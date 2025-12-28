"""
Sequence Packing Collators for Training Optimization

This module implements sequence packing to eliminate padding waste and improve
GPU utilization. Multiple short sequences are packed into single training
examples up to max_length, separated by EOS tokens.

CRITICAL: Document boundary masking is implemented to prevent cross-document
attention. Without this, the model learns broken grammar from unrelated sequences.

Classes:
    SequencePackingCollator: Greedy bin-packing for fast, simple packing
    DynamicSequencePackingCollator: Adaptive packing with statistics tracking
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import numpy as np
from collections import defaultdict
import warnings


def validate_attention_mask(
    attention_mask: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    name: str = "attention_mask"
) -> None:
    """
    Validate attention mask shape and dtype.

    Args:
        attention_mask: The mask tensor to validate
        expected_shape: Expected shape tuple (use -1 for any dimension)
        expected_dtype: Expected dtype
        name: Name for error messages

    Raises:
        ValueError: If validation fails
    """
    if attention_mask is None:
        return

    # Check dtype
    if expected_dtype is not None and attention_mask.dtype != expected_dtype:
        warnings.warn(
            f"{name} has dtype {attention_mask.dtype}, expected {expected_dtype}. "
            f"This may cause dtype mismatches in mixed precision training.",
            UserWarning
        )

    # Check shape
    if expected_shape is not None:
        if len(attention_mask.shape) != len(expected_shape):
            raise ValueError(
                f"{name} has {len(attention_mask.shape)} dimensions, "
                f"expected {len(expected_shape)}. Shape: {attention_mask.shape}"
            )
        for i, (actual, expected) in enumerate(zip(attention_mask.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} is {actual}, expected {expected}. "
                    f"Shape: {attention_mask.shape}"
                )

    # Check for NaN values
    if torch.isnan(attention_mask).any():
        raise ValueError(f"{name} contains NaN values")

    # Check mask values are valid (0 or -inf for additive masks)
    finite_mask = torch.isfinite(attention_mask)
    finite_values = attention_mask[finite_mask]
    if finite_values.numel() > 0:
        if not torch.allclose(finite_values, torch.zeros_like(finite_values), atol=1e-6):
            non_zero = finite_values[finite_values.abs() > 1e-6]
            if non_zero.numel() > 0:
                warnings.warn(
                    f"{name} contains unexpected finite values (not 0): "
                    f"range [{non_zero.min().item():.4f}, {non_zero.max().item():.4f}]",
                    UserWarning
                )


def create_document_attention_mask(
    document_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a 2D attention mask that blocks cross-document attention.

    For packed sequences like [Doc1_tokens... Doc2_tokens... PAD...],
    this creates a mask where tokens can only attend to tokens from
    the same document.

    Args:
        document_ids: [seq_len] tensor where each position contains
                     the document ID (0, 1, 2, ...) or -1 for padding
        dtype: Output dtype for the mask

    Returns:
        [seq_len, seq_len] attention mask where:
        - 0.0 = can attend (same document)
        - -inf = cannot attend (different document or padding)
    """
    seq_len = document_ids.size(0)

    # Create mask: positions can attend if they have the same document_id
    # and neither is padding (-1)
    # Broadcast: [seq_len, 1] vs [1, seq_len] -> [seq_len, seq_len]
    same_doc = (document_ids.unsqueeze(1) == document_ids.unsqueeze(0))

    # Mask out padding positions (document_id == -1)
    not_padding = (document_ids != -1)
    valid_positions = not_padding.unsqueeze(1) & not_padding.unsqueeze(0)

    # Combine: can attend only if same document AND both not padding
    can_attend = same_doc & valid_positions

    # Also apply causal mask (can only attend to earlier positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=document_ids.device))
    can_attend = can_attend & causal_mask

    # Convert to attention mask format: 0 for attend, -inf for block
    attention_mask = torch.where(
        can_attend,
        torch.tensor(0.0, dtype=dtype, device=document_ids.device),
        torch.tensor(float('-inf'), dtype=dtype, device=document_ids.device)
    )

    return attention_mask


def create_document_position_ids(
    document_ids: torch.Tensor
) -> torch.Tensor:
    """
    Create position IDs that reset to 0 at each document boundary.

    This ensures RoPE positional embeddings restart for each document,
    preventing position information from leaking across documents.

    Args:
        document_ids: [seq_len] tensor where each position contains
                     the document ID (0, 1, 2, ...) or -1 for padding

    Returns:
        [seq_len] tensor of position IDs (0, 1, 2, ... resetting per doc)
    """
    seq_len = document_ids.size(0)
    position_ids = torch.zeros(seq_len, dtype=torch.long, device=document_ids.device)

    current_doc = -1
    current_pos = 0

    for i in range(seq_len):
        doc_id = document_ids[i].item()
        if doc_id == -1:  # Padding
            position_ids[i] = 0
        elif doc_id != current_doc:  # New document
            current_doc = doc_id
            current_pos = 0
            position_ids[i] = current_pos
            current_pos += 1
        else:  # Same document
            position_ids[i] = current_pos
            current_pos += 1

    return position_ids


class SequencePackingCollator:
    """
    Pack multiple short sequences into single training examples.

    Uses greedy bin-packing: concatenates sequences with EOS separator
    until max_length is reached, then starts a new packed sequence.

    This eliminates padding waste and improves throughput.

    Args:
        max_length: Maximum sequence length for packed examples
        pad_token_id: Token ID used for padding
        eos_token_id: Token ID for end-of-sequence (used as separator)
        pack_sequences: Whether to enable packing (True) or use standard collation
        mask_dtype: Dtype for attention masks (should match model's working dtype)
        validate_masks: Whether to validate attention masks (slower but catches bugs)
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        pack_sequences: bool = True,
        sort_by_length: bool = False,  # False preserves random order for better coherence
        preserve_batch_size: bool = True,
        mask_dtype: torch.dtype = torch.bfloat16,  # FIXED: Must match model dtype for -inf precision
        validate_masks: bool = False,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pack_sequences = pack_sequences
        self.sort_by_length = sort_by_length
        self.preserve_batch_size = preserve_batch_size  # Keep output batch size = input batch size
        self.mask_dtype = mask_dtype
        self.validate_masks = validate_masks

        # Statistics tracking
        self.total_tokens_before = 0
        self.total_tokens_after = 0
        self.num_batches = 0

        # Pre-allocated tensors for efficiency (avoid per-batch allocations)
        self._eos_tensor: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None

    def _to_tensor(self, arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert numpy array to tensor if needed."""
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr.copy())
        return arr

    def _get_sequence_length(self, item: Union[Dict, torch.Tensor, np.ndarray]) -> int:
        """Get the actual length of a sequence (excluding padding)."""
        if isinstance(item, dict):
            if 'attention_mask' in item:
                mask = item['attention_mask']
                if isinstance(mask, np.ndarray):
                    return int(mask.sum())
                return int(mask.sum())
            elif 'input_ids' in item:
                ids = item['input_ids']
                if isinstance(ids, np.ndarray):
                    return int((ids != self.pad_token_id).sum())
                # Count non-padding tokens
                return int((ids != self.pad_token_id).sum())
        elif isinstance(item, np.ndarray):
            return int((item != self.pad_token_id).sum())
        elif isinstance(item, torch.Tensor):
            return int((item != self.pad_token_id).sum())
        return self.max_length

    def _extract_ids(self, item: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Extract input_ids from various input formats and convert to tensor."""
        if isinstance(item, dict):
            ids = item['input_ids']
            return self._to_tensor(ids)
        return self._to_tensor(item)

    def _get_eos_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        """Get cached EOS tensor, creating if needed."""
        if self._eos_tensor is None or self._eos_tensor.dtype != dtype:
            self._eos_tensor = torch.tensor([self.eos_token_id], dtype=dtype)
        return self._eos_tensor

    def _pack_sequences_greedy(
        self,
        sequences: List[torch.Tensor],
        lengths: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Optimized greedy bin-packing algorithm with document boundary tracking.

        Sorts sequences by length (longest first) and packs them
        into bins until each bin reaches max_length.
        Uses pre-allocated buffers to minimize tensor allocations.

        Returns:
            Tuple of (packed_sequences, document_ids_list)
            - packed_sequences: List of packed input_ids tensors
            - document_ids_list: List of document ID tensors for attention masking
        """
        if not sequences:
            return [], []

        # Optionally sort by length descending for better packing
        if self.sort_by_length:
            sorted_indices = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)
        else:
            sorted_indices = list(range(len(sequences)))  # Keep original order for randomness

        packed = []
        document_ids_list = []
        dtype = sequences[0].dtype
        eos_tensor = self._get_eos_tensor(dtype)

        # Pre-allocate output buffer (reused for each pack)
        output_buffer = torch.full((self.max_length,), self.pad_token_id, dtype=dtype)
        doc_id_buffer = torch.full((self.max_length,), -1, dtype=torch.long)  # -1 = padding

        current_parts: List[Tuple[torch.Tensor, int, int]] = []  # (tensor, length, doc_id) tuples
        current_length = 0
        current_doc_id = 0  # Document ID within this packed sequence

        for idx in sorted_indices:
            seq = sequences[idx]
            seq_len = lengths[idx]

            # +1 for EOS separator between sequences (EOS belongs to previous doc)
            needed_length = seq_len + (1 if current_parts else 0)

            if current_length + needed_length <= self.max_length:
                # Add to current pack
                if current_parts:
                    # EOS separator belongs to the PREVIOUS document
                    prev_doc_id = current_parts[-1][2]
                    current_parts.append((eos_tensor, 1, prev_doc_id))
                    current_length += 1
                    current_doc_id += 1  # New document starts after EOS

                current_parts.append((seq[:seq_len], seq_len, current_doc_id))
                current_length += seq_len
            else:
                # Finalize current pack and start new one
                if current_parts:
                    packed_seq, doc_ids = self._finalize_pack_with_doc_ids(
                        current_parts, current_length, output_buffer.clone(), doc_id_buffer.clone()
                    )
                    packed.append(packed_seq)
                    document_ids_list.append(doc_ids)

                current_parts = [(seq[:seq_len], seq_len, 0)]  # Reset doc_id to 0 for new pack
                current_length = seq_len
                current_doc_id = 0

        # Don't forget the last pack
        if current_parts:
            packed_seq, doc_ids = self._finalize_pack_with_doc_ids(
                current_parts, current_length, output_buffer.clone(), doc_id_buffer.clone()
            )
            packed.append(packed_seq)
            document_ids_list.append(doc_ids)

        return packed, document_ids_list

    def _finalize_pack_with_doc_ids(
        self,
        parts: List[Tuple[torch.Tensor, int, int]],
        total_length: int,
        output: torch.Tensor,
        doc_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast pack finalization with document ID tracking."""
        pos = 0
        for tensor, length, doc_id in parts:
            output[pos:pos + length] = tensor[:length]
            doc_ids[pos:pos + length] = doc_id
            pos += length
        return output, doc_ids

    def _finalize_pack_fast(
        self,
        parts: List[Tuple[torch.Tensor, int]],
        total_length: int,
        output: torch.Tensor
    ) -> torch.Tensor:
        """Fast pack finalization using pre-allocated buffer."""
        pos = 0
        for tensor, length in parts:
            output[pos:pos + length] = tensor[:length]
            pos += length
        return output

    def _finalize_pack(self, pack: List[torch.Tensor], pack_length: int) -> torch.Tensor:
        """Concatenate pack and add padding to max_length."""
        concatenated = torch.cat(pack)

        # Pad to max_length
        if len(concatenated) < self.max_length:
            padding = torch.full(
                (self.max_length - len(concatenated),),
                self.pad_token_id,
                dtype=concatenated.dtype
            )
            concatenated = torch.cat([concatenated, padding])

        return concatenated[:self.max_length]

    def _adjust_to_batch_size(
        self,
        packed: List[torch.Tensor],
        sequences: List[torch.Tensor],
        lengths: List[int],
        target_size: int
    ) -> List[torch.Tensor]:
        """
        Adjust packed sequences to match target batch size.

        - If too few packed: add unpacked sequences (padded to max_length)
        - If too many packed: truncate to target size

        This ensures consistent batch sizes during training when packing is enabled.
        """
        current_size = len(packed)

        if current_size == target_size:
            return packed

        if current_size > target_size:
            # Too many packed sequences - truncate (keeps most densely packed ones)
            return packed[:target_size]

        # Too few packed sequences - need to add more
        # Add unpacked sequences (padded individually) to reach target size
        needed = target_size - current_size
        dtype = packed[0].dtype if packed else torch.long

        # Find sequences not yet used (short ones that didn't fit well into packs)
        # We'll pad them individually to max_length
        for i in range(min(needed, len(sequences))):
            seq = sequences[i]
            seq_len = lengths[i]

            # Create padded sequence
            padded_seq = torch.full((self.max_length,), self.pad_token_id, dtype=dtype)
            padded_seq[:seq_len] = seq[:seq_len]
            packed.append(padded_seq)

            if len(packed) >= target_size:
                break

        # If still not enough (rare), duplicate last packed sequence
        while len(packed) < target_size:
            packed.append(packed[-1].clone())

        return packed[:target_size]

    def __call__(self, batch: List[Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pack a batch of sequences with document boundary masking.

        Args:
            batch: List of items, each being either a dict with 'input_ids'
                   or a tensor of token IDs

        Returns:
            Dict with:
            - 'input_ids': [batch, seq_len] packed token IDs
            - 'attention_mask': [batch, seq_len, seq_len] 2D mask blocking cross-document attention
            - 'labels': [batch, seq_len] labels with -100 for padding
            - 'position_ids': [batch, seq_len] position IDs that reset per document
            - 'document_ids': [batch, seq_len] document IDs for debugging
        """
        if not batch:
            return {}

        if not self.pack_sequences:
            # Standard collation without packing
            return self._standard_collate(batch)

        # Extract sequences and their lengths
        sequences = [self._extract_ids(item) for item in batch]
        lengths = [self._get_sequence_length(item) for item in batch]

        # Track statistics
        self.total_tokens_before += sum(lengths)
        self.num_batches += 1

        # Pack sequences with document boundary tracking
        packed, document_ids_list = self._pack_sequences_greedy(sequences, lengths)

        if not packed:
            return self._standard_collate(batch)

        # Preserve batch size if enabled (prevents BS fluctuation during training)
        target_batch_size = len(batch)
        if self.preserve_batch_size and len(packed) != target_batch_size:
            packed, document_ids_list = self._adjust_to_batch_size_with_doc_ids(
                packed, document_ids_list, sequences, lengths, target_batch_size
            )

        # Stack into batch
        input_ids = torch.stack(packed)
        document_ids = torch.stack(document_ids_list)

        # Create 2D attention mask with document boundaries
        # This BLOCKS cross-document attention - critical for coherent learning!
        batch_size, seq_len = input_ids.shape
        # DTYPE FIX: Explicitly specify dtype to match model's working dtype
        # This prevents dtype mismatches in mixed precision training (FP16/BF16)
        attention_mask_2d = torch.zeros(batch_size, seq_len, seq_len, dtype=self.mask_dtype)
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        for i in range(batch_size):
            attention_mask_2d[i] = create_document_attention_mask(document_ids[i], dtype=self.mask_dtype)
            position_ids[i] = create_document_position_ids(document_ids[i])

        # Validate masks if enabled (catches bugs early but adds overhead)
        if self.validate_masks:
            validate_attention_mask(
                attention_mask_2d,
                expected_shape=(batch_size, seq_len, seq_len),
                expected_dtype=self.mask_dtype,
                name="packed_attention_mask"
            )

        # Labels are same as input_ids for causal LM (shifted internally by model)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # Ignore padding in loss

        # Also mask out EOS tokens in labels between documents (optional, prevents learning to predict separator)
        # This is a design choice - you may want the model to learn EOS prediction
        # For now, we keep EOS in labels so model learns when to end

        # Track packed tokens (using simple mask for counting)
        simple_mask = (input_ids != self.pad_token_id).long()
        self.total_tokens_after += int(simple_mask.sum())

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask_2d,  # 2D mask for document boundaries
            'labels': labels,
            'position_ids': position_ids,  # Reset per document for RoPE
            'document_ids': document_ids,  # For debugging
        }

    def _adjust_to_batch_size_with_doc_ids(
        self,
        packed: List[torch.Tensor],
        document_ids_list: List[torch.Tensor],
        sequences: List[torch.Tensor],
        lengths: List[int],
        target_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Adjust packed sequences and document IDs to match target batch size.
        """
        current_size = len(packed)

        if current_size == target_size:
            return packed, document_ids_list

        if current_size > target_size:
            # Truncate
            return packed[:target_size], document_ids_list[:target_size]

        # Need to add more - create single-document packs
        needed = target_size - current_size
        dtype = packed[0].dtype if packed else torch.long

        for i in range(min(needed, len(sequences))):
            seq = sequences[i]
            seq_len = lengths[i]

            # Create padded sequence (single document)
            padded_seq = torch.full((self.max_length,), self.pad_token_id, dtype=dtype)
            padded_seq[:seq_len] = seq[:seq_len]

            # Create document IDs (all same doc_id=0, padding=-1)
            doc_ids = torch.full((self.max_length,), -1, dtype=torch.long)
            doc_ids[:seq_len] = 0

            packed.append(padded_seq)
            document_ids_list.append(doc_ids)

            if len(packed) >= target_size:
                break

        # If still not enough, duplicate last
        while len(packed) < target_size:
            packed.append(packed[-1].clone())
            document_ids_list.append(document_ids_list[-1].clone())

        return packed[:target_size], document_ids_list[:target_size]

    def _standard_collate(self, batch: List[Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Standard collation without packing (fallback)."""
        sequences = [self._extract_ids(item) for item in batch]

        # Pad to max length in batch
        max_len = min(max(len(s) for s in sequences), self.max_length)

        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = torch.full(
                    (max_len - len(seq),),
                    self.pad_token_id,
                    dtype=seq.dtype
                )
                seq = torch.cat([seq, padding])
            padded.append(seq[:max_len])

        input_ids = torch.stack(padded)
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def get_packing_efficiency(self) -> float:
        """Return the packing efficiency ratio."""
        if self.total_tokens_before == 0:
            return 1.0
        return self.total_tokens_after / self.total_tokens_before

    def reset_statistics(self):
        """Reset packing statistics."""
        self.total_tokens_before = 0
        self.total_tokens_after = 0
        self.num_batches = 0


class DynamicSequencePackingCollator(SequencePackingCollator):
    """
    Adaptive sequence packing with statistics tracking and dynamic optimization.

    Extends SequencePackingCollator with:
    - Length distribution tracking for better packing decisions
    - Adaptive binning based on sequence length statistics
    - Detailed efficiency metrics

    Args:
        max_length: Maximum sequence length for packed examples
        pad_token_id: Token ID used for padding
        eos_token_id: Token ID for end-of-sequence (used as separator)
        target_packing_ratio: Target ratio of utilized tokens (0.0-1.0)
        adaptive_binning: Whether to use adaptive bin sizes based on length distribution
        mask_dtype: Dtype for attention masks (should match model's working dtype)
        validate_masks: Whether to validate attention masks (slower but catches bugs)
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        target_packing_ratio: float = 0.9,
        adaptive_binning: bool = True,
        preserve_batch_size: bool = True,
        mask_dtype: torch.dtype = torch.bfloat16,  # FIXED: Must match model dtype for -inf precision
        validate_masks: bool = False,
    ):
        super().__init__(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            pack_sequences=True,
            preserve_batch_size=preserve_batch_size,
            mask_dtype=mask_dtype,
            validate_masks=validate_masks,
        )
        self.target_packing_ratio = target_packing_ratio
        self.adaptive_binning = adaptive_binning

        # Length distribution tracking
        self.length_counts = defaultdict(int)
        self.length_sum = 0
        self.length_count = 0

        # Binning state
        self.bins: List[List[int]] = []  # Bin boundaries
        self._update_bins()

    def _update_bins(self):
        """Update bin boundaries based on length distribution."""
        if not self.adaptive_binning or self.length_count < 100:
            # Default bins: powers of 2
            self.bins = [16, 32, 64, 128, 256, 512, 1024, self.max_length]
        else:
            # Adaptive bins based on length distribution
            avg_length = self.length_sum / self.length_count
            self.bins = [
                int(avg_length * 0.25),
                int(avg_length * 0.5),
                int(avg_length * 0.75),
                int(avg_length),
                int(avg_length * 1.5),
                int(avg_length * 2),
                self.max_length,
            ]

    def _get_bin(self, length: int) -> int:
        """Get the bin index for a sequence length."""
        for i, boundary in enumerate(self.bins):
            if length <= boundary:
                return i
        return len(self.bins) - 1

    def _pack_sequences_adaptive(
        self,
        sequences: List[torch.Tensor],
        lengths: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Adaptive packing that groups similar-length sequences.

        Groups sequences by length bins, then applies greedy packing
        within each bin for more efficient packing.

        Returns:
            Tuple of (packed_sequences, document_ids_list)
        """
        if not sequences:
            return [], []

        # Group by bins
        binned: Dict[int, List[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            bin_idx = self._get_bin(length)
            binned[bin_idx].append(idx)

        all_packed = []
        all_doc_ids = []

        # Pack each bin separately
        for bin_idx in sorted(binned.keys()):
            indices = binned[bin_idx]
            bin_sequences = [sequences[i] for i in indices]
            bin_lengths = [lengths[i] for i in indices]

            # Use greedy packing within bin (returns tuple now)
            packed, doc_ids = self._pack_sequences_greedy(bin_sequences, bin_lengths)
            all_packed.extend(packed)
            all_doc_ids.extend(doc_ids)

        return all_packed, all_doc_ids

    def __call__(self, batch: List[Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pack a batch with adaptive optimization and document boundary masking.

        Returns:
            Dict with:
            - 'input_ids': [batch, seq_len] packed token IDs
            - 'attention_mask': [batch, seq_len, seq_len] 2D mask blocking cross-document attention
            - 'labels': [batch, seq_len] labels with -100 for padding
            - 'position_ids': [batch, seq_len] position IDs that reset per document
            - 'document_ids': [batch, seq_len] document IDs for debugging
        """
        if not batch:
            return {}

        # Extract sequences and their lengths
        sequences = [self._extract_ids(item) for item in batch]
        lengths = [self._get_sequence_length(item) for item in batch]

        # Update length statistics
        for length in lengths:
            self.length_counts[length] += 1
            self.length_sum += length
            self.length_count += 1

        # Periodically update bins
        if self.length_count % 1000 == 0:
            self._update_bins()

        # Track statistics
        self.total_tokens_before += sum(lengths)
        self.num_batches += 1

        # Pack sequences using adaptive method (returns packed + document_ids)
        if self.adaptive_binning:
            packed, document_ids_list = self._pack_sequences_adaptive(sequences, lengths)
        else:
            packed, document_ids_list = self._pack_sequences_greedy(sequences, lengths)

        if not packed:
            return self._standard_collate(batch)

        # Preserve batch size if enabled (prevents BS fluctuation during training)
        target_batch_size = len(batch)
        if self.preserve_batch_size and len(packed) != target_batch_size:
            packed, document_ids_list = self._adjust_to_batch_size_with_doc_ids(
                packed, document_ids_list, sequences, lengths, target_batch_size
            )

        # Stack into batch
        input_ids = torch.stack(packed)
        document_ids = torch.stack(document_ids_list)

        # Create 2D attention mask with document boundaries
        # This BLOCKS cross-document attention - critical for coherent learning!
        batch_size, seq_len = input_ids.shape
        # DTYPE FIX: Explicitly specify dtype to match model's working dtype
        # This prevents dtype mismatches in mixed precision training (FP16/BF16)
        attention_mask_2d = torch.zeros(batch_size, seq_len, seq_len, dtype=self.mask_dtype)
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        for i in range(batch_size):
            attention_mask_2d[i] = create_document_attention_mask(document_ids[i], dtype=self.mask_dtype)
            position_ids[i] = create_document_position_ids(document_ids[i])

        # Validate masks if enabled (catches bugs early but adds overhead)
        if self.validate_masks:
            validate_attention_mask(
                attention_mask_2d,
                expected_shape=(batch_size, seq_len, seq_len),
                expected_dtype=self.mask_dtype,
                name="dynamic_packed_attention_mask"
            )

        # Labels
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        # Track packed tokens (using simple mask for counting)
        simple_mask = (input_ids != self.pad_token_id).long()
        self.total_tokens_after += int(simple_mask.sum())

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask_2d,  # 2D mask for document boundaries
            'labels': labels,
            'position_ids': position_ids,  # Reset per document for RoPE
            'document_ids': document_ids,  # For debugging
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return detailed packing statistics."""
        return {
            'packing_efficiency': self.get_packing_efficiency(),
            'total_batches': self.num_batches,
            'total_tokens_before': self.total_tokens_before,
            'total_tokens_after': self.total_tokens_after,
            'avg_sequence_length': self.length_sum / max(1, self.length_count),
            'unique_lengths': len(self.length_counts),
            'bin_boundaries': self.bins,
        }

    def reset_statistics(self):
        """Reset all statistics."""
        super().reset_statistics()
        self.length_counts.clear()
        self.length_sum = 0
        self.length_count = 0


# Convenience function for creating the appropriate collator
def create_packing_collator(
    max_length: int,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    strategy: str = 'greedy',
    **kwargs
) -> Union[SequencePackingCollator, DynamicSequencePackingCollator]:
    """
    Factory function to create the appropriate packing collator.

    Args:
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        eos_token_id: EOS token ID
        strategy: 'greedy' or 'adaptive'
        **kwargs: Additional arguments passed to collator

    Returns:
        Configured collator instance
    """
    if strategy == 'adaptive':
        return DynamicSequencePackingCollator(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    else:
        return SequencePackingCollator(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
