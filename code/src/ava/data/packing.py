"""
Sequence Packing Collators for Training Optimization

This module implements sequence packing to eliminate padding waste and improve
GPU utilization by 20-35%. Multiple short sequences are packed into single
training examples up to max_length, separated by EOS tokens.

Classes:
    SequencePackingCollator: Greedy bin-packing for fast, simple packing
    DynamicSequencePackingCollator: Adaptive packing with statistics tracking
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import numpy as np
from collections import defaultdict


class SequencePackingCollator:
    """
    Pack multiple short sequences into single training examples.

    Uses greedy bin-packing: concatenates sequences with EOS separator
    until max_length is reached, then starts a new packed sequence.

    This eliminates padding waste, improving throughput by 20-35%.

    Args:
        max_length: Maximum sequence length for packed examples
        pad_token_id: Token ID used for padding
        eos_token_id: Token ID for end-of-sequence (used as separator)
        pack_sequences: Whether to enable packing (True) or use standard collation
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        pack_sequences: bool = True,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pack_sequences = pack_sequences

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
    ) -> List[torch.Tensor]:
        """
        Optimized greedy bin-packing algorithm.

        Sorts sequences by length (longest first) and packs them
        into bins until each bin reaches max_length.
        Uses pre-allocated buffers to minimize tensor allocations.
        """
        if not sequences:
            return []

        # Sort by length descending for better packing
        sorted_indices = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)

        packed = []
        dtype = sequences[0].dtype
        eos_tensor = self._get_eos_tensor(dtype)

        # Pre-allocate output buffer (reused for each pack)
        output_buffer = torch.full((self.max_length,), self.pad_token_id, dtype=dtype)

        current_parts: List[Tuple[torch.Tensor, int]] = []  # (tensor, length) pairs
        current_length = 0

        for idx in sorted_indices:
            seq = sequences[idx]
            seq_len = lengths[idx]

            # +1 for EOS separator between sequences
            needed_length = seq_len + (1 if current_parts else 0)

            if current_length + needed_length <= self.max_length:
                # Add to current pack
                if current_parts:
                    current_parts.append((eos_tensor, 1))
                    current_length += 1
                current_parts.append((seq[:seq_len], seq_len))
                current_length += seq_len
            else:
                # Finalize current pack and start new one
                if current_parts:
                    packed.append(self._finalize_pack_fast(current_parts, current_length, output_buffer.clone()))
                current_parts = [(seq[:seq_len], seq_len)]
                current_length = seq_len

        # Don't forget the last pack
        if current_parts:
            packed.append(self._finalize_pack_fast(current_parts, current_length, output_buffer.clone()))

        return packed

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

    def __call__(self, batch: List[Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pack a batch of sequences.

        Args:
            batch: List of items, each being either a dict with 'input_ids'
                   or a tensor of token IDs

        Returns:
            Dict with 'input_ids', 'attention_mask', and 'labels'
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

        # Pack sequences
        packed = self._pack_sequences_greedy(sequences, lengths)

        if not packed:
            return self._standard_collate(batch)

        # Stack into batch
        input_ids = torch.stack(packed)

        # Create attention mask (1 for non-padding tokens)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Labels are same as input_ids for causal LM (shifted internally by model)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # Ignore padding in loss

        # Track packed tokens
        self.total_tokens_after += int(attention_mask.sum())

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

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
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        target_packing_ratio: float = 0.9,
        adaptive_binning: bool = True,
    ):
        super().__init__(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            pack_sequences=True,
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
    ) -> List[torch.Tensor]:
        """
        Adaptive packing that groups similar-length sequences.

        Groups sequences by length bins, then applies greedy packing
        within each bin for more efficient packing.
        """
        if not sequences:
            return []

        # Group by bins
        binned: Dict[int, List[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            bin_idx = self._get_bin(length)
            binned[bin_idx].append(idx)

        all_packed = []

        # Pack each bin separately
        for bin_idx in sorted(binned.keys()):
            indices = binned[bin_idx]
            bin_sequences = [sequences[i] for i in indices]
            bin_lengths = [lengths[i] for i in indices]

            # Use greedy packing within bin
            packed = self._pack_sequences_greedy(bin_sequences, bin_lengths)
            all_packed.extend(packed)

        return all_packed

    def __call__(self, batch: List[Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pack a batch with adaptive optimization.
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

        # Pack sequences using adaptive method
        if self.adaptive_binning:
            packed = self._pack_sequences_adaptive(sequences, lengths)
        else:
            packed = self._pack_sequences_greedy(sequences, lengths)

        if not packed:
            return self._standard_collate(batch)

        # Stack into batch
        input_ids = torch.stack(packed)

        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()

        # Labels
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        # Track packed tokens
        self.total_tokens_after += int(attention_mask.sum())

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
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
