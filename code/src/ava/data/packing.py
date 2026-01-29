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
import os
import torch
import numpy as np
from collections import defaultdict
import warnings

# Numba JIT compilation for performance-critical functions
# Falls back to pure Python if Numba is not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator when Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Environment variable to disable Numba optimization
USE_NUMBA_PACKING = os.environ.get("AVA_USE_NUMBA_PACKING", "1") == "1"


@njit(cache=True)
def _compute_pack_assignments_numba(lengths: np.ndarray, max_length: int) -> np.ndarray:
    """
    Numba-optimized pack assignment computation using greedy bin-packing.

    Computes which pack each sequence belongs to without creating the actual
    packed tensors. This is the hot loop that benefits most from JIT compilation.

    Args:
        lengths: Array of sequence lengths (int64)
        max_length: Maximum length for packed sequences

    Returns:
        pack_ids: Array where pack_ids[i] = pack index for sequence i
    """
    n = len(lengths)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Sort indices by length descending for better packing
    sorted_indices = np.argsort(-lengths)

    pack_ids = np.zeros(n, dtype=np.int32)
    current_pack = 0
    current_length = 0

    for i in range(n):
        idx = sorted_indices[i]
        seq_len = lengths[idx]

        # +1 for EOS separator between sequences
        needed = seq_len + (1 if current_length > 0 else 0)

        if current_length + needed <= max_length:
            pack_ids[idx] = current_pack
            current_length += needed
        else:
            # Start new pack
            current_pack += 1
            pack_ids[idx] = current_pack
            current_length = seq_len

    return pack_ids


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
                # Batch min/max into single sync
                minmax = torch.stack([non_zero.min(), non_zero.max()]).tolist()
                warnings.warn(
                    f"{name} contains unexpected finite values (not 0): "
                    f"range [{minmax[0]:.4f}, {minmax[1]:.4f}]",
                    UserWarning
                )


def validate_document_boundaries(
    attention_mask: torch.Tensor,
    document_ids: torch.Tensor,
    name: str = "packed_batch"
) -> bool:
    """
    Validate that attention mask correctly blocks cross-document attention.

    This is a critical runtime check to ensure document boundary masking is working.
    Cross-document attention causes the model to learn broken grammar from unrelated
    sequences, which can silently degrade model quality.

    Args:
        attention_mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len] attention mask
        document_ids: [batch_size, seq_len] or [seq_len] document IDs (-1 for padding)
        name: Name for error messages

    Returns:
        True if validation passes

    Raises:
        ValueError: If cross-document attention is detected (mask allows attention
                   between positions with different document IDs)
    """
    if attention_mask is None or document_ids is None:
        return True

    # Handle both batched and unbatched inputs
    if attention_mask.ndim == 2:
        attention_mask = attention_mask.unsqueeze(0)
        document_ids = document_ids.unsqueeze(0)

    batch_size, seq_len, _ = attention_mask.shape

    # Check each batch item
    violations_found = []
    for b in range(min(batch_size, 4)):  # Check first 4 items for performance
        mask = attention_mask[b]
        doc_ids = document_ids[b]

        # Find positions where attention is allowed (mask value is 0, not -inf)
        # For additive masks: 0 means attend, -inf means block
        can_attend = torch.isfinite(mask) & (mask > -1e9)

        # For each attending pair, check if they're in the same document
        for i in range(min(seq_len, 64)):  # Check first 64 positions
            for j in range(i + 1):  # Only check causal positions
                if can_attend[i, j]:
                    # Position i attends to position j - must be same document
                    doc_i = doc_ids[i].item()
                    doc_j = doc_ids[j].item()

                    # Skip padding positions
                    if doc_i == -1 or doc_j == -1:
                        continue

                    if doc_i != doc_j:
                        violations_found.append((b, i, j, doc_i, doc_j))
                        if len(violations_found) >= 5:  # Limit violations to report
                            break
            if len(violations_found) >= 5:
                break
        if len(violations_found) >= 5:
            break

    if violations_found:
        violation_str = "\n".join([
            f"  batch[{b}]: pos {i} (doc={di}) attends to pos {j} (doc={dj})"
            for b, i, j, di, dj in violations_found[:5]
        ])
        raise ValueError(
            f"{name}: Cross-document attention detected! "
            f"Document boundary masking is NOT working correctly.\n"
            f"Violations found:\n{violation_str}\n"
            f"This will cause the model to learn broken grammar. "
            f"Check that document_ids are correctly assigned during packing."
        )

    return True


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


def create_document_attention_mask_batched(
    document_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create batched 2D attention masks that block cross-document attention.

    OPTIMIZED: Processes entire batch in parallel with broadcasting.
    Eliminates Python for loop over batch dimension.

    For packed sequences like [Doc1_tokens... Doc2_tokens... PAD...],
    this creates masks where tokens can only attend to tokens from
    the same document.

    Args:
        document_ids: [batch_size, seq_len] tensor where each position contains
                     the document ID (0, 1, 2, ...) or -1 for padding
        dtype: Output dtype for the mask

    Returns:
        [batch_size, seq_len, seq_len] attention mask where:
        - 0.0 = can attend (same document)
        - -inf = cannot attend (different document or padding)
    """
    batch_size, seq_len = document_ids.shape
    device = document_ids.device

    # Broadcast document IDs for comparison
    # [batch, seq, 1] vs [batch, 1, seq] -> [batch, seq, seq]
    doc_ids_row = document_ids.unsqueeze(2)  # [batch, seq, 1]
    doc_ids_col = document_ids.unsqueeze(1)  # [batch, 1, seq]

    # Same document check (vectorized across batch)
    same_doc = (doc_ids_row == doc_ids_col)  # [batch, seq, seq]

    # Padding mask: neither position is padding (-1)
    not_padding = (document_ids != -1)  # [batch, seq]
    not_padding_row = not_padding.unsqueeze(2)  # [batch, seq, 1]
    not_padding_col = not_padding.unsqueeze(1)  # [batch, 1, seq]
    valid_positions = not_padding_row & not_padding_col  # [batch, seq, seq]

    # Combined mask: same document AND both valid
    can_attend = same_doc & valid_positions

    # Causal mask (shared across batch) - query can only attend to earlier keys
    # Use 1D causal and broadcast
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )  # [seq, seq]

    # Apply causal mask to all batch items
    can_attend = can_attend & causal_mask.unsqueeze(0)  # [batch, seq, seq]

    # Convert to attention mask format: 0 for attend, -inf for block
    # Use in-place operations for memory efficiency
    attention_mask = torch.full(
        (batch_size, seq_len, seq_len),
        float('-inf'),
        dtype=dtype,
        device=device
    )
    attention_mask[can_attend] = 0.0

    return attention_mask


def create_document_position_ids_batched(
    document_ids: torch.Tensor
) -> torch.Tensor:
    """
    Create batched position IDs that reset to 0 at each document boundary.

    OPTIMIZED: Fully vectorized across batch dimension.

    Args:
        document_ids: [batch_size, seq_len] tensor

    Returns:
        [batch_size, seq_len] tensor of position IDs
    """
    batch_size, seq_len = document_ids.shape
    device = document_ids.device

    if seq_len == 0:
        return torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    # Detect document boundaries for each sequence in batch
    doc_change = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    doc_change[:, 0] = True  # First position always starts a segment

    if seq_len > 1:
        # Document changes or previous was padding
        doc_change[:, 1:] = (
            (document_ids[:, 1:] != document_ids[:, :-1]) |
            (document_ids[:, :-1] == -1)
        )

    # Create position indices [seq_len] and broadcast to [batch, seq_len]
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(batch_size, -1)

    # Get segment start positions using cummax along sequence dimension
    segment_start_positions = torch.where(
        doc_change,
        positions,
        torch.zeros_like(positions)
    )
    segment_starts, _ = torch.cummax(segment_start_positions, dim=1)

    # Position within document
    position_ids = positions - segment_starts

    # Padding positions get position 0
    position_ids = torch.where(
        document_ids == -1,
        torch.zeros_like(position_ids),
        position_ids
    )

    return position_ids


def create_cu_seqlens_from_document_ids(
    document_ids: torch.Tensor,
    pad_token_id: int = -1
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert document IDs to cu_seqlens format for Flash Attention varlen.

    This eliminates the need for 2D attention masks entirely by using
    Flash Attention's native variable-length sequence support.

    CRITICAL: This format is ~40-60% faster than 2D masks because:
    - No 512MB mask allocation (for batch=32, seq=2048)
    - Flash Attention can skip padding tokens entirely
    - Better memory locality

    Args:
        document_ids: [batch_size, seq_len] tensor where each position contains
                     the document ID (0, 1, 2, ...) or -1 for padding

    Returns:
        Tuple of:
        - cu_seqlens: [total_documents + 1] cumulative sequence lengths
        - input_ids_packed: [total_tokens] concatenated non-padding tokens
        - max_seqlen: maximum document length in batch

    Example:
        document_ids = [[0, 0, 0, 1, 1, -1],   # Doc0=3 tokens, Doc1=2 tokens, 1 pad
                        [0, 0, 1, 1, 1, -1]]   # Doc0=2 tokens, Doc1=3 tokens, 1 pad

        Returns:
        - cu_seqlens = [0, 3, 5, 7, 10]  # Cumulative: 0, 3, 3+2=5, 5+2=7, 7+3=10
        - max_seqlen = 3
    """
    batch_size, seq_len = document_ids.shape
    device = document_ids.device

    # Find non-padding mask
    not_padding = (document_ids != pad_token_id)

    # Count documents and their lengths per batch item
    # Each unique document ID (excluding padding) is a separate document
    cu_seqlens_list = [torch.tensor([0], dtype=torch.int32, device=device)]
    doc_lengths = []
    total_tokens = 0

    for b in range(batch_size):
        row = document_ids[b]
        valid_mask = row != pad_token_id

        if not valid_mask.any():
            # Empty sequence - skip
            continue

        # Get document boundaries in this row
        valid_docs = row[valid_mask]
        if len(valid_docs) == 0:
            continue

        # Count tokens per document
        unique_docs = valid_docs.unique()
        for doc_id in unique_docs:
            doc_len = (valid_docs == doc_id).sum().item()
            doc_lengths.append(doc_len)
            total_tokens += doc_len
            cu_seqlens_list.append(torch.tensor([total_tokens], dtype=torch.int32, device=device))

    if len(cu_seqlens_list) == 1:
        # No valid documents
        return (
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            0
        )

    cu_seqlens = torch.cat(cu_seqlens_list)
    max_seqlen = max(doc_lengths) if doc_lengths else 0

    return cu_seqlens, not_padding, max_seqlen


def create_cu_seqlens_vectorized(
    document_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Create cu_seqlens format for Flash Attention varlen - fully vectorized.

    This format completely eliminates 2D attention mask allocation.
    For batch=32, seq=2048, this saves ~512MB of GPU memory per batch.

    Args:
        document_ids: [batch_size, seq_len] document IDs (-1 for padding)
        input_ids: [batch_size, seq_len] token IDs

    Returns:
        Dict with:
        - 'input_ids_packed': [total_tokens] concatenated non-padding tokens
        - 'cu_seqlens': [num_documents + 1] cumulative sequence lengths
        - 'max_seqlen': maximum document length
        - 'position_ids_packed': [total_tokens] position IDs (reset per doc)
        - 'indices': [total_tokens] original indices for unpacking results
        - 'batch_size': original batch size
        - 'seq_len': original sequence length

    Usage with Flash Attention:
        packed = create_cu_seqlens_vectorized(document_ids, input_ids)
        output = flash_attn_varlen_func(
            q.view(-1, num_heads, head_dim),  # Flatten to [total_tokens, heads, dim]
            k.view(-1, num_heads, head_dim),
            v.view(-1, num_heads, head_dim),
            packed['cu_seqlens'],
            packed['cu_seqlens'],
            packed['max_seqlen'],
            packed['max_seqlen'],
            causal=True
        )
    """
    batch_size, seq_len = document_ids.shape
    device = document_ids.device

    # Find non-padding positions
    not_padding = (document_ids != -1)  # [batch, seq]

    # Flatten and extract non-padding tokens
    not_padding_flat = not_padding.view(-1)  # [batch * seq]
    input_ids_flat = input_ids.view(-1)  # [batch * seq]

    # Get packed input_ids (non-padding only)
    input_ids_packed = input_ids_flat[not_padding_flat]  # [total_tokens]

    # Compute document lengths for cu_seqlens
    # Each row in document_ids may have multiple documents
    # We need to find boundaries within each row AND across rows

    # Approach: Create a unique ID for each document across the batch
    # unique_doc_id = batch_idx * max_docs_per_seq + doc_id
    # But simpler: just iterate and count (document order matters)

    # For each document, find its length
    # A document boundary is where doc_id changes (excluding padding->anything)

    # Create flattened document IDs for non-padding only
    doc_ids_flat = document_ids.view(-1)
    doc_ids_packed = doc_ids_flat[not_padding_flat]  # [total_tokens]

    if len(doc_ids_packed) == 0:
        return {
            'input_ids_packed': torch.tensor([], dtype=torch.long, device=device),
            'cu_seqlens': torch.tensor([0], dtype=torch.int32, device=device),
            'max_seqlen': 0,
            'position_ids_packed': torch.tensor([], dtype=torch.long, device=device),
            'indices': torch.tensor([], dtype=torch.long, device=device),
            'batch_size': batch_size,
            'seq_len': seq_len,
        }

    # Add batch offset to make document IDs unique across batches
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    batch_indices_flat = batch_indices.reshape(-1)  # Use reshape for non-contiguous tensors
    batch_indices_packed = batch_indices_flat[not_padding_flat]

    # Unique document ID = batch * 1000 + doc_id (assuming < 1000 docs per sequence)
    max_docs = 1000
    unique_doc_ids = batch_indices_packed * max_docs + doc_ids_packed

    # Find document boundaries (where unique_doc_id changes)
    doc_boundaries = torch.zeros(len(unique_doc_ids), dtype=torch.bool, device=device)
    doc_boundaries[0] = True  # First token starts a document
    doc_boundaries[1:] = (unique_doc_ids[1:] != unique_doc_ids[:-1])

    # Compute document lengths using boundaries
    # boundary_indices = where boundaries occur
    boundary_indices = torch.where(doc_boundaries)[0]  # [num_docs]
    # Document lengths = diff of consecutive boundaries, plus final doc length
    doc_ends = torch.cat([boundary_indices[1:], torch.tensor([len(unique_doc_ids)], device=device)])
    doc_lengths = doc_ends - boundary_indices  # [num_docs]

    # Create cu_seqlens from cumsum of lengths
    cu_seqlens = torch.zeros(len(doc_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(doc_lengths, dim=0).to(torch.int32)

    max_seqlen = int(doc_lengths.max().item()) if len(doc_lengths) > 0 else 0

    # Create position IDs (reset per document)
    # position = offset within current document
    segment_ids = torch.cumsum(doc_boundaries.long(), dim=0) - 1  # 0-indexed segment IDs
    # Position = current_idx - start_of_segment
    segment_starts = torch.zeros_like(segment_ids)
    positions = torch.arange(len(unique_doc_ids), device=device, dtype=torch.long)
    segment_starts[doc_boundaries] = positions[doc_boundaries]
    segment_starts_expanded, _ = torch.cummax(segment_starts, dim=0)
    position_ids_packed = positions - segment_starts_expanded

    # Store indices for unpacking results back to [batch, seq, ...]
    indices = torch.where(not_padding_flat)[0]

    return {
        'input_ids_packed': input_ids_packed,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
        'position_ids_packed': position_ids_packed,
        'indices': indices,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }


def create_document_position_ids(
    document_ids: torch.Tensor
) -> torch.Tensor:
    """
    Create position IDs that reset to 0 at each document boundary.

    OPTIMIZED: Fully vectorized - no Python loops, no GPU→CPU sync.

    This ensures RoPE positional embeddings restart for each document,
    preventing position information from leaking across documents.

    Args:
        document_ids: [seq_len] tensor where each position contains
                     the document ID (0, 1, 2, ...) or -1 for padding

    Returns:
        [seq_len] tensor of position IDs (0, 1, 2, ... resetting per doc)
    """
    seq_len = document_ids.size(0)
    device = document_ids.device

    if seq_len == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    # Detect document boundaries (where doc_id changes or previous was padding)
    # A new segment starts when:
    # 1. It's the first position, OR
    # 2. doc_id differs from previous position, OR
    # 3. Previous position was padding (-1)
    doc_change = torch.zeros(seq_len, dtype=torch.bool, device=device)
    doc_change[0] = True  # First position always starts a segment

    if seq_len > 1:
        # Document changes or previous was padding
        doc_change[1:] = (document_ids[1:] != document_ids[:-1]) | (document_ids[:-1] == -1)

    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=torch.long)

    # Get the start position of each segment using cummax
    # segment_starts[i] = position of the start of document containing position i
    segment_start_positions = torch.where(doc_change, positions, torch.zeros_like(positions))
    segment_starts, _ = torch.cummax(segment_start_positions, dim=0)

    # Position within document = global position - segment start position
    position_ids = positions - segment_starts

    # Padding positions get position 0
    position_ids = torch.where(document_ids == -1, torch.zeros_like(position_ids), position_ids)

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
        use_cu_seqlens: bool = True,  # Use Flash Attention varlen format (40-60% faster, 0 mask memory)
        # P2-6: Length-bucketed shuffling improves packing efficiency from ~95% to ~98%
        length_bucket_shuffle: bool = True,  # Groups similar-length sequences for better packing
        num_length_buckets: int = 8,  # Number of length buckets for grouping
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pack_sequences = pack_sequences
        self.sort_by_length = sort_by_length
        self.preserve_batch_size = preserve_batch_size  # Keep output batch size = input batch size
        self.mask_dtype = mask_dtype
        self.validate_masks = validate_masks
        self.use_cu_seqlens = use_cu_seqlens
        # P2-6: Length-bucketed shuffling for better packing efficiency
        self.length_bucket_shuffle = length_bucket_shuffle
        self.num_length_buckets = num_length_buckets

        # Statistics tracking
        self.total_tokens_before = 0
        self.total_tokens_after = 0
        self.num_batches = 0

        # Pre-allocated tensors for efficiency (avoid per-batch allocations)
        self._eos_tensor: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None

        # Buffer pool for packing (eliminates clone overhead - 2-5% speedup)
        self._buffer_pool: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._max_pool_size: int = 64  # Max buffers to keep in pool

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

    def _get_buffer_from_pool(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a buffer pair from the pool, or create new ones if pool is empty."""
        if self._buffer_pool:
            output_buffer, doc_id_buffer = self._buffer_pool.pop()
            # Reset buffers for reuse
            output_buffer.fill_(self.pad_token_id)
            doc_id_buffer.fill_(-1)
            return output_buffer, doc_id_buffer
        # Create new buffers
        return (
            torch.full((self.max_length,), self.pad_token_id, dtype=dtype),
            torch.full((self.max_length,), -1, dtype=torch.long)
        )

    def _return_buffer_to_pool(self, output_buffer: torch.Tensor, doc_id_buffer: torch.Tensor) -> None:
        """Return a buffer pair to the pool for reuse."""
        if len(self._buffer_pool) < self._max_pool_size:
            self._buffer_pool.append((output_buffer, doc_id_buffer))

    def _length_bucket_shuffle(
        self,
        sequences: List[torch.Tensor],
        lengths: List[int]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        P2-6: Group sequences by length buckets and shuffle within each bucket.

        This improves packing efficiency from ~95% to ~98% by grouping similar-length
        sequences while maintaining some randomness within each bucket.

        Groups sequences into length buckets, shuffles within each bucket, then
        concatenates buckets in random order. This achieves better packing than
        pure random order (similar lengths pack better together) while avoiding
        the training bias of pure length sorting.

        Args:
            sequences: List of input sequences
            lengths: List of sequence lengths

        Returns:
            Tuple of (shuffled_sequences, shuffled_lengths)
        """
        if not sequences or len(sequences) <= 1:
            return sequences, lengths

        # Create buckets based on length ranges
        max_len = max(lengths)
        min_len = min(lengths)
        if max_len == min_len:
            return sequences, lengths

        # Calculate bucket boundaries
        bucket_size = max(1, (max_len - min_len) // self.num_length_buckets)

        # Group indices by bucket
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket_idx = min(self.num_length_buckets - 1, (length - min_len) // bucket_size)
            buckets[bucket_idx].append(idx)

        # Shuffle indices within each bucket
        import random
        for bucket_indices in buckets.values():
            random.shuffle(bucket_indices)

        # Concatenate buckets (iterate in sorted order for deterministic grouping)
        # This groups similar lengths together for better packing
        shuffled_indices = []
        for bucket_idx in sorted(buckets.keys()):
            shuffled_indices.extend(buckets[bucket_idx])

        # Apply shuffled order
        shuffled_sequences = [sequences[i] for i in shuffled_indices]
        shuffled_lengths = [lengths[i] for i in shuffled_indices]

        return shuffled_sequences, shuffled_lengths

    def _pack_sequences_numba(
        self,
        sequences: List[torch.Tensor],
        lengths: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Numba-accelerated greedy bin-packing with document boundary tracking.

        Uses Numba JIT-compiled function for pack assignment computation,
        then assembles the actual tensors. Falls back to Python if Numba unavailable.

        Returns:
            Tuple of (packed_sequences, document_ids_list)
        """
        if not sequences:
            return [], []

        # Use Numba for pack assignment computation (the hot loop)
        lengths_np = np.array(lengths, dtype=np.int64)
        pack_ids = _compute_pack_assignments_numba(lengths_np, self.max_length)

        # Group sequences by pack ID
        num_packs = int(pack_ids.max()) + 1 if len(pack_ids) > 0 else 0
        packs: Dict[int, List[Tuple[torch.Tensor, int, int]]] = {i: [] for i in range(num_packs)}

        # Sort by length descending (same order as Numba function)
        if self.sort_by_length:
            sorted_indices = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)
        else:
            sorted_indices = list(range(len(sequences)))

        # Track document IDs within each pack
        pack_doc_counters = {i: 0 for i in range(num_packs)}

        for idx in sorted_indices:
            pack_id = pack_ids[idx]
            seq = sequences[idx]
            seq_len = lengths[idx]
            doc_id = pack_doc_counters[pack_id]
            packs[pack_id].append((seq[:seq_len], seq_len, doc_id))
            pack_doc_counters[pack_id] += 1

        # Assemble packed sequences with EOS separators
        dtype = sequences[0].dtype
        eos_tensor = self._get_eos_tensor(dtype)
        packed = []
        document_ids_list = []

        for pack_id in range(num_packs):
            parts = packs[pack_id]
            if not parts:
                continue

            # Build parts list with EOS separators
            final_parts: List[Tuple[torch.Tensor, int, int]] = []
            for i, (seq, seq_len, doc_id) in enumerate(parts):
                if i > 0:
                    # EOS belongs to previous document
                    prev_doc_id = final_parts[-1][2]
                    final_parts.append((eos_tensor, 1, prev_doc_id))
                final_parts.append((seq, seq_len, doc_id))

            # Calculate total length
            total_length = sum(length for _, length, _ in final_parts)

            # Get buffers and finalize
            output_buffer, doc_id_buffer = self._get_buffer_from_pool(dtype)
            packed_seq, doc_ids = self._finalize_pack_with_doc_ids(
                final_parts, total_length, output_buffer, doc_id_buffer
            )
            packed.append(packed_seq)
            document_ids_list.append(doc_ids)

        return packed, document_ids_list

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
                    # Get buffers from pool (avoids allocation overhead)
                    output_buffer, doc_id_buffer = self._get_buffer_from_pool(dtype)
                    packed_seq, doc_ids = self._finalize_pack_with_doc_ids(
                        current_parts, current_length, output_buffer, doc_id_buffer
                    )
                    packed.append(packed_seq)
                    document_ids_list.append(doc_ids)

                current_parts = [(seq[:seq_len], seq_len, 0)]  # Reset doc_id to 0 for new pack
                current_length = seq_len
                current_doc_id = 0

        # Don't forget the last pack
        if current_parts:
            # Get buffers from pool (avoids allocation overhead)
            output_buffer, doc_id_buffer = self._get_buffer_from_pool(dtype)
            packed_seq, doc_ids = self._finalize_pack_with_doc_ids(
                current_parts, current_length, output_buffer, doc_id_buffer
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

        # P2-6: Apply length-bucketed shuffling for better packing efficiency
        # Groups similar-length sequences together for ~3% better packing
        if self.length_bucket_shuffle and len(sequences) > 1:
            sequences, lengths = self._length_bucket_shuffle(sequences, lengths)

        # Track statistics
        self.total_tokens_before += sum(lengths)
        self.num_batches += 1

        # Pack sequences with document boundary tracking
        # Use Numba-accelerated path when available (5-10x faster for pack assignment)
        if USE_NUMBA_PACKING and NUMBA_AVAILABLE:
            try:
                packed, document_ids_list = self._pack_sequences_numba(sequences, lengths)
            except Exception:
                # Fall back to Python implementation on any error
                packed, document_ids_list = self._pack_sequences_greedy(sequences, lengths)
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

        # Labels are same as input_ids for causal LM (shifted internally by model)
        # OPTIMIZATION: Use torch.where instead of clone() + indexed assignment
        # This is a single fused operation, avoiding unnecessary tensor copy
        labels = torch.where(input_ids == self.pad_token_id, -100, input_ids)

        # Track packed tokens (using simple mask for counting)
        simple_mask = (input_ids != self.pad_token_id).long()
        self.total_tokens_after += int(simple_mask.sum())

        # === FAST PATH: cu_seqlens for Flash Attention varlen (40-60% faster) ===
        # This eliminates 512MB 2D mask allocation entirely
        if self.use_cu_seqlens:
            cu_seqlens_info = create_cu_seqlens_vectorized(document_ids, input_ids)
            position_ids = create_document_position_ids_batched(document_ids)

            return {
                'input_ids': input_ids,
                'labels': labels,
                'position_ids': position_ids,
                'document_ids': document_ids,
                # cu_seqlens for Flash Attention varlen
                'cu_seqlens': cu_seqlens_info['cu_seqlens'],
                'max_seqlen': cu_seqlens_info['max_seqlen'],
            }

        # === Standard path: 2D attention masks ===
        # Create 2D attention mask with document boundaries
        # This BLOCKS cross-document attention - critical for coherent learning!
        # OPTIMIZED: Use batched functions to eliminate Python for loop (15-25% speedup)
        attention_mask_2d = create_document_attention_mask_batched(document_ids, dtype=self.mask_dtype)
        position_ids = create_document_position_ids_batched(document_ids)

        # Validate masks if enabled (catches bugs early but adds overhead)
        if self.validate_masks:
            batch_size, seq_len = input_ids.shape
            validate_attention_mask(
                attention_mask_2d,
                expected_shape=(batch_size, seq_len, seq_len),
                expected_dtype=self.mask_dtype,
                name="packed_attention_mask"
            )
            # FIX: Also validate document boundaries are correctly enforced
            # This catches silent bugs where cross-document attention occurs
            validate_document_boundaries(
                attention_mask_2d,
                document_ids,
                name="packed_attention_mask"
            )

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
        # OPTIMIZATION: Use torch.where instead of clone() + indexed assignment
        labels = torch.where(input_ids == self.pad_token_id, -100, input_ids)

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
        use_cu_seqlens: bool = True,  # Use Flash Attention varlen format (40-60% faster, 0 mask memory)
        length_bucket_shuffle: bool = True,  # P2-6: Groups similar-length sequences
        num_length_buckets: int = 8,
    ):
        super().__init__(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            pack_sequences=True,
            preserve_batch_size=preserve_batch_size,
            mask_dtype=mask_dtype,
            validate_masks=validate_masks,
            use_cu_seqlens=use_cu_seqlens,
            length_bucket_shuffle=length_bucket_shuffle,
            num_length_buckets=num_length_buckets,
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

        # Pack each bin separately (use Numba when available)
        for bin_idx in sorted(binned.keys()):
            indices = binned[bin_idx]
            bin_sequences = [sequences[i] for i in indices]
            bin_lengths = [lengths[i] for i in indices]

            # Use Numba-accelerated path when available
            if USE_NUMBA_PACKING and NUMBA_AVAILABLE:
                try:
                    packed, doc_ids = self._pack_sequences_numba(bin_sequences, bin_lengths)
                except Exception:
                    packed, doc_ids = self._pack_sequences_greedy(bin_sequences, bin_lengths)
            else:
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
            # Use Numba-accelerated path when available
            if USE_NUMBA_PACKING and NUMBA_AVAILABLE:
                try:
                    packed, document_ids_list = self._pack_sequences_numba(sequences, lengths)
                except Exception:
                    packed, document_ids_list = self._pack_sequences_greedy(sequences, lengths)
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

        # Labels are same as input_ids for causal LM (shifted internally by model)
        # OPTIMIZATION: Use torch.where instead of clone() + indexed assignment
        # This is a single fused operation, avoiding unnecessary tensor copy
        labels = torch.where(input_ids == self.pad_token_id, -100, input_ids)

        # Track packed tokens (using simple mask for counting)
        simple_mask = (input_ids != self.pad_token_id).long()
        self.total_tokens_after += int(simple_mask.sum())

        # === FAST PATH: cu_seqlens for Flash Attention varlen (40-60% faster) ===
        # This eliminates 512MB 2D mask allocation entirely
        if self.use_cu_seqlens:
            cu_seqlens_info = create_cu_seqlens_vectorized(document_ids, input_ids)
            position_ids = create_document_position_ids_batched(document_ids)

            return {
                'input_ids': input_ids,
                'labels': labels,
                'position_ids': position_ids,
                'document_ids': document_ids,
                # cu_seqlens for Flash Attention varlen
                'cu_seqlens': cu_seqlens_info['cu_seqlens'],
                'max_seqlen': cu_seqlens_info['max_seqlen'],
            }

        # === Standard path: 2D attention masks ===
        # Create 2D attention mask with document boundaries
        # This BLOCKS cross-document attention - critical for coherent learning!
        # OPTIMIZED: Use batched functions to eliminate Python for loop (15-25% speedup)
        attention_mask_2d = create_document_attention_mask_batched(document_ids, dtype=self.mask_dtype)
        position_ids = create_document_position_ids_batched(document_ids)

        # Validate masks if enabled (catches bugs early but adds overhead)
        if self.validate_masks:
            batch_size, seq_len = input_ids.shape
            validate_attention_mask(
                attention_mask_2d,
                expected_shape=(batch_size, seq_len, seq_len),
                expected_dtype=self.mask_dtype,
                name="dynamic_packed_attention_mask"
            )

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
