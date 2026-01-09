"""
Unified Collator Module for Data Loading.

This module consolidates collation utilities:
- BasePaddingCollator: Common padding logic for all collators
- DynamicPaddingCollator: Pads to batch max (from indexed.py)
- SequencePackingCollator: Bin-packing for zero padding (from packing.py)
- ConversationBatchCollator: Conversation-specific (from conversation.py)

Usage:
    from ava.data.collators import DynamicPaddingCollator, SequencePackingCollator

    # Dynamic padding (pads to batch max)
    collator = DynamicPaddingCollator(pad_token_id=0, max_length=2048)

    # Sequence packing (zero padding waste)
    packing_collator = SequencePackingCollator(
        max_length=2048,
        pad_token_id=0,
        eos_token_id=1,
    )
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Base Collator Interface
# ============================================================================

class BaseCollator(ABC):
    """
    Abstract base class for all collators.

    Provides a common interface and shared utilities.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: int = 2048,
        padding_side: str = 'right',
    ):
        """
        Initialize base collator.

        Args:
            pad_token_id: Token ID used for padding
            max_length: Maximum sequence length (safety cap)
            padding_side: Where to add padding ('right' or 'left')
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding_side = padding_side

        # Statistics tracking
        self._total_tokens = 0
        self._padding_tokens = 0
        self._batch_count = 0

    @abstractmethod
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        pass

    def get_efficiency(self) -> float:
        """Get padding efficiency (1.0 = no padding waste)."""
        if self._total_tokens == 0:
            return 1.0
        return 1.0 - (self._padding_tokens / self._total_tokens)

    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self._total_tokens = 0
        self._padding_tokens = 0
        self._batch_count = 0

    def _pad_tensor(
        self,
        tensor: torch.Tensor,
        target_length: int,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """
        Pad a tensor to target length.

        Args:
            tensor: Input tensor [seq_len] or [batch, seq_len]
            target_length: Target sequence length
            pad_value: Value to use for padding

        Returns:
            Padded tensor
        """
        if tensor.dim() == 1:
            current_length = tensor.size(0)
            if current_length >= target_length:
                return tensor[:target_length]

            pad_length = target_length - current_length

            if self.padding_side == 'right':
                return torch.cat([
                    tensor,
                    torch.full((pad_length,), pad_value, dtype=tensor.dtype)
                ])
            else:
                return torch.cat([
                    torch.full((pad_length,), pad_value, dtype=tensor.dtype),
                    tensor
                ])
        else:
            raise NotImplementedError("Batch padding not implemented in base class")


# ============================================================================
# Dynamic Padding Collator
# ============================================================================

class DynamicPaddingCollator(BaseCollator):
    """
    Collator that pads to batch maximum, not global maximum.

    With LengthBinnedSampler, sequences in each batch have similar lengths,
    so padding waste is typically only ~5% instead of 50-80% with fixed padding.

    This is the primary collator for indexed datasets.
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with dynamic padding to batch max.

        SPEED OPTIMIZED: Single-pass approach with minimal per-item overhead.
        """
        if not batch:
            return {}

        # OPTIMIZATION: Fast path - assume no None items (common case)
        # Only filter if needed to avoid list creation overhead
        if batch[0] is None or batch[-1] is None:
            batch = [b for b in batch if b is not None]
            if not batch:
                return {}

        batch_size = len(batch)

        # OPTIMIZATION: Single-pass length extraction with fast path for 'length' field
        # Avoid isinstance checks by using getattr with fallback
        first_item = batch[0]
        has_length_field = 'length' in first_item

        if has_length_field:
            # Fast path: length field present (pre-tokenized data)
            lengths = [item['length'] for item in batch]
        else:
            # Slower path: compute from input_ids
            lengths = [len(item['input_ids']) for item in batch]

        max_len = min(max(lengths), self.max_length)

        # Pre-allocate output tensors
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        # OPTIMIZATION: Check padding side once outside loop
        is_right_pad = self.padding_side == 'right'

        # OPTIMIZATION: Single-pass fill with minimal type checks
        # Assume tensors (standard case), handle lists only on AttributeError
        total_padding = 0
        for i, item in enumerate(batch):
            seq_len = min(lengths[i], max_len)
            ids = item['input_ids']

            # Fast tensor slice (avoid isinstance)
            try:
                ids_slice = ids[:seq_len]
            except TypeError:
                # Fallback for list input
                ids_slice = torch.tensor(ids[:seq_len], dtype=torch.long)

            # Get or create attention mask
            mask = item.get('attention_mask')
            if mask is None:
                # Create ones mask inline (common case for pre-tokenized data)
                mask_slice = attention_mask.new_ones(seq_len)
            else:
                try:
                    mask_slice = mask[:seq_len]
                except TypeError:
                    mask_slice = torch.tensor(mask[:seq_len], dtype=torch.long)

            # Get or use input_ids as labels
            lab = item.get('labels')
            if lab is None:
                lab_slice = ids_slice.clone() if hasattr(ids_slice, 'clone') else torch.tensor(ids_slice, dtype=torch.long)
            else:
                try:
                    lab_slice = lab[:seq_len]
                except TypeError:
                    lab_slice = torch.tensor(lab[:seq_len], dtype=torch.long)

            # Fill tensors (branch once on padding side)
            if is_right_pad:
                input_ids[i, :seq_len] = ids_slice
                attention_mask[i, :seq_len] = mask_slice
                labels[i, :seq_len] = lab_slice
            else:
                offset = max_len - seq_len
                input_ids[i, offset:] = ids_slice
                attention_mask[i, offset:] = mask_slice
                labels[i, offset:] = lab_slice

            total_padding += max_len - seq_len

        # Track efficiency (single computation at end)
        self._total_tokens += batch_size * max_len
        self._padding_tokens += total_padding
        self._batch_count += 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================================
# Fixed Padding Collator
# ============================================================================

class FixedPaddingCollator(BaseCollator):
    """
    Collator that always pads to max_length.

    Simpler but less efficient than DynamicPaddingCollator.
    Use when you need consistent tensor shapes across batches.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with fixed padding to max_length."""
        if not batch:
            return {}

        batch = [b for b in batch if b is not None]
        if not batch:
            return {}

        batch_size = len(batch)

        # Pre-allocate tensors at max_length
        input_ids = torch.full((batch_size, self.max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        labels = torch.full((batch_size, self.max_length), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            ids = item['input_ids']
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            seq_len = min(len(ids), self.max_length)

            mask = item.get('attention_mask', torch.ones(seq_len, dtype=torch.long))
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)

            lab = item.get('labels', ids.clone())
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab)

            if self.padding_side == 'right':
                input_ids[i, :seq_len] = ids[:seq_len]
                attention_mask[i, :seq_len] = mask[:seq_len]
                labels[i, :seq_len] = lab[:seq_len]
            else:
                offset = self.max_length - seq_len
                input_ids[i, offset:] = ids[:seq_len]
                attention_mask[i, offset:] = mask[:seq_len]
                labels[i, offset:] = lab[:seq_len]

            self._padding_tokens += self.max_length - seq_len
            self._total_tokens += self.max_length

        self._batch_count += 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================================
# Re-exports from specialized modules
# ============================================================================

# Import and re-export packing collators
try:
    from .packing import (
        SequencePackingCollator,
        DynamicSequencePackingCollator,
        create_packing_collator,
        create_document_attention_mask,
        create_document_position_ids,
    )
except ImportError:
    SequencePackingCollator = None
    DynamicSequencePackingCollator = None
    create_packing_collator = None
    create_document_attention_mask = None
    create_document_position_ids = None

# Import and re-export conversation collator
try:
    from .conversation import ConversationBatchCollator
except ImportError:
    ConversationBatchCollator = None


# ============================================================================
# Factory Function
# ============================================================================

def create_collator(
    mode: str = 'dynamic',
    pad_token_id: int = 0,
    max_length: int = 2048,
    eos_token_id: Optional[int] = None,
    **kwargs,
) -> BaseCollator:
    """
    Create a collator based on mode.

    Args:
        mode: Collation mode:
            - 'dynamic': Pads to batch max (default)
            - 'fixed': Pads to max_length always
            - 'packing': Sequence packing (zero padding)
        pad_token_id: Padding token ID
        max_length: Maximum sequence length
        eos_token_id: EOS token ID (required for packing)
        **kwargs: Additional arguments for specific collators

    Returns:
        Configured collator instance

    Example:
        >>> collator = create_collator('dynamic', pad_token_id=0)
        >>> collator = create_collator('packing', eos_token_id=1)
    """
    if mode == 'dynamic':
        return DynamicPaddingCollator(
            pad_token_id=pad_token_id,
            max_length=max_length,
            padding_side=kwargs.get('padding_side', 'right'),
        )
    elif mode == 'fixed':
        return FixedPaddingCollator(
            pad_token_id=pad_token_id,
            max_length=max_length,
            padding_side=kwargs.get('padding_side', 'right'),
        )
    elif mode == 'packing':
        if SequencePackingCollator is None:
            raise ImportError("Packing collator not available")
        if eos_token_id is None:
            raise ValueError("eos_token_id required for packing mode")
        return SequencePackingCollator(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown collation mode: {mode}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    'BaseCollator',
    # Padding collators
    'DynamicPaddingCollator',
    'FixedPaddingCollator',
    # Packing collators (from packing.py)
    'SequencePackingCollator',
    'DynamicSequencePackingCollator',
    'create_packing_collator',
    'create_document_attention_mask',
    'create_document_position_ids',
    # Conversation collator (from conversation.py)
    'ConversationBatchCollator',
    # Factory
    'create_collator',
]
