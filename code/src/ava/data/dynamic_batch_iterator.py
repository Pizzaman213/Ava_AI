"""
Dynamic Batch Iterator - Simple, Understandable Implementation

This module wraps a PyTorch DataLoader to provide dynamic batch sizing.
Mini-batches are accumulated and concatenated based on the scheduler's
current batch size target.

Key Classes:
    - DynamicBatchIterator: Main iterator with dynamic batch accumulation
    - PassThroughIterator: No-op wrapper when dynamic batching is disabled

Usage:
    from ava.optimizations.dynamic_batching import create_dynamic_batch_scheduler

    scheduler = create_dynamic_batch_scheduler(config)
    iterator = DynamicBatchIterator(dataloader, scheduler)

    for batch in iterator:
        # batch size varies based on memory
        loss = model(batch)
"""

import logging
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Union

import torch
from torch.utils.data import DataLoader

from ava.core.data_utils import extract_dynamic_batching_config
from ava.optimizations.dynamic_batching import (
    DynamicBatchScheduler,
    create_dynamic_batch_scheduler,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Dynamic Batch Iterator
# =============================================================================

class DynamicBatchIterator:
    """
    Wraps a DataLoader and yields dynamically-sized batches.

    How it works:
        1. Base DataLoader yields small mini-batches (e.g., batch_size=16)
        2. This iterator accumulates mini-batches in a buffer
        3. When buffer reaches target size, concatenate and yield
        4. Target size is controlled by DynamicBatchScheduler

    Simple, clear logic - no complex token budgets or unified strategies.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        scheduler: DynamicBatchScheduler,
        min_batch_size: int = 16,
        max_batch_size: int = 256,
    ):
        """
        Initialize dynamic batch iterator.

        Args:
            dataloader: Base DataLoader with small batch_size (e.g., 16)
            scheduler: DynamicBatchScheduler for batch size decisions
            min_batch_size: Minimum batch size (should match dataloader.batch_size)
            max_batch_size: Maximum batch size to accumulate
        """
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Statistics
        self._step_count = 0
        self._batches_yielded = 0
        self._samples_processed = 0
        # MEMORY FIX: Use bounded deque instead of unbounded list
        # Prevents memory growth during long training runs
        self._batch_sizes: Deque[int] = deque(maxlen=1000)

        # Instance-level buffer for OOM recovery
        # Can be cleared via clear_buffer() when batch size is reduced
        self._buffer: List[Dict[str, torch.Tensor]] = []
        self._buffer_samples = 0

        logger.info(
            f"DynamicBatchIterator initialized: "
            f"min={min_batch_size}, max={max_batch_size}"
        )

    def clear_buffer(self) -> None:
        """
        Clear the internal accumulation buffer.

        Call this after OOM to discard any accumulated samples that may
        exceed the new (reduced) batch size target.
        """
        discarded = self._buffer_samples
        self._buffer = []
        self._buffer_samples = 0
        if discarded > 0:
            logger.info(f"Cleared iterator buffer ({discarded} samples discarded)")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataloader, yielding dynamically-sized batches.

        Simple logic:
            1. Accumulate mini-batches in buffer
            2. Check if buffer size >= target batch size
            3. If yes, concatenate and yield
            4. Update scheduler and repeat
        """
        # Use instance buffer (can be cleared on OOM)
        self._buffer = []
        self._buffer_samples = 0

        for mini_batch in self.dataloader:
            # Get mini-batch size
            batch_size = self._get_batch_size(mini_batch)

            # Add to buffer
            self._buffer.append(mini_batch)
            self._buffer_samples += batch_size

            # Check if we should yield
            target_size = self.scheduler.get_batch_size()

            if self._buffer_samples >= target_size:
                # Concatenate and yield
                combined = self._concatenate(self._buffer)
                actual_size = self._get_batch_size(combined)

                # SAFETY: If actual batch exceeds target significantly, truncate
                # This prevents oversized batches after constraint changes
                max_allowed = int(target_size * 1.2)  # Allow 20% overshoot for mini-batch granularity
                if actual_size > max_allowed and actual_size > self.min_batch_size:
                    # Truncate the batch
                    logger.warning(
                        f"Batch size {actual_size} exceeds max {max_allowed}, truncating"
                    )
                    for key, value in combined.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 0:
                            combined[key] = value[:max_allowed]
                    actual_size = max_allowed

                # Update statistics
                self._batches_yielded += 1
                self._samples_processed += actual_size
                self._batch_sizes.append(actual_size)

                # Log periodically
                if self._batches_yielded % 100 == 0:
                    mem = self.scheduler.get_memory_stats()
                    logger.debug(
                        f"[Batch {self._batches_yielded}] "
                        f"BS={actual_size}, target={target_size}, "
                        f"mem={mem['smoothed']:.1%}"
                    )

                yield combined

                # Reset buffer
                self._buffer = []
                self._buffer_samples = 0

                # Update scheduler (may adjust batch size)
                self._step_count += 1
                self.scheduler.step(self._step_count)

        # Yield remaining samples at end of epoch
        if self._buffer:
            combined = self._concatenate(self._buffer)
            actual_size = self._get_batch_size(combined)
            self._batches_yielded += 1
            self._samples_processed += actual_size
            self._batch_sizes.append(actual_size)
            yield combined
            self._buffer = []
            self._buffer_samples = 0

    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Get number of samples in a batch."""
        if 'input_ids' in batch:
            return batch['input_ids'].size(0)
        elif 'attention_mask' in batch:
            return batch['attention_mask'].size(0)
        # Fallback: try first tensor
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return v.size(0)
        return 1

    def _concatenate(self, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Concatenate multiple mini-batches into one batch.

        Args:
            batches: List of batch dictionaries

        Returns:
            Single concatenated batch
        """
        if len(batches) == 1:
            return batches[0]

        result = {}

        # Get all keys from first batch
        keys = list(batches[0].keys())

        for key in keys:
            values = [b[key] for b in batches if key in b]

            if not values:
                continue

            # Check if all values are tensors
            if all(isinstance(v, torch.Tensor) for v in values):
                # Validate shapes are compatible before concatenation
                shapes = [v.shape for v in values]
                non_batch_shapes = [s[1:] for s in shapes]  # Shapes excluding batch dim

                if len(set(non_batch_shapes)) > 1:
                    # Shape mismatch - log warning with details
                    logger.warning(
                        f"Shape mismatch for key '{key}': {shapes}. "
                        f"Using first batch only (data loss: {len(values)-1} batches)."
                    )
                    result[key] = values[0]
                    continue

                try:
                    result[key] = torch.cat(values, dim=0)
                except (RuntimeError, TypeError) as e:
                    # Log the failure with context
                    logger.warning(
                        f"Batch concatenation failed for key '{key}': {e}. "
                        f"Shapes: {shapes}. Using first batch only."
                    )
                    result[key] = values[0]
            else:
                # Non-tensor value - use first
                result[key] = values[0]

        return result

    def __len__(self) -> int:
        """
        Estimate number of batches.

        This is approximate since batch sizes vary dynamically.
        """
        try:
            base_len = len(self.dataloader)
            base_batch = getattr(self.dataloader, 'batch_size', self.min_batch_size) or self.min_batch_size
            total_samples = base_len * base_batch

            # Estimate based on current target batch size
            target = self.scheduler.get_batch_size()
            return max(1, total_samples // target)
        except TypeError:
            # IterableDataset may not have __len__
            return self._batches_yielded + 1000

    def set_batch_size(self, batch_size: int) -> None:
        """
        Manually set target batch size.

        Args:
            batch_size: New target batch size
        """
        self.scheduler.set_batch_size(batch_size)

    def record_oom(self) -> int:
        """
        Record OOM event and reduce batch size.

        Returns:
            New (reduced) batch size
        """
        return self.scheduler.record_oom()

    def set_batch_controller(self, controller: Any) -> None:
        """
        Connect BatchSizeController as constraint provider.

        This links the DynamicBatchScheduler to the central batch size
        authority (BatchSizeController), preventing the scheduler from
        increasing batch size beyond known-safe values after OOM events.

        Args:
            controller: BatchSizeController instance (or any object with
                       max_batch_size property)
        """
        if hasattr(self.scheduler, 'set_constraint_provider'):
            self.scheduler.set_constraint_provider(controller)
            logger.info(
                f"BatchController connected to DynamicBatchIterator, "
                f"effective max={self.scheduler._get_effective_max()}"
            )
        else:
            logger.warning(
                "Scheduler does not support constraint provider - "
                "batch size limits may not be enforced"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get iterator statistics."""
        avg_batch = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0

        return {
            "batches_yielded": self._batches_yielded,
            "samples_processed": self._samples_processed,
            "avg_batch_size": avg_batch,
            "min_batch_used": min(self._batch_sizes) if self._batch_sizes else 0,
            "max_batch_used": max(self._batch_sizes) if self._batch_sizes else 0,
            "current_target": self.scheduler.get_batch_size(),
            "scheduler": self.scheduler.get_statistics(),
        }

    def log_summary(self) -> None:
        """Log a summary of iterator performance."""
        stats = self.get_statistics()
        logger.info("=" * 60)
        logger.info("DYNAMIC BATCH ITERATOR SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Batches yielded: {stats['batches_yielded']}")
        logger.info(f"Samples processed: {stats['samples_processed']:,}")
        logger.info(f"Avg batch size: {stats['avg_batch_size']:.1f}")
        logger.info(f"Batch size range: [{stats['min_batch_used']}, {stats['max_batch_used']}]")
        logger.info("=" * 60)
        self.scheduler.log_summary()

    def reset_statistics(self) -> None:
        """Reset statistics for new epoch."""
        self._batches_yielded = 0
        self._samples_processed = 0
        self._batch_sizes.clear()  # Clear deque, preserves maxlen

    @property
    def current_batch_size(self) -> int:
        """Get current target batch size."""
        return self.scheduler.get_batch_size()


# =============================================================================
# Pass-Through Iterator
# =============================================================================

class PassThroughIterator:
    """
    No-op iterator that passes through batches unchanged.

    Used when dynamic batching is disabled but code expects a
    DynamicBatchIterator interface.
    """

    def __init__(self, dataloader: DataLoader):
        """
        Initialize pass-through iterator.

        Args:
            dataloader: DataLoader to wrap
        """
        self.dataloader = dataloader
        self._batches_yielded = 0
        self._samples_processed = 0

        # Get batch size from dataloader
        self.current_batch_size = getattr(dataloader, 'batch_size', 1) or 1

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Pass through batches unchanged."""
        for batch in self.dataloader:
            self._batches_yielded += 1

            # Count samples
            if isinstance(batch, dict) and 'input_ids' in batch:
                self._samples_processed += batch['input_ids'].size(0)

            yield batch

    def __len__(self) -> int:
        """Return dataloader length."""
        try:
            return len(self.dataloader)
        except TypeError:
            return self._batches_yielded + 1000

    def set_batch_size(self, _batch_size: int) -> None:
        """No-op for pass-through."""
        pass

    def record_oom(self) -> int:
        """No-op for pass-through."""
        return self.current_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "batches_yielded": self._batches_yielded,
            "samples_processed": self._samples_processed,
            "mode": "pass-through",
        }

    def log_summary(self) -> None:
        """Log summary."""
        logger.info(
            f"PassThroughIterator: {self._batches_yielded} batches, "
            f"{self._samples_processed:,} samples"
        )

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._batches_yielded = 0
        self._samples_processed = 0


# =============================================================================
# Factory Function
# =============================================================================

def create_dynamic_batch_iterator(
    dataloader: DataLoader,
    config_dict: Dict[str, Any],
) -> Union[DynamicBatchIterator, PassThroughIterator]:
    """
    Create a dynamic batch iterator from config.

    Args:
        dataloader: Base DataLoader with small batch_size
        config_dict: Configuration dictionary

    Returns:
        DynamicBatchIterator if enabled, PassThroughIterator otherwise
    """
    # CONFIG FIX: Use shared utility to handle multiple config paths
    # This consolidates the 3 different config locations into one lookup
    db_config = extract_dynamic_batching_config(config_dict) or {}

    # Check if enabled
    if not db_config.get('enabled', False):
        logger.info("Dynamic batching disabled - using pass-through")
        return PassThroughIterator(dataloader)

    # Get batch size bounds
    min_batch = db_config.get('min_batch_size', 16)
    max_batch = db_config.get('max_batch_size', 256)

    # Create scheduler
    scheduler = create_dynamic_batch_scheduler(config_dict)

    return DynamicBatchIterator(
        dataloader=dataloader,
        scheduler=scheduler,
        min_batch_size=min_batch,
        max_batch_size=max_batch,
    )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These aliases provide backward compatibility with old code
TokenBudgetBatchIterator = DynamicBatchIterator
UnifiedBatchingConfig = None  # Removed - use DynamicBatchConfig instead
UnifiedBatchingStrategy = None  # Removed - functionality merged into scheduler
