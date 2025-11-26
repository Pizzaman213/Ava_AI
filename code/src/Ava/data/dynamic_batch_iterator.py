"""
Dynamic Batch Iterator for Memory-Aware Batch Size Adjustment

This module provides a wrapper around PyTorch DataLoader that dynamically
adjusts batch sizes based on GPU memory utilization, implementing the
dynamic_batching configuration from YAML.

The approach uses mini-batch concatenation:
- Base DataLoader yields batches of min_batch_size
- Iterator concatenates multiple mini-batches based on memory availability
- Yields batch sizes that are multiples of min_batch_size (64, 128, 192, 256)

This works with IterableDataset (unlike batch_sampler which requires map-style).
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

# Import from the training optimizations module
# Note: This is a cross-package import, handled at runtime
try:
    from src.Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
    )
except ImportError:
    # Fallback for different import contexts
    from Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
    )

logger = logging.getLogger(__name__)


class DynamicBatchIterator:
    """
    Wraps a DataLoader and yields dynamically-sized batches by concatenating
    multiple mini-batches based on GPU memory utilization.

    This enables dynamic batch sizing with IterableDataset, which doesn't
    support PyTorch's native batch_sampler.

    The base DataLoader should be created with batch_size=min_batch_size.
    This iterator will concatenate 1-N mini-batches to create larger batches
    when GPU memory allows.

    Example:
        ```python
        # Create base DataLoader with min_batch_size
        base_loader = DataLoader(dataset, batch_size=64, ...)

        # Create scheduler from config
        scheduler = create_dynamic_batch_scheduler(config_dict)

        # Wrap with dynamic iterator
        dynamic_loader = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        # Use in training loop
        for batch in dynamic_loader:
            # batch size varies: 64, 128, 192, or 256
            loss = model(batch)
        ```

    Args:
        base_dataloader: DataLoader with batch_size=min_batch_size
        scheduler: DynamicBatchScheduler for memory-aware decisions
        min_batch_size: Minimum batch size (should match base_dataloader.batch_size)
        max_batch_size: Maximum batch size (must be multiple of min_batch_size)
    """

    def __init__(
        self,
        base_dataloader: DataLoader,
        scheduler: DynamicBatchScheduler,
        min_batch_size: int = 64,
        max_batch_size: int = 256,
    ):
        self.base_dataloader = base_dataloader
        self.scheduler = scheduler
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Validate max is multiple of min
        if max_batch_size % min_batch_size != 0:
            logger.warning(
                f"max_batch_size ({max_batch_size}) is not a multiple of "
                f"min_batch_size ({min_batch_size}). Adjusting to nearest multiple."
            )
            self.max_batch_size = (max_batch_size // min_batch_size) * min_batch_size

        self.max_multiplier = self.max_batch_size // self.min_batch_size

        # Statistics
        self._step_count = 0
        self._total_batches_yielded = 0
        self._batch_size_history: List[int] = []

        logger.info(
            f"DynamicBatchIterator initialized: "
            f"min={self.min_batch_size}, max={self.max_batch_size}, "
            f"max_multiplier={self.max_multiplier}"
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over base DataLoader, yielding dynamically-sized batches.

        Accumulates mini-batches in a buffer and yields when the target
        multiplier (based on memory) is reached.
        """
        buffer: List[Dict[str, torch.Tensor]] = []

        for mini_batch in self.base_dataloader:
            buffer.append(mini_batch)

            # Get target batch size from scheduler
            target_size = self.scheduler.get_current_batch_size(self._step_count)
            # Use rounding instead of floor division to handle decimals (e.g., 42/32=1.3 -> 1)
            target_multiplier = max(1, round(target_size / self.min_batch_size))
            target_multiplier = min(target_multiplier, self.max_multiplier)

            # Yield when we have enough mini-batches
            if len(buffer) >= target_multiplier:
                concatenated = self._concatenate_batches(buffer)
                actual_batch_size = concatenated['input_ids'].size(0)

                # Track statistics
                self._batch_size_history.append(actual_batch_size)
                self._total_batches_yielded += 1

                # Log batch size periodically (every 100 batches or on change)
                if self._total_batches_yielded % 100 == 0 or (
                    len(self._batch_size_history) >= 2 and
                    self._batch_size_history[-1] != self._batch_size_history[-2]
                ):
                    # Get memory stats for debugging
                    mem_info = ""
                    if hasattr(self.scheduler, 'get_memory_stats'):
                        mem = self.scheduler.get_memory_stats()
                        mem_info = f" | GPU mem: {mem.get('utilization', 0):.1%}"
                    print(f" [DynamicBatch] Step {self._step_count}: BS={actual_batch_size} "
                          f"(target={target_size}, multiplier={target_multiplier}x){mem_info}")

                yield concatenated
                buffer = []

                # Update scheduler with current memory state
                self._step_count += 1
                new_size = self.scheduler.step(self._step_count)

                if new_size is not None:
                    logger.debug(
                        f"Step {self._step_count}: Batch size adjusted to {new_size} "
                        f"(multiplier: {new_size // self.min_batch_size}x)"
                    )

        # Yield remaining mini-batches at end of epoch
        if buffer:
            concatenated = self._concatenate_batches(buffer)
            self._batch_size_history.append(concatenated['input_ids'].size(0))
            self._total_batches_yielded += 1
            yield concatenated

    def _concatenate_batches(
        self, batches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate multiple mini-batches into one large batch.

        Args:
            batches: List of mini-batches (dicts with input_ids, attention_mask, labels)

        Returns:
            Single concatenated batch
        """
        if len(batches) == 1:
            return batches[0]

        # Concatenate along batch dimension (dim=0)
        result = {}

        # Handle standard keys
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batches[0]:
                result[key] = torch.cat([b[key] for b in batches], dim=0)

        # Handle any additional keys (e.g., position_ids, token_type_ids)
        for key in batches[0].keys():
            if key not in result:
                try:
                    result[key] = torch.cat([b[key] for b in batches], dim=0)
                except (TypeError, RuntimeError):
                    # Non-concatenatable value, take from first batch
                    result[key] = batches[0][key]

        return result

    def __len__(self) -> int:
        """
        Return estimated number of batches.

        Note: This is an approximation since batch sizes vary dynamically.
        Returns the minimum possible batches (if always using max_batch_size).
        """
        base_len = len(self.base_dataloader)
        return max(1, base_len // self.max_multiplier)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about dynamic batching performance."""
        if not self._batch_size_history:
            return {
                'total_batches': 0,
                'avg_batch_size': 0,
                'min_batch_size_used': 0,
                'max_batch_size_used': 0,
                'batch_size_variance': 0,
            }

        import statistics

        return {
            'total_batches': self._total_batches_yielded,
            'avg_batch_size': statistics.mean(self._batch_size_history),
            'min_batch_size_used': min(self._batch_size_history),
            'max_batch_size_used': max(self._batch_size_history),
            'batch_size_variance': statistics.variance(self._batch_size_history)
                if len(self._batch_size_history) > 1 else 0,
            'scheduler_stats': self.scheduler.get_statistics(),
        }

    def log_summary(self) -> None:
        """Log a summary of dynamic batching performance."""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("Dynamic Batch Iterator Summary")
        logger.info("=" * 60)
        logger.info(f"Total batches yielded: {stats['total_batches']}")
        logger.info(f"Average batch size: {stats['avg_batch_size']:.1f}")
        logger.info(f"Batch size range: [{stats['min_batch_size_used']}, {stats['max_batch_size_used']}]")
        logger.info(f"Batch size variance: {stats['batch_size_variance']:.2f}")
        logger.info("=" * 60)

        # Also log scheduler summary
        self.scheduler.log_summary()

    def reset_statistics(self) -> None:
        """Reset statistics for new epoch."""
        self._batch_size_history = []
        self._total_batches_yielded = 0
        # Don't reset step_count - scheduler needs continuous steps


def create_dynamic_batch_iterator(
    base_dataloader: DataLoader,
    config_dict: Dict[str, Any],
) -> DynamicBatchIterator:
    """
    Factory function to create a DynamicBatchIterator from config.

    Args:
        base_dataloader: DataLoader with batch_size=min_batch_size
        config_dict: Configuration dictionary containing dynamic_batching section

    Returns:
        DynamicBatchIterator wrapping the base DataLoader
    """
    # Extract dynamic batching config
    db_config = config_dict.get('dynamic_batching', {})

    if not db_config.get('enabled', False):
        logger.warning(
            "create_dynamic_batch_iterator called but dynamic_batching.enabled=False. "
            "Returning base dataloader wrapped in pass-through iterator."
        )

    # Get batch size parameters
    min_batch_size = db_config.get('min_batch_size', 64)
    max_batch_size = db_config.get('max_batch_size', 256)

    # Create scheduler
    scheduler = create_dynamic_batch_scheduler(config_dict)

    return DynamicBatchIterator(
        base_dataloader=base_dataloader,
        scheduler=scheduler,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )


class PassThroughBatchIterator:
    """
    Pass-through iterator that doesn't modify batches.

    Used when dynamic batching is disabled but code expects a DynamicBatchIterator.
    """

    def __init__(self, base_dataloader: DataLoader):
        self.base_dataloader = base_dataloader
        self._total_batches_yielded = 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.base_dataloader:
            self._total_batches_yielded += 1
            yield batch

    def __len__(self) -> int:
        return len(self.base_dataloader)

    def get_statistics(self) -> Dict[str, Any]:
        return {'total_batches': self._total_batches_yielded, 'mode': 'pass-through'}

    def log_summary(self) -> None:
        logger.info(f"PassThroughBatchIterator: {self._total_batches_yielded} batches yielded")

    def reset_statistics(self) -> None:
        self._total_batches_yielded = 0
