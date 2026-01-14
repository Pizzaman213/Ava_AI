"""Out-of-memory recovery utilities.

This module provides comprehensive OOM recovery that resets all relevant state
to prevent cascading errors after batch size reduction.

Key Features:
- CUDA cache clearing
- Optimizer state reset (momentum buffers)
- Prefetcher queue flushing
- DataLoader recreation with new batch size
- Activation cache clearing

Example:
    >>> oom_manager = OOMRecoveryManager(training_context)
    >>> try:
    ...     forward_pass(batch)
    ... except RuntimeError as e:
    ...     if "out of memory" in str(e):
    ...         new_bs = oom_manager.recover_from_oom(current_batch_size)
"""

import logging
import queue
import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..training.context import TrainingContext

logger = logging.getLogger(__name__)


class OOMRecoveryManager:
    """Manages complete state reset after OOM errors.

    When an OOM error occurs, simply reducing the batch size is not enough.
    Many components cache state based on the old batch size:
    - Optimizer momentum buffers
    - Prefetcher queue with old-size batches
    - DataLoader workers with cached samples
    - Activation checkpointing buffers
    - CUDA memory allocator fragmentation

    This class performs a complete reset to ensure stable training after OOM.

    Attributes:
        context: Training context with model, optimizer, data_manager, etc.
        min_batch_size: Minimum allowed batch size (default: 1)
        reduction_factor: Factor to reduce batch size by (default: 0.75)
        max_recovery_attempts: Maximum OOM recovery attempts before giving up (default: 3)
    """

    def __init__(
        self,
        context: "TrainingContext",
        min_batch_size: int = 1,
        reduction_factor: float = 0.75,
        max_recovery_attempts: int = 3
    ):
        """Initialize OOM recovery manager.

        Args:
            context: Training context containing model, optimizer, data managers
            min_batch_size: Minimum batch size to try (default: 1)
            reduction_factor: Multiply batch size by this on OOM (default: 0.75 = 25% reduction)
            max_recovery_attempts: Maximum recovery attempts before failure (default: 3)
        """
        self.context = context
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_count = 0

    def recover_from_oom(self, current_batch_size: int) -> Optional[int]:
        """
        Perform complete OOM recovery with state reset.

        This method:
        1. Calculates new reduced batch size
        2. Clears CUDA cache and synchronizes
        3. Resets optimizer state (momentum/Adam buffers)
        4. Flushes prefetcher queue if present
        5. Recreates DataLoader with new batch size
        6. Clears model activation cache if present

        Args:
            current_batch_size: Current batch size that caused OOM

        Returns:
            New reduced batch size, or None if recovery is impossible

        Raises:
            RuntimeError: If max recovery attempts exceeded
        """
        self.recovery_count += 1

        if self.recovery_count > self.max_recovery_attempts:
            logger.error(
                f"Max OOM recovery attempts ({self.max_recovery_attempts}) exceeded! "
                "Training cannot continue."
            )
            return None

        # Step 1: Calculate new batch size
        new_batch_size = max(
            self.min_batch_size,
            int(current_batch_size * self.reduction_factor)
        )

        if new_batch_size == current_batch_size:
            logger.error(
                f"Cannot reduce batch size further! "
                f"Current: {current_batch_size}, Min: {self.min_batch_size}"
            )
            return None

        logger.warning(
            f"OOM detected (attempt {self.recovery_count}/{self.max_recovery_attempts}), "
            f"reducing batch size: {current_batch_size} -> {new_batch_size}"
        )

        # Step 2: Clear CUDA cache and synchronize
        logger.info("Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Step 3: Reset optimizer state (momentum buffers can be large)
        if hasattr(self.context, 'optimizer') and self.context.optimizer is not None:
            logger.info("Resetting optimizer state...")
            self._reset_optimizer_state(self.context.optimizer)

        # Step 4: Clear prefetcher queue (critical!)
        if hasattr(self.context, 'data_manager') and self.context.data_manager is not None:
            logger.info("Flushing prefetcher queue...")
            self._flush_prefetcher_queue(self.context.data_manager)

        # Step 5: Recreate dataloader with new batch size
        # Note: This may not be possible in all contexts, depends on data_manager API
        if hasattr(self.context, 'data_manager') and self.context.data_manager is not None:
            if hasattr(self.context.data_manager, 'recreate_train_dataloader'):
                logger.info(f"Recreating dataloader with batch size {new_batch_size}...")
                try:
                    self.context.data_manager.recreate_train_dataloader(
                        batch_size=new_batch_size
                    )
                except Exception as e:
                    logger.warning(f"Failed to recreate dataloader: {e}")
            else:
                logger.warning(
                    "DataLoader recreation not supported, "
                    "continuing with truncated batches"
                )

        # Step 6: Clear model activation cache if present
        if hasattr(self.context, 'model') and self.context.model is not None:
            if hasattr(self.context.model, 'clear_activation_cache'):
                logger.info("Clearing model activation cache...")
                self.context.model.clear_activation_cache()

        # Step 7: Final CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()

        logger.info(
            f"OOM recovery complete, new batch size: {new_batch_size} "
            f"(reduced by {int((1 - self.reduction_factor) * 100)}%)"
        )

        return new_batch_size

    def _reset_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Reset optimizer state to free momentum/Adam buffers.

        VRAM OPTIMIZATION: Uses optimizer.state.clear() instead of O(n) loop
        through all parameters. This is faster for large models (100K+ params).

        Args:
            optimizer: PyTorch optimizer
        """
        try:
            # Zero gradients first (set_to_none=True frees memory immediately)
            optimizer.zero_grad(set_to_none=True)

            # VRAM OPTIMIZATION: Single operation to clear all state instead of O(n) loop
            # This clears momentum buffers, Adam variance state, etc.
            state_count = len(optimizer.state)
            optimizer.state.clear()

            logger.debug(f"Cleared optimizer state ({state_count} parameter entries)")

        except Exception as e:
            logger.warning(f"Error resetting optimizer state: {e}")

    def _flush_prefetcher_queue(self, data_manager) -> None:
        """Flush prefetcher queue to remove old-size batches.

        Args:
            data_manager: DataLoaderManager instance
        """
        try:
            if hasattr(data_manager, 'train_loader'):
                train_loader = data_manager.train_loader

                if train_loader is not None and hasattr(train_loader, '_prefetcher'):
                    prefetcher = train_loader._prefetcher

                    # Flush the queue
                    flushed_count = 0
                    if hasattr(prefetcher, 'queue'):
                        while not prefetcher.queue.empty():
                            try:
                                prefetcher.queue.get_nowait()
                                flushed_count += 1
                            except queue.Empty:
                                break

                        logger.debug(f"Flushed {flushed_count} batches from prefetcher queue")

                        # Reset prefetcher state if possible
                        if hasattr(prefetcher, 'reset'):
                            prefetcher.reset()

        except Exception as e:
            logger.warning(f"Error flushing prefetcher queue: {e}")

    def reset_recovery_count(self) -> None:
        """Reset recovery attempt counter.

        Call this after successful training to allow future recovery attempts.
        For example, call this at the end of each epoch or after N successful steps.
        """
        if self.recovery_count > 0:
            logger.info(
                f"Resetting OOM recovery counter (was {self.recovery_count}), "
                "training has stabilized"
            )
            self.recovery_count = 0

    def get_recovery_stats(self) -> dict:
        """Get OOM recovery statistics.

        Returns:
            Dictionary with recovery statistics
        """
        return {
            'recovery_attempts': self.recovery_count,
            'max_attempts': self.max_recovery_attempts,
            'min_batch_size': self.min_batch_size,
            'reduction_factor': self.reduction_factor,
        }


def get_memory_fragmentation() -> float:
    """Calculate GPU memory fragmentation ratio.

    Fragmentation occurs when allocated memory is much smaller than reserved memory,
    indicating the CUDA allocator is holding onto freed blocks that can't be coalesced.

    Returns:
        Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
        Returns 0.0 if CUDA is not available or no memory is reserved.
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        stats = torch.cuda.memory_stats()
        allocated = stats.get('allocated_bytes.all.current', 0)
        reserved = stats.get('reserved_bytes.all.current', 0)

        if reserved == 0:
            return 0.0

        # Fragmentation = 1 - (allocated / reserved)
        # High fragmentation means lots of reserved but unused memory
        return 1.0 - (allocated / reserved)
    except Exception:
        return 0.0


def get_memory_stats() -> dict:
    """Get detailed GPU memory statistics.

    Returns:
        Dictionary with memory statistics:
        - allocated_gb: Currently allocated GPU memory in GB
        - reserved_gb: Total reserved GPU memory in GB
        - fragmentation: Fragmentation ratio (0.0-1.0)
        - free_gb: Free GPU memory in GB (total - reserved)
        - total_gb: Total GPU memory in GB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'fragmentation': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
        }

    try:
        stats = torch.cuda.memory_stats()
        allocated = stats.get('allocated_bytes.all.current', 0)
        reserved = stats.get('reserved_bytes.all.current', 0)
        total = torch.cuda.get_device_properties(0).total_memory

        gb = 1024 ** 3
        return {
            'allocated_gb': allocated / gb,
            'reserved_gb': reserved / gb,
            'fragmentation': get_memory_fragmentation(),
            'free_gb': (total - reserved) / gb,
            'total_gb': total / gb,
        }
    except Exception:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'fragmentation': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
        }


def proactive_memory_cleanup(
    fragmentation_threshold: float = 0.30,
    force: bool = False
) -> bool:
    """Proactively clean up GPU memory if fragmentation is high.

    This helps prevent OOM errors by periodically defragmenting the CUDA allocator.
    Call this during natural pauses in training (after validation, generation, etc.).

    Args:
        fragmentation_threshold: Trigger cleanup when fragmentation exceeds this (default: 0.30)
        force: If True, always perform cleanup regardless of fragmentation level

    Returns:
        True if cleanup was performed, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    fragmentation = get_memory_fragmentation()

    if force or fragmentation > fragmentation_threshold:
        # Get stats before cleanup for logging
        stats_before = get_memory_stats()

        # Perform cleanup
        torch.cuda.empty_cache()

        # Force Python garbage collection to release any PyTorch tensors
        import gc
        gc.collect()

        # Get stats after cleanup
        stats_after = get_memory_stats()

        freed_gb = stats_before['reserved_gb'] - stats_after['reserved_gb']

        if freed_gb > 0.01:  # Only log if we freed more than 10MB
            logger.debug(
                f"Proactive memory cleanup: freed {freed_gb:.2f} GB, "
                f"fragmentation {stats_before['fragmentation']:.1%} -> {stats_after['fragmentation']:.1%}"
            )

        return True

    return False
