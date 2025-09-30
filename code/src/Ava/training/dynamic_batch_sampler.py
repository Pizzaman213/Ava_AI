"""
Dynamic Batch Size Adjustment for Hardware-Adaptive Training

Automatically adjusts batch size based on GPU memory utilization to:
- Maximize throughput by using optimal batch size for available hardware
- Prevent OOM by reducing batch size when memory is critical
- Increase batch size when GPU is underutilized
- Provide seamless hardware portability (same config works on 8GB to 80GB GPUs)
"""

import torch
import logging
from typing import Optional, Dict, List, Tuple
from collections import deque
import time

logger = logging.getLogger(__name__)


class DynamicBatchSizer:
    """
    Dynamically adjust batch size based on GPU memory utilization.

    Monitors memory usage and automatically scales batch size to maximize
    hardware utilization while preventing OOM errors.
    """

    def __init__(
        self,
        initial_batch_size: int = 8,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        target_memory_utilization: float = 0.85,
        adjustment_frequency: int = 100,
        adjustment_factor: float = 1.25,
        warmup_steps: int = 500,
        smooth_transitions: bool = True,
        memory_monitor=None
    ):
        """
        Initialize dynamic batch sizer.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory_utilization: Target GPU memory usage (0.85 = 85%)
            adjustment_frequency: Check and adjust every N steps
            adjustment_factor: Multiply/divide batch size by this factor
            warmup_steps: Don't adjust during first N steps
            smooth_transitions: Use gradual adjustments vs aggressive changes
            memory_monitor: MemoryMonitor instance for memory stats
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = max(1, min_batch_size)
        self.max_batch_size = max_batch_size
        self.target_utilization = target_memory_utilization
        self.adjustment_frequency = adjustment_frequency
        self.adjustment_factor = adjustment_factor
        self.warmup_steps = warmup_steps
        self.smooth_transitions = smooth_transitions
        self.memory_monitor = memory_monitor

        # Tracking
        self.step_count = 0
        self.last_adjustment_step = 0
        self.adjustment_history = deque(maxlen=100)
        self.batch_size_history = deque(maxlen=1000)
        self.total_adjustments = 0
        self.increases = 0
        self.decreases = 0

        # Memory thresholds for different actions
        self.critical_threshold = 0.98  # Emergency reduction (higher - be less conservative)
        self.warning_threshold = 0.95  # Significant reduction (higher)
        self.high_threshold = 0.92     # Modest reduction (higher)
        self.low_threshold = 0.85      # Consider increase (much higher - increase more aggressively)

        logger.info(
            f"Dynamic batch sizer initialized: "
            f"initial={initial_batch_size}, min={min_batch_size}, max={max_batch_size}, "
            f"target_util={target_memory_utilization:.1%}"
        )

    def should_adjust(self, step: int) -> bool:
        """
        Check if it's time to consider batch size adjustment.

        Args:
            step: Current training step

        Returns:
            True if should check for adjustment
        """
        # Reduced warmup to 100 steps - adjust batch size early
        if step < 100:
            return False

        # Check every 50 steps instead of adjustment_frequency for faster increases
        if step - self.last_adjustment_step < 50:
            return False

        return True

    def calculate_new_batch_size(
        self,
        current_batch_size: int,
        memory_stats: Dict[str, float]
    ) -> Tuple[int, str]:
        """
        Calculate new batch size based on memory utilization.

        Args:
            current_batch_size: Current batch size
            memory_stats: Memory statistics from MemoryMonitor

        Returns:
            Tuple of (new_batch_size, reason)
        """
        gpu_util = memory_stats.get('gpu_utilization', 0.0)
        available_gb = memory_stats.get('gpu_available_gb', 0.0)

        # Critical: Emergency reduction
        if gpu_util >= self.critical_threshold:
            new_size = max(self.min_batch_size, current_batch_size // 4)
            reason = f"EMERGENCY: GPU {gpu_util:.1%} >= {self.critical_threshold:.1%}"
            return new_size, reason

        # Warning: Significant reduction
        if gpu_util >= self.warning_threshold:
            factor = 0.5 if not self.smooth_transitions else 0.8
            new_size = max(self.min_batch_size, int(current_batch_size * factor))
            reason = f"WARNING: GPU {gpu_util:.1%} >= {self.warning_threshold:.1%}"
            return new_size, reason

        # High: Modest reduction
        if gpu_util >= self.high_threshold:
            factor = 0.7 if not self.smooth_transitions else 0.9
            new_size = max(self.min_batch_size, int(current_batch_size * factor))
            reason = f"HIGH: GPU {gpu_util:.1%} >= {self.high_threshold:.1%}"
            return new_size, reason

        # Low: Consider increase - AGGRESSIVE mode to fill GPU memory
        if gpu_util < self.low_threshold and available_gb > 0.5:
            # Simplified check - increase if we have free memory (no stability check needed)
            # Calculate how much we can increase based on available memory
            factor = self.adjustment_factor if not self.smooth_transitions else 1.2
            new_size = min(self.max_batch_size, int(current_batch_size * factor))

            # Don't increase if already at max
            if new_size == current_batch_size:
                return current_batch_size, "at_maximum"

            reason = f"INCREASE: GPU {gpu_util:.1%} < {self.low_threshold:.1%}, {available_gb:.1f}GB free"
            return new_size, reason

        # Optimal range: No change needed
        return current_batch_size, f"optimal: GPU {gpu_util:.1%}"

    def adjust_batch_size(
        self,
        step: int,
        memory_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[int, bool, str]:
        """
        Adjust batch size if needed based on memory utilization.

        Args:
            step: Current training step
            memory_stats: Memory statistics (if None, will query memory_monitor)

        Returns:
            Tuple of (new_batch_size, changed, reason)
        """
        self.step_count = step

        # DEBUG: Log that we're being called
        if step % 500 == 0:
            logger.info(f"🔍 DynamicBatchSizer called at step {step}: current_batch_size={self.current_batch_size}")

        # Check if we should adjust
        if not self.should_adjust(step):
            self.batch_size_history.append(self.current_batch_size)
            return self.current_batch_size, False, "not_time_yet"

        # Get memory stats
        if memory_stats is None:
            if self.memory_monitor is None:
                # No memory monitor, can't adjust
                self.batch_size_history.append(self.current_batch_size)
                return self.current_batch_size, False, "no_memory_monitor"

            memory_stats = self.memory_monitor.get_memory_stats()

        # Calculate new batch size
        new_batch_size, reason = self.calculate_new_batch_size(
            self.current_batch_size,
            memory_stats
        )

        # DEBUG: Log decision every 200 steps
        if step % 200 == 0:
            gpu_util = memory_stats.get('gpu_utilization', 0.0)
            available_gb = memory_stats.get('gpu_available_gb', 0.0)
            logger.info(
                f"🔍 Step {step}: GPU={gpu_util:.1%}, available={available_gb:.1f}GB, "
                f"current_bs={self.current_batch_size}, new_bs={new_batch_size}, reason={reason}"
            )

        # Check if changed
        changed = new_batch_size != self.current_batch_size

        if changed:
            old_batch_size = self.current_batch_size
            self.current_batch_size = new_batch_size
            self.last_adjustment_step = step
            self.total_adjustments += 1

            if new_batch_size > old_batch_size:
                self.increases += 1
                direction = "INCREASED"
            else:
                self.decreases += 1
                direction = "DECREASED"

            # Record adjustment
            self.adjustment_history.append({
                'step': step,
                'old_size': old_batch_size,
                'new_size': new_batch_size,
                'reason': reason,
                'memory_util': memory_stats.get('gpu_utilization', 0.0),
                'direction': direction
            })

            logger.info(
                f"Step {step}: Batch size {direction} {old_batch_size} → {new_batch_size} "
                f"({reason})"
            )

        self.batch_size_history.append(self.current_batch_size)

        return self.current_batch_size, changed, reason

    def get_statistics(self) -> Dict[str, any]:
        """
        Get batch size adjustment statistics.

        Returns:
            Dictionary with adjustment statistics
        """
        if len(self.batch_size_history) > 0:
            history_list = list(self.batch_size_history)
            min_batch = min(history_list)
            max_batch = max(history_list)
            avg_batch = sum(history_list) / len(history_list)
        else:
            min_batch = self.current_batch_size
            max_batch = self.current_batch_size
            avg_batch = self.current_batch_size

        return {
            'current_batch_size': self.current_batch_size,
            'min_batch_size': min_batch,
            'max_batch_size': max_batch,
            'avg_batch_size': avg_batch,
            'total_adjustments': self.total_adjustments,
            'increases': self.increases,
            'decreases': self.decreases,
            'adjustment_rate': self.total_adjustments / max(1, self.step_count),
            'recent_adjustments': list(self.adjustment_history)[-5:] if self.adjustment_history else []
        }

    def reset(self, new_batch_size: Optional[int] = None):
        """
        Reset batch sizer state.

        Args:
            new_batch_size: Optional new starting batch size
        """
        if new_batch_size is not None:
            self.current_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))

        logger.info(f"Dynamic batch sizer reset to batch_size={self.current_batch_size}")


class DynamicBatchSamplerWrapper:
    """
    Wrapper around PyTorch DataLoader to support dynamic batch sizing.

    Handles dataloader recreation when batch size changes.
    """

    def __init__(
        self,
        dataset,
        batch_sizer: DynamicBatchSizer,
        collate_fn=None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        """
        Initialize dynamic batch sampler wrapper.

        Args:
            dataset: PyTorch dataset
            batch_sizer: DynamicBatchSizer instance
            collate_fn: Collate function for batching
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_sizer = batch_sizer
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Create initial dataloader
        self.dataloader = self._create_dataloader(self.batch_sizer.current_batch_size)
        self.dataloader_iter = None

    def _create_dataloader(self, batch_size: int):
        """Create a new dataloader with specified batch size."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False  # Shuffling handled by dataset if needed
        )

    def recreate_dataloader_if_needed(self, new_batch_size: int):
        """
        Recreate dataloader if batch size changed.

        Args:
            new_batch_size: New batch size
        """
        if new_batch_size != self.dataloader.batch_size:
            logger.debug(f"Recreating dataloader with batch_size={new_batch_size}")
            self.dataloader = self._create_dataloader(new_batch_size)
            self.dataloader_iter = None  # Will be recreated on next iteration

    def __iter__(self):
        """Create iterator."""
        self.dataloader_iter = iter(self.dataloader)
        return self

    def __next__(self):
        """Get next batch."""
        return next(self.dataloader_iter)