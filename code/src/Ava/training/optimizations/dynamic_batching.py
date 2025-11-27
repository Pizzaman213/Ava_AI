"""
Dynamic Batching with Memory Awareness

Based on research from:
- arXiv:2412.21124: Dynamic batching for efficient GPU utilization
- arXiv:2503.05248: Memory-aware batch scheduling

This module implements adaptive batch sizing that monitors GPU memory in real-time
and adjusts the batch size to maximize throughput without causing OOM errors.

Key features:
- Real-time GPU memory monitoring
- Automatic batch size adjustment based on memory usage
- Configurable memory thresholds and adjustment rates
- Gradual scaling to avoid sudden memory spikes
- Safety mechanisms to prevent OOM

Expected improvement: 15-25% throughput increase
"""

import torch
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs


@dataclass
class DynamicBatchConfig:
    """Configuration for dynamic batching"""
    enabled: bool = True
    initial_batch_size: int = 128
    min_batch_size: int = 32
    max_batch_size: int = 512

    # Memory thresholds (as fraction of total GPU memory)
    low_memory_threshold: float = 0.50  # Below this, increase batch size
    target_memory_threshold: float = 0.70  # Target utilization
    high_memory_threshold: float = 0.85  # Above this, decrease batch size
    critical_memory_threshold: float = 0.95  # Emergency decrease

    # Adjustment parameters
    increase_factor: float = 1.2  # Multiply by this when increasing
    decrease_factor: float = 0.8  # Multiply by this when decreasing
    adjustment_frequency: int = 10  # Adjust every N steps
    warmup_steps: int = 100  # Don't adjust during warmup

    # Safety parameters
    max_adjustments_per_session: int = 50  # Prevent oscillation
    cooldown_steps: int = 5  # Steps to wait after adjustment


class DynamicBatchScheduler:
    """
    Manages dynamic batch size adjustment based on GPU memory usage.

    This scheduler monitors GPU memory and automatically adjusts the batch size
    to maximize GPU utilization while avoiding OOM errors.

    Example usage:
        scheduler = DynamicBatchScheduler(config)

        for step in range(num_steps):
            batch_size = scheduler.get_current_batch_size(step)
            batch = get_batch(batch_size)

            # Train on batch
            loss = model(batch)
            loss.backward()

            # Update scheduler with current memory usage
            scheduler.step(step)
    """

    def __init__(self, config: DynamicBatchConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.original_batch_size = config.initial_batch_size

        # Tracking
        self.step_count = 0
        self.adjustment_count = 0
        self.last_adjustment_step = -config.cooldown_steps
        self.memory_history = []
        self.batch_size_history = []

        # Statistics
        self.total_increases = 0
        self.total_decreases = 0
        self.max_batch_size_reached = config.initial_batch_size
        self.min_batch_size_reached = config.initial_batch_size

        # Get GPU info
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            logger.info(f"Dynamic batching initialized on GPU {self.device}")
            logger.info(f"Total GPU memory: {self.total_memory / 1e9:.2f} GB")
            logger.info(f"Initial batch size: {self.current_batch_size}")
            logger.info(f"Batch size range: [{config.min_batch_size}, {config.max_batch_size}]")
        else:
            self.device = None
            self.total_memory = None
            logger.warning("CUDA not available, dynamic batching disabled")
            self.config.enabled = False

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics

        Uses memory_reserved() instead of memory_allocated() because:
        - memory_allocated() only shows currently active tensors
        - memory_reserved() shows all memory held by PyTorch's caching allocator
        - This better reflects actual GPU memory pressure and prevents OOM
        """
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'peak_gb': 0.0,
                'utilization': 0.0
            }

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)  # Use reserved peak

        # Use RESERVED memory for utilization calculation - this is what actually
        # matters for OOM prevention. allocated() can be misleadingly low.
        utilization = reserved / self.total_memory if self.total_memory else 0.0

        return {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'peak_gb': peak / 1e9,
            'utilization': utilization  # Now based on reserved, not allocated
        }

    def should_adjust(self, step: int) -> bool:
        """Check if we should adjust batch size at this step"""
        if not self.config.enabled:
            return False

        # Don't adjust during warmup
        if step < self.config.warmup_steps:
            return False

        # Don't adjust too frequently
        if step - self.last_adjustment_step < self.config.cooldown_steps:
            return False

        # Don't adjust on every step
        if step % self.config.adjustment_frequency != 0:
            return False

        # Stop if we've adjusted too many times (prevent oscillation)
        if self.adjustment_count >= self.config.max_adjustments_per_session:
            return False

        return True

    def calculate_new_batch_size(self, memory_utilization: float) -> Tuple[int, str]:
        """
        Calculate new batch size based on memory utilization

        increase_factor: max multiplier to increase to (e.g., 3 = up to 3x min_batch_size)
        decrease_factor: ignored (always decrease by 1 step)

        Batch sizes are always multiples of min_batch_size (32, 64, 96, etc.)

        Returns:
            Tuple of (new_batch_size, reason)
        """
        min_bs = self.config.min_batch_size
        current_multiplier = self.current_batch_size // min_bs  # Integer multiplier
        max_multiplier = self.config.max_batch_size // min_bs

        # Critical memory - decrease to minimum immediately
        if memory_utilization > self.config.critical_memory_threshold:
            new_multiplier = 1  # Go to minimum
            reason = f"CRITICAL memory {memory_utilization:.1%}"

        # High memory - decrease by 1 multiplier step
        elif memory_utilization > self.config.high_memory_threshold:
            new_multiplier = max(1, current_multiplier - 1)
            reason = f"HIGH memory {memory_utilization:.1%}"

        # Low memory - increase by 1 multiplier step (up to max_multiplier)
        elif memory_utilization < self.config.low_memory_threshold:
            new_multiplier = min(current_multiplier + 1, max_multiplier)
            reason = f"LOW memory {memory_utilization:.1%}"

        # Target range - no change
        else:
            new_multiplier = current_multiplier
            reason = f"OPTIMAL memory {memory_utilization:.1%}"

        # Convert multiplier to batch size (always multiple of min_bs)
        new_size = new_multiplier * min_bs

        # Clamp to valid range
        new_size = max(min_bs, min(self.config.max_batch_size, new_size))

        return new_size, reason

    def step(self, step: int) -> Optional[int]:
        """
        Update scheduler state and potentially adjust batch size

        Args:
            step: Current training step

        Returns:
            New batch size if adjusted, None otherwise
        """
        self.step_count = step

        # Get memory stats
        mem_stats = self.get_memory_stats()
        utilization = mem_stats['utilization']

        # Record history
        self.memory_history.append(utilization)
        self.batch_size_history.append(self.current_batch_size)

        # Keep history bounded
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
            self.batch_size_history = self.batch_size_history[-1000:]

        # Check if we should adjust
        if not self.should_adjust(step):
            return None

        # Calculate new batch size
        new_batch_size, reason = self.calculate_new_batch_size(utilization)

        # Check if adjustment needed
        if new_batch_size == self.current_batch_size:
            logger.debug(f"Step {step}: {reason}, keeping batch size {new_batch_size}")
            return None

        # Apply adjustment
        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size
        self.last_adjustment_step = step
        self.adjustment_count += 1

        # Update statistics
        if new_batch_size > old_batch_size:
            self.total_increases += 1
            direction = "INCREASED"
        else:
            self.total_decreases += 1
            direction = "DECREASED"

        self.max_batch_size_reached = max(self.max_batch_size_reached, new_batch_size)
        self.min_batch_size_reached = min(self.min_batch_size_reached, new_batch_size)

        # Log adjustment
        logger.info(
            f"Step {step}: {direction} batch size: {old_batch_size} -> {new_batch_size} "
            f"({reason}, mem={mem_stats['reserved_gb']:.2f}GB reserved)"
        )

        return new_batch_size

    def get_current_batch_size(self, step: Optional[int] = None) -> int:
        """Get the current batch size (optionally update first)"""
        if step is not None:
            self.step(step)
        return self.current_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch size adjustments"""
        if not self.memory_history:
            return {}

        return {
            'current_batch_size': self.current_batch_size,
            'original_batch_size': self.original_batch_size,
            'min_batch_size_reached': self.min_batch_size_reached,
            'max_batch_size_reached': self.max_batch_size_reached,
            'total_adjustments': self.adjustment_count,
            'total_increases': self.total_increases,
            'total_decreases': self.total_decreases,
            'avg_memory_utilization': sum(self.memory_history) / len(self.memory_history),
            'current_memory_utilization': self.memory_history[-1] if self.memory_history else 0.0,
            'batch_size_improvement': (self.current_batch_size / self.original_batch_size - 1.0) * 100,
        }

    def log_summary(self):
        """Log a summary of dynamic batching performance"""
        stats = self.get_statistics()

        if not stats:
            logger.info("Dynamic batching: No statistics available")
            return

        logger.info("=" * 70)
        logger.info("DYNAMIC BATCHING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Original batch size: {stats['original_batch_size']}")
        logger.info(f"Final batch size: {stats['current_batch_size']}")
        logger.info(f"Batch size range: [{stats['min_batch_size_reached']}, {stats['max_batch_size_reached']}]")
        logger.info(f"Total adjustments: {stats['total_adjustments']} ({stats['total_increases']} increases, {stats['total_decreases']} decreases)")
        logger.info(f"Avg memory utilization: {stats['avg_memory_utilization']:.1%}")
        logger.info(f"Batch size improvement: {stats['batch_size_improvement']:+.1f}%")
        logger.info("=" * 70)

    def reset(self):
        """Reset scheduler to initial state"""
        self.current_batch_size = self.config.initial_batch_size
        self.step_count = 0
        self.adjustment_count = 0
        self.last_adjustment_step = -self.config.cooldown_steps
        self.memory_history = []
        self.batch_size_history = []
        self.total_increases = 0
        self.total_decreases = 0
        self.max_batch_size_reached = self.config.initial_batch_size
        self.min_batch_size_reached = self.config.initial_batch_size

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)


def create_dynamic_batch_scheduler(config_dict: Dict[str, Any]) -> DynamicBatchScheduler:
    """
    Create a dynamic batch scheduler from a configuration dictionary

    Args:
        config_dict: Configuration dictionary with dynamic_batching section

    Returns:
        DynamicBatchScheduler instance
    """
    # Extract dynamic batching config
    db_config = config_dict.get('dynamic_batching', {})

    # Get initial batch size - use min_batch_size from dynamic_batching config
    # (training.batch_size may be null when dynamic batching is enabled)
    initial_batch_size = db_config.get('min_batch_size', 32)

    # Create config
    config = DynamicBatchConfig(
        enabled=db_config.get('enabled', True),
        initial_batch_size=initial_batch_size,
        min_batch_size=db_config.get('min_batch_size', max(32, initial_batch_size // 4)),
        max_batch_size=db_config.get('max_batch_size', initial_batch_size * 4),
        low_memory_threshold=db_config.get('low_memory_threshold', 0.50),
        target_memory_threshold=db_config.get('target_memory_threshold', 0.70),
        high_memory_threshold=db_config.get('high_memory_threshold', 0.85),
        critical_memory_threshold=db_config.get('critical_memory_threshold', 0.95),
        increase_factor=db_config.get('increase_factor', 1.2),
        decrease_factor=db_config.get('decrease_factor', 0.8),
        adjustment_frequency=db_config.get('adjustment_frequency', 10),
        warmup_steps=db_config.get('warmup_steps', 100),
        max_adjustments_per_session=db_config.get('max_adjustments_per_session', 50),
        cooldown_steps=db_config.get('cooldown_steps', 5),
    )

    return DynamicBatchScheduler(config)
