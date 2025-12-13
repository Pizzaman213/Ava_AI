"""
Dynamic Batching - Simple, Understandable Implementation

This module provides memory-aware batch size adjustment during training.
The design prioritizes clarity and maintainability over feature count.

Key Classes:
    - MemoryMonitor: Tracks GPU memory usage with smoothing
    - BatchSizeCalculator: Simple, direct batch size decisions
    - DynamicBatchScheduler: Main coordinator for dynamic batching

Usage:
    scheduler = create_dynamic_batch_scheduler(config_dict)
    for step in range(num_steps):
        new_batch_size = scheduler.step(step)
        if new_batch_size:
            # Batch size changed, update dataloader
            pass
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DynamicBatchConfig:
    """
    Configuration for dynamic batching.

    Simple, flat config with sensible defaults.
    """
    # Enable/disable
    enabled: bool = True

    # Batch size bounds
    min_batch_size: int = 16
    max_batch_size: int = 256
    step_size: int = 16  # How much to increase/decrease at a time

    # Memory thresholds (as fractions 0-1)
    target_memory: float = 0.75   # Ideal memory usage
    high_memory: float = 0.85     # Start decreasing batch size
    critical_memory: float = 0.92 # Emergency - drop to minimum
    low_memory: float = 0.60      # Can increase batch size

    # Smoothing
    smoothing_factor: float = 0.1  # EMA alpha (0.1 = smooth, 0.5 = responsive)

    # Timing
    adjustment_interval: int = 10  # Steps between adjustments
    warmup_steps: int = 100        # Steps before allowing increases
    cooldown_steps: int = 5        # Steps to wait after adjustment

    # Optional features (disabled by default)
    enable_warmup: bool = False        # Gradual batch size ramp-up
    warmup_start_batch: int = 16       # Starting batch size for warmup
    warmup_end_step: int = 500         # When warmup completes

    enable_token_budget: bool = False  # Target token count instead of samples
    target_tokens: int = 4096          # Target tokens per batch
    max_tokens: int = 8192             # Maximum tokens per batch

    def __post_init__(self):
        """Validate configuration values."""
        if self.min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be > 0, got {self.min_batch_size}")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"max_batch_size ({self.max_batch_size}) must be >= "
                f"min_batch_size ({self.min_batch_size})"
            )
        if self.step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor:
    """
    Simple GPU memory monitoring with exponential moving average smoothing.

    Tracks:
        - Raw memory usage (current snapshot)
        - Smoothed memory usage (EMA for stable decisions)
        - Peak memory (for safety bounds)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        smoothing_factor: float = 0.1,
    ):
        """
        Initialize memory monitor.

        Args:
            device: GPU device to monitor (default: cuda:0)
            smoothing_factor: EMA alpha - lower = smoother, higher = more responsive
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.smoothing_factor = smoothing_factor

        # State
        self._smoothed_usage: float = 0.0
        self._peak_usage: float = 0.0
        self._total_memory: int = 0
        self._history: List[float] = []
        self._max_history: int = 100

        # Initialize total memory
        if torch.cuda.is_available():
            self._total_memory = torch.cuda.get_device_properties(self.device).total_memory

    def get_usage(self) -> float:
        """
        Get current GPU memory usage as a fraction (0-1).

        Returns:
            Memory utilization (0.0 to 1.0)
        """
        if not torch.cuda.is_available() or self._total_memory == 0:
            return 0.0

        # Use reserved memory (matches nvidia-smi more closely)
        reserved = torch.cuda.memory_reserved(self.device)
        usage = reserved / self._total_memory

        # Update smoothed value
        if self._smoothed_usage == 0.0:
            self._smoothed_usage = usage
        else:
            self._smoothed_usage = (
                self.smoothing_factor * usage +
                (1 - self.smoothing_factor) * self._smoothed_usage
            )

        # Update peak
        self._peak_usage = max(self._peak_usage, usage)

        # Record history
        self._history.append(usage)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return usage

    def get_smoothed_usage(self) -> float:
        """Get smoothed memory usage (EMA)."""
        return self._smoothed_usage

    def get_peak_usage(self) -> float:
        """Get peak memory usage seen so far."""
        return self._peak_usage

    def get_total_memory_gb(self) -> float:
        """Get total GPU memory in GB."""
        return self._total_memory / (1024 ** 3)

    def get_stats(self) -> Dict[str, float]:
        """Get all memory statistics."""
        current = self.get_usage()
        return {
            "current": current,
            "smoothed": self._smoothed_usage,
            "peak": self._peak_usage,
            "total_gb": self.get_total_memory_gb(),
        }

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        self._peak_usage = 0.0


# =============================================================================
# Batch Size Calculator
# =============================================================================

class BatchSizeCalculator:
    """
    Simple, direct batch size calculation.

    Logic is straightforward:
        - memory > critical: emergency drop to minimum
        - memory > high: decrease by step_size
        - memory < low: increase by step_size
        - otherwise: no change

    No confusing multipliers, no complex state machines.
    """

    def __init__(
        self,
        min_batch_size: int = 16,
        max_batch_size: int = 256,
        step_size: int = 16,
        target_memory: float = 0.75,
        high_memory: float = 0.85,
        critical_memory: float = 0.92,
        low_memory: float = 0.60,
    ):
        """
        Initialize batch size calculator.

        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            step_size: How much to increase/decrease at a time
            target_memory: Ideal memory utilization
            high_memory: Threshold to start decreasing
            critical_memory: Emergency threshold - drop to minimum
            low_memory: Threshold to start increasing
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.step_size = step_size
        self.target_memory = target_memory
        self.high_memory = high_memory
        self.critical_memory = critical_memory
        self.low_memory = low_memory

        # Validate
        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be > 0, got {min_batch_size}")
        if max_batch_size < min_batch_size:
            raise ValueError(f"max_batch_size ({max_batch_size}) must be >= min_batch_size ({min_batch_size})")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

    def calculate(
        self,
        current_batch_size: int,
        memory_usage: float,
        allow_increase: bool = True,
        effective_max: Optional[int] = None,
    ) -> tuple[int, str]:
        """
        Calculate new batch size based on memory usage.

        Args:
            current_batch_size: Current batch size
            memory_usage: Current memory utilization (0-1)
            allow_increase: Whether increases are allowed (False during warmup)
            effective_max: Override max_batch_size (from constraint provider)

        Returns:
            Tuple of (new_batch_size, reason_string)
        """
        # Use effective_max if provided, otherwise use configured max
        max_bs = effective_max if effective_max is not None else self.max_batch_size

        # Critical memory - emergency drop to minimum
        if memory_usage > self.critical_memory:
            return self.min_batch_size, f"CRITICAL: {memory_usage:.0%} > {self.critical_memory:.0%}"

        # High memory - decrease
        if memory_usage > self.high_memory:
            new_size = max(self.min_batch_size, current_batch_size - self.step_size)
            return new_size, f"HIGH: {memory_usage:.0%} > {self.high_memory:.0%}"

        # Low memory - increase (if allowed and below effective max)
        if memory_usage < self.low_memory and allow_increase:
            new_size = min(max_bs, current_batch_size + self.step_size)
            return new_size, f"LOW: {memory_usage:.0%} < {self.low_memory:.0%}"

        # Optimal range - no change
        return current_batch_size, f"OPTIMAL: {memory_usage:.0%}"


# =============================================================================
# Dynamic Batch Scheduler
# =============================================================================

class DynamicBatchScheduler:
    """
    Main scheduler for dynamic batch sizing.

    Coordinates memory monitoring and batch size decisions.
    Simple interface: call step() each training step, get batch size changes.
    """

    def __init__(self, config: DynamicBatchConfig):
        """
        Initialize dynamic batch scheduler.

        Args:
            config: Dynamic batching configuration
        """
        self.config = config

        # Components
        self.memory_monitor = MemoryMonitor(smoothing_factor=config.smoothing_factor)
        self.calculator = BatchSizeCalculator(
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size,
            step_size=config.step_size,
            target_memory=config.target_memory,
            high_memory=config.high_memory,
            critical_memory=config.critical_memory,
            low_memory=config.low_memory,
        )

        # State
        self.current_batch_size = config.min_batch_size
        self.step_count = 0
        self.last_adjustment_step = 0
        self.last_reason = "INIT"

        # Constraint provider (e.g., BatchSizeController)
        # When set, this limits max_batch_size to prevent OOM oscillation
        self._constraint_provider: Optional[Any] = None

        # Statistics
        self.adjustments_made = 0
        self.oom_count = 0
        self.batch_size_history: List[int] = []

        # OOM recovery tracking - prevent rapid increases after OOM
        self._last_oom_step: int = -1000  # Long ago by default
        self._oom_cooldown_steps: int = 50  # Wait 50 steps after OOM before increasing

        logger.info(
            f"DynamicBatchScheduler initialized: "
            f"batch_size={config.min_batch_size}-{config.max_batch_size}, "
            f"step_size={config.step_size}, "
            f"thresholds=[low={config.low_memory:.0%}, target={config.target_memory:.0%}, "
            f"high={config.high_memory:.0%}, critical={config.critical_memory:.0%}]"
        )

    def set_constraint_provider(self, provider: Any) -> None:
        """
        Set external constraint provider (e.g., BatchSizeController).

        The provider limits max_batch_size to prevent the scheduler from
        increasing batch size beyond known-safe values after OOM events.

        The provider should have:
        - max_batch_size property: Maximum allowed batch size
        - is_size_safe(size) method (optional): Check if a size is safe

        Args:
            provider: Constraint provider (typically BatchSizeController)
        """
        self._constraint_provider = provider
        if provider is not None:
            # Immediately apply constraint
            effective_max = self._get_effective_max()
            if effective_max < self.config.max_batch_size:
                logger.info(
                    f"Constraint provider set: max_batch_size {self.config.max_batch_size} "
                    f"-> {effective_max} (constrained)"
                )
            # Also clamp current batch size if it exceeds the new max
            if self.current_batch_size > effective_max:
                old_bs = self.current_batch_size
                self.current_batch_size = effective_max
                logger.info(f"Current batch size clamped: {old_bs} -> {effective_max}")

    def _get_effective_max(self) -> int:
        """
        Get the effective maximum batch size considering constraints.

        If a constraint provider is set, returns the minimum of:
        - The configured max_batch_size
        - The provider's max_batch_size

        Returns:
            Effective maximum batch size
        """
        if self._constraint_provider is not None:
            provider_max = getattr(self._constraint_provider, 'max_batch_size', None)
            if provider_max is not None:
                return min(self.config.max_batch_size, provider_max)
        return self.config.max_batch_size

    def step(self, step_num: int, force_check: bool = False) -> Optional[int]:
        """
        Update scheduler and potentially adjust batch size.

        Args:
            step_num: Current training step number
            force_check: Force memory check even if not at interval

        Returns:
            New batch size if changed, None otherwise
        """
        self.step_count = step_num

        # Record history
        self.batch_size_history.append(self.current_batch_size)
        if len(self.batch_size_history) > 1000:
            self.batch_size_history = self.batch_size_history[-1000:]

        # Check if we should evaluate
        if not force_check:
            # Respect cooldown
            if step_num - self.last_adjustment_step < self.config.cooldown_steps:
                return None

            # Only check at intervals
            if step_num % self.config.adjustment_interval != 0:
                return None

        # Handle warmup mode
        if self.config.enable_warmup and step_num < self.config.warmup_end_step:
            return self._handle_warmup(step_num)

        # Get memory usage
        memory_usage = self.memory_monitor.get_smoothed_usage()

        # Check if increases are allowed:
        # 1. Must be past warmup period
        # 2. Must be past OOM cooldown period (prevent rapid re-increase after OOM)
        past_warmup = step_num >= self.config.warmup_steps
        past_oom_cooldown = (step_num - self._last_oom_step) >= self._oom_cooldown_steps
        allow_increase = past_warmup and past_oom_cooldown

        # Get effective max (respects constraint provider like BatchSizeController)
        effective_max = self._get_effective_max()

        # Calculate new batch size with constraint
        new_batch_size, reason = self.calculator.calculate(
            current_batch_size=self.current_batch_size,
            memory_usage=memory_usage,
            allow_increase=allow_increase,
            effective_max=effective_max,
        )

        # Check if changed
        if new_batch_size != self.current_batch_size:
            old_size = self.current_batch_size
            self.current_batch_size = new_batch_size
            self.last_adjustment_step = step_num
            self.last_reason = reason
            self.adjustments_made += 1

            logger.info(
                f"[Step {step_num}] Batch size: {old_size} -> {new_batch_size} ({reason})"
            )
            return new_batch_size

        return None

    def _handle_warmup(self, step_num: int) -> Optional[int]:
        """Handle warmup period with gradual batch size increase."""
        # Respect OOM cooldown - don't increase during recovery
        if (step_num - self._last_oom_step) < self._oom_cooldown_steps:
            return None  # Stay at current batch size during OOM recovery

        start = self.config.warmup_start_batch
        effective_max = self._get_effective_max()  # Respect constraint provider
        total_steps = self.config.warmup_end_step

        # Linear interpolation (respecting effective max, not config max)
        progress = step_num / total_steps
        target = int(start + (effective_max - start) * progress)

        # Round to step_size and clamp to effective range
        target = (target // self.config.step_size) * self.config.step_size
        target = max(self.config.min_batch_size, min(effective_max, target))

        # Only allow increases, never decrease during warmup (unless OOM happened)
        if target < self.current_batch_size:
            target = self.current_batch_size

        if target != self.current_batch_size:
            old_size = self.current_batch_size
            self.current_batch_size = target
            self.last_adjustment_step = step_num
            self.last_reason = f"WARMUP: {progress:.0%}"
            self.adjustments_made += 1

            logger.info(
                f"[Step {step_num}] Warmup batch size: {old_size} -> {target}"
            )
            return target

        return None

    def record_oom(self) -> int:
        """
        Record an OOM event and reduce batch size.

        Call this when a CUDA OOM error occurs.

        Returns:
            New (reduced) batch size
        """
        self.oom_count += 1
        old_size = self.current_batch_size

        # Drop by 2 steps on OOM for safety
        self.current_batch_size = max(
            self.config.min_batch_size,
            self.current_batch_size - 2 * self.config.step_size
        )

        self.last_adjustment_step = self.step_count
        self._last_oom_step = self.step_count  # Track OOM for cooldown
        self.last_reason = f"OOM #{self.oom_count}"
        self.adjustments_made += 1

        logger.warning(
            f"[OOM #{self.oom_count}] Batch size: {old_size} -> {self.current_batch_size} "
            f"(cooldown until step {self.step_count + self._oom_cooldown_steps})"
        )

        return self.current_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size, constrained by effective max."""
        effective_max = self._get_effective_max()
        return min(self.current_batch_size, effective_max)

    def set_batch_size(self, batch_size: int) -> None:
        """Manually set batch size (clamped to valid range and constraint)."""
        effective_max = self._get_effective_max()
        self.current_batch_size = max(
            self.config.min_batch_size,
            min(effective_max, batch_size)
        )

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return self.memory_monitor.get_stats()

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "step_count": self.step_count,
            "adjustments_made": self.adjustments_made,
            "oom_count": self.oom_count,
            "last_reason": self.last_reason,
            "memory": self.memory_monitor.get_stats(),
            "config": {
                "min_batch_size": self.config.min_batch_size,
                "max_batch_size": self.config.max_batch_size,
                "step_size": self.config.step_size,
            }
        }

    def log_summary(self) -> None:
        """Log a summary of scheduler performance."""
        stats = self.get_statistics()
        logger.info("=" * 60)
        logger.info("DYNAMIC BATCH SCHEDULER SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Final batch size: {stats['current_batch_size']}")
        logger.info(f"Total steps: {stats['step_count']}")
        logger.info(f"Adjustments made: {stats['adjustments_made']}")
        logger.info(f"OOM events: {stats['oom_count']}")
        logger.info(f"Peak memory: {stats['memory']['peak']:.1%}")
        logger.info("=" * 60)


# =============================================================================
# Factory Function
# =============================================================================

def create_dynamic_batch_scheduler(config_dict: Dict[str, Any]) -> DynamicBatchScheduler:
    """
    Create a DynamicBatchScheduler from a config dictionary.

    Looks for 'dynamic_batching' section in various locations:
        - config_dict['dynamic_batching']
        - config_dict['training']['dynamic_batching']
        - config_dict['training']['batching']['dynamic_batching']

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configured DynamicBatchScheduler
    """
    # Find dynamic batching config
    db_config = config_dict.get('dynamic_batching', {})

    if not db_config:
        training = config_dict.get('training', {})
        db_config = training.get('dynamic_batching', {})

        if not db_config:
            batching = training.get('batching', {})
            db_config = batching.get('dynamic_batching', {})

    # Build config object
    config = DynamicBatchConfig(
        enabled=db_config.get('enabled', True),
        min_batch_size=db_config.get('min_batch_size', 16),
        max_batch_size=db_config.get('max_batch_size', 256),
        step_size=db_config.get('step_size', 16),
        target_memory=db_config.get('target_memory', 0.75),
        high_memory=db_config.get('high_memory', 0.85),
        critical_memory=db_config.get('critical_memory', 0.92),
        low_memory=db_config.get('low_memory', 0.60),
        smoothing_factor=db_config.get('smoothing_factor', 0.1),
        adjustment_interval=db_config.get('adjustment_interval', 10),
        warmup_steps=db_config.get('warmup_steps', 100),
        cooldown_steps=db_config.get('cooldown_steps', 5),
        enable_warmup=db_config.get('enable_warmup', False),
        warmup_start_batch=db_config.get('warmup_start_batch', 16),
        warmup_end_step=db_config.get('warmup_end_step', 500),
        enable_token_budget=db_config.get('enable_token_budget', False),
        target_tokens=db_config.get('target_tokens', 4096),
        max_tokens=db_config.get('max_tokens', 8192),
    )

    return DynamicBatchScheduler(config)


# =============================================================================
# Convenience Aliases (for backward compatibility)
# =============================================================================

# These provide backward compatibility with code that imports the old names
DynamicBatchingConfig = DynamicBatchConfig
