"""
Batch Size Controller - Central Authority for Dynamic Batch Sizing

This module provides a unified controller for batch size decisions during training.
It replaces the fragmented approach where DynamicBatchScheduler and DynamicBatchIterator
competed for control, leading to oscillating batch sizes and OOM errors.

Key Features:
- Startup calibration via binary search to find optimal batch size
- Explicit tracking of safe/unsafe batch sizes
- Immediate OOM response with known-safe fallback
- Convergence to stable batch size (no oscillation)

Usage:
    controller = BatchSizeController(
        min_batch_size=16,
        max_batch_size=256,
        target_memory=0.75,
    )

    # Optional: Run calibration at startup
    optimal = controller.startup_calibration(model, sample_batch_fn)

    # During training, record outcomes
    controller.record_success(batch_size=64, memory_util=0.72)

    # On OOM, get safe fallback immediately
    safe_size = controller.record_failure(batch_size=128)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeState:
    """Immutable snapshot of current batch size state for logging/debugging."""
    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    safe_ceiling: int
    mode: str
    known_safe_count: int
    known_unsafe_count: int
    consecutive_successes: int


class MemoryMonitor:
    """
    Simple GPU memory monitor with caching.

    Provides memory utilization readings without the complexity of the old
    DynamicBatchScheduler. All batch size decisions are delegated to
    BatchSizeController.
    """

    def __init__(self, cache_interval_sec: float = 0.5):
        """
        Initialize memory monitor.

        Args:
            cache_interval_sec: How often to refresh memory stats (seconds)
        """
        self._cache: Dict[str, float] = {}
        self._cache_time: float = 0.0
        self._cache_interval: float = cache_interval_sec
        self._device: Optional[int] = None
        self._total_memory: int = 0

        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            self._total_memory = torch.cuda.get_device_properties(self._device).total_memory

    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        return time.monotonic() - self._cache_time > self._cache_interval

    def _refresh_cache(self) -> None:
        """Query GPU memory (causes sync)."""
        if not torch.cuda.is_available() or self._total_memory == 0:
            self._cache = {'utilization': 0.0, 'reserved_gb': 0.0, 'allocated_gb': 0.0}
            return

        reserved = torch.cuda.memory_reserved(self._device)
        allocated = torch.cuda.memory_allocated(self._device)

        self._cache = {
            'utilization': reserved / self._total_memory,
            'reserved_gb': reserved / 1e9,
            'allocated_gb': allocated / 1e9,
            'total_gb': self._total_memory / 1e9,
        }
        self._cache_time = time.monotonic()

    def get_utilization(self, force_refresh: bool = False) -> float:
        """
        Get current memory utilization (0.0 to 1.0).

        Args:
            force_refresh: If True, query GPU even if cache is fresh

        Returns:
            Memory utilization as fraction (0.0 to 1.0)
        """
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.get('utilization', 0.0)

    def get_stats(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get full memory statistics.

        Args:
            force_refresh: If True, query GPU even if cache is fresh

        Returns:
            Dict with utilization, reserved_gb, allocated_gb, total_gb
        """
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.copy()


class BatchSizeController:
    """
    Central authority for batch size decisions.

    Design Principles:
    1. Single source of truth - only this component changes batch size
    2. Explicit safe/unsafe tracking - remembers what worked and what didn't
    3. Convergence - finds stable batch size through calibration, then locks
    4. Immediate OOM response - returns known-safe size instantly on failure

    Modes:
    - 'calibrating': Finding optimal batch size at startup
    - 'stable': Locked onto optimal size, no changes unless OOM
    - 'recovering': Recently had OOM, using conservative size
    """

    def __init__(
        self,
        min_batch_size: int = 16,
        max_batch_size: int = 256,
        target_memory: float = 0.75,
        oom_recovery_factor: float = 0.5,
        stability_threshold: int = 10,
    ):
        """
        Initialize batch size controller.

        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory: Target GPU memory utilization (0.0 to 1.0)
            oom_recovery_factor: Factor to reduce batch size on OOM (0.5 = halve)
            stability_threshold: Consecutive successes before considering stable
        """
        self._min_batch_size = max(1, min_batch_size)
        self._max_batch_size = max_batch_size
        self._target_memory = target_memory
        self._oom_recovery_factor = oom_recovery_factor
        self._stability_threshold = stability_threshold

        # Current state
        self._current_batch_size = min_batch_size
        self._mode = 'calibrating'

        # Safety tracking
        self._known_safe: Dict[int, float] = {}  # batch_size -> max_memory_util seen
        self._known_unsafe: Set[int] = set()     # batch_sizes that caused OOM
        self._safe_ceiling = min_batch_size       # Highest confirmed safe size

        # Stability tracking
        self._consecutive_successes = 0
        self._stable_batch_size: Optional[int] = None

        # Memory monitor
        self._memory_monitor = MemoryMonitor()

        logger.info(
            f"BatchSizeController initialized: "
            f"range=[{min_batch_size}, {max_batch_size}], "
            f"target_memory={target_memory:.0%}"
        )

    def startup_calibration(
        self,
        model: nn.Module,
        sample_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
        target_memory: Optional[float] = None,
        max_time_seconds: float = 30.0,
        seq_len: int = 512,
    ) -> int:
        """
        Binary search for optimal batch size at training start.

        Algorithm:
        1. Start at min_batch_size (definitely works)
        2. Double until OOM or util > target
        3. Binary search between last_good and first_bad
        4. Lock onto stable size

        Args:
            model: The model to calibrate with
            sample_batch_fn: Function that creates a batch given batch_size
            target_memory: Override target memory utilization
            max_time_seconds: Maximum time to spend calibrating
            seq_len: Sequence length for calibration batches

        Returns:
            Optimal batch size found
        """
        if target_memory is None:
            target_memory = self._target_memory

        start_time = time.time()
        logger.info(f"Starting batch size calibration (target={target_memory:.0%}, timeout={max_time_seconds}s)")

        # Phase 1: Find upper bound by doubling
        test_size = self._min_batch_size
        last_good = test_size
        first_bad = self._max_batch_size + 1  # Sentinel for "not found"

        model.eval()  # Calibrate in eval mode (less memory than train)

        while test_size <= self._max_batch_size:
            if time.time() - start_time > max_time_seconds:
                logger.warning(f"Calibration timeout after {max_time_seconds}s")
                break

            try:
                torch.cuda.empty_cache()
                batch = sample_batch_fn(test_size)

                # Move to GPU if not already
                device = next(model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                with torch.no_grad():
                    # Forward pass only for calibration
                    _ = model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                    )

                util = self._memory_monitor.get_utilization(force_refresh=True)
                logger.debug(f"Calibration: BS={test_size}, util={util:.1%}")

                if util > target_memory:
                    # Over target, but not OOM
                    first_bad = test_size
                    logger.info(f"Found upper bound at BS={test_size} (util={util:.1%} > {target_memory:.0%})")
                    break
                else:
                    last_good = test_size
                    self._record_safe(test_size, util)
                    test_size *= 2

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    first_bad = test_size
                    self._known_unsafe.add(test_size)
                    logger.info(f"OOM at BS={test_size}, upper bound found")
                    break
                else:
                    raise

        # Phase 2: Binary search for optimal
        logger.info(f"Binary search between {last_good} and {first_bad}")

        while first_bad - last_good > self._min_batch_size:
            if time.time() - start_time > max_time_seconds:
                logger.warning(f"Calibration timeout during binary search")
                break

            # Ensure mid is a multiple of min_batch_size
            mid = ((last_good + first_bad) // 2 // self._min_batch_size) * self._min_batch_size

            if mid <= last_good or mid >= first_bad:
                break

            try:
                torch.cuda.empty_cache()
                batch = sample_batch_fn(mid)

                device = next(model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                with torch.no_grad():
                    _ = model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                    )

                util = self._memory_monitor.get_utilization(force_refresh=True)
                logger.debug(f"Binary search: BS={mid}, util={util:.1%}")

                if util <= target_memory:
                    last_good = mid
                    self._record_safe(mid, util)
                else:
                    first_bad = mid

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    first_bad = mid
                    self._known_unsafe.add(mid)
                else:
                    raise

        # Finalize
        self._current_batch_size = last_good
        self._stable_batch_size = last_good
        self._safe_ceiling = last_good
        self._mode = 'stable'

        torch.cuda.empty_cache()
        model.train()  # Return to training mode

        elapsed = time.time() - start_time
        logger.info(
            f"Calibration complete in {elapsed:.1f}s: "
            f"optimal batch size = {last_good}, "
            f"tested {len(self._known_safe)} safe sizes, "
            f"found {len(self._known_unsafe)} unsafe sizes"
        )

        return last_good

    def _record_safe(self, batch_size: int, memory_util: float) -> None:
        """Internal: Record a safe batch size."""
        self._known_safe[batch_size] = max(
            self._known_safe.get(batch_size, 0.0),
            memory_util
        )
        self._safe_ceiling = max(self._safe_ceiling, batch_size)

    def record_success(self, batch_size: int, memory_util: Optional[float] = None) -> None:
        """
        Record that a batch completed successfully.

        Call this after each successful forward+backward pass to help the
        controller learn which batch sizes are safe.

        Args:
            batch_size: Batch size that succeeded
            memory_util: Memory utilization during the batch (optional)
        """
        if memory_util is None:
            memory_util = self._memory_monitor.get_utilization()

        self._record_safe(batch_size, memory_util)
        self._consecutive_successes += 1

        # Transition from recovering to stable after enough successes
        if self._mode == 'recovering' and self._consecutive_successes >= self._stability_threshold:
            self._mode = 'stable'
            logger.info(f"Recovered to stable mode at BS={batch_size}")

    def record_failure(self, batch_size: int) -> int:
        """
        Record an OOM error and get a safe batch size to use.

        Call this when OOM occurs. Returns a known-safe batch size to use
        for the retry.

        Args:
            batch_size: Batch size that caused OOM

        Returns:
            Safe batch size to use for retry
        """
        self._known_unsafe.add(batch_size)
        self._consecutive_successes = 0
        self._mode = 'recovering'

        # Find largest known-safe size below the failed size
        safe_sizes = [s for s in self._known_safe.keys() if s < batch_size]

        if safe_sizes:
            new_size = max(safe_sizes)
        else:
            # Fallback: reduce by oom_recovery_factor
            new_size = max(
                self._min_batch_size,
                int(batch_size * self._oom_recovery_factor)
            )
            # Round to multiple of min_batch_size
            new_size = (new_size // self._min_batch_size) * self._min_batch_size
            new_size = max(self._min_batch_size, new_size)

        self._current_batch_size = new_size

        # Update safe ceiling to be below the unsafe size
        self._safe_ceiling = min(self._safe_ceiling, batch_size - 1)

        logger.warning(
            f"OOM at BS={batch_size}, falling back to BS={new_size} "
            f"(known_safe={len(self._known_safe)}, known_unsafe={len(self._known_unsafe)})"
        )

        return new_size

    def get_batch_size(self) -> int:
        """
        Get the current recommended batch size.

        After calibration, this returns a stable value. Only changes on OOM.

        Returns:
            Current batch size to use
        """
        return self._current_batch_size

    def get_safe_batch_for_seq_len(self, seq_len: int, base_seq_len: int = 512) -> int:
        """
        Get safe batch size adjusted for sequence length.

        Longer sequences require smaller batch sizes due to quadratic
        attention memory scaling.

        Args:
            seq_len: Current sequence length
            base_seq_len: Reference sequence length for current batch size

        Returns:
            Adjusted batch size safe for the given sequence length
        """
        if seq_len <= base_seq_len:
            return self._current_batch_size

        # Memory scales roughly as O(batch * seq^2) for attention
        # So batch_new = batch_old * (seq_old / seq_new)^2
        ratio = (base_seq_len / seq_len) ** 2
        adjusted = int(self._current_batch_size * ratio)

        # Round to multiple of min_batch_size
        adjusted = (adjusted // self._min_batch_size) * self._min_batch_size
        adjusted = max(self._min_batch_size, min(adjusted, self._current_batch_size))

        return adjusted

    def get_state(self) -> BatchSizeState:
        """Get current state for logging/debugging."""
        return BatchSizeState(
            current_batch_size=self._current_batch_size,
            min_batch_size=self._min_batch_size,
            max_batch_size=self._max_batch_size,
            safe_ceiling=self._safe_ceiling,
            mode=self._mode,
            known_safe_count=len(self._known_safe),
            known_unsafe_count=len(self._known_unsafe),
            consecutive_successes=self._consecutive_successes,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for logging."""
        return {
            'current_batch_size': self._current_batch_size,
            'safe_ceiling': self._safe_ceiling,
            'mode': self._mode,
            'known_safe_sizes': sorted(self._known_safe.keys()),
            'known_unsafe_sizes': sorted(self._known_unsafe),
            'consecutive_successes': self._consecutive_successes,
            'target_memory': self._target_memory,
        }

    @property
    def memory_monitor(self) -> MemoryMonitor:
        """Access the memory monitor for direct utilization queries."""
        return self._memory_monitor

    @property
    def is_stable(self) -> bool:
        """Check if batch size has stabilized."""
        return self._mode == 'stable'

    @property
    def safe_ceiling(self) -> int:
        """Get the highest known-safe batch size."""
        return self._safe_ceiling


def create_batch_size_controller(config: Dict[str, Any]) -> Optional[BatchSizeController]:
    """
    Factory function to create BatchSizeController from config dict.

    Args:
        config: Configuration dictionary (full config or just dynamic_batching section)

    Returns:
        BatchSizeController instance, or None if disabled
    """
    # Handle nested config
    db_config = config.get('dynamic_batching', config)

    if not db_config.get('enabled', False):
        return None

    return BatchSizeController(
        min_batch_size=db_config.get('min_batch_size', 16),
        max_batch_size=db_config.get('max_batch_size', 256),
        target_memory=db_config.get('target_memory_utilization',
                                    db_config.get('target_memory_threshold', 0.75)),
        oom_recovery_factor=db_config.get('oom_recovery_factor', 0.5),
        stability_threshold=db_config.get('stability_threshold', 10),
    )
