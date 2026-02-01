"""
Shared GPU memory utilities for the Ava training framework.

This module provides centralized memory querying and management utilities
used across batch_controller, oom_recovery, prefetch, and other modules.

Centralizing these utilities:
1. Eliminates duplicate GPU->CPU sync patterns
2. Provides consistent memory threshold handling
3. Simplifies optional dependency management (ava.cuda.memory_cache)

Key Functions:
    get_memory_stats() -> Dict[str, float]
    get_memory_utilization() -> float
    get_memory_fragmentation() -> float
    proactive_memory_cleanup(threshold, force) -> bool
    get_memory_cache_if_available() -> Optional[MemoryMetricsCache]
"""

import gc
import logging
import time
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Cache for optional memory_cache module
_MEMORY_CACHE_CHECKED = False
_MEMORY_CACHE_AVAILABLE = False
_MEMORY_CACHE_MODULE = None


def get_memory_cache_if_available() -> Optional[Any]:
    """
    Get the unified MemoryMetricsCache if available.

    This function handles the optional import of ava.cuda.memory_cache
    and caches the result to avoid repeated import attempts.

    Returns:
        The memory cache instance if available, None otherwise.

    Example:
        >>> cache = get_memory_cache_if_available()
        >>> if cache:
        ...     util = cache.get_utilization()
        ... else:
        ...     # Fallback to direct querying
        ...     util = get_memory_utilization()
    """
    global _MEMORY_CACHE_CHECKED, _MEMORY_CACHE_AVAILABLE, _MEMORY_CACHE_MODULE

    if not _MEMORY_CACHE_CHECKED:
        try:
            from ava.cuda.memory_cache import get_memory_cache
            _MEMORY_CACHE_MODULE = get_memory_cache
            _MEMORY_CACHE_AVAILABLE = True
        except ImportError:
            _MEMORY_CACHE_AVAILABLE = False
            _MEMORY_CACHE_MODULE = None
        _MEMORY_CACHE_CHECKED = True

    if _MEMORY_CACHE_AVAILABLE and _MEMORY_CACHE_MODULE is not None:
        try:
            return _MEMORY_CACHE_MODULE()
        except Exception:
            return None
    return None


def is_memory_cache_available() -> bool:
    """
    Check if the unified memory cache is available.

    Returns:
        True if ava.cuda.memory_cache is available, False otherwise.
    """
    # Trigger cache check if not done
    get_memory_cache_if_available()
    return _MEMORY_CACHE_AVAILABLE


def get_memory_utilization(device: Optional[torch.device] = None) -> float:
    """
    Get current GPU memory utilization (0.0 to 1.0).

    Uses unified cache if available, otherwise queries CUDA directly.

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        Memory utilization as a fraction (0.0 to 1.0).
        Returns 0.0 if CUDA is not available.

    Example:
        >>> util = get_memory_utilization()
        >>> if util > 0.9:
        ...     print("Memory pressure detected!")
    """
    if not torch.cuda.is_available():
        return 0.0

    # Use unified cache if available
    cache = get_memory_cache_if_available()
    if cache is not None:
        try:
            return cache.get_utilization()
        except Exception:
            pass  # Fall through to direct query

    # Direct query fallback
    try:
        if device is None:
            device = torch.cuda.current_device()

        total = torch.cuda.get_device_properties(device).total_memory
        if total == 0:
            return 0.0

        # Use peak allocated for accurate measurement
        stats = torch.cuda.memory_stats(device)
        allocated = stats.get('allocated_bytes.all.peak', 0)

        return allocated / total
    except Exception:
        return 0.0


def get_memory_fragmentation(device: Optional[torch.device] = None) -> float:
    """
    Calculate GPU memory fragmentation ratio.

    Fragmentation occurs when allocated memory is much smaller than reserved memory,
    indicating the CUDA allocator is holding onto freed blocks that can't be coalesced.

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
        Returns 0.0 if CUDA is not available or no memory is reserved.

    Example:
        >>> frag = get_memory_fragmentation()
        >>> if frag > 0.3:
        ...     proactive_memory_cleanup(force=True)
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        if device is None:
            device = torch.cuda.current_device()

        stats = torch.cuda.memory_stats(device)
        allocated = stats.get('allocated_bytes.all.current', 0)
        reserved = stats.get('reserved_bytes.all.current', 0)

        if reserved == 0:
            return 0.0

        # Fragmentation = 1 - (allocated / reserved)
        # High fragmentation means lots of reserved but unused memory
        return 1.0 - (allocated / reserved)
    except Exception:
        return 0.0


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get detailed GPU memory statistics.

    Uses unified cache if available, otherwise queries CUDA directly.

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        Dictionary with memory statistics:
        - allocated_gb: Currently allocated GPU memory in GB
        - reserved_gb: Total reserved GPU memory in GB
        - peak_allocated_gb: Peak allocated memory in GB
        - fragmentation: Fragmentation ratio (0.0-1.0)
        - free_gb: Free GPU memory in GB (total - reserved)
        - total_gb: Total GPU memory in GB
        - utilization: Memory utilization ratio (0.0-1.0)

    Example:
        >>> stats = get_memory_stats()
        >>> print(f"Using {stats['allocated_gb']:.1f} GB of {stats['total_gb']:.1f} GB")
    """
    empty_stats = {
        'allocated_gb': 0.0,
        'reserved_gb': 0.0,
        'peak_allocated_gb': 0.0,
        'fragmentation': 0.0,
        'free_gb': 0.0,
        'total_gb': 0.0,
        'utilization': 0.0,
        'utilization_current': 0.0,
        'utilization_reserved': 0.0,
    }

    if not torch.cuda.is_available():
        return empty_stats

    # Use unified cache if available
    cache = get_memory_cache_if_available()
    if cache is not None:
        try:
            return cache.get_stats()
        except Exception:
            pass  # Fall through to direct query

    # Direct query fallback
    try:
        if device is None:
            device = torch.cuda.current_device()

        stats = torch.cuda.memory_stats(device)
        allocated = stats.get('allocated_bytes.all.current', 0)
        reserved = stats.get('reserved_bytes.all.current', 0)
        peak_allocated = stats.get('allocated_bytes.all.peak', 0)
        total = torch.cuda.get_device_properties(device).total_memory

        gb = 1024 ** 3

        # Calculate fragmentation
        fragmentation = 0.0
        if reserved > 0:
            fragmentation = 1.0 - (allocated / reserved)

        return {
            'allocated_gb': allocated / gb,
            'reserved_gb': reserved / gb,
            'peak_allocated_gb': peak_allocated / gb,
            'fragmentation': fragmentation,
            'free_gb': (total - reserved) / gb,
            'total_gb': total / gb,
            'utilization': peak_allocated / total if total > 0 else 0.0,
            'utilization_current': allocated / total if total > 0 else 0.0,
            'utilization_reserved': reserved / total if total > 0 else 0.0,
        }
    except Exception:
        return empty_stats


def proactive_memory_cleanup(
    fragmentation_threshold: float = 0.30,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Proactively clean up GPU memory if fragmentation is high.

    This helps prevent OOM errors by periodically defragmenting the CUDA allocator.
    Call this during natural pauses in training (after validation, generation, etc.).

    Args:
        fragmentation_threshold: Trigger cleanup when fragmentation exceeds this (default: 0.30)
        force: If True, always perform cleanup regardless of fragmentation level
        device: CUDA device to clean. If None, cleans all devices.

    Returns:
        True if cleanup was performed, False otherwise

    Example:
        >>> # After validation
        >>> proactive_memory_cleanup()

        >>> # Force cleanup before generation
        >>> proactive_memory_cleanup(force=True)
    """
    if not torch.cuda.is_available():
        return False

    fragmentation = get_memory_fragmentation(device)

    if force or fragmentation > fragmentation_threshold:
        # Get stats before cleanup for logging
        stats_before = get_memory_stats(device)

        # Perform cleanup
        torch.cuda.empty_cache()

        # Force Python garbage collection to release any PyTorch tensors
        gc.collect()

        # Get stats after cleanup
        stats_after = get_memory_stats(device)

        freed_gb = stats_before['reserved_gb'] - stats_after['reserved_gb']

        if freed_gb > 0.01:  # Only log if we freed more than 10MB
            logger.debug(
                f"Proactive memory cleanup: freed {freed_gb:.2f} GB, "
                f"fragmentation {stats_before['fragmentation']:.1%} -> {stats_after['fragmentation']:.1%}"
            )

        return True

    return False


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    """
    Reset peak memory tracking.

    Call this before a measurement period to get accurate peak memory readings.

    Args:
        device: CUDA device. If None, resets for current device.
    """
    if not torch.cuda.is_available():
        return

    try:
        if device is None:
            device = torch.cuda.current_device()

        torch.cuda.reset_peak_memory_stats(device)

        # Also reset unified cache if available
        cache = get_memory_cache_if_available()
        if cache is not None and hasattr(cache, 'reset_peak_memory'):
            cache.reset_peak_memory()
    except Exception:
        pass


def is_under_memory_pressure(
    threshold: float = 0.90,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Check if GPU memory utilization exceeds threshold.

    Useful for implementing memory-aware backpressure in prefetchers
    and batch controllers.

    Args:
        threshold: Memory utilization threshold (default: 0.90 = 90%)
        device: CUDA device to check. If None, uses current device.

    Returns:
        True if memory utilization exceeds threshold, False otherwise.

    Example:
        >>> while is_under_memory_pressure():
        ...     time.sleep(0.1)
        ...     gc.collect()
        ...     torch.cuda.empty_cache()
    """
    # Use unified cache if available for lower overhead
    cache = get_memory_cache_if_available()
    if cache is not None:
        try:
            return cache.is_under_pressure(threshold=threshold)
        except Exception:
            pass

    return get_memory_utilization(device) > threshold


# Cached memory info for reducing sync overhead
class MemoryStatsCache:
    """
    Thread-safe cached memory statistics with configurable TTL.

    Reduces GPU->CPU sync overhead by caching memory stats and
    only refreshing when the cache is stale.

    Example:
        >>> cache = MemoryStatsCache(cache_interval_sec=5.0)
        >>> # Fast path - uses cache if fresh
        >>> util = cache.get_utilization()
        >>> # Force refresh
        >>> util = cache.get_utilization(force_refresh=True)
    """

    def __init__(
        self,
        cache_interval_sec: float = 5.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize memory stats cache.

        Args:
            cache_interval_sec: How often to refresh memory stats (seconds).
            device: CUDA device to monitor. If None, uses current device.
        """
        self._cache_interval = cache_interval_sec
        self._device = device
        self._cache: Dict[str, float] = {}
        self._cache_time: float = 0.0
        self._total_memory: int = 0

        if torch.cuda.is_available():
            dev = device if device is not None else torch.cuda.current_device()
            try:
                self._total_memory = torch.cuda.get_device_properties(dev).total_memory
            except Exception:
                pass

    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        return time.monotonic() - self._cache_time > self._cache_interval

    def _refresh_cache(self) -> None:
        """Query GPU memory and update cache."""
        self._cache = get_memory_stats(self._device)
        self._cache_time = time.monotonic()

    def get_utilization(self, force_refresh: bool = False) -> float:
        """Get memory utilization (0.0 to 1.0)."""
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.get('utilization', 0.0)

    def get_stats(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get full memory statistics."""
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.copy()

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        reset_peak_memory(self._device)
        # Invalidate cache after reset
        self._cache_time = 0.0


__all__ = [
    'get_memory_cache_if_available',
    'is_memory_cache_available',
    'get_memory_utilization',
    'get_memory_fragmentation',
    'get_memory_stats',
    'proactive_memory_cleanup',
    'reset_peak_memory',
    'is_under_memory_pressure',
    'MemoryStatsCache',
]
