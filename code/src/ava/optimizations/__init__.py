"""
Training optimizations for improved performance.

This package provides:
- Gradient checkpointing utilities (checkpointing.py)
- Hybrid caching (hybrid_cache.py)
- FP8 training support (fp8.py)
- Async prefetching (prefetch.py)
- Batch size calibration (batch_controller.py)
- Learning rate management (lr_managers/)
- GPU synchronization utilities (gpu_sync.py)
- Memory utilities (memory_utils.py)
- OOM recovery (oom_recovery.py)
- Gradient monitoring (gradients.py -> gpu_sync.py)
"""

# Learning rate management (subpackage)
from .lr_managers import AdaptiveLearningRateManager, AdaptiveLRConfig

# GPU sync utilities (minimize cudaStreamSynchronize overhead)
from .gpu_sync import (
    MetricsBatcher,
    DeferredMetrics,
    batch_to_list,
    batch_mean_to_list,
    accumulate_and_sync,
    # Gradient utilities (merged from gradients.py)
    check_gradients,
    check_gradients_deferred,
    detect_gradient_issues,
)

# Batch size control and memory monitoring
from .batch_controller import BatchSizeController, MemoryMonitor

# OOM recovery
from .oom_recovery import OOMRecoveryManager, proactive_memory_cleanup

# Async batch prefetching
from .prefetch import AsyncBatchPrefetcher

# Checkpointing utilities
from .checkpointing import double_checkpoint, DoubleCheckpointConfig

# Memory utilities
from .memory_utils import (
    get_memory_stats,
    get_memory_utilization,
    get_memory_fragmentation,
    get_memory_cache_if_available,
    is_memory_cache_available,
    is_under_memory_pressure,
    reset_peak_memory,
    MemoryStatsCache,
)

__all__ = [
    # Learning rate management
    'AdaptiveLearningRateManager',
    'AdaptiveLRConfig',
    # GPU sync utilities
    'MetricsBatcher',
    'DeferredMetrics',
    'batch_to_list',
    'batch_mean_to_list',
    'accumulate_and_sync',
    # Gradient utilities
    'check_gradients',
    'check_gradients_deferred',
    'detect_gradient_issues',
    # Batch size control
    'BatchSizeController',
    'MemoryMonitor',
    # OOM recovery
    'OOMRecoveryManager',
    'proactive_memory_cleanup',
    # Async prefetching
    'AsyncBatchPrefetcher',
    # Checkpointing
    'double_checkpoint',
    'DoubleCheckpointConfig',
    # Memory utilities
    'get_memory_stats',
    'get_memory_utilization',
    'get_memory_fragmentation',
    'get_memory_cache_if_available',
    'is_memory_cache_available',
    'is_under_memory_pressure',
    'reset_peak_memory',
    'MemoryStatsCache',
]
