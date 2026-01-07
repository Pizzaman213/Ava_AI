"""
Training optimizations for improved performance.

This package provides:
- Gradient checkpointing utilities
- Hybrid caching
- FP8 training support
- Async prefetching
- Batch size calibration
- Learning rate management (moved from optim/)
"""

# Learning rate management (moved from optim/)
from .lr_managers import AdaptiveLearningRateManager, AdaptiveLRConfig

__all__ = [
    'AdaptiveLearningRateManager',
    'AdaptiveLRConfig',
]
