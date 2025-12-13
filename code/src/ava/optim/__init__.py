"""
Optimizers and learning rate scheduling.

This package provides:
- AdaptiveLearningRateManager: Dynamic LR adjustment
- PlateauDetector: Loss plateau detection
- AdvancedWarmupScheduler: Warmup scheduling
"""

from .lr_managers import (
    AdaptiveLRConfig,
    AdaptiveLearningRateManager,
    LRConfig,
)

__all__ = [
    'AdaptiveLRConfig',
    'AdaptiveLearningRateManager',
    'LRConfig',
]
