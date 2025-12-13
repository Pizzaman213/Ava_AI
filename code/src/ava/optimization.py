"""
Backward compatibility module for optimizers.

Re-exports from optim package for backward compatibility with older import paths.
"""

from .optim import (
    AdaptiveLRConfig,
    AdaptiveLearningRateManager,
    LRConfig,
)

__all__ = [
    'AdaptiveLRConfig',
    'AdaptiveLearningRateManager',
    'LRConfig',
]
