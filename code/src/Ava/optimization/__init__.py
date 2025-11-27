"""
Optimization Module

Provides learning rate management for training.
"""

# Learning Rate Management (used by finetune.py)
try:
    from .learning_rate.managers import (
        AdaptiveLearningRateManager,
        PlateauDetector,
        IntelligentLRManager,
        AdvancedWarmupScheduler,
        AdaptiveLRConfig,
        LRConfig,
        WarmupConfig,
    )
except ImportError:
    AdaptiveLearningRateManager = None
    PlateauDetector = None
    IntelligentLRManager = None
    AdvancedWarmupScheduler = None
    AdaptiveLRConfig = None
    LRConfig = None
    WarmupConfig = None

__all__ = [
    # Learning Rate Management
    "AdaptiveLearningRateManager",
    "PlateauDetector",
    "IntelligentLRManager",
    "AdvancedWarmupScheduler",
    "AdaptiveLRConfig",
    "LRConfig",
    "WarmupConfig",
]
