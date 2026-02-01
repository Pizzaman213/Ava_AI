"""
Learning Rate Management Package

This package provides comprehensive learning rate management utilities:

Modules:
- configs: Configuration dataclasses (AdaptiveLRConfig, LRConfig, WarmupConfig)
- adaptive: AdaptiveLearningRateManager for real-time LR adaptation
- warmup: AdvancedWarmupScheduler for sophisticated warmup scheduling
- intelligent: IntelligentLRManager and PlateauDetector for high-level coordination
- factories: Convenience factory functions for common configurations

Usage:
    from ava.optimizations.lr_managers import (
        AdaptiveLearningRateManager,
        AdaptiveLRConfig,
        create_cosine_warmup_config,
    )

Backward Compatibility:
    All imports that worked from lr_managers.py still work:
    - from ava.optimizations.lr_managers import AdaptiveLearningRateManager
    - from ava.optimizations.lr_managers import AdaptiveLRConfig
    etc.
"""

# Configuration classes
from .configs import (
    WarmupSchedule,
    AdaptiveLRConfig,
    LRConfig,
    WarmupConfig,
)

# Adaptive LR management
from .adaptive import AdaptiveLearningRateManager

# Advanced warmup
from .warmup import AdvancedWarmupScheduler

# Intelligent LR management
from .intelligent import PlateauDetector, IntelligentLRManager

# Factory functions
from .factories import (
    create_conservative_lr_config,
    create_aggressive_lr_config,
    create_balanced_lr_config,
    create_linear_warmup_config,
    create_cosine_warmup_config,
    create_polynomial_warmup_config,
    create_adaptive_warmup_config,
)

__all__ = [
    # Configuration classes
    'WarmupSchedule',
    'AdaptiveLRConfig',
    'LRConfig',
    'WarmupConfig',
    # Managers
    'AdaptiveLearningRateManager',
    'AdvancedWarmupScheduler',
    'PlateauDetector',
    'IntelligentLRManager',
    # Factory functions
    'create_conservative_lr_config',
    'create_aggressive_lr_config',
    'create_balanced_lr_config',
    'create_linear_warmup_config',
    'create_cosine_warmup_config',
    'create_polynomial_warmup_config',
    'create_adaptive_warmup_config',
]
