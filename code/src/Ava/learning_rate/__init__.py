"""
Learning Rate Management and Scheduling Components

This module provides comprehensive learning rate management for training.

Main Components (Production Use):
- AdaptiveLRManager: Real-time adaptive LR with loss monitoring
- IntelligentLRManager: High-level LR coordinator
- AdvancedWarmupScheduler: Comprehensive warmup system
- PlateauDetector: Validation-based plateau detection
- LRFinder: Learning rate range testing

Research Components (Experimental):
- CosineAnnealingWarmRestarts: SGDR scheduler
- OneCycleLR: Super-convergence scheduler
- PolynomialDecayLR: Polynomial decay scheduler
- AdaptiveLRScheduler: Performance-based adaptive scheduler
- NoisyStudentScheduler: Noisy student training
- SchedulerFactory: Easy scheduler instantiation

Configuration Classes:
- AdaptiveLRConfig: Config for adaptive LR management
- LRConfig: Config for intelligent LR management
- WarmupConfig: Config for warmup scheduling
"""

# ============================================================================
# Primary Exports from New Consolidated Modules
# ============================================================================

from .adaptive_manager import (
    # Main classes
    AdaptiveLearningRateManager,
    IntelligentLRManager,
    PlateauDetector,
    AdvancedWarmupScheduler,
    # Configuration classes
    AdaptiveLRConfig,
    LRConfig,
    WarmupConfig,
    WarmupSchedule,
    # Convenience functions
    create_conservative_lr_config,
    create_aggressive_lr_config,
    create_balanced_lr_config,
    create_linear_warmup_config,
    create_cosine_warmup_config,
    create_polynomial_warmup_config,
    create_adaptive_warmup_config,
)

from .research_schedulers import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecayLR,
    AdaptiveLRScheduler,
    NoisyStudentScheduler,
    SchedulerFactory,
)

from .lr_finder import LRFinder, LRFinderConfig

# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Main aliases for backward compatibility
AdaptiveLRManager = AdaptiveLearningRateManager
LRManager = IntelligentLRManager

# Backward compatibility for deprecated imports from advanced_warmup_scheduling
# These were rarely/never used, but we keep them for safety
try:
    from .advanced_warmup_scheduling import (
        GradientNoiseScale,
        LearningRateFinder,
        CyclicalBatchScheduler,
        AdaptiveWarmupScheduler as AdaptiveWarmup,
    )
    _has_deprecated_scheduling = True
except ImportError:
    # If file has been removed, provide stub implementations that warn users
    _has_deprecated_scheduling = False
    import warnings

    def _deprecated_import(name):
        warnings.warn(
            f"{name} has been deprecated and removed. "
            f"Use adaptive_manager.py or research_schedulers.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(f"{name} is no longer available")

    class GradientNoiseScale:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _deprecated_import("GradientNoiseScale")

    class LearningRateFinder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _deprecated_import("LearningRateFinder")

    class CyclicalBatchScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _deprecated_import("CyclicalBatchScheduler")

    class AdaptiveWarmup:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _deprecated_import("AdaptiveWarmup")


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Production components
    "AdaptiveLearningRateManager",
    "AdaptiveLRManager",  # Alias
    "IntelligentLRManager",
    "LRManager",  # Alias
    "PlateauDetector",
    "AdvancedWarmupScheduler",
    "LRFinder",
    "LRFinderConfig",

    # Configuration classes
    "AdaptiveLRConfig",
    "LRConfig",
    "WarmupConfig",
    "WarmupSchedule",

    # Convenience functions
    "create_conservative_lr_config",
    "create_aggressive_lr_config",
    "create_balanced_lr_config",
    "create_linear_warmup_config",
    "create_cosine_warmup_config",
    "create_polynomial_warmup_config",
    "create_adaptive_warmup_config",

    # Research schedulers
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecayLR",
    "AdaptiveLRScheduler",
    "NoisyStudentScheduler",
    "SchedulerFactory",

    # Deprecated but maintained for compatibility
    "GradientNoiseScale",
    "LearningRateFinder",
    "CyclicalBatchScheduler",
    "AdaptiveWarmup",
]
