"""
Training utilities and advanced training techniques.
"""

from .gradient_surgery import (
    GradientSurgeon,
    AdaptiveGradientSurgeon,
    GradientConflictAnalyzer
)

from .advanced_schedulers import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecayLR,
    AdaptiveLRScheduler,
    NoisyStudentScheduler,
    SchedulerFactory
)

from .progressive_training import (
    ProgressiveTrainingConfig,
    CurriculumLearning,
    GrowLengthScheduler,
    DynamicBatchSizer,
    ProgressiveModelScaler,
    ProgressiveTrainer
)

from .advanced_warmup import (
    AdvancedWarmupScheduler,
    WarmupConfig,
    WarmupSchedule,
    create_linear_warmup_config,
    create_cosine_warmup_config,
    create_polynomial_warmup_config,
    create_adaptive_warmup_config
)

from .adaptive_lr import (
    AdaptiveLearningRateManager,
    AdaptiveLRConfig,
    create_conservative_lr_config,
    create_aggressive_lr_config,
    create_balanced_lr_config
)

from .performance_modes import (
    PerformanceModeManager,
    PerformanceModeConfig,
    PerformanceMode,
    create_ultra_fast_config,
    create_fast_progress_config,
    create_minimal_progress_config,
    create_express_config,
    create_no_sync_config,
    create_standard_config,
    detect_performance_mode_from_args,
    create_config_from_args
)

from .metrics import (
    TrainingMetricsCollector,
    MetricConfig,
    TrainingStep,
    MetricType,
    TimingContext,
    create_comprehensive_metrics_config,
    create_fast_metrics_config,
    create_minimal_metrics_config
)

__all__ = [
    # Gradient Surgery
    "GradientSurgeon",
    "AdaptiveGradientSurgeon",
    "GradientConflictAnalyzer",

    # Advanced Schedulers
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecayLR",
    "AdaptiveLRScheduler",
    "NoisyStudentScheduler",
    "SchedulerFactory",

    # Progressive Training
    "ProgressiveTrainingConfig",
    "CurriculumLearning",
    "GrowLengthScheduler",
    "DynamicBatchSizer",
    "ProgressiveModelScaler",
    "ProgressiveTrainer",

    # Advanced Warmup
    "AdvancedWarmupScheduler",
    "WarmupConfig",
    "WarmupSchedule",
    "create_linear_warmup_config",
    "create_cosine_warmup_config",
    "create_polynomial_warmup_config",
    "create_adaptive_warmup_config",

    # Adaptive Learning Rate
    "AdaptiveLearningRateManager",
    "AdaptiveLRConfig",
    "create_conservative_lr_config",
    "create_aggressive_lr_config",
    "create_balanced_lr_config",

    # Performance Modes
    "PerformanceModeManager",
    "PerformanceModeConfig",
    "PerformanceMode",
    "create_ultra_fast_config",
    "create_fast_progress_config",
    "create_minimal_progress_config",
    "create_express_config",
    "create_no_sync_config",
    "create_standard_config",
    "detect_performance_mode_from_args",
    "create_config_from_args",

    # Training Metrics
    "TrainingMetricsCollector",
    "MetricConfig",
    "TrainingStep",
    "MetricType",
    "TimingContext",
    "create_comprehensive_metrics_config",
    "create_fast_metrics_config",
    "create_minimal_metrics_config"
]