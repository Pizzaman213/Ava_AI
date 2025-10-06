"""
Training utilities and advanced training techniques.
"""

# Note: The following files have been moved to _archived/training/:
# - profiling_tools.py
# - dynamic_batch_sampler.py
# - progressive_batch_scheduler.py
# - qlora_utils.py
# - distributed_optimizations.py
# These are optional training strategies/tools not used in the default pipeline

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