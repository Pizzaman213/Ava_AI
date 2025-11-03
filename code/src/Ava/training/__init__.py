"""
Training utilities and advanced training techniques.

This module has been reorganized into subdirectories for better organization:
- core/: Core training infrastructure (EnhancedTrainer, RunManager, OptimizationIntegration)
- learning_rate/: Learning rate management and scheduling
- distributed/: Distributed training management
- gradients/: Gradient management and optimization
- strategies/: Training strategies (progressive training, performance modes)
- monitoring/: Training monitoring and metrics

For backward compatibility, all public APIs are re-exported from this module.
"""

# Note: The following files have been moved to _archived/training/:
# - profiling_tools.py
# - dynamic_batch_sampler.py
# - progressive_batch_scheduler.py
# - qlora_utils.py
# - distributed_optimizations.py
# These are optional training strategies/tools not used in the default pipeline

# Core training infrastructure
from .core.enhanced_trainer import EnhancedModularTrainer
from .core.run_manager import RunManager
from .core.optimization_integration import OptimizedTrainingSetup as OptimizationIntegration

# Learning rate management
from .learning_rate import (
    AdaptiveLearningRateManager,
    AdaptiveLRConfig,
    IntelligentLRManager,
    LRConfig,
    LRFinder,
    LRFinderConfig
)

# Distributed training
from .distributed.distributed_manager import (
    DistributedManager,
    DistributedConfig,
    get_distributed_manager,
    get_rank,
    get_world_size
)
from .distributed.unified_distributed_manager import UnifiedDistributedManager
from .distributed.distributed_health_checker import (
    DistributedHealthChecker,
    get_health_checker,
    record_training_metrics
)
from .distributed.rank_aware_error_handler import (
    RankAwareErrorHandler,
    ErrorSeverity,
    ErrorType,
    get_error_handler
)

# Gradient management
from .gradients.gradient_surgery import (
    GradientSurgeon,
    AdaptiveGradientSurgeon,
    GradientConflictAnalyzer,
)
from .gradients.gradient_health import GradientHealthMonitor, LossHealthMonitor

# Monitoring
from .monitoring.unified_optimizations import UnifiedOptimizer as UnifiedOptimizations

from .learning_rate import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecayLR,
    AdaptiveLRScheduler,
    NoisyStudentScheduler,
    SchedulerFactory,
    AdvancedWarmupScheduler,
    WarmupConfig,
    WarmupSchedule,
    create_linear_warmup_config,
    create_cosine_warmup_config,
    create_polynomial_warmup_config,
    create_adaptive_warmup_config
)

from .strategies.progressive_training import (
    ProgressiveTrainingConfig,
    CurriculumLearning,
    GrowLengthScheduler,
    DynamicBatchSizer,
    ProgressiveModelScaler,
    ProgressiveTrainer,
    ProgressiveTrainingManager
)


from .strategies.performance_modes import (
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

from .monitoring.metrics import (
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
    # Core
    "EnhancedModularTrainer",
    "RunManager",
    "OptimizationIntegration",

    # Learning Rate
    "AdaptiveLearningRateManager",
    "AdaptiveLRConfig",
    "IntelligentLRManager",
    "LRConfig",
    "LRFinder",
    "LRFinderConfig",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecayLR",
    "AdaptiveLRScheduler",
    "NoisyStudentScheduler",
    "SchedulerFactory",
    "AdvancedWarmupScheduler",
    "WarmupConfig",
    "WarmupSchedule",
    "create_linear_warmup_config",
    "create_cosine_warmup_config",
    "create_polynomial_warmup_config",
    "create_adaptive_warmup_config",

    # Distributed
    "DistributedManager",
    "DistributedConfig",
    "get_distributed_manager",
    "get_rank",
    "get_world_size",
    "UnifiedDistributedManager",
    "DistributedHealthChecker",
    "get_health_checker",
    "record_training_metrics",
    "RankAwareErrorHandler",
    "ErrorSeverity",
    "ErrorType",
    "get_error_handler",

    # Gradient Surgery & Health
    "GradientSurgeon",
    "AdaptiveGradientSurgeon",
    "GradientConflictAnalyzer",
    "GradientHealthMonitor",
    "LossHealthMonitor",

    # Progressive Training & Strategies
    "ProgressiveTrainingConfig",
    "CurriculumLearning",
    "GrowLengthScheduler",
    "DynamicBatchSizer",
    "ProgressiveModelScaler",
    "ProgressiveTrainer",
    "ProgressiveTrainingManager",

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

    # Training Metrics & Monitoring
    "TrainingMetricsCollector",
    "MetricConfig",
    "TrainingStep",
    "MetricType",
    "TimingContext",
    "create_comprehensive_metrics_config",
    "create_fast_metrics_config",
    "create_minimal_metrics_config",
    "UnifiedOptimizations",
]