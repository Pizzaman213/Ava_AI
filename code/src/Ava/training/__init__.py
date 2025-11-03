"""
Training Infrastructure Module

Organized into submodules:
- core: Enhanced trainer (core training loop)
- orchestration: Run management, optimization coordination
- monitoring: Metrics collection, performance modes
- strategies: Progressive training, curriculum learning

Consolidated training functionality including:
- Enhanced trainer (core training loop)
- Run management (directory and logging setup)
- Optimization integration (optimizer and scheduler setup)
- Performance modes (resource management)
- Progressive training (curriculum and scaling)
- Metrics (training metrics collection)
- Optimizations (unified optimization orchestration)
"""

# Core training loop
from .core import EnhancedTrainer

# Orchestration and optimization coordination
from .orchestration.optimizations import (
    UnifiedOptimizer,
    OptimizationConfig,
    quick_optimize,
)
try:
    from .orchestration.run_manager import RunManager
except ImportError:
    RunManager = None

# Monitoring and performance
from .monitoring.metrics import (
    TrainingMetricsCollector,
    MoEMetricsTracker,
    MetricConfig,
)
from .monitoring.performance_modes import (
    PerformanceMode,
    PerformanceModeManager,
)

# Advanced training strategies
from .strategies.progressive_training import (
    CurriculumLearning,
    GrowLengthScheduler,
    DynamicBatchSizer,
    ProgressiveModelScaler,
    ProgressiveTrainer,
)

__all__ = [
    # Core
    "EnhancedTrainer",
    # Orchestration
    "RunManager",
    "UnifiedOptimizer",
    "OptimizationConfig",
    "quick_optimize",
    # Monitoring
    "TrainingMetricsCollector",
    "MoEMetricsTracker",
    "MetricConfig",
    "PerformanceMode",
    "PerformanceModeManager",
    # Strategies
    "CurriculumLearning",
    "GrowLengthScheduler",
    "DynamicBatchSizer",
    "ProgressiveModelScaler",
    "ProgressiveTrainer",
]
