"""Training orchestration and optimization coordination."""

from .run_manager import RunManager
from .pipeline import TrainingPipeline, ErrorSeverity, ComponentError

try:
    from .optimizations import UnifiedOptimizer, OptimizationConfig
    from .optimization_integration import OptimizedTrainingSetup, quick_optimize
    _OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    _OPTIMIZATIONS_AVAILABLE = False
    UnifiedOptimizer = None
    OptimizationConfig = None
    OptimizedTrainingSetup = None
    quick_optimize = None

__all__ = [
    "RunManager",
    "TrainingPipeline",
    "ErrorSeverity",
    "ComponentError",
]

if _OPTIMIZATIONS_AVAILABLE:
    __all__.extend([
        "UnifiedOptimizer",
        "OptimizationConfig",
        "OptimizedTrainingSetup",
        "quick_optimize",
    ])
