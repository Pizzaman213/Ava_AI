"""Training orchestration and optimization coordination."""

from .run_manager import RunManager
from .optimizations import UnifiedOptimizer, OptimizationConfig
from .optimization_integration import OptimizedTrainingSetup

# Import both quick_optimize functions with aliases to avoid conflicts
from .optimizations import quick_optimize as quick_optimize_unified
from .optimization_integration import quick_optimize as quick_optimize_setup

__all__ = [
    "RunManager",
    "UnifiedOptimizer",
    "OptimizationConfig",
    "OptimizedTrainingSetup",
    "quick_optimize_unified",
    "quick_optimize_setup",
]
