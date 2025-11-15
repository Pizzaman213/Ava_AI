"""Training orchestration and optimization coordination."""

from .run_manager import RunManager
from .optimizations import UnifiedOptimizer, OptimizationConfig
from .optimization_integration import OptimizedTrainingSetup

# Use the comprehensive version from optimization_integration as the primary
from .optimization_integration import quick_optimize

# Keep backward compatibility alias (optimizations version is now a wrapper)
from .optimizations import quick_optimize as quick_optimize_legacy

__all__ = [
    "RunManager",
    "UnifiedOptimizer",
    "OptimizationConfig",
    "OptimizedTrainingSetup",
    "quick_optimize",
    "quick_optimize_legacy",
]
