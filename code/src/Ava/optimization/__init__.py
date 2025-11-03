"""
Optimization Module

Consolidated optimization functionality including:
- Advanced optimizers (Lion, Sophia, AdaFactor)
- Learning rate schedulers and management
- LR finder (range test)
- Gradient operations (health monitoring, surgery)
- Quantization (model compression)
- FP8 training (mixed precision)

Organized into submodules:
- gradients: Gradient health monitoring and surgery
- learning_rate: LR schedulers, finders, and warmup strategies
- optimizers: Advanced optimizer implementations
- precision: FP8 training and quantization
"""

# Note: The following files are in _archived/optimization/ but re-exported here for compatibility:
# - a100_optimizer.py
# - flash_attention_v3.py
# - nvlink_optimizer.py
# - memory_optimizer.py
# - fused_optimizers.py
# - gradient_optimizations.py
# - compilation_optimizations.py
# - hardware_optimizations.py
# These are experimental/optional optimizations used by the unified optimization system

# Import from new submodule structure
from .precision.quantization import (
    ModelQuantizer,
    LinearQuantized,
    DynamicQuantization,
    INT4Quantization,
    QuantizationObserver,
)

from .optimizers.advanced import (
    LionOptimizer,
    SophiaOptimizer,
    AdaFactorOptimizer,
    OptimizerFactory
)

from .learning_rate.managers import (
    AdaptiveLearningRateManager,
    PlateauDetector,
    IntelligentLRManager,
    AdvancedWarmupScheduler,
    AdaptiveLRConfig,
    LRConfig,
    WarmupConfig,
)

from .learning_rate.finder import (
    LRFinder,
    LRFinderConfig,
    find_lr,
)

from .learning_rate.research import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecayLR,
    AdaptiveLRScheduler,
    NoisyStudentScheduler,
    SchedulerFactory,
)

from .gradients.health import GradientHealthMonitor, LossHealthMonitor
from .gradients.surgery import (
    GradientSurgeon,
    AdaptiveGradientSurgeon,
    GradientConflictAnalyzer,
)

from .precision.fp8 import (
    FP8Handler,
    FP8Linear,
    FP8MultiHeadAttention,
    FP8LayerNorm,
    FP8TransformerLayer,
    FP8ModelWrapper,
)

# Backward compatibility aliases for old import paths
GradientSurgery = GradientSurgeon  # Old name compatibility

# Re-export modules from _archived for compatibility with optimization_integration
try:
    from .._archived.optimization import (  # type: ignore[import-not-found]
        gradient_optimizations,
        fused_optimizers,
        hardware_optimizations,
        compilation_optimizations,
    )
except ImportError:
    # Graceful fallback if _archived modules not available
    gradient_optimizations = None
    fused_optimizers = None
    hardware_optimizations = None
    compilation_optimizations = None

__all__ = [
    # Quantization
    "ModelQuantizer",
    "LinearQuantized",
    "DynamicQuantization",
    "INT4Quantization",
    "QuantizationObserver",

    # Advanced Optimizers
    "LionOptimizer",
    "SophiaOptimizer",
    "AdaFactorOptimizer",
    "OptimizerFactory",

    # Learning Rate Management
    "AdaptiveLearningRateManager",
    "PlateauDetector",
    "IntelligentLRManager",
    "AdvancedWarmupScheduler",
    "AdaptiveLRConfig",
    "LRConfig",
    "WarmupConfig",

    # LR Finder
    "LRFinder",
    "LRFinderConfig",
    "find_lr",

    # Research Schedulers
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecayLR",
    "AdaptiveLRScheduler",
    "NoisyStudentScheduler",
    "SchedulerFactory",

    # Gradient Operations
    "GradientHealthMonitor",
    "LossHealthMonitor",
    "GradientSurgeon",
    "AdaptiveGradientSurgeon",
    "GradientConflictAnalyzer",
    "GradientSurgery",  # Backward compatibility alias

    # FP8 Training
    "FP8Handler",
    "FP8Linear",
    "FP8MultiHeadAttention",
    "FP8LayerNorm",
    "FP8TransformerLayer",
    "FP8ModelWrapper",

    # Re-exported optimization modules
    "gradient_optimizations",
    "fused_optimizers",
    "hardware_optimizations",
    "compilation_optimizations",
]