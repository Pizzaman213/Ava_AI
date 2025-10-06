"""
Optimization module for model quantization, compression, and advanced optimizers.
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

from .quantization import (
    ModelQuantizer,
    LinearQuantized,
    DynamicQuantization,
    INT4Quantization,
    QuantizationConfig,
    QuantizationObserver,
    quantize_model_pipeline
)

from .advanced_optimizers import (
    LionOptimizer,
    SophiaOptimizer,
    AdaFactorOptimizer,
    OptimizerFactory
)

from .fp8_training import (
    FP8Format,
    FP8Config,
    FP8Handler,
    FP8Linear,
    FP8MultiHeadAttention,
    FP8LayerNorm,
    FP8TransformerLayer,
    FP8ModelWrapper,
    create_fp8_model,
    benchmark_fp8_training
)

# Re-export modules from _archived for compatibility with optimization_integration
try:
    from .._archived.optimization import (
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
    "QuantizationConfig",
    "QuantizationObserver",
    "quantize_model_pipeline",

    # Advanced Optimizers
    "LionOptimizer",
    "SophiaOptimizer",
    "AdaFactorOptimizer",
    "OptimizerFactory",

    # FP8 Training
    "FP8Format",
    "FP8Config",
    "FP8Handler",
    "FP8Linear",
    "FP8MultiHeadAttention",
    "FP8LayerNorm",
    "FP8TransformerLayer",
    "FP8ModelWrapper",
    "create_fp8_model",
    "benchmark_fp8_training",

    # Re-exported optimization modules
    "gradient_optimizations",
    "fused_optimizers",
    "hardware_optimizations",
    "compilation_optimizations",
]