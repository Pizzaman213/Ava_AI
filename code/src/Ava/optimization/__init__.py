"""
Optimization module for model quantization, compression, and advanced optimizers.
"""

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
    "benchmark_fp8_training"
]