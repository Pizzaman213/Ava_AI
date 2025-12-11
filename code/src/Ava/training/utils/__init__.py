"""
Training utilities for the Ava pipeline.

Extracted from train_100m_full.py for modularity.
"""

from .distributed import setup_distributed, cleanup_distributed, DISTRIBUTED_AVAILABLE
from .gradient_utils import check_gradients
from .async_prefetcher import AsyncBatchPrefetcher
from .quantization import (
    quantize_model_int8,
    quantize_model_nf4,
    quantize_model_nvfp4,
    apply_quantization,
    BITSANDBYTES_AVAILABLE,
    TORCHAO_AVAILABLE,
)

__all__ = [
    # Distributed
    "setup_distributed",
    "cleanup_distributed",
    "DISTRIBUTED_AVAILABLE",
    # Gradients
    "check_gradients",
    # Async prefetching
    "AsyncBatchPrefetcher",
    # Quantization
    "quantize_model_int8",
    "quantize_model_nf4",
    "quantize_model_nvfp4",
    "apply_quantization",
    "BITSANDBYTES_AVAILABLE",
    "TORCHAO_AVAILABLE",
]
