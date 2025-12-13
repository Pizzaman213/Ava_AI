"""
Triton kernels for GPU-optimized operations.

This package provides fused Triton kernels for:
- MoE gating and top-k operations
- Fused activation functions (SwiGLU, GeGLU)
"""

from .moe import (
    KernelConfig,
    set_kernel_config,
    get_kernel_config,
    TRITON_AVAILABLE,
)

__all__ = [
    'KernelConfig',
    'set_kernel_config',
    'get_kernel_config',
    'TRITON_AVAILABLE',
]
