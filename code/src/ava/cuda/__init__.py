"""
CUDA utilities for GPU operations.

This package provides:
- StreamPool: CUDA stream management
- CUDATimer: Event-based timing
- Pinned buffer management
- Async metrics tracking
- Nsight profiling integration
- Triton kernels for MoE, activations, and expert dispatch
"""

# Stream management
from .streams import StreamPool, CUDATimer, get_buffer_pool, PinnedBufferPool

# Triton kernels (moved from kernels/)
try:
    from .moe_kernels import (
        fused_gating_topk,
        fused_softmax_topk_renorm,
        TRITON_AVAILABLE,
    )
except ImportError:
    fused_gating_topk = None
    fused_softmax_topk_renorm = None
    TRITON_AVAILABLE = False

try:
    from .kernel_activations import fused_swiglu, fused_geglu
except ImportError:
    fused_swiglu = None
    fused_geglu = None

try:
    from .fused_experts import fused_expert_forward
except ImportError:
    fused_expert_forward = None

try:
    from .fused_norm import (
        fused_add_layer_norm,
        fused_add_layer_norm_simple,
        FusedAddLayerNorm,
    )
except ImportError:
    fused_add_layer_norm = None
    fused_add_layer_norm_simple = None
    FusedAddLayerNorm = None

__all__ = [
    # Stream management
    'StreamPool',
    'CUDATimer',
    'get_buffer_pool',
    'PinnedBufferPool',
    # Triton kernels
    'fused_gating_topk',
    'fused_softmax_topk_renorm',
    'fused_swiglu',
    'fused_geglu',
    'fused_expert_forward',
    # Fused norm
    'fused_add_layer_norm',
    'fused_add_layer_norm_simple',
    'FusedAddLayerNorm',
    'TRITON_AVAILABLE',
]
