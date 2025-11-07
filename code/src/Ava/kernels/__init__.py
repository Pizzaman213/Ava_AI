"""
Custom CUDA/Triton kernels for high-performance MoE operations.
"""

from .moe_kernels import (
    fused_gating_topk,
    expert_scatter_gather,
    load_balancing_loss_kernel,
    TRITON_AVAILABLE,
)

__all__ = [
    'fused_gating_topk',
    'expert_scatter_gather',
    'load_balancing_loss_kernel',
    'TRITON_AVAILABLE',
]
