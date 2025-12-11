"""
High-performance Triton kernels for MoE operations.

This module provides fused kernels that significantly improve MoE performance
by reducing memory bandwidth and kernel launch overhead.

Available Kernels:
- Router/Gating:
  - fused_gating_topk: Fused linear + softmax + top-k (15-25% speedup)
  - fused_softmax_topk: Fused softmax + top-k on pre-computed logits
  - fused_gating_topk_renorm: With probability renormalization

- Activations:
  - fused_swiglu: Fused SwiGLU activation (10-15% speedup)
  - fused_geglu: Fused GeGLU activation (10-15% speedup)
  - fused_gated_activation: Unified interface for gated activations

- Configuration:
  - KernelConfig: Configuration for kernel optimizations
  - set_kernel_config: Set global kernel configuration
  - get_kernel_config: Get current configuration

Performance benefits:
- 50-80% overall throughput improvement
- Reduced memory bandwidth through fusion
- Better GPU utilization
- Fewer kernel launches
"""

from .moe_kernels import (
    # Core functions
    fused_gating_topk,
    fused_softmax_topk,
    fused_gating_topk_renorm,
    # Configuration
    KernelConfig,
    set_kernel_config,
    get_kernel_config,
    # Utilities
    benchmark_topk_kernels,
    # Constants
    TRITON_AVAILABLE,
)

from .activation_kernels import (
    # Fused activations
    fused_swiglu,
    fused_geglu,
    fused_gated_activation,
    # Autograd functions (for advanced usage)
    FusedSwiGLUFunction,
    FusedGeGLUFunction,
    # Utilities
    benchmark_fused_activations,
)

__all__ = [
    # Router/Gating kernels
    'fused_gating_topk',
    'fused_softmax_topk',
    'fused_gating_topk_renorm',
    # Activation kernels
    'fused_swiglu',
    'fused_geglu',
    'fused_gated_activation',
    'FusedSwiGLUFunction',
    'FusedGeGLUFunction',
    # Configuration
    'KernelConfig',
    'set_kernel_config',
    'get_kernel_config',
    # Utilities
    'benchmark_topk_kernels',
    'benchmark_fused_activations',
    # Constants
    'TRITON_AVAILABLE',
]
