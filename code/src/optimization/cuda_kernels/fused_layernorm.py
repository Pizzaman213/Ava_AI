"""
Fused LayerNorm CUDA Kernels

Implements optimized layer normalization with optional affine transformations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def layer_norm_forward_kernel(
        # Inputs
        x_ptr, gamma_ptr, beta_ptr,
        # Outputs  
        y_ptr, mean_ptr, rstd_ptr,
        # Shape
        N, D,
        # Strides
        stride_xn, stride_xd,
        stride_yn, stride_yd,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Forward pass for layer normalization"""
        # Program ID
        row = tl.program_id(0)
        
        # Compute mean
        offs_d = tl.arange(0, BLOCK_SIZE)
        mask = offs_d < D
        
        # Load input row
        x_ptrs = x_ptr + row * stride_xn + offs_d * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute mean
        mean = tl.sum(x, axis=0) / D
        
        # Compute variance
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / D
        
        # Compute reciprocal standard deviation
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Normalize
        x_norm = x_centered * rstd
        
        # Apply affine transformation
        if gamma_ptr is not None:
            gamma = tl.load(gamma_ptr + offs_d, mask=mask, other=1.0)
            x_norm = x_norm * gamma
            
        if beta_ptr is not None:
            beta = tl.load(beta_ptr + offs_d, mask=mask, other=0.0)
            x_norm = x_norm + beta
        
        # Store outputs
        y_ptrs = y_ptr + row * stride_yn + offs_d * stride_yd
        tl.store(y_ptrs, x_norm, mask=mask)
        
        # Store statistics for backward pass
        if mean_ptr is not None:
            tl.store(mean_ptr + row, mean)
        if rstd_ptr is not None:
            tl.store(rstd_ptr + row, rstd)
    
    
    @triton.jit
    def layer_norm_backward_kernel(
        # Inputs
        dy_ptr, x_ptr, mean_ptr, rstd_ptr, gamma_ptr,
        # Outputs
        dx_ptr, dgamma_ptr, dbeta_ptr,
        # Shape
        N, D,
        # Strides
        stride_dyn, stride_dyd,
        stride_xn, stride_xd,
        stride_dxn, stride_dxd,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward pass for layer normalization"""
        row = tl.program_id(0)
        
        # Load data
        offs_d = tl.arange(0, BLOCK_SIZE)
        mask = offs_d < D
        
        dy_ptrs = dy_ptr + row * stride_dyn + offs_d * stride_dyd
        dy = tl.load(dy_ptrs, mask=mask, other=0.0)
        
        x_ptrs = x_ptr + row * stride_xn + offs_d * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        
        # Normalize x
        x_centered = x - mean
        x_norm = x_centered * rstd
        
        # Load gamma if provided
        if gamma_ptr is not None:
            gamma = tl.load(gamma_ptr + offs_d, mask=mask, other=1.0)
        else:
            gamma = 1.0
        
        # Compute gradients
        if dgamma_ptr is not None:
            dgamma = dy * x_norm
            tl.atomic_add(dgamma_ptr + offs_d, dgamma, mask=mask)
            
        if dbeta_ptr is not None:
            tl.atomic_add(dbeta_ptr + offs_d, dy, mask=mask)
        
        # Compute dx
        dy_gamma = dy * gamma
        mean_dy_gamma = tl.sum(dy_gamma, axis=0) / D
        mean_dy_gamma_x_centered = tl.sum(dy_gamma * x_centered, axis=0) / D
        
        dx = rstd * (dy_gamma - mean_dy_gamma - x_centered * mean_dy_gamma_x_centered * rstd * rstd)
        
        # Store dx
        dx_ptrs = dx_ptr + row * stride_dxn + offs_d * stride_dxd
        tl.store(dx_ptrs, dx, mask=mask)


    @triton.jit
    def rms_norm_forward_kernel(
        # Inputs
        x_ptr, gamma_ptr,
        # Outputs
        y_ptr, rstd_ptr,
        # Shape
        N, D,
        # Strides
        stride_xn, stride_xd,
        stride_yn, stride_yd,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Forward pass for RMS normalization"""
        row = tl.program_id(0)
        
        # Load input
        offs_d = tl.arange(0, BLOCK_SIZE)
        mask = offs_d < D
        
        x_ptrs = x_ptr + row * stride_xn + offs_d * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute RMS
        x_squared = x * x
        mean_squared = tl.sum(x_squared, axis=0) / D
        rms = tl.sqrt(mean_squared + eps)
        rstd = 1.0 / rms
        
        # Normalize
        x_norm = x * rstd
        
        # Apply gamma
        if gamma_ptr is not None:
            gamma = tl.load(gamma_ptr + offs_d, mask=mask, other=1.0)
            x_norm = x_norm * gamma
        
        # Store outputs
        y_ptrs = y_ptr + row * stride_yn + offs_d * stride_yd
        tl.store(y_ptrs, x_norm, mask=mask)
        
        if rstd_ptr is not None:
            tl.store(rstd_ptr + row, rstd)


class FusedLayerNormKernel(nn.Module):
    """
    Fused LayerNorm implementation with custom CUDA kernels
    
    Supports both standard LayerNorm and RMSNorm
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias and norm_type == "layernorm"
        self.norm_type = norm_type
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Check if Triton is available
        self.use_triton = TRITON_AVAILABLE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_triton and x.is_cuda:
            return self._forward_triton(x)
        else:
            return self._forward_pytorch(x)
    
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using Triton kernel"""
        orig_shape = x.shape
        x = x.view(-1, self.normalized_shape)
        N, D = x.shape
        
        # Configure kernel
        BLOCK_SIZE = triton.next_power_of_2(D)
        
        # Allocate outputs
        y = torch.empty_like(x)
        
        if self.norm_type == "layernorm":
            mean = torch.empty(N, dtype=x.dtype, device=x.device)
            rstd = torch.empty(N, dtype=x.dtype, device=x.device)
            
            # Launch kernel
            layer_norm_forward_kernel[(N,)](
                x, self.weight, self.bias,
                y, mean, rstd,
                N, D,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
                eps=self.eps,
            )
        else:  # RMSNorm
            rstd = torch.empty(N, dtype=x.dtype, device=x.device)
            
            # Launch kernel
            rms_norm_forward_kernel[(N,)](
                x, self.weight,
                y, rstd,
                N, D,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
                eps=self.eps,
            )
        
        return y.view(orig_shape)
    
    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback PyTorch implementation"""
        if self.norm_type == "layernorm":
            return F.layer_norm(
                x, (self.normalized_shape,), self.weight, self.bias, self.eps
            )
        else:  # RMSNorm
            # RMS normalization
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            x_norm = x / rms
            
            if self.weight is not None:
                x_norm = x_norm * self.weight
                
            return x_norm


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation
    
    More efficient than LayerNorm for large models
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.fused_kernel = FusedLayerNormKernel(
            hidden_size, eps=eps, bias=False, norm_type="rmsnorm"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.fused_kernel(x)


def get_norm_layer(
    norm_type: str,
    hidden_size: int,
    eps: float = 1e-5,
    **kwargs
) -> nn.Module:
    """Factory function to get normalization layer"""
    norm_types = {
        "layernorm": lambda: FusedLayerNormKernel(
            hidden_size, eps=eps, norm_type="layernorm"
        ),
        "rmsnorm": lambda: FusedRMSNorm(hidden_size, eps=eps),
        "nonorm": lambda: nn.Identity(),
    }
    
    if norm_type.lower() not in norm_types:
        raise ValueError(f"Unknown norm type: {norm_type}")
        
    return norm_types[norm_type.lower()]()