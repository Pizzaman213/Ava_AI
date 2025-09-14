"""
Fused MLP CUDA Kernels

Implements optimized MLP operations with fused activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def fused_mlp_forward_kernel(
        # Inputs
        x_ptr, w1_ptr, w2_ptr, b1_ptr, b2_ptr,
        # Output
        out_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_w1k, stride_w1n,
        stride_w2n, stride_w2k,
        stride_outm, stride_outk,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """
        Fused MLP forward kernel: out = W2 @ activation(W1 @ x + b1) + b2
        
        Supports multiple activation functions:
        - ACTIVATION = 0: ReLU
        - ACTIVATION = 1: GELU
        - ACTIVATION = 2: SiLU/Swish
        - ACTIVATION = 3: GeGLU
        """
        # Program ID
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        pid_m = pid % num_pid_m
        
        # Compute offsets
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Initialize accumulator for first linear layer
        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # First linear layer: W1 @ x
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load input block
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk
            x_mask = (offs_m[:, None] < M) & ((k * BLOCK_SIZE_K + offs_k[None, :]) < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            
            # Load weight block
            w1_ptrs = w1_ptr + (k * BLOCK_SIZE_K + offs_k[:, None]) * stride_w1k + offs_n[None, :] * stride_w1n
            w1_mask = ((k * BLOCK_SIZE_K + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
            
            # Accumulate
            acc1 += tl.dot(x, w1, allow_tf32=True)
        
        # Add bias if provided
        if b1_ptr is not None:
            b1 = tl.load(b1_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc1 += b1[None, :]
        
        # Apply activation function
        if ACTIVATION == 0:  # ReLU
            acc1 = tl.maximum(acc1, 0.0)
        elif ACTIVATION == 1:  # GELU
            acc1 = acc1 * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (acc1 + 0.044715 * acc1 * acc1 * acc1)))
        elif ACTIVATION == 2:  # SiLU
            acc1 = acc1 * tl.sigmoid(acc1)
        elif ACTIVATION == 3:  # GeGLU
            # Split activation for gated units
            half_n = N // 2
            gate = acc1[:, :half_n]
            value = acc1[:, half_n:]
            gate = gate * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)))
            acc1 = gate * value
        
        # Second linear layer: W2 @ activation(...)
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        
        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            # Get activated values
            if ACTIVATION == 3:  # GeGLU uses half the hidden dimension
                act_n = n * BLOCK_SIZE_N // 2
                act = acc1[:, act_n:act_n + BLOCK_SIZE_N // 2]
            else:
                act = acc1[:, n * BLOCK_SIZE_N:(n + 1) * BLOCK_SIZE_N]
            
            # Load second weight block
            w2_ptrs = w2_ptr + (n * BLOCK_SIZE_N + offs_n[:, None]) * stride_w2n + offs_k[None, :] * stride_w2k
            w2_mask = ((n * BLOCK_SIZE_N + offs_n[:, None]) < N) & (offs_k[None, :] < K)
            w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)
            
            # Accumulate
            acc2 += tl.dot(act, w2, allow_tf32=True)
        
        # Add second bias if provided
        if b2_ptr is not None:
            b2 = tl.load(b2_ptr + offs_k, mask=offs_k < K, other=0.0)
            acc2 += b2[None, :]
        
        # Store output
        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_k[None, :] * stride_outk
        out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        tl.store(out_ptrs, acc2.to(tl.float16), mask=out_mask)


class FusedMLPKernel(nn.Module):
    """
    Fused MLP implementation with custom CUDA kernels
    
    Performs: out = W2(activation(W1(x))) in a single kernel launch
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = bias
        self.dropout = dropout
        
        # Weights
        self.w1 = nn.Parameter(torch.empty(in_features, hidden_features))
        self.w2 = nn.Parameter(torch.empty(hidden_features, out_features))
        
        # Biases
        if bias:
            self.b1 = nn.Parameter(torch.zeros(hidden_features))
            self.b2 = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Initialize weights
        self.reset_parameters()
        
        # Check if Triton is available
        self.use_triton = TRITON_AVAILABLE
        
        # Activation function mapping
        self.activation_map = {
            'relu': 0,
            'gelu': 1,
            'silu': 2,
            'swish': 2,
            'geglu': 3,
        }
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        if self.use_bias:
            nn.init.zeros_(self.b1)
            nn.init.zeros_(self.b2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused operations"""
        if self.use_triton and x.is_cuda and not self.training:
            # Use fused kernel for inference
            return self._forward_triton(x)
        else:
            # Fallback to PyTorch implementation
            return self._forward_pytorch(x)
    
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using Triton kernel"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.in_features)
        M, K = x.shape
        N = self.hidden_features
        
        # Configure kernel
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        
        # Get activation ID
        activation_id = self.activation_map.get(self.activation.lower(), 1)
        
        # Allocate output
        out = torch.empty((M, self.out_features), dtype=x.dtype, device=x.device)
        
        # Launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
        
        fused_mlp_forward_kernel[grid](
            x, self.w1, self.w2,
            self.b1 if self.use_bias else None,
            self.b2 if self.use_bias else None,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.w1.stride(0), self.w1.stride(1),
            self.w2.stride(0), self.w2.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            ACTIVATION=activation_id,
        )
        
        return out.reshape(batch_size, seq_len, self.out_features)
    
    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback PyTorch implementation"""
        # First linear layer
        x = F.linear(x, self.w1.t(), self.b1)
        
        # Activation
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation in ['silu', 'swish']:
            x = F.silu(x)
        elif self.activation == 'geglu':
            x, gate = x.chunk(2, dim=-1)
            x = x * F.gelu(gate)
        
        # Dropout
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        
        # Second linear layer
        x = F.linear(x, self.w2.t(), self.b2)
        
        return x


class FusedGatedMLP(nn.Module):
    """
    Fused Gated MLP for models like LLaMA
    
    Implements: out = W2(silu(W1(x)) * W3(x))
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        activation: str = "silu",
        bias: bool = False
    ):
        super().__init__()
        out_features = out_features or in_features
        
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)
        
        self.activation = activation
        self.act_fn = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'swish': F.silu,
        }
        return activations.get(activation.lower(), F.silu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Could be optimized with custom kernel
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))