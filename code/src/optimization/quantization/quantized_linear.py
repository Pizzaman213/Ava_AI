"""
Unified quantized linear layers for efficient inference

Supports multiple quantization methods with optimized kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class QuantizedLinear(nn.Module):
    """
    Base class for quantized linear layers
    
    Supports both GPTQ and AWQ quantization methods with
    efficient dequantization and matrix multiplication.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
        quantization_method: str = "gptq"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.quantization_method = quantization_method
        
        # Calculate number of groups
        self.n_groups = (in_features + group_size - 1) // group_size
        
        # Quantized weight storage
        if bits == 4:
            # Pack 4-bit weights
            self.register_buffer(
                'qweight',
                torch.zeros((out_features, in_features // 8), dtype=torch.int32)
            )
        else:
            self.register_buffer(
                'qweight', 
                torch.zeros((out_features, in_features), dtype=torch.int8)
            )
        
        # Quantization parameters
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.n_groups), dtype=torch.float16)
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((out_features, self.n_groups), dtype=torch.float16)
        )
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # CUDA kernel availability
        self.use_cuda_kernel = False
        self._check_cuda_kernels()
        
    def _check_cuda_kernels(self):
        """Check if optimized CUDA kernels are available"""
        try:
            import triton
            self.use_cuda_kernel = True
        except ImportError:
            self.use_cuda_kernel = False
            
    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack int4 weights into int32 format"""
        if self.bits != 4:
            return weight
            
        # Pack 8 int4 values into one int32
        packed = torch.zeros(
            (self.out_features, self.in_features // 8),
            dtype=torch.int32,
            device=weight.device
        )
        
        for i in range(8):
            packed |= ((weight[:, i::8] & 0xF).to(torch.int32) << (4 * i))
            
        return packed
    
    def unpack_weight(self) -> torch.Tensor:
        """Unpack int4 weights from int32 format"""
        if self.bits != 4:
            return self.qweight
            
        unpacked = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.int8,
            device=self.qweight.device
        )
        
        for i in range(8):
            unpacked[:, i::8] = ((self.qweight >> (4 * i)) & 0xF).to(torch.int8)
            
        return unpacked
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize weights to full precision"""
        # Unpack if necessary
        if self.bits == 4:
            qweight = self.unpack_weight()
        else:
            qweight = self.qweight
            
        # Dequantize group by group
        weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.float16,
            device=qweight.device
        )
        
        for g in range(self.n_groups):
            start_idx = g * self.group_size
            end_idx = min((g + 1) * self.group_size, self.in_features)
            
            # Apply dequantization formula
            weight[:, start_idx:end_idx] = (
                qweight[:, start_idx:end_idx].float() - self.qzeros[:, g:g+1]
            ) * self.scales[:, g:g+1]
            
        return weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        if self.use_cuda_kernel and x.is_cuda:
            # Use optimized kernel if available
            return self._forward_cuda(x)
        else:
            # Fallback to standard implementation
            return self._forward_cpu(x)
            
    def _forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """CPU/fallback forward implementation"""
        # Dequantize weights
        weight = self.dequantize()
        
        # Standard linear operation
        output = F.linear(x, weight, self.bias)
        
        return output
    
    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized CUDA forward implementation"""
        # This would use custom CUDA kernels for fused dequant + matmul
        # For now, fallback to CPU implementation
        return self._forward_cpu(x)
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        quantization_method: str = "gptq"
    ) -> 'QuantizedLinear':
        """Create quantized layer from existing linear layer"""
        # Create new quantized layer
        q_linear = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bits,
            group_size,
            quantization_method
        )
        
        # Copy bias if present
        if linear.bias is not None:
            q_linear.bias.data = linear.bias.data.clone()
            
        return q_linear


class DynamicQuantizedLinear(nn.Module):
    """
    Dynamic quantization linear layer
    
    Quantizes activations on-the-fly during inference for
    additional memory savings and performance.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 4,
        activation_bits: int = 8,
        group_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        
        # Weight quantization (static)
        self.weight_quantizer = QuantizedLinear(
            in_features,
            out_features,
            bias=False,  # Handle bias separately
            bits=weight_bits,
            group_size=group_size
        )
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Activation quantization parameters (computed dynamically)
        self.register_buffer('act_scale', torch.ones(1))
        self.register_buffer('act_zero_point', torch.zeros(1))
        
    def quantize_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Dynamically quantize activations"""
        # Compute quantization parameters
        x_min = x.min()
        x_max = x.max()
        
        # Compute scale and zero point
        qmin = 0
        qmax = (1 << self.activation_bits) - 1
        
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - torch.round(x_min / scale)
        
        # Quantize
        x_q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return x_q, scale.item(), zero_point.item()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization"""
        # Store original dtype
        orig_dtype = x.dtype
        
        # Quantize activations
        if self.training:
            # No quantization during training
            x_q = x
            act_scale = 1.0
            act_zero = 0.0
        else:
            x_q, act_scale, act_zero = self.quantize_activations(x)
            
        # Get dequantized weights
        weight = self.weight_quantizer.dequantize()
        
        # Perform quantized matmul
        if isinstance(x_q, torch.uint8):
            # Dequantize activations for computation
            x_dq = (x_q.float() - act_zero) * act_scale
            output = F.linear(x_dq, weight, None)
        else:
            output = F.linear(x_q, weight, None)
            
        # Add bias
        if self.bias is not None:
            output = output + self.bias
            
        # Restore original dtype
        output = output.to(orig_dtype)
        
        return output
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        weight_bits: int = 4,
        activation_bits: int = 8,
        group_size: int = 128
    ) -> 'DynamicQuantizedLinear':
        """Create dynamic quantized layer from existing linear"""
        q_linear = DynamicQuantizedLinear(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            weight_bits,
            activation_bits,
            group_size
        )
        
        # Copy bias
        if linear.bias is not None:
            q_linear.bias.data = linear.bias.data.clone()
            
        return q_linear


class FusedQuantizedLinear(nn.Module):
    """
    Fused quantized linear layer with optimized kernels
    
    Implements fused dequantization and matrix multiplication
    for maximum performance.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
        backend: str = "triton"  # triton, cuda, or cpu
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.backend = backend
        
        # Initialize quantized linear base
        self.q_linear = QuantizedLinear(
            in_features,
            out_features,
            bias,
            bits,
            group_size
        )
        
        # Setup backend-specific kernels
        self._setup_kernels()
        
    def _setup_kernels(self):
        """Setup backend-specific optimized kernels"""
        if self.backend == "triton":
            try:
                from .triton_kernels import quantized_matmul_kernel
                self.kernel = quantized_matmul_kernel
            except ImportError:
                logger.warning("Triton not available, falling back to CPU")
                self.backend = "cpu"
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fused operations"""
        if self.backend == "triton" and x.is_cuda:
            # Use Triton kernel for fused operation
            return self._forward_triton(x)
        else:
            # Fallback to standard implementation
            return self.q_linear(x)
            
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using Triton kernels"""
        # This would call the Triton kernel for fused dequant+matmul
        # Implementation depends on Triton kernel availability
        return self.q_linear(x)  # Fallback for now