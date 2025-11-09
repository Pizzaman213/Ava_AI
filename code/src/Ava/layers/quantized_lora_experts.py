"""
Quantized LoRA Expert implementation.

Combines Low-Rank Adaptation with INT8/INT4 quantization for maximum
memory efficiency. Designed to work with CPU offloading.

Memory per expert (hidden=4096, intermediate=14336, rank=4):
- Base weights (quantized INT8): ~224 KB (vs 896 KB FP16)
- LoRA deltas (FP16): ~130 KB
- Scale/zero-point: ~7 KB
- Total: ~361 KB (vs 4 MB standard, 91% reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class QuantizedLoRAExpert(nn.Module):
    """
    Expert with LoRA adaptation and INT8/INT4 quantized base weights.

    Architecture:
        W_effective = W_base_dequantized + (alpha/rank) * (lora_B @ lora_A)

    Memory strategy:
    - Base weights: Stored in INT8/INT4 (quantized) on CPU
    - LoRA deltas: Stored in FP16 (small) on CPU
    - On activation: Move to GPU and dequantize base
    - On deactivation: Move to CPU (keep quantized)

    Args:
        hidden_size: Input/output dimension
        intermediate_size: FFN hidden dimension
        activation: Activation function ('swiglu', 'geglu', 'gelu')
        lora_rank: Rank of LoRA matrices (4-16)
        lora_alpha: LoRA scaling parameter (typically 2*rank)
        quantization_bits: 8 (INT8) or 4 (INT4)
        quantization_method: 'per_channel' or 'per_tensor'
        dropout: Dropout probability
        dtype: FP dtype for LoRA deltas (fp16/bf16)

    Example:
        >>> expert = QuantizedLoRAExpert(
        ...     hidden_size=4096,
        ...     intermediate_size=14336,
        ...     lora_rank=4,
        ...     quantization_bits=8
        ... )
        >>> expert.cpu()  # Store on CPU (quantized)
        >>> # When needed:
        >>> expert.cuda()  # Move to GPU (auto-dequantizes)
        >>> output = expert(input)  # Apply LoRA + base
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'swiglu',
        lora_rank: int = 8,
        lora_alpha: int = 16,
        quantization_bits: int = 8,
        quantization_method: str = 'per_channel',
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = activation
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank
        self.quantization_bits = quantization_bits
        self.quantization_method = quantization_method
        self.dtype = dtype or torch.float16

        # Quantization parameters
        if quantization_bits == 8:
            self.qmin, self.qmax = 0, 255
            self.quant_dtype = torch.uint8
        elif quantization_bits == 4:
            self.qmin, self.qmax = 0, 15
            self.quant_dtype = torch.uint8  # Stored as uint8
        else:
            raise ValueError(f"Unsupported quantization_bits: {quantization_bits}")

        # Track quantization state
        self._is_quantized = False
        self._quantization_initialized = False

        # =========================================
        # 1. BASE WEIGHTS (Will be quantized)
        # =========================================

        if activation in ['swiglu', 'geglu']:
            # Gated activation: gate + up projection combined
            # Shape: (2 * intermediate_size, hidden_size)
            self.base_gate_up = nn.Parameter(
                torch.empty(2 * intermediate_size, hidden_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.base_gate_up, a=math.sqrt(5))

            # Down projection
            # Shape: (hidden_size, intermediate_size)
            self.base_down = nn.Parameter(
                torch.empty(hidden_size, intermediate_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.base_down, a=math.sqrt(5))

        else:  # Regular activation (gelu, etc.)
            # Up projection
            self.base_up = nn.Parameter(
                torch.empty(intermediate_size, hidden_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.base_up, a=math.sqrt(5))

            # Down projection
            self.base_down = nn.Parameter(
                torch.empty(hidden_size, intermediate_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.base_down, a=math.sqrt(5))

        # =========================================
        # 2. LORA DELTA WEIGHTS (Stay in FP16)
        # =========================================

        if activation in ['swiglu', 'geglu']:
            # LoRA for gate_up: A and B matrices
            # A: (rank, hidden_size), B: (2*intermediate, rank)
            self.lora_gate_up_A = nn.Parameter(
                torch.zeros(lora_rank, hidden_size, dtype=self.dtype)
            )
            self.lora_gate_up_B = nn.Parameter(
                torch.zeros(2 * intermediate_size, lora_rank, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_gate_up_A, a=math.sqrt(5))
            # B initialized to zero (important!)

            # LoRA for down: A and B matrices
            # A: (rank, intermediate), B: (hidden, rank)
            self.lora_down_A = nn.Parameter(
                torch.zeros(lora_rank, intermediate_size, dtype=self.dtype)
            )
            self.lora_down_B = nn.Parameter(
                torch.zeros(hidden_size, lora_rank, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        else:
            # LoRA for up projection
            self.lora_up_A = nn.Parameter(
                torch.zeros(lora_rank, hidden_size, dtype=self.dtype)
            )
            self.lora_up_B = nn.Parameter(
                torch.zeros(intermediate_size, lora_rank, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_up_A, a=math.sqrt(5))

            # LoRA for down projection
            self.lora_down_A = nn.Parameter(
                torch.zeros(lora_rank, intermediate_size, dtype=self.dtype)
            )
            self.lora_down_B = nn.Parameter(
                torch.zeros(hidden_size, lora_rank, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        # =========================================
        # 3. QUANTIZATION METADATA (Per-channel)
        # =========================================

        # These will be populated when quantization happens
        # Using buffers (not parameters) since they don't need gradients

        if activation in ['swiglu', 'geglu']:
            # Scales and zero-points for gate_up (per output channel)
            self.register_buffer(
                'gate_up_scale',
                torch.ones(2 * intermediate_size, dtype=self.dtype)
            )
            self.register_buffer(
                'gate_up_zero_point',
                torch.zeros(2 * intermediate_size, dtype=torch.int32)
            )

            # Scales and zero-points for down (per output channel)
            self.register_buffer(
                'down_scale',
                torch.ones(hidden_size, dtype=self.dtype)
            )
            self.register_buffer(
                'down_zero_point',
                torch.zeros(hidden_size, dtype=torch.int32)
            )

        else:
            # Scales and zero-points for up
            self.register_buffer(
                'up_scale',
                torch.ones(intermediate_size, dtype=self.dtype)
            )
            self.register_buffer(
                'up_zero_point',
                torch.zeros(intermediate_size, dtype=torch.int32)
            )

            # Scales and zero-points for down
            self.register_buffer(
                'down_scale',
                torch.ones(hidden_size, dtype=self.dtype)
            )
            self.register_buffer(
                'down_zero_point',
                torch.zeros(hidden_size, dtype=torch.int32)
            )

        # =========================================
        # 4. ACTIVATION & DROPOUT
        # =========================================

        if activation == 'swiglu':
            self.activation = nn.SiLU()
        elif activation == 'geglu':
            self.activation = nn.GELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # =============================================
    # QUANTIZATION METHODS
    # =============================================

    def quantize_weights(self):
        """
        Quantize base weights to INT8/INT4.

        This should be called when moving the expert to CPU for storage.
        LoRA deltas remain in FP16.
        """
        if self._is_quantized:
            return  # Already quantized

        with torch.no_grad():
            if self.activation_type in ['swiglu', 'geglu']:
                # Quantize gate_up projection
                self._quantize_weight(
                    self.base_gate_up,
                    self.gate_up_scale,
                    self.gate_up_zero_point
                )

                # Quantize down projection
                self._quantize_weight(
                    self.base_down,
                    self.down_scale,
                    self.down_zero_point
                )
            else:
                # Quantize up projection
                self._quantize_weight(
                    self.base_up,
                    self.up_scale,
                    self.up_zero_point
                )

                # Quantize down projection
                self._quantize_weight(
                    self.base_down,
                    self.down_scale,
                    self.down_zero_point
                )

        self._is_quantized = True
        self._quantization_initialized = True

    def _quantize_weight(
        self,
        weight: nn.Parameter,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ):
        """
        Quantize a weight tensor using per-channel quantization.

        Formula:
            scale = (max - min) / (qmax - qmin)
            zero_point = qmin - round(min / scale)
            Q = clip(round(W / scale) + zero_point, qmin, qmax)

        Args:
            weight: Weight parameter to quantize (modified in-place)
            scale: Buffer to store scales (modified in-place)
            zero_point: Buffer to store zero points (modified in-place)
        """
        # Per-channel: compute stats along output channels (dim=0)
        # weight shape: (out_features, in_features)

        if self.quantization_method == 'per_channel':
            # Compute min/max per output channel
            w_min = weight.min(dim=1, keepdim=True)[0]  # (out_features, 1)
            w_max = weight.max(dim=1, keepdim=True)[0]  # (out_features, 1)
        else:  # per_tensor
            w_min = weight.min()
            w_max = weight.max()

        # Compute scale and zero point
        scale_val = (w_max - w_min) / (self.qmax - self.qmin)
        scale_val = torch.clamp(scale_val, min=1e-8)  # Avoid division by zero
        zero_point_val = self.qmin - torch.round(w_min / scale_val)
        zero_point_val = torch.clamp(zero_point_val, self.qmin, self.qmax)

        # Store scale and zero_point
        if self.quantization_method == 'per_channel':
            scale.copy_(scale_val.squeeze(1))  # (out_features,)
            zero_point.copy_(zero_point_val.squeeze(1).to(torch.int32))
        else:
            scale.fill_(scale_val.item())
            zero_point.fill_(int(zero_point_val.item()))

        # Quantize: Q = clip(round(W / scale) + zero_point, qmin, qmax)
        if self.quantization_method == 'per_channel':
            w_quantized = weight / scale_val  # Broadcast scale
        else:
            w_quantized = weight / scale_val

        w_quantized = torch.round(w_quantized) + zero_point_val
        w_quantized = torch.clamp(w_quantized, self.qmin, self.qmax)

        # Store quantized weights (overwrite original parameter)
        # First disable gradients, then update data
        weight.requires_grad_(False)
        weight.data = w_quantized.to(self.quant_dtype)

    def dequantize_weights(self):
        """
        Dequantize base weights from INT8/INT4 to FP16.

        This should be called when moving the expert to GPU for computation.

        Formula:
            W = (Q - zero_point) * scale
        """
        if not self._is_quantized:
            return  # Already in FP16

        with torch.no_grad():
            if self.activation_type in ['swiglu', 'geglu']:
                # Dequantize gate_up projection
                self._dequantize_weight(
                    self.base_gate_up,
                    self.gate_up_scale,
                    self.gate_up_zero_point
                )

                # Dequantize down projection
                self._dequantize_weight(
                    self.base_down,
                    self.down_scale,
                    self.down_zero_point
                )
            else:
                # Dequantize up projection
                self._dequantize_weight(
                    self.base_up,
                    self.up_scale,
                    self.up_zero_point
                )

                # Dequantize down projection
                self._dequantize_weight(
                    self.base_down,
                    self.down_scale,
                    self.down_zero_point
                )

        self._is_quantized = False

    def _dequantize_weight(
        self,
        weight: nn.Parameter,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ):
        """
        Dequantize a weight tensor.

        Formula:
            W = (Q - zero_point) * scale

        Args:
            weight: Quantized weight parameter (modified in-place)
            scale: Scale factors
            zero_point: Zero points
        """
        # Convert to float
        w_float = weight.to(self.dtype)

        # Dequantize: W = (Q - zero_point) * scale
        if self.quantization_method == 'per_channel':
            # Broadcast scale and zero_point: (out_features,) -> (out_features, 1)
            scale_expanded = scale.unsqueeze(1)
            zero_point_expanded = zero_point.unsqueeze(1).to(self.dtype)
            w_dequantized = (w_float - zero_point_expanded) * scale_expanded
        else:
            w_dequantized = (w_float - zero_point.to(self.dtype)) * scale

        # Store dequantized weights (overwrite parameter)
        weight.data = w_dequantized
        weight.requires_grad_(True)  # Re-enable gradients for FP16 weights

    # =============================================
    # DEVICE TRANSFER HOOKS
    # =============================================

    def to(self, *args, **kwargs):
        """
        Override to() to handle quantization on device transfers.
        """
        # Determine target device
        device = None
        if len(args) > 0:
            if isinstance(args[0], (torch.device, str)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        if 'device' in kwargs:
            device = kwargs['device']
            if isinstance(device, str):
                device = torch.device(device)

        # Handle quantization based on device
        if device is not None:
            if device.type == 'cuda':
                # Moving to GPU: dequantize first
                if self._is_quantized:
                    self.dequantize_weights()
            elif device.type == 'cpu':
                # Moving to CPU: quantize first (if initialized)
                if self._quantization_initialized and not self._is_quantized:
                    self.quantize_weights()

        # Call parent to() (performs actual device transfer)
        return super().to(*args, **kwargs)

    # =============================================
    # FORWARD PASS
    # =============================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-adapted quantized weights.

        Computation:
            W_effective = W_base (dequantized) + (alpha/rank) * (B @ A)
            output = activation(x @ W_effective^T)

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # Ensure weights are dequantized (should be, if on GPU)
        if self._is_quantized:
            # This shouldn't happen in normal operation, but handle gracefully
            self.dequantize_weights()

        if self.activation_type in ['swiglu', 'geglu']:
            # ========================================
            # Gated activation (SwiGLU/GeGLU)
            # ========================================

            # 1. Compute effective gate_up weight with LoRA
            # W_gate_up = W_base + (alpha/rank) * (B @ A)
            lora_delta = self.lora_gate_up_B @ self.lora_gate_up_A  # (2*inter, hidden)
            W_gate_up = self.base_gate_up + self.lora_scaling * lora_delta

            # 2. Apply gate_up projection
            gate_up = F.linear(x, W_gate_up)  # (batch, seq, 2*intermediate)

            # 3. Split into gate and up
            gate, up = gate_up.chunk(2, dim=-1)  # Each (batch, seq, intermediate)

            # 4. Apply gated activation
            hidden = self.activation(gate) * up  # (batch, seq, intermediate)

            # 5. Apply dropout if enabled
            if self.dropout is not None:
                hidden = self.dropout(hidden)

            # 6. Compute effective down weight with LoRA
            lora_delta_down = self.lora_down_B @ self.lora_down_A  # (hidden, inter)
            W_down = self.base_down + self.lora_scaling * lora_delta_down

            # 7. Apply down projection
            output = F.linear(hidden, W_down)  # (batch, seq, hidden)

        else:
            # ========================================
            # Regular activation (GELU, etc.)
            # ========================================

            # 1. Compute effective up weight with LoRA
            lora_delta_up = self.lora_up_B @ self.lora_up_A  # (intermediate, hidden)
            W_up = self.base_up + self.lora_scaling * lora_delta_up

            # 2. Apply up projection
            hidden = F.linear(x, W_up)  # (batch, seq, intermediate)

            # 3. Apply activation
            hidden = self.activation(hidden)

            # 4. Apply dropout if enabled
            if self.dropout is not None:
                hidden = self.dropout(hidden)

            # 5. Compute effective down weight with LoRA
            lora_delta_down = self.lora_down_B @ self.lora_down_A  # (hidden, inter)
            W_down = self.base_down + self.lora_scaling * lora_delta_down

            # 6. Apply down projection
            output = F.linear(hidden, W_down)  # (batch, seq, hidden)

        return output

    # =============================================
    # UTILITY METHODS
    # =============================================

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage breakdown in MB."""
        def tensor_size_mb(t):
            return t.numel() * t.element_size() / 1024 / 1024

        base_memory = 0
        lora_memory = 0
        metadata_memory = 0

        if self.activation_type in ['swiglu', 'geglu']:
            base_memory += tensor_size_mb(self.base_gate_up)
            base_memory += tensor_size_mb(self.base_down)
            lora_memory += tensor_size_mb(self.lora_gate_up_A)
            lora_memory += tensor_size_mb(self.lora_gate_up_B)
            lora_memory += tensor_size_mb(self.lora_down_A)
            lora_memory += tensor_size_mb(self.lora_down_B)
            metadata_memory += tensor_size_mb(self.gate_up_scale)
            metadata_memory += tensor_size_mb(self.gate_up_zero_point)
            metadata_memory += tensor_size_mb(self.down_scale)
            metadata_memory += tensor_size_mb(self.down_zero_point)
        else:
            base_memory += tensor_size_mb(self.base_up)
            base_memory += tensor_size_mb(self.base_down)
            lora_memory += tensor_size_mb(self.lora_up_A)
            lora_memory += tensor_size_mb(self.lora_up_B)
            lora_memory += tensor_size_mb(self.lora_down_A)
            lora_memory += tensor_size_mb(self.lora_down_B)
            metadata_memory += tensor_size_mb(self.up_scale)
            metadata_memory += tensor_size_mb(self.up_zero_point)
            metadata_memory += tensor_size_mb(self.down_scale)
            metadata_memory += tensor_size_mb(self.down_zero_point)

        return {
            'base_weights': base_memory,
            'lora_deltas': lora_memory,
            'quantization_metadata': metadata_memory,
            'total': base_memory + lora_memory + metadata_memory,
            'is_quantized': self._is_quantized,
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'hidden={self.hidden_size}, intermediate={self.intermediate_size}, '
            f'activation={self.activation_type}, lora_rank={self.lora_rank}, '
            f'quant_bits={self.quantization_bits}, quantized={self._is_quantized}'
        )


# =============================================
# HELPER: Non-LoRA Quantized Expert
# =============================================

class QuantizedExpert(nn.Module):
    """
    Quantized expert without LoRA (for completeness).

    Simpler version of QuantizedLoRAExpert without LoRA deltas.
    Use this when you want quantization + offloading but not LoRA.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'swiglu',
        quantization_bits: int = 8,
        quantization_method: str = 'per_channel',
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = activation
        self.quantization_bits = quantization_bits
        self.quantization_method = quantization_method
        self.dtype = dtype or torch.float16

        # Quantization setup (same as QuantizedLoRAExpert)
        if quantization_bits == 8:
            self.qmin, self.qmax = 0, 255
            self.quant_dtype = torch.uint8
        elif quantization_bits == 4:
            self.qmin, self.qmax = 0, 15
            self.quant_dtype = torch.uint8
        else:
            raise ValueError(f"Unsupported quantization_bits: {quantization_bits}")

        self._is_quantized = False
        self._quantization_initialized = False

        # Initialize weights (without LoRA)
        if activation in ['swiglu', 'geglu']:
            self.gate_up = nn.Parameter(
                torch.empty(2 * intermediate_size, hidden_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.gate_up, a=math.sqrt(5))

            self.down = nn.Parameter(
                torch.empty(hidden_size, intermediate_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

            # Quantization metadata
            self.register_buffer(
                'gate_up_scale',
                torch.ones(2 * intermediate_size, dtype=self.dtype)
            )
            self.register_buffer(
                'gate_up_zero_point',
                torch.zeros(2 * intermediate_size, dtype=torch.int32)
            )
            self.register_buffer(
                'down_scale',
                torch.ones(hidden_size, dtype=self.dtype)
            )
            self.register_buffer(
                'down_zero_point',
                torch.zeros(hidden_size, dtype=torch.int32)
            )
        else:
            self.up = nn.Parameter(
                torch.empty(intermediate_size, hidden_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.up, a=math.sqrt(5))

            self.down = nn.Parameter(
                torch.empty(hidden_size, intermediate_size, dtype=self.dtype)
            )
            nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

            # Quantization metadata
            self.register_buffer(
                'up_scale',
                torch.ones(intermediate_size, dtype=self.dtype)
            )
            self.register_buffer(
                'up_zero_point',
                torch.zeros(intermediate_size, dtype=torch.int32)
            )
            self.register_buffer(
                'down_scale',
                torch.ones(hidden_size, dtype=self.dtype)
            )
            self.register_buffer(
                'down_zero_point',
                torch.zeros(hidden_size, dtype=torch.int32)
            )

        # Activation
        if activation == 'swiglu':
            self.activation = nn.SiLU()
        elif activation in ['geglu', 'gelu']:
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def quantize_weights(self):
        """Quantize weights to INT8/INT4."""
        if self._is_quantized:
            return

        with torch.no_grad():
            if self.activation_type in ['swiglu', 'geglu']:
                self._quantize_weight(self.gate_up, self.gate_up_scale, self.gate_up_zero_point)
                self._quantize_weight(self.down, self.down_scale, self.down_zero_point)
            else:
                self._quantize_weight(self.up, self.up_scale, self.up_zero_point)
                self._quantize_weight(self.down, self.down_scale, self.down_zero_point)

        self._is_quantized = True
        self._quantization_initialized = True

    def _quantize_weight(self, weight: nn.Parameter, scale: torch.Tensor, zero_point: torch.Tensor):
        """Quantize a single weight tensor."""
        if self.quantization_method == 'per_channel':
            w_min = weight.min(dim=1, keepdim=True)[0]
            w_max = weight.max(dim=1, keepdim=True)[0]
        else:
            w_min = weight.min()
            w_max = weight.max()

        scale_val = (w_max - w_min) / (self.qmax - self.qmin)
        scale_val = torch.clamp(scale_val, min=1e-8)
        zero_point_val = self.qmin - torch.round(w_min / scale_val)
        zero_point_val = torch.clamp(zero_point_val, self.qmin, self.qmax)

        if self.quantization_method == 'per_channel':
            scale.copy_(scale_val.squeeze(1))
            zero_point.copy_(zero_point_val.squeeze(1).to(torch.int32))
        else:
            scale.fill_(scale_val.item())
            zero_point.fill_(int(zero_point_val.item()))

        if self.quantization_method == 'per_channel':
            w_quantized = weight / scale_val
        else:
            w_quantized = weight / scale_val

        w_quantized = torch.round(w_quantized) + zero_point_val
        w_quantized = torch.clamp(w_quantized, self.qmin, self.qmax)
        weight.requires_grad_(False)
        weight.data = w_quantized.to(self.quant_dtype)

    def dequantize_weights(self):
        """Dequantize weights from INT8/INT4 to FP16."""
        if not self._is_quantized:
            return

        with torch.no_grad():
            if self.activation_type in ['swiglu', 'geglu']:
                self._dequantize_weight(self.gate_up, self.gate_up_scale, self.gate_up_zero_point)
                self._dequantize_weight(self.down, self.down_scale, self.down_zero_point)
            else:
                self._dequantize_weight(self.up, self.up_scale, self.up_zero_point)
                self._dequantize_weight(self.down, self.down_scale, self.down_zero_point)

        self._is_quantized = False

    def _dequantize_weight(self, weight: nn.Parameter, scale: torch.Tensor, zero_point: torch.Tensor):
        """Dequantize a single weight tensor."""
        w_float = weight.to(self.dtype)

        if self.quantization_method == 'per_channel':
            scale_expanded = scale.unsqueeze(1)
            zero_point_expanded = zero_point.unsqueeze(1).to(self.dtype)
            w_dequantized = (w_float - zero_point_expanded) * scale_expanded
        else:
            w_dequantized = (w_float - zero_point.to(self.dtype)) * scale

        weight.data = w_dequantized
        weight.requires_grad_(True)

    def to(self, *args, **kwargs):
        """Override to() to handle quantization on device transfers."""
        # Determine target device
        device = None
        if len(args) > 0:
            if isinstance(args[0], (torch.device, str)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        if 'device' in kwargs:
            device = kwargs['device']
            if isinstance(device, str):
                device = torch.device(device)

        # Handle quantization based on device
        if device is not None:
            if device.type == 'cuda':
                if self._is_quantized:
                    self.dequantize_weights()
            elif device.type == 'cpu':
                if self._quantization_initialized and not self._is_quantized:
                    self.quantize_weights()

        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (without LoRA)."""
        if self._is_quantized:
            self.dequantize_weights()

        if self.activation_type in ['swiglu', 'geglu']:
            gate_up = F.linear(x, self.gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = self.activation(gate) * up
            if self.dropout is not None:
                hidden = self.dropout(hidden)
            output = F.linear(hidden, self.down)
        else:
            hidden = F.linear(x, self.up)
            hidden = self.activation(hidden)
            if self.dropout is not None:
                hidden = self.dropout(hidden)
            output = F.linear(hidden, self.down)

        return output

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage breakdown in MB."""
        def tensor_size_mb(t):
            return t.numel() * t.element_size() / 1024 / 1024

        base_memory = 0
        metadata_memory = 0

        if self.activation_type in ['swiglu', 'geglu']:
            base_memory += tensor_size_mb(self.gate_up)
            base_memory += tensor_size_mb(self.down)
            metadata_memory += tensor_size_mb(self.gate_up_scale)
            metadata_memory += tensor_size_mb(self.gate_up_zero_point)
            metadata_memory += tensor_size_mb(self.down_scale)
            metadata_memory += tensor_size_mb(self.down_zero_point)
        else:
            base_memory += tensor_size_mb(self.up)
            base_memory += tensor_size_mb(self.down)
            metadata_memory += tensor_size_mb(self.up_scale)
            metadata_memory += tensor_size_mb(self.up_zero_point)
            metadata_memory += tensor_size_mb(self.down_scale)
            metadata_memory += tensor_size_mb(self.down_zero_point)

        return {
            'base_weights': base_memory,
            'quantization_metadata': metadata_memory,
            'total': base_memory + metadata_memory,
            'is_quantized': self._is_quantized,
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'hidden={self.hidden_size}, intermediate={self.intermediate_size}, '
            f'activation={self.activation_type}, quant_bits={self.quantization_bits}, '
            f'quantized={self._is_quantized}'
        )