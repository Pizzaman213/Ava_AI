"""
Mixture of Activations (MoA) implementation for dynamic activation function selection.

This module implements adaptive activation function selection where different
tokens or features can use different activation functions based on their content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import math


class MixtureOfActivations(nn.Module):
    """
    Mixture of Activations layer that dynamically selects activation functions.

    This layer routes different inputs to different activation functions
    based on learned routing weights, allowing for more expressive models.
    """

    def __init__(
        self,
        input_dim: int,
        activation_functions: Optional[List[str]] = None,
        num_experts: int = 4,
        router_hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_gating = use_gating

        # Default activation functions
        if activation_functions is None:
            activation_functions = ['relu', 'gelu', 'swish', 'mish']

        self.activation_names = activation_functions[:num_experts]

        # Build activation function modules
        self.activations = nn.ModuleDict()
        for i, activation_name in enumerate(self.activation_names):
            self.activations[f'expert_{i}'] = self._build_activation(activation_name)

        # Router network for activation selection
        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts)
        )

        # Optional gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, router_hidden_dim),
                nn.ReLU(),
                nn.Linear(router_hidden_dim, num_experts),
                nn.Sigmoid()
            )

        self.dropout = nn.Dropout(dropout)

        # Load balancing parameters
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def _build_activation(self, activation_name: str) -> nn.Module:
        """Build activation function module."""
        if activation_name.lower() == 'relu':
            return nn.ReLU()
        elif activation_name.lower() == 'gelu':
            return nn.GELU()
        elif activation_name.lower() == 'swish' or activation_name.lower() == 'silu':
            return nn.SiLU()
        elif activation_name.lower() == 'mish':
            return Mish()
        elif activation_name.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name.lower() == 'elu':
            return nn.ELU()
        elif activation_name.lower() == 'prelu':
            return nn.PReLU()
        elif activation_name.lower() == 'tanh':
            return nn.Tanh()
        elif activation_name.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name.lower() == 'softplus':
            return nn.Softplus()
        else:
            # Default to ReLU
            return nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through mixture of activations.

        Args:
            x: Input tensor [batch_size, ..., input_dim]
            return_routing_info: Whether to return routing information

        Returns:
            Tuple of (output, routing_info)
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.input_dim)
        batch_size = x_flat.shape[0]

        # Compute routing weights
        routing_logits = self.router(x_flat)
        routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)

        # Optional gating
        if self.use_gating:
            gates = self.gate(x_flat)
            routing_weights = routing_weights * gates

        # Apply activations
        expert_outputs = []
        for i, (expert_name, activation_fn) in enumerate(self.activations.items()):
            expert_output = activation_fn(x_flat)
            expert_outputs.append(expert_output)

        # Stack expert outputs
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, input_dim]

        # Weighted combination
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # [batch, num_experts, 1]
        mixed_output = (stacked_outputs * routing_weights_expanded).sum(dim=1)

        # Apply dropout
        mixed_output = self.dropout(mixed_output)

        # Reshape back to original shape
        output = mixed_output.view(original_shape)

        # Update expert usage statistics (for load balancing)
        if self.training:
            expert_usage = routing_weights.sum(dim=0)
            self.expert_counts += expert_usage.detach()
            self.total_tokens += batch_size

        routing_info = None
        if return_routing_info:
            routing_info = {
                'routing_weights': routing_weights.view(*original_shape[:-1], self.num_experts),
                'routing_logits': routing_logits.view(*original_shape[:-1], self.num_experts),
                'expert_utilization': self.expert_counts / (self.total_tokens + 1e-8) if self.training else None,
                'selected_activations': self.activation_names
            }

        return output, routing_info

    def get_load_balancing_loss(self) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        if self.total_tokens == 0:
            return torch.tensor(0.0, device=self.expert_counts.device)

        # Compute variance in expert usage
        avg_usage = self.expert_counts / self.total_tokens
        target_usage = 1.0 / self.num_experts
        variance = ((avg_usage - target_usage) ** 2).sum()
        return variance


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation that learns parameters for parametric activations.

    This module can adapt activation functions based on input characteristics.
    """

    def __init__(
        self,
        input_dim: int,
        activation_type: str = 'learnable_relu',
        num_parameters: int = 2,
        init_values: Optional[List[float]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.activation_type = activation_type.lower()
        self.num_parameters = num_parameters

        # Initialize learnable parameters
        if init_values is None:
            if activation_type == 'learnable_relu':
                init_values = [0.01, 1.0]  # negative_slope, positive_slope
            elif activation_type == 'learnable_swish':
                init_values = [1.0]  # beta parameter
            elif activation_type == 'learnable_gelu':
                init_values = [1.0]  # approximation parameter
            else:
                init_values = [1.0] * num_parameters

        self.parameters_raw = nn.Parameter(torch.tensor(init_values[:num_parameters]))

        # Parameter constraints to ensure numerical stability
        self.register_buffer('min_values', torch.tensor([-10.0] * num_parameters))
        self.register_buffer('max_values', torch.tensor([10.0] * num_parameters))

    @property
    def activation_parameters(self):
        """Get constrained activation parameters."""
        return torch.clamp(self.parameters_raw, self.min_values, self.max_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive activation function."""
        params = self.activation_parameters

        if self.activation_type == 'learnable_relu':
            negative_slope = params[0]
            positive_slope = params[1] if len(params) > 1 else 1.0
            return torch.where(x >= 0, positive_slope * x, negative_slope * x)

        elif self.activation_type == 'learnable_swish':
            beta = params[0]
            return x * torch.sigmoid(beta * x)

        elif self.activation_type == 'learnable_gelu':
            approx_param = params[0]
            return 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + approx_param * torch.pow(x, 3))
            ))

        elif self.activation_type == 'learnable_elu':
            alpha = params[0]
            return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))

        elif self.activation_type == 'learnable_softplus':
            beta = params[0]
            threshold = params[1] if len(params) > 1 else 20.0
            return F.softplus(beta * x, beta, threshold) / beta

        else:
            # Default to learnable ReLU
            return F.leaky_relu(x, negative_slope=params[0])


class ContextualActivation(nn.Module):
    """
    Contextual activation that adapts based on input context.

    This module uses attention mechanisms to compute context-aware
    activation parameters.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 64,
        num_heads: int = 4,
        base_activation: str = 'gelu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_heads = num_heads

        # Context extraction
        self.context_extractor = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU()
        )

        # Self-attention for context
        self.context_attention = nn.MultiheadAttention(
            context_dim, num_heads, batch_first=True
        )

        # Parameter prediction
        self.param_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, 2),  # [scale, shift] parameters
            nn.Tanh()
        )

        # Base activation
        self.base_activation = self._get_base_activation(base_activation)

    def _get_base_activation(self, activation_name: str) -> nn.Module:
        """Get base activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': Mish(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation_name.lower(), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with contextual activation.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Activated tensor with same shape as input
        """
        original_shape = x.shape

        # Handle different input shapes
        if len(original_shape) == 2:
            # [batch, features] -> [batch, 1, features] for attention
            x_reshaped = x.unsqueeze(1)
            needs_squeeze = True
        elif len(original_shape) == 3:
            # [batch, seq, features] - already good for attention
            x_reshaped = x
            needs_squeeze = False
        else:
            # Flatten to 3D
            x_reshaped = x.view(-1, 1, self.input_dim)
            needs_squeeze = True

        # Extract context
        context = self.context_extractor(x_reshaped)

        # Apply self-attention to context
        attended_context, _ = self.context_attention(context, context, context)

        # Predict activation parameters
        params = self.param_predictor(attended_context)  # [..., 2]
        scale = 1.0 + 0.5 * params[..., 0:1]  # Scale factor around 1.0
        shift = 0.1 * params[..., 1:2]        # Small shift

        # Apply base activation
        if needs_squeeze:
            x_for_activation = x_reshaped.squeeze(1) if x_reshaped.shape[1] == 1 else x_reshaped.view(-1, self.input_dim)
        else:
            x_for_activation = x_reshaped

        activated = self.base_activation(x_for_activation)

        # Apply contextual modulation
        if needs_squeeze and len(original_shape) == 2:
            scale_expanded = scale.squeeze(1)
            shift_expanded = shift.squeeze(1)
        elif needs_squeeze:
            scale_expanded = scale.view(original_shape)
            shift_expanded = shift.view(original_shape)
        else:
            scale_expanded = scale
            shift_expanded = shift

        modulated = activated * scale_expanded + shift_expanded

        # Reshape back to original shape
        if needs_squeeze and len(original_shape) > 3:
            modulated = modulated.view(original_shape)

        return modulated


class HierarchicalActivation(nn.Module):
    """
    Hierarchical activation that applies different activations at different scales.

    This module processes inputs at multiple scales and combines the results.
    """

    def __init__(
        self,
        input_dim: int,
        scales: List[int] = None,
        activation_per_scale: Optional[List[str]] = None,
        combination_method: str = 'weighted_sum'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales or [1, 2, 4]
        self.combination_method = combination_method

        # Default activations per scale
        if activation_per_scale is None:
            activation_per_scale = ['gelu', 'swish', 'mish']

        # Build activations for each scale
        self.scale_activations = nn.ModuleList()
        for scale, activation_name in zip(self.scales, activation_per_scale):
            activation_module = self._build_activation(activation_name)
            self.scale_activations.append(activation_module)

        # Scale processing
        self.scale_processors = nn.ModuleList()
        for scale in self.scales:
            if scale > 1:
                # Downsampling and upsampling for multi-scale processing
                processor = nn.ModuleDict({
                    'downsample': nn.AvgPool1d(kernel_size=scale, stride=scale),
                    'upsample': nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
                })
            else:
                processor = nn.ModuleDict({
                    'downsample': nn.Identity(),
                    'upsample': nn.Identity()
                })
            self.scale_processors.append(processor)

        # Combination weights
        if combination_method == 'weighted_sum':
            self.combination_weights = nn.Parameter(torch.ones(len(self.scales)))
        elif combination_method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, len(self.scales)),
                nn.Softmax(dim=-1)
            )

    def _build_activation(self, activation_name: str) -> nn.Module:
        """Build activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': Mish(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation_name.lower(), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical activation.

        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]

        Returns:
            Hierarchically activated tensor
        """
        # Handle 2D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, input_dim = x.shape
        scale_outputs = []

        # Process at each scale
        for scale_idx, (scale, activation, processor) in enumerate(
            zip(self.scales, self.scale_activations, self.scale_processors)
        ):
            # Prepare input for processing
            x_for_scale = x.transpose(1, 2)  # [batch, input_dim, seq_len]

            # Downsample if needed
            if scale > 1 and seq_len >= scale:
                x_downsampled = processor['downsample'](x_for_scale)
            else:
                x_downsampled = x_for_scale

            # Transpose back for activation
            x_downsampled = x_downsampled.transpose(1, 2)

            # Apply activation
            activated = activation(x_downsampled)

            # Upsample back if needed
            if scale > 1 and seq_len >= scale:
                activated = activated.transpose(1, 2)
                # Only upsample if the downsampled length is different
                if activated.shape[-1] != seq_len:
                    activated = F.interpolate(activated, size=seq_len, mode='linear', align_corners=False)
                activated = activated.transpose(1, 2)

            scale_outputs.append(activated)

        # Combine outputs from different scales
        if self.combination_method == 'weighted_sum':
            weights = F.softmax(self.combination_weights, dim=0)
            combined = sum(w * output for w, output in zip(weights, scale_outputs))

        elif self.combination_method == 'attention':
            # Use attention weights based on input
            attention_weights = self.attention_weights(x.mean(dim=1))  # Pool sequence
            attention_weights = attention_weights.unsqueeze(1)  # [batch, 1, num_scales]

            stacked_outputs = torch.stack(scale_outputs, dim=-1)  # [batch, seq, input_dim, num_scales]
            combined = (stacked_outputs * attention_weights.unsqueeze(2)).sum(dim=-1)

        else:  # 'mean'
            combined = torch.stack(scale_outputs).mean(dim=0)

        # Remove sequence dimension if it was added
        if squeeze_output:
            combined = combined.squeeze(1)

        return combined