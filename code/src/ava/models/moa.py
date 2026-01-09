"""
Mixture of Activations (MoA) - Dynamic activation function selection.

This module implements Mixture of Activations which allows the model to
dynamically select different activation functions for different tokens,
providing:
1. Adaptive non-linearity based on input characteristics
2. Potential for better gradient flow via activation diversity
3. Combination of benefits from different activation families

Example:
    >>> moa = MixtureOfActivations(
    ...     hidden_size=1024,
    ...     intermediate_size=4096,
    ...     num_activations=4,
    ... )
    >>> x = torch.randn(2, 128, 1024)
    >>> output, aux_loss = moa(x)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Available activation functions
ACTIVATION_REGISTRY: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'gelu': F.gelu,
    'relu': F.relu,
    'silu': F.silu,  # SiLU / Swish
    'tanh': torch.tanh,
    'leaky_relu': lambda x: F.leaky_relu(x, 0.1),
    'elu': F.elu,
    'mish': lambda x: x * torch.tanh(F.softplus(x)),
    'softplus': F.softplus,
    'gelu_approx': lambda x: F.gelu(x, approximate='tanh'),
}


@dataclass
class MoAConfig:
    """Configuration for Mixture of Activations."""
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_activations: int = 4
    activation_types: Tuple[str, ...] = ('swiglu', 'geglu', 'gelu', 'silu')
    dropout: float = 0.0
    router_jitter: float = 0.0
    load_balance_loss_coef: float = 0.01
    use_gated: bool = True  # Use gated activations (SwiGLU, GeGLU, etc.)
    routing_type: str = 'softmax'  # 'softmax', 'topk', 'sigmoid'
    num_active: int = 1  # Number of activations to use per token (1 = hard routing)


class ActivationRouter(nn.Module):
    """
    Router for selecting which activation function to use.

    Args:
        hidden_size: Input hidden dimension
        num_activations: Number of activation functions to route to
        routing_type: Type of routing computation
        jitter: Jitter noise for exploration
    """

    def __init__(
        self,
        hidden_size: int,
        num_activations: int,
        routing_type: str = 'softmax',
        jitter: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_activations = num_activations
        self.routing_type = routing_type
        self.jitter = jitter

        # Router projection
        self.router = nn.Linear(hidden_size, num_activations, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for balanced start."""
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute routing probabilities.

        Args:
            hidden_states: [batch, seq, hidden_size]
            training: Whether in training mode

        Returns:
            routing_probs: [batch, seq, num_activations]
            routing_info: Dictionary with metrics
        """
        # Compute router logits
        router_logits = self.router(hidden_states)

        # Add jitter during training
        if training and self.jitter > 0:
            noise = torch.randn_like(router_logits) * self.jitter
            router_logits = router_logits + noise

        # Compute probabilities
        if self.routing_type == 'softmax':
            routing_probs = F.softmax(router_logits, dim=-1)
        elif self.routing_type == 'sigmoid':
            routing_probs = torch.sigmoid(router_logits)
            # Normalize
            routing_probs = routing_probs / (routing_probs.sum(dim=-1, keepdim=True) + 1e-9)
        elif self.routing_type == 'topk':
            # Hard top-1 routing with straight-through estimator
            topk_idx = router_logits.argmax(dim=-1, keepdim=True)
            routing_probs = torch.zeros_like(router_logits)
            routing_probs.scatter_(-1, topk_idx, 1.0)
            # Straight-through: use soft probs for gradients
            soft_probs = F.softmax(router_logits, dim=-1)
            routing_probs = routing_probs - soft_probs.detach() + soft_probs
        else:
            routing_probs = F.softmax(router_logits, dim=-1)

        # Compute metrics
        routing_info: Dict[str, Any] = {}

        # Load balance: activation usage distribution
        act_usage = routing_probs.mean(dim=(0, 1))  # [num_activations]
        target_usage = 1.0 / self.num_activations
        load_balance_loss = ((act_usage - target_usage) ** 2).sum() * self.num_activations
        routing_info['load_balance_loss'] = load_balance_loss
        routing_info['activation_usage'] = act_usage

        # Entropy (routing diversity)
        entropy = -(routing_probs * (routing_probs + 1e-9).log()).sum(dim=-1).mean()
        routing_info['routing_entropy'] = entropy

        return routing_probs, routing_info


class GatedActivation(nn.Module):
    """
    Gated activation function (SwiGLU, GeGLU, etc.).

    Implements: activation(gate) * up, where gate and up are split from input.

    Args:
        activation_fn: Base activation function
    """

    def __init__(self, activation_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated activation.

        Args:
            x: [batch, seq, intermediate_size * 2]

        Returns:
            output: [batch, seq, intermediate_size]
        """
        gate, up = x.chunk(2, dim=-1)
        return self.activation_fn(gate) * up


class MixtureOfActivations(nn.Module):
    """
    Mixture of Activations FFN layer.

    Routes tokens to different activation functions and combines outputs.
    Can use either hard routing (one activation per token) or soft routing
    (weighted combination of activations).

    Architecture:
    - Up projection to intermediate size (2x for gated)
    - Parallel activation functions
    - Routing-weighted combination
    - Down projection back to hidden size

    Args:
        config: MoAConfig with all settings

    Example:
        >>> config = MoAConfig(hidden_size=1024, intermediate_size=4096)
        >>> moa = MixtureOfActivations(config)
        >>> x = torch.randn(2, 128, 1024)
        >>> output, aux_loss, info = moa(x, training=True)
    """

    def __init__(self, config: MoAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_activations = config.num_activations
        self.use_gated = config.use_gated

        # Determine projection sizes
        if config.use_gated:
            self.proj_size = config.intermediate_size * 2
        else:
            self.proj_size = config.intermediate_size

        # Up projection (shared across activations)
        self.up_proj = nn.Linear(config.hidden_size, self.proj_size, bias=False)

        # Down projection (shared)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # Create activation functions
        self.activations = nn.ModuleList()
        activation_types = config.activation_types[:config.num_activations]

        # Pad with defaults if not enough
        while len(activation_types) < config.num_activations:
            activation_types = activation_types + ('gelu',)

        for act_type in activation_types:
            if act_type == 'swiglu':
                self.activations.append(GatedActivation(F.silu))
            elif act_type == 'geglu':
                self.activations.append(GatedActivation(F.gelu))
            elif act_type == 'reglu':
                self.activations.append(GatedActivation(F.relu))
            elif act_type in ACTIVATION_REGISTRY:
                if config.use_gated:
                    self.activations.append(GatedActivation(ACTIVATION_REGISTRY[act_type]))
                else:
                    # Wrap in module for consistency
                    self.activations.append(
                        nn.Sequential(
                            nn.Lambda(ACTIVATION_REGISTRY[act_type])  # type: ignore
                        ) if hasattr(nn, 'Lambda') else
                        _ActivationWrapper(ACTIVATION_REGISTRY[act_type])
                    )
            else:
                # Default to gated GELU
                self.activations.append(GatedActivation(F.gelu))

        # Router
        self.router = ActivationRouter(
            hidden_size=config.hidden_size,
            num_activations=config.num_activations,
            routing_type=config.routing_type,
            jitter=config.router_jitter,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        self._init_weights()

        logger.info(
            f"MixtureOfActivations: {config.num_activations} activations, "
            f"hidden={config.hidden_size}, intermediate={config.intermediate_size}"
        )

    def _init_weights(self):
        """Initialize projections."""
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dynamic activation selection.

        Args:
            hidden_states: [batch, seq, hidden_size]
            training: Whether in training mode

        Returns:
            output: [batch, seq, hidden_size]
            aux_loss: Auxiliary loss
            info: Routing metrics
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get routing probabilities
        routing_probs, routing_info = self.router(hidden_states, training)
        # [batch, seq, num_activations]

        # Up projection
        up = self.up_proj(hidden_states)  # [batch, seq, proj_size]

        # Apply each activation and weight by routing probs
        combined_output = torch.zeros(
            batch_size, seq_len, self.intermediate_size,
            device=device, dtype=dtype,
        )

        for i, activation in enumerate(self.activations):
            # Apply activation
            act_output = activation(up)  # [batch, seq, intermediate_size]

            # Weight by routing probability
            weight = routing_probs[..., i:i+1]  # [batch, seq, 1]
            combined_output = combined_output + weight * act_output

        # Dropout
        if self.dropout is not None:
            combined_output = self.dropout(combined_output)

        # Down projection
        output = self.down_proj(combined_output)

        # Compute auxiliary loss
        aux_loss = self.config.load_balance_loss_coef * routing_info['load_balance_loss']

        return output, aux_loss, routing_info


class _ActivationWrapper(nn.Module):
    """Simple wrapper to make a function callable as a module."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class MoAExpertLayer(nn.Module):
    """
    FFN layer with MoA for use in MoE architectures.

    Combines Mixture of Experts (for capacity) with Mixture of Activations
    (for non-linearity diversity).

    Args:
        config: MoAConfig with settings
    """

    def __init__(self, config: MoAConfig):
        super().__init__()
        self.moa = MixtureOfActivations(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward with pre-norm and residual.

        Args:
            hidden_states: [batch, seq, hidden_size]
            training: Whether in training mode

        Returns:
            output: [batch, seq, hidden_size]
            aux_loss: Auxiliary loss
            info: Metrics
        """
        # Pre-norm
        normed = self.layer_norm(hidden_states)

        # MoA forward
        moa_output, aux_loss, info = self.moa(normed, training)

        # Residual
        output = hidden_states + moa_output

        return output, aux_loss, info


def create_moa_from_config(
    hidden_size: int,
    intermediate_size: int,
    activation_types: Optional[Tuple[str, ...]] = None,
    num_activations: int = 4,
    use_gated: bool = True,
    **kwargs,
) -> MixtureOfActivations:
    """
    Create MoA from common parameters.

    Args:
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        activation_types: Tuple of activation names
        num_activations: Number of activations to route between
        use_gated: Whether to use gated activations
        **kwargs: Additional MoAConfig parameters

    Returns:
        Configured MixtureOfActivations
    """
    config = MoAConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_activations=num_activations,
        activation_types=activation_types or ('swiglu', 'geglu', 'gelu', 'silu'),
        use_gated=use_gated,
        **kwargs,
    )
    return MixtureOfActivations(config)


__all__ = [
    'MoAConfig',
    'ActivationRouter',
    'GatedActivation',
    'MixtureOfActivations',
    'MoAExpertLayer',
    'create_moa_from_config',
    'ACTIVATION_REGISTRY',
]
