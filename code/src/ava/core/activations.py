"""
Activation Function Factory.

This module provides a centralized registry and factory for activation functions,
replacing duplicate if/elif chains throughout the codebase.

Supported activations:
- swiglu/silu/swish: SiLU (Sigmoid Linear Unit)
- geglu/gelu: GELU (Gaussian Error Linear Unit)
- relu: ReLU (Rectified Linear Unit)
- tanh: Hyperbolic tangent
- sigmoid: Sigmoid function

Gated activations (swiglu, geglu) require 2x intermediate size in MLP layers.
"""

import torch.nn as nn

# Registry mapping activation names to PyTorch modules
ACTIVATION_REGISTRY = {
    'swiglu': nn.SiLU,   # SwiGLU uses SiLU (Swish) as the gating activation
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'geglu': nn.GELU,    # GeGLU uses GELU as the gating activation
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}

# Set of gated activations that require 2x intermediate size
GATED_ACTIVATIONS = frozenset({'swiglu', 'geglu'})


def get_activation(activation: str) -> nn.Module:
    """
    Get activation module by name.

    This is the single source of truth for activation selection,
    replacing 5+ duplicate if/elif chains across the codebase.

    Args:
        activation: Activation name ('swiglu', 'geglu', 'gelu', 'relu', 'silu')

    Returns:
        PyTorch activation module instance

    Raises:
        ValueError: If activation name is unknown

    Example:
        >>> act = get_activation('swiglu')
        >>> output = act(input_tensor)
    """
    activation = activation.lower()

    if activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Supported: {list(ACTIVATION_REGISTRY.keys())}"
        )

    return ACTIVATION_REGISTRY[activation]()


def is_gated_activation(activation: str) -> bool:
    """
    Check if activation uses gating (requires 2x intermediate size).

    Gated activations like SwiGLU and GeGLU split the intermediate
    representation into two parts: one for gating and one for values.
    This requires doubling the intermediate_size in MLP configurations.

    Args:
        activation: Activation name to check

    Returns:
        True if activation is gated (swiglu, geglu), False otherwise

    Example:
        >>> if is_gated_activation('swiglu'):
        ...     intermediate_size *= 2
    """
    return activation.lower() in GATED_ACTIVATIONS


__all__ = [
    'ACTIVATION_REGISTRY',
    'GATED_ACTIVATIONS',
    'get_activation',
    'is_gated_activation',
]
