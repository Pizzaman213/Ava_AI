"""
ALiBi (Attention with Linear Biases) Positional Encoding.

This module implements ALiBi from "Train Short, Test Long: Attention with
Linear Biases Enables Input Length Extrapolation" (Press et al., 2021).

ALiBi replaces learned or sinusoidal positional embeddings with linear
biases added directly to attention scores. Benefits:
1. No learned positional parameters
2. Better length extrapolation
3. Efficient computation
4. Works well with long sequences

Example:
    >>> alibi = ALiBiPositionalBias(num_heads=16)
    >>> attn_scores = torch.randn(2, 16, 128, 128)  # [batch, heads, q_len, k_len]
    >>> biased_scores = attn_scores + alibi(128)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    The slopes follow a geometric sequence to provide different
    attention patterns per head:
    - Heads with larger slopes focus on nearby tokens
    - Heads with smaller slopes can attend to distant tokens

    Args:
        num_heads: Number of attention heads

    Returns:
        slopes: [num_heads] tensor of slope values
    """
    # For power of 2 num_heads, use exact geometric sequence
    if num_heads <= 0:
        return torch.tensor([])

    def get_slopes_power_of_2(n: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        # For non-power-of-2, interpolate between nearest powers of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base_slopes = get_slopes_power_of_2(closest_power_of_2)

        # Get extra slopes by interpolation
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2][:num_heads - closest_power_of_2]

        slopes = base_slopes + extra_slopes

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(
    seq_len: int,
    num_heads: int,
    slopes: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build the ALiBi bias matrix.

    Creates a bias matrix where bias[i,j] = -slope * |i - j|
    This encourages attention to nearby positions.

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        slopes: Pre-computed slopes [num_heads]
        device: Target device
        dtype: Target dtype

    Returns:
        bias: [1, num_heads, seq_len, seq_len] bias matrix
    """
    if device is not None:
        slopes = slopes.to(device)

    # Create position difference matrix
    # positions[i, j] = j - i (positive = future, negative = past)
    positions = torch.arange(seq_len, device=device or slopes.device)
    # relative_positions[i, j] = i - j (position distance)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

    # Convert to float for computation
    relative_positions = relative_positions.float()

    # Apply slopes: bias = -slope * distance
    # slopes: [num_heads] -> [num_heads, 1, 1]
    slopes = slopes.view(-1, 1, 1)
    bias = -slopes * relative_positions.abs().unsqueeze(0)  # [num_heads, seq_len, seq_len]

    # Add batch dimension
    bias = bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

    if dtype is not None:
        bias = bias.to(dtype)

    return bias


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi positional bias module.

    Computes and caches linear biases to add to attention scores.
    The biases encode relative position information without learned
    parameters.

    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length to cache (optional)

    Example:
        >>> alibi = ALiBiPositionalBias(num_heads=16)
        >>> # Get bias for sequence length 128
        >>> bias = alibi(128)  # [1, 16, 128, 128]
        >>> # Add to attention scores
        >>> attn_scores = attn_scores + bias
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute and register slopes as buffer (not a parameter)
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes, persistent=True)

        # Optionally pre-compute bias for max_seq_len
        self._cached_bias: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

        logger.debug(f"ALiBiPositionalBias: {num_heads} heads, slopes range [{slopes.min():.4f}, {slopes.max():.4f}]")

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias matrix for given sequence length.

        Uses caching to avoid recomputation when possible.

        Args:
            seq_len: Sequence length
            device: Target device (uses buffer device if None)
            dtype: Target dtype (uses buffer dtype if None)

        Returns:
            bias: [1, num_heads, seq_len, seq_len] bias matrix
        """
        # Check cache
        if (
            self._cached_bias is not None
            and self._cached_seq_len >= seq_len
            and (device is None or self._cached_bias.device == device)
            and (dtype is None or self._cached_bias.dtype == dtype)
        ):
            # Return slice of cached bias
            return self._cached_bias[:, :, :seq_len, :seq_len]

        # Compute new bias
        target_device = device or self.slopes.device
        target_dtype = dtype or torch.float32

        bias = build_alibi_bias(
            seq_len=seq_len,
            num_heads=self.num_heads,
            slopes=self.slopes,
            device=target_device,
            dtype=target_dtype,
        )

        # Update cache if this is the new longest sequence
        if seq_len > self._cached_seq_len:
            self._cached_bias = bias
            self._cached_seq_len = seq_len

        return bias

    def get_slopes(self) -> torch.Tensor:
        """Get the ALiBi slopes for each head."""
        return self.slopes

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}"


class ALiBiAttention(nn.Module):
    """
    Attention layer with built-in ALiBi positional encoding.

    Drop-in replacement for standard attention that uses ALiBi
    instead of learned positional embeddings.

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        dropout: Attention dropout probability
        use_bias: Whether to use bias in projections

    Example:
        >>> attn = ALiBiAttention(hidden_size=1024, num_heads=16)
        >>> x = torch.randn(2, 128, 1024)
        >>> output = attn(x, x, x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # ALiBi
        self.alibi = ALiBiPositionalBias(num_heads)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with ALiBi positional encoding.

        Args:
            query: [batch, seq_q, hidden_size]
            key: [batch, seq_k, hidden_size]
            value: [batch, seq_k, hidden_size]
            attention_mask: Optional mask [batch, 1, seq_q, seq_k] or [batch, seq_q, seq_k]
            is_causal: Whether to apply causal masking

        Returns:
            output: [batch, seq_q, hidden_size]
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add ALiBi bias
        alibi_bias = self.alibi(max(seq_q, seq_k), device=q.device, dtype=q.dtype)
        # Handle different q/k lengths by slicing appropriately
        attn_scores = attn_scores + alibi_bias[:, :, :seq_q, :seq_k]

        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask

        # Softmax and dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back: [batch, num_heads, seq_q, head_dim] -> [batch, seq_q, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output


def apply_alibi_to_attention_scores(
    attention_scores: torch.Tensor,
    alibi_bias: ALiBiPositionalBias,
) -> torch.Tensor:
    """
    Apply ALiBi bias to pre-computed attention scores.

    Utility function for integrating ALiBi into existing attention
    implementations.

    Args:
        attention_scores: [batch, num_heads, seq_q, seq_k]
        alibi_bias: ALiBiPositionalBias module

    Returns:
        biased_scores: [batch, num_heads, seq_q, seq_k]
    """
    seq_q, seq_k = attention_scores.shape[-2:]
    bias = alibi_bias(max(seq_q, seq_k), device=attention_scores.device, dtype=attention_scores.dtype)
    return attention_scores + bias[:, :, :seq_q, :seq_k]


__all__ = [
    'get_alibi_slopes',
    'build_alibi_bias',
    'ALiBiPositionalBias',
    'ALiBiAttention',
    'apply_alibi_to_attention_scores',
]
