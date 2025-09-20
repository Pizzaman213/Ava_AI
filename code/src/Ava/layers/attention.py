"""
Enhanced attention mechanisms for Qwen MoE++ model.

This module implements advanced attention mechanisms including multi-head
attention with various enhancements for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EnhancedMultiheadAttention(nn.Module):
    """
    Enhanced multi-head attention with additional features.

    This attention mechanism includes:
    - Standard multi-head self-attention
    - Optional rotary position embeddings
    - Attention dropout
    - Efficient scaled dot-product attention

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in projections

    Example:
        >>> attention = EnhancedMultiheadAttention(embed_dim=768, num_heads=12)
        >>> output, weights = attention(query, key, value)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Padding mask of shape (batch_size, seq_len)
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.shape

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask for all heads
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        if need_weights:
            return output, attn_weights
        else:
            return output, None


class ALiBiPositionEmbedding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for position encoding.

    ALiBi adds linear biases to attention scores based on distance,
    providing better extrapolation to longer sequences than learned embeddings.

    Args:
        num_heads (int): Number of attention heads
        max_seq_len (int): Maximum sequence length for precomputed slopes

    Example:
        >>> alibi = ALiBiPositionEmbedding(num_heads=12, max_seq_len=2048)
        >>> bias = alibi(seq_len=100)
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Precompute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute bias matrix for efficiency
        self._precompute_bias_matrix()

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute slopes for ALiBi biases."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads), dtype=torch.float32)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = self._get_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            return torch.tensor(slopes_a + slopes_b, dtype=torch.float32)

    def _precompute_bias_matrix(self):
        """Precompute bias matrix for common sequence lengths."""
        seq_len = self.max_seq_len
        # Create position matrix
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        positions = positions.abs()

        # Apply slopes to get biases
        biases = positions.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        self.register_buffer('precomputed_biases', -biases)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get ALiBi biases for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Bias tensor of shape [num_heads, seq_len, seq_len]
        """
        if seq_len <= self.max_seq_len:
            return self.precomputed_biases[:, :seq_len, :seq_len]
        else:
            # Compute biases for longer sequences on-the-fly
            positions = torch.arange(seq_len, device=self.slopes.device)
            positions = positions.unsqueeze(0) - positions.unsqueeze(1)
            positions = positions.abs()
            biases = positions.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
            return -biases


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for improved position encoding.

    This implements the rotary position embedding from the RoFormer paper,
    which has shown improvements in handling longer sequences.

    Args:
        dim (int): Dimension of the embeddings
        max_position_embeddings (int): Maximum sequence length
        base (int): Base for the geometric progression

    Example:
        >>> rope = RotaryPositionEmbedding(dim=64, max_position_embeddings=1024)
        >>> q_embedded = rope(query, seq_len=100)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin
        self._precompute_cos_sin()

    def _precompute_cos_sin(self):
        """Precompute cosine and sine values for positions."""
        max_seq_len = self.max_position_embeddings
        t = torch.arange(max_seq_len).to(dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply rotary position embedding.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Sequence length (uses x.shape[1] if not provided)

        Returns:
            Tensor with rotary position embedding applied
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Get cached cos and sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)

        # Split x for rotation
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

        # Apply rotation formula
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        ], dim=-1)

        return rotated


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for memory-efficient attention computation.

    This implements a memory-efficient attention mechanism that reduces
    memory usage from O(N²) to O(N) for sequence length N.

    Note: This is a simplified implementation. Production use should
    leverage optimized kernels.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability

    Example:
        >>> flash_attn = FlashAttention(embed_dim=768, num_heads=12)
        >>> output = flash_attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with flash attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention with memory-efficient algorithm
        # Note: This is simplified - real flash attention uses tiling
        scale = self.head_dim ** -0.5

        # Chunk processing for memory efficiency
        chunk_size = min(64, seq_len)
        output_chunks = []

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]

            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                scores = scores.masked_fill(
                    attention_mask[:, None, i:end_i, :],
                    float('-inf')
                )

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            output_chunk = torch.matmul(attn_weights, v)
            output_chunks.append(output_chunk)

        # Concatenate chunks
        output = torch.cat(output_chunks, dim=2)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output