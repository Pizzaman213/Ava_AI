"""
Fusion strategies for RAG.

This module provides different ways to combine retrieved document
context with model hidden states during RAG inference.

Supported strategies:
- ConcatFusion: Concatenate context and project
- GatedFusion: Learnable gate for context mixing
- CrossAttentionFusion: Cross-attention over retrieved documents

Usage:
    from ava.rag import GatedFusion

    fusion = GatedFusion(hidden_size=768, context_size=768)
    output = fusion(hidden_states, retrieved_context)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RAGFusion(nn.Module):
    """
    Base class for RAG fusion strategies.

    Fusion modules take model hidden states and retrieved context,
    and produce fused hidden states that incorporate the retrieved
    information.
    """

    def __init__(self, hidden_size: int, context_size: Optional[int] = None):
        """
        Initialize fusion module.

        Args:
            hidden_size: Dimension of model hidden states
            context_size: Dimension of context embeddings (defaults to hidden_size)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size or hidden_size

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse hidden states with retrieved context.

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            context: Retrieved context [batch, num_docs, context_size] or
                    [batch, seq_len, context_size] if already expanded

        Returns:
            Fused hidden states [batch, seq_len, hidden_size]
        """
        raise NotImplementedError


class ConcatFusion(RAGFusion):
    """
    Concatenate retrieved context with hidden states and project.

    Simple but effective fusion: concatenates context (expanded to sequence
    length) with hidden states and uses a linear projection to reduce back
    to hidden_size.

    Architecture:
        fused = Linear([hidden_states; context_expanded])
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, context_size)

        # Project concatenated representation back to hidden_size
        self.projection = nn.Linear(hidden_size + self.context_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse by concatenation and projection.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            context: [batch, num_docs, context_size]

        Returns:
            Fused hidden states [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Expand context to match sequence length
        # Take mean over documents and expand
        if context.dim() == 3 and context.size(1) != seq_len:
            context_pooled = context.mean(dim=1, keepdim=True)  # [batch, 1, context]
            context_expanded = context_pooled.expand(-1, seq_len, -1)  # [batch, seq, context]
        else:
            context_expanded = context

        # Concatenate and project
        combined = torch.cat([hidden_states, context_expanded], dim=-1)
        fused = self.projection(combined)
        fused = self.dropout(fused)

        # Residual connection with layer norm
        output = self.layer_norm(hidden_states + fused)

        return output


class GatedFusion(RAGFusion):
    """
    Gated fusion of retrieved context.

    Uses a learnable gate to control how much retrieved context
    affects the hidden states. This allows the model to learn
    when to rely on retrieval vs. its own knowledge.

    Architecture:
        context_proj = Linear(context)
        gate = sigmoid(Linear([hidden_states; context_proj]))
        output = hidden_states + gate * context_proj
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, context_size)

        # Project context to hidden size
        self.context_proj = nn.Linear(self.context_size, hidden_size)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse with learned gating.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            context: [batch, num_docs, context_size]

        Returns:
            Fused hidden states [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pool and expand context
        if context.dim() == 3 and context.size(1) != seq_len:
            context_pooled = context.mean(dim=1, keepdim=True)
            context_expanded = context_pooled.expand(-1, seq_len, -1)
        else:
            context_expanded = context

        # Project context to hidden size
        context_proj = self.context_proj(context_expanded)
        context_proj = self.dropout(context_proj)

        # Compute gate
        gate_input = torch.cat([hidden_states, context_proj], dim=-1)
        gate = self.gate(gate_input)

        # Apply gated fusion
        fused = hidden_states + gate * context_proj
        output = self.layer_norm(fused)

        return output


class CrossAttentionFusion(RAGFusion):
    """
    Cross-attention fusion over retrieved documents.

    Uses multi-head cross-attention where the hidden states are
    the queries and retrieved documents are keys/values. This
    allows token-level attention over retrieved content.

    Architecture:
        attn_output = MultiheadAttention(Q=hidden, K=context, V=context)
        output = LayerNorm(hidden + attn_output)
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, context_size)

        self.num_heads = num_heads

        # Project context to hidden size if different
        if self.context_size != hidden_size:
            self.context_proj = nn.Linear(self.context_size, hidden_size)
        else:
            self.context_proj = None

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse with cross-attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            context: [batch, num_docs, context_size]

        Returns:
            Fused hidden states [batch, seq_len, hidden_size]
        """
        # Project context if needed
        if self.context_proj is not None:
            context = self.context_proj(context)

        # Cross-attention: hidden states attend to context
        attn_output, attn_weights = self.cross_attn(
            query=hidden_states,
            key=context,
            value=context,
        )

        attn_output = self.dropout(attn_output)

        # Residual connection with layer norm
        output = self.layer_norm(hidden_states + attn_output)

        return output

    def forward_with_weights(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> tuple:
        """
        Fuse with cross-attention and return attention weights.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            context: [batch, num_docs, context_size]

        Returns:
            Tuple of (fused hidden states, attention weights)
        """
        if self.context_proj is not None:
            context = self.context_proj(context)

        attn_output, attn_weights = self.cross_attn(
            query=hidden_states,
            key=context,
            value=context,
            average_attn_weights=False,
        )

        attn_output = self.dropout(attn_output)
        output = self.layer_norm(hidden_states + attn_output)

        return output, attn_weights


class AdaptiveFusion(RAGFusion):
    """
    Adaptive fusion that combines multiple strategies.

    Uses a learned router to choose between different fusion
    strategies based on the input.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, context_size)

        # Multiple fusion strategies
        self.concat_fusion = ConcatFusion(hidden_size, self.context_size, dropout)
        self.gated_fusion = GatedFusion(hidden_size, self.context_size, dropout)
        self.cross_attn_fusion = CrossAttentionFusion(
            hidden_size, self.context_size, num_heads, dropout
        )

        # Router to select strategy
        self.router = nn.Sequential(
            nn.Linear(hidden_size + self.context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse with adaptive strategy selection.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            context: [batch, num_docs, context_size]

        Returns:
            Fused hidden states [batch, seq_len, hidden_size]
        """
        # Compute routing weights
        hidden_mean = hidden_states.mean(dim=1)  # [batch, hidden]
        context_mean = context.mean(dim=1)  # [batch, context]
        router_input = torch.cat([hidden_mean, context_mean], dim=-1)
        weights = self.router(router_input)  # [batch, 3]

        # Apply each fusion strategy
        concat_out = self.concat_fusion(hidden_states, context)
        gated_out = self.gated_fusion(hidden_states, context)
        cross_out = self.cross_attn_fusion(hidden_states, context)

        # Weighted combination
        weights = weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, 3, 1]
        stacked = torch.stack([concat_out, gated_out, cross_out], dim=2)  # [batch, seq, 3, hidden]
        output = (stacked * weights).sum(dim=2)  # [batch, seq, hidden]

        return output


def create_fusion(config: any, hidden_size: int) -> RAGFusion:
    """
    Factory function to create a fusion module from config.

    Args:
        config: RAGConfig with fusion settings
        hidden_size: Model hidden size

    Returns:
        RAGFusion instance
    """
    fusion_type = getattr(config, "rag_fusion_type", "attention").lower()
    context_size = getattr(config, "embedding_dim", hidden_size)
    dropout = getattr(config, "fusion_dropout", 0.1)
    num_heads = getattr(config, "fusion_num_heads", 8)

    if fusion_type == "concat":
        return ConcatFusion(hidden_size, context_size, dropout)
    elif fusion_type == "gated":
        return GatedFusion(hidden_size, context_size, dropout)
    elif fusion_type in ["attention", "cross_attention"]:
        return CrossAttentionFusion(hidden_size, context_size, num_heads, dropout)
    elif fusion_type == "adaptive":
        return AdaptiveFusion(hidden_size, context_size, num_heads, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
