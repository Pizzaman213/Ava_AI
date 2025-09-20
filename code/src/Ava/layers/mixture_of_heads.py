"""
Mixture of Heads (MoH) implementation for dynamic attention head selection.

This module implements dynamic attention head selection where different tokens
can use different attention heads based on their content and requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class MixtureOfHeads(nn.Module):
    """
    Mixture of Heads (MoH) - Dynamic attention head selection.

    This module allows different tokens to use different attention heads
    based on content-aware routing, enabling specialized attention patterns
    for different types of content.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_head_experts (int): Number of specialized head groups
        dropout (float): Dropout probability
        use_head_routing (bool): Whether to use dynamic head routing
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_head_experts: int = 4,
        dropout: float = 0.1,
        use_head_routing: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_head_experts = num_head_experts
        self.head_dim = embed_dim // num_heads
        self.use_head_routing = use_head_routing

        assert self.head_dim * num_heads == embed_dim

        # Head routing network
        if use_head_routing:
            self.head_router = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, num_head_experts),
                nn.Softmax(dim=-1)
            )

        # Specialized head groups
        self.head_experts = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
                'k_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
                'v_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
                'out_proj': nn.Linear(embed_dim, embed_dim, bias=bias),
            }) for _ in range(num_head_experts)
        ])

        # Attention patterns for different head experts
        self.attention_patterns = nn.Parameter(
            torch.randn(num_head_experts, num_heads, num_heads)
        )

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scale factor
        self.scale = self.head_dim ** -0.5

        # Head specialization embeddings
        self.head_specializations = nn.Parameter(
            torch.randn(num_head_experts, embed_dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass with dynamic head selection.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights, routing_info)
        """
        batch_size, seq_len, _ = query.shape

        if self.use_head_routing:
            # Compute head routing weights
            routing_weights = self.head_router(query)  # [batch, seq, num_head_experts]

            # Select top-k head experts per token
            top_k = min(2, self.num_head_experts)
            top_weights, top_indices = torch.topk(routing_weights, top_k, dim=-1)

            # Initialize output
            output = torch.zeros_like(query)
            all_attn_weights = []

            # Process each selected head expert
            for k in range(top_k):
                expert_indices = top_indices[:, :, k]  # [batch, seq]
                expert_weights = top_weights[:, :, k]   # [batch, seq]

                # Process each expert
                for expert_idx in range(self.num_head_experts):
                    # Find tokens assigned to this expert
                    expert_mask = (expert_indices == expert_idx)
                    if not expert_mask.any():
                        continue

                    # Get expert projections
                    expert = self.head_experts[expert_idx]

                    # Project Q, K, V for this expert
                    q = expert['q_proj'](query)
                    k = expert['k_proj'](key)
                    v = expert['v_proj'](value)

                    # Reshape for multi-head attention
                    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    # Apply attention pattern for this expert
                    pattern = self.attention_patterns[expert_idx]
                    q = torch.einsum('bhsd,hh->bhsd', q, pattern)

                    # Compute attention
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                    if attention_mask is not None:
                        scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = self.attn_dropout(attn_weights)

                    attn_output = torch.matmul(attn_weights, v)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

                    # Final projection
                    expert_output = expert['out_proj'](attn_output)

                    # Weight by routing weights and expert assignment
                    mask_expanded = expert_mask.unsqueeze(-1).float()
                    weight_expanded = expert_weights.unsqueeze(-1)
                    weighted_output = expert_output * mask_expanded * weight_expanded

                    output += weighted_output

                    if need_weights:
                        all_attn_weights.append(attn_weights)

            output = self.resid_dropout(output)

            # Routing information
            routing_info = {
                'routing_weights': routing_weights,
                'top_weights': top_weights,
                'top_indices': top_indices,
                'head_utilization': self._compute_head_utilization(top_indices)
            }

        else:
            # Standard multi-head attention without routing
            expert = self.head_experts[0]  # Use first expert

            q = expert['q_proj'](query)
            k = expert['k_proj'](key)
            v = expert['v_proj'](value)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

            output = expert['out_proj'](attn_output)
            output = self.resid_dropout(output)

            all_attn_weights = [attn_weights] if need_weights else []
            routing_info = {}

        final_attn_weights = torch.stack(all_attn_weights).mean(0) if all_attn_weights else None

        return output, final_attn_weights, routing_info

    def _compute_head_utilization(self, top_indices: torch.Tensor) -> torch.Tensor:
        """Compute utilization statistics for each head expert."""
        batch_size, seq_len, top_k = top_indices.shape
        utilization = torch.zeros(self.num_head_experts, device=top_indices.device)

        for expert_idx in range(self.num_head_experts):
            expert_count = (top_indices == expert_idx).sum()
            utilization[expert_idx] = expert_count.float() / (batch_size * seq_len * top_k)

        return utilization


class AdaptiveHeadAttention(nn.Module):
    """
    Adaptive Head Attention with dynamic head count selection.

    This module dynamically selects the number of attention heads
    to use based on input complexity and computational budget.
    """

    def __init__(
        self,
        embed_dim: int,
        max_heads: int = 16,
        min_heads: int = 2,
        complexity_threshold: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_heads = max_heads
        self.min_heads = min_heads
        self.complexity_threshold = complexity_threshold

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Attention heads with different capacities
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim, num_heads=i+1, dropout=dropout, batch_first=True
            ) for i in range(min_heads-1, max_heads)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive head count.
        """
        # Estimate complexity
        complexity = self.complexity_estimator(query).mean()

        # Select number of heads based on complexity
        if complexity > self.complexity_threshold:
            head_idx = self.max_heads - self.min_heads
        else:
            # Linear interpolation between min and max heads
            head_ratio = complexity / self.complexity_threshold
            head_count = int(self.min_heads + head_ratio * (self.max_heads - self.min_heads))
            head_idx = head_count - self.min_heads

        head_idx = max(0, min(head_idx, len(self.attention_heads) - 1))

        # Apply selected attention
        attention_layer = self.attention_heads[head_idx]
        output, attn_weights = attention_layer(
            query, key, value,
            key_padding_mask=attention_mask,
            need_weights=True
        )

        info = {
            'complexity_score': complexity.item(),
            'selected_heads': head_idx + self.min_heads,
            'attention_weights': attn_weights
        }

        return output, info