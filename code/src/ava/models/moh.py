"""
Mixture of Heads (MoH) - Dynamic attention head selection.

This module implements Mixture of Heads which allows the model to dynamically
select which attention heads to use for each token, providing:
1. Adaptive computation based on input complexity
2. Improved parameter efficiency
3. Potential for faster inference via head pruning

Based on research from:
- "Mixture-of-Head Attention" (Csordás et al., 2023)
- "Multi-Head Attention: Collaborate Instead of Concatenate" (Cordonnier et al., 2020)

Example:
    >>> moh = MixtureOfHeads(
    ...     num_heads=16,
    ...     hidden_size=1024,
    ...     num_active_heads=4,
    ... )
    >>> x = torch.randn(2, 128, 1024)
    >>> output, routing_info = moh(query, key, value, x)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MoHConfig:
    """Configuration for Mixture of Heads."""
    num_heads: int = 16
    hidden_size: int = 1024
    num_active_heads: int = 4  # Top-k heads to select per token
    head_dim: Optional[int] = None  # If None, computed as hidden_size // num_heads
    dropout: float = 0.0
    router_jitter: float = 0.0  # Jitter for exploration during training
    load_balance_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    use_learned_routing: bool = True  # vs fixed/random routing
    routing_type: str = 'softmax'  # 'softmax', 'sigmoid', 'topk_softmax'
    use_bias: bool = False


class HeadRouter(nn.Module):
    """
    Router for selecting which attention heads to use.

    Computes routing weights for each attention head based on the input.

    Args:
        hidden_size: Input hidden dimension
        num_heads: Total number of attention heads
        num_active: Number of heads to activate per token
        jitter: Jitter noise for exploration
        routing_type: Type of routing ('softmax', 'sigmoid', 'topk_softmax')
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_active: int = 4,
        jitter: float = 0.0,
        routing_type: str = 'softmax',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_active = num_active
        self.jitter = jitter
        self.routing_type = routing_type

        # Router projection
        self.router = nn.Linear(hidden_size, num_heads, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize router with small weights for balanced start."""
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute routing weights and select top-k heads.

        Args:
            hidden_states: [batch, seq, hidden_size]
            training: Whether in training mode

        Returns:
            routing_weights: [batch, seq, num_active] selected head weights
            selected_heads: [batch, seq, num_active] indices of selected heads
            routing_info: Dictionary with auxiliary losses and metrics
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch, seq, num_heads]

        # Add jitter during training
        if training and self.jitter > 0:
            noise = torch.randn_like(router_logits) * self.jitter
            router_logits = router_logits + noise

        # Compute routing probabilities
        if self.routing_type == 'softmax':
            router_probs = F.softmax(router_logits, dim=-1)
        elif self.routing_type == 'sigmoid':
            router_probs = torch.sigmoid(router_logits)
        elif self.routing_type == 'topk_softmax':
            # Softmax only over top-k (sparse softmax)
            topk_vals, topk_idx = router_logits.topk(self.num_active, dim=-1)
            sparse_probs = F.softmax(topk_vals, dim=-1)
            router_probs = torch.zeros_like(router_logits)
            router_probs.scatter_(-1, topk_idx, sparse_probs)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k heads
        routing_weights, selected_heads = router_probs.topk(self.num_active, dim=-1)
        # [batch, seq, num_active], [batch, seq, num_active]

        # Renormalize weights
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute auxiliary losses
        routing_info: Dict[str, Any] = {}

        # Load balance loss: encourage uniform head usage
        # Count how often each head is selected
        head_counts = torch.zeros(self.num_heads, device=device)
        for h in range(self.num_heads):
            head_counts[h] = (selected_heads == h).float().sum()
        head_probs = head_counts / (head_counts.sum() + 1e-9)
        target_prob = 1.0 / self.num_heads
        load_balance_loss = ((head_probs - target_prob) ** 2).sum() * self.num_heads
        routing_info['load_balance_loss'] = load_balance_loss

        # Router z-loss: prevent router logits from growing too large
        router_z_loss = (router_logits ** 2).mean()
        routing_info['router_z_loss'] = router_z_loss

        # Head utilization metrics
        routing_info['head_utilization'] = head_probs
        routing_info['routing_entropy'] = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1).mean()

        return routing_weights, selected_heads, routing_info


class MixtureOfHeads(nn.Module):
    """
    Mixture of Heads attention with dynamic head selection.

    Instead of using all attention heads, this module dynamically selects
    a subset of heads based on the input, allowing for adaptive computation.

    Architecture:
    - Router: Selects top-k heads per token
    - Attention: Standard multi-head attention with selected heads
    - Output: Weighted combination of selected head outputs

    Args:
        config: MoHConfig with all settings

    Example:
        >>> config = MoHConfig(num_heads=16, hidden_size=1024, num_active_heads=4)
        >>> moh = MixtureOfHeads(config)
        >>> q = k = v = torch.randn(2, 128, 1024)
        >>> output, aux_loss, info = moh(q, k, v, need_weights=False)
    """

    def __init__(self, config: MoHConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.num_active = config.num_active_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)

        # Projections for all heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

        # Router for head selection
        if config.use_learned_routing:
            self.router = HeadRouter(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_active=config.num_active_heads,
                jitter=config.router_jitter,
                routing_type=config.routing_type,
            )
        else:
            self.router = None

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Loss coefficients
        self.load_balance_loss_coef = config.load_balance_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef

        self._init_weights()

        logger.info(
            f"MixtureOfHeads: {config.num_heads} total heads, "
            f"{config.num_active_heads} active per token"
        )

    def _init_weights(self):
        """Initialize projections."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dynamic head selection.

        Args:
            query: [batch, seq_q, hidden_size]
            key: [batch, seq_k, hidden_size]
            value: [batch, seq_k, hidden_size]
            attention_mask: Optional mask [batch, 1, seq_q, seq_k]
            need_weights: Whether to return attention weights
            training: Whether in training mode

        Returns:
            output: [batch, seq_q, hidden_size]
            aux_loss: Scalar auxiliary loss
            info: Dictionary with routing metrics
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]
        device = query.device
        dtype = query.dtype

        # Project to all heads
        q = self.q_proj(query)  # [batch, seq_q, num_heads * head_dim]
        k = self.k_proj(key)    # [batch, seq_k, num_heads * head_dim]
        v = self.v_proj(value)  # [batch, seq_k, num_heads * head_dim]

        # Reshape for attention: [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Get routing weights and selected heads
        if self.router is not None:
            routing_weights, selected_heads, routing_info = self.router(query, training)
            # routing_weights: [batch, seq_q, num_active]
            # selected_heads: [batch, seq_q, num_active]
        else:
            # Random selection (for ablation)
            selected_heads = torch.randint(
                0, self.num_heads, (batch_size, seq_q, self.num_active),
                device=device,
            )
            routing_weights = torch.ones(
                batch_size, seq_q, self.num_active,
                device=device, dtype=dtype,
            ) / self.num_active
            routing_info = {}

        # Compute attention scores for all heads (we'll select later)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_q, seq_k]

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        # Compute attention output for all heads
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_q, head_dim]

        # Select only the active heads and weight them
        # This is the key MoH operation
        output = torch.zeros(batch_size, seq_q, self.hidden_size, device=device, dtype=dtype)

        # Gather selected head outputs
        # attn_output: [batch, num_heads, seq_q, head_dim]
        # selected_heads: [batch, seq_q, num_active]
        for b in range(batch_size):
            for s in range(seq_q):
                for a in range(self.num_active):
                    head_idx = selected_heads[b, s, a]
                    weight = routing_weights[b, s, a]
                    head_output = attn_output[b, head_idx, s]  # [head_dim]

                    # Compute start index in output (treating all heads as if they were concat)
                    # For MoH, we weight and sum the outputs
                    # Using o_proj implicitly handles the combination
                    start_idx = head_idx * self.head_dim
                    end_idx = start_idx + self.head_dim

                    # Accumulate weighted output
                    output[b, s, start_idx:end_idx] += weight * head_output

        # Apply output projection
        output = self.o_proj(output)

        # Compute auxiliary loss
        aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if 'load_balance_loss' in routing_info:
            aux_loss = aux_loss + self.load_balance_loss_coef * routing_info['load_balance_loss']
        if 'router_z_loss' in routing_info:
            aux_loss = aux_loss + self.router_z_loss_coef * routing_info['router_z_loss']

        return output, aux_loss, routing_info


class MoHAttentionLayer(nn.Module):
    """
    Complete attention layer with MoH integration.

    Drop-in replacement for standard attention layers with added
    dynamic head selection capability.

    Args:
        config: MoHConfig with settings
    """

    def __init__(self, config: MoHConfig):
        super().__init__()
        self.attention = MixtureOfHeads(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward with pre-norm and residual connection.

        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: Optional attention mask
            training: Whether in training mode

        Returns:
            output: [batch, seq, hidden_size]
            aux_loss: Auxiliary loss
            info: Routing metrics
        """
        # Pre-norm
        normed = self.layer_norm(hidden_states)

        # Self-attention with MoH
        attn_output, aux_loss, info = self.attention(
            normed, normed, normed,
            attention_mask=attention_mask,
            training=training,
        )

        # Dropout
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        # Residual
        output = hidden_states + attn_output

        return output, aux_loss, info


__all__ = [
    'MoHConfig',
    'HeadRouter',
    'MixtureOfHeads',
    'MoHAttentionLayer',
]
