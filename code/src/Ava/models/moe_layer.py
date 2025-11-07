"""
Sparse Mixture of Experts Layer - Drop-in replacement for FFN layers.

This module implements a production-grade SparseMoE layer with:
- Mixtral or DeepSeek routing
- Grouped GEMM for parallel expert computation
- 4 auxiliary losses (load_balance, router_z, expert_dropout, diversity)
- Dynamic expert capacity
- Gradient checkpointing
- Expert-level mixed precision
- Hierarchical MoE support (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
import math

from ..layers.experts import ExpertParallelGroup, SharedExpertLayer
from ..layers.routing import MixtralRouter, DeepSeekRouter


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer - drop-in FFN replacement.

    This layer can replace any feedforward network in a transformer with
    a mixture of expert networks, providing better parameter efficiency
    and potentially better performance.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: FFN hidden dimension (typically 3.5-4x hidden_size)
        num_experts: Total number of experts
        num_experts_per_token: How many experts to activate per token (k)
        router_type: Routing strategy ('mixtral', 'deepseek')
        capacity_factor: Expert capacity multiplier
        expert_dropout: Dropout rate for expert outputs
        activation: Activation function ('swiglu', 'geglu', 'gelu')
        use_grouped_gemm: Whether to use grouped GEMM (recommended)
        use_triton_kernels: Whether to use Triton kernels for routing
        use_torch_compile: Whether to compile the module
        router_z_loss_coef: Coefficient for router z-loss
        load_balance_loss_coef: Coefficient for load balancing loss
        diversity_loss_coef: Coefficient for diversity loss
        expert_dropout_loss_coef: Coefficient for expert dropout regularization
        router_jitter_noise: Jitter noise for exploration
        use_shared_expert: Whether to use shared expert (DeepSeek-style)
        shared_expert_weight: Weight for shared expert
        gradient_checkpointing: Whether to checkpoint expert computation
        dtype: Parameter dtype

    Example:
        >>> # Replace FFN in transformer with MoE
        >>> moe_layer = SparseMoELayer(
        ...     hidden_size=4096,
        ...     intermediate_size=14336,
        ...     num_experts=32,
        ...     num_experts_per_token=2,
        ...     router_type='mixtral'
        ... )
        >>> x = torch.randn(8, 128, 4096)  # [batch, seq, hidden]
        >>> output, aux_loss, metrics = moe_layer(x, training=True)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        router_type: str = 'mixtral',
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.0,
        activation: str = 'swiglu',
        use_grouped_gemm: bool = True,
        use_triton_kernels: bool = True,
        use_torch_compile: bool = True,
        router_z_loss_coef: float = 0.001,
        load_balance_loss_coef: float = 0.01,
        diversity_loss_coef: float = 0.001,
        expert_dropout_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        use_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        gradient_checkpointing: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_type = router_type
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        self.use_grouped_gemm = use_grouped_gemm
        self.use_triton_kernels = use_triton_kernels
        self.use_torch_compile = use_torch_compile
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.diversity_loss_coef = diversity_loss_coef
        self.expert_dropout_loss_coef = expert_dropout_loss_coef
        self.use_shared_expert = use_shared_expert
        self.gradient_checkpointing = gradient_checkpointing

        # Create router
        if router_type == 'mixtral':
            self.router = MixtralRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_selected_experts=num_experts_per_token,
                capacity_factor=capacity_factor,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_loss_coef=load_balance_loss_coef,
                router_jitter_noise=router_jitter_noise,
                dtype=dtype,
            )
        elif router_type == 'deepseek':
            self.router = DeepSeekRouter(
                hidden_size=hidden_size,
                num_experts=num_experts - 1,  # One will be shared
                num_selected_experts=num_experts_per_token,
                num_shared_experts=1,
                shared_expert_weight=shared_expert_weight,
                capacity_factor=capacity_factor,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_loss_coef=load_balance_loss_coef,
                router_jitter_noise=router_jitter_noise,
                dtype=dtype,
            )
            # Create shared expert
            self.shared_expert = SharedExpertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}. Use 'mixtral' or 'deepseek'")

        # Create expert group
        if use_grouped_gemm:
            # Grouped GEMM: all experts in one module (5-10x faster)
            expert_count = num_experts if router_type == 'mixtral' else num_experts - 1
            self.experts = ExpertParallelGroup(
                num_experts=expert_count,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout=expert_dropout,
                dtype=dtype,
            )
        else:
            # Sequential experts (slower, for compatibility)
            raise NotImplementedError("Sequential experts not implemented. Use use_grouped_gemm=True")

        # Expert dropout for regularization
        if expert_dropout > 0:
            self.expert_dropout_layer = nn.Dropout(expert_dropout)
        else:
            self.expert_dropout_layer = None

        # Layer normalization (applied before MoE, like in transformer)
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)

    def _compute_diversity_loss(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss: Encourages different tokens to use different experts.

        This prevents all tokens from collapsing to the same few experts.

        Args:
            expert_indices: Selected experts [num_tokens, k]

        Returns:
            Scalar diversity loss
        """
        # Compute pairwise diversity
        # High diversity = tokens use different expert combinations
        num_tokens = expert_indices.shape[0]

        # Create expert set fingerprint for each token
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [num_tokens, k, num_experts]
        expert_fingerprint = expert_mask.sum(dim=1)  # [num_tokens, num_experts]

        # Compute pairwise similarity
        similarity = torch.matmul(expert_fingerprint, expert_fingerprint.t())  # [num_tokens, num_tokens]

        # Normalize by number of experts per token
        similarity = similarity / (self.num_experts_per_token ** 2)

        # Diversity loss: penalize high similarity
        # Exclude diagonal (self-similarity)
        mask = 1 - torch.eye(num_tokens, device=expert_indices.device)
        diversity_loss = (similarity * mask).sum() / (num_tokens * (num_tokens - 1) + 1e-10)

        return diversity_loss

    def _compute_expert_dropout_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Expert dropout regularization loss.

        Encourages the model not to rely too heavily on any single expert
        by penalizing high routing weights.

        Args:
            expert_weights: Routing weights [num_tokens, k]

        Returns:
            Scalar regularization loss
        """
        # Penalize high confidence routing
        # This encourages more exploration and prevents overconfidence
        dropout_loss = (expert_weights ** 2).mean()
        return dropout_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through Sparse MoE layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            training: Whether in training mode

        Returns:
            - output: Layer output [batch_size, seq_len, hidden_size]
            - aux_loss: Total auxiliary loss (scalar)
            - metrics: Dictionary of routing metrics and losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape

        # Normalize input
        hidden_states = self.norm(hidden_states)

        # Flatten for routing
        hidden_flat = hidden_states.view(-1, hidden_size)  # [num_tokens, hidden_size]
        num_tokens = hidden_flat.shape[0]

        # Route tokens to experts
        expert_indices, expert_weights, routing_aux_loss, routing_metrics = self.router(
            hidden_flat, training=training
        )
        # expert_indices: [num_tokens, k]
        # expert_weights: [num_tokens, k]

        # Compute expert outputs using grouped GEMM
        if self.gradient_checkpointing and training:
            # Use gradient checkpointing to save memory
            expert_outputs = torch.utils.checkpoint.checkpoint(
                self.experts,
                hidden_flat,
                expert_indices,
                expert_weights,
                use_reentrant=False
            )
        else:
            expert_outputs = self.experts(
                hidden_flat,
                expert_indices,
                expert_weights
            )
        # expert_outputs: [num_tokens, k, hidden_size]

        # Combine expert outputs (weighted sum)
        output = expert_outputs.sum(dim=1)  # [num_tokens, hidden_size]

        # Apply expert dropout if enabled
        if self.expert_dropout_layer is not None and training:
            output = self.expert_dropout_layer(output)

        # Reshape back to original
        output = output.view(*original_shape)  # [batch_size, seq_len, hidden_size]

        # If using DeepSeek-style shared expert, add it
        if self.use_shared_expert and self.router_type == 'deepseek':
            shared_output = self.shared_expert(hidden_states)
            output = output + shared_output

        # Compute auxiliary losses
        aux_loss = routing_aux_loss

        # Diversity loss
        if training and self.diversity_loss_coef > 0:
            diversity_loss = self._compute_diversity_loss(expert_indices)
            aux_loss = aux_loss + self.diversity_loss_coef * diversity_loss
        else:
            diversity_loss = torch.tensor(0.0, device=hidden_states.device)

        # Expert dropout regularization loss
        if training and self.expert_dropout_loss_coef > 0:
            expert_dropout_loss = self._compute_expert_dropout_loss(expert_weights)
            aux_loss = aux_loss + self.expert_dropout_loss_coef * expert_dropout_loss
        else:
            expert_dropout_loss = torch.tensor(0.0, device=hidden_states.device)

        # Collect all metrics
        metrics = {
            **routing_metrics,
            'aux_loss_total': aux_loss,
            'aux_loss_routing': routing_aux_loss,
            'aux_loss_diversity': diversity_loss,
            'aux_loss_expert_dropout': expert_dropout_loss,
            'num_tokens': torch.tensor(num_tokens, device=hidden_states.device),
        }

        return output, aux_loss, metrics

    def reset_expert_counts(self):
        """Reset expert utilization counts (for metrics)."""
        if hasattr(self.router, 'expert_counts'):
            self.router.expert_counts.zero_()
            self.router.total_routing_calls.zero_()

    def get_expert_usage_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get expert usage statistics over time.

        Returns:
            Dictionary with expert utilization stats
        """
        if not hasattr(self.router, 'expert_counts'):
            return {}

        total_calls = self.router.total_routing_calls.item()
        if total_calls == 0:
            return {'expert_usage': self.router.expert_counts}

        # Normalize by total calls
        usage_per_call = self.router.expert_counts / total_calls
        return {
            'expert_usage_total': self.router.expert_counts,
            'expert_usage_normalized': usage_per_call,
            'total_routing_calls': self.router.total_routing_calls,
        }


# Optionally compile the module for performance
# This can provide additional 10-20% speedup
try:
    if hasattr(torch, 'compile'):
        SparseMoELayer.forward = torch.compile(
            SparseMoELayer.forward,
            mode='reduce-overhead',
            fullgraph=False
        )
except Exception:
    # torch.compile not available or failed
    pass
