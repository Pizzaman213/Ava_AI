"""
Hierarchical routing for clustered MoE.

Implements two-level routing:
Level 1: Route to expert cluster (coarse-grained)
Level 2: Route to expert within cluster (fine-grained)

This reduces routing complexity and enables cluster-based loading,
where only active cluster(s) need to be in GPU memory.

Memory savings: 70-90% depending on number of clusters and cluster size.

References:
- Mixture-of-Clustered-Experts (2024)
- HC-SMoE: Hierarchically Clustered Sparse MoE (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class HierarchicalRouter(nn.Module):
    """
    Two-level hierarchical router for clustered experts.

    Level 1: Selects which expert cluster(s) to use
    Level 2: Selects which experts within the cluster

    This architecture:
    - Reduces routing search space by factor of num_clusters
    - Enables loading only active cluster to GPU
    - Provides better load balancing across clusters

    Args:
        hidden_size: Input dimension
        num_experts: Total number of experts
        num_clusters: Number of expert clusters
        num_selected_experts: Number of experts to select per token
        num_selected_clusters: Number of clusters to select (typically 1-2)
        capacity_factor: Expert capacity factor
        router_z_loss_coef: Router z-loss coefficient
        load_balance_loss_coef: Load balancing loss coefficient
        router_jitter_noise: Jitter noise for exploration
        dtype: Parameter dtype

    Example:
        >>> router = HierarchicalRouter(
        ...     hidden_size=4096,
        ...     num_experts=32,
        ...     num_clusters=4,  # 4 clusters of 8 experts each
        ...     num_selected_experts=2
        ... )
        >>> # Routes to 1-2 clusters, then 2 experts within each cluster
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_clusters: int = 4,
        num_selected_experts: int = 2,
        num_selected_clusters: int = 1,
        capacity_factor: float = 1.25,
        router_z_loss_coef: float = 0.001,
        load_balance_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert num_experts % num_clusters == 0, \
            f"num_experts ({num_experts}) must be divisible by num_clusters ({num_clusters})"

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_clusters = num_clusters
        self.experts_per_cluster = num_experts // num_clusters
        self.num_selected_experts = num_selected_experts
        self.num_selected_clusters = num_selected_clusters
        self.capacity_factor = capacity_factor
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # Level 1: Cluster routing (lightweight)
        self.cluster_gate = nn.Linear(hidden_size, num_clusters, dtype=dtype)

        # Level 2: Per-cluster expert routing
        # We use separate gates for each cluster for flexibility
        self.expert_gates = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_cluster, dtype=dtype)
            for _ in range(num_clusters)
        ])

        # Cluster assignments (can be learned or fixed)
        # Maps expert_id -> cluster_id
        self.register_buffer(
            'cluster_assignments',
            torch.arange(num_experts) // self.experts_per_cluster
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize routing weights with small random values."""
        # Cluster gate
        nn.init.normal_(self.cluster_gate.weight, mean=0.0, std=0.01)
        if self.cluster_gate.bias is not None:
            nn.init.zeros_(self.cluster_gate.bias)

        # Expert gates
        for gate in self.expert_gates:
            nn.init.normal_(gate.weight, mean=0.0, std=0.01)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def _compute_load_balance_loss(
        self,
        cluster_probs: torch.Tensor,
        expert_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss for both levels.

        Args:
            cluster_probs: Cluster probabilities [num_tokens, num_clusters]
            expert_probs: Expert probabilities [num_tokens, experts_per_cluster]
            expert_indices: Selected expert indices [num_tokens, k]

        Returns:
            Load balance loss
        """
        # Cluster-level balance
        cluster_mean_prob = cluster_probs.mean(dim=0)  # [num_clusters]
        cluster_frac = torch.zeros_like(cluster_mean_prob)
        for i in range(self.num_clusters):
            # Count tokens routed to this cluster
            cluster_mask = (expert_indices // self.experts_per_cluster == i).any(dim=-1)
            cluster_frac[i] = cluster_mask.float().mean()

        cluster_balance_loss = (
            self.num_clusters * (cluster_mean_prob * cluster_frac).sum()
        )

        # Expert-level balance (within clusters)
        expert_mean_prob = expert_probs.mean(dim=0)  # [experts_per_cluster]
        expert_frac = torch.zeros_like(expert_mean_prob)
        for i in range(self.experts_per_cluster):
            expert_mask = ((expert_indices % self.experts_per_cluster) == i).any(dim=-1)
            expert_frac[i] = expert_mask.float().mean()

        expert_balance_loss = (
            self.experts_per_cluster * (expert_mean_prob * expert_frac).sum()
        )

        # Combined loss
        return (cluster_balance_loss + expert_balance_loss) / 2

    def _compute_router_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Router z-loss to encourage confidence.

        Args:
            logits: Router logits

        Returns:
            Z-loss
        """
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        return z_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Hierarchical routing forward pass.

        Args:
            hidden_states: Input [batch, seq, hidden] or [tokens, hidden]
            training: Whether in training mode

        Returns:
            expert_indices: Selected expert IDs [num_tokens, k]
            expert_weights: Routing weights [num_tokens, k]
            aux_loss: Auxiliary loss (z-loss + load balance)
            metrics: Routing metrics dict
        """
        # Flatten if needed
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
        else:
            batch_size = hidden_states.shape[0]
            seq_len = 1

        num_tokens = hidden_states.shape[0]

        # Add jitter noise during training
        if training and self.router_jitter_noise > 0:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * self.router_jitter_noise

        # ========================================
        # Level 1: Cluster Selection
        # ========================================

        cluster_logits = self.cluster_gate(hidden_states)  # [num_tokens, num_clusters]
        cluster_probs = F.softmax(cluster_logits, dim=-1)

        # Select top-K clusters
        top_cluster_probs, top_cluster_indices = torch.topk(
            cluster_probs, k=min(self.num_selected_clusters, self.num_clusters), dim=-1
        )  # [num_tokens, num_selected_clusters]

        # ========================================
        # Level 2: Expert Selection within Clusters
        # ========================================

        all_expert_indices = []
        all_expert_weights = []
        all_expert_probs_for_loss = []

        for cluster_idx in range(self.num_selected_clusters):
            # Get cluster ID for each token
            cluster_id = top_cluster_indices[:, cluster_idx]  # [num_tokens]

            # Compute expert logits for selected cluster
            # We need to handle dynamic cluster selection efficiently
            expert_logits_list = []
            for token_idx in range(num_tokens):
                cid = cluster_id[token_idx].item()
                expert_logits = self.expert_gates[cid](hidden_states[token_idx:token_idx+1])
                expert_logits_list.append(expert_logits)

            expert_logits = torch.cat(expert_logits_list, dim=0)  # [num_tokens, experts_per_cluster]
            expert_probs = F.softmax(expert_logits, dim=-1)

            # Select top-k experts within cluster
            k_per_cluster = max(1, self.num_selected_experts // self.num_selected_clusters)
            top_expert_probs, top_expert_local_indices = torch.topk(
                expert_probs, k=min(k_per_cluster, self.experts_per_cluster), dim=-1
            )  # [num_tokens, k_per_cluster]

            # Convert local indices to global expert IDs
            # global_id = cluster_id * experts_per_cluster + local_id
            global_expert_indices = (
                cluster_id.unsqueeze(1) * self.experts_per_cluster +
                top_expert_local_indices
            )  # [num_tokens, k_per_cluster]

            # Weight by cluster probability
            weighted_probs = top_expert_probs * top_cluster_probs[:, cluster_idx:cluster_idx+1]

            all_expert_indices.append(global_expert_indices)
            all_expert_weights.append(weighted_probs)
            all_expert_probs_for_loss.append(expert_probs)

        # Combine experts from all selected clusters
        expert_indices = torch.cat(all_expert_indices, dim=1)  # [num_tokens, k]
        expert_weights = torch.cat(all_expert_weights, dim=1)  # [num_tokens, k]

        # Normalize weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Take top-k if we got more than needed
        if expert_indices.shape[1] > self.num_selected_experts:
            top_k_probs, top_k_idx = torch.topk(expert_weights, k=self.num_selected_experts, dim=-1)
            expert_indices = torch.gather(expert_indices, 1, top_k_idx)
            expert_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # ========================================
        # Auxiliary Losses
        # ========================================

        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        if training:
            # Router z-loss
            cluster_z_loss = self._compute_router_z_loss(cluster_logits)
            aux_loss += self.router_z_loss_coef * cluster_z_loss

            # Load balance loss
            avg_expert_probs = torch.stack(all_expert_probs_for_loss).mean(dim=0)
            balance_loss = self._compute_load_balance_loss(
                cluster_probs, avg_expert_probs, expert_indices
            )
            aux_loss += self.load_balance_loss_coef * balance_loss

        # ========================================
        # Metrics
        # ========================================

        metrics = {
            'active_clusters': len(torch.unique(expert_indices // self.experts_per_cluster)),
            'active_experts': len(torch.unique(expert_indices)),
            'cluster_entropy': -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=-1).mean(),
        }

        return expert_indices, expert_weights, aux_loss, metrics

    def set_cluster_assignments(self, assignments: torch.Tensor):
        """
        Set fixed cluster assignments for experts.

        Args:
            assignments: Tensor of shape [num_experts] with cluster IDs
        """
        assert assignments.shape[0] == self.num_experts
        assert assignments.max() < self.num_clusters
        self.cluster_assignments = assignments.to(self.cluster_assignments.device)
