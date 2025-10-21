"""
Expert layers for MoE++ architecture with advanced routing and balancing.

This module implements the core expert components including:
- ExpertBalancer: Advanced load balancing strategies (Sinkhorn, capacity-based)
- SparseExpert: Conditional computation expert with sparsity masks
- Expert selection and routing mechanisms

These components enable efficient mixture-of-experts training with improved
load balancing and reduced computational overhead.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Optional, Tuple


class ExpertBalancer:
    """
    Advanced expert balancing for MoE++ using various strategies.

    This class implements multiple load balancing strategies to ensure
    efficient expert utilization and prevent mode collapse where all
    tokens get routed to the same few experts.

    Args:
        num_experts (int): Number of experts in the MoE layer
        balance_strategy (str): Strategy for balancing expert loads
            - 'sinkhorn': Uses Sinkhorn-Knopp algorithm for doubly stochastic normalization
            - 'capacity': Enforces hard capacity constraints per expert
            - 'standard': Standard softmax routing without balancing

    Attributes:
        token_count_history (torch.Tensor): Historical token counts per expert
        expert_utilization (torch.Tensor): Utilization statistics per expert

    Example:
        >>> balancer = ExpertBalancer(num_experts=8, balance_strategy='sinkhorn')
        >>> router_logits = torch.randn(2, 64, 8)  # [batch, tokens, experts]
        >>> weights, indices = balancer.compute_balanced_routing(router_logits, k=2)
        >>> # weights: [2, 64, 2], indices: [2, 64, 2]
    """

    def __init__(self, num_experts: int, balance_strategy: str = 'sinkhorn'):
        self.num_experts = num_experts
        self.balance_strategy = balance_strategy
        self.token_count_history = torch.zeros(num_experts)
        self.expert_utilization = torch.zeros(num_experts)

    def compute_balanced_routing(self, router_logits: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute balanced routing weights and indices for top-k expert selection.

        Args:
            router_logits (torch.Tensor): Raw routing logits of shape [batch_size, num_tokens, num_experts]
            k (int): Number of experts to select per token

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - top_k_weights: Normalized routing weights for selected experts [batch_size, num_tokens, k]
                - top_k_indices: Indices of selected experts [batch_size, num_tokens, k]
        """
        batch_size, num_tokens, _ = router_logits.shape

        if self.balance_strategy == 'sinkhorn':
            # Apply Sinkhorn-Knopp algorithm for balanced assignment
            routing_weights = F.softmax(router_logits, dim=-1)

            # Iterative balancing - alternates between normalizing rows and columns
            # to achieve doubly stochastic matrix (rows and columns sum to 1)
            for _ in range(3):  # 3 iterations typically sufficient for convergence
                # Normalize rows: each token's weights sum to 1
                routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
                # Normalize columns: balance expert load
                routing_weights = routing_weights / (routing_weights.sum(dim=-2, keepdim=True) + 1e-8)

            # Select top-k experts from balanced weights
            top_k_weights, top_k_indices = torch.topk(routing_weights, k, dim=-1)
            return top_k_weights, top_k_indices

        elif self.balance_strategy == 'capacity':
            # Capacity-constrained routing with hard limits per expert
            routing_weights = F.softmax(router_logits, dim=-1)
            capacity = num_tokens // self.num_experts * 2  # 2x capacity buffer for flexibility

            # Track expert load and apply capacity constraints
            expert_load = torch.zeros(self.num_experts, device=router_logits.device)
            selected_weights = torch.zeros_like(routing_weights[:, :, :k])
            selected_indices = torch.zeros(batch_size, num_tokens, k, dtype=torch.long, device=router_logits.device)

            # Greedy assignment respecting capacity constraints
            flat_weights = routing_weights.view(-1, self.num_experts)
            sorted_indices = torch.argsort(flat_weights, dim=1, descending=True)

            for i in range(k):
                expert_choices = sorted_indices[:, i]
                # Check capacity and assign only if under limit
                mask = expert_load[expert_choices] < capacity
                selected_indices.view(-1, k)[:, i] = torch.where(
                    mask, expert_choices, torch.zeros_like(expert_choices)
                )
                expert_load.scatter_add_(0, expert_choices[mask],
                                       torch.ones_like(expert_choices[mask], dtype=torch.float))

            return selected_weights, selected_indices

        else:
            # Standard softmax routing without balancing
            routing_weights = F.softmax(router_logits, dim=-1)
            return torch.topk(routing_weights, k, dim=-1)


class SparseExpert(nn.Module):
    """
    Sparse expert with conditional computation for efficiency.

    This expert implements conditional computation where only a subset of
    activations are processed based on importance scores, reducing compute
    while maintaining model capacity.

    Args:
        input_size (int): Dimension of input features
        hidden_size (int): Dimension of hidden layer
        output_size (int): Dimension of output features
        sparsity_level (float): Fraction of activations to zero out (0.0 to 1.0)

    Architecture:
        - Main network: Linear -> GELU -> Dropout -> Linear
        - Mask generator: Produces importance scores for sparsity
        - Computation gate: Binary decision for conditional computation

    Example:
        >>> expert = SparseExpert(768, 3072, 768, sparsity_level=0.5)
        >>> x = torch.randn(32, 768)  # [batch, features]
        >>> output, mask, gate = expert(x)
        >>> # output: [32, 768], mask: sparsity mask, gate: computation decisions
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, sparsity_level: float = 0.5, use_true_sparsity: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sparsity_level = sparsity_level
        # FIXED: Add option for true conditional computation (saves compute but not compile-friendly)
        self.use_true_sparsity = use_true_sparsity

        # Main expert network - standard FFN architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),  # GELU activation for better gradient flow
            nn.Dropout(0.1),  # Regularization
            nn.Linear(hidden_size, output_size)
        )

        # Sparsity mask generator - learns which activations are important
        self.mask_generator = nn.Sequential(
            nn.Linear(input_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()  # Output importance scores in [0, 1]
        )

        # Conditional computation gate - binary decision for each token
        self.computation_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()  # Output probability of computation
        )

    def forward(self, x: torch.Tensor, compute_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with conditional sparse computation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size] or [batch_size, input_size]
            compute_mask (torch.Tensor, optional): Pre-computed sparsity mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - output: Expert output of shape matching input
                - compute_mask: Binary mask indicating which activations were computed
                - gate_score: Confidence scores for computation decisions
        """
        # Handle both 2D and 3D input tensors
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            flat_x = x.view(-1, self.input_size)
            needs_reshape = True
        else:
            # Already flat (2D)
            flat_x = x
            batch_size = x.shape[0]
            seq_len = 1
            needs_reshape = False

        # Generate sparsity mask if not provided
        if compute_mask is None:
            activation_importance = self.mask_generator(flat_x)
            # Select top (1-sparsity_level) fraction of activations
            threshold = torch.kthvalue(activation_importance.view(-1),
                                     int(activation_importance.numel() * self.sparsity_level)).values
            compute_mask = activation_importance > threshold

        # Apply conditional computation gate
        gate_score = self.computation_gate(flat_x).squeeze(-1)

        # Type assertion to ensure compute_mask is not None
        assert compute_mask is not None
        should_compute = gate_score > 0.5  # Binary decision threshold

        # Compute mask: combine gate decision with activation sparsity
        compute_indices = should_compute & compute_mask.any(dim=1)

        # FIXED: Two computation modes - true sparsity (saves compute) vs compile-friendly (masks output)
        is_scripting = torch.jit.is_scripting() if hasattr(torch.jit, 'is_scripting') else False  # type: ignore[attr-defined]
        if self.use_true_sparsity and not is_scripting:
            # True conditional computation: only process selected tokens
            # This saves actual computation but uses dynamic control flow
            compute_mask_bool = compute_indices
            num_compute = compute_mask_bool.sum().item()

            if num_compute > 0:
                # Only compute for selected tokens
                selected_x = flat_x[compute_mask_bool]
                selected_output = self.network(selected_x)

                # Scatter back to full output
                output = torch.zeros(flat_x.shape[0], self.output_size, device=flat_x.device, dtype=flat_x.dtype)
                output[compute_mask_bool] = selected_output
            else:
                # Nothing to compute
                output = torch.zeros(flat_x.shape[0], self.output_size, device=flat_x.device, dtype=flat_x.dtype)
        else:
            # Compile-friendly mode: compute everything, then mask
            # This avoids dynamic control flow and boolean indexing
            compute_mask_float = compute_indices.float().unsqueeze(-1)  # Shape: [N, 1]
            expert_output = self.network(flat_x)

            # Apply mask: zero out outputs where we shouldn't compute
            # This is compile-friendly as it uses element-wise multiplication
            output = expert_output * compute_mask_float

        # Reshape output back to original dimensions if needed
        if needs_reshape:
            return output.view(batch_size, seq_len, self.output_size), compute_mask, gate_score
        else:
            return output, compute_mask, gate_score