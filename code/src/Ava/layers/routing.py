"""
Expert routing and selection for MoE++ architecture.

This module implements intelligent routing mechanisms including:
- ExpertSelector: Dynamic expert selection with confidence scoring
- MoEPlusPlusLayer: Complete MoE layer with load balancing and auxiliary losses

The routing system uses confidence-based dynamic selection where high-confidence
tokens use fewer experts while uncertain tokens engage more experts for better accuracy.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Dict, Tuple, Any

from .experts import ExpertBalancer, SparseExpert


class SwitchTransformerRouting(nn.Module):
    """
    Switch Transformer routing implementation with capacity factors and load balancing.

    This implements the routing mechanism from the Switch Transformer paper
    with improvements for better load balancing and efficiency.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        dropout_rate: float = 0.1,
        jitter_eps: float = 0.1,
        router_bias: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.dropout_rate = dropout_rate
        self.jitter_eps = jitter_eps

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts, bias=router_bias)
        self.dropout = nn.Dropout(dropout_rate)

        # Auxiliary loss tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Forward pass through Switch Transformer routing.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            training: Whether in training mode

        Returns:
            Tuple of (dispatch_tensor, combine_tensor, expert_capacity, aux_info)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Flatten for routing
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [num_tokens, hidden_dim]

        # Add jitter during training for better load balancing
        if training and self.jitter_eps > 0:
            noise = torch.empty_like(hidden_flat).uniform_(-self.jitter_eps, self.jitter_eps)
            hidden_flat = hidden_flat + noise

        # Compute routing logits
        router_logits = self.router(hidden_flat)  # [num_tokens, num_experts]

        # Apply routing
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top expert for each token
        expert_gate, expert_index = torch.max(router_probs, dim=-1)

        # FIXED: Compute expert capacity with adaptive adjustment for small batches
        # Use a dynamic capacity factor that increases for small batches to reduce token dropping
        min_capacity_per_expert = 4  # Minimum tokens per expert
        base_capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        if base_capacity < min_capacity_per_expert:
            # For small batches, increase capacity factor dynamically
            adjusted_capacity_factor = (min_capacity_per_expert * self.num_experts) / max(num_tokens, 1)
            expert_capacity = max(min_capacity_per_expert, int(adjusted_capacity_factor * num_tokens / self.num_experts))
            if training and num_tokens > 0:
                import warnings
                warnings.warn(
                    f"Small batch detected: Adjusting expert capacity from {base_capacity} to {expert_capacity} "
                    f"for {num_tokens} tokens and {self.num_experts} experts. "
                    f"Consider increasing batch size for better performance.",
                    UserWarning
                )
        else:
            expert_capacity = base_capacity

        # Create dispatch and combine tensors
        dispatch_tensor = torch.zeros(
            num_tokens, self.num_experts, expert_capacity,
            dtype=hidden_flat.dtype, device=hidden_flat.device
        )
        combine_tensor = torch.zeros(
            num_tokens, self.num_experts, expert_capacity,
            dtype=hidden_flat.dtype, device=hidden_flat.device
        )

        # Track expert assignments
        expert_usage = torch.zeros(self.num_experts, device=hidden_flat.device)

        # Vectorized routing instead of double loop (much faster!)
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_index == expert_id)
            selected_tokens = torch.where(expert_mask)[0]

            # Respect capacity constraints
            if len(selected_tokens) > expert_capacity:
                # Select top tokens by gate value
                selected_gates = expert_gate[selected_tokens]
                _, top_indices = torch.topk(selected_gates, expert_capacity)
                selected_tokens = selected_tokens[top_indices]

            if len(selected_tokens) > 0:
                # Vectorized assignment (no inner loop!)
                num_selected = len(selected_tokens)
                positions = torch.arange(num_selected, device=hidden_flat.device)
                dispatch_tensor[selected_tokens, expert_id, positions] = 1.0
                combine_tensor[selected_tokens, expert_id, positions] = expert_gate[selected_tokens]
                expert_usage[expert_id] = num_selected

        # Update statistics for load balancing loss
        if training:
            self.expert_counts += expert_usage.detach()
            self.total_tokens += num_tokens

        # Calculate tokens dropped
        tokens_dropped = max(0, num_tokens - dispatch_tensor.sum().item())

        # Warn if significant tokens dropped
        if tokens_dropped > num_tokens * 0.1:  # More than 10% dropped
            import warnings
            warnings.warn(
                f"High token drop rate: {tokens_dropped}/{num_tokens} ({100*tokens_dropped/num_tokens:.1f}%) tokens dropped. "
                f"Consider increasing capacity_factor or num_experts_per_token."
            )

        # Auxiliary information
        aux_info = {
            'router_probs': router_probs,
            'expert_usage': expert_usage,
            'load_balancing_loss': self._compute_load_balancing_loss(router_probs),
            'router_z_loss': torch.mean(router_logits ** 2),
            'tokens_dropped': tokens_dropped
        }

        return dispatch_tensor, combine_tensor, expert_capacity, aux_info

    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Probability that each expert is selected
        prob_per_expert = router_probs.mean(dim=0)  # [num_experts]

        # Fraction of tokens routed to each expert
        routing_per_expert = router_probs.argmax(dim=-1)
        usage_per_expert = torch.bincount(
            routing_per_expert, minlength=self.num_experts
        ).float() / router_probs.shape[0]

        # Load balancing loss encourages uniform distribution
        load_loss = self.num_experts * torch.sum(prob_per_expert * usage_per_expert)
        return load_loss


class GSERouting(nn.Module):
    """
    GShard Expert routing with top-2 gating and capacity factors.

    This implements the routing from GShard with top-2 expert selection.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float = 2.0,
        second_expert_policy: str = "random",
        normalize_gate: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.second_expert_policy = second_expert_policy
        self.normalize_gate = normalize_gate

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through GShard routing.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (routing_weights, expert_indices, aux_info)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        router_logits = self.router(hidden_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-2 experts
        top2_probs, top2_indices = torch.topk(router_probs, k=2, dim=-1)

        # Normalize top-2 probabilities
        if self.normalize_gate:
            top2_probs = top2_probs / top2_probs.sum(dim=-1, keepdim=True)

        # Apply capacity constraints
        expert_capacity = int(self.capacity_factor * hidden_flat.shape[0] / self.num_experts)

        # Create routing tensors
        routing_weights = torch.zeros_like(router_probs)
        for i in range(2):  # For top-2
            expert_id = top2_indices[:, i]
            # Simple capacity enforcement - could be more sophisticated
            routing_weights.scatter_(1, expert_id.unsqueeze(1), top2_probs[:, i].unsqueeze(1))

        aux_info = {
            'router_probs': router_probs,
            'top2_probs': top2_probs,
            'top2_indices': top2_indices,
            'expert_capacity': expert_capacity
        }

        return routing_weights, top2_indices, aux_info


class HashingExpertRouting(nn.Module):
    """
    Hash-based expert routing for deterministic load balancing.

    This routing method uses hashing to deterministically assign tokens
    to experts, ensuring perfect load balancing.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_hash_functions: int = 4,
        hash_type: str = "learned"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_hash_functions = num_hash_functions
        self.hash_type = hash_type

        if hash_type == "learned":
            # Learnable hash functions
            self.hash_functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1)
                ) for _ in range(num_hash_functions)
            ])
        else:
            # Fixed random hash functions
            self.register_buffer('hash_weights', torch.randn(num_hash_functions, hidden_dim))
            self.hash_weights: torch.Tensor  # Type hint for registered buffer

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through hash-based routing.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (routing_weights, expert_indices, aux_info)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_flat.shape[0]

        if self.hash_type == "learned":
            # Compute learned hash values
            hash_values = []
            for hash_fn in self.hash_functions:
                hash_val = hash_fn(hidden_flat).squeeze(-1)  # [num_tokens]
                hash_values.append(hash_val)
            hash_tensor = torch.stack(hash_values, dim=1)  # [num_tokens, num_hash_functions]
        else:
            # Compute fixed hash values
            hash_tensor = torch.matmul(hidden_flat, self.hash_weights.T)  # [num_tokens, num_hash_functions]

        # Convert hash values to expert assignments
        expert_assignments = torch.remainder(
            torch.sum(hash_tensor, dim=1).long(), self.num_experts
        )

        # Create one-hot routing weights
        routing_weights = F.one_hot(expert_assignments, self.num_experts).float()

        aux_info = {
            'hash_values': hash_tensor,
            'expert_assignments': expert_assignments,
            'load_balance': torch.bincount(expert_assignments, minlength=self.num_experts).float()
        }

        return routing_weights, expert_assignments.unsqueeze(1), aux_info


class StochasticExpertRouting(nn.Module):
    """
    Stochastic expert routing with exploration-exploitation trade-off.

    This routing method balances between exploitation (using best experts)
    and exploration (trying different experts) for better learning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        exploration_rate: float = 0.1,
        temperature: float = 1.0,
        use_gumbel: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts)

        # Exploration policy
        self.exploration_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through stochastic routing.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            training: Whether in training mode

        Returns:
            Tuple of (routing_weights, expert_indices, aux_info)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing logits
        router_logits = self.router(hidden_flat)  # [num_tokens, num_experts]

        # Compute exploration probabilities
        exploration_probs = self.exploration_network(hidden_flat).squeeze(-1)  # [num_tokens]

        if training:
            if self.use_gumbel:
                # Gumbel-Softmax for differentiable sampling
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-8) + 1e-8)
                router_logits = router_logits + gumbel_noise

            # Sample routing decisions
            routing_probs = F.softmax(router_logits / self.temperature, dim=-1)

            # Decide between exploitation and exploration
            exploit_mask = torch.bernoulli(1 - exploration_probs * self.exploration_rate)

            # Exploitation: use learned routing
            exploit_indices = torch.multinomial(routing_probs, 1).squeeze(-1)

            # Exploration: random assignment
            explore_indices = torch.randint(0, self.num_experts, (hidden_flat.shape[0],), device=hidden_flat.device)

            # Combine based on exploration decision
            expert_indices = torch.where(exploit_mask.bool(), exploit_indices, explore_indices)

        else:
            # During inference, use deterministic routing (argmax)
            expert_indices = torch.argmax(router_logits, dim=-1)
            routing_probs = F.softmax(router_logits, dim=-1)

        # Create routing weights
        routing_weights = F.one_hot(expert_indices, self.num_experts).float()

        # If using soft routing, use the probabilities
        if self.use_gumbel and training:
            routing_weights = routing_probs

        aux_info = {
            'router_logits': router_logits,
            'exploration_probs': exploration_probs,
            'expert_indices': expert_indices,
            'routing_entropy': -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1).mean()
        }

        return routing_weights, expert_indices.unsqueeze(1), aux_info


class ExpertSelector(nn.Module):
    """
    Intelligent expert selection with confidence-based dynamic routing.

    This module implements a confidence-aware routing mechanism that dynamically
    adjusts the number of experts used per token based on the model's confidence.
    High-confidence predictions use fewer experts (efficiency), while uncertain
    predictions engage more experts (accuracy).

    Args:
        hidden_size (int): Dimension of hidden states
        num_experts (int): Total number of experts available
        min_experts (int): Minimum number of experts per token
        max_experts (int): Maximum number of experts per token

    Attributes:
        confidence_net: Network that estimates confidence for each token
        expert_specialization: Learnable expert embedding vectors
        router: Main routing network that produces expert scores

    Example:
        >>> selector = ExpertSelector(hidden_size=768, num_experts=8, min_experts=1, max_experts=4)
        >>> hidden_states = torch.randn(2, 64, 768)  # [batch, seq_len, hidden]
        >>> weights, indices, confidence, logits = selector(hidden_states)
        >>> # weights: [2, 64, 4] - routing weights for top experts
        >>> # indices: [2, 64, 4] - indices of selected experts
    """

    def __init__(self, hidden_size: int, num_experts: int, min_experts: int = 1, max_experts: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts

        # Confidence scoring network - estimates uncertainty
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )

        # Expert specialization indicators - learnable expert embeddings
        self.expert_specialization = nn.Parameter(torch.randn(num_experts, hidden_size))

        # Main routing network with bias for better initialization
        self.router = nn.Linear(hidden_size, num_experts, bias=True)

    def forward(self, hidden_states: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expert routing with confidence-based dynamic selection.

        Args:
            hidden_states (torch.Tensor): Input of shape [batch_size, seq_len, hidden_size]
            temperature (float): Temperature for softmax, lower = more peaked distribution

        Returns:
            Tuple containing:
                - selected_weights: Weights for selected experts [batch_size, seq_len, max_experts]
                - selected_indices: Indices of selected experts [batch_size, seq_len, max_experts]
                - confidence_scores: Confidence scores per token [batch_size * seq_len]
                - router_logits: Raw routing logits [batch_size * seq_len, num_experts]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # Compute routing logits with temperature scaling
        router_logits = self.router(flat_hidden) / temperature

        # Compute confidence scores for each token
        confidence_scores = self.confidence_net(flat_hidden).squeeze(-1)

        # Dynamic k: high confidence -> few experts, low confidence -> many experts
        # This implements adaptive computation based on model uncertainty
        dynamic_k = self.min_experts + (self.max_experts - self.min_experts) * (1 - confidence_scores)
        dynamic_k = torch.clamp(dynamic_k.round().long(), self.min_experts, self.max_experts)

        # Convert logits to probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select variable number of experts per token
        max_k = self.max_experts
        selected_weights = torch.zeros(batch_size * seq_len, max_k, device=hidden_states.device)
        selected_indices = torch.zeros(batch_size * seq_len, max_k, dtype=torch.long, device=hidden_states.device)

        for k in range(max_k):
            # Mask for tokens that need k+1 or more experts
            use_expert_mask = (dynamic_k > k)
            if use_expert_mask.any():
                sorted_probs, sorted_indices = torch.sort(routing_probs[use_expert_mask],
                                                        descending=True, dim=1)
                # Check bounds to avoid indexing errors
                if k < sorted_probs.shape[1]:
                    selected_weights[use_expert_mask, k] = sorted_probs[:, k]
                    selected_indices[use_expert_mask, k] = sorted_indices[:, k]

        # Reshape back to batch dimensions
        selected_weights = selected_weights.view(batch_size, seq_len, max_k)
        selected_indices = selected_indices.view(batch_size, seq_len, max_k)

        return selected_weights, selected_indices, confidence_scores, router_logits


class MoEPlusPlusLayer(nn.Module):
    """
    Enhanced Mixture of Experts layer with advanced routing and load balancing.

    This layer implements the MoE++ architecture with:
    - Dynamic expert selection based on confidence
    - Load balancing with Sinkhorn normalization
    - Sparse experts with conditional computation
    - Auxiliary losses for training stability

    Args:
        config: Configuration object with model parameters

    Key Features:
        - Balanced routing to prevent expert collapse
        - Diversity loss to encourage expert specialization
        - Confidence-based dynamic computation
        - Sparse activation patterns for efficiency

    Example:
        >>> config = EnhancedMoEConfig(hidden_size=768, num_experts=8, ...)
        >>> moe_layer = MoEPlusPlusLayer(config)
        >>> hidden_states = torch.randn(2, 64, 768)
        >>> output, aux_losses, aux_info = moe_layer(hidden_states)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if hasattr(config, 'intermediate_size') else config.hidden_size * 4
        self.balance_loss_weight = getattr(config, 'balance_loss_weight', 0.01)

        # FIXED: Support for gradient accumulation-aware loss scaling
        # Auxiliary losses should be scaled by 1/gradient_accumulation_steps to prevent over-regularization
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

        # Advanced expert selector with confidence scoring
        self.expert_selector = ExpertSelector(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            min_experts=getattr(config, 'min_experts_per_tok', 1),
            max_experts=getattr(config, 'max_experts_per_tok', 4)
        )

        # Expert balancer for load distribution
        self.expert_balancer = ExpertBalancer(
            num_experts=self.num_experts,
            balance_strategy=getattr(config, 'balance_strategy', 'sinkhorn')
        )

        # Create sparse experts
        self.experts = nn.ModuleList([
            SparseExpert(
                input_size=self.hidden_size,
                hidden_size=self.intermediate_size,
                output_size=self.hidden_size,
                sparsity_level=getattr(config, 'expert_sparsity', 0.3)
            ) for _ in range(self.num_experts)
        ])

        # Expert diversity enhancement
        self.expert_diversity_weight = getattr(config, 'expert_diversity_weight', 0.01)
        self.diversity_projection = nn.Linear(self.hidden_size, 64)

        # Router dropout for regularization
        self.router_dropout = nn.Dropout(getattr(config, 'router_dropout', 0.0))

        # FIXED: Add expert dropout and routing jitter for robustness
        # Expert dropout randomly masks experts during training to prevent overfitting
        self.expert_dropout = getattr(config, 'expert_dropout', 0.0)
        # Routing jitter adds noise to logits for exploration
        self.routing_jitter = getattr(config, 'routing_jitter', 0.0)

    def forward(self, hidden_states: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass through the MoE++ layer.

        Args:
            hidden_states (torch.Tensor): Input of shape [batch_size, seq_len, hidden_size]
            temperature (float): Temperature for routing softmax

        Returns:
            Tuple containing:
                - output: Processed hidden states [batch_size, seq_len, hidden_size]
                - aux_losses: Dictionary of auxiliary losses for training
                - aux_info: Dictionary of auxiliary information for monitoring
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Expert selection with confidence scoring
        routing_weights, selected_indices, confidence_scores, router_logits = \
            self.expert_selector(hidden_states, temperature)

        # FIXED: Add routing jitter during training for exploration
        if self.training and self.routing_jitter > 0:
            noise = torch.empty_like(router_logits).uniform_(-self.routing_jitter, self.routing_jitter)
            router_logits = router_logits + noise

        # FIXED: Router logits shape handling
        # expert_selector returns flattened logits: [batch_size * seq_len, num_experts]
        # expert_balancer expects: [batch_size, seq_len, num_experts]
        # Need to reshape properly to preserve batch structure
        router_logits_3d = router_logits.view(batch_size, seq_len, -1)

        # Apply load balancing with correctly shaped logits
        balanced_weights, balanced_indices = self.expert_balancer.compute_balanced_routing(
            router_logits_3d, self.num_experts_per_tok
        )
        # Flatten back to [batch_size * seq_len, top_k] for expert processing
        balanced_weights = balanced_weights.view(-1, balanced_weights.shape[-1])
        balanced_indices = balanced_indices.view(-1, balanced_indices.shape[-1])

        # Apply router dropout for regularization
        routing_weights = self.router_dropout(routing_weights)

        # Initialize output and tracking variables
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_outputs = []
        total_compute_tokens = 0
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device)

        # FIXED: Expert dropout - randomly disable experts during training for robustness
        expert_dropout_mask = None
        if self.training and self.expert_dropout > 0:
            # Randomly mask experts (but always keep at least one expert available)
            dropout_probs = torch.full((self.num_experts,), 1.0 - self.expert_dropout, device=hidden_states.device)
            expert_dropout_mask = torch.bernoulli(dropout_probs).bool()
            # Ensure at least one expert is available
            if not expert_dropout_mask.any():
                expert_dropout_mask[torch.randint(0, self.num_experts, (1,))] = True

        # Process each expert
        for expert_idx in range(self.num_experts):
            # FIXED: Skip expert if dropped out during training
            if expert_dropout_mask is not None and not expert_dropout_mask[expert_idx]:
                continue
            # Find tokens assigned to this expert
            expert_mask = (balanced_indices == expert_idx).any(dim=-1)
            # Flatten for proper indexing
            flat_hidden = hidden_states.view(-1, hidden_dim)
            flat_mask = expert_mask.view(-1)
            expert_tokens = flat_hidden[flat_mask]

            if len(expert_tokens) > 0:
                # Process through sparse expert
                expert_output, compute_mask, gate_score = self.experts[expert_idx](expert_tokens)
                expert_outputs.append(expert_output)

                # Update load statistics
                expert_load[expert_idx] = len(expert_tokens)
                total_compute_tokens += (compute_mask.sum() * len(expert_tokens)).item()

                # FIXED: Apply proper per-token weighted combination
                token_indices = torch.where(flat_mask)[0]

                # Get the appropriate weights for this expert (per-token, not averaged!)
                flat_balanced_weights = balanced_weights.view(-1, balanced_weights.shape[-1])
                flat_balanced_indices = balanced_indices.view(-1, balanced_indices.shape[-1])

                # For each token assigned to this expert, find its routing weight
                # Shape: tokens_for_expert[i] corresponds to expert_tokens[i]
                tokens_for_expert_indices = flat_balanced_indices[flat_mask]  # [num_expert_tokens, top_k]
                tokens_for_expert_weights = flat_balanced_weights[flat_mask]  # [num_expert_tokens, top_k]

                # Find which position in top_k corresponds to this expert
                expert_weight_per_token = torch.zeros(len(expert_tokens), device=hidden_states.device)
                for token_idx in range(len(expert_tokens)):
                    # Find where this expert appears in the top-k for this token
                    expert_positions = (tokens_for_expert_indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_positions) > 0:
                        # Use the weight from the first position where this expert appears
                        pos = expert_positions[0]
                        expert_weight_per_token[token_idx] = tokens_for_expert_weights[token_idx, pos]
                    else:
                        # Fallback: if expert not in top-k (shouldn't happen), use small weight
                        expert_weight_per_token[token_idx] = 0.1

                # Apply per-token weights (preserves routing signal!)
                weighted_output = expert_output * expert_weight_per_token.unsqueeze(-1)

                # Accumulate weighted expert outputs
                flat_final = final_hidden_states.view(-1, hidden_dim)
                flat_final[token_indices] += weighted_output
                final_hidden_states = flat_final.view(batch_size, seq_len, hidden_dim)

        # Compute auxiliary losses for training stability
        aux_losses = {}

        # FIXED: Scale auxiliary losses for gradient accumulation
        # When using gradient accumulation, losses are summed over multiple micro-steps
        # before the optimizer steps. This effectively multiplies auxiliary losses by
        # gradient_accumulation_steps, causing over-regularization. Scale them down.
        aux_loss_scale = 1.0 / max(1, self.gradient_accumulation_steps)

        # 1. Load balancing loss - encourages uniform expert usage
        load_balancing_loss = self._compute_load_balancing_loss(expert_load, batch_size * seq_len)
        aux_losses['load_balancing'] = load_balancing_loss * aux_loss_scale

        # 2. Expert diversity loss - encourages diverse expert representations
        diversity_loss = self._compute_diversity_loss(expert_outputs)
        aux_losses['diversity'] = diversity_loss * aux_loss_scale

        # 3. Confidence regularization - prevents overconfidence
        confidence_loss = self._compute_confidence_loss(confidence_scores)
        aux_losses['confidence'] = confidence_loss * aux_loss_scale

        # 4. Router z-loss - numerical stability
        z_loss = self._compute_z_loss(router_logits)
        aux_losses['z_loss'] = z_loss * aux_loss_scale

        # Auxiliary information for monitoring
        aux_info = {
            'routing_weights': balanced_weights,
            'selected_indices': balanced_indices,
            'confidence_scores': confidence_scores,
            'expert_load': expert_load,
            'compute_efficiency': total_compute_tokens / (batch_size * seq_len * self.hidden_size) if total_compute_tokens > 0 else 0.0
        }

        return final_hidden_states, aux_losses, aux_info

    def _compute_load_balancing_loss(self, expert_load: torch.Tensor, total_tokens: int) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        avg_load = total_tokens / self.num_experts
        load_variance = torch.var(expert_load)
        return load_variance / (avg_load ** 2 + 1e-8)

    def _compute_diversity_loss(self, expert_outputs: list) -> torch.Tensor:
        """
        Encourage experts to learn diverse representations.

        FIXED: Removed .detach() to allow gradients to flow back to experts.
        The diversity loss should influence expert specialization through gradients.
        """
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device) if expert_outputs else torch.tensor(0.0)

        # Project outputs to lower dimension for efficiency
        projected_outputs = []
        for out in expert_outputs:
            if out.numel() > 0:  # Only process non-empty outputs
                # Ensure 2D shape for projection
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                elif out.dim() > 2:
                    out = out.view(-1, out.shape[-1])
                # FIXED: Remove .detach() to allow gradient flow to experts
                projected = self.diversity_projection(out)
                projected_outputs.append(projected)

        if len(projected_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        # Compute pairwise cosine similarities
        diversity_loss = torch.tensor(0.0, device=projected_outputs[0].device)
        num_pairs = 0

        for i in range(len(projected_outputs)):
            for j in range(i + 1, len(projected_outputs)):
                # Use mean pooling to handle different sizes
                feat_i = projected_outputs[i].mean(dim=0)
                feat_j = projected_outputs[j].mean(dim=0)

                # Cosine similarity
                sim = F.cosine_similarity(feat_i, feat_j, dim=0)
                diversity_loss = diversity_loss + sim.abs()
                num_pairs += 1

        return diversity_loss / max(num_pairs, 1)

    def _compute_confidence_loss(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        """Regularize confidence scores to prevent overconfidence."""
        # Encourage moderate confidence scores around 0.7
        confidence_mean = torch.mean(confidence_scores)
        confidence_var = torch.var(confidence_scores)
        target_mean = 0.7

        return (confidence_mean - target_mean) ** 2 + 0.1 * confidence_var

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Z-loss for numerical stability in routing."""
        # Encourages logits to not be too large, preventing numerical instability
        z_loss = torch.mean(router_logits ** 2) * 1e-4
        return z_loss