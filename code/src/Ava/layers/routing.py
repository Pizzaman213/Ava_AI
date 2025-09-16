"""
Expert routing and selection for MoE++ architecture.

This module implements intelligent routing mechanisms including:
- ExpertSelector: Dynamic expert selection with confidence scoring
- MoEPlusPlusLayer: Complete MoE layer with load balancing and auxiliary losses

The routing system uses confidence-based dynamic selection where high-confidence
tokens use fewer experts while uncertain tokens engage more experts for better accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

from .experts import ExpertBalancer, SparseExpert


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

        # Apply load balancing
        balanced_weights, balanced_indices = self.expert_balancer.compute_balanced_routing(
            router_logits.unsqueeze(0), self.num_experts_per_tok
        )
        balanced_weights = balanced_weights.squeeze(0)
        balanced_indices = balanced_indices.squeeze(0)

        # Apply router dropout for regularization
        routing_weights = self.router_dropout(routing_weights)

        # Initialize output and tracking variables
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_outputs = []
        total_compute_tokens = 0
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device)

        # Process each expert
        for expert_idx in range(self.num_experts):
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

                # Apply weighted combination
                token_indices = torch.where(flat_mask)[0]

                # Get the appropriate weights for this expert
                flat_balanced_weights = balanced_weights.view(-1, balanced_weights.shape[-1])
                flat_balanced_indices = balanced_indices.view(-1, balanced_indices.shape[-1])

                expert_weight_mask = (flat_balanced_indices[flat_mask] == expert_idx)
                expert_weights = flat_balanced_weights[flat_mask][expert_weight_mask.any(dim=-1)]

                if len(expert_weights) > 0:
                    weighted_output = expert_output * expert_weights.mean().unsqueeze(-1)
                else:
                    weighted_output = expert_output * 0.5  # Default weight if no specific weight found

                # Accumulate weighted expert outputs
                flat_final = final_hidden_states.view(-1, hidden_dim)
                flat_final[token_indices] += weighted_output
                final_hidden_states = flat_final.view(batch_size, seq_len, hidden_dim)

        # Compute auxiliary losses for training stability
        aux_losses = {}

        # 1. Load balancing loss - encourages uniform expert usage
        load_balancing_loss = self._compute_load_balancing_loss(expert_load, batch_size * seq_len)
        aux_losses['load_balancing'] = load_balancing_loss

        # 2. Expert diversity loss - encourages diverse expert representations
        diversity_loss = self._compute_diversity_loss(expert_outputs)
        aux_losses['diversity'] = diversity_loss

        # 3. Confidence regularization - prevents overconfidence
        confidence_loss = self._compute_confidence_loss(confidence_scores)
        aux_losses['confidence'] = confidence_loss

        # 4. Router z-loss - numerical stability
        z_loss = self._compute_z_loss(router_logits)
        aux_losses['z_loss'] = z_loss

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
        """Encourage experts to learn diverse representations."""
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
                projected = self.diversity_projection(out.detach())
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