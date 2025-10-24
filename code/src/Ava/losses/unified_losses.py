"""
Unified Loss Functions Module

Consolidates all loss functions from:
- repetition_penalty_loss.py
- anti_repetition_loss.py
- advanced_losses.py
- adaptive_mtp_loss.py
- deepseek_loss.py

Provides unified interface for all loss computation with consistent APIs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

# ============================================================================
# Repetition Penalty Losses
# ============================================================================

class UnifiedRepetitionPenalty(nn.Module):
    """
    Unified repetition penalty combining multiple approaches.

    Consolidates:
    - N-gram repetition penalty
    - Immediate token repetition
    - Sequence-level diversity
    """

    def __init__(
        self,
        ngram_size: int = 3,
        ngram_weight: float = 0.5,
        immediate_weight: float = 1.0,
        diversity_weight: float = 0.3,
        vocab_size: int = 50257
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.ngram_weight = ngram_weight
        self.immediate_weight = immediate_weight
        self.diversity_weight = diversity_weight
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute unified repetition penalty.

        Args:
            logits: [batch, seq_len, vocab_size]
            input_ids: [batch, seq_len]
            labels: Optional labels

        Returns:
            Repetition penalty loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        total_penalty = torch.tensor(0.0, device=device)

        # 1. N-gram repetition penalty
        if self.ngram_weight > 0 and seq_len >= self.ngram_size:
            ngram_penalty = self._compute_ngram_penalty(logits, input_ids)
            total_penalty += self.ngram_weight * ngram_penalty

        # 2. Immediate repetition penalty (token-to-token)
        if self.immediate_weight > 0 and seq_len > 1:
            immediate_penalty = self._compute_immediate_penalty(logits, input_ids)
            total_penalty += self.immediate_weight * immediate_penalty

        # 3. Diversity penalty (encourage varied output)
        if self.diversity_weight > 0:
            diversity_penalty = self._compute_diversity_penalty(logits)
            total_penalty += self.diversity_weight * diversity_penalty

        return total_penalty

    def _compute_ngram_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Penalize repeated n-grams."""
        batch_size, seq_len, vocab_size = logits.shape
        penalty = 0.0
        count = 0

        for b in range(batch_size):
            for i in range(seq_len - self.ngram_size + 1):
                ngram = tuple(input_ids[b, i:i+self.ngram_size].tolist())

                # Count occurrences
                occurrences = 0
                for j in range(seq_len - self.ngram_size + 1):
                    if tuple(input_ids[b, j:j+self.ngram_size].tolist()) == ngram:
                        occurrences += 1

                if occurrences > 1:
                    penalty += math.log(occurrences)
                    count += 1

        return torch.tensor(penalty / max(count, 1), device=logits.device)

    def _compute_immediate_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Penalize immediate token repetition."""
        batch_size, seq_len = input_ids.shape

        # Compare each token with previous
        repeated = (input_ids[:, 1:] == input_ids[:, :-1]).float()
        penalty = repeated.mean()

        return penalty

    def _compute_diversity_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Encourage output diversity."""
        # Compute entropy across vocabulary
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Negative entropy is penalty (we want high entropy/diversity)
        max_entropy = math.log(self.vocab_size)
        diversity_loss = 1.0 - (entropy.mean() / max_entropy)

        return diversity_loss


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Focuses training on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Compute focal loss."""
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of correct class
        batch_size, seq_len, vocab_size = logits.shape
        labels_flat = labels.view(-1)
        probs_flat = probs.view(-1, vocab_size)

        # Gather probabilities for true labels
        target_probs = probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - target_probs) ** self.gamma

        # Cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels_flat,
            reduction='none',
            label_smoothing=label_smoothing
        )

        # Focal loss
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Contrastive Loss
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning."""

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: [batch_size, hidden_size]
            labels: Optional labels for supervised contrastive

        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature

        batch_size = embeddings.size(0)

        # Create positive/negative masks
        if labels is not None:
            # Supervised: same label = positive
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.t()).float()
            # Remove diagonal
            mask.fill_diagonal_(0)
        else:
            # Self-supervised: diagonal pairs are positive
            mask = torch.eye(batch_size, device=embeddings.device)

        # Compute loss
        exp_sim = torch.exp(similarity)

        # Sum of all similarities
        sum_exp = exp_sim.sum(dim=1, keepdim=True)

        # Log probability of positives
        log_prob = similarity - torch.log(sum_exp)

        # Mean of log-likelihood over positives
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        loss = -mean_log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# MoE Balancing Loss
# ============================================================================

class MoEBalancingLoss(nn.Module):
    """Loss for balancing expert utilization in MoE models."""

    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MoE balancing loss.

        Args:
            router_logits: [batch, seq_len, num_experts]
            expert_indices: Selected expert indices

        Returns:
            (loss, stats_dict)
        """
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Average probability per expert across batch
        expert_probs = routing_probs.mean(dim=[0, 1])  # [num_experts]

        # Target: uniform distribution
        target_prob = 1.0 / self.num_experts

        # Load balancing loss (variance from uniform)
        balance_loss = ((expert_probs - target_prob) ** 2).mean()

        # Compute statistics
        stats = {
            'expert_utilization': expert_probs.max().item() / expert_probs.min().item(),
            'max_expert_prob': expert_probs.max().item(),
            'min_expert_prob': expert_probs.min().item()
        }

        return self.weight * balance_loss, stats


# ============================================================================
# Multi-Token Prediction Loss
# ============================================================================

class MultiTokenPredictionLoss(nn.Module):
    """Loss for predicting multiple future tokens."""

    def __init__(
        self,
        num_future_tokens: int = 3,
        weights: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        self.reduction = reduction

        # Default: exponentially decreasing weights
        if weights is None:
            weights = [0.5 ** i for i in range(num_future_tokens)]
        self.weights = weights

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        label_smoothing: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-token prediction loss.

        Args:
            predictions: List of [batch, seq_len, vocab_size] for each future position
            targets: List of target token ids for each position
            label_smoothing: Label smoothing factor

        Returns:
            (total_loss, stats_dict)
        """
        total_loss = 0.0
        losses = []

        for i, (pred, target, weight) in enumerate(zip(predictions, targets, self.weights)):
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                target.view(-1),
                reduction=self.reduction,
                label_smoothing=label_smoothing,
                ignore_index=-100
            )
            losses.append(loss.item())
            total_loss += weight * loss

        stats = {f'mtp_loss_{i+1}': loss for i, loss in enumerate(losses)}
        stats['mtp_loss_total'] = total_loss.item()

        return total_loss, stats


# ============================================================================
# Unified Loss Computer
# ============================================================================

class UnifiedLossComputer:
    """
    Main interface for computing all types of losses with consistent API.

    Consolidates all loss computation into single unified class.
    """

    def __init__(
        self,
        vocab_size: int = 50257,

        # Standard CE loss config
        label_smoothing: float = 0.0,
        ignore_index: int = -100,

        # Repetition penalty config
        use_repetition_penalty: bool = False,
        repetition_config: Optional[Dict] = None,

        # Focal loss config
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,

        # Contrastive loss config
        use_contrastive: bool = False,
        contrastive_temperature: float = 0.07,
        contrastive_weight: float = 0.1,

        # MoE balancing config
        use_moe_balancing: bool = False,
        num_experts: int = 8,
        moe_weight: float = 0.01,

        # Multi-token prediction config
        use_mtp: bool = False,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1
    ):
        """Initialize unified loss computer."""
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Initialize loss modules
        self.use_repetition_penalty = use_repetition_penalty
        if use_repetition_penalty:
            rep_config = repetition_config or {}
            self.repetition_loss = UnifiedRepetitionPenalty(vocab_size=vocab_size, **rep_config)

        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
            self.contrastive_weight = contrastive_weight

        self.use_moe_balancing = use_moe_balancing
        if use_moe_balancing:
            self.moe_balancing_loss = MoEBalancingLoss(num_experts=num_experts, weight=moe_weight)

        self.use_mtp = use_mtp
        if use_mtp:
            self.mtp_loss = MultiTokenPredictionLoss(num_future_tokens=num_future_tokens)
            self.mtp_weight = mtp_weight

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        mtp_predictions: Optional[List[torch.Tensor]] = None,
        mtp_targets: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all configured losses.

        Args:
            logits: Main prediction logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            input_ids: Input token ids (for repetition penalty)
            hidden_states: Hidden states (for contrastive)
            router_logits: Router logits for MoE (for balancing)
            mtp_predictions: Predictions for multi-token prediction
            mtp_targets: Targets for multi-token prediction

        Returns:
            (total_loss, stats_dict)
        """
        stats = {}
        total_loss = 0.0

        # 1. Main loss (CE or Focal)
        if self.use_focal_loss:
            main_loss = self.focal_loss(logits, labels, self.label_smoothing)
            stats['focal_loss'] = main_loss.item()
        else:
            main_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index
            )
            stats['ce_loss'] = main_loss.item()

        total_loss += main_loss

        # 2. Repetition penalty
        if self.use_repetition_penalty and input_ids is not None:
            rep_loss = self.repetition_loss(logits, input_ids, labels)
            stats['repetition_loss'] = rep_loss.item()
            total_loss += rep_loss

        # 3. Contrastive loss
        if self.use_contrastive and hidden_states is not None:
            cont_loss = self.contrastive_loss(hidden_states)
            stats['contrastive_loss'] = cont_loss.item()
            total_loss += self.contrastive_weight * cont_loss

        # 4. MoE balancing
        if self.use_moe_balancing and router_logits is not None:
            moe_loss, moe_stats = self.moe_balancing_loss(router_logits)
            stats['moe_balance_loss'] = moe_loss.item()
            stats.update(moe_stats)
            total_loss += moe_loss

        # 5. Multi-token prediction
        if self.use_mtp and mtp_predictions is not None and mtp_targets is not None:
            mtp_loss_val, mtp_stats = self.mtp_loss(mtp_predictions, mtp_targets, self.label_smoothing)
            stats.update(mtp_stats)
            total_loss += self.mtp_weight * mtp_loss_val

        stats['total_loss'] = total_loss.item()

        return total_loss, stats


# Convenience function
def create_loss_computer(
    vocab_size: int,
    loss_config: Optional[Dict] = None
) -> UnifiedLossComputer:
    """Convenience function to create loss computer from config dict."""
    config = loss_config or {}
    return UnifiedLossComputer(vocab_size=vocab_size, **config)
