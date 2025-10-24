"""
ALL LOSS FUNCTIONS - Complete Consolidated Module

This module consolidates ALL loss functions from the entire codebase into ONE file:
- Adaptive MTP Loss
- Advanced Losses (Contrastive, Focal, Label Smoothing, etc.)
- Anti-Repetition Losses
- DeepSeek Losses (MTP, Temperature Scaling, MoE Balancing)
- Repetition Penalty Losses
- And more!

All loss functions are available from this single import.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter
import math


# ============================================================================
# ADAPTIVE MTP LOSS
# ============================================================================

class AdaptiveMTPLoss(nn.Module):
    """
    Adaptive Multi-Token Prediction Loss.

    Combines:
    - Primary token prediction loss (always full weight)
    - Confidence-weighted additional token losses
    - Regularization to encourage confident predictions
    """

    def __init__(
        self,
        vocab_size: int,
        primary_loss_weight: float = 1.0,
        additional_loss_base_weight: float = 0.1,
        confidence_reg_strength: float = 0.01,
        use_confidence_weighting: bool = True,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.primary_loss_weight = primary_loss_weight
        self.additional_loss_base_weight = additional_loss_base_weight
        self.confidence_reg_strength = confidence_reg_strength
        self.use_confidence_weighting = use_confidence_weighting
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        self.register_buffer('total_primary_loss', torch.tensor(0.0))
        self.register_buffer('total_additional_loss', torch.tensor(0.0))
        self.register_buffer('total_confidence_reg', torch.tensor(0.0))
        self.register_buffer('loss_count', torch.tensor(0))

    def compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        if self.label_smoothing > 0:
            log_probs = F.log_softmax(logits_flat, dim=-1)
            with torch.no_grad():
                smoothed_targets = torch.zeros_like(log_probs)
                smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))
                smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), 1.0 - self.label_smoothing)
                if self.ignore_index >= 0:
                    padding_mask = targets_flat == self.ignore_index
                    smoothed_targets[padding_mask] = 0.0
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index, reduction='none')

        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum().clamp(min=1.0)
        else:
            if self.ignore_index >= 0:
                valid_tokens = (targets_flat != self.ignore_index).float()
                loss = loss.sum() / valid_tokens.sum().clamp(min=1.0)
            else:
                loss = loss.mean()
        return loss

    def forward(
        self, primary_logits: torch.Tensor, targets: torch.Tensor,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mtp_active: bool = False
    ) -> Dict[str, Any]:
        primary_loss = self.compute_cross_entropy(primary_logits, targets, attention_mask)
        primary_loss = primary_loss * self.primary_loss_weight

        additional_loss = torch.tensor(0.0, device=primary_logits.device)
        confidence_reg = torch.tensor(0.0, device=primary_logits.device)
        avg_confidence = torch.tensor(0.0, device=primary_logits.device)
        effective_mtp_weight = 0.0

        if mtp_active and additional_logits is not None:
            head_losses = []
            for i, logits in enumerate(additional_logits):
                future_offset = i + 1
                if future_offset >= targets.shape[1]:
                    continue
                shifted_targets = targets[:, future_offset:]
                shifted_logits = logits[:, :-future_offset, :]
                shifted_mask = attention_mask[:, future_offset:] if attention_mask is not None else None
                head_loss = self.compute_cross_entropy(shifted_logits, shifted_targets, shifted_mask)
                head_losses.append(head_loss)

            if head_losses:
                additional_loss = torch.stack(head_losses).mean()
                if self.use_confidence_weighting and confidence_scores is not None:
                    avg_confidence = confidence_scores.mean()
                    confidence_weight = avg_confidence.clamp(min=0.0, max=1.0)
                    effective_mtp_weight = self.additional_loss_base_weight * confidence_weight
                else:
                    effective_mtp_weight = self.additional_loss_base_weight
                additional_loss = additional_loss * effective_mtp_weight

        if confidence_scores is not None and self.confidence_reg_strength > 0:
            epsilon = 1e-7
            deviation_from_half = torch.abs(2 * confidence_scores - 1).clamp(min=epsilon)
            confidence_reg = -torch.log(deviation_from_half).mean() * self.confidence_reg_strength
            if confidence_scores.numel() > 0:
                avg_confidence = confidence_scores.mean()

        total_loss = primary_loss + additional_loss + confidence_reg

        with torch.no_grad():
            self.total_primary_loss += primary_loss.item()
            self.total_additional_loss += additional_loss.item()
            self.total_confidence_reg += confidence_reg.item()
            self.loss_count += 1

        return {
            'loss': total_loss,
            'primary_loss': primary_loss,
            'additional_loss': additional_loss,
            'confidence_reg': confidence_reg,
            'avg_confidence': avg_confidence,
            'effective_mtp_weight': effective_mtp_weight,
            'mtp_active': mtp_active
        }


# ============================================================================
# CONTRASTIVE LOSS
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning representations."""

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        loss_type: str = "infonce",
        normalize_embeddings: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type.lower()
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self.loss_type == "infonce":
            return self._infonce_loss(embeddings, labels, positive_pairs)
        elif self.loss_type == "standard":
            return self._standard_contrastive_loss(embeddings, labels, positive_pairs, negative_pairs)
        else:
            return F.triplet_margin_loss(embeddings, embeddings, embeddings, margin=self.margin)

    def _infonce_loss(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        mid_point = batch_size // 2
        anchor_embeds = embeddings[:mid_point]
        positive_embeds = embeddings[mid_point:mid_point*2]

        all_similarities = torch.matmul(anchor_embeds, embeddings.T) / self.temperature
        pos_labels = torch.arange(len(anchor_embeds), len(anchor_embeds)*2, device=embeddings.device)
        return F.cross_entropy(all_similarities, pos_labels)

    def _standard_contrastive_loss(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if positive_pairs is None or negative_pairs is None:
            if labels is None:
                raise ValueError("Either pairs or labels must be provided")
            positive_pairs, negative_pairs = self._generate_pairs_from_labels(labels)

        pos_distances = torch.norm(embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]], p=2, dim=-1)
        neg_distances = torch.norm(embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]], p=2, dim=-1)
        pos_loss = pos_distances.pow(2)
        neg_loss = F.relu(self.margin - neg_distances).pow(2)
        return (pos_loss.mean() + neg_loss.mean()) / 2

    def _generate_pairs_from_labels(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = labels.shape[0]
        positive_pairs, negative_pairs = [], []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if labels[i] == labels[j]:
                    positive_pairs.append([i, j])
                else:
                    negative_pairs.append([i, j])
        return torch.tensor(positive_pairs, device=labels.device), torch.tensor(negative_pairs, device=labels.device)


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha if isinstance(self.alpha, (float, int)) else self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# DIVERSITY LOSS
# ============================================================================

class DiversityLoss(nn.Module):
    """Diversity loss to encourage diverse representations in MoE models."""

    def __init__(self, similarity_metric: str = "cosine", diversity_weight: float = 1.0):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.diversity_weight = diversity_weight

    def forward(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        device = expert_outputs[0].device
        diversity_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                if self.similarity_metric == "cosine":
                    similarity = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).mean()
                else:
                    similarity = (expert_outputs[i] * expert_outputs[j]).sum(dim=-1).mean()
                diversity_loss += similarity
                num_pairs += 1

        return self.diversity_weight * diversity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)


# ============================================================================
# AUXILIARY LOSS (MOE)
# ============================================================================

class AuxiliaryLoss(nn.Module):
    """Auxiliary loss for MoE routing."""

    def __init__(
        self,
        load_balancing_weight: float = 0.0001,
        router_z_weight: float = 0.001,
        expert_diversity_weight: float = 0.0001
    ):
        super().__init__()
        self.load_balancing_weight = load_balancing_weight
        self.router_z_weight = router_z_weight
        self.expert_diversity_weight = expert_diversity_weight

    def load_balancing_loss(self, gate_logits: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_mask = F.one_hot(expert_indices, num_experts).float()
        expert_usage = expert_mask.sum(dim=0).sum(dim=0)
        gate_prob_sums = gate_probs.sum(dim=0)
        total_tokens = gate_logits.shape[0] * expert_indices.shape[1]
        load_loss = num_experts * torch.sum(gate_prob_sums * expert_usage) / (total_tokens ** 2)
        return torch.clamp(load_loss, max=10.0)

    def router_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)
        logsumexp_vals = torch.logsumexp(gate_logits, dim=-1)
        return torch.clamp(torch.mean(logsumexp_vals ** 2), max=100.0)

    def forward(
        self,
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        num_experts: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        if gate_logits is not None and expert_indices is not None and num_experts is not None:
            losses['load_balancing'] = self.load_balancing_weight * self.load_balancing_loss(gate_logits, expert_indices, num_experts)
        if gate_logits is not None:
            losses['router_z'] = self.router_z_weight * self.router_z_loss(gate_logits)
        if expert_outputs is not None:
            diversity_loss_fn = DiversityLoss()
            losses['expert_diversity'] = self.expert_diversity_weight * diversity_loss_fn(expert_outputs)
        return losses


# ============================================================================
# ANTI-REPETITION LOSS
# ============================================================================

class AntiRepetitionLoss(nn.Module):
    """Enhanced loss function that penalizes repetitive outputs."""

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        pad_token_id: int,
        repetition_penalty_weight: float = 0.1,
        eos_penalty_weight: float = 0.05,
        diversity_bonus_weight: float = 0.05,
        ngram_size: int = 4,
        ignore_index: int = -100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.eos_penalty_weight = eos_penalty_weight
        self.diversity_bonus_weight = diversity_bonus_weight
        self.ngram_size = ngram_size
        self.ignore_index = ignore_index
        self.base_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def calculate_ngram_repetition(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        if seq_len < self.ngram_size:
            return torch.zeros(batch_size, device=device)

        repetition_scores = []
        for i in range(batch_size):
            tokens = token_ids[i]
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            if len(tokens) < self.ngram_size:
                repetition_scores.append(0.0)
                continue

            ngrams = []
            for j in range(len(tokens) - self.ngram_size + 1):
                ngram = tuple(tokens[j:j+self.ngram_size].tolist())
                if self.pad_token_id not in ngram:
                    ngrams.append(ngram)

            if len(ngrams) == 0:
                repetition_scores.append(0.0)
                continue

            unique_ngrams = len(set(ngrams))
            repetition = 1.0 - (unique_ngrams / len(ngrams))
            repetition_scores.append(repetition)

        return torch.tensor(repetition_scores, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        batch_size, seq_len, vocab_size = logits.shape
        base_loss_per_token = self.base_loss(logits.view(-1, vocab_size), labels.view(-1))
        base_loss_per_token = base_loss_per_token.view(batch_size, seq_len)

        if attention_mask is not None:
            mask = (labels != self.ignore_index).float()
            base_loss = (base_loss_per_token * mask).sum() / mask.sum()
        else:
            base_loss = base_loss_per_token.mean()

        repetition_scores = self.calculate_ngram_repetition(labels, attention_mask)
        repetition_penalty = repetition_scores.mean()

        total_loss = base_loss + self.repetition_penalty_weight * repetition_penalty

        if return_components:
            return total_loss, {'base_loss': base_loss.item(), 'repetition_penalty': repetition_penalty.item()}
        return total_loss, None


# ============================================================================
# DEEPSEEK MULTI-TOKEN PREDICTION LOSS
# ============================================================================

class MultiTokenPredictionLoss(nn.Module):
    """Multi-Token Prediction (MTP) loss for improved long-range dependency learning."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1,
        shared_projection: bool = False,
        temperature: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_future_tokens = num_future_tokens
        self.mtp_weight = mtp_weight
        self.temperature = temperature

        if shared_projection:
            self.projection = nn.Linear(hidden_size, vocab_size * num_future_tokens)
        else:
            self.projections = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for _ in range(num_future_tokens)])
        self.shared_projection = shared_projection
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        hidden_states = self.layer_norm(hidden_states)

        total_mtp_loss = torch.tensor(0.0, device=device)
        per_token_losses = []

        for future_idx in range(1, self.num_future_tokens + 1):
            if future_idx >= seq_len:
                continue

            pred_hidden = hidden_states[:, :-future_idx, :]
            future_targets = target_ids[:, future_idx:]

            if self.shared_projection:
                start_idx = (future_idx - 1) * self.vocab_size
                end_idx = future_idx * self.vocab_size
                logits = self.projection(pred_hidden)[:, :, start_idx:end_idx]
            else:
                logits = self.projections[future_idx - 1](pred_hidden)

            logits = logits / self.temperature
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = future_targets.reshape(-1)

            if attention_mask is not None:
                mask_flat = attention_mask[:, future_idx:].reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

            total_mtp_loss = total_mtp_loss + loss
            per_token_losses.append(loss.detach().item())

        if self.num_future_tokens > 0:
            total_mtp_loss = total_mtp_loss / min(self.num_future_tokens, seq_len - 1)

        return {
            'mtp_loss': total_mtp_loss * self.mtp_weight,
            'per_token_losses': per_token_losses,
            'mtp_weight': self.mtp_weight
        }


# ============================================================================
# TEMPERATURE-SCALED CROSS ENTROPY
# ============================================================================

class TemperatureScaledCrossEntropy(nn.Module):
    """Temperature-scaled cross-entropy loss with adaptive temperature."""

    def __init__(
        self,
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        label_smoothing: float = 0.1,
        vocab_size: Optional[int] = None,
        temperature_bounds: Tuple[float, float] = (0.5, 2.0),
        adaptation_rate: float = 0.01,
        eos_token_id: Optional[int] = None,
        min_sequence_length: int = 20,
        eos_penalty_weight: float = 5.0
    ):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.adaptive_temperature = adaptive_temperature
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.temperature_bounds = temperature_bounds
        self.adaptation_rate = adaptation_rate
        self.eos_token_id = eos_token_id
        self.min_sequence_length = min_sequence_length
        self.eos_penalty_weight = eos_penalty_weight

        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_size', torch.tensor(0))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> Dict[str, Any]:
        scaled_logits = logits / self.temperature
        batch_size, seq_len = targets.shape
        vocab_size = logits.shape[-1]

        logits_flat = scaled_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smoothed_targets = torch.zeros_like(logits_flat)
                smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))
                smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            loss = loss * mask_flat
            if reduction == 'mean':
                loss = loss.sum() / mask_flat.sum()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()

        return {'loss': loss, 'temperature': self.temperature.item(), 'label_smoothing': self.label_smoothing}


# ============================================================================
# MOE BALANCER (AUXILIARY-FREE)
# ============================================================================

class AuxiliaryFreeMoEBalancer(nn.Module):
    """Auxiliary-loss-free load balancing for Mixture of Experts."""

    def __init__(
        self,
        num_experts: int,
        balance_loss_weight: float = 0.0,
        gradient_balance_weight: float = 0.1,
        target_balance_ratio: float = 1.0,
        momentum: float = 0.9
    ):
        super().__init__()
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        self.gradient_balance_weight = gradient_balance_weight
        self.target_balance_ratio = target_balance_ratio
        self.momentum = momentum

        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('expert_scores', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        compute_loss: bool = False
    ) -> Dict[str, Any]:
        expert_scores = F.softmax(gate_logits, dim=-1)

        with torch.no_grad():
            for i in range(self.num_experts):
                count = (expert_indices == i).float().sum()
                self.expert_counts[i] = self.momentum * self.expert_counts[i] + (1 - self.momentum) * count

            self.expert_scores = self.momentum * self.expert_scores + (1 - self.momentum) * expert_scores.mean(dim=0)
            self.total_tokens = self.momentum * self.total_tokens + (1 - self.momentum) * float(expert_indices.shape[0])

        balance_loss = torch.tensor(0.0, device=gate_logits.device)
        return {'balance_loss': balance_loss, 'expert_counts': self.expert_counts.clone()}


# ============================================================================
# N-GRAM REPETITION PENALTY
# ============================================================================

class NGramRepetitionPenalty(nn.Module):
    """Penalizes n-gram repetitions in generated sequences."""

    def __init__(self, ngram_size: int = 3, penalty_weight: float = 0.1, vocab_size: int = 50257, ignore_index: int = -100):
        super().__init__()
        self.ngram_size = ngram_size
        self.penalty_weight = penalty_weight
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def compute_ngram_repetition_penalty(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len < self.ngram_size:
            return torch.tensor(0.0, device=device)

        total_penalty = torch.tensor(0.0, device=device)
        total_ngrams = 0

        for b in range(batch_size):
            sequence = token_ids[b]
            mask = attention_mask[b] if attention_mask is not None else None

            ngrams = []
            for i in range(seq_len - self.ngram_size + 1):
                if mask is not None and mask[i:i+self.ngram_size].sum() < self.ngram_size:
                    continue
                ngram = tuple(sequence[i:i+self.ngram_size].tolist())
                ngrams.append(ngram)

            if len(ngrams) == 0:
                continue

            ngram_counts = Counter(ngrams)
            for ngram, count in ngram_counts.items():
                if count > 1:
                    repetition_penalty = (count - 1) * count / 2
                    total_penalty = total_penalty + repetition_penalty
            total_ngrams += len(ngrams)

        normalized_penalty = total_penalty / total_ngrams if total_ngrams > 0 else torch.tensor(0.0, device=device)
        return normalized_penalty * self.penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
        ngram_penalty = self.compute_ngram_repetition_penalty(targets, attention_mask)
        return {'ngram_penalty': ngram_penalty, 'penalty_weight': self.penalty_weight}


# ============================================================================
# UNIFIED LOSS COMPUTER (Main Interface)
# ============================================================================

class UnifiedLossComputer:
    """
    Main unified loss computer combining ALL loss types.

    This is the primary interface for using all loss functions.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        use_repetition_penalty: bool = False,
        repetition_config: Optional[Dict] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_contrastive: bool = False,
        contrastive_temperature: float = 0.07,
        contrastive_weight: float = 0.1,
        use_moe_balancing: bool = False,
        num_experts: int = 8,
        moe_weight: float = 0.01,
        use_mtp: bool = False,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        self.use_repetition_penalty = use_repetition_penalty
        if use_repetition_penalty:
            rep_config = repetition_config or {}
            self.repetition_loss = NGramRepetitionPenalty(vocab_size=vocab_size, **rep_config)

        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
            self.contrastive_weight = contrastive_weight

        self.use_moe_balancing = use_moe_balancing
        if use_moe_balancing:
            self.moe_balancing_loss = AuxiliaryLoss(load_balancing_weight=moe_weight)

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats = {}
        total_loss = 0.0

        # Main loss
        if self.use_focal_loss:
            main_loss = self.focal_loss(logits, labels)
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

        # Repetition penalty
        if self.use_repetition_penalty and input_ids is not None:
            rep_dict = self.repetition_loss(logits, labels)
            stats['repetition_loss'] = rep_dict['ngram_penalty'].item()
            total_loss += rep_dict['ngram_penalty']

        # Contrastive loss
        if self.use_contrastive and hidden_states is not None:
            cont_loss = self.contrastive_loss(hidden_states)
            stats['contrastive_loss'] = cont_loss.item()
            total_loss += self.contrastive_weight * cont_loss

        # MoE balancing
        if self.use_moe_balancing and router_logits is not None and num_experts is not None:
            moe_dict = self.moe_balancing_loss(gate_logits=router_logits, num_experts=num_experts)
            for k, v in moe_dict.items():
                stats[f'moe_{k}'] = v.item()
                total_loss += v

        stats['total_loss'] = total_loss.item()
        return total_loss, stats


# Convenience function
def create_loss_computer(vocab_size: int, loss_config: Optional[Dict] = None) -> UnifiedLossComputer:
    """Convenience function to create loss computer from config dict."""
    config = loss_config or {}
    return UnifiedLossComputer(vocab_size=vocab_size, **config)
