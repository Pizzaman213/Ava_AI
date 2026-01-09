"""
Advanced Loss Functions for Training.

This module provides specialized loss functions for various training scenarios:
1. FocalLoss - Handles class imbalance by down-weighting easy examples
2. ContrastiveLoss - InfoNCE-style contrastive learning
3. DiversityLoss - Encourages diverse representations
4. AdaptiveTemperatureLoss - Auto-adjusts temperature based on loss statistics
5. LabelSmoothingCrossEntropy - Cross-entropy with label smoothing
6. DistillationLoss - Knowledge distillation from teacher model

Example:
    >>> focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    >>> loss = focal_loss(logits, targets)

    >>> contrastive = ContrastiveLoss(temperature=0.07)
    >>> loss = contrastive(embeddings, labels)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Focal Loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Contrastive Loss
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.0

    # Diversity Loss
    diversity_weight: float = 0.01

    # Adaptive Temperature
    initial_temperature: float = 1.0
    min_temperature: float = 0.5
    max_temperature: float = 2.0
    temperature_ema_decay: float = 0.99

    # Label Smoothing
    label_smoothing: float = 0.1

    # Distillation
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Weight between hard and soft targets


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard
    negatives, which is especially useful for imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class balancing weight (can be scalar or per-class tensor)
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Index to ignore in loss computation

    Reference:
        "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> logits = torch.randn(32, 1000)  # [batch, vocab]
        >>> targets = torch.randint(0, 1000, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = 0.25,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha_weights', alpha)
        else:
            self.alpha_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [batch, num_classes] or [batch, seq, num_classes]
            targets: [batch] or [batch, seq]

        Returns:
            loss: Scalar or tensor depending on reduction
        """
        # Handle different input shapes
        if logits.dim() == 3:
            # [batch, seq, vocab] -> [batch*seq, vocab]
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

        # Compute standard cross-entropy
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction='none',
            ignore_index=self.ignore_index,
        )

        # Get probabilities
        p_t = torch.exp(-ce_loss)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha_weights is not None:
            # Per-class alpha
            alpha_t = self.alpha_weights.gather(0, targets.clamp(0))
            focal_weight = alpha_t * focal_weight
        elif self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Mask out ignored indices
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float()
            focal_loss = focal_loss * mask

        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return focal_loss.sum() / mask.sum().clamp(min=1)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style Contrastive Loss.

    Encourages similar representations for positive pairs and
    dissimilar representations for negative pairs.

    L = -log(exp(sim(z_i, z_j) / tau) / sum_k exp(sim(z_i, z_k) / tau))

    Args:
        temperature: Temperature scaling parameter
        margin: Optional margin for triplet-style loss
        normalize: Whether to L2 normalize embeddings

    Reference:
        "A Simple Framework for Contrastive Learning" (Chen et al., 2020)

    Example:
        >>> loss_fn = ContrastiveLoss(temperature=0.07)
        >>> embeddings = torch.randn(32, 768)  # [batch, hidden]
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, ...])  # Positive pairs share labels
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.normalize = normalize

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: [batch, hidden_size]
            labels: Optional [batch] labels (samples with same label are positive pairs)
            positive_pairs: Optional [batch, 2] indices of positive pairs

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Normalize embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask
        if labels is not None:
            # Samples with same label are positive pairs
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # Remove self-similarity
            positive_mask.fill_diagonal_(0)
        elif positive_pairs is not None:
            positive_mask = torch.zeros(batch_size, batch_size, device=device)
            for i, j in positive_pairs:
                positive_mask[i, j] = 1
                positive_mask[j, i] = 1
        else:
            # Assume consecutive pairs are positives (e.g., augmented views)
            positive_mask = torch.zeros(batch_size, batch_size, device=device)
            for i in range(0, batch_size - 1, 2):
                positive_mask[i, i + 1] = 1
                positive_mask[i + 1, i] = 1

        # Apply margin
        if self.margin > 0:
            similarity = similarity - self.margin * positive_mask

        # Create negative mask (all pairs except self)
        negative_mask = 1 - torch.eye(batch_size, device=device)

        # Compute loss (NCE-style)
        # For each sample, compute log probability of positive pairs
        exp_sim = torch.exp(similarity) * negative_mask

        # Numerator: positive pairs
        pos_sim = (torch.exp(similarity) * positive_mask).sum(dim=-1)

        # Denominator: all pairs except self
        all_sim = exp_sim.sum(dim=-1)

        # Avoid log(0)
        loss = -torch.log(pos_sim / (all_sim + 1e-9) + 1e-9)

        # Only compute loss for samples that have positive pairs
        valid_mask = (positive_mask.sum(dim=-1) > 0).float()
        if valid_mask.sum() > 0:
            loss = (loss * valid_mask).sum() / valid_mask.sum()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class DiversityLoss(nn.Module):
    """
    Diversity Loss to encourage diverse representations.

    Penalizes representations that are too similar within a batch,
    promoting exploration of the representation space.

    L = mean(sim(z_i, z_j)) for i != j

    Args:
        weight: Loss weight
        target_similarity: Target average similarity (0 = orthogonal)
        margin: Margin for hinge-style loss

    Example:
        >>> loss_fn = DiversityLoss(weight=0.01)
        >>> embeddings = torch.randn(32, 768)
        >>> loss = loss_fn(embeddings)
    """

    def __init__(
        self,
        weight: float = 0.01,
        target_similarity: float = 0.0,
        margin: Optional[float] = None,
    ):
        super().__init__()
        self.weight = weight
        self.target_similarity = target_similarity
        self.margin = margin

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss.

        Args:
            embeddings: [batch, hidden_size] or [batch, seq, hidden_size]

        Returns:
            loss: Scalar diversity loss
        """
        # Handle sequence dimension
        if embeddings.dim() == 3:
            # [batch, seq, hidden] -> [batch * seq, hidden]
            embeddings = embeddings.view(-1, embeddings.shape[-1])

        batch_size = embeddings.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T)

        # Remove diagonal (self-similarity)
        mask = 1 - torch.eye(batch_size, device=embeddings.device)
        similarity = similarity * mask

        # Compute mean off-diagonal similarity
        mean_similarity = similarity.sum() / (batch_size * (batch_size - 1))

        # Compute loss
        if self.margin is not None:
            # Hinge-style: only penalize if above target + margin
            loss = F.relu(mean_similarity - self.target_similarity - self.margin)
        else:
            # MSE-style: penalize deviation from target
            loss = (mean_similarity - self.target_similarity) ** 2

        return self.weight * loss


class AdaptiveTemperatureLoss(nn.Module):
    """
    Cross-entropy with adaptive temperature scaling.

    Automatically adjusts temperature based on loss statistics to
    improve training stability and convergence.

    When loss is high -> lower temperature (sharper predictions)
    When loss is low -> higher temperature (softer predictions)

    Args:
        initial_temperature: Starting temperature
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        ema_decay: Exponential moving average decay for loss tracking
        adaptation_rate: How fast to adapt temperature

    Example:
        >>> loss_fn = AdaptiveTemperatureLoss()
        >>> logits = torch.randn(32, 1000)
        >>> targets = torch.randint(0, 1000, (32,))
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Temperature: {loss_fn.temperature:.4f}")
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.5,
        max_temperature: float = 2.0,
        ema_decay: float = 0.99,
        adaptation_rate: float = 0.01,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.ema_decay = ema_decay
        self.adaptation_rate = adaptation_rate
        self.ignore_index = ignore_index

        # Learnable or adaptive temperature
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('loss_var_ema', torch.tensor(1.0))
        self.register_buffer('step_count', torch.tensor(0))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with adaptive temperature.

        Args:
            logits: [batch, num_classes] or [batch, seq, num_classes]
            targets: [batch] or [batch, seq]

        Returns:
            loss: Scaled cross-entropy loss
        """
        # Handle different input shapes
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            scaled_logits, targets,
            reduction='mean',
            ignore_index=self.ignore_index,
        )

        # Update temperature (only during training)
        if self.training:
            self._update_temperature(loss.detach())

        return loss

    def _update_temperature(self, loss: torch.Tensor):
        """Update temperature based on loss statistics."""
        self.step_count += 1

        # Update EMA of loss
        if self.step_count == 1:
            self.loss_ema = loss
        else:
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss

        # Update variance estimate
        loss_diff = (loss - self.loss_ema) ** 2
        self.loss_var_ema = self.ema_decay * self.loss_var_ema + (1 - self.ema_decay) * loss_diff

        # Compute z-score
        std = torch.sqrt(self.loss_var_ema + 1e-9)
        z_score = (loss - self.loss_ema) / std

        # Adapt temperature
        # High loss (z > 0) -> decrease temperature (sharper)
        # Low loss (z < 0) -> increase temperature (softer)
        temp_delta = -self.adaptation_rate * z_score
        new_temp = self.temperature + temp_delta

        # Clamp to valid range
        self.temperature = torch.clamp(new_temp, self.min_temperature, self.max_temperature)

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    Instead of hard 0/1 targets, uses soft targets:
    - True class: 1 - smoothing
    - Other classes: smoothing / (num_classes - 1)

    This helps prevent overconfident predictions and improves generalization.

    Args:
        smoothing: Label smoothing factor (0 = no smoothing)
        ignore_index: Index to ignore in loss computation

    Example:
        >>> loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> logits = torch.randn(32, 1000)
        >>> targets = torch.randint(0, 1000, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: [batch, num_classes] or [batch, seq, num_classes]
            targets: [batch] or [batch, seq]

        Returns:
            loss: Scalar loss value
        """
        # Handle different input shapes
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

        num_classes = logits.shape[-1]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            # Start with uniform distribution
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            # Set true class
            smooth_targets.scatter_(
                -1,
                targets.unsqueeze(-1).clamp(0),  # Clamp for ignore_index
                1 - self.smoothing,
            )

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.

    Combines hard targets (true labels) with soft targets (teacher predictions)
    to transfer knowledge from a larger teacher model.

    L = alpha * soft_loss + (1 - alpha) * hard_loss

    Args:
        temperature: Temperature for softening teacher predictions
        alpha: Weight between soft and hard targets
        ignore_index: Index to ignore in loss computation

    Reference:
        "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

    Example:
        >>> loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
        >>> student_logits = torch.randn(32, 1000)
        >>> teacher_logits = torch.randn(32, 1000)
        >>> targets = torch.randint(0, 1000, (32,))
        >>> loss = loss_fn(student_logits, teacher_logits, targets)
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute distillation loss.

        Args:
            student_logits: [batch, num_classes] student predictions
            teacher_logits: [batch, num_classes] teacher predictions
            targets: [batch] true labels

        Returns:
            loss: Combined distillation loss
            info: Dict with hard_loss and soft_loss components
        """
        # Handle different input shapes
        if student_logits.dim() == 3:
            batch_size, seq_len, vocab_size = student_logits.shape
            student_logits = student_logits.view(-1, vocab_size)
            teacher_logits = teacher_logits.view(-1, vocab_size)
            targets = targets.view(-1)

        # Hard loss (standard cross-entropy)
        hard_loss = F.cross_entropy(
            student_logits, targets,
            reduction='mean',
            ignore_index=self.ignore_index,
        )

        # Soft loss (KL divergence with teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean',
        ) * (self.temperature ** 2)  # Scale by T^2 as per original paper

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        info = {
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
        }

        return loss, info


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.

    Useful for multi-task learning or when using auxiliary losses.

    Args:
        losses: Dict mapping loss names to (loss_fn, weight) tuples

    Example:
        >>> combined = CombinedLoss({
        ...     'ce': (nn.CrossEntropyLoss(), 1.0),
        ...     'diversity': (DiversityLoss(), 0.01),
        ...     'contrastive': (ContrastiveLoss(), 0.1),
        ... })
        >>> loss, info = combined({
        ...     'ce': (logits, targets),
        ...     'diversity': (embeddings,),
        ...     'contrastive': (embeddings, labels),
        ... })
    """

    def __init__(
        self,
        losses: Dict[str, Tuple[nn.Module, float]],
    ):
        super().__init__()
        self.loss_modules = nn.ModuleDict()
        self.loss_weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.loss_modules[name] = loss_fn
            self.loss_weights[name] = weight

    def forward(
        self,
        inputs: Dict[str, Tuple],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            inputs: Dict mapping loss names to input tuples

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values
        """
        total_loss = 0.0
        loss_dict = {}

        for name, args in inputs.items():
            if name not in self.loss_modules:
                logger.warning(f"Unknown loss: {name}")
                continue

            loss_fn = self.loss_modules[name]
            weight = self.loss_weights.get(name, 1.0)

            # Compute loss
            loss = loss_fn(*args)

            # Handle tuple returns (loss, info)
            if isinstance(loss, tuple):
                loss = loss[0]

            loss_dict[name] = loss
            total_loss = total_loss + weight * loss

        loss_dict['total'] = total_loss

        return total_loss, loss_dict


def create_loss_from_config(config: LossConfig, loss_type: str) -> nn.Module:
    """
    Create a loss function from configuration.

    Args:
        config: LossConfig with parameters
        loss_type: Type of loss ('focal', 'contrastive', 'diversity', 'adaptive', 'smooth')

    Returns:
        Configured loss function
    """
    if loss_type == 'focal':
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
        )
    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            temperature=config.contrastive_temperature,
            margin=config.contrastive_margin,
        )
    elif loss_type == 'diversity':
        return DiversityLoss(
            weight=config.diversity_weight,
        )
    elif loss_type == 'adaptive':
        return AdaptiveTemperatureLoss(
            initial_temperature=config.initial_temperature,
            min_temperature=config.min_temperature,
            max_temperature=config.max_temperature,
            ema_decay=config.temperature_ema_decay,
        )
    elif loss_type == 'smooth':
        return LabelSmoothingCrossEntropy(
            smoothing=config.label_smoothing,
        )
    elif loss_type == 'distillation':
        return DistillationLoss(
            temperature=config.distillation_temperature,
            alpha=config.distillation_alpha,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    'LossConfig',
    'FocalLoss',
    'ContrastiveLoss',
    'DiversityLoss',
    'AdaptiveTemperatureLoss',
    'LabelSmoothingCrossEntropy',
    'DistillationLoss',
    'CombinedLoss',
    'create_loss_from_config',
]
