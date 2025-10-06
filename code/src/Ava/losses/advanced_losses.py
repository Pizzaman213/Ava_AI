"""
Advanced loss functions for enhanced training.

This module implements various advanced loss functions including contrastive
learning, auxiliary losses, and specialized training objectives.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Union
import math


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning representations by contrasting positive and negative pairs.

    This implementation supports both standard contrastive loss and InfoNCE-style loss.
    """

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
        """
        Compute contrastive loss.

        Args:
            embeddings: Embeddings tensor [batch_size, embedding_dim]
            labels: Labels for creating positive/negative pairs [batch_size]
            positive_pairs: Explicit positive pairs [num_pairs, 2]
            negative_pairs: Explicit negative pairs [num_pairs, 2]

        Returns:
            Contrastive loss value
        """
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self.loss_type == "infonce":
            return self._infonce_loss(embeddings, labels, positive_pairs)
        elif self.loss_type == "standard":
            return self._standard_contrastive_loss(embeddings, labels, positive_pairs, negative_pairs)
        elif self.loss_type == "triplet":
            if labels is None:
                raise ValueError("Labels required for triplet loss")
            return self._triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _infonce_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """InfoNCE loss implementation."""
        batch_size = embeddings.shape[0]

        if positive_pairs is not None:
            # Use explicit positive pairs
            anchor_indices = positive_pairs[:, 0]
            positive_indices = positive_pairs[:, 1]
            anchor_embeds = embeddings[anchor_indices]
            positive_embeds = embeddings[positive_indices]
        elif labels is not None:
            # Create positive pairs from labels
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = mask.float() - torch.eye(batch_size, device=mask.device)

            # For simplicity, use consecutive augmentations as positive pairs
            anchor_embeds = embeddings[::2]  # Even indices
            positive_embeds = embeddings[1::2]  # Odd indices
            batch_size = min(anchor_embeds.shape[0], positive_embeds.shape[0])
            anchor_embeds = anchor_embeds[:batch_size]
            positive_embeds = positive_embeds[:batch_size]
        else:
            # Assume first half are anchors, second half are positives
            mid_point = batch_size // 2
            anchor_embeds = embeddings[:mid_point]
            positive_embeds = embeddings[mid_point:mid_point*2]

        # Compute similarities
        positive_sim = F.cosine_similarity(anchor_embeds, positive_embeds, dim=-1) / self.temperature

        # Compute similarities with all negatives
        all_similarities = torch.matmul(anchor_embeds, embeddings.T) / self.temperature

        # Create labels (positive indices)
        if positive_pairs is not None:
            pos_labels = positive_pairs[:len(anchor_embeds), 1]
        else:
            pos_labels = torch.arange(len(anchor_embeds), len(anchor_embeds)*2, device=embeddings.device)

        # InfoNCE loss
        loss = F.cross_entropy(all_similarities, pos_labels)
        return loss

    def _standard_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard contrastive loss with margin."""
        if positive_pairs is None or negative_pairs is None:
            # Generate pairs from labels
            if labels is None:
                raise ValueError("Either pairs or labels must be provided")
            positive_pairs, negative_pairs = self._generate_pairs_from_labels(labels)

        # At this point, pairs are guaranteed to be tensors
        pos_pairs: torch.Tensor = positive_pairs
        neg_pairs: torch.Tensor = negative_pairs

        # Compute distances for positive pairs
        pos_distances = torch.norm(
            embeddings[pos_pairs[:, 0]] - embeddings[pos_pairs[:, 1]],
            p=2, dim=-1
        )

        # Compute distances for negative pairs
        neg_distances = torch.norm(
            embeddings[neg_pairs[:, 0]] - embeddings[neg_pairs[:, 1]],
            p=2, dim=-1
        )

        # Contrastive loss
        pos_loss = pos_distances.pow(2)
        neg_loss = F.relu(self.margin - neg_distances).pow(2)

        return (pos_loss.mean() + neg_loss.mean()) / 2

    def _triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Triplet loss implementation."""
        return F.triplet_margin_loss(
            anchor=embeddings,
            positive=embeddings,  # Simplified - would need proper positive mining
            negative=embeddings,  # Simplified - would need proper negative mining
            margin=self.margin
        )

    def _generate_pairs_from_labels(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate positive and negative pairs from labels."""
        batch_size = labels.shape[0]
        positive_pairs = []
        negative_pairs = []

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if labels[i] == labels[j]:
                    positive_pairs.append([i, j])
                else:
                    negative_pairs.append([i, j])

        positive_pairs = torch.tensor(positive_pairs, device=labels.device)
        negative_pairs = torch.tensor(negative_pairs, device=labels.device)

        return positive_pairs, negative_pairs


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]

        Returns:
            Focal loss value
        """
        # Flatten if needed
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization.

    Prevents the model from becoming too confident on training data.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class DiversityLoss(nn.Module):
    """
    Diversity loss to encourage diverse representations in MoE models.

    This loss encourages different experts to learn diverse representations.
    """

    def __init__(self, similarity_metric: str = "cosine", diversity_weight: float = 1.0):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.diversity_weight = diversity_weight

    def forward(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute diversity loss across expert outputs.

        Args:
            expert_outputs: List of expert outputs [batch_size, hidden_dim]

        Returns:
            Diversity loss encouraging different experts to be different
        """
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        device = expert_outputs[0].device
        diversity_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                expert_i = expert_outputs[i]
                expert_j = expert_outputs[j]

                if self.similarity_metric == "cosine":
                    # Cosine similarity - we want this to be low (diverse)
                    similarity = F.cosine_similarity(expert_i, expert_j, dim=-1).mean()
                elif self.similarity_metric == "l2":
                    # L2 distance - we want experts to be far apart
                    distance = torch.norm(expert_i - expert_j, p=2, dim=-1).mean()
                    similarity = 1.0 / (1.0 + distance)  # Convert to similarity (high=bad)
                else:
                    # Dot product similarity
                    similarity = (expert_i * expert_j).sum(dim=-1).mean()

                diversity_loss += similarity
                num_pairs += 1

        if num_pairs > 0:
            return self.diversity_weight * diversity_loss / num_pairs
        else:
            device = expert_outputs[0].device if expert_outputs else 'cpu'
            return torch.tensor(0.0, device=device)


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for MoE routing and other auxiliary objectives.

    This implements various auxiliary losses commonly used in MoE models.
    """

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

    def load_balancing_loss(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int
    ) -> torch.Tensor:
        """
        Load balancing loss to encourage uniform expert usage.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size * seq_len, top_k]
            num_experts: Total number of experts

        Returns:
            Load balancing loss
        """
        # Gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Expert usage frequency
        expert_mask = F.one_hot(expert_indices, num_experts).float()
        expert_usage = expert_mask.sum(dim=0).sum(dim=0)

        # Gate probability sums
        gate_prob_sums = gate_probs.sum(dim=0)

        # Load balancing loss (CV^2 - coefficient of variation squared)
        total_tokens = gate_logits.shape[0] * expert_indices.shape[1]
        load_loss = num_experts * torch.sum(gate_prob_sums * expert_usage) / (total_tokens ** 2)

        # Cap load balancing loss to prevent runaway values
        load_loss = torch.clamp(load_loss, max=10.0)

        return load_loss

    def router_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Router Z-loss for numerical stability.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]

        Returns:
            Router Z-loss
        """
        # Clip gate logits to prevent extreme values
        gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)

        # Z-loss encourages smaller logits to prevent overflow
        logsumexp_vals = torch.logsumexp(gate_logits, dim=-1)
        z_loss = torch.mean(logsumexp_vals ** 2)

        # Cap the z-loss to prevent runaway values
        z_loss = torch.clamp(z_loss, max=100.0)

        return z_loss

    def expert_diversity_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Expert diversity loss to encourage specialization.

        Args:
            expert_outputs: List of expert outputs

        Returns:
            Expert diversity loss
        """
        diversity_loss_fn = DiversityLoss()
        return diversity_loss_fn(expert_outputs)

    def forward(
        self,
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        num_experts: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses.

        Returns:
            Dictionary of auxiliary losses
        """
        losses = {}

        if gate_logits is not None and expert_indices is not None and num_experts is not None:
            losses['load_balancing'] = self.load_balancing_weight * self.load_balancing_loss(
                gate_logits, expert_indices, num_experts
            )

        if gate_logits is not None:
            losses['router_z'] = self.router_z_weight * self.router_z_loss(gate_logits)

        if expert_outputs is not None:
            losses['expert_diversity'] = self.expert_diversity_weight * self.expert_diversity_loss(expert_outputs)

        return losses


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for semi-supervised learning and augmentation consistency.

    Encourages the model to produce consistent predictions for augmented versions
    of the same input.
    """

    def __init__(
        self,
        consistency_type: str = "mse",
        temperature: float = 1.0,
        threshold: float = 0.95
    ):
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.threshold = threshold

    def forward(
        self,
        outputs_original: torch.Tensor,
        outputs_augmented: torch.Tensor,
        confidence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute consistency loss between original and augmented outputs.

        Args:
            outputs_original: Outputs for original inputs
            outputs_augmented: Outputs for augmented inputs
            confidence_mask: Optional mask for high-confidence predictions

        Returns:
            Consistency loss
        """
        if self.consistency_type == "mse":
            # MSE between softmax outputs
            probs_orig = F.softmax(outputs_original / self.temperature, dim=-1)
            probs_aug = F.softmax(outputs_augmented / self.temperature, dim=-1)
            consistency_loss = F.mse_loss(probs_aug, probs_orig, reduction='none').mean(dim=-1)

        elif self.consistency_type == "kl":
            # KL divergence
            log_probs_orig = F.log_softmax(outputs_original / self.temperature, dim=-1)
            probs_aug = F.softmax(outputs_augmented / self.temperature, dim=-1)
            consistency_loss = F.kl_div(log_probs_orig, probs_aug, reduction='none').sum(dim=-1)

        elif self.consistency_type == "ce":
            # Cross-entropy with original as pseudo-labels
            pseudo_labels = torch.argmax(outputs_original, dim=-1)
            consistency_loss = F.cross_entropy(outputs_augmented, pseudo_labels, reduction='none')

        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")

        # Apply confidence mask if provided
        if confidence_mask is not None:
            consistency_loss = consistency_loss * confidence_mask

        return consistency_loss.mean()


class PerplexityLoss(nn.Module):
    """
    Perplexity-based loss for language modeling evaluation.

    This loss computes perplexity and can be used as an auxiliary loss.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute perplexity and cross-entropy loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]

        Returns:
            Dictionary with 'loss' and 'perplexity'
        """
        # Flatten logits and targets
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index)

        # Compute perplexity
        perplexity = torch.exp(loss)

        return {
            'loss': loss,
            'perplexity': perplexity
        }


class AdaptiveLossScaling(nn.Module):
    """
    Adaptive loss scaling for balancing multiple loss components.

    This module learns to weight different loss components dynamically.
    """

    def __init__(self, num_losses: int, init_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_losses = num_losses

        if init_weights is None:
            init_weights = [1.0] * num_losses

        # Learnable loss weights (in log space for stability)
        self.log_weights = nn.Parameter(torch.tensor(init_weights).log())

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptively weighted loss.

        Args:
            losses: List of individual loss values

        Returns:
            Tuple of (combined weighted loss, normalized weights)
        """
        weights = torch.exp(self.log_weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Compute weighted loss
        weighted_loss_val = sum(w * loss for w, loss in zip(weights, losses))
        # Ensure it's a tensor, not just 0
        if not isinstance(weighted_loss_val, torch.Tensor):
            weighted_loss_val = torch.tensor(0.0, device=weights.device)

        return weighted_loss_val, weights


class CompositeLoss(nn.Module):
    """
    Composite loss that combines multiple loss functions.

    This is a convenient wrapper for combining different loss types.
    """

    def __init__(self, loss_config: Dict[str, Dict]):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for loss_name, config in loss_config.items():
            loss_type = config.pop('type')
            weight = config.pop('weight', 1.0)

            self.weights[loss_name] = weight

            if loss_type == 'focal':
                self.losses[loss_name] = FocalLoss(**config)
            elif loss_type == 'contrastive':
                self.losses[loss_name] = ContrastiveLoss(**config)
            elif loss_type == 'label_smoothing':
                self.losses[loss_name] = LabelSmoothingLoss(**config)
            elif loss_type == 'auxiliary':
                self.losses[loss_name] = AuxiliaryLoss(**config)
            elif loss_type == 'consistency':
                self.losses[loss_name] = ConsistencyLoss(**config)
            elif loss_type == 'diversity':
                self.losses[loss_name] = DiversityLoss(**config)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute all configured losses.

        Args:
            **kwargs: Arguments for different loss functions

        Returns:
            Dictionary of computed losses
        """
        computed_losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.losses.items():
            try:
                if loss_name == 'focal' and 'inputs' in kwargs and 'targets' in kwargs:
                    loss_value = loss_fn(kwargs['inputs'], kwargs['targets'])
                elif loss_name == 'contrastive' and 'embeddings' in kwargs:
                    loss_value = loss_fn(kwargs['embeddings'], kwargs.get('labels'))
                elif loss_name == 'auxiliary':
                    loss_dict = loss_fn(
                        gate_logits=kwargs.get('gate_logits'),
                        expert_indices=kwargs.get('expert_indices'),
                        expert_outputs=kwargs.get('expert_outputs'),
                        num_experts=kwargs.get('num_experts')
                    )
                    loss_value = sum(loss_dict.values())
                    computed_losses.update({f"aux_{k}": v for k, v in loss_dict.items()})
                else:
                    continue  # Skip if required args not available

                computed_losses[loss_name] = loss_value
                total_loss += self.weights[loss_name] * loss_value

            except Exception as e:
                # Skip losses that can't be computed with available inputs
                continue

        computed_losses['total'] = total_loss
        return computed_losses