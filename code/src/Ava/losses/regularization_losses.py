"""
Regularization Loss Functions

This module contains regularization and anti-collapse mechanisms:
- N-gram repetition penalties
- Immediate sequence repetition detection
- Diversity losses for MoE
- Contrastive learning
- Consistency regularization

Consolidated from: repetition_penalty_loss.py, advanced_losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class NGramRepetitionPenalty(nn.Module):
    """
    Penalizes n-gram repetitions in generated sequences.

    This helps prevent mode collapse where models learn to repeat
    the same tokens/phrases instead of generating diverse text.

    From: repetition_penalty_loss.py
    """

    def __init__(
        self,
        ngram_size: int = 3,
        penalty_weight: float = 0.1,
        vocab_size: int = 50257,
        ignore_index: int = -100
    ):
        """
        Initialize n-gram repetition penalty.

        Args:
            ngram_size: Size of n-grams to track (2-4 recommended)
            penalty_weight: Weight of penalty loss (0.01-0.1)
            vocab_size: Vocabulary size
            ignore_index: Token ID to ignore (padding)
        """
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
        """
        Compute penalty for repeated n-grams.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Penalty value (higher = more repetition)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len < self.ngram_size:
            return torch.tensor(0.0, device=device)

        # Track unique n-grams and repetition counts
        total_penalty = torch.tensor(0.0, device=device)
        total_ngrams = 0

        for b in range(batch_size):
            sequence = token_ids[b]
            mask = attention_mask[b] if attention_mask is not None else None

            # Extract n-grams
            ngrams = []
            for i in range(seq_len - self.ngram_size + 1):
                # Skip if any token in ngram is masked
                if mask is not None and mask[i:i+self.ngram_size].sum() < self.ngram_size:
                    continue

                ngram = tuple(sequence[i:i+self.ngram_size].tolist())
                ngrams.append(ngram)

            if len(ngrams) == 0:
                continue

            # Count repetitions
            from collections import Counter
            ngram_counts = Counter(ngrams)

            # Penalize repeated n-grams
            for ngram, count in ngram_counts.items():
                if count > 1:
                    # Penalty increases with repetition count
                    # count=2: penalty=1, count=3: penalty=3, count=4: penalty=6
                    repetition_penalty = (count - 1) * count / 2
                    total_penalty = total_penalty + repetition_penalty

            total_ngrams += len(ngrams)

        # Normalize by number of n-grams
        if total_ngrams > 0:
            normalized_penalty = total_penalty / total_ngrams
        else:
            normalized_penalty = torch.tensor(0.0, device=device)

        return normalized_penalty * self.penalty_weight

    def compute_token_diversity_penalty(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Penalize low entropy (non-diverse) predictions.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax

        Returns:
            Diversity penalty (lower entropy = higher penalty)
        """
        # Apply temperature
        scaled_logits = logits / temperature

        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Compute entropy (higher entropy = more diverse)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize entropy (max entropy = log(vocab_size))
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        # Penalty for low entropy (1 - entropy)
        # High entropy (diverse) = low penalty
        # Low entropy (repetitive) = high penalty
        diversity_penalty = (1.0 - normalized_entropy).mean()

        return diversity_penalty * self.penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Compute repetition penalties.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with penalty losses
        """
        # Compute n-gram repetition penalty on targets
        ngram_penalty = self.compute_ngram_repetition_penalty(
            targets, attention_mask
        )

        # Compute diversity penalty on predictions
        diversity_penalty = self.compute_token_diversity_penalty(logits)

        # Total penalty
        total_penalty = ngram_penalty + diversity_penalty

        return {
            'ngram_penalty': ngram_penalty,
            'diversity_penalty': diversity_penalty,
            'total_repetition_penalty': total_penalty,
            'penalty_weight': self.penalty_weight
        }


class SequenceRepetitionDetector(nn.Module):
    """
    Detects and heavily penalizes immediate token repetition.

    This is a stronger version that specifically targets the
    "time time time" style repetition.

    From: repetition_penalty_loss.py
    """

    def __init__(
        self,
        penalty_weight: float = 1.0,
        max_repeat_length: int = 10
    ):
        """
        Initialize sequence repetition detector.

        Args:
            penalty_weight: Weight of penalty (higher = stronger)
            max_repeat_length: Maximum repetition sequence to detect
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.max_repeat_length = max_repeat_length

    def detect_immediate_repetition(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Detect immediate token repetition (e.g., "time time time").

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Penalty value
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        total_penalty = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            sequence = token_ids[b]
            mask = attention_mask[b] if attention_mask is not None else None

            consecutive_count = 1
            max_consecutive = 1

            for i in range(1, seq_len):
                # Skip masked tokens
                if mask is not None and mask[i] == 0:
                    continue

                # Check if same as previous token
                if sequence[i] == sequence[i-1]:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1

            # Heavy penalty for consecutive repetitions
            # 2 repeats: penalty=1, 3 repeats: penalty=4, 4 repeats: penalty=9
            if max_consecutive > 1:
                repetition_penalty = (max_consecutive - 1) ** 2
                total_penalty = total_penalty + repetition_penalty

        # Average over batch
        normalized_penalty = total_penalty / batch_size

        return normalized_penalty * self.penalty_weight

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Detect and penalize sequence repetition.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with repetition penalties
        """
        immediate_penalty = self.detect_immediate_repetition(
            token_ids, attention_mask
        )

        return {
            'immediate_repetition_penalty': immediate_penalty,
            'repetition_detector_weight': self.penalty_weight
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning representations by contrasting positive and negative pairs.

    This implementation supports both standard contrastive loss and InfoNCE-style loss.

    From: advanced_losses.py
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


class DiversityLoss(nn.Module):
    """
    Diversity loss to encourage diverse representations in MoE models.

    This loss encourages different experts to learn diverse representations.

    From: advanced_losses.py
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


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for semi-supervised learning and augmentation consistency.

    Encourages the model to produce consistent predictions for augmented versions
    of the same input.

    From: advanced_losses.py
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
