"""
Anti-Repetition Loss Function

Adds penalties for:
1. Token repetition (n-gram overlap)
2. EOS token over-use
3. Low diversity

This wraps around your existing loss and adds these penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AntiRepetitionLoss(nn.Module):
    """
    Enhanced loss function that penalizes repetitive outputs.

    Combines:
    - Base cross-entropy loss
    - N-gram repetition penalty
    - EOS token over-use penalty
    - Diversity bonus
    """

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
        """
        Args:
            vocab_size: Size of vocabulary
            eos_token_id: ID of EOS token to penalize
            pad_token_id: ID of padding token (ignored in calculations)
            repetition_penalty_weight: Weight for n-gram repetition penalty (0.1 = 10%)
            eos_penalty_weight: Weight for EOS over-use penalty (0.05 = 5%)
            diversity_bonus_weight: Weight for diversity bonus (0.05 = 5%)
            ngram_size: Size of n-grams to check (default: 4)
            ignore_index: Token ID to ignore in loss calculation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.eos_penalty_weight = eos_penalty_weight
        self.diversity_bonus_weight = diversity_bonus_weight
        self.ngram_size = ngram_size
        self.ignore_index = ignore_index

        # Base loss
        self.base_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def calculate_ngram_repetition(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate n-gram repetition ratio for each sequence.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            repetition_scores: [batch_size] - ratio of repeated n-grams
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len < self.ngram_size:
            return torch.zeros(batch_size, device=device)

        repetition_scores = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            # Skip if too short
            if len(tokens) < self.ngram_size:
                repetition_scores.append(0.0)
                continue

            # Extract n-grams
            ngrams = []
            for j in range(len(tokens) - self.ngram_size + 1):
                ngram = tuple(tokens[j:j+self.ngram_size].tolist())
                # Skip if contains padding
                if self.pad_token_id not in ngram:
                    ngrams.append(ngram)

            if len(ngrams) == 0:
                repetition_scores.append(0.0)
                continue

            # Calculate repetition ratio
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            repetition = 1.0 - (unique_ngrams / total_ngrams)
            repetition_scores.append(repetition)

        return torch.tensor(repetition_scores, device=device)

    def calculate_eos_penalty(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate penalty for excessive EOS token usage.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            eos_penalties: [batch_size] - ratio of EOS tokens
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        eos_ratios = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            if len(tokens) == 0:
                eos_ratios.append(0.0)
                continue

            # Count EOS tokens (excluding the final legitimate one)
            eos_count = (tokens == self.eos_token_id).sum().item()

            # If EOS appears, it should ideally be only at the end
            # Penalize if it appears multiple times or early
            if eos_count > 1:
                # Multiple EOS tokens = problem
                penalty = eos_count / len(tokens)
            elif eos_count == 1 and tokens[-1] != self.eos_token_id:
                # EOS in middle of sequence = problem
                penalty = 0.5 / len(tokens)
            else:
                # Normal case: single EOS at end (or no EOS)
                penalty = 0.0

            eos_ratios.append(penalty)

        return torch.tensor(eos_ratios, device=device)

    def calculate_diversity_bonus(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate diversity bonus (negative penalty) for diverse outputs.
        Uses unique token ratio as a simple diversity measure.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            diversity_scores: [batch_size] - higher = more diverse (bonus)
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        diversity_scores = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            if len(tokens) == 0:
                diversity_scores.append(0.0)
                continue

            # Filter out special tokens
            valid_tokens = tokens[
                (tokens != self.pad_token_id) &
                (tokens != self.eos_token_id)
            ]

            if len(valid_tokens) == 0:
                diversity_scores.append(0.0)
                continue

            # Diversity = ratio of unique tokens
            unique_tokens = len(torch.unique(valid_tokens))
            diversity = unique_tokens / len(valid_tokens)
            diversity_scores.append(diversity)

        return torch.tensor(diversity_scores, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Calculate combined loss with anti-repetition penalties.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            return_components: If True, return dict with loss components

        Returns:
            loss: Scalar loss
            components: Dict with loss components (if return_components=True)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Base cross-entropy loss
        base_loss_per_token = self.base_loss(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )
        base_loss_per_token = base_loss_per_token.view(batch_size, seq_len)

        # Average over valid tokens
        if attention_mask is not None:
            # Mask out padding
            mask = (labels != self.ignore_index).float()
            base_loss = (base_loss_per_token * mask).sum() / mask.sum()
        else:
            base_loss = base_loss_per_token.mean()

        # 2. Get predicted tokens for penalty calculation
        predicted_tokens = logits.argmax(dim=-1)  # [batch_size, seq_len]

        # 3. Calculate penalties
        repetition_scores = self.calculate_ngram_repetition(
            predicted_tokens, attention_mask
        )
        repetition_penalty = repetition_scores.mean()

        eos_penalties = self.calculate_eos_penalty(
            predicted_tokens, attention_mask
        )
        eos_penalty = eos_penalties.mean()

        diversity_scores = self.calculate_diversity_bonus(
            predicted_tokens, attention_mask
        )
        diversity_bonus = diversity_scores.mean()

        # 4. Combined loss
        # Base loss + repetition penalty + EOS penalty - diversity bonus
        total_loss = (
            base_loss +
            self.repetition_penalty_weight * repetition_penalty +
            self.eos_penalty_weight * eos_penalty -
            self.diversity_bonus_weight * diversity_bonus
        )

        if return_components:
            components = {
                'base_loss': base_loss.item(),
                'repetition_penalty': repetition_penalty.item(),
                'eos_penalty': eos_penalty.item(),
                'diversity_bonus': diversity_bonus.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, components

        return total_loss, None


class AdaptiveAntiRepetitionLoss(AntiRepetitionLoss):
    """
    Adaptive version that adjusts penalty weights based on training progress.

    Starts with high penalties and gradually reduces them as model improves.
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        pad_token_id: int,
        initial_repetition_weight: float = 0.2,  # Start high
        final_repetition_weight: float = 0.05,    # End low
        initial_eos_weight: float = 0.1,
        final_eos_weight: float = 0.02,
        warmup_steps: int = 10000,
        **kwargs
    ):
        """
        Args:
            warmup_steps: Number of steps to linearly reduce penalties
            Other args same as AntiRepetitionLoss
        """
        super().__init__(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty_weight=initial_repetition_weight,
            eos_penalty_weight=initial_eos_weight,
            **kwargs
        )

        self.initial_repetition_weight = initial_repetition_weight
        self.final_repetition_weight = final_repetition_weight
        self.initial_eos_weight = initial_eos_weight
        self.final_eos_weight = final_eos_weight
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def update_weights(self, step: int):
        """Update penalty weights based on training step."""
        self.current_step = step

        if step >= self.warmup_steps:
            # Use final weights
            self.repetition_penalty_weight = self.final_repetition_weight
            self.eos_penalty_weight = self.final_eos_weight
        else:
            # Linear interpolation
            progress = step / self.warmup_steps

            self.repetition_penalty_weight = (
                self.initial_repetition_weight * (1 - progress) +
                self.final_repetition_weight * progress
            )

            self.eos_penalty_weight = (
                self.initial_eos_weight * (1 - progress) +
                self.final_eos_weight * progress
            )
