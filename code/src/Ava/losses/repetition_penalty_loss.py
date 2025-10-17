"""
N-gram Repetition Penalty Loss

This module implements losses that explicitly penalize repetitive token patterns,
preventing mode collapse where the model learns to repeat tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NGramRepetitionPenalty(nn.Module):
    """
    Penalizes n-gram repetitions in generated sequences.

    This helps prevent mode collapse where models learn to repeat
    the same tokens/phrases instead of generating diverse text.
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
    ) -> Dict[str, torch.Tensor]:
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
    "time time time" style repetition we're seeing.
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
    ) -> Dict[str, torch.Tensor]:
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
