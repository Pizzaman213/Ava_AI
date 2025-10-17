"""
Adaptive Multi-Token Prediction Loss

This module implements the loss function for adaptive MTP that:
1. Always computes full loss for primary token prediction
2. Weights additional token losses by confidence scores
3. Includes regularization to encourage confident predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class AdaptiveMTPLoss(nn.Module):
    """
    Loss function for Adaptive Multi-Token Prediction.

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
        """
        Initialize Adaptive MTP Loss.

        Args:
            vocab_size: Size of vocabulary
            primary_loss_weight: Weight for primary token loss (always 1.0)
            additional_loss_base_weight: Base weight for additional tokens before confidence scaling
            confidence_reg_strength: Strength of confidence regularization
            use_confidence_weighting: Whether to weight additional losses by confidence
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation (padding)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.primary_loss_weight = primary_loss_weight
        self.additional_loss_base_weight = additional_loss_base_weight
        self.confidence_reg_strength = confidence_reg_strength
        self.use_confidence_weighting = use_confidence_weighting
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Track statistics
        self.register_buffer('total_primary_loss', torch.tensor(0.0))
        self.register_buffer('total_additional_loss', torch.tensor(0.0))
        self.register_buffer('total_confidence_reg', torch.tensor(0.0))
        self.register_buffer('loss_count', torch.tensor(0))

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(logits_flat, dim=-1)

            # Create smoothed target distribution
            with torch.no_grad():
                smoothed_targets = torch.zeros_like(log_probs)
                smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))
                smoothed_targets.scatter_(
                    1,
                    targets_flat.unsqueeze(1),
                    1.0 - self.label_smoothing
                )

                # Mask padding tokens
                if self.ignore_index >= 0:
                    padding_mask = targets_flat == self.ignore_index
                    smoothed_targets[padding_mask] = 0.0

            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                reduction='none'
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum().clamp(min=1.0)
        else:
            if self.ignore_index >= 0:
                # Count non-ignored tokens
                valid_tokens = (targets_flat != self.ignore_index).float()
                loss = loss.sum() / valid_tokens.sum().clamp(min=1.0)
            else:
                loss = loss.mean()

        return loss

    def compute_confidence_regularization(
        self,
        confidence_scores: torch.Tensor,
        target_mode: str = 'binary'
    ) -> torch.Tensor:
        """
        Regularization to encourage confident predictions.

        Pushes confidence scores away from uncertain middle ground (0.5)
        toward either high confidence (1.0) or low confidence (0.0).

        Args:
            confidence_scores: Confidence scores [batch_size, 1] or [batch_size, seq_len, 1]
            target_mode: 'binary' (push to 0 or 1) or 'high' (push toward 1)

        Returns:
            Regularization loss
        """
        if target_mode == 'binary':
            # Penalize scores near 0.5 (uncertain)
            # Loss is minimal at 0 and 1, maximal at 0.5
            # Use: -log(|2*conf - 1|) which is high when conf ≈ 0.5
            epsilon = 1e-7
            deviation_from_half = torch.abs(2 * confidence_scores - 1).clamp(min=epsilon)
            reg_loss = -torch.log(deviation_from_half).mean()

        elif target_mode == 'high':
            # Encourage high confidence
            # Negative log likelihood of confidence
            epsilon = 1e-7
            reg_loss = -torch.log(confidence_scores.clamp(min=epsilon)).mean()

        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

        return reg_loss

    def forward(
        self,
        primary_logits: torch.Tensor,
        targets: torch.Tensor,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mtp_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute adaptive MTP loss.

        Args:
            primary_logits: Logits for next token [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            additional_logits: List of logits for future positions (if MTP active)
            confidence_scores: Confidence scores [batch_size, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            mtp_active: Whether MTP was activated

        Returns:
            Dictionary containing:
                - loss: Total loss
                - primary_loss: Loss for primary token prediction
                - additional_loss: Loss for additional tokens (if MTP active)
                - confidence_reg: Confidence regularization loss
                - avg_confidence: Average confidence score
                - effective_mtp_weight: Effective weight applied to additional losses
        """
        # Always compute primary token loss
        primary_loss = self.compute_cross_entropy(
            primary_logits, targets, attention_mask
        )
        primary_loss = primary_loss * self.primary_loss_weight

        # Initialize additional loss and regularization
        additional_loss = torch.tensor(0.0, device=primary_logits.device)
        confidence_reg = torch.tensor(0.0, device=primary_logits.device)
        avg_confidence = torch.tensor(0.0, device=primary_logits.device)
        effective_mtp_weight = 0.0

        # Compute additional losses if MTP is active
        if mtp_active and additional_logits is not None:
            num_heads = len(additional_logits)

            # Compute loss for each future position
            head_losses = []
            for i, logits in enumerate(additional_logits):
                # Shift targets for future position (i+1 positions ahead)
                future_offset = i + 1

                # Ensure we have enough positions
                if future_offset >= targets.shape[1]:
                    continue

                # Get shifted targets
                shifted_targets = targets[:, future_offset:]

                # Get corresponding logits (remove last few positions)
                shifted_logits = logits[:, :-future_offset, :]

                # Shifted attention mask
                if attention_mask is not None:
                    shifted_mask = attention_mask[:, future_offset:]
                else:
                    shifted_mask = None

                # Compute loss for this head
                head_loss = self.compute_cross_entropy(
                    shifted_logits, shifted_targets, shifted_mask
                )
                head_losses.append(head_loss)

            # Average losses across heads
            if head_losses:
                additional_loss = torch.stack(head_losses).mean()

                # Apply confidence weighting
                if self.use_confidence_weighting and confidence_scores is not None:
                    avg_confidence = confidence_scores.mean()
                    confidence_weight = avg_confidence.clamp(min=0.0, max=1.0)
                    effective_mtp_weight = (
                        self.additional_loss_base_weight * confidence_weight
                    )
                else:
                    effective_mtp_weight = self.additional_loss_base_weight

                # Weight the additional loss
                additional_loss = additional_loss * effective_mtp_weight

        # Compute confidence regularization
        if confidence_scores is not None and self.confidence_reg_strength > 0:
            confidence_reg = self.compute_confidence_regularization(
                confidence_scores, target_mode='binary'
            )
            confidence_reg = confidence_reg * self.confidence_reg_strength

            if confidence_scores.numel() > 0:
                avg_confidence = confidence_scores.mean()

        # Total loss
        total_loss = primary_loss + additional_loss + confidence_reg

        # Update statistics
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
            'mtp_active': mtp_active,
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get loss statistics for monitoring.

        Returns:
            Dictionary with average losses
        """
        if self.loss_count > 0:
            avg_primary = (self.total_primary_loss / self.loss_count)  # type: ignore[operator]
            avg_additional = (self.total_additional_loss / self.loss_count)  # type: ignore[operator]
            avg_conf_reg = (self.total_confidence_reg / self.loss_count)  # type: ignore[operator]
            # Convert tensors to floats
            avg_primary = avg_primary.item() if isinstance(avg_primary, torch.Tensor) else float(avg_primary)
            avg_additional = avg_additional.item() if isinstance(avg_additional, torch.Tensor) else float(avg_additional)
            avg_conf_reg = avg_conf_reg.item() if isinstance(avg_conf_reg, torch.Tensor) else float(avg_conf_reg)
        else:
            avg_primary = 0.0
            avg_additional = 0.0
            avg_conf_reg = 0.0

        loss_count_val = self.loss_count.item() if isinstance(self.loss_count, torch.Tensor) else int(self.loss_count)
        return {
            'avg_primary_loss': avg_primary,
            'avg_additional_loss': avg_additional,
            'avg_confidence_reg': avg_conf_reg,
            'total_computations': loss_count_val,
        }

    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_primary_loss.zero_()  # type: ignore[attr-defined]
        self.total_additional_loss.zero_()  # type: ignore[attr-defined]
        self.total_confidence_reg.zero_()  # type: ignore[attr-defined]
        self.loss_count.zero_()  # type: ignore[attr-defined]
