"""
Unified Loss Module for Ava AI Training Pipeline

This module combines all loss functions into a single, cohesive system:
- DeepSeek-style loss (temperature-scaled cross-entropy, multi-token prediction, MoE balancing)
- Adaptive MTP loss (confidence-weighted multi-token prediction)
- Repetition penalties (n-gram, immediate repetition, EOS penalties)
- Advanced losses (focal, contrastive, diversity, auxiliary)
- Anti-repetition loss

The unified loss can be configured to use any combination of these components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from collections import Counter

# Import base loss components
from .deepseek_loss import (
    MultiTokenPredictionLoss,
    TemperatureScaledCrossEntropy,
    AuxiliaryFreeMoEBalancer
)
from .adaptive_mtp_loss import AdaptiveMTPLoss
from .repetition_penalty_loss import NGramRepetitionPenalty, SequenceRepetitionDetector
from .advanced_losses import (
    ContrastiveLoss,
    FocalLoss,
    LabelSmoothingLoss,
    DiversityLoss,
    AuxiliaryLoss,
    ConsistencyLoss,
    PerplexityLoss,
    AdaptiveLossScaling,
    CompositeLoss
)


class UnifiedLoss(nn.Module):
    """
    Unified loss function combining all available loss components.

    This is the main loss class for the Ava AI training pipeline, providing
    a single interface to all loss functions with flexible configuration.

    Features:
    - Temperature-scaled cross-entropy with adaptive temperature
    - Multi-token prediction (both DeepSeek-style and Adaptive MTP)
    - N-gram repetition penalties
    - Immediate repetition detection
    - EOS token penalties
    - MoE load balancing (auxiliary-free)
    - Diversity and contrastive losses
    - Label smoothing
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        # Primary loss settings
        primary_loss_type: str = "deepseek",  # "deepseek", "adaptive_mtp", "standard"
        ignore_index: int = -100,
        # Temperature scaling
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        label_smoothing: float = 0.1,
        # Multi-token prediction
        use_mtp: bool = False,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1,
        mtp_type: str = "deepseek",  # "deepseek" or "adaptive"
        # Adaptive MTP specific
        use_confidence_weighting: bool = True,
        confidence_reg_strength: float = 0.01,
        # Repetition penalties
        use_ngram_penalty: bool = True,
        ngram_size: int = 4,
        ngram_penalty_weight: float = 0.1,
        use_immediate_repetition_penalty: bool = True,
        immediate_repetition_weight: float = 0.5,
        # EOS penalties
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        min_sequence_length: int = 20,
        eos_penalty_weight: float = 0.05,
        # MoE balancing
        num_experts: Optional[int] = None,
        use_moe_balancing: bool = False,
        gradient_balance_weight: float = 0.1,
        # Advanced losses
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_diversity_loss: bool = False,
        diversity_weight: float = 0.01,
        # Auxiliary losses
        use_auxiliary_loss: bool = False,
        load_balancing_weight: float = 0.0001,
        router_z_weight: float = 0.001,
    ):
        """
        Initialize unified loss function.

        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension (required for MTP)
            primary_loss_type: Type of primary loss ("deepseek", "adaptive_mtp", "standard")
            ignore_index: Token ID to ignore in loss calculation
            initial_temperature: Starting temperature for scaling
            adaptive_temperature: Whether to adapt temperature during training
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_mtp: Whether to use multi-token prediction
            num_future_tokens: Number of future tokens to predict
            mtp_weight: Weight for MTP loss
            mtp_type: Type of MTP ("deepseek" or "adaptive")
            use_confidence_weighting: Whether to weight MTP by confidence
            confidence_reg_strength: Strength of confidence regularization
            use_ngram_penalty: Whether to penalize n-gram repetitions
            ngram_size: Size of n-grams to track
            ngram_penalty_weight: Weight for n-gram penalty
            use_immediate_repetition_penalty: Whether to penalize immediate repetition
            immediate_repetition_weight: Weight for immediate repetition penalty
            eos_token_id: EOS token ID for penalties
            pad_token_id: Padding token ID
            min_sequence_length: Minimum length before allowing EOS
            eos_penalty_weight: Weight for EOS penalty
            num_experts: Number of experts (for MoE)
            use_moe_balancing: Whether to use MoE balancing
            gradient_balance_weight: Weight for gradient-based balancing
            use_focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            use_diversity_loss: Whether to use diversity loss
            diversity_weight: Weight for diversity loss
            use_auxiliary_loss: Whether to use auxiliary loss
            load_balancing_weight: Weight for load balancing loss
            router_z_weight: Weight for router z-loss
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.primary_loss_type = primary_loss_type
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Initialize primary loss based on type
        if primary_loss_type == "deepseek":
            # DeepSeek-style loss with temperature scaling
            self.main_loss = TemperatureScaledCrossEntropy(
                initial_temperature=initial_temperature,
                adaptive_temperature=adaptive_temperature,
                label_smoothing=label_smoothing,
                vocab_size=vocab_size,
                eos_token_id=eos_token_id,
                min_sequence_length=min_sequence_length,
                eos_penalty_weight=eos_penalty_weight
            )
        elif primary_loss_type == "adaptive_mtp":
            # Adaptive MTP as primary loss
            if hidden_size is None:
                raise ValueError("hidden_size required for adaptive_mtp loss")
            self.main_loss = AdaptiveMTPLoss(
                vocab_size=vocab_size,
                primary_loss_weight=1.0,
                additional_loss_base_weight=mtp_weight,
                confidence_reg_strength=confidence_reg_strength,
                use_confidence_weighting=use_confidence_weighting,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index
            )
        else:
            # Standard cross-entropy with label smoothing
            if label_smoothing > 0:
                self.main_loss = LabelSmoothingLoss(
                    num_classes=vocab_size,
                    smoothing=label_smoothing
                )
            else:
                self.main_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Multi-token prediction
        self.use_mtp = use_mtp
        self.mtp_type = mtp_type
        if use_mtp:
            if hidden_size is None:
                raise ValueError("hidden_size required for multi-token prediction")

            if mtp_type == "deepseek":
                self.mtp_loss = MultiTokenPredictionLoss(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_future_tokens=num_future_tokens,
                    mtp_weight=mtp_weight
                )
            elif mtp_type == "adaptive":
                self.mtp_loss = AdaptiveMTPLoss(
                    vocab_size=vocab_size,
                    primary_loss_weight=0.0,  # Only use MTP component
                    additional_loss_base_weight=mtp_weight,
                    confidence_reg_strength=confidence_reg_strength,
                    use_confidence_weighting=use_confidence_weighting,
                    label_smoothing=label_smoothing,
                    ignore_index=ignore_index
                )

        # N-gram repetition penalty
        self.use_ngram_penalty = use_ngram_penalty
        if use_ngram_penalty:
            self.ngram_penalty = NGramRepetitionPenalty(
                ngram_size=ngram_size,
                penalty_weight=ngram_penalty_weight,
                vocab_size=vocab_size,
                ignore_index=ignore_index
            )

        # Immediate repetition detector
        self.use_immediate_repetition_penalty = use_immediate_repetition_penalty
        if use_immediate_repetition_penalty:
            self.immediate_repetition_detector = SequenceRepetitionDetector(
                penalty_weight=immediate_repetition_weight
            )

        # MoE load balancing
        self.use_moe_balancing = use_moe_balancing and num_experts is not None
        if self.use_moe_balancing:
            assert num_experts is not None
            self.moe_balancer = AuxiliaryFreeMoEBalancer(
                num_experts=num_experts,
                gradient_balance_weight=gradient_balance_weight
            )

        # Focal loss
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma
            )

        # Diversity loss
        self.use_diversity_loss = use_diversity_loss
        if use_diversity_loss:
            self.diversity_loss = DiversityLoss(
                diversity_weight=diversity_weight
            )

        # Auxiliary loss
        self.use_auxiliary_loss = use_auxiliary_loss
        if use_auxiliary_loss:
            self.auxiliary_loss = AuxiliaryLoss(
                load_balancing_weight=load_balancing_weight,
                router_z_weight=router_z_weight
            )

    def compute_main_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        primary_logits: Optional[torch.Tensor] = None,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        mtp_active: bool = False
    ) -> Dict[str, Any]:
        """Compute primary loss based on configured type."""
        if self.primary_loss_type == "deepseek":
            # DeepSeek-style temperature-scaled cross-entropy
            result = self.main_loss(logits, targets, attention_mask)
            return {
                'loss': result['loss'],
                'temperature': result.get('temperature', 1.0)
            }
        elif self.primary_loss_type == "adaptive_mtp":
            # Adaptive MTP as primary loss
            if primary_logits is None:
                primary_logits = logits
            result = self.main_loss(
                primary_logits=primary_logits,
                targets=targets,
                additional_logits=additional_logits,
                confidence_scores=confidence_scores,
                attention_mask=attention_mask,
                mtp_active=mtp_active
            )
            return result
        else:
            # Standard cross-entropy or label smoothing
            if isinstance(self.main_loss, LabelSmoothingLoss):
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = self.main_loss(logits_flat, targets_flat)
            else:
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = self.main_loss(logits_flat, targets_flat)

            return {'loss': loss}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        # Adaptive MTP specific
        primary_logits: Optional[torch.Tensor] = None,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        mtp_active: bool = False,
        # MoE specific
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        # Control flags
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute unified loss with all configured components.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            hidden_states: Hidden states for MTP [batch_size, seq_len, hidden_size]
            primary_logits: Primary token logits (for Adaptive MTP)
            additional_logits: Additional token logits (for Adaptive MTP)
            confidence_scores: Confidence scores (for Adaptive MTP)
            mtp_active: Whether MTP was activated
            gate_logits: MoE gate logits [batch_size * seq_len, num_experts]
            expert_indices: Selected experts [batch_size * seq_len, top_k]
            expert_outputs: Expert outputs for diversity loss
            return_detailed: Whether to return detailed loss breakdown

        Returns:
            If return_detailed=False: Total loss tensor
            If return_detailed=True: Dictionary with all loss components
        """
        losses = {}

        # 1. Compute main loss
        main_result = self.compute_main_loss(
            logits=logits,
            targets=targets,
            attention_mask=attention_mask,
            primary_logits=primary_logits,
            additional_logits=additional_logits,
            confidence_scores=confidence_scores,
            mtp_active=mtp_active
        )

        losses['main_loss'] = main_result['loss']
        total_loss = main_result['loss']

        # Store additional main loss info
        for key, value in main_result.items():
            if key != 'loss':
                losses[f'main_{key}'] = value

        # 2. Add multi-token prediction loss (if not already primary)
        if self.use_mtp and self.primary_loss_type != "adaptive_mtp":
            if hidden_states is None:
                losses['mtp_warning'] = "MTP enabled but hidden_states not provided"
            else:
                if self.mtp_type == "deepseek":
                    mtp_result = self.mtp_loss(hidden_states, targets, attention_mask)
                    losses['mtp_loss'] = mtp_result['mtp_loss']
                    total_loss = total_loss + mtp_result['mtp_loss']
                    losses['mtp_per_token_losses'] = mtp_result['per_token_losses']
                elif self.mtp_type == "adaptive":
                    if primary_logits is not None and additional_logits is not None:
                        mtp_result = self.mtp_loss(
                            primary_logits=primary_logits,
                            targets=targets,
                            additional_logits=additional_logits,
                            confidence_scores=confidence_scores,
                            attention_mask=attention_mask,
                            mtp_active=mtp_active
                        )
                        losses['adaptive_mtp_loss'] = mtp_result['loss']
                        total_loss = total_loss + mtp_result['loss']
                        losses.update({f'adaptive_mtp_{k}': v for k, v in mtp_result.items() if k != 'loss'})

        # 3. Add n-gram repetition penalty
        if self.use_ngram_penalty:
            ngram_result = self.ngram_penalty(logits, targets, attention_mask)
            losses['ngram_penalty'] = ngram_result['total_repetition_penalty']
            total_loss = total_loss + ngram_result['total_repetition_penalty']
            losses['ngram_penalty_breakdown'] = {
                'ngram': ngram_result['ngram_penalty'],
                'diversity': ngram_result['diversity_penalty']
            }

        # 4. Add immediate repetition penalty
        if self.use_immediate_repetition_penalty:
            immediate_result = self.immediate_repetition_detector(targets, attention_mask)
            losses['immediate_repetition_penalty'] = immediate_result['immediate_repetition_penalty']
            total_loss = total_loss + immediate_result['immediate_repetition_penalty']

        # 5. Add MoE balancing
        if self.use_moe_balancing and gate_logits is not None and expert_indices is not None:
            balance_result = self.moe_balancer(gate_logits, expert_indices, expert_outputs)
            losses['moe_balancing_term'] = balance_result['balancing_term']
            total_loss = total_loss + balance_result['balancing_term']
            losses['moe_balance_loss'] = balance_result['balance_loss']
            losses['moe_expert_balance_ratio'] = balance_result['expert_balance_ratio']
            losses['moe_cv'] = balance_result['coefficient_of_variation']

        # 6. Add focal loss
        if self.use_focal_loss:
            focal_result = self.focal_loss(logits, targets)
            losses['focal_loss'] = focal_result
            # Focal loss replaces main loss, don't add to total

        # 7. Add diversity loss
        if self.use_diversity_loss and expert_outputs is not None:
            diversity_result = self.diversity_loss(expert_outputs)
            losses['diversity_loss'] = diversity_result
            total_loss = total_loss + diversity_result

        # 8. Add auxiliary loss
        if self.use_auxiliary_loss:
            aux_result = self.auxiliary_loss(
                gate_logits=gate_logits,
                expert_indices=expert_indices,
                expert_outputs=expert_outputs,
                num_experts=self.moe_balancer.num_experts if self.use_moe_balancing else None
            )
            for key, value in aux_result.items():
                losses[f'aux_{key}'] = value
                total_loss = total_loss + value

        # Store total loss
        losses['total_loss'] = total_loss

        if return_detailed:
            return losses
        else:
            return total_loss

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics from all loss components."""
        stats = {}

        # Adaptive MTP statistics
        if self.primary_loss_type == "adaptive_mtp":
            stats['adaptive_mtp'] = self.main_loss.get_statistics()
        elif self.use_mtp and self.mtp_type == "adaptive":
            stats['adaptive_mtp'] = self.mtp_loss.get_statistics()

        # MoE balancing statistics
        if self.use_moe_balancing:
            stats['moe_balancing'] = {
                'expert_counts': self.moe_balancer.expert_counts.tolist(),
                'expert_scores': self.moe_balancer.expert_scores.tolist(),
                'total_tokens': self.moe_balancer.total_tokens.item()
            }

        return stats

    def reset_statistics(self):
        """Reset statistics in all loss components."""
        if self.primary_loss_type == "adaptive_mtp":
            self.main_loss.reset_statistics()
        elif self.use_mtp and self.mtp_type == "adaptive":
            self.mtp_loss.reset_statistics()


# Convenience function for creating unified loss
def create_unified_loss(config: Dict[str, Any]) -> UnifiedLoss:
    """
    Create a unified loss from a configuration dictionary.

    Args:
        config: Configuration dictionary with loss parameters

    Returns:
        Configured UnifiedLoss instance
    """
    return UnifiedLoss(**config)


# Export all loss classes
__all__ = [
    'UnifiedLoss',
    'create_unified_loss',
    # Base components
    'MultiTokenPredictionLoss',
    'TemperatureScaledCrossEntropy',
    'AuxiliaryFreeMoEBalancer',
    'AdaptiveMTPLoss',
    'NGramRepetitionPenalty',
    'SequenceRepetitionDetector',
    'ContrastiveLoss',
    'FocalLoss',
    'LabelSmoothingLoss',
    'DiversityLoss',
    'AuxiliaryLoss',
    'ConsistencyLoss',
    'PerplexityLoss',
    'AdaptiveLossScaling',
    'CompositeLoss',
]
