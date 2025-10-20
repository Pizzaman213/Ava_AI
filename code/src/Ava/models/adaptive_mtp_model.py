"""
Adaptive Multi-Token Prediction Model

This module implements a model wrapper that adds adaptive multi-token prediction
capabilities to any base transformer model. It intelligently decides whether to
predict multiple future tokens based on confidence scoring.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .confidence_gate import ConfidenceGate
from .prediction_heads import MultiTokenPredictionHeads


@dataclass
class AdaptiveMTPConfig:
    """Configuration for Adaptive Multi-Token Prediction."""

    # Core MTP settings
    num_prediction_heads: int = 3  # Number of future tokens to predict (2-4 recommended)
    confidence_threshold_train: float = 0.6  # Threshold during training
    confidence_threshold_inference: float = 0.7  # Threshold during inference (higher)

    # Confidence gate settings
    gate_hidden_dims: Tuple[int, ...] = (512, 256)
    gate_dropout: float = 0.1
    gate_activation: str = 'gelu'
    use_attention_pooling: bool = False

    # Prediction head settings
    head_type: str = 'linear'  # 'linear' or 'mlp'
    head_intermediate_size: Optional[int] = None
    head_dropout: float = 0.1
    share_projections: bool = False

    # Training settings
    mtp_warmup_epochs: int = 2  # Train only primary head for first N epochs
    confidence_reg_strength: float = 0.01  # Regularization for confident predictions

    # Loss weighting
    use_confidence_weighting: bool = True  # Weight losses by confidence
    primary_loss_weight: float = 1.0  # Primary token always gets full weight
    additional_loss_base_weight: float = 0.1  # Base weight for additional tokens

    # Efficiency settings
    enable_dynamic_prediction: bool = True  # Skip MTP computation when low confidence
    min_confidence_for_computation: float = 0.3  # Don't compute heads below this


class AdaptiveMTPModel(nn.Module):
    """
    Adaptive Multi-Token Prediction Model wrapper.

    This model wraps a base transformer (e.g., EnhancedMoEModel) and adds:
    1. Confidence gating to decide when to predict multiple tokens
    2. Multiple lightweight prediction heads for future positions
    3. Adaptive loss weighting based on confidence scores
    """

    # Type annotations for registered buffers
    current_epoch: torch.Tensor
    training_steps: torch.Tensor
    mtp_activations: torch.Tensor
    total_predictions: torch.Tensor

    def __init__(
        self,
        base_model: nn.Module,
        config: AdaptiveMTPConfig,
        vocab_size: int,
        hidden_size: int,
    ):
        """
        Initialize Adaptive MTP Model.

        Args:
            base_model: Base transformer model (e.g., EnhancedMoEModel)
            config: AdaptiveMTPConfig with MTP settings
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension of the model
        """
        super().__init__()

        self.base_model = base_model
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Confidence gating network
        self.confidence_gate = ConfidenceGate(
            hidden_size=hidden_size,
            gate_hidden_dims=config.gate_hidden_dims,
            dropout=config.gate_dropout,
            activation=config.gate_activation,
            use_attention_pooling=config.use_attention_pooling,
        )

        # Multi-token prediction heads
        self.prediction_heads = MultiTokenPredictionHeads(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=config.num_prediction_heads,
            head_type=config.head_type,
            intermediate_size=config.head_intermediate_size,
            dropout=config.head_dropout,
            share_projections=config.share_projections,
        )

        # Track training progress for warmup
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('training_steps', torch.tensor(0))

        # Statistics tracking
        self.register_buffer('mtp_activations', torch.tensor(0))
        self.register_buffer('total_predictions', torch.tensor(0))

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup tracking."""
        self.current_epoch = torch.tensor(epoch)

    def in_warmup_period(self) -> bool:
        """Check if we're in the warmup period (single-token only)."""
        epoch_val = self.current_epoch.item() if isinstance(self.current_epoch, torch.Tensor) else int(self.current_epoch)
        return epoch_val < self.config.mtp_warmup_epochs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Forward pass with adaptive multi-token prediction.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments for base model

        Returns:
            Dictionary or tuple containing:
                - loss: Total loss (if labels provided)
                - primary_logits: Logits for next token [batch_size, seq_len, vocab_size]
                - additional_logits: List of logits for future positions (if MTP active)
                - confidence_scores: Confidence scores
                - hidden_states: Final hidden states
                - mtp_active: Whether MTP was activated
        """
        # Forward pass through base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Extract hidden states and primary logits
        if isinstance(base_outputs, dict):
            hidden_states = base_outputs.get('hidden_states')
            if hidden_states is None:
                hidden_states = base_outputs.get('last_hidden_state')
            primary_logits = base_outputs.get('logits')
        else:
            # Handle tuple output
            hidden_states = base_outputs[0] if len(base_outputs) > 0 else None
            primary_logits = base_outputs[1] if len(base_outputs) > 1 else None

        # If base model doesn't provide hidden states, we need them
        if hidden_states is None:
            raise ValueError("Base model must return hidden_states for MTP")

        # Get primary logits if not provided by base model
        if primary_logits is None and hasattr(self.base_model, 'lm_head'):
            lm_head = self.base_model.lm_head
            if callable(lm_head):
                primary_logits = lm_head(hidden_states)
            elif isinstance(lm_head, nn.Module):
                primary_logits = lm_head(hidden_states)

        # Compute confidence scores
        confidence_outputs = self.confidence_gate(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_per_token=False  # Get batch-level confidence
        )
        confidence_scores = confidence_outputs['confidence']
        avg_confidence = confidence_outputs['avg_confidence']

        # Determine if we should activate MTP
        mtp_active = (
            not self.in_warmup_period() and
            avg_confidence.item() >= self.config.confidence_threshold_train and
            self.config.enable_dynamic_prediction
        )

        additional_logits = None

        if mtp_active:
            # Predict multiple future tokens
            mtp_outputs = self.prediction_heads(
                hidden_states=hidden_states,
                return_all_logits=True
            )
            additional_logits = mtp_outputs['all_logits']

            # Update statistics
            with torch.no_grad():
                self.mtp_activations += 1
                self.total_predictions += 1
        else:
            # Skip MTP computation for efficiency
            with torch.no_grad():
                self.total_predictions += 1

        # Update training steps
        if self.training:
            self.training_steps += 1

        # Prepare outputs
        outputs = {
            'primary_logits': primary_logits,
            'additional_logits': additional_logits,
            'confidence_scores': confidence_scores,
            'avg_confidence': avg_confidence,
            'hidden_states': hidden_states,
            'mtp_active': mtp_active,
            'num_prediction_heads': self.config.num_prediction_heads if mtp_active else 0,
        }

        # Include base model outputs
        if isinstance(base_outputs, dict):
            for key, value in base_outputs.items():
                if key not in outputs:
                    outputs[key] = value

        if return_dict:
            return outputs
        else:
            return tuple(v for v in outputs.values())

    def generate_multi_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Union[torch.Tensor, bool, int, None]]:
        """
        Generate multiple tokens at inference time with confidence gating.

        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling

        Returns:
            Dictionary with generated tokens and metadata
        """
        # Check confidence
        confidence_outputs = self.confidence_gate(hidden_states, attention_mask)
        confidence = confidence_outputs['avg_confidence'].item()

        # Use higher threshold for inference
        if confidence >= self.config.confidence_threshold_inference:
            # Predict multiple tokens
            predictions = self.prediction_heads.predict_tokens(
                hidden_states=hidden_states,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            return {
                'predictions': predictions['predictions'],
                'num_tokens': predictions['num_tokens'],
                'confidence': confidence,
                'used_mtp': True,
            }
        else:
            # Fall back to single-token prediction
            return {
                'predictions': None,
                'num_tokens': 0,
                'confidence': confidence,
                'used_mtp': False,
            }

    def get_mtp_statistics(self) -> Dict[str, float]:
        """
        Get statistics about MTP usage.

        Returns:
            Dictionary with MTP statistics
        """
        if self.total_predictions > 0:
            mtp_usage_ratio = (self.mtp_activations / self.total_predictions).item()
        else:
            mtp_usage_ratio = 0.0

        confidence_stats = self.confidence_gate.get_statistics()

        return {
            'mtp_usage_ratio': mtp_usage_ratio,
            'mtp_activations': self.mtp_activations.item(),
            'total_predictions': self.total_predictions.item(),
            'current_epoch': self.current_epoch.item(),
            'in_warmup': self.in_warmup_period(),
            **confidence_stats,
        }

    def reset_statistics(self):
        """Reset all tracking statistics."""
        self.mtp_activations.zero_()
        self.total_predictions.zero_()
        self.confidence_gate.reset_statistics()

    def get_config(self) -> AdaptiveMTPConfig:
        """Get the configuration."""
        return self.config

    def get_base_model(self) -> nn.Module:
        """Get the base transformer model."""
        return self.base_model
