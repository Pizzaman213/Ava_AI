"""
Adaptive Multi-Token Prediction (MTP) System

This module implements the complete adaptive multi-token prediction system including:
- ConfidenceGate: Neural network for predicting confidence scores
- MultiTokenPredictionHeads: Lightweight heads for predicting future tokens
- AdaptiveMTPModel: Model wrapper with adaptive MTP capabilities

The system intelligently decides whether to predict multiple future tokens based on
confidence scoring, enabling faster inference when the model is confident.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence Gating Network
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConfidenceGate(nn.Module):
    """
    Neural network that outputs a confidence score (0-1) indicating whether
    the model should predict multiple future tokens.

    The gate learns to predict when the model's next-token prediction will be
    reliable enough to warrant predicting even further ahead.
    """

    # Type annotations for registered buffers
    confidence_sum: torch.Tensor
    confidence_count: torch.Tensor
    high_confidence_count: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        gate_hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'gelu',
        use_attention_pooling: bool = False,
    ):
        """
        Initialize confidence gating network.

        Args:
            hidden_size: Dimension of input hidden states
            gate_hidden_dims: Hidden dimensions for gate MLP layers
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ('gelu', 'relu', 'silu')
            use_attention_pooling: Use attention-based pooling instead of mean
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.use_attention_pooling = use_attention_pooling

        # Optional attention pooling for better representation
        if use_attention_pooling:
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )

        # Build MLP layers for confidence scoring
        layers = []
        prev_dim = hidden_size

        for hidden_dim in gate_hidden_dims:
            if use_layer_norm:
                layers.append(nn.LayerNorm(prev_dim))

            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation function
            if activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection to confidence score
        if use_layer_norm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Confidence in [0, 1]

        self.gate_network = nn.Sequential(*layers)

        # Statistics tracking for analysis
        self.register_buffer('confidence_sum', torch.tensor(0.0))
        self.register_buffer('confidence_count', torch.tensor(0))
        self.register_buffer('high_confidence_count', torch.tensor(0))

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool hidden states to get a single representation for confidence scoring.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] optional mask

        Returns:
            pooled_states: [batch_size, hidden_size]
        """
        if self.use_attention_pooling:
            # Attention-based pooling
            attention_weights = self.attention_pool(hidden_states)  # [B, L, 1]

            if attention_mask is not None:
                # Mask out padding positions
                attention_weights = attention_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    float('-inf')
                )
                attention_weights = F.softmax(attention_weights, dim=1)

            pooled = (hidden_states * attention_weights).sum(dim=1)
        else:
            # Mean pooling
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                # Simple mean pooling
                pooled = hidden_states.mean(dim=1)

        return pooled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_per_token: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute confidence scores for multi-token prediction.

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_per_token: If True, return per-token confidence scores

        Returns:
            Dictionary containing:
                - confidence: Scalar confidence score [batch_size, 1] or [batch_size, seq_len, 1]
                - pooled_hidden: Pooled hidden states (if not per-token)
                - avg_confidence: Average confidence across batch (for logging)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if return_per_token:
            # Compute per-token confidence scores
            # Reshape to [batch_size * seq_len, hidden_size]
            hidden_flat = hidden_states.view(-1, hidden_size)
            confidence_flat = self.gate_network(hidden_flat)  # [B*L, 1]
            confidence = confidence_flat.view(batch_size, seq_len, 1)

            # Apply mask if provided
            if attention_mask is not None:
                confidence = confidence.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    0.0
                )

            pooled_hidden = None
        else:
            # Pool hidden states and compute single confidence per batch
            pooled_hidden = self.pool_hidden_states(hidden_states, attention_mask)
            confidence = self.gate_network(pooled_hidden)  # [B, 1]

        # Track statistics during training
        if self.training:
            with torch.no_grad():
                self.confidence_sum += confidence.sum()
                self.confidence_count += confidence.numel()
                self.high_confidence_count += (confidence > 0.7).sum()

        # Calculate average confidence for logging
        avg_confidence = confidence.mean()

        return {
            'confidence': confidence,
            'pooled_hidden': pooled_hidden,
            'avg_confidence': avg_confidence,
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get confidence statistics for monitoring.

        Returns:
            Dictionary with confidence stats
        """
        if self.confidence_count > 0:
            avg_conf = (self.confidence_sum / self.confidence_count).item()
            high_conf_ratio = (self.high_confidence_count / self.confidence_count).item()
        else:
            avg_conf = 0.0
            high_conf_ratio = 0.0

        return {
            'avg_confidence': avg_conf,
            'high_confidence_ratio': high_conf_ratio,
            'total_predictions': self.confidence_count.item(),
        }

    def reset_statistics(self):
        """Reset tracking statistics."""
        self.confidence_sum.zero_()
        self.confidence_count.zero_()
        self.high_confidence_count.zero_()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-Token Prediction Heads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiTokenPredictionHeads(nn.Module):
    """
    Lightweight prediction heads for predicting multiple future tokens.

    These heads share the same transformer backbone but branch off at the
    final layers to predict different future positions efficiently.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        head_type: str = 'linear',
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        share_projections: bool = False,
        use_layer_norm: bool = True,
    ):
        """
        Initialize multi-token prediction heads.

        Args:
            hidden_size: Dimension of hidden states from backbone
            vocab_size: Size of vocabulary
            num_heads: Number of future positions to predict (2-4 recommended)
            head_type: Type of head ('linear' or 'mlp')
            intermediate_size: Intermediate dimension for MLP heads
            dropout: Dropout probability
            share_projections: Whether to share weights across heads
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_type = head_type
        self.share_projections = share_projections

        # Input layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        # Create prediction heads
        if share_projections:
            # Single shared head for all positions
            self.shared_head: Optional[nn.Module] = self._create_head(
                hidden_size, vocab_size * num_heads,
                intermediate_size, dropout
            )
            self.heads: Optional[nn.ModuleList] = None
        else:
            # Separate head for each future position
            self.heads = nn.ModuleList([
                self._create_head(hidden_size, vocab_size, intermediate_size, dropout)
                for _ in range(num_heads)
            ])
            self.shared_head = None

    def _create_head(
        self,
        input_size: int,
        output_size: int,
        intermediate_size: Optional[int],
        dropout: float
    ) -> nn.Module:
        """
        Create a single prediction head.

        Args:
            input_size: Input dimension
            output_size: Output dimension (vocab_size)
            intermediate_size: Intermediate dimension for MLP
            dropout: Dropout probability

        Returns:
            Prediction head module
        """
        if self.head_type == 'linear':
            # Simple linear projection
            return nn.Linear(input_size, output_size)

        elif self.head_type == 'mlp':
            # Small MLP for richer representations
            if intermediate_size is None:
                intermediate_size = input_size * 2

            return nn.Sequential(
                nn.Linear(input_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, output_size)
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict multiple future tokens from hidden states.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            return_all_logits: Return logits for all positions separately

        Returns:
            Dictionary containing:
                - all_logits: List of logits for each future position
                  Each: [batch_size, seq_len, vocab_size]
                - combined_logits: Optional combined logits if needed
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply layer normalization if configured
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        if self.share_projections:
            # Single forward pass for all heads
            assert self.shared_head is not None, "shared_head should not be None when share_projections is True"
            combined_logits = self.shared_head(hidden_states)
            # [batch_size, seq_len, vocab_size * num_heads]

            # Split into separate predictions
            all_logits = torch.chunk(combined_logits, self.num_heads, dim=-1)
            # List of [batch_size, seq_len, vocab_size]

        else:
            # Separate forward pass for each head
            assert self.heads is not None, "heads should not be None when share_projections is False"
            all_logits = [
                head(hidden_states)  # [batch_size, seq_len, vocab_size]
                for head in self.heads
            ]

        result = {
            'all_logits': all_logits,
            'num_heads': self.num_heads,
        }

        # Optionally return combined logits
        if return_all_logits:
            result['logits_list'] = all_logits

        return result

    def predict_tokens(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict future tokens with sampling strategies.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Dictionary with predicted tokens for each position
        """
        # Get logits for all heads
        outputs = self.forward(hidden_states)
        all_logits = outputs['all_logits']

        predictions = []
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        for logits in all_logits:
            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            predicted_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(batch_size, seq_len)

            predictions.append(predicted_tokens)

        return {
            'predictions': predictions,  # List of [batch_size, seq_len]
            'num_tokens': len(predictions),
        }

    def get_parameters_count(self) -> Dict[str, Any]:
        """
        Get the number of parameters in prediction heads.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())

        if self.share_projections:
            assert self.shared_head is not None
            shared_params = sum(p.numel() for p in self.shared_head.parameters())
            per_head_params = shared_params // self.num_heads
        else:
            assert self.heads is not None
            per_head_params = sum(p.numel() for p in self.heads[0].parameters())
            shared_params = 0

        return {
            'total_params': total_params,
            'per_head_params': per_head_params,
            'shared_params': shared_params,
            'num_heads': self.num_heads,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adaptive MTP Model Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    'AdaptiveMTPConfig',
    'ConfidenceGate',
    'MultiTokenPredictionHeads',
    'AdaptiveMTPModel',
]
