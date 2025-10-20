"""
Confidence Gating Network for Adaptive Multi-Token Prediction

This module implements a neural network that predicts the model's confidence
in its next-token prediction, determining whether to activate additional
prediction heads for multi-token generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


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
