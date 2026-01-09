"""
Multi-Token Prediction (MTP) Head for speculative decoding and improved training.

This module implements multi-token prediction which:
1. Predicts multiple future tokens simultaneously during training
2. Enables speculative decoding for faster inference
3. Uses confidence gating to weight predictions

Based on research from:
- "Better & Faster Large Language Models via Multi-token Prediction" (Meta, 2024)
- "Medusa: Simple Framework for Accelerating LLM Generation" (Cai et al., 2023)

Example:
    >>> mtp_head = MultiTokenPredictionHead(
    ...     hidden_size=1024,
    ...     vocab_size=50000,
    ...     num_heads=4,
    ...     confidence_threshold=0.6,
    ... )
    >>> hidden_states = model.forward(input_ids)  # [batch, seq, hidden]
    >>> predictions, confidences, loss = mtp_head(hidden_states, labels)
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
class MTPConfig:
    """Configuration for Multi-Token Prediction head."""
    hidden_size: int = 1024
    vocab_size: int = 50000
    num_prediction_heads: int = 3
    confidence_threshold_train: float = 0.6
    confidence_threshold_inference: float = 0.7
    gate_hidden_dims: Tuple[int, ...] = (512, 256)
    gate_dropout: float = 0.1
    gate_activation: str = 'gelu'
    use_attention_pooling: bool = False
    head_type: str = 'linear'
    head_intermediate_size: Optional[int] = None
    head_dropout: float = 0.1
    share_projections: bool = False
    mtp_warmup_epochs: int = 2
    confidence_reg_strength: float = 0.01
    use_confidence_weighting: bool = False
    primary_loss_weight: float = 1.0
    additional_loss_base_weight: float = 0.1
    enable_dynamic_prediction: bool = False
    min_confidence_for_computation: float = 0.3


class ConfidenceGate(nn.Module):
    """
    Confidence gate for determining prediction reliability.

    Learns to predict how confident the model should be about each
    prediction head's output. Used to:
    1. Weight losses during training
    2. Decide which predictions to accept during inference
    3. Enable early exit in speculative decoding

    Args:
        hidden_size: Input hidden dimension
        hidden_dims: Tuple of hidden layer dimensions
        dropout: Dropout probability
        activation: Activation function name
        use_attention_pooling: Use attention for context aggregation
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_attention_pooling: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention_pooling = use_attention_pooling

        # Get activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        # Build MLP layers
        layers = []
        in_dim = hidden_size
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_proj = nn.Linear(in_dim, 1)

        # Optional attention pooling for context aggregation
        if use_attention_pooling:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            confidence: [batch, seq] confidence scores in [0, 1]
        """
        # Optional attention pooling
        if self.use_attention_pooling:
            attended, _ = self.attention(
                hidden_states, hidden_states, hidden_states,
                need_weights=False,
            )
            hidden_states = self.attention_norm(hidden_states + attended)

        # MLP processing
        features = self.mlp(hidden_states)
        confidence = torch.sigmoid(self.output_proj(features).squeeze(-1))

        return confidence


class PredictionHead(nn.Module):
    """
    Single prediction head for one future token.

    Can be linear (single projection) or MLP (with intermediate layer).

    Args:
        hidden_size: Input hidden dimension
        vocab_size: Output vocabulary size
        head_type: 'linear' or 'mlp'
        intermediate_size: Intermediate size for MLP (if None, uses 4x hidden)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        head_type: str = 'linear',
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.head_type = head_type

        if head_type == 'linear':
            self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        elif head_type == 'mlp':
            intermediate = intermediate_size or (hidden_size * 4)
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, intermediate),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate, vocab_size, bias=False),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to start predictions close to uniform."""
        if self.head_type == 'linear':
            nn.init.normal_(self.proj.weight, std=0.02)
        else:
            for module in self.proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for this head's token.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        return self.proj(hidden_states)


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction head for predicting multiple future tokens.

    Architecture:
    - Primary head: Predicts next token (standard LM head)
    - Additional heads: Predict tokens 2, 3, ... N
    - Confidence gates: Estimate prediction reliability per head

    Training:
    - Computes cross-entropy loss for each head
    - Optionally weights by confidence
    - Includes confidence regularization

    Inference:
    - Uses confidence thresholds for speculative decoding
    - High-confidence predictions can be verified in parallel

    Args:
        config: MTPConfig with all settings
        existing_lm_head: Optional existing LM head to use as primary

    Example:
        >>> config = MTPConfig(hidden_size=1024, vocab_size=50000, num_prediction_heads=4)
        >>> mtp = MultiTokenPredictionHead(config)
        >>> hidden = torch.randn(2, 128, 1024)
        >>> labels = torch.randint(0, 50000, (2, 128))
        >>> result = mtp(hidden, labels, training=True)
        >>> print(result['loss'], result['predictions'][0].shape)
    """

    def __init__(
        self,
        config: MTPConfig,
        existing_lm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_prediction_heads

        # Primary head (can reuse existing LM head)
        if existing_lm_head is not None:
            self.primary_head = existing_lm_head
        else:
            self.primary_head = PredictionHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                head_type=config.head_type,
                intermediate_size=config.head_intermediate_size,
                dropout=config.head_dropout,
            )

        # Additional prediction heads for future tokens
        self.additional_heads = nn.ModuleList([
            PredictionHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                head_type=config.head_type,
                intermediate_size=config.head_intermediate_size,
                dropout=config.head_dropout,
            )
            for _ in range(config.num_prediction_heads - 1)
        ])

        # Confidence gates for each additional head
        gate_dims = config.gate_hidden_dims
        if isinstance(gate_dims, str):
            gate_dims = tuple(int(d) for d in gate_dims.split(','))

        self.confidence_gates = nn.ModuleList([
            ConfidenceGate(
                hidden_size=config.hidden_size,
                hidden_dims=gate_dims,
                dropout=config.gate_dropout,
                activation=config.gate_activation,
                use_attention_pooling=config.use_attention_pooling,
            )
            for _ in range(config.num_prediction_heads - 1)
        ])

        # Shared projection option (reduces parameters)
        if config.share_projections and len(self.additional_heads) > 1:
            # Share weights of additional heads
            base_head = self.additional_heads[0]
            for head in self.additional_heads[1:]:
                head.proj = base_head.proj

        # Training state
        self._current_epoch = 0
        self._warmup_complete = False

        logger.info(
            f"MultiTokenPredictionHead: {self.num_heads} heads, "
            f"vocab_size={config.vocab_size}, hidden_size={config.hidden_size}"
        )

    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling."""
        self._current_epoch = epoch
        self._warmup_complete = epoch >= self.config.mtp_warmup_epochs

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
        return_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional loss computation.

        Args:
            hidden_states: [batch, seq, hidden_size]
            labels: [batch, seq] target token IDs (optional)
            training: Whether in training mode
            return_logits: Whether to return full logits (memory-intensive)

        Returns:
            Dictionary with:
            - predictions: List of [batch, seq] predicted token IDs
            - confidences: List of [batch, seq] confidence scores (for additional heads)
            - logits: List of [batch, seq, vocab] logits (if return_logits=True)
            - loss: Scalar loss (if labels provided)
            - losses: Per-head losses
            - metrics: Additional metrics
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        result: Dict[str, Any] = {
            'predictions': [],
            'confidences': [],
            'logits': [] if return_logits else None,
            'loss': None,
            'losses': [],
            'metrics': {},
        }

        # Threshold based on mode
        threshold = (
            self.config.confidence_threshold_train if training
            else self.config.confidence_threshold_inference
        )

        # Primary head prediction (always computed)
        primary_logits = self.primary_head(hidden_states)  # [batch, seq, vocab]
        primary_preds = primary_logits.argmax(dim=-1)  # [batch, seq]
        result['predictions'].append(primary_preds)
        result['confidences'].append(torch.ones(batch_size, seq_len, device=device))

        if return_logits:
            result['logits'].append(primary_logits)

        # Additional heads with confidence gating
        for i, (head, gate) in enumerate(zip(self.additional_heads, self.confidence_gates)):
            # Compute confidence
            confidence = gate(hidden_states)  # [batch, seq]
            result['confidences'].append(confidence)

            # Dynamic computation: skip low-confidence predictions during inference
            if (
                self.config.enable_dynamic_prediction
                and not training
                and confidence.mean() < self.config.min_confidence_for_computation
            ):
                # Return dummy predictions
                result['predictions'].append(torch.zeros_like(primary_preds))
                if return_logits:
                    result['logits'].append(torch.zeros_like(primary_logits))
                continue

            # Compute prediction
            logits = head(hidden_states)  # [batch, seq, vocab]
            preds = logits.argmax(dim=-1)  # [batch, seq]
            result['predictions'].append(preds)

            if return_logits:
                result['logits'].append(logits)

        # Compute losses if labels provided
        if labels is not None:
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
            losses = []

            # Primary head loss (always computed)
            # Shift for next-token prediction
            shift_logits = primary_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            primary_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses.append(primary_loss)
            total_loss = total_loss + self.config.primary_loss_weight * primary_loss

            # Additional head losses (with warmup)
            if self._warmup_complete:
                for i, (head, gate) in enumerate(zip(self.additional_heads, self.confidence_gates)):
                    # Shift by head index + 2 (head 0 predicts token +2)
                    offset = i + 2
                    if seq_len <= offset:
                        # Not enough context for this head
                        losses.append(torch.tensor(0.0, device=device))
                        continue

                    # Get logits for this head
                    if return_logits and len(result['logits']) > i + 1:
                        head_logits = result['logits'][i + 1]
                    else:
                        head_logits = head(hidden_states)

                    # Shift appropriately for this head's prediction target
                    shift_logits = head_logits[..., :-offset, :].contiguous()
                    shift_labels = labels[..., offset:].contiguous()

                    # Compute loss
                    head_loss = F.cross_entropy(
                        shift_logits.view(-1, self.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='none',
                    )
                    head_loss = head_loss.view(batch_size, -1)  # [batch, seq-offset]

                    # Get confidence for weighting
                    confidence = result['confidences'][i + 1][..., :-offset]  # Match loss shape

                    if self.config.use_confidence_weighting:
                        # Weight loss by confidence
                        weighted_loss = (head_loss * confidence).mean()
                    else:
                        weighted_loss = head_loss.mean()

                    losses.append(weighted_loss)

                    # Compute weight decay for additional heads
                    weight = self.config.additional_loss_base_weight * (0.5 ** i)
                    total_loss = total_loss + weight * weighted_loss

                # Confidence regularization: encourage calibrated confidence
                if self.config.confidence_reg_strength > 0:
                    conf_reg = torch.tensor(0.0, device=device, dtype=dtype)
                    for conf in result['confidences'][1:]:
                        # Regularize toward 0.5 (encourage exploration)
                        conf_reg = conf_reg + ((conf - 0.5) ** 2).mean()
                    total_loss = total_loss + self.config.confidence_reg_strength * conf_reg
                    # GPU SYNC FIX: Keep as tensor, defer .item() to log time
                    result['metrics']['confidence_regularization'] = conf_reg.detach()

            result['loss'] = total_loss
            # GPU SYNC FIX: Keep as tensors, defer .item() to log time
            result['losses'] = [l.detach() if isinstance(l, torch.Tensor) else l for l in losses]

        # Compute metrics
        result['metrics']['num_heads_active'] = len(result['predictions'])
        if len(result['confidences']) > 1:
            avg_conf = torch.stack([c.mean() for c in result['confidences'][1:]]).mean()
            # GPU SYNC FIX: Keep as tensor, defer .item() to log time
            result['metrics']['avg_additional_confidence'] = avg_conf.detach()

        return result

    def speculative_decode(
        self,
        hidden_states: torch.Tensor,
        num_candidates: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate token sequences for speculative decoding.

        Args:
            hidden_states: [batch, 1, hidden_size] - last position only
            num_candidates: Max number of speculative tokens

        Returns:
            candidates: [batch, num_candidates] predicted token IDs
            confidences: [batch, num_candidates] confidence scores
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        threshold = self.config.confidence_threshold_inference

        candidates = []
        confidences = []

        # Primary prediction (always included)
        primary_logits = self.primary_head(hidden_states)
        primary_pred = primary_logits.argmax(dim=-1).squeeze(-1)  # [batch]
        candidates.append(primary_pred)
        confidences.append(torch.ones(batch_size, device=device))

        # Additional speculative predictions
        for i, (head, gate) in enumerate(zip(self.additional_heads, self.confidence_gates)):
            if len(candidates) >= num_candidates:
                break

            confidence = gate(hidden_states).squeeze(-1)  # [batch]

            # Only include if confidence exceeds threshold
            if confidence.mean() >= threshold:
                logits = head(hidden_states)
                pred = logits.argmax(dim=-1).squeeze(-1)  # [batch]
                candidates.append(pred)
                confidences.append(confidence)
            else:
                break

        # Stack results
        candidates = torch.stack(candidates, dim=1)  # [batch, num_candidates]
        confidences = torch.stack(confidences, dim=1)  # [batch, num_candidates]

        return candidates, confidences


def create_mtp_head_from_config(
    config: Any,
    hidden_size: int,
    vocab_size: int,
    existing_lm_head: Optional[nn.Module] = None,
) -> MultiTokenPredictionHead:
    """
    Create MTP head from training config.

    Args:
        config: AdaptiveMTPConfig or similar config object
        hidden_size: Model hidden size
        vocab_size: Vocabulary size
        existing_lm_head: Optional existing LM head to reuse

    Returns:
        Configured MultiTokenPredictionHead
    """
    # Parse gate hidden dims
    gate_dims = config.gate_hidden_dims
    if isinstance(gate_dims, str):
        gate_dims = tuple(int(d) for d in gate_dims.split(','))
    else:
        gate_dims = tuple(gate_dims) if gate_dims else (512, 256)

    mtp_config = MTPConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_prediction_heads=config.num_prediction_heads,
        confidence_threshold_train=config.confidence_threshold_train,
        confidence_threshold_inference=config.confidence_threshold_inference,
        gate_hidden_dims=gate_dims,
        gate_dropout=config.gate_dropout,
        gate_activation=config.gate_activation,
        use_attention_pooling=config.use_attention_pooling,
        head_type=config.head_type,
        head_intermediate_size=config.head_intermediate_size,
        head_dropout=config.head_dropout,
        share_projections=config.share_projections,
        mtp_warmup_epochs=config.mtp_warmup_epochs,
        confidence_reg_strength=config.confidence_reg_strength,
        use_confidence_weighting=config.use_confidence_weighting,
        primary_loss_weight=config.primary_loss_weight,
        additional_loss_base_weight=config.additional_loss_base_weight,
        enable_dynamic_prediction=config.enable_dynamic_prediction,
        min_confidence_for_computation=config.min_confidence_for_computation,
    )

    return MultiTokenPredictionHead(mtp_config, existing_lm_head)


__all__ = [
    'MTPConfig',
    'ConfidenceGate',
    'PredictionHead',
    'MultiTokenPredictionHead',
    'create_mtp_head_from_config',
]
