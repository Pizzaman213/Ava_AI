"""
Cross-Attention Layers for Multi-Modal Fusion.

This module implements cross-attention mechanisms for integrating
information from different modalities (text, vision, audio) into
the main transformer model.

Cross-attention allows the model to:
1. Attend to encoded representations from other modalities
2. Fuse multi-modal information at each transformer layer
3. Condition text generation on visual/audio context

Example:
    >>> cross_attn = CrossAttentionLayer(
    ...     hidden_size=1024,
    ...     num_heads=16,
    ...     encoder_hidden_size=768,
    ... )
    >>> # text_hidden: [batch, seq, 1024], encoder_hidden: [batch, enc_seq, 768]
    >>> output, attn_weights = cross_attn(text_hidden, encoder_hidden)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CrossAttentionConfig:
    """Configuration for Cross-Attention layers."""
    hidden_size: int = 1024
    num_heads: int = 16
    encoder_hidden_size: int = 768  # Size of encoder (vision/audio) hidden states
    head_dim: Optional[int] = None  # If None, computed as hidden_size // num_heads
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_bias: bool = False
    use_flash_attention: bool = True
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    pre_norm: bool = True  # Pre-LayerNorm vs Post-LayerNorm
    residual_connection: bool = True
    gated_cross_attention: bool = False  # Gated cross-attention (Flamingo-style)
    gate_init_value: float = 0.0  # Initial value for gating parameter


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for multi-modal fusion.

    Query comes from the decoder (text), while Key and Value
    come from the encoder (vision/audio).

    Args:
        config: CrossAttentionConfig with all settings

    Example:
        >>> config = CrossAttentionConfig(hidden_size=1024, encoder_hidden_size=768)
        >>> cross_attn = CrossAttention(config)
        >>> q_hidden = torch.randn(2, 128, 1024)  # Text hidden states
        >>> kv_hidden = torch.randn(2, 196, 768)  # Image patches
        >>> output = cross_attn(q_hidden, kv_hidden)
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder_hidden_size = config.encoder_hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)

        assert config.hidden_size % config.num_heads == 0, \
            f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})"

        # Query projection (from decoder hidden states)
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)

        # Key and Value projections (from encoder hidden states)
        self.k_proj = nn.Linear(config.encoder_hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.encoder_hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)

        # Output projection
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else None
        self.output_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Flash attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

        self._init_weights()

        logger.debug(
            f"CrossAttention: hidden={config.hidden_size}, encoder_hidden={config.encoder_hidden_size}, "
            f"heads={config.num_heads}, head_dim={self.head_dim}"
        )

    def _init_weights(self):
        """Initialize weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-attention.

        Args:
            hidden_states: [batch, seq_q, hidden_size] - Query from decoder
            encoder_hidden_states: [batch, seq_kv, encoder_hidden_size] - KV from encoder
            attention_mask: Optional mask for query positions
            encoder_attention_mask: Optional mask for encoder positions [batch, seq_kv]
            output_attentions: Whether to return attention weights

        Returns:
            output: [batch, seq_q, hidden_size]
            attention_weights: Optional [batch, num_heads, seq_q, seq_kv]
        """
        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = encoder_hidden_states.shape[1]

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        # Reshape: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle encoder attention mask
        attn_mask = None
        if encoder_attention_mask is not None:
            # [batch, seq_kv] -> [batch, 1, 1, seq_kv]
            attn_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attn_mask.float()) * torch.finfo(q.dtype).min

        # Compute attention
        if self.use_flash_attention and not output_attentions:
            # Use Flash Attention via scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=False,  # Cross-attention is not causal
            )
            attention_weights = None
        else:
            # Manual attention computation
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attention_weights = F.softmax(attn_scores, dim=-1)

            if self.attention_dropout is not None:
                attention_weights = self.attention_dropout(attention_weights)

            attn_output = torch.matmul(attention_weights, v)

        # Reshape back: [batch, num_heads, seq_q, head_dim] -> [batch, seq_q, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)

        # Output projection
        output = self.o_proj(attn_output)

        if self.output_dropout is not None:
            output = self.output_dropout(output)

        return output, attention_weights


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention (Flamingo-style).

    Adds a learnable gating mechanism that controls how much
    cross-modal information flows into the model. This allows
    gradual integration of multi-modal features.

    gate_output = tanh(gate) * cross_attention_output

    Args:
        config: CrossAttentionConfig with gated_cross_attention=True

    Example:
        >>> config = CrossAttentionConfig(hidden_size=1024, gated_cross_attention=True)
        >>> gated_attn = GatedCrossAttention(config)
        >>> output = gated_attn(text_hidden, image_hidden)
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.cross_attention = CrossAttention(config)

        # Learnable gate parameter (initialized near zero)
        self.gate = nn.Parameter(torch.tensor([config.gate_init_value]))

        logger.info(f"GatedCrossAttention: gate_init={config.gate_init_value}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with gating.

        Args:
            hidden_states: [batch, seq_q, hidden_size]
            encoder_hidden_states: [batch, seq_kv, encoder_hidden_size]
            attention_mask: Optional query mask
            encoder_attention_mask: Optional encoder mask
            output_attentions: Whether to return attention weights

        Returns:
            gated_output: [batch, seq_q, hidden_size]
            attention_weights: Optional attention weights
        """
        # Cross-attention
        attn_output, attention_weights = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )

        # Apply gating (tanh keeps gate bounded)
        gated_output = torch.tanh(self.gate) * attn_output

        return gated_output, attention_weights

    def get_gate_value(self) -> float:
        """Get current gate value (for monitoring)."""
        return torch.tanh(self.gate).item()


class CrossAttentionLayer(nn.Module):
    """
    Complete Cross-Attention layer with LayerNorm and residual.

    Can be inserted into transformer blocks for multi-modal fusion.

    Architecture (pre-norm):
        output = hidden_states + CrossAttention(LayerNorm(hidden_states), encoder_hidden)

    Architecture (post-norm):
        output = LayerNorm(hidden_states + CrossAttention(hidden_states, encoder_hidden))

    Args:
        config: CrossAttentionConfig with all settings

    Example:
        >>> config = CrossAttentionConfig(hidden_size=1024, encoder_hidden_size=768)
        >>> layer = CrossAttentionLayer(config)
        >>> output, attn = layer(text_hidden, image_hidden)
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config

        # Cross-attention (optionally gated)
        if config.gated_cross_attention:
            self.cross_attention = GatedCrossAttention(config)
        else:
            self.cross_attention = CrossAttention(config)

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.encoder_hidden_size != config.hidden_size:
                self.encoder_layer_norm = nn.LayerNorm(config.encoder_hidden_size, eps=config.layer_norm_eps)
            else:
                self.encoder_layer_norm = None
        else:
            self.layer_norm = None
            self.encoder_layer_norm = None

        self.pre_norm = config.pre_norm
        self.residual = config.residual_connection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LayerNorm and residual.

        Args:
            hidden_states: [batch, seq_q, hidden_size]
            encoder_hidden_states: [batch, seq_kv, encoder_hidden_size]
            attention_mask: Optional query mask
            encoder_attention_mask: Optional encoder mask
            output_attentions: Whether to return attention weights

        Returns:
            output: [batch, seq_q, hidden_size]
            attention_weights: Optional attention weights
        """
        residual = hidden_states

        # Pre-norm
        if self.pre_norm and self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            if self.encoder_layer_norm is not None:
                encoder_hidden_states = self.encoder_layer_norm(encoder_hidden_states)

        # Cross-attention
        attn_output, attention_weights = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )

        # Residual connection
        if self.residual:
            output = residual + attn_output
        else:
            output = attn_output

        # Post-norm
        if not self.pre_norm and self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attention_weights


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module supporting multiple encoder modalities.

    Can fuse information from multiple sources (e.g., image + audio)
    into the text representation.

    Args:
        hidden_size: Decoder hidden size
        modality_configs: Dict mapping modality names to their configs
        fusion_type: How to combine modalities ('sequential', 'parallel', 'hierarchical')

    Example:
        >>> fusion = MultiModalFusion(
        ...     hidden_size=1024,
        ...     modality_configs={
        ...         'vision': {'encoder_hidden_size': 768},
        ...         'audio': {'encoder_hidden_size': 512},
        ...     }
        ... )
        >>> output = fusion(
        ...     text_hidden,
        ...     {'vision': image_features, 'audio': audio_features}
        ... )
    """

    def __init__(
        self,
        hidden_size: int,
        modality_configs: Dict[str, Dict[str, Any]],
        fusion_type: str = 'sequential',
        num_heads: int = 16,
        dropout: float = 0.0,
        gated: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.modality_names = list(modality_configs.keys())
        self.fusion_type = fusion_type

        # Create cross-attention for each modality
        self.cross_attention_layers = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            config = CrossAttentionConfig(
                hidden_size=hidden_size,
                num_heads=num_heads,
                encoder_hidden_size=cfg.get('encoder_hidden_size', hidden_size),
                dropout=dropout,
                gated_cross_attention=gated,
            )
            self.cross_attention_layers[name] = CrossAttentionLayer(config)

        # For parallel fusion, need a combiner
        if fusion_type == 'parallel':
            self.combiner = nn.Linear(hidden_size * len(modality_configs), hidden_size)
        elif fusion_type == 'hierarchical':
            self.hierarchy_weights = nn.Parameter(torch.ones(len(modality_configs)))

        logger.info(
            f"MultiModalFusion: modalities={list(modality_configs.keys())}, "
            f"fusion={fusion_type}, gated={gated}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Dict[str, torch.Tensor],
        encoder_attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Fuse multiple modalities into text representation.

        Args:
            hidden_states: [batch, seq, hidden_size] - Text hidden states
            encoder_hidden_states: Dict mapping modality names to their features
            encoder_attention_masks: Optional dict of masks per modality
            output_attentions: Whether to return attention weights

        Returns:
            fused_output: [batch, seq, hidden_size]
            attention_weights: Dict of attention weights per modality
        """
        encoder_attention_masks = encoder_attention_masks or {}
        attention_weights_all: Dict[str, Optional[torch.Tensor]] = {}

        if self.fusion_type == 'sequential':
            # Apply cross-attention sequentially
            output = hidden_states
            for name in self.modality_names:
                if name in encoder_hidden_states:
                    enc_hidden = encoder_hidden_states[name]
                    enc_mask = encoder_attention_masks.get(name)
                    output, attn_weights = self.cross_attention_layers[name](
                        output, enc_hidden,
                        encoder_attention_mask=enc_mask,
                        output_attentions=output_attentions,
                    )
                    attention_weights_all[name] = attn_weights

        elif self.fusion_type == 'parallel':
            # Apply cross-attention in parallel, then combine
            outputs = []
            for name in self.modality_names:
                if name in encoder_hidden_states:
                    enc_hidden = encoder_hidden_states[name]
                    enc_mask = encoder_attention_masks.get(name)
                    out, attn_weights = self.cross_attention_layers[name](
                        hidden_states, enc_hidden,
                        encoder_attention_mask=enc_mask,
                        output_attentions=output_attentions,
                    )
                    outputs.append(out)
                    attention_weights_all[name] = attn_weights
                else:
                    outputs.append(hidden_states)  # Use original if modality missing

            # Combine parallel outputs
            combined = torch.cat(outputs, dim=-1)
            output = self.combiner(combined)

        elif self.fusion_type == 'hierarchical':
            # Weighted combination with learnable hierarchy
            weights = F.softmax(self.hierarchy_weights, dim=0)
            output = torch.zeros_like(hidden_states)

            for i, name in enumerate(self.modality_names):
                if name in encoder_hidden_states:
                    enc_hidden = encoder_hidden_states[name]
                    enc_mask = encoder_attention_masks.get(name)
                    out, attn_weights = self.cross_attention_layers[name](
                        hidden_states, enc_hidden,
                        encoder_attention_mask=enc_mask,
                        output_attentions=output_attentions,
                    )
                    output = output + weights[i] * out
                    attention_weights_all[name] = attn_weights

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return output, attention_weights_all


def create_cross_attention_from_config(
    hidden_size: int,
    encoder_hidden_size: int,
    num_heads: int = 16,
    gated: bool = False,
    **kwargs,
) -> CrossAttentionLayer:
    """
    Create a cross-attention layer from common parameters.

    Args:
        hidden_size: Decoder hidden dimension
        encoder_hidden_size: Encoder hidden dimension
        num_heads: Number of attention heads
        gated: Whether to use gated cross-attention
        **kwargs: Additional CrossAttentionConfig parameters

    Returns:
        Configured CrossAttentionLayer
    """
    config = CrossAttentionConfig(
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_heads=num_heads,
        gated_cross_attention=gated,
        **kwargs,
    )
    return CrossAttentionLayer(config)


__all__ = [
    'CrossAttentionConfig',
    'CrossAttention',
    'GatedCrossAttention',
    'CrossAttentionLayer',
    'MultiModalFusion',
    'create_cross_attention_from_config',
]
