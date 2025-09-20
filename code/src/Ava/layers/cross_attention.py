"""
Cross-attention layers for multi-modal capabilities.

This module implements various cross-attention mechanisms that enable
the model to attend across different modalities (text, vision, audio, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class MultiModalCrossAttention(nn.Module):
    """
    Multi-modal cross-attention layer for handling different input modalities.

    This layer enables attention between different modalities such as text,
    vision, and audio, allowing for rich multi-modal understanding.
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 1024,
        audio_dim: int = 512,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # Modality-specific projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-attention components
        self.text_to_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vision_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vision_to_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        if use_layer_norm:
            self.text_norm = nn.LayerNorm(hidden_dim)
            self.vision_norm = nn.LayerNorm(hidden_dim)
            self.audio_norm = nn.LayerNorm(hidden_dim)

        # Output projections
        self.text_output = nn.Linear(hidden_dim, text_dim)
        self.vision_output = nn.Linear(hidden_dim, vision_dim)
        self.audio_output = nn.Linear(hidden_dim, audio_dim)

        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through multi-modal cross-attention.

        Args:
            text_features: Text features [batch, text_seq_len, text_dim]
            vision_features: Vision features [batch, vision_seq_len, vision_dim]
            audio_features: Audio features [batch, audio_seq_len, audio_dim]
            attention_masks: Optional attention masks for each modality

        Returns:
            Tuple of (enhanced_text, enhanced_vision, enhanced_audio, attention_weights)
        """
        attention_weights = {}

        # Project to common dimension
        if text_features is not None:
            text_proj = self.text_proj(text_features)
        if vision_features is not None:
            vision_proj = self.vision_proj(vision_features)
        if audio_features is not None:
            audio_proj = self.audio_proj(audio_features)

        # Cross-modal attention computations
        enhanced_text = text_proj if text_features is not None else None
        enhanced_vision = vision_proj if vision_features is not None else None
        enhanced_audio = audio_proj if audio_features is not None else None

        # Text attending to other modalities
        if text_features is not None:
            if vision_features is not None:
                text_vision_attn, text_vision_weights = self.text_to_vision(
                    text_proj, vision_proj, vision_proj,
                    key_padding_mask=attention_masks.get('vision') if attention_masks else None
                )
                enhanced_text = enhanced_text + self.dropout(text_vision_attn)
                attention_weights['text_to_vision'] = text_vision_weights

            if audio_features is not None:
                text_audio_attn, text_audio_weights = self.text_to_audio(
                    text_proj, audio_proj, audio_proj,
                    key_padding_mask=attention_masks.get('audio') if attention_masks else None
                )
                enhanced_text = enhanced_text + self.dropout(text_audio_attn)
                attention_weights['text_to_audio'] = text_audio_weights

        # Vision attending to other modalities
        if vision_features is not None:
            if text_features is not None:
                vision_text_attn, vision_text_weights = self.vision_to_text(
                    vision_proj, text_proj, text_proj,
                    key_padding_mask=attention_masks.get('text') if attention_masks else None
                )
                enhanced_vision = enhanced_vision + self.dropout(vision_text_attn)
                attention_weights['vision_to_text'] = vision_text_weights

            if audio_features is not None:
                vision_audio_attn, vision_audio_weights = self.vision_to_audio(
                    vision_proj, audio_proj, audio_proj,
                    key_padding_mask=attention_masks.get('audio') if attention_masks else None
                )
                enhanced_vision = enhanced_vision + self.dropout(vision_audio_attn)
                attention_weights['vision_to_audio'] = vision_audio_weights

        # Audio attending to other modalities
        if audio_features is not None:
            if text_features is not None:
                audio_text_attn, audio_text_weights = self.audio_to_text(
                    audio_proj, text_proj, text_proj,
                    key_padding_mask=attention_masks.get('text') if attention_masks else None
                )
                enhanced_audio = enhanced_audio + self.dropout(audio_text_attn)
                attention_weights['audio_to_text'] = audio_text_weights

            if vision_features is not None:
                audio_vision_attn, audio_vision_weights = self.audio_to_vision(
                    audio_proj, vision_proj, vision_proj,
                    key_padding_mask=attention_masks.get('vision') if attention_masks else None
                )
                enhanced_audio = enhanced_audio + self.dropout(audio_vision_attn)
                attention_weights['audio_to_vision'] = audio_vision_weights

        # Apply layer normalization
        if self.use_layer_norm:
            if enhanced_text is not None:
                enhanced_text = self.text_norm(enhanced_text)
            if enhanced_vision is not None:
                enhanced_vision = self.vision_norm(enhanced_vision)
            if enhanced_audio is not None:
                enhanced_audio = self.audio_norm(enhanced_audio)

        # Project back to original dimensions
        if enhanced_text is not None:
            enhanced_text = self.text_output(enhanced_text)
        if enhanced_vision is not None:
            enhanced_vision = self.vision_output(enhanced_vision)
        if enhanced_audio is not None:
            enhanced_audio = self.audio_output(enhanced_audio)

        return enhanced_text, enhanced_vision, enhanced_audio, attention_weights


class PerceiversCrossAttention(nn.Module):
    """
    Perceiver-style cross-attention for handling variable input modalities.

    This implementation uses a learned latent array that attends to
    different modalities, providing a unified representation.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        num_latents: int = 256,
        input_dims: Dict[str, int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.input_dims = input_dims or {'text': 768, 'vision': 1024, 'audio': 512}
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Learnable latent array
        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input projections for each modality
        self.input_projections = nn.ModuleDict({
            modality: nn.Linear(dim, latent_dim)
            for modality, dim in self.input_dims.items()
        })

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Self-attention layers for latents
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 4, latent_dim)
            ) for _ in range(num_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        input_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through Perceiver cross-attention.

        Args:
            inputs: Dictionary of input tensors for each modality
            input_masks: Optional attention masks for each modality

        Returns:
            Tuple of (latent_representations, attention_weights)
        """
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device

        # Initialize latents for the batch
        latents = self.latent_array.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate all inputs
        all_inputs = []
        all_masks = []
        modality_info = {}
        start_idx = 0

        for modality, tensor in inputs.items():
            # Project to latent dimension
            projected = self.input_projections[modality](tensor)
            all_inputs.append(projected)

            # Track modality positions
            seq_len = tensor.shape[1]
            modality_info[modality] = (start_idx, start_idx + seq_len)
            start_idx += seq_len

            # Handle masks
            if input_masks and modality in input_masks:
                all_masks.append(input_masks[modality])
            else:
                all_masks.append(torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device))

        # Concatenate all inputs and masks
        concatenated_inputs = torch.cat(all_inputs, dim=1)
        concatenated_masks = torch.cat(all_masks, dim=1) if any(mask.any() for mask in all_masks) else None

        attention_weights = {}

        # Apply cross-attention and self-attention layers
        for layer_idx in range(self.num_layers):
            # Cross-attention: latents attend to inputs
            cross_attn, cross_weights = self.cross_attention_layers[layer_idx](
                latents, concatenated_inputs, concatenated_inputs,
                key_padding_mask=concatenated_masks
            )
            latents = self.cross_norms[layer_idx](latents + cross_attn)

            # Self-attention among latents
            self_attn, self_weights = self.self_attention_layers[layer_idx](
                latents, latents, latents
            )
            latents = self.self_norms[layer_idx](latents + self_attn)

            # Feed-forward
            ffn_out = self.ffn_layers[layer_idx](latents)
            latents = self.ffn_norms[layer_idx](latents + ffn_out)

            # Store attention weights
            attention_weights[f'cross_layer_{layer_idx}'] = cross_weights
            attention_weights[f'self_layer_{layer_idx}'] = self_weights

        return latents, attention_weights


class AdaptiveCrossAttention(nn.Module):
    """
    Adaptive cross-attention that dynamically adjusts attention patterns
    based on input content and modality importance.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_modalities: int = 3,
        adaptation_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = hidden_dim // num_heads

        # Modality importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, num_modalities),
            nn.Softmax(dim=-1)
        )

        # Adaptive attention patterns
        self.pattern_generator = nn.Sequential(
            nn.Linear(hidden_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, num_heads * num_heads),
            nn.Sigmoid()
        )

        # Standard attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query_modality: torch.Tensor,
        key_modalities: List[torch.Tensor],
        value_modalities: List[torch.Tensor],
        modality_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Adaptive cross-attention forward pass.

        Args:
            query_modality: Query modality tensor [batch, seq_len, hidden_dim]
            key_modalities: List of key modality tensors
            value_modalities: List of value modality tensors
            modality_weights: Optional pre-computed modality weights

        Returns:
            Tuple of (attended_output, attention_info)
        """
        batch_size, seq_len, _ = query_modality.shape

        # Predict modality importance if not provided
        if modality_weights is None:
            pooled_query = query_modality.mean(dim=1)  # Pool sequence dimension
            modality_weights = self.importance_predictor(pooled_query)

        # Generate adaptive attention patterns
        attention_patterns = self.pattern_generator(query_modality.mean(dim=1))
        attention_patterns = attention_patterns.view(batch_size, self.num_heads, self.num_heads)

        # Combine modalities based on importance weights
        combined_keys = torch.zeros_like(query_modality)
        combined_values = torch.zeros_like(query_modality)

        for i, (key_mod, value_mod) in enumerate(zip(key_modalities, value_modalities)):
            if i < self.num_modalities:
                weight = modality_weights[:, i].unsqueeze(1).unsqueeze(2)
                combined_keys += weight * key_mod
                combined_values += weight * value_mod

        # Standard attention computation
        q = self.query_proj(query_modality)
        k = self.key_proj(combined_keys)
        v = self.value_proj(combined_values)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply adaptive attention patterns
        q = torch.einsum('bhsd,bhh->bhsd', q, attention_patterns)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.output_proj(attn_output)

        attention_info = {
            'modality_weights': modality_weights,
            'attention_patterns': attention_patterns,
            'attention_weights': attn_weights
        }

        return output, attention_info


class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical cross-attention for handling multi-scale multi-modal inputs.

    This module processes inputs at different scales and hierarchically
    combines information across modalities.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_scales: int = 3,
        scale_factors: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.scale_factors = scale_factors or [1, 2, 4]

        # Multi-scale processing layers
        self.scale_processors = nn.ModuleList([
            nn.ModuleDict({
                'downsample': nn.AvgPool1d(kernel_size=factor, stride=factor) if factor > 1 else nn.Identity(),
                'attention': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm': nn.LayerNorm(hidden_dim),
                'upsample': nn.Upsample(scale_factor=factor, mode='linear', align_corners=False) if factor > 1 else nn.Identity()
            }) for factor in self.scale_factors
        ])

        # Cross-scale fusion
        self.cross_scale_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical cross-attention forward pass.

        Args:
            query: Query tensor [batch, seq_len, hidden_dim]
            key_value: Key-value tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Tuple of (hierarchical_output, scale_attention_weights)
        """
        batch_size, seq_len, hidden_dim = query.shape
        scale_outputs = []
        scale_attention_weights = {}

        # Process at different scales
        for scale_idx, (scale_factor, processor) in enumerate(zip(self.scale_factors, self.scale_processors)):
            # Downsample key-value if needed
            if scale_factor > 1:
                # Reshape for 1D pooling
                kv_reshaped = key_value.transpose(1, 2)  # [batch, hidden_dim, seq_len]
                kv_downsampled = processor['downsample'](kv_reshaped)
                kv_downsampled = kv_downsampled.transpose(1, 2)  # [batch, downsampled_seq_len, hidden_dim]
            else:
                kv_downsampled = key_value

            # Cross-attention at this scale
            scale_output, scale_weights = processor['attention'](
                query, kv_downsampled, kv_downsampled,
                key_padding_mask=attention_mask
            )

            # Normalize
            scale_output = processor['norm'](scale_output + query)

            scale_outputs.append(scale_output)
            scale_attention_weights[f'scale_{scale_idx}'] = scale_weights

        # Cross-scale attention
        stacked_outputs = torch.stack(scale_outputs, dim=2)  # [batch, seq_len, num_scales, hidden_dim]
        stacked_outputs = stacked_outputs.view(batch_size, seq_len * self.num_scales, hidden_dim)

        # Query attends to all scales
        cross_scale_output, cross_scale_weights = self.cross_scale_fusion(
            query, stacked_outputs, stacked_outputs
        )

        scale_attention_weights['cross_scale'] = cross_scale_weights

        # Combine all scale outputs
        concatenated = torch.cat(scale_outputs, dim=-1)  # [batch, seq_len, hidden_dim * num_scales]
        final_output = self.output_projection(concatenated)
        final_output = self.final_norm(final_output + cross_scale_output)

        return final_output, scale_attention_weights