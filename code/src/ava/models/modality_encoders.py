"""
Modality Encoders for Multi-Modal Learning.

This module implements encoders for different modalities (vision, audio)
that transform raw features into representations suitable for cross-attention
with the text transformer.

Supported modalities:
1. Vision: Patch-based encoding (ViT-style) with position embeddings
2. Audio: Frame-based encoding for spectrograms/waveforms
3. Generic: Configurable encoder for arbitrary tensor inputs

Example:
    >>> vision_encoder = VisionEncoder(
    ...     image_size=224,
    ...     patch_size=16,
    ...     hidden_size=768,
    ...     num_layers=6,
    ... )
    >>> image = torch.randn(2, 3, 224, 224)
    >>> features = vision_encoder(image)  # [2, 196, 768]
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
class VisionEncoderConfig:
    """Configuration for Vision Encoder."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_cls_token: bool = True
    use_flash_attention: bool = True
    layer_norm_eps: float = 1e-6
    # Pre-trained weights path (optional)
    pretrained_path: Optional[str] = None


@dataclass
class AudioEncoderConfig:
    """Configuration for Audio Encoder."""
    input_size: int = 80  # Mel bins or input features
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    max_length: int = 3000  # Max frames
    use_flash_attention: bool = True
    layer_norm_eps: float = 1e-6
    conv_downsample: bool = True  # Use conv for downsampling
    conv_kernel_size: int = 3
    conv_stride: int = 2


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.

    Splits image into non-overlapping patches and projects to hidden dimension.

    Args:
        image_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        hidden_size: Output embedding dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        # Patch projection (conv with kernel_size=stride=patch_size)
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            x: [batch, channels, height, width]

        Returns:
            patches: [batch, num_patches, hidden_size]
        """
        # [batch, hidden_size, h/patch, w/patch]
        x = self.projection(x)
        # Flatten spatial dimensions: [batch, hidden_size, num_patches]
        x = x.flatten(2)
        # Transpose: [batch, num_patches, hidden_size]
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with self-attention and FFN.

    Used as building block for both vision and audio encoders.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: FFN hidden size = hidden_size * mlp_ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        use_flash_attention: Use Flash Attention
        layer_norm_eps: LayerNorm epsilon
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # FFN
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: Optional [batch, seq] or [batch, 1, seq, seq]

        Returns:
            output: [batch, seq, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,  # Dropout handled separately
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = F.softmax(attn_scores, dim=-1)
            if self.attn_dropout is not None:
                attn_probs = self.attn_dropout(attn_probs)

            attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        hidden_states = residual + attn_output

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for image features.

    Encodes images into a sequence of patch embeddings that can be
    used with cross-attention in the main transformer.

    Architecture:
    1. Patch embedding (Conv2d)
    2. Add CLS token (optional) + position embeddings
    3. Transformer layers
    4. Output: [batch, num_patches (+1 for CLS), hidden_size]

    Args:
        config: VisionEncoderConfig with all settings

    Example:
        >>> config = VisionEncoderConfig(image_size=224, patch_size=16, hidden_size=768)
        >>> encoder = VisionEncoder(config)
        >>> image = torch.randn(2, 3, 224, 224)
        >>> features = encoder(image)  # [2, 197, 768] (196 patches + CLS)
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            num_positions = num_patches + 1
        else:
            self.cls_token = None
            num_positions = num_patches

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                use_flash_attention=config.use_flash_attention,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._init_weights()

        logger.info(
            f"VisionEncoder: image={config.image_size}, patch={config.patch_size}, "
            f"hidden={config.hidden_size}, layers={config.num_layers}"
        )

    def _init_weights(self):
        """Initialize weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize patch embedding
        w = self.patch_embed.projection.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode image to feature sequence.

        Args:
            pixel_values: [batch, channels, height, width]
            attention_mask: Optional mask (usually not needed for images)

        Returns:
            features: [batch, num_patches (+1 for CLS), hidden_size]
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding
        x = self.patch_embed(pixel_values)  # [batch, num_patches, hidden]

        # Prepend CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        if self.dropout is not None:
            x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final norm
        x = self.norm(x)

        return x

    def get_output_dim(self) -> int:
        """Get output hidden dimension."""
        return self.config.hidden_size

    def get_num_patches(self) -> int:
        """Get number of output patches (excluding CLS)."""
        return self.patch_embed.num_patches


class AudioEncoder(nn.Module):
    """
    Transformer encoder for audio features (spectrograms/waveforms).

    Encodes audio features into a sequence that can be used with
    cross-attention in the main transformer.

    Architecture:
    1. Optional convolutional downsampling
    2. Linear projection to hidden size
    3. Add position embeddings
    4. Transformer layers
    5. Output: [batch, seq_len, hidden_size]

    Args:
        config: AudioEncoderConfig with all settings

    Example:
        >>> config = AudioEncoderConfig(input_size=80, hidden_size=512, num_layers=4)
        >>> encoder = AudioEncoder(config)
        >>> spectrogram = torch.randn(2, 80, 500)  # [batch, mel_bins, frames]
        >>> features = encoder(spectrogram)  # [2, 250, 512] (downsampled 2x)
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # Convolutional frontend for downsampling
        if config.conv_downsample:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    config.input_size,
                    config.hidden_size,
                    kernel_size=config.conv_kernel_size,
                    stride=config.conv_stride,
                    padding=config.conv_kernel_size // 2,
                ),
                nn.GELU(),
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=config.conv_kernel_size,
                    stride=config.conv_stride,
                    padding=config.conv_kernel_size // 2,
                ),
                nn.GELU(),
            )
            # Compute effective max length after downsampling
            effective_length = config.max_length // (config.conv_stride ** 2)
        else:
            self.conv = None
            self.input_proj = nn.Linear(config.input_size, config.hidden_size)
            effective_length = config.max_length

        # Position embeddings (sinusoidal)
        self.pos_embed = self._create_sinusoidal_positions(effective_length, config.hidden_size)
        self.register_buffer('position_embeddings', self.pos_embed)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                use_flash_attention=config.use_flash_attention,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        logger.info(
            f"AudioEncoder: input={config.input_size}, hidden={config.hidden_size}, "
            f"layers={config.num_layers}, conv_downsample={config.conv_downsample}"
        )

    def _create_sinusoidal_positions(self, max_len: int, hidden_size: int) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        pos_embed = torch.zeros(1, max_len, hidden_size)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)

        return pos_embed

    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode audio features.

        Args:
            audio_features: [batch, input_size, frames] or [batch, frames, input_size]
            attention_mask: Optional [batch, frames] mask for padding

        Returns:
            features: [batch, seq_len, hidden_size]
            output_mask: Updated attention mask after downsampling
        """
        # Handle both [batch, features, time] and [batch, time, features]
        if audio_features.dim() == 3:
            if audio_features.shape[1] == self.config.input_size:
                # [batch, features, time] - expected for conv
                pass
            else:
                # [batch, time, features] - transpose
                audio_features = audio_features.transpose(1, 2)

        if self.conv is not None:
            # Convolutional frontend: [batch, input_size, frames] -> [batch, hidden, frames']
            x = self.conv(audio_features)
            # Transpose for transformer: [batch, frames', hidden]
            x = x.transpose(1, 2)

            # Update attention mask for downsampling
            if attention_mask is not None:
                # Downsample mask to match sequence length
                output_mask = attention_mask[:, ::self.config.conv_stride ** 2]
                output_mask = output_mask[:, :x.shape[1]]
            else:
                output_mask = None
        else:
            # Simple projection
            x = audio_features.transpose(1, 2)  # [batch, frames, input_size]
            x = self.input_proj(x)  # [batch, frames, hidden]
            output_mask = attention_mask

        seq_len = x.shape[1]

        # Add position embeddings (truncate or pad as needed)
        pos_embed = self.position_embeddings[:, :seq_len, :]
        x = x + pos_embed.to(x.device)

        if self.dropout is not None:
            x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=None)  # Mask handling TODO

        # Final norm
        x = self.norm(x)

        return x, output_mask

    def get_output_dim(self) -> int:
        """Get output hidden dimension."""
        return self.config.hidden_size


class GenericEncoder(nn.Module):
    """
    Generic encoder for arbitrary tensor inputs.

    Useful for encoding pre-computed embeddings or other modalities
    not covered by vision/audio encoders.

    Args:
        input_size: Input feature dimension
        hidden_size: Output hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads

    Example:
        >>> encoder = GenericEncoder(input_size=256, hidden_size=512, num_layers=2)
        >>> features = torch.randn(2, 100, 256)
        >>> output = encoder(features)  # [2, 100, 512]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
            )
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode generic features.

        Args:
            features: [batch, seq, input_size]
            attention_mask: Optional [batch, seq]

        Returns:
            output: [batch, seq, hidden_size]
        """
        # Project to hidden size
        x = self.input_proj(features)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final norm
        x = self.norm(x)

        return x

    def get_output_dim(self) -> int:
        """Get output hidden dimension."""
        return self.hidden_size


class MultiModalEncoder(nn.Module):
    """
    Combined encoder supporting multiple modalities.

    Manages multiple modality-specific encoders and provides
    a unified interface for encoding different input types.

    Args:
        modality_configs: Dict mapping modality names to encoder configs
        output_hidden_size: Common output hidden size (projects if needed)

    Example:
        >>> encoder = MultiModalEncoder(
        ...     modality_configs={
        ...         'vision': VisionEncoderConfig(hidden_size=768),
        ...         'audio': AudioEncoderConfig(hidden_size=512),
        ...     },
        ...     output_hidden_size=1024,
        ... )
        >>> features = encoder({
        ...     'vision': image_tensor,
        ...     'audio': audio_tensor,
        ... })
    """

    def __init__(
        self,
        modality_configs: Dict[str, Union[VisionEncoderConfig, AudioEncoderConfig]],
        output_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.modality_names = list(modality_configs.keys())
        self.output_hidden_size = output_hidden_size

        # Create encoders
        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        for name, config in modality_configs.items():
            if isinstance(config, VisionEncoderConfig):
                encoder = VisionEncoder(config)
                encoder_hidden = config.hidden_size
            elif isinstance(config, AudioEncoderConfig):
                encoder = AudioEncoder(config)
                encoder_hidden = config.hidden_size
            else:
                raise ValueError(f"Unknown config type for modality {name}: {type(config)}")

            self.encoders[name] = encoder

            # Add projection if output size differs
            if output_hidden_size is not None and encoder_hidden != output_hidden_size:
                self.projections[name] = nn.Linear(encoder_hidden, output_hidden_size)

        logger.info(f"MultiModalEncoder: modalities={list(modality_configs.keys())}")

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities.

        Args:
            inputs: Dict mapping modality names to input tensors
            attention_masks: Optional dict of attention masks

        Returns:
            Dict mapping modality names to encoded features
        """
        attention_masks = attention_masks or {}
        outputs = {}

        for name, tensor in inputs.items():
            if name not in self.encoders:
                logger.warning(f"Unknown modality: {name}, skipping")
                continue

            encoder = self.encoders[name]
            mask = attention_masks.get(name)

            # Encode
            if isinstance(encoder, AudioEncoder):
                features, _ = encoder(tensor, attention_mask=mask)
            else:
                features = encoder(tensor, attention_mask=mask)

            # Project if needed
            if name in self.projections:
                features = self.projections[name](features)

            outputs[name] = features

        return outputs

    def get_output_dim(self) -> int:
        """Get output hidden dimension."""
        if self.output_hidden_size is not None:
            return self.output_hidden_size
        # Return first encoder's output dim
        for name in self.modality_names:
            return self.encoders[name].get_output_dim()
        return 0


def create_vision_encoder(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_size: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    **kwargs,
) -> VisionEncoder:
    """Create a vision encoder from common parameters."""
    config = VisionEncoderConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )
    return VisionEncoder(config)


def create_audio_encoder(
    input_size: int = 80,
    hidden_size: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs,
) -> AudioEncoder:
    """Create an audio encoder from common parameters."""
    config = AudioEncoderConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )
    return AudioEncoder(config)


__all__ = [
    'VisionEncoderConfig',
    'AudioEncoderConfig',
    'PatchEmbedding',
    'TransformerBlock',
    'VisionEncoder',
    'AudioEncoder',
    'GenericEncoder',
    'MultiModalEncoder',
    'create_vision_encoder',
    'create_audio_encoder',
]
