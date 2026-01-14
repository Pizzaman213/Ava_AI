"""
Model architectures for Ava training.

This package provides:
- EnhancedMoEModel: Main MoE++ model architecture
- SparseMoELayer: Sparse mixture of experts layer
- Expert layers: HighPerformanceExpert, ExpertParallelGroup, SequentialExpertGroup
- Routing: MixtralRouter, DeepSeekRouter, UnifiedMoERouter
- ALiBi: ALiBiPositionalBias, ALiBiAttention
- Cross-Attention: CrossAttentionLayer, MultiModalFusion
- Modality Encoders: VisionEncoder, AudioEncoder
"""

from .moe import EnhancedMoEModel, EnhancedMoEConfig
from .moe_layer import SparseMoELayer

# Expert layers
from .experts import (
    HighPerformanceExpert,
    ExpertParallelGroup,
    SharedExpertLayer,
    SequentialExpertGroup,
)

# Routing
from .routing import MixtralRouter, DeepSeekRouter, UnifiedMoERouter

# ALiBi Positional Encoding
from .alibi import (
    get_alibi_slopes,
    build_alibi_bias,
    ALiBiPositionalBias,
    ALiBiAttention,
    apply_alibi_to_attention_scores,
)

# Cross-Attention for Multi-Modal
from .cross_attention import (
    CrossAttentionConfig,
    CrossAttention,
    GatedCrossAttention,
    CrossAttentionLayer,
    MultiModalFusion,
    create_cross_attention_from_config,
)

# Modality Encoders
from .modality_encoders import (
    VisionEncoderConfig,
    AudioEncoderConfig,
    PatchEmbedding,
    TransformerBlock,
    VisionEncoder,
    AudioEncoder,
    GenericEncoder,
    MultiModalEncoder,
    create_vision_encoder,
    create_audio_encoder,
)

__all__ = [
    # Core models
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
    'SparseMoELayer',
    # Expert layers
    'HighPerformanceExpert',
    'ExpertParallelGroup',
    'SharedExpertLayer',
    'SequentialExpertGroup',
    # Routing
    'MixtralRouter',
    'DeepSeekRouter',
    'UnifiedMoERouter',
    # ALiBi
    'get_alibi_slopes',
    'build_alibi_bias',
    'ALiBiPositionalBias',
    'ALiBiAttention',
    'apply_alibi_to_attention_scores',
    # Cross-Attention
    'CrossAttentionConfig',
    'CrossAttention',
    'GatedCrossAttention',
    'CrossAttentionLayer',
    'MultiModalFusion',
    'create_cross_attention_from_config',
    # Modality Encoders
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
