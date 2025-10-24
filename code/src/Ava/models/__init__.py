"""
Ava Model Architectures

This module contains the core model architectures including:
- EnhancedMoEModel: Base Mixture of Experts model
- AdaptiveMTPModel: Adaptive Multi-Token Prediction wrapper
- ConfidenceGate: Confidence scoring network
- MultiTokenPredictionHeads: Multi-token prediction heads
"""

# Import MTP components from consolidated module
from .mtp import (
    AdaptiveMTPModel,
    AdaptiveMTPConfig,
    ConfidenceGate,
    MultiTokenPredictionHeads,
)

# Import MoE model
try:
    from .moe_model import EnhancedMoEModel, EnhancedMoEConfig
except ImportError:
    # Model might be defined elsewhere or not yet created
    EnhancedMoEModel = None
    EnhancedMoEConfig = None

# Import Colossal-AI extension (optional)
try:
    from .colossalai_moe_model import ColossalAIMoEModel
except ImportError:
    ColossalAIMoEModel = None

__all__ = [
    # MTP System
    'AdaptiveMTPModel',
    'AdaptiveMTPConfig',
    'ConfidenceGate',
    'MultiTokenPredictionHeads',
    # MoE Models
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
    'ColossalAIMoEModel',
]
