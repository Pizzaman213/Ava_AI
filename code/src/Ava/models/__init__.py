"""
Ava Model Architectures

This module contains the core model architectures including:
- EnhancedMoEModel: Base Mixture of Experts model
- AdaptiveMTPModel: Adaptive Multi-Token Prediction wrapper
- ConfidenceGate: Confidence scoring network
- PredictionHeads: Multi-token prediction heads
"""

from .adaptive_mtp_model import AdaptiveMTPModel, AdaptiveMTPConfig
from .confidence_gate import ConfidenceGate
from .prediction_heads import MultiTokenPredictionHeads

# EnhancedMoEModel will be imported separately as it may exist elsewhere
try:
    from .moe_model import EnhancedMoEModel, EnhancedMoEConfig
except ImportError:
    # Model might be defined elsewhere or not yet created
    EnhancedMoEModel = None
    EnhancedMoEConfig = None

__all__ = [
    'AdaptiveMTPModel',
    'AdaptiveMTPConfig',
    'ConfidenceGate',
    'MultiTokenPredictionHeads',
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
]
