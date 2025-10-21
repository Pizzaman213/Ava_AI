# Type stubs for Ava.models module
from typing import Optional, Type
import torch.nn as nn

from .adaptive_mtp_model import AdaptiveMTPModel as AdaptiveMTPModel, AdaptiveMTPConfig as AdaptiveMTPConfig
from .confidence_gate import ConfidenceGate as ConfidenceGate
from .prediction_heads import MultiTokenPredictionHeads as MultiTokenPredictionHeads

# EnhancedMoEModel may or may not be available
EnhancedMoEModel: Optional[Type[nn.Module]]
EnhancedMoEConfig: Optional[Type]

__all__ = [
    'AdaptiveMTPModel',
    'AdaptiveMTPConfig',
    'ConfidenceGate',
    'MultiTokenPredictionHeads',
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
]
