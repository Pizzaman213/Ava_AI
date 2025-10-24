# Type stubs for Ava.models module
from typing import Optional, Type
import torch.nn as nn

# Import from consolidated MTP module
from .mtp import AdaptiveMTPModel as AdaptiveMTPModel, AdaptiveMTPConfig as AdaptiveMTPConfig
from .mtp import ConfidenceGate as ConfidenceGate
from .mtp import MultiTokenPredictionHeads as MultiTokenPredictionHeads

# EnhancedMoEModel may or may not be available
EnhancedMoEModel: Optional[Type[nn.Module]]
EnhancedMoEConfig: Optional[Type]
ColossalAIMoEModel: Optional[Type[nn.Module]]

__all__ = [
    'AdaptiveMTPModel',
    'AdaptiveMTPConfig',
    'ConfidenceGate',
    'MultiTokenPredictionHeads',
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
    'ColossalAIMoEModel',
]
