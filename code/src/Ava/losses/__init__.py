"""
Advanced loss functions module.

This module provides a comprehensive suite of loss functions for training,
including the unified loss that combines all components.
"""

# Import unified loss (recommended for most use cases)
from .unified_loss import UnifiedLoss, create_unified_loss

# Import individual loss components
from .advanced_losses import (
    ContrastiveLoss,
    FocalLoss,
    LabelSmoothingLoss,
    DiversityLoss,
    AuxiliaryLoss,
    ConsistencyLoss,
    PerplexityLoss,
    AdaptiveLossScaling,
    CompositeLoss
)

from .adaptive_mtp_loss import AdaptiveMTPLoss
from .repetition_penalty_loss import NGramRepetitionPenalty, SequenceRepetitionDetector
from .deepseek_loss import (
    DeepSeekLoss,
    MultiTokenPredictionLoss,
    TemperatureScaledCrossEntropy,
    AuxiliaryFreeMoEBalancer
)
from .anti_repetition_loss import AntiRepetitionLoss, AdaptiveAntiRepetitionLoss

__all__ = [
    # Unified loss (recommended)
    "UnifiedLoss",
    "create_unified_loss",
    # DeepSeek losses
    "DeepSeekLoss",
    "MultiTokenPredictionLoss",
    "TemperatureScaledCrossEntropy",
    "AuxiliaryFreeMoEBalancer",
    # Adaptive MTP
    "AdaptiveMTPLoss",
    # Repetition penalties
    "NGramRepetitionPenalty",
    "SequenceRepetitionDetector",
    "AntiRepetitionLoss",
    "AdaptiveAntiRepetitionLoss",
    # Advanced losses
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiversityLoss",
    "AuxiliaryLoss",
    "ConsistencyLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss",
]