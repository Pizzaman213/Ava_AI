"""
Advanced loss functions module.

This module provides a comprehensive suite of loss functions for training,
all consolidated in a single losses.py file for easier maintenance and imports.
"""

# Import all loss components from the unified losses.py file
from .losses import (
    # Unified interface (recommended for most use cases)
    UnifiedLoss,
    create_unified_loss,
    # DeepSeek components
    MultiTokenPredictionLoss,
    TemperatureScaledCrossEntropy,
    AuxiliaryFreeMoEBalancer,
    DeepSeekLoss,
    # Adaptive MTP
    AdaptiveMTPLoss,
    # Repetition penalties
    NGramRepetitionPenalty,
    SequenceRepetitionDetector,
    # Anti-repetition
    AntiRepetitionLoss,
    AdaptiveAntiRepetitionLoss,
    # Advanced losses
    ContrastiveLoss,
    FocalLoss,
    LabelSmoothingLoss,
    DiversityLoss,
    AuxiliaryLoss,
    ConsistencyLoss,
    PerplexityLoss,
    AdaptiveLossScaling,
    CompositeLoss,
)

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