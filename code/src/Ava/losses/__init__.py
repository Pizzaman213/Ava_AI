"""
Advanced loss functions module.

This module provides a comprehensive collection of loss functions for training
advanced language models, including:

Core Losses:
- FocalLoss: For addressing class imbalance
- LabelSmoothingLoss: For better generalization
- ContrastiveLoss: For representation learning
- DiversityLoss: For encouraging diverse expert outputs
- AuxiliaryLoss: For MoE routing
- NGramRepetitionPenalty: For preventing repetitive text
- SequenceRepetitionDetector: For detecting immediate repetition
- ConsistencyLoss: For semi-supervised learning
- PerplexityLoss: For language model evaluation
- AdaptiveLossScaling: For balancing multiple losses
- CompositeLoss: For combining multiple loss functions

Advanced Losses:
- AdaptiveMTPLoss: Adaptive Multi-Token Prediction
- MultiTokenPredictionLoss: DeepSeek-style MTP
- TemperatureScaledCrossEntropy: Temperature-scaled CE with adaptive temperature
- AuxiliaryFreeMoEBalancer: Auxiliary-free load balancing for MoE
- DeepSeekLoss: Combined DeepSeek-style loss
- AntiRepetitionLoss: Enhanced anti-repetition loss
"""

# Core losses
from .core_losses import (
    FocalLoss,
    LabelSmoothingLoss,
    ContrastiveLoss,
    DiversityLoss,
    AuxiliaryLoss,
    NGramRepetitionPenalty,
    SequenceRepetitionDetector,
    ConsistencyLoss,
    PerplexityLoss,
    AdaptiveLossScaling,
    CompositeLoss,
)

# Advanced losses
from .advanced_losses import (
    AdaptiveMTPLoss,
    MultiTokenPredictionLoss,
    TemperatureScaledCrossEntropy,
    AuxiliaryFreeMoEBalancer,
    DeepSeekLoss,
    AntiRepetitionLoss,
)

__all__ = [
    # Core losses
    "FocalLoss",
    "LabelSmoothingLoss",
    "ContrastiveLoss",
    "DiversityLoss",
    "AuxiliaryLoss",
    "NGramRepetitionPenalty",
    "SequenceRepetitionDetector",
    "ConsistencyLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss",
    # Advanced losses
    "AdaptiveMTPLoss",
    "MultiTokenPredictionLoss",
    "TemperatureScaledCrossEntropy",
    "AuxiliaryFreeMoEBalancer",
    "DeepSeekLoss",
    "AntiRepetitionLoss",
]
