"""
Advanced loss functions module.

Consolidated architecture:
- core_losses.py: Fundamental training objectives
- regularization_losses.py: Regularization and anti-collapse mechanisms
- mtp_moe_losses.py: Multi-token prediction and MoE-specific losses

All original imports are maintained for backward compatibility.
"""

# Core losses
from .core_losses import (
    TemperatureScaledCrossEntropy,
    FocalLoss,
    LabelSmoothingLoss,
    PerplexityLoss,
    AdaptiveLossScaling,
    CompositeLoss,
)

# Regularization losses
from .regularization_losses import (
    NGramRepetitionPenalty,
    SequenceRepetitionDetector,
    ContrastiveLoss,
    DiversityLoss,
    ConsistencyLoss,
)

# MTP and MoE losses
from .mtp_moe_losses import (
    MultiTokenPredictionLoss,
    AdaptiveMTPLoss,
    AuxiliaryFreeMoEBalancer,
    AuxiliaryLoss,
    DeepSeekLoss,
)

__all__ = [
    # Core losses
    "TemperatureScaledCrossEntropy",
    "FocalLoss",
    "LabelSmoothingLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss",
    # Regularization losses
    "NGramRepetitionPenalty",
    "SequenceRepetitionDetector",
    "ContrastiveLoss",
    "DiversityLoss",
    "ConsistencyLoss",
    # MTP and MoE losses
    "MultiTokenPredictionLoss",
    "AdaptiveMTPLoss",
    "AuxiliaryFreeMoEBalancer",
    "AuxiliaryLoss",
    "DeepSeekLoss",
]