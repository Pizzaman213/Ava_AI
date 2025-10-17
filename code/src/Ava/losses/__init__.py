"""
Advanced loss functions module.
"""

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

__all__ = [
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiversityLoss",
    "AuxiliaryLoss",
    "ConsistencyLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss",
    "AdaptiveMTPLoss",
    "NGramRepetitionPenalty",
    "SequenceRepetitionDetector",
]