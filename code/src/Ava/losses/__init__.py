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

__all__ = [
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiversityLoss",
    "AuxiliaryLoss",
    "ConsistencyLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss"
]