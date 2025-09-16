"""
Ava MoE++ Architecture Package

A modular implementation of the enhanced Mixture of Experts model with:
- Advanced routing mechanisms
- Dynamic expert selection
- Conditional sparse computation
- Comprehensive training utilities
"""

from .models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from .layers.experts import ExpertBalancer, SparseExpert
from .layers.routing import ExpertSelector, MoEPlusPlusLayer

__version__ = "1.0.0"

__all__ = [
    "EnhancedMoEModel",
    "EnhancedMoEConfig",
    "ExpertBalancer",
    "SparseExpert",
    "ExpertSelector",
    "MoEPlusPlusLayer",
]