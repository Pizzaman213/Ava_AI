"""
Neural network layers for Qwen MoE++ architecture.
"""

# Note: The following files have been moved to _archived/layers/:
# - attention.py
# - advanced_attention.py
# - mixture_of_heads.py
# - mixture_of_activations.py
# - cross_attention.py
# These are experimental features not used in the default training pipeline

from .experts import ExpertBalancer, SparseExpert
from .routing import ExpertSelector, MoEPlusPlusLayer

__all__ = [
    "ExpertBalancer",
    "SparseExpert",
    "ExpertSelector",
    "MoEPlusPlusLayer",
]