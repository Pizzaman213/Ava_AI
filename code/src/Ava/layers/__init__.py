"""
Neural network layers for high-performance Sparse MoE architecture.
"""

# Note: The following files have been moved to _archived/layers/:
# - attention.py
# - advanced_attention.py
# - mixture_of_heads.py
# - mixture_of_activations.py
# - cross_attention.py
# These are experimental features not used in the default training pipeline

from .experts import HighPerformanceExpert, ExpertParallelGroup, SharedExpertLayer
from .routing import UnifiedMoERouter, MixtralRouter, DeepSeekRouter

__all__ = [
    "HighPerformanceExpert",
    "ExpertParallelGroup",
    "SharedExpertLayer",
    "UnifiedMoERouter",
    "MixtralRouter",
    "DeepSeekRouter",
]