"""
Model architectures for Ava training.

This package provides:
- EnhancedMoEModel: Main MoE++ model architecture
- SparseMoELayer: Sparse mixture of experts layer
- Expert layers: HighPerformanceExpert, ExpertParallelGroup
- Routing: MixtralRouter, DeepSeekRouter, UnifiedMoERouter
"""

from .moe import EnhancedMoEModel, EnhancedMoEConfig
from .moe_layer import SparseMoELayer

# Expert layers (moved from nn/)
from .experts import HighPerformanceExpert, ExpertParallelGroup, SharedExpertLayer

# Routing (moved from nn/)
from .routing import MixtralRouter, DeepSeekRouter, UnifiedMoERouter

try:
    from .moe import OptimizedMoETransformer, OptimizedMoEConfig
except ImportError:
    OptimizedMoETransformer = None
    OptimizedMoEConfig = None

__all__ = [
    # Core models
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
    'SparseMoELayer',
    'OptimizedMoETransformer',
    'OptimizedMoEConfig',
    # Expert layers
    'HighPerformanceExpert',
    'ExpertParallelGroup',
    'SharedExpertLayer',
    # Routing
    'MixtralRouter',
    'DeepSeekRouter',
    'UnifiedMoERouter',
]
