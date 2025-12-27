"""
Model architectures for Ava training.

This package provides:
- EnhancedMoEModel: Main MoE++ model architecture
- SparseMoELayer: Sparse mixture of experts layer
"""

from .moe import EnhancedMoEModel, EnhancedMoEConfig
from .moe_layer import SparseMoELayer

try:
    from .moe import OptimizedMoETransformer, OptimizedMoEConfig
except ImportError:
    OptimizedMoETransformer = None
    OptimizedMoEConfig = None

__all__ = [
    'EnhancedMoEModel',
    'EnhancedMoEConfig',
    'SparseMoELayer',
    'OptimizedMoETransformer',
    'OptimizedMoEConfig',
]
