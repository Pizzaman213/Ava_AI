"""
Ava - Advanced MoE++ Training Framework.

This package provides a comprehensive framework for training
Mixture of Experts (MoE++) language models with optimized
performance and memory efficiency.

Subpackages:
    - config: Configuration management and constants
    - core: Core utilities (paths, logging, activations)
    - data: Data loading and processing
    - nn: Neural network layers (experts, routing)
    - kernels: Triton kernels for GPU optimization
    - models: Model architectures
    - training: Training loop components
    - optimizations: Training optimizations
    - strategies: Training strategies (progressive training)
    - optim: Optimizers and LR scheduling
    - cuda: CUDA utilities
    - eval: Evaluation utilities

Usage:
    from ava.data import create_streaming_dataloaders
    from ava.core import get_project_root, setup_colored_logging
    from ava.config import DATA_CONSTANTS

Example:
    >>> from ava.core import get_project_root
    >>> root = get_project_root()
    >>> print(f"Project root: {root}")
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
__all__ = [
    '__version__',
]
