"""
Modular training framework components.

This package provides:
- TrainingContext: Context object for training state
- TrainingComponent: Base class for training components
- DataLoaderManager: Manages data loading
"""

from .base import TrainingComponent, TrainingContext, ManagerInterface
from .data_loader_manager import DataLoaderManager

# Optional trainer import
try:
    from .trainer import SimplifiedEnhancedTrainer
    EnhancedTrainer = SimplifiedEnhancedTrainer
except ImportError:
    SimplifiedEnhancedTrainer = None
    EnhancedTrainer = None

__all__ = [
    "TrainingComponent",
    "TrainingContext",
    "ManagerInterface",
    "DataLoaderManager",
    "SimplifiedEnhancedTrainer",
    "EnhancedTrainer",
]
