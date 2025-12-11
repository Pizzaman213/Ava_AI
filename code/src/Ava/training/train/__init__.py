"""
Modular training framework components.

This package provides:
- TrainingContext: Context object for training state
- TrainingComponent: Base class for training components
- ManagerInterface: Extended interface for lifecycle hooks
- DataLoaderManager: Manages data loading
- ModelBuilder: Model creation and optimization
- OptimizerManager: Optimizer and scheduler setup
- TrainingLoopManager: Core training loop
- ValidationManager: Validation during training
- GenerationManager: Sample generation for monitoring
- MetricsManager: Metrics tracking with TensorBoard/WandB
"""

from .base import TrainingComponent, TrainingContext, ManagerInterface
from .data_loader_manager import DataLoaderManager
from .model_builder import ModelBuilder
from .optimizer_manager import OptimizerManager
from .training_loop_manager import (
    TrainingLoopManager,
    TrainingLoopConfig,
    # Protocol interfaces for dependency injection
    MetricsLoggerProtocol,
    GenerationProviderProtocol,
    CheckpointSaverProtocol,
)
from .validation_manager import ValidationManager
from .generation_manager import GenerationManager
from .metrics_manager import MetricsManager

__all__ = [
    # Base classes
    "TrainingComponent",
    "TrainingContext",
    "ManagerInterface",
    # Managers
    "DataLoaderManager",
    "ModelBuilder",
    "OptimizerManager",
    "TrainingLoopManager",
    "TrainingLoopConfig",
    "ValidationManager",
    "GenerationManager",
    "MetricsManager",
    # Protocol interfaces
    "MetricsLoggerProtocol",
    "GenerationProviderProtocol",
    "CheckpointSaverProtocol",
]
