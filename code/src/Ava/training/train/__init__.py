"""
Modular training framework - Breaking down the monolithic Trainer class.

This package provides a compositional training framework that separates concerns:
- DistributedTrainingManager: Handles distributed setup, synchronization, fault tolerance
- CheckpointManager: Manages model state persistence and recovery
- LossComputationManager: Handles loss calculation, numerical stability, and gradient surgery
- MonitoringManager: Tracks metrics, logs, W&B integration
- EnhancedTrainer: Composes all managers into a cohesive training loop

Architecture:
    EnhancedTrainer
     DistributedTrainingManager
     CheckpointManager
     LossComputationManager
     MonitoringManager
     DeepSpeed/Optimizer Integration

Benefits:
    - Each manager is testable independently
    - Clear single responsibility
    - Easier to understand and modify
    - Reduced cognitive load (each file ~400-600 lines)
    - Easy to mock/replace components
"""

from .base import TrainingComponent, TrainingContext, ManagerInterface
from .distributed_manager import DistributedTrainingManager
from .checkpoint_manager import CheckpointManager
from .loss_manager import LossComputationManager
from .monitoring_manager import MonitoringManager
from .trainer import SimplifiedEnhancedTrainer

# Backwards compatibility alias
EnhancedTrainer = SimplifiedEnhancedTrainer

__all__ = [
    "TrainingComponent",
    "TrainingContext",
    "ManagerInterface",
    "DistributedTrainingManager",
    "CheckpointManager",
    "LossComputationManager",
    "MonitoringManager",
    "SimplifiedEnhancedTrainer",
    "EnhancedTrainer",  # For backwards compatibility
]
