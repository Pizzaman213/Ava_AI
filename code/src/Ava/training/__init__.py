"""
Training Infrastructure Module

Organized into submodules:
- train: Data loader management, training context
- orchestration: Run management
- strategies: Progressive training, curriculum learning
- optimizations: Dynamic batching, checkpointing
"""

# Run management (optional - used by finetune.py)
try:
    from .orchestration.run_manager import RunManager
except ImportError:
    RunManager = None

# Advanced training strategies (optional - used by finetune.py)
try:
    from .strategies.progressive_training import (
        CurriculumLearning,
        GrowLengthScheduler,
        DynamicBatchSizer,
        ProgressiveModelScaler,
        ProgressiveTrainer,
        ProgressiveTrainingConfig,
        ProgressiveTrainingManager,
    )
except ImportError:
    CurriculumLearning = None
    GrowLengthScheduler = None
    DynamicBatchSizer = None
    ProgressiveModelScaler = None
    ProgressiveTrainer = None
    ProgressiveTrainingConfig = None
    ProgressiveTrainingManager = None

__all__ = [
    # Orchestration
    "RunManager",
    # Strategies
    "CurriculumLearning",
    "GrowLengthScheduler",
    "DynamicBatchSizer",
    "ProgressiveModelScaler",
    "ProgressiveTrainer",
    "ProgressiveTrainingConfig",
    "ProgressiveTrainingManager",
]
