"""
Configuration management for Ava training.

This package provides:
- Training configuration dataclasses
- YAML configuration loading
- Centralized constants
- Configuration validation

Usage:
    from ava.config import DATA_CONSTANTS, TRAINER_CONSTANTS
    from ava.config.constants import update_constants_from_config
    from ava.config import EnhancedTrainingConfig, TrainingConfigManager
"""

from .constants import (
    DATA_CONSTANTS,
    DataPipelineConstants,
    MOE_CONSTANTS,
    MoEConstants,
    TRAINER_CONSTANTS,
    TrainerConstants,
    update_constants_from_config,
)

from .training_config import (
    DynamicConfig,
    EnhancedTrainingConfig,
    TrainingConfigManager,
)

__all__ = [
    # Constants instances
    'DATA_CONSTANTS',
    'TRAINER_CONSTANTS',
    'MOE_CONSTANTS',
    # Constants classes
    'DataPipelineConstants',
    'TrainerConstants',
    'MoEConstants',
    # Functions
    'update_constants_from_config',
    # Training config
    'DynamicConfig',
    'EnhancedTrainingConfig',
    'TrainingConfigManager',
]
