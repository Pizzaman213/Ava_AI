"""
Configuration Management Package

This package provides configuration management for all training features
including validation, feature compatibility checking, and structured config handling.
"""

from .training_config import (
    # Configuration dataclasses
    ArchitectureConfig,
    RAGConfig,
    LossConfig,
    GradientConfig,
    EvaluationConfig,
    QuantizationConfig,
    EpisodicMemoryConfig,
    DataConfig,
    MultiColumnDataConfig,
    TrainingConfig,
    OutputConfig,
    RunManagementConfig,
    WandBConfig,
    PerformanceConfig,
    EnhancedTrainingConfig,

    # Manager class
    TrainingConfigManager
)

__all__ = [
    # Configuration dataclasses
    'ArchitectureConfig',
    'RAGConfig',
    'LossConfig',
    'GradientConfig',
    'EvaluationConfig',
    'QuantizationConfig',
    'EpisodicMemoryConfig',
    'DataConfig',
    'MultiColumnDataConfig',
    'TrainingConfig',
    'OutputConfig',
    'RunManagementConfig',
    'WandBConfig',
    'PerformanceConfig',
    'EnhancedTrainingConfig',

    # Manager class
    'TrainingConfigManager'
]