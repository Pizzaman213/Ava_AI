"""
Training loop components for Ava.

This package provides:
- TrainingLoopManager: Main training loop
- DataLoaderManager: Data loading management
- OptimizerManager: Optimizer and scheduler management
- ValidationManager: Validation and evaluation
- RunManager: Experiment tracking
- TrainingPipeline: Pipeline orchestration
- Coherence evaluation (moved from eval/)
- Progressive training (moved from strategies/)
"""

from .context import TrainingContext, TrainingComponent, ManagerInterface
from .loop import TrainingLoopConfig, TrainingLoopManager
from .data_manager import DataLoaderManager
from .optimizer import OptimizerManager
from .model_builder import ModelBuilder
from .validation import ValidationManager
from .generation import GenerationManager
from ava.core.wandb_logger import MetricsManager
from .run_manager import RunManager
from .pipeline import TrainingPipeline
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    DISTRIBUTED_AVAILABLE,
)

# Coherence evaluation (moved from eval/)
from .coherence import (
    CoherenceMetrics,
    CoherenceMeasurer,
    FastCoherenceMeasurer,
    CoherenceConfig,
)

# Progressive training (moved from strategies/)
from .progressive import (
    ProgressiveStrategyConfig,
    ProgressiveTrainingManager,
)
# Backward compatibility alias
ProgressiveTrainingConfig = ProgressiveStrategyConfig

# Quality evaluation (unified)
from .quality_evaluator import (
    QualityMetrics,
    QualityEvaluator,
    create_quality_evaluator,
)

__all__ = [
    # Context and base classes
    'TrainingContext',
    'TrainingComponent',
    'ManagerInterface',
    # Loop management
    'TrainingLoopConfig',
    'TrainingLoopManager',
    # Component managers
    'DataLoaderManager',
    'OptimizerManager',
    'ModelBuilder',
    'ValidationManager',
    'GenerationManager',
    'MetricsManager',
    # Orchestration
    'RunManager',
    'TrainingPipeline',
    # Distributed utilities
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'DISTRIBUTED_AVAILABLE',
    # Coherence evaluation
    'CoherenceMetrics',
    'CoherenceMeasurer',
    'FastCoherenceMeasurer',
    'CoherenceConfig',
    # Progressive training
    'ProgressiveStrategyConfig',
    'ProgressiveTrainingConfig',  # Backward compatibility alias
    'ProgressiveTrainingManager',
    # Quality evaluation (unified)
    'QualityMetrics',
    'QualityEvaluator',
    'create_quality_evaluator',
]
