# Training Module Organization

This document describes the reorganized structure of the `Ava.training` module, which has been organized into logical subdirectories for better maintainability and discoverability.

## Directory Structure

```
training/
├── __init__.py                 # Main module with backward-compatible exports
├── README.md                   # This file
├── core/                       # Core training infrastructure
│   ├── __init__.py
│   ├── enhanced_trainer.py     # Main trainer (EnhancedModularTrainer)
│   ├── run_manager.py          # Training run organization
│   └── optimization_integration.py  # Unified optimization setup
├── learning_rate/              # Learning rate management
│   ├── __init__.py
│   ├── adaptive_lr.py          # Real-time adaptive LR adjustments
│   ├── advanced_schedulers.py  # SGDR, OneCycle, Polynomial, etc.
│   ├── advanced_warmup.py      # Sophisticated warmup strategies
│   ├── advanced_warmup_scheduling.py  # Gradient noise, LR finder
│   ├── lr_manager.py           # Intelligent LR calculation
│   └── lr_finder.py            # Learning rate range testing
├── distributed/                # Distributed training
│   ├── __init__.py
│   ├── distributed_manager.py  # Core DDP implementation
│   ├── unified_distributed_manager.py  # Simplified DDP wrapper
│   ├── distributed_health_checker.py   # Cross-rank health monitoring
│   └── rank_aware_error_handler.py     # Distributed error handling
├── gradients/                  # Gradient management
│   ├── __init__.py
│   ├── gradient_surgery.py     # Multi-task gradient conflict resolution
│   └── gradient_health.py      # Gradient monitoring and adaptive clipping
├── strategies/                 # Training strategies
│   ├── __init__.py
│   ├── progressive_training.py # Curriculum learning, dynamic batching
│   └── performance_modes.py    # Performance optimization modes
└── monitoring/                 # Training monitoring
    ├── __init__.py
    ├── metrics.py              # Comprehensive metrics collection
    └── unified_optimizations.py # Optimization aggregation
```

## Module Categories

### 1. Core Training Infrastructure (`core/`)

The fundamental components for running training:

- **EnhancedModularTrainer**: Main trainer class that integrates all modular components
- **RunManager**: Manages training run organization, logging, and metadata
- **OptimizationIntegration**: Orchestrates optimization components (hardware, mixed precision, throughput, memory)

**Key classes**: `EnhancedModularTrainer`, `RunManager`, `OptimizationIntegration`

### 2. Learning Rate Management (`learning_rate/`)

Everything related to learning rate scheduling and adaptation:

- **adaptive_lr.py**: Real-time LR adjustments with loss monitoring, plateau detection, and stability-based adjustments
- **advanced_schedulers.py**: State-of-the-art schedulers (SGDR, OneCycle, Polynomial Decay, Adaptive LR, Noisy Student)
- **advanced_warmup.py**: Sophisticated warmup with linear/cosine/polynomial/exponential schedules
- **advanced_warmup_scheduling.py**: Gradient noise scale, LR finder, cyclical batch sizing
- **lr_manager.py**: Intelligent LR management integrating warmup, main scheduler, and adaptive adjustments
- **lr_finder.py**: Learning rate range testing with Savitzky-Golay smoothing and divergence detection

**Key classes**: `AdaptiveLearningRateManager`, `LRFinder`, `LRManager`, `SchedulerFactory`, `AdvancedWarmupScheduler`

### 3. Distributed Training (`distributed/`)

Components for multi-GPU and multi-node training:

- **distributed_manager.py**: Core DDP implementation with barrier synchronization and error handling
- **unified_distributed_manager.py**: Simplified wrapper providing consistent DDP interface
- **distributed_health_checker.py**: Health monitoring across ranks with loss sync and gradient analysis
- **rank_aware_error_handler.py**: Distributed error handling with severity levels and coordinated responses

**Key classes**: `DistributedManager`, `UnifiedDistributedManager`, `DistributedHealthChecker`, `RankAwareErrorHandler`

### 4. Gradient Management (`gradients/`)

Gradient optimization and conflict resolution:

- **gradient_surgery.py**: Multi-task gradient conflict resolution (GradientSurgeon, AdaptiveGradientSurgeon)
- **gradient_health.py**: Gradient monitoring with adaptive clipping, explosion detection, and automatic recovery

**Key classes**: `GradientSurgeon`, `AdaptiveGradientSurgeon`, `GradientHealthMonitor`, `LossHealthMonitor`

### 5. Training Strategies (`strategies/`)

Advanced training techniques:

- **progressive_training.py**: Progressive training with sequence length scaling, curriculum learning, dynamic batch sizing
- **performance_modes.py**: Performance optimization modes (UltraFast, FastProgress, MinimalProgress, Express, NoSync)

**Key classes**: `ProgressiveTrainingManager`, `ProgressiveTrainer`, `CurriculumLearning`, `PerformanceModeManager`

### 6. Monitoring (`monitoring/`)

Training metrics and observation:

- **metrics.py**: Comprehensive metrics collection (loss, LR, gradients, memory, timing) with trend analysis
- **unified_optimizations.py**: Aggregates and manages all optimization modules

**Key classes**: `TrainingMetricsCollector`, `MetricConfig`, `UnifiedOptimizations`

## Usage

All public APIs are re-exported from the main `training` module for backward compatibility:

```python
# Import from main training module (backward compatible)
from Ava.training import (
    EnhancedModularTrainer,
    RunManager,
    LRFinder,
    AdaptiveLearningRateManager,
    DistributedManager,
    GradientHealthMonitor,
    ProgressiveTrainingManager,
    TrainingMetricsCollector,
)

# Or import directly from subdirectories
from Ava.training.core import EnhancedModularTrainer
from Ava.training.learning_rate import LRFinder, AdaptiveLearningRateManager
from Ava.training.distributed import DistributedManager
from Ava.training.gradients import GradientHealthMonitor
from Ava.training.strategies import ProgressiveTrainingManager
from Ava.training.monitoring import TrainingMetricsCollector
```

## Statistics

- **Total Files**: 20 Python modules
- **Total Lines**: ~15,660 lines of code
- **Largest Module**: `enhanced_trainer.py` (3,897 lines)

### Lines of Code by Category

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Core | 3 | 4,970 | Main trainer and run management |
| Learning Rate | 6 | 4,729 | LR scheduling and adaptation |
| Distributed | 4 | 2,492 | Multi-GPU/node training |
| Gradients | 2 | 1,014 | Gradient optimization |
| Strategies | 2 | 1,990 | Progressive training and performance modes |
| Monitoring | 2 | 1,339 | Metrics and observation |

## Migration Notes

### For External Code

If you're importing from `Ava.training` in external scripts, **no changes are needed**. All public APIs are re-exported from the main module.

### For Internal Code

Internal imports within the training module have been updated to use relative imports:

```python
# Before (when files were in training/)
from .gradient_health import GradientHealthMonitor

# After (when files are in training/gradients/)
from ..gradients.gradient_health import GradientHealthMonitor
```

## Benefits of This Organization

1. **Improved Discoverability**: Related functionality is grouped together
2. **Reduced Cognitive Load**: Each subdirectory focuses on a specific concern
3. **Better Maintainability**: Changes to one area are isolated from others
4. **Clearer Dependencies**: Import structure makes relationships explicit
5. **Backward Compatibility**: Existing code continues to work without changes
6. **Git History Preserved**: Files were moved with `git mv` to maintain history

## Archived Files

The following files have been moved to `_archived/training/`:
- `profiling_tools.py`
- `dynamic_batch_sampler.py`
- `progressive_batch_scheduler.py`
- `qlora_utils.py`
- `distributed_optimizations.py`

These are optional training strategies/tools not used in the default pipeline.
