# Ava - Enhanced MoE Training Framework

This directory contains the core implementation of the Ava training framework for Mixture-of-Experts (MoE) models.

## Directory Structure

```
src/Ava/
├── _archived/              # Experimental/optional features (not in training pipeline)
│   ├── data/              # Data preparation tools
│   ├── evaluation/        # Alternative evaluators
│   ├── generation/        # Text generation (inference only)
│   ├── layers/            # Experimental layer architectures
│   ├── losses/            # Alternative loss functions
│   ├── models/            # Model wrappers
│   ├── optimization/      # Experimental optimizations
│   └── training/          # Alternative training strategies
│
├── config/                # Training configuration
│   ├── training_config.py
│   └── feature_compatibility.py
│
├── data/                  # Data loading and processing
│   ├── arrow_reader.py
│   └── encoding_detector.py
│
├── evaluation/            # Model evaluation
│   └── comprehensive_eval.py
│
├── layers/                # Neural network layers
│   ├── experts.py
│   └── routing.py
│
├── losses/                # Loss functions
│   ├── advanced_losses.py
│   └── deepseek_loss.py
│
├── memory/                # Memory management
│   └── episodic_memory.py
│
├── models/                # Model architectures
│   └── moe_model.py
│
├── optimization/          # Model optimization
│   ├── advanced_optimizers.py
│   ├── fp8_training.py
│   └── quantization.py
│
├── training/              # Training utilities (16 modules)
│   ├── enhanced_trainer.py         # Core trainer
│   ├── adaptive_lr.py
│   ├── advanced_schedulers.py
│   ├── advanced_warmup.py
│   ├── distributed_health_checker.py
│   ├── distributed_manager.py
│   ├── gradient_health.py
│   ├── gradient_surgery.py
│   ├── lr_manager.py
│   ├── memory_monitor.py
│   ├── metrics.py
│   ├── performance_modes.py
│   ├── progressive_training.py
│   ├── rank_aware_error_handler.py
│   └── run_manager.py
│
├── utils/                 # Utility functions
│   ├── async_logging.py
│   ├── checkpoint.py
│   ├── gpu_memory.py
│   └── logging.py
│
├── data_streaming.py      # Streaming data loaders
├── multi_column_data.py   # Multi-column data handling
│
├── ARCHIVED_FEATURES.md   # Documentation of archived features
└── README.md             # This file
```

## Core Components

### Training Pipeline (Essential)

1. **Trainer** - [training/enhanced_trainer.py](training/enhanced_trainer.py)
   - Main training loop and orchestration
   - Integrates all training components

2. **Model** - [models/moe_model.py](models/moe_model.py)
   - Enhanced Mixture-of-Experts architecture
   - Expert routing and balancing

3. **Data** - [data_streaming.py](data_streaming.py)
   - Efficient streaming data loaders
   - Multi-column data support

4. **Config** - [config/training_config.py](config/training_config.py)
   - Training hyperparameters
   - Feature flags

### Training Features

- **Distributed Training** - Multi-GPU/multi-node support
- **Progressive Training** - Curriculum learning and dynamic batching
- **Gradient Surgery** - Advanced gradient manipulation
- **Memory Optimization** - Efficient memory management
- **Performance Modes** - Speed vs. quality tradeoffs
- **Comprehensive Metrics** - Detailed training analytics

### Optimization Features

- **Quantization** - INT4/INT8/FP8 model quantization
- **Advanced Optimizers** - Lion, Sophia, AdaFactor
- **FP8 Training** - Mixed-precision with FP8
- **Learning Rate** - Adaptive LR with warmup

## Archived Features

**25 modules** have been moved to `_archived/` as they are not used in the default training pipeline. These include:

- **Experimental optimizations** (FlashAttention v3, A100-specific, etc.)
- **Alternative layer architectures** (MoH, MoA, cross-attention)
- **Data preparation tools** (profiling, deduplication)
- **Inference utilities** (text generation)
- **Alternative training strategies** (QLoRA, dynamic batching)

See [ARCHIVED_FEATURES.md](ARCHIVED_FEATURES.md) for complete documentation.

## Quick Start

### Basic Training
```python
from src.Ava.config import EnhancedTrainingConfig
from src.Ava.models.moe_model import EnhancedMoEModel
from src.Ava.training.enhanced_trainer import EnhancedModularTrainer

# Load config
config = EnhancedTrainingConfig.from_yaml("config.yaml")

# Create model
model = EnhancedMoEModel(config.model)

# Create trainer
trainer = EnhancedModularTrainer(model, config)

# Train
trainer.train()
```

### Using Archived Features
```python
# Import from archive
from src.Ava._archived.generation.generator import TextGenerator

# Or move back to active directory
# mv _archived/optimization/flash_attention_v3.py optimization/
```

## Import Structure

All modules follow the pattern:
```python
from src.Ava.<module> import <class>
```

Example:
```python
from src.Ava.training import EnhancedModularTrainer
from src.Ava.models import EnhancedMoEModel
from src.Ava.config import EnhancedTrainingConfig
```

## Configuration

Training is configured via YAML files or Python dataclasses:

```yaml
# config.yaml
model:
  hidden_size: 768
  num_layers: 12
  num_experts: 8

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 10
```

## Documentation

- **[ARCHIVED_FEATURES.md](ARCHIVED_FEATURES.md)** - Complete documentation of archived features
- **Training scripts** - See `code/scripts/training/`
- **Configuration examples** - See `code/configs/`

## Module Statistics

- **Active modules:** 50 Python files
- **Archived modules:** 25 Python files
- **Total LOC (active):** ~15,000 lines
- **Reduction:** 33% decrease in active codebase

## Development

### Adding New Features
1. Implement in appropriate directory
2. Add imports to `__init__.py`
3. Update configuration if needed
4. Add tests
5. Document in README

### Archiving Features
1. Move to `_archived/<category>/`
2. Update `__init__.py` to remove imports
3. Document in `ARCHIVED_FEATURES.md`
4. Test that training still works

## Support

For issues or questions:
- Check [ARCHIVED_FEATURES.md](ARCHIVED_FEATURES.md) for archived features
- Review configuration documentation
- See training scripts for examples

---

**Last Updated:** 2025-10-06
**Status:** Production-ready
**Active Modules:** 50
**Archived Modules:** 25
