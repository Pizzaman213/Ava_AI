# Ava LLM Training Framework Documentation

Welcome to the Ava documentation. This guide covers everything from getting started to advanced optimization techniques.

## Quick Navigation

| Guide | Description |
|-------|-------------|
| [Getting Started](./01_GETTING_STARTED.md) | Installation, first training run, and basic concepts |
| [Architecture](./02_ARCHITECTURE.md) | MoE++ architecture, model components, and design |
| [Training Guide](./03_TRAINING_GUIDE.md) | Complete training workflow and best practices |
| [Configuration Reference](./04_CONFIGURATION.md) | All 110+ configuration parameters explained |
| [Data Pipeline](./05_DATA_PIPELINE.md) | Data loading, preprocessing, and streaming |
| [Distributed Training](./06_DISTRIBUTED.md) | Multi-GPU, DeepSpeed, and FSDP setup |
| [RLHF Training](./07_RLHF.md) | Reinforcement Learning from Human Feedback |
| [API Reference](./08_API_REFERENCE.md) | Module-by-module API documentation |
| [Troubleshooting](./09_TROUBLESHOOTING.md) | Common issues and solutions |
| [Performance Tuning](./10_PERFORMANCE.md) | Optimization strategies and benchmarks |

## Framework Overview

Ava is an advanced LLM training framework featuring:

- **MoE++ Architecture**: 8-32 experts with Mixtral/DeepSeek/Switch routing
- **Memory Efficiency**: Gradient checkpointing, expert offloading, FP8 support
- **Scalability**: Single GPU to multi-node distributed training
- **Production Ready**: WandB integration, checkpointing, evaluation metrics

## Quick Start

```bash
# Single GPU training
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU training
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

## Documentation Conventions

- **Code blocks** show exact commands to run
- **Configuration examples** use YAML format
- **File paths** are relative to `/root/Ava_AI/`
- **Line references** use `file.py:123` format

## Contributing

Documentation improvements are welcome. Each guide follows this structure:
1. Overview and purpose
2. Quick examples
3. Detailed explanations
4. Configuration options
5. Troubleshooting section
