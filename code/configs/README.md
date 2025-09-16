# Ava MoE++ Configuration Files

This directory contains pre-configured settings for training Ava MoE++ models optimized for different hardware and use cases.

## 📁 Configuration Structure

```
configs/
├── cpu/                  # CPU-optimized configurations
│   ├── small.yaml       # 50M params - Development/testing
│   └── medium.yaml      # 200M params - Small-scale training
├── gpu/                  # GPU-optimized configurations
│   ├── base.yaml        # 500M params - Standard training
│   └── large.yaml       # 1.5B params - Large-scale training
├── dev.yaml             # 10M params - Quick testing/debugging
├── memory_optimized.yaml # Optimized for low memory systems
├── speed_optimized.yaml  # Optimized for fast training
└── quality_optimized.yaml # Optimized for best model quality
```

## 🚀 Quick Start

```bash
# For testing and development (fastest)
python scripts/training/train.py --config configs/dev.yaml

# For CPU training
python scripts/training/train.py --config configs/cpu/small.yaml

# For GPU training
python scripts/training/train.py --config configs/gpu/base.yaml
```

## 📊 Configuration Comparison

| Config | Parameters | Memory | Hardware | Training Speed | Use Case |
|--------|-----------|--------|----------|----------------|----------|
| `dev.yaml` | ~10M | 2GB | Any | Very Fast | Development, testing |
| `cpu/small.yaml` | ~50M | 8GB | 8+ CPU cores | Fast | Small experiments |
| `cpu/medium.yaml` | ~200M | 16GB | 16+ CPU cores | Moderate | CPU training |
| `gpu/base.yaml` | ~500M | 12GB VRAM | RTX 3060+ | Fast | Standard GPU |
| `gpu/large.yaml` | ~1.5B | 24GB VRAM | RTX 3090+ | Moderate | Large models |
| `memory_optimized.yaml` | ~30M | 4GB | Any | Slow | Low memory |
| `speed_optimized.yaml` | ~100M | 8GB | GPU preferred | Very Fast | Quick training |
| `quality_optimized.yaml` | ~1B | 16GB+ VRAM | High-end GPU | Slow | Best quality |

## 🔧 Configuration Details

### CPU Configurations

#### `cpu/small.yaml`
- **Model Size**: ~50M parameters
- **Hardware**: 8+ CPU cores, 16GB RAM
- **Features**: 4 experts, 8 layers, 512 hidden size
- **Best For**: Development, testing, small datasets

#### `cpu/medium.yaml`
- **Model Size**: ~200M parameters
- **Hardware**: 16+ CPU cores, 32GB RAM
- **Features**: 8 experts, 12 layers, 768 hidden size
- **Best For**: Small-scale training, fine-tuning

### GPU Configurations

#### `gpu/base.yaml`
- **Model Size**: ~500M parameters
- **Hardware**: GPU with 8GB+ VRAM
- **Features**: 16 experts, 16 layers, mixed precision
- **Best For**: Standard training, research

#### `gpu/large.yaml`
- **Model Size**: ~1.5B parameters
- **Hardware**: GPU with 24GB+ VRAM
- **Features**: 32 experts, 24 layers, BF16 training
- **Best For**: Large-scale training, production models

### Specialized Configurations

#### `dev.yaml`
- **Purpose**: Quick testing and debugging
- **Features**: Minimal model (2 layers, 2 experts)
- **Training**: 100 steps max, small batches
- **Use**: Code development, unit testing

#### `memory_optimized.yaml`
- **Purpose**: Training on memory-constrained systems
- **Features**: Gradient checkpointing, small batches, streaming data
- **Techniques**: CPU offloading, activation checkpointing
- **Use**: Limited RAM/VRAM systems

#### `speed_optimized.yaml`
- **Purpose**: Maximum training speed
- **Features**: Large batches, no dropout, compiled kernels
- **Optimizations**: Flash attention, torch.compile, CUDA graphs
- **Use**: Quick experiments, hyperparameter search

#### `quality_optimized.yaml`
- **Purpose**: Best model quality
- **Features**: 64 experts, extensive regularization
- **Techniques**: Curriculum learning, EMA, SAM optimizer
- **Use**: Final training, research, benchmarks

## 🎯 Choosing the Right Configuration

### By Hardware

**CPU Only:**
- Limited RAM (< 16GB): `memory_optimized.yaml`
- Standard CPU (16GB): `cpu/small.yaml`
- Powerful CPU (32GB+): `cpu/medium.yaml`

**GPU Available:**
- Consumer GPU (8GB): `gpu/base.yaml`
- High-end GPU (24GB+): `gpu/large.yaml`
- Multiple GPUs: Modify `gpu/large.yaml` with distributed settings

### By Use Case

**Development:**
- Quick testing: `dev.yaml`
- Feature development: `cpu/small.yaml`

**Training:**
- First experiments: `speed_optimized.yaml`
- Hyperparameter search: `cpu/small.yaml` or `gpu/base.yaml`
- Final training: `quality_optimized.yaml`

**Production:**
- Inference optimization: `speed_optimized.yaml` (modified)
- Quality focus: `gpu/large.yaml` or `quality_optimized.yaml`

## 🛠️ Customizing Configurations

### Override from Command Line

```bash
# Override specific parameters
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --epochs 20
```

### Create Custom Configuration

1. Copy a base configuration:
```bash
cp configs/cpu/small.yaml configs/custom.yaml
```

2. Edit the parameters:
```yaml
model:
  hidden_size: 640  # Custom size
  num_experts: 6    # Custom expert count

training:
  batch_size: 16
  learning_rate: 2e-4
```

3. Use your custom config:
```bash
python scripts/training/train.py --config configs/custom.yaml
```

## 📈 Performance Tips

### Memory Optimization
- Enable `gradient_checkpointing: true`
- Reduce `batch_size` and increase `gradient_accumulation_steps`
- Use `memory_optimized.yaml` as base

### Speed Optimization
- Disable `gradient_checkpointing`
- Increase `batch_size`
- Enable `mixed_precision` and `torch_compile`
- Use `speed_optimized.yaml` as base

### Quality Optimization
- Increase `num_experts` and `top_k_experts`
- Enable curriculum learning and regularization
- Use longer training with lower learning rate
- Use `quality_optimized.yaml` as base

## 🔍 Configuration Validation

Test your configuration before full training:

```bash
# Quick validation (1 epoch, small data)
python scripts/training/train.py \
    --config configs/your_config.yaml \
    --epochs 1 \
    --max-length 256
```

## 📝 Configuration Keys

### Model Parameters
- `hidden_size`: Model dimension
- `num_layers`: Number of transformer layers
- `num_experts`: Total experts in MoE
- `top_k_experts`: Active experts per token

### Training Parameters
- `batch_size`: Samples per batch
- `learning_rate`: Initial learning rate
- `num_epochs`: Training epochs
- `gradient_accumulation_steps`: Gradient accumulation

### Data Parameters
- `data_dir`: Path to training data
- `max_length`: Maximum sequence length
- `max_train_examples`: Limit training samples

### Hardware Parameters
- `device`: Target device (cpu/cuda/auto)
- `mixed_precision`: Enable FP16/BF16
- `num_workers`: Data loading threads

## 🆘 Troubleshooting

### Out of Memory
- Use `memory_optimized.yaml`
- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Reduce `max_length`

### Slow Training
- Use `speed_optimized.yaml`
- Increase `batch_size`
- Reduce `num_experts`
- Disable logging

### Poor Quality
- Use `quality_optimized.yaml`
- Train longer (`num_epochs`)
- Reduce `learning_rate`
- Increase `num_experts`

## 📚 Further Reading

- [Configuration Guide](../docs/configuration.md)
- [Training Guide](../docs/training_guide.md)
- [Architecture Overview](../docs/architecture.md)