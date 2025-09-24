# Training Guide

Comprehensive guide for training Ava MoE++ models with DeepSpeed integration and enhanced features.

## Table of Contents
- [Basic Training](#basic-training)
- [DeepSpeed Integration](#deepspeed-integration)
- [Enhanced Features](#enhanced-features)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Advanced Training](#advanced-training)
- [Monitoring](#monitoring)
- [Optimization Tips](#optimization-tips)
- [Troubleshooting](#troubleshooting)

## Basic Training

### Simple Training Command

```bash
# Basic CPU training
python scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --epochs 10 \
    --batch-size 4

# GPU training with enhanced features
python scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --epochs 10 \
    --batch-size 8 \
    --use-moh \
    --use-episodic-memory
```

### Full Training Command with All Options

```bash
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --data-dir /project/code/data/pretraining/processed \
    --max-length 1024 \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 2e-4 \
    --gradient-accumulation 8 \
    --output-dir /project/code/outputs \
    --save-every 500 \
    --device auto \
    --num-workers 8 \
    --seed 42 \
    --use-deepspeed \
    --zero-stage 2 \
    --use-rag \
    --use-advanced-losses \
    --use-gradient-surgery \
    --ultra-fast-mode
```

## DeepSpeed Integration

### DeepSpeed ZeRO Training

DeepSpeed ZeRO enables training much larger models by sharding optimizer states, gradients, and parameters:

#### ZeRO Stage 1 (Optimizer Sharding)
```bash
# Basic DeepSpeed training with optimizer sharding
python scripts/training/train.py \
    --config configs/distributed/deepspeed_zero1.yaml \
    --use-deepspeed \
    --zero-stage 1

# Multi-GPU training
deepspeed --num_gpus=2 scripts/training/train.py \
    --config configs/distributed/deepspeed_zero1.yaml \
    --use-deepspeed
```

#### ZeRO Stage 2 (Optimizer + Gradient Sharding)
```bash
# More memory efficient training
deepspeed --num_gpus=4 scripts/training/train.py \
    --config configs/distributed/deepspeed_zero2.yaml \
    --use-deepspeed \
    --zero-stage 2
```

#### ZeRO Stage 3 (Full Parameter Sharding)
```bash
# Maximum memory efficiency for very large models
deepspeed --num_gpus=8 scripts/training/train.py \
    --config configs/distributed/deepspeed_zero3.yaml \
    --use-deepspeed \
    --zero-stage 3 \
    --cpu-offload
```

### DeepSpeed Features

#### CPU/NVMe Offloading
```bash
# Offload optimizer states to CPU
python scripts/training/train.py \
    --config configs/distributed/deepspeed_zero3.yaml \
    --use-deepspeed \
    --cpu-offload

# Offload to NVMe for even larger models
python scripts/training/train.py \
    --config configs/distributed/deepspeed_zero3.yaml \
    --use-deepspeed \
    --nvme-offload
```

#### Mixed Precision Training
```bash
# FP16 training
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-deepspeed \
    --fp16

# BF16 training (better numerical stability)
python scripts/training/train.py \
    --config configs/hardware/a100_80gb.yaml \
    --use-deepspeed \
    --bf16

# FP8 training (H100 only)
python scripts/training/train.py \
    --config configs/hardware/h100_80gb.yaml \
    --use-deepspeed \
    --fp8
```

## Enhanced Features

### RAG (Retrieval Augmented Generation)
```bash
# Enable RAG system
python scripts/training/train.py \
    --config configs/research/rag_enabled.yaml \
    --use-rag \
    --knowledge-base-path data/knowledge_base
```

### NVFP4 Quantization
```bash
# 4-bit quantization for memory efficiency
python scripts/training/train.py \
    --config configs/research/quantization_nvfp4.yaml \
    --use-nvfp4 \
    --bit-width 4
```

### MoH (Mixture of Heads)
```bash
# Enable specialized attention heads
python scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --use-moh \
    --moh-num-heads 8
```

### Advanced Loss Functions
```bash
# Enable focal loss for hard example focus
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-focal-loss \
    --focal-gamma 2.0

# Enable contrastive loss for better representations
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-contrastive-loss

# Enable diversity loss for expert specialization
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-diversity-loss
```

### Gradient Surgery
```bash
# Enable PCGrad for multi-task learning
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-gradient-surgery \
    --gradient-surgery-method pcgrad

# Enable CAGrad for conflict-aware optimization
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --use-gradient-surgery \
    --gradient-surgery-method cagrad
```

### Episodic Memory
```bash
# Enable continual learning with episodic memory
python scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --use-episodic-memory \
    --memory-size 1000 \
    --replay-ratio 0.3
```

### Performance Modes
```bash
# Ultra-fast mode for rapid prototyping
python scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --ultra-fast-mode

# Express mode for production training
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --express-mode

# Minimal progress mode for long training runs
python scripts/training/train.py \
    --config configs/hardware/a100_80gb.yaml \
    --minimal-progress
```

## Data Preparation

### Data Directory Structure

```
/project/code/data/pretraining/processed/
├── train_*.parquet    # Training files
├── val_*.parquet      # Validation files
└── test_*.parquet     # Test files
```

### Creating Training Data

#### From Text Files
```python
import pandas as pd
from pathlib import Path

# Read text files
texts = []
for file in Path("raw_data/").glob("*.txt"):
    with open(file, 'r') as f:
        texts.append(f.read())

# Create DataFrame
df = pd.DataFrame({"text": texts})

# Save as parquet
df.to_parquet("/project/code/data/pretraining/processed/train_custom.parquet")
```

#### From CSV
```python
import pandas as pd

# Load CSV
df = pd.read_csv("data.csv")

# Ensure 'text' column exists
if 'content' in df.columns:
    df = df.rename(columns={'content': 'text'})

# Save as parquet
df.to_parquet("/project/code/data/pretraining/processed/train_data.parquet")
```

#### From JSON Lines
```python
import pandas as pd

# Load JSONL
df = pd.read_json("data.jsonl", lines=True)

# Process and save
df[['text']].to_parquet("/project/code/data/pretraining/processed/train_data.parquet")
```

### Data Requirements

- **Format**: Parquet files with a 'text' column or multi-column datasets
- **Size**: At least 1000 samples recommended (10K+ for enhanced features)
- **Length**: Texts should be meaningful (>50 tokens, up to 8192 for H100)
- **Quality**: Clean, deduplicated text
- **Split**: 90% train, 10% validation
- **Streaming**: Support for large datasets with streaming data loading
- **Multi-Column**: Support for complex dataset structures with multiple columns

## Configuration

### Key Configuration Parameters

```yaml
model:
  hidden_size: 512          # Model dimension
  num_layers: 8             # Number of transformer layers
  num_attention_heads: 8    # Attention heads
  num_experts: 4            # Number of experts in MoE
  num_experts_per_tok: 2    # Active experts per token

training:
  batch_size: 4             # Batch size per device
  learning_rate: 5e-4       # Learning rate
  num_epochs: 10            # Training epochs
  gradient_accumulation_steps: 4  # Gradient accumulation
  warmup_steps: 500         # LR warmup steps
```

### Configuration Presets

#### GPU Configurations
| Config | Parameters | VRAM | Features | Use Case |
|--------|------------|------|----------|----------|
| `gpu/tiny.yaml` | ~100M | 6-8GB | Basic + Streaming | Entry-level development |
| `gpu/small.yaml` | ~150M | 8-10GB | MoH + Memory + Quantization | Development & research |
| `gpu/medium.yaml` | ~300M | 12-16GB | MoH + Cross-attention | Moderate research |
| `gpu/large.yaml` | ~1.5B | 24GB+ | All features + DeepSpeed | Production training |
| `gpu/xl.yaml` | ~900M | 22GB+ | Advanced features | Large-scale research |
| `gpu/1b.yaml` | ~1B | 24GB+ | Full feature set | State-of-the-art |

#### Distributed Configurations
| Config | ZeRO Stage | Memory Savings | Min GPUs | Use Case |
|--------|------------|----------------|----------|----------|
| `distributed/deepspeed_zero1.yaml` | 1 | ~4x optimizer | 2-8 GPUs | Optimizer sharding |
| `distributed/deepspeed_zero2.yaml` | 2 | ~8x opt+grad | 4-16 GPUs | Large model training |
| `distributed/deepspeed_zero3.yaml` | 3 | ~64x parameters | 8+ GPUs | Ultra-large models |

#### Research Configurations
| Config | Focus | Features | Use Case |
|--------|-------|----------|----------|
| `research/rag_enabled.yaml` | RAG Research | Retrieval + Cross-attention | Knowledge-grounded generation |
| `research/quantization_nvfp4.yaml` | Quantization | 4-bit NVFP4 | Memory-efficient deployment |

#### Hardware Configurations
| Config | Hardware | Memory | Special Features |
|--------|----------|--------|------------------|
| `hardware/a100_80gb.yaml` | A100 80GB | 80GB HBM2e | BF16, Tensor Cores, 70GB pool |
| `hardware/h100_80gb.yaml` | H100 80GB | 80GB HBM3 | FP8, 4th Gen Tensor Cores |

### Custom Configuration

Create `configs/custom.yaml`:
```yaml
model:
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12
  num_experts: 8
  vocab_size: 50257

training:
  batch_size: 8
  learning_rate: 3e-4
  num_epochs: 20
```

## Advanced Training

### Resume from Checkpoint

```bash
# Resume standard training
python scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --resume /project/code/outputs/checkpoint_epoch_5.pt

# Resume DeepSpeed training
deepspeed --num_gpus=4 scripts/training/train.py \
    --config configs/distributed/deepspeed_zero2.yaml \
    --use-deepspeed \
    --resume /project/code/outputs/deepspeed_checkpoint
```

### Multi-Stage Training

```bash
# Stage 1: Warm-up with basic features
python scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4

# Stage 2: Main training with enhanced features
python scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --resume outputs/checkpoint_epoch_5.pt \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 5e-4 \
    --use-moh \
    --use-episodic-memory

# Stage 3: Advanced training with DeepSpeed
deepspeed --num_gpus=4 scripts/training/train.py \
    --config configs/distributed/deepspeed_zero2.yaml \
    --resume outputs/checkpoint_epoch_15.pt \
    --epochs 15 \
    --use-deepspeed \
    --use-rag \
    --use-advanced-losses

# Stage 4: Fine-tuning with gradient surgery
python scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --resume outputs/checkpoint_epoch_30.pt \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --use-gradient-surgery \
    --gradient-surgery-method cagrad
```

### Curriculum Learning

The model supports automatic curriculum learning:

```yaml
training:
  curriculum_learning: true
  initial_sequence_length: 256
  curriculum_warmup_steps: 1000
  progressive_sequence_growth: true
  sequence_growth_rate: 1.5
  sequence_growth_interval: 5000
```

### Expert Analysis During Training

```python
# In your training loop, access expert statistics
outputs = model(input_ids, return_expert_stats=True)
expert_stats = outputs['expert_stats']
print(f"Expert utilization: {expert_stats['expert_loads']}")
```

## Monitoring

### Real-time Monitoring

```bash
# Watch training log
tail -f /project/code/outputs/training_*.log

# Monitor system resources
htop  # or top

# Watch GPU usage (if applicable)
nvidia-smi -l 1

# Monitor outputs directory
watch -n 10 ls -lah /project/code/outputs/
```

### Training Metrics

Track these key metrics:

#### Core Metrics
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better (good: <20)
- **Learning Rate**: Check warmup and decay
- **Gradient Norm**: Should be stable (<10)

#### Enhanced Metrics
- **Expert Utilization**: Should be balanced across experts
- **Memory Usage**: Monitor episodic memory efficiency
- **RAG Retrieval Accuracy**: Quality of retrieved knowledge
- **Gradient Surgery Conflicts**: Number of conflicting gradients resolved
- **Auxiliary Loss Components**: Focal, contrastive, diversity losses
- **DeepSpeed Memory Savings**: Actual vs. theoretical savings

#### Performance Metrics
- **Throughput**: Tokens/second with enhanced features
- **Memory Pool Utilization**: Efficiency of memory management
- **Expert Load Balance**: Distribution of expert usage
- **Quantization Efficiency**: Memory reduction with NVFP4

### Using TensorBoard

```bash
# If tensorboard logging is enabled
tensorboard --logdir /project/code/outputs/runs/
```

### Custom Logging

Add to your training script:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/project/code/outputs/custom.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Optimization Tips

### Memory Optimization

1. **DeepSpeed ZeRO Stages**
   ```bash
   # ZeRO-1: Optimizer sharding (~4x memory savings)
   --use-deepspeed --zero-stage 1

   # ZeRO-2: Optimizer + gradient sharding (~8x memory savings)
   --use-deepspeed --zero-stage 2

   # ZeRO-3: Full parameter sharding (~64x memory savings)
   --use-deepspeed --zero-stage 3
   ```

2. **CPU/NVMe Offloading**
   ```bash
   # Offload optimizer states to CPU
   --use-deepspeed --zero-stage 3 --cpu-offload

   # Offload to NVMe for ultra-large models
   --use-deepspeed --zero-stage 3 --nvme-offload
   ```

3. **NVFP4 Quantization**
   ```bash
   # 4-bit quantization (4x memory reduction)
   --use-nvfp4 --bit-width 4
   ```

4. **Memory Pool Management**
   ```yaml
   memory:
     enable_memory_pool: true
     pool_size_gb: 20.0
     clear_cache_frequency: 25
   ```

5. **Traditional Optimizations**
   ```yaml
   model:
     gradient_checkpointing: true
   training:
     batch_size: 1
     gradient_accumulation_steps: 16
   data:
     max_length: 256
   ```

### Speed Optimization

1. **Performance Modes**
   ```bash
   # Ultra-fast mode (3x speed increase)
   --ultra-fast-mode

   # Express mode (2x speed increase)
   --express-mode

   # Minimal progress mode (reduce logging overhead)
   --minimal-progress
   ```

2. **Hardware-Specific Optimizations**
   ```bash
   # A100 optimized configuration
   --config configs/hardware/a100_80gb.yaml --bf16

   # H100 optimized with FP8
   --config configs/hardware/h100_80gb.yaml --fp8
   ```

3. **Compilation and Optimization**
   ```yaml
   compilation:
     enabled: true
     backend: "inductor"
     fullgraph: true
     dynamic: false
   ```

4. **Streaming Data Loading**
   ```yaml
   data_loading:
     streaming: true
     buffer_size: 20000
     distributed: true
   ```

5. **Traditional Optimizations**
   ```bash
   --batch-size 16  # If memory allows
   --num-workers 16  # More workers for data loading
   --fp16  # Mixed precision training
   ```

### Quality Optimization

1. **Learning Rate Schedule**
   ```yaml
   training:
     lr_scheduler_type: cosine
     warmup_ratio: 0.1
   ```

2. **Regularization**
   ```yaml
   training:
     weight_decay: 0.01
     dropout_rate: 0.1
   ```

3. **Gradient Clipping**
   ```yaml
   training:
     max_grad_norm: 1.0
   ```

4. **Longer Training**
   ```bash
   --epochs 50 --save-every 1000
   ```

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory (OOM)
```bash
# Solution 1: Use DeepSpeed ZeRO
--use-deepspeed --zero-stage 3 --cpu-offload

# Solution 2: Enable NVFP4 quantization
--use-nvfp4 --bit-width 4

# Solution 3: Reduce batch size with gradient accumulation
--batch-size 1 --gradient-accumulation 16

# Solution 4: Use smaller model configuration
--config configs/gpu/tiny.yaml

# Solution 5: Reduce sequence length
--max-length 512

# Solution 6: Enable memory pool management
# (configured in YAML with enable_memory_pool: true)
```

#### Slow Training
```bash
# Solution 1: Enable performance modes
--ultra-fast-mode  # or --express-mode

# Solution 2: Use hardware-optimized configs
--config configs/hardware/a100_80gb.yaml

# Solution 3: Enable streaming data loading
# (configured in YAML with streaming: true)

# Solution 4: Reduce logging and validation frequency
--log-level WARNING --minimal-progress

# Solution 5: Optimize data loading
--num-workers 16 --prefetch-factor 4

# Solution 6: Enable compilation (PyTorch 2.0+)
# (configured in YAML with compilation enabled)
```

#### Loss Not Decreasing
```bash
# Solution 1: Lower learning rate
--learning-rate 1e-5

# Solution 2: Check data quality
python -c "import pandas as pd; df = pd.read_parquet('data.parquet'); print(df.head())"

# Solution 3: Smaller model
--config configs/cpu/small.yaml

# Solution 4: More training steps
--epochs 50
```

#### NaN Loss
```bash
# Solution 1: Reduce learning rate
--learning-rate 1e-6

# Solution 2: Enable gradient clipping
# Already enabled in config

# Solution 3: Check for bad data
# Remove samples with very long sequences
```

### Debug Mode

Enable detailed debugging:
```bash
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --log-level DEBUG \
    --batch-size 1 \
    --epochs 1
```

### Validation

Validate your setup:
```python
# Check data using multi-column loader
python -c "
from src.Ava.multi_column_data import create_multi_column_dataloader
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Default configuration for simple text data
config = {
    'columns': [{'name': 'text', 'type': 'text', 'role': 'input', 'max_length': 512}],
    'combine_strategy': 'concatenate',
    'max_samples': 10,
    'validation_enabled': True
}

train_loader = create_multi_column_dataloader(
    config=config, tokenizer=tokenizer, batch_size=1, split='train'
)
val_loader = create_multi_column_dataloader(
    config=config, tokenizer=tokenizer, batch_size=1, split='val'
)
print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
"

# Check model
python -c "
from src.Ava.models import EnhancedMoEModel, EnhancedMoEConfig
config = EnhancedMoEConfig(hidden_size=128, num_layers=2)
model = EnhancedMoEModel(config)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

## Best Practices

1. **Start Small**: Begin with small models and datasets
2. **Monitor Regularly**: Check logs every few hours
3. **Save Checkpoints**: Use `--save-every` for regular saves
4. **Track Experiments**: Document hyperparameters and results
5. **Validate Early**: Run validation after first epoch
6. **Use Version Control**: Track config changes in git
7. **Clean Data**: Ensure high-quality training data
8. **Incremental Changes**: Adjust one parameter at a time

## Next Steps

- [Evaluation Guide](evaluation.md) - Evaluate your trained models
- [Generation Guide](generation.md) - Generate text with your models
- [Fine-tuning Guide](advanced/fine_tuning.md) - Adapt to specific tasks
- [Architecture Overview](architecture.md) - Understand model internals