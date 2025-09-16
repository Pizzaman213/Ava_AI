# Training Guide

Comprehensive guide for training Ava MoE++ models.

## Table of Contents
- [Basic Training](#basic-training)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Advanced Training](#advanced-training)
- [Monitoring](#monitoring)
- [Optimization Tips](#optimization-tips)
- [Troubleshooting](#troubleshooting)

## Basic Training

### Simple Training Command

```bash
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --epochs 10 \
    --batch-size 4
```

### Full Training Command with All Options

```bash
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --data-dir /project/code/data/pretraining/processed \
    --max-length 512 \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 5e-4 \
    --gradient-accumulation 4 \
    --output-dir /project/code/outputs \
    --save-every 500 \
    --device cpu \
    --num-workers 4 \
    --seed 42
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

- **Format**: Parquet files with a 'text' column
- **Size**: At least 1000 samples recommended
- **Length**: Texts should be meaningful (>50 tokens)
- **Quality**: Clean, deduplicated text
- **Split**: 90% train, 10% validation

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

| Config | Use Case | Parameters | Memory | Training Speed |
|--------|----------|------------|--------|----------------|
| `cpu/small.yaml` | Testing/Development | 50M | 8GB | Fast |
| `cpu/medium.yaml` | Small datasets | 200M | 16GB | Medium |
| `gpu/base.yaml` | Standard training | 500M | 12GB VRAM | Fast |
| `gpu/large.yaml` | Production | 1.5B | 24GB VRAM | Medium |

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
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --resume /project/code/outputs/checkpoint_epoch_5.pt
```

### Multi-Stage Training

```bash
# Stage 1: Warm-up with small batches
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4

# Stage 2: Main training
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --resume outputs/checkpoint_epoch_5.pt \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 5e-4

# Stage 3: Fine-tuning
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --resume outputs/checkpoint_epoch_15.pt \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-5
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
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better (good: <20)
- **Learning Rate**: Check warmup and decay
- **Gradient Norm**: Should be stable (<10)

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

1. **Gradient Checkpointing** (enabled by default)
   ```yaml
   model:
     gradient_checkpointing: true
   ```

2. **Reduce Batch Size**
   ```bash
   --batch-size 1 --gradient-accumulation 16
   ```

3. **Reduce Sequence Length**
   ```bash
   --max-length 256
   ```

4. **Use Smaller Model**
   ```bash
   --config configs/cpu/small.yaml
   ```

### Speed Optimization

1. **Increase Batch Size** (if memory allows)
   ```bash
   --batch-size 16
   ```

2. **Use Multiple Workers**
   ```bash
   --num-workers 8
   ```

3. **Disable Unnecessary Features**
   ```yaml
   model:
     use_cache: false  # During training
     use_memory_efficient_attention: false  # If not needed
   ```

4. **Mixed Precision** (GPU only)
   ```yaml
   training:
     mixed_precision: true
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
# Solution 1: Reduce batch size
--batch-size 1

# Solution 2: Enable gradient accumulation
--batch-size 1 --gradient-accumulation 8

# Solution 3: Reduce model size
--config configs/cpu/small.yaml

# Solution 4: Reduce sequence length
--max-length 256
```

#### Slow Training
```bash
# Solution 1: Check data loading
--num-workers 0  # Test without multiprocessing

# Solution 2: Reduce logging frequency
--log-level WARNING

# Solution 3: Use smaller validation set
# Edit max_val_samples in train.py
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
# Check data
python -c "
from src.Ava.data import create_dataloaders
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
train_loader, val_loader = create_dataloaders(
    tokenizer=tokenizer,
    batch_size=1,
    max_train_samples=10
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