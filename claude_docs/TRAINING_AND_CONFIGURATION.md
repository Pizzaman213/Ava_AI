# Training & Configuration Guide

Complete guide to training models and configuring the system.

## 🚀 Quick Start

### 3-Minute Setup

**Option 1: Train from Scratch (Recommended)**
```bash
# 1. Copy config
cp configs/gpu/small.yaml configs/gpu/custom.yaml

# 2. Edit config if needed (optional)
# nano configs/gpu/custom.yaml

# 3. Start training
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

**Option 2: Fine-tune Existing Model**
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --resume outputs/runs/latest/checkpoint
```

**Option 3: RLHF Fine-tuning**
```bash
python scripts/6_rhlf_Finetuning/prepare_prompts.py --create-samples
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/small.yaml
```

## ⚙️ Configuration System

### YAML Structure

```yaml
# Model architecture
model:
  num_layers: 12
  hidden_size: 768
  num_attention_heads: 12
  vocab_size: 65536

# Training hyperparameters
training:
  learning_rate: 0.006
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 2000

# Data loading
data:
  tokenizer_name: /path/to/tokenizer
  dataset_path: ./data/processed

# Optimization features
performance:
  use_flash_attention: true
  use_torch_compile: true
  mixed_precision: bf16

# RLHF fine-tuning
rlhf:
  policy_model_path: ./outputs/latest
  prompt_dataset_path: ./data/rlhf/prompts.json
  ppo:
    learning_rate: 5.0e-7
    max_gen_length: 128
```

### Key Parameters Explained

**Learning Rate** (critical):
- Too low: Training stalls, no learning
- Too high: Loss spikes, NaN values
- Good range for pre-training: 0.003-0.01
- **Common mistake**: Using fine-tuning LR (0.0001) for pre-training

**Batch Size**:
- Effective batch = batch_size * gradient_accumulation_steps
- Larger batch: smoother training, faster GPU utilization
- Smaller batch: lower memory, noisier gradients
- Typical range: 32-64 effective batch size

**Gradient Accumulation**:
- Updates weights after N batches
- Simulates larger batch size with limited GPU memory
- Example: batch_size=8, accumulate=4 → effective=32

**Warmup Steps**:
- Gradually increase LR at training start
- Prevents early instability
- Typical: 2-5% of total training steps

**Number of Epochs**:
- Typically 1-3 for large datasets
- More epochs for small datasets
- Training converges after first epoch usually

## 📋 Configuration Templates

### For Development/Testing (Tiny Model)
```yaml
model:
  vocab_size: 65536
  hidden_size: 512
  num_layers: 6
  
training:
  learning_rate: 0.008
  batch_size: 32
  gradient_accumulation_steps: 1
  num_epochs: 1
  
performance:
  use_flash_attention: false
  mixed_precision: "fp32"
```

### For Production (Small Model) ⭐ RECOMMENDED
```yaml
model:
  vocab_size: 65536
  hidden_size: 768
  num_layers: 12
  
training:
  learning_rate: 0.006
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 2
  
performance:
  use_flash_attention: true
  use_torch_compile: true
  mixed_precision: "bf16"
```

### For Research (Base Model)
```yaml
model:
  vocab_size: 65536
  hidden_size: 1024
  num_layers: 24
  
training:
  learning_rate: 0.003
  batch_size: 8
  gradient_accumulation_steps: 8
  num_epochs: 3
  
performance:
  use_flash_attention: true
  use_torch_compile: true
  mixed_precision: "bf16"
  gradient_checkpointing: true
```

### For Maximum Quality (Large Model)
```yaml
model:
  vocab_size: 65536
  hidden_size: 1280
  num_layers: 32
  
training:
  learning_rate: 0.001
  batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 5
  
performance:
  use_flash_attention: true
  use_torch_compile: true
  mixed_precision: "bf16"
  gradient_checkpointing: true
  deepspeed: true
```

## 🔄 Configuration Customization

### Change Learning Rate
```yaml
training:
  learning_rate: 0.003  # Adjust from 0.006
```

### Enable/Disable Features
```yaml
performance:
  use_flash_attention: true     # Enable fast attention
  use_torch_compile: true        # Enable torch.compile
  gradient_checkpointing: true   # Enable for memory saving
```

### Set Model Size
```yaml
model:
  num_layers: 24         # More layers = larger model
  hidden_size: 1024      # Larger hidden dim
  num_attention_heads: 16 # More attention heads
```

### Distributed Training
```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2          # 1, 2, or 3
  train_batch_size: 32
  gradient_accumulation_steps: 1
```

## 📊 Hardware Requirements

### GPU Memory by Model Size

| Model | GPU Memory | Batch Size | Effective Batch |
|-------|-----------|-----------|-----------------|
| Tiny (100M) | 4GB | 32 | 32 |
| Small (233M) | 8GB | 8 | 32 |
| Base (500M) | 16GB | 8 | 64 |
| Large (1.3B) | 24GB | 4 | 64 |

### Recommended GPUs

**Best**: A100 (40/80GB) or H100 (80GB)
**Good**: RTX 3090/4090 (24GB), A6000 (48GB)
**OK**: RTX 3080/4080 (10/16GB), RTX A5000 (24GB)
**Minimum**: RTX 3060 (12GB)

### CPU/RAM Requirements

| Model | CPU Type | RAM |
|-------|----------|-----|
| Tiny | i5 or better | 16GB |
| Small | i7 or better | 32GB |
| Base | Xeon/EPYC | 64GB |
| Large | Xeon/EPYC | 128GB+ |

## 📁 Data Preparation

### Directory Structure
```
data/
├── raw/                 # Your raw text files
├── processed/           # Prepared training data
└── rlhf/                # RLHF prompts
```

### Prepare Data
```bash
# 1. Place raw data
cp my_data.txt data/raw/

# 2. Prepare and tokenize
python scripts/data_prep/prepare_data.py \
  --input data/raw/my_data.txt \
  --output data/processed/my_data.jsonl

# 3. Verify data
python -c "import jsonlines; count = sum(1 for _ in jsonlines.open('data/processed/my_data.jsonl')); print(f'Examples: {count}')"
```

### Data Format

Input format (one JSON per line):
```json
{"text": "Your training text here..."}
{"text": "Another example..."}
```

## 🎯 Training Commands

### Basic Training
```bash
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

### Resume from Checkpoint
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --resume outputs/runs/latest/checkpoint
```

### Fresh Start (discard old checkpoints)
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --fresh-start
```

### Custom Data Path
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --data-path ./my_custom_data
```

### Disable W&B Logging
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --no-wandb
```

### Override Specific Parameters
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --batch-size 16 \
  --learning-rate 0.003 \
  --num-epochs 5
```

## 📈 Learning Rate Tuning

### LR Finder Tool
```bash
# Find optimal learning rate
python scripts/training/lr_finder.py \
  --config configs/gpu/small.yaml \
  --output lr_finder_plot.png
```

This will:
1. Train for a few iterations with increasing LR
2. Plot loss vs LR
3. Suggest optimal LR range

### Manual LR Selection

**Rule of thumb**:
```
LR ~ 1e-3 for small models (< 500M params)
LR ~ 1e-4 for medium models (500M - 5B)
LR ~ 1e-5 for large models (> 5B)
```

**Pre-training vs Fine-tuning**:
```
Pre-training (from scratch):     0.001 - 0.01   (higher)
Fine-tuning (existing model):    0.0001 - 0.001 (lower)
RLHF fine-tuning:                1e-7 - 1e-6    (very low)
```

## 🔍 Monitoring Training

### Check Training Status
```bash
# Watch loss decrease
tail -f outputs/runs/latest/logs/training.log

# Monitor GPU memory
watch nvidia-smi

# Check trainer metrics
python -c "
import json
with open('outputs/runs/latest/logs/metrics.json') as f:
    for line in f:
        data = json.loads(line)
        print(f\"Step {data['step']}: Loss={data['loss']:.4f}\")
"
```

### Expected Loss Curve
```
Step 1,000:   Loss ≈ 6.0-7.0  ✓ Learning started
Step 10,000:  Loss ≈ 3.0-4.0  ✓ Steady progress
Step 50,000:  Loss ≈ 2.0-2.5  ✓ Good convergence
Step 100,000: Loss ≈ 1.8-2.0  ✓ Excellent
```

### Warning Signs
```
Step 1,000:   Loss still 10+   ❌ LR too low
Step 1,000:   Loss = NaN       ❌ LR too high
Loss oscillating wildly         ❌ Batch size too small
Loss plateaus early             ❌ May need more data
```

## 🚀 Optimization Strategies

### For Speed (Fastest Training)
```yaml
performance:
  use_flash_attention: true
  use_torch_compile: true
  
training:
  batch_size: 32
  gradient_accumulation_steps: 1
```

### For Memory Efficiency
```yaml
performance:
  gradient_checkpointing: true
  
training:
  batch_size: 4
  gradient_accumulation_steps: 8
```

### Balanced (Recommended)
```yaml
performance:
  use_flash_attention: true
  use_torch_compile: true
  gradient_checkpointing: false
  
training:
  batch_size: 8
  gradient_accumulation_steps: 4
```

### Multi-GPU (DeepSpeed)
```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2

training:
  batch_size: 32
  gradient_accumulation_steps: 4
```

## 🔧 Troubleshooting Configuration

### Problem: Out of Memory (OOM)
**Solution**:
```yaml
training:
  batch_size: 4           # Reduce
  gradient_accumulation_steps: 8  # Increase
  
performance:
  gradient_checkpointing: true
  max_gen_length: 64      # Reduce sequences
```

### Problem: Training Very Slow
**Solution**:
```yaml
performance:
  use_flash_attention: true    # Enable
  use_torch_compile: true      # Enable
  
training:
  batch_size: 32              # Increase if memory allows
```

### Problem: Loss Not Decreasing
**Solution**:
```yaml
training:
  learning_rate: 0.01     # Increase (but not too much)
  warmup_steps: 1000      # Decrease
  num_epochs: 2           # Increase
```

### Problem: Loss Spikes/NaN
**Solution**:
```yaml
training:
  learning_rate: 0.001    # Decrease significantly
  max_gradient_norm: 0.5  # Tighter clipping
  
performance:
  mixed_precision: "fp32" # Use full precision
```

## 📚 Examples

### Example 1: Quick Test
```bash
python scripts/5_training/train.py --config configs/gpu/tiny.yaml
# Runs on small model, takes ~1-2 hours
```

### Example 2: Production Training
```bash
python scripts/5_training/train.py --config configs/gpu/small.yaml
# Full training, ~24-48 hours on RTX 3090 Ti
```

### Example 3: Custom Setup
```bash
# Edit config first
sed -i 's/learning_rate: 0.006/learning_rate: 0.003/' configs/gpu/small.yaml
sed -i 's/batch_size: 8/batch_size: 16/' configs/gpu/small.yaml

# Then train
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

### Example 4: Resume After Interruption
```bash
# Find latest checkpoint
ls -lt outputs/runs/*/checkpoints/ | head -1

# Resume training
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --resume outputs/runs/run_20251021_123456_abc123/checkpoints/checkpoint_step_50000
```

---

**Status**: ✅ Complete
**Last Updated**: 2025-10-21
