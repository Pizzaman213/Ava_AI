# Complete Training Implementation Guide
## 100M Parameter Model Training with Full Features

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Features Overview](#features-overview)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Advanced Usage](#advanced-usage)
6. [Distributed Training](#distributed-training)
7. [Customization Guide](#customization-guide)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation & Setup
```bash
# Navigate to project
cd /project

# Ensure dependencies are installed
pip install torch torchvision torchaudio -U
pip install tensorboard tqdm pyyaml transformers

# Make script executable
chmod +x code/scripts/5_training/train_100m_full.py
```

### Basic Training Command
```bash
# Single GPU, default settings (3 epochs, batch=8)
python code/scripts/5_training/train_100m_full.py

# With config file
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml

# Custom parameters
python code/scripts/5_training/train_100m_full.py \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 5e-5
```

### Output Structure
```
./logs/                          # Tensorboard logs and metrics
├── training_YYYYMMDD_HHMMSS.log
├── metrics_summary.json
└── events.out.tfevents.*

./checkpoints/                   # Model checkpoints
├── checkpoint_epoch_0_step_0.pt
├── checkpoint_epoch_1_step_0.pt
└── checkpoint_epoch_2_step_0.pt
```

---

## Features Overview

### ✅ Implemented Features

#### 1. **Model Architecture**
- 100M parameter Transformer
- 12 layers, 768 hidden size, 12 attention heads
- Configurable via YAML
- Supports custom model substitution

#### 2. **Data Pipeline**
- Dummy dataset for demonstration
- Support for custom datasets
- Data loading with multiple workers
- Pin memory for GPU efficiency
- Distributed data loading

#### 3. **Training Loop**
- Epoch-based training
- Batch-level metrics tracking
- Loss computation and backprop
- Gradient accumulation support
- Progress bars with tqdm

#### 4. **Optimization**
- AdamW optimizer
- Learning rate scheduling (warmup + cosine annealing)
- Gradient clipping (max_grad_norm=1.0)
- Mixed precision training (bfloat16)

#### 5. **Logging & Metrics**
- Console logging with timestamps
- File logging to log directory
- TensorBoard integration
- Metrics JSON summary
- Validation metrics tracking

#### 6. **Checkpointing**
- Save model and optimizer state
- Automatic checkpoint management (keep last 3)
- Resume from checkpoint
- Epoch and step tracking

#### 7. **Distributed Training**
- Single-node multi-GPU (DDP)
- Multi-node support (with torchrun)
- Proper rank-aware logging
- Synchronized operations

#### 8. **Validation**
- Validation loop
- Validation interval control
- Best checkpoint saving
- Validation loss tracking

---

## Architecture

### Code Structure

```python
# 1. LOGGING (Lines 62-89)
setup_logging()                    # Setup loggers with rank awareness

# 2. DATA LOADING (Lines 92-154)
DummyDataset                       # Implement your Dataset here
create_dataloaders()               # Create train/val loaders

# 3. MODEL (Lines 157-203)
TransformerModel100M               # Replace with your model

# 4. CHECKPOINTING (Lines 206-251)
CheckpointManager                  # Save/load model states

# 5. METRICS (Lines 254-289)
MetricsTracker                     # Track and log metrics

# 6. DISTRIBUTED (Lines 292-314)
setup_distributed()                # Initialize DDP
cleanup_distributed()              # Cleanup after training

# 7. TRAINING (Lines 317-452)
train_epoch()                      # Training loop
validate()                         # Validation loop

# 8. MAIN (Lines 455-623)
main()                             # Orchestration

# 9. CLI (Lines 626+)
argparse setup                     # Command-line interface
```

### Component Relationships

```
main()
├── setup_distributed()           # Init DDP if multi-GPU
├── setup_logging()               # Setup loggers
├── Load config (YAML)            # Load hyperparameters
├── Create model                  # Initialize TransformerModel100M
├── Create optimizer              # AdamW
├── Create dataloaders            # train_loader, val_loader
├── Create scheduler              # Warmup + Cosine Annealing
├── CheckpointManager             # Checkpoint operations
├── MetricsTracker                # Metrics tracking
└── Training loop
    ├── train_epoch()             # Per-epoch training
    │   ├── Forward pass
    │   ├── Loss computation
    │   ├── Backward pass
    │   ├── Gradient accumulation
    │   └── Metrics update
    ├── validate()                # Validation
    └── Checkpointing
```

---

## Configuration

### YAML Config Format

```yaml
# Model configuration
model:
  vocab_size: 50680
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12

# Training configuration
training:
  batch_size: 8
  learning_rate: 5e-5
  num_epochs: 3
  gradient_accumulation_steps: 1
  warmup_steps: 1000
```

### CLI Arguments Override Config

```bash
# Config file values
python train_100m_full.py --config config.yaml

# CLI overrides config file
python train_100m_full.py \
    --config config.yaml \
    --batch-size 32 \              # Override batch_size from config
    --learning-rate 1e-4 \         # Override learning_rate from config
    --epochs 10                     # Override num_epochs from config
```

### Environment Variables

```bash
# Distributed training setup
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# PyTorch settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Advanced Usage

### 1. Resuming from Checkpoint

```bash
# Continue training from epoch 5
python train_100m_full.py \
    --config config.yaml \
    --resume checkpoints/checkpoint_epoch_5_step_0.pt \
    --epochs 10                     # Will train epochs 6-10
```

### 2. Validation Every N Epochs

```bash
# Only validate every 2 epochs (reduces overhead)
python train_100m_full.py \
    --val-interval 2
```

### 3. Custom Log Interval

```bash
# Log every 50 batches instead of 10
python train_100m_full.py \
    --log-interval 50
```

### 4. Monitoring with TensorBoard

```bash
# Terminal 1: Run training
python train_100m_full.py --config config.yaml

# Terminal 2: Start TensorBoard
tensorboard --logdir ./logs --port 6006

# Open browser to http://localhost:6006
```

### 5. Distributed Training (Multi-GPU)

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml

# 8 GPUs across 2 nodes
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml
```

---

## Customization Guide

### 1. Custom Dataset

**Replace `DummyDataset` in the script:**

```python
class MyCustomDataset(Dataset):
    """Your custom dataset."""

    def __init__(self, data_path: str, tokenizer, seq_length: int = 512):
        self.data = load_data(data_path)  # Implement
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'labels': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
        }

# Update create_dataloaders():
def create_dataloaders(...):
    train_dataset = MyCustomDataset(
        data_path='path/to/train_data',
        tokenizer=tokenizer,
        seq_length=512
    )
    # ... rest of code
```

### 2. Custom Model Architecture

**Replace `TransformerModel100M`:**

```python
class MyCustomModel(nn.Module):
    """Your custom model."""

    def __init__(self, vocab_size: int, hidden_size: int, **kwargs):
        super().__init__()
        # Your architecture here

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Your forward pass
        logits = ...  # Your model output

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# Update main():
model = MyCustomModel(vocab_size, hidden_size, ...)
```

### 3. Custom Learning Rate Scheduler

```python
# Replace scheduler creation in main():
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(step):
    """Custom learning rate schedule."""
    if step < 1000:
        return step / 1000  # Linear warmup
    else:
        return max(0.1, (10000 - step) / 10000)  # Linear decay

scheduler = LambdaLR(optimizer, lr_lambda)
```

### 4. Custom Metrics Tracking

```python
# Add to train_epoch():
if metrics_tracker is not None:
    metrics_tracker.update(
        global_step,
        loss=loss_value,
        lr=optimizer.param_groups[0]['lr'],
        accuracy=accuracy,  # Add custom metric
        perplexity=perplexity,  # Add custom metric
    )
```

### 5. Integrate with Weights & Biases

```bash
# Install wandb
pip install wandb

# In main():
import wandb

wandb.init(
    project="llm-training",
    config={
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
)

# In train_epoch(), after metrics update:
wandb.log({
    'train_loss': loss_value,
    'learning_rate': optimizer.param_groups[0]['lr'],
})

# At end:
wandb.finish()
```

### 6. Integrate with DeepSpeed

```bash
# Install deepspeed
pip install deepspeed

# Create ds_config.json:
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": "dynamic"
    },
    "zero_optimization": {
        "stage": 2
    }
}

# In main():
import deepspeed

model, optimizer, train_loader, scheduler = deepspeed.initialize(
    args=None,
    model=model,
    optimizer=optimizer,
    config='ds_config.json'
)

# In train_epoch():
# Use model.backward(loss) instead of loss.backward()
```

---

## Distributed Training

### Single-Node Multi-GPU (Recommended for single machine)

```bash
# 4 GPUs on single machine
torchrun --nproc_per_node=4 train_100m_full.py

# Explicitly specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train_100m_full.py
```

### Multi-Node Distributed

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_100m_full.py

# Node 1 (worker)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_100m_full.py
```

### Debugging Distributed Training

```bash
# Check DDP initialization
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun --nproc_per_node=2 train_100m_full.py

# Use nccl backend debugging
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=4 train_100m_full.py
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:** `CUDA out of memory` error

**Solutions:**
```bash
# Reduce batch size
python train_100m_full.py --batch-size 4

# Increase gradient accumulation
# (in config.yaml) gradient_accumulation_steps: 8

# Enable pytorch memory efficient mode
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 2. Slow Training on Multi-GPU

**Symptoms:** Multi-GPU training is slower than single GPU

**Solutions:**
```bash
# Check for bottlenecks
# 1. Reduce number of data workers
# 2. Increase batch size per GPU
# 3. Check GPU utilization with nvidia-smi

nvidia-smi -l 1  # Monitor GPU usage every 1 second
```

#### 3. DDP Hanging/Deadlock

**Symptoms:** Training hangs at initialization or during training

**Solutions:**
```bash
# Check network connectivity
ping other_node

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Use DEBUG mode
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 train_100m_full.py
```

#### 4. Checkpoint Resume Issues

**Symptoms:** Error when resuming from checkpoint

**Solutions:**
```bash
# Verify checkpoint file exists
ls -lh checkpoints/

# Check checkpoint compatibility
python -c "
import torch
ckpt = torch.load('checkpoint.pt')
print('Keys:', ckpt.keys())
"
```

### Performance Tuning

#### 1. Optimize Data Loading

```python
# In create_dataloaders():
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,           # Increase if CPU has spare cores
    pin_memory=True,         # Always use on GPU training
    prefetch_factor=2,       # Prefetch batches
    persistent_workers=True, # Keep workers alive between epochs
)
```

#### 2. Optimize Mixed Precision

```python
# In train_epoch():
with torch.autocast(
    device_type='cuda',
    dtype=torch.bfloat16,  # Or torch.float16
    enabled=True
):
    outputs = model(...)
```

#### 3. Enable Gradient Checkpointing

```python
# In TransformerModel100M:
def forward(self, ...):
    x = self.embedding(input_ids)
    for layer in self.layers:
        x = torch.utils.checkpoint.checkpoint(
            layer,
            x,
            use_reentrant=False  # For mixed precision
        )
```

---

## Summary

This training script provides a complete, production-ready training pipeline with:

✅ **Basics:** Model creation, training loop, validation
✅ **Advanced:** Checkpointing, distributed training, metrics tracking
✅ **Customizable:** Easy to swap models, datasets, optimizers
✅ **Professional:** Logging, error handling, CLI interface
✅ **Scalable:** Single GPU → Multi-node training

**Start with:** `python train_100m_full.py --config config.yaml`
**Then customize:** Follow the customization guide for your specific needs

