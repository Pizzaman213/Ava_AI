# Getting Started with Ava

This guide walks you through setting up and running your first training job with Ava.

## Prerequisites

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB+ recommended)
- **CUDA**: 11.8+

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Ava_AI
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: DeepSpeed for distributed training
pip install deepspeed

# Optional: Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

### 3. Verify Installation

```bash
python code/scripts/check_dependencies.py
```

## Project Structure

```
/root/Ava_AI/
├── code/
│   ├── src/ava/           # Core framework
│   ├── scripts/           # Training scripts
│   ├── configs/           # Configuration files
│   └── outputs/           # Training outputs
├── data/                  # Training data
└── models/                # Model checkpoints
```

## Your First Training Run

### Step 1: Prepare Data

Download and preprocess training data:

```bash
python code/scripts/1_data_download/unified_download.py
```

This downloads datasets and creates pre-tokenized Arrow files in `code/data/`.

### Step 2: Choose a Configuration

| Config | Parameters | GPU Memory | Use Case |
|--------|------------|------------|----------|
| `minimal_working.yaml` | 62M | 8GB | Testing/Development |
| `large.yaml` | 200M+ | 24GB | Production |
| `large_Multy.yaml` | 200M+ | 24GB x N | Multi-GPU |

### Step 3: Start Training

```bash
# Development/testing (smaller model)
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/minimal_working.yaml

# Production training
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml
```

### Step 4: Monitor Progress

Training logs appear in the terminal. For detailed monitoring:

```bash
# Enable Weights & Biases
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --use-wandb
```

View runs at [wandb.ai](https://wandb.ai).

## Understanding the Output

Training creates a run directory:

```
code/outputs/pretraining/ava_training_YYYYMMDD_HHMMSS_ID/
├── checkpoints/           # Model checkpoints
│   ├── checkpoint_step_1000.pt
│   └── best_model.pt
├── logs/                  # Training logs
├── metrics/               # JSON metrics
└── config.yaml            # Run configuration
```

## Key Concepts

### Mixture of Experts (MoE)

Ava uses sparse MoE architecture where each token is processed by only `k` experts out of `N` total:

```yaml
model:
  num_experts: 8          # Total experts
  num_experts_per_token: 2  # Active per token
  router_type: 'mixtral'  # Routing algorithm
```

### Training Pipeline

The pipeline manages components via lifecycle hooks:

```
Initialize → Epoch Loop → Step Loop → Cleanup
                ↓
    [DataLoader → Forward → Loss → Backward → Optimizer]
```

### Gradient Checkpointing

Reduces memory by recomputing activations during backward pass:

```yaml
model:
  gradient_checkpointing: true
```

## Next Steps

- [Architecture Guide](./02_ARCHITECTURE.md) - Understand the model
- [Training Guide](./03_TRAINING_GUIDE.md) - Advanced training options
- [Configuration Reference](./04_CONFIGURATION.md) - All parameters
- [Troubleshooting](./09_TROUBLESHOOTING.md) - Common issues

## Quick Reference

```bash
# Training commands
python code/scripts/5_training/train_pipeline.py --config <config.yaml>
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config <config.yaml>

# Fine-tuning
python code/scripts/5_training/finetune.py --checkpoint <path>

# Generation
python code/scripts/7_generation/generate.py --prompt "Once upon a time"

# RLHF
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config <config.yaml>
```
