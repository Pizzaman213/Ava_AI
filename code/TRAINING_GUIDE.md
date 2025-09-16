# Ava MoE++ Training Guide

## Quick Start

Training now uses **streaming by default** for memory efficiency:

```bash
# Default: streaming enabled (memory efficient)
python3 scripts/training/train.py --config configs/cpu/small.yaml

# Explicitly disable streaming (loads all data into memory)
python3 scripts/training/train.py --config configs/cpu/small.yaml --no-streaming
```

## Data Loading Modes

### 🌊 Streaming Mode (Default)
- **Memory Efficient**: Constant memory usage regardless of dataset size
- **Handles Large Datasets**: Can train on datasets larger than RAM
- **Immediate Start**: Begins training without loading delay
- **Automatic Shuffling**: Maintains randomness via buffer

```bash
# Streaming with custom buffer size
python3 scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --buffer-size 2000  # Larger buffer for better shuffling
```

### 💾 In-Memory Mode
- **Faster Iteration**: All data in RAM
- **Full Shuffling**: Complete dataset randomization
- **Best for Small Datasets**: When dataset fits in memory

```bash
# Load all data into memory
python3 scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --no-streaming
```

## Configuration Examples

### Ultra-Tiny (Testing)
```bash
# Quick test with minimal resources
python3 scripts/training/train.py \
    --config configs/cpu/ultra_tiny.yaml \
    --max-samples 100 \
    --batch-size 2 \
    --epochs 1
```

### Small (Development)
```bash
# Development with reasonable performance
python3 scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --batch-size 4 \
    --epochs 10
```

### Medium (Training)
```bash
# Full training with medium model
python3 scripts/training/train.py \
    --config configs/cpu/medium.yaml \
    --batch-size 8 \
    --epochs 20
```

### GPU Training
```bash
# GPU with large batch size
python3 scripts/training/train.py \
    --config configs/gpu/base.yaml \
    --batch-size 32 \
    --epochs 50
```

## Data Preparation

### Using Existing Arrow Files
The training scripts automatically detect and use Arrow files in:
- `/project/code/data/pretraining/processed/train_*/`
- `/project/code/data/pretraining/processed/val_*/`

### Preparing New Data
```bash
# Fast parallel processing
python3 scripts/data_prep/prepare_data_fast.py \
    --input-path your_data.jsonl \
    --output-dir data/pretraining/processed \
    --num-workers 8

# Simple processing
python3 scripts/data_prep/prepare_data_simple.py \
    --input-path your_data.jsonl \
    --output-dir data/pretraining/processed
```

## Memory Management

### For Limited Memory Systems
```bash
# Minimal memory usage
python3 scripts/training/train.py \
    --config configs/cpu/ultra_tiny.yaml \
    --buffer-size 100 \
    --batch-size 1 \
    --max-samples 1000
```

### For High Memory Systems
```bash
# Maximum performance with available memory
python3 scripts/training/train.py \
    --config configs/cpu/medium.yaml \
    --no-streaming \
    --batch-size 16
```

## Monitoring Training

Training outputs are saved to `/project/code/outputs/`:
- Logs: `training_YYYYMMDD_HHMMSS.log`
- Checkpoints: `checkpoint_epoch_N.pt`
- Metrics: Automatically logged

View training progress:
```bash
tail -f outputs/training_*.log
```

## Common Issues

### Out of Memory
- Use streaming mode (default)
- Reduce batch size
- Use smaller model config
- Reduce buffer size

### Slow Training
- Increase batch size if memory allows
- Use GPU configuration
- Reduce max sequence length
- Use --no-streaming for small datasets

### Data Not Found
- Check data exists in `/project/code/data/pretraining/processed/`
- Ensure Arrow/Parquet files are present
- Run data preparation scripts if needed

## Advanced Options

```bash
python3 scripts/training/train.py --help
```

Key options:
- `--streaming/--no-streaming`: Control data loading mode
- `--buffer-size`: Streaming buffer size (default: 1000)
- `--max-samples`: Limit training samples (for testing)
- `--batch-size`: Override config batch size
- `--epochs`: Override config epochs
- `--learning-rate`: Override config learning rate
- `--resume`: Resume from checkpoint