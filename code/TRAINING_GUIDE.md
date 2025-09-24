# Ava MoE++ Training Guide

## Quick Start

Training now uses **streaming by default** for memory efficiency and supports **multi-column data loading**:

```bash
# Default: streaming enabled (memory efficient)
python3 scripts/training/train.py --config configs/cpu/small.yaml

# With multi-column data loading
python3 scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --use-multi-column \
    --column-names text \
    --column-types text \
    --column-roles input

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

## Multi-Column Data Loading

### 📊 Multi-Column Support
The framework now supports training on datasets with multiple columns and mixed data types:

- **Text columns**: Natural language with configurable tokenization
- **Numeric columns**: Numerical data with normalization
- **Categorical columns**: Categories with vocabulary mapping
- **Image columns**: Image data with preprocessing
- **Tensor columns**: Pre-computed tensors

### Basic Multi-Column Training
```bash
# Simple text column
python3 scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --use-multi-column \
    --column-names text \
    --column-types text \
    --column-roles input \
    --data-dir /project/code/processed
```

### Advanced Multi-Column Examples

#### Instruction-Response Training
```bash
# Template-based instruction tuning
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names instruction,response \
    --column-types text,text \
    --column-roles input,target \
    --combine-strategy template \
    --column-template "Instruction: {instruction}\nResponse: {response}"
```

#### Multi-Input Training
```bash
# Multiple input types (text + categorical + numeric)
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names text_input,category,difficulty \
    --column-types text,categorical,numeric \
    --column-roles input,auxiliary,auxiliary \
    --combine-strategy separate
```

#### Configuration File Approach
```bash
# Use predefined configuration
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --dataset-config configs/multi_column_tests/instruction_response.yaml
```

### HuggingFace Dataset Integration
```bash
# Load directly from HuggingFace Hub
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --hf-dataset squad \
    --column-names context,question,answer \
    --column-types text,text,text \
    --column-roles input,input,target
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

### New Data Output Location
Processed data is now saved to `/project/code/processed/` for better organization.

### Using Existing Processed Files
The training scripts automatically detect and use processed files in:
- `/project/code/processed/*.jsonl` (new default location)
- `/project/code/data/pretraining/processed/train_*/`
- `/project/code/data/pretraining/processed/val_*/`

### Preparing New Data
```bash
# Enhanced multi-column data preparation
python3 scripts/data_prep/prepare_data_rapids.py \
    --raw-data-dir data \
    --output-dir /project/code/processed \
    --max-samples 50000 \
    --format-strategy multi_column

# Fast parallel processing (legacy)
python3 scripts/data_prep/prepare_data_fast.py \
    --input-path your_data.jsonl \
    --output-dir /project/code/processed \
    --num-workers 8

# Simple processing (legacy)
python3 scripts/data_prep/prepare_data_simple.py \
    --input-path your_data.jsonl \
    --output-dir /project/code/processed
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

Multi-column specific options:
- `--use-multi-column`: Enable multi-column data loading
- `--column-names`: Comma-separated list of column names
- `--column-types`: Comma-separated list of column types (text,numeric,categorical,image,tensor)
- `--column-roles`: Comma-separated list of column roles (input,target,auxiliary,weight)
- `--combine-strategy`: How to combine columns (concatenate,template,separate)
- `--column-template`: Template string for template strategy
- `--dataset-config`: Path to multi-column configuration file
- `--hf-dataset`: HuggingFace dataset name for direct loading
- `--data-dir`: Directory containing processed data files