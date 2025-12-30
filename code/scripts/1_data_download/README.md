# Data Download and Pre-tokenization

Complete workflow for downloading and preparing training data.

## Quick Start

```bash
# Download all default datasets (OpenOrca, C4, TinyStories)
python code/scripts/1_data_download/unified_download.py

# Download a specific dataset
python code/scripts/1_data_download/unified_download.py --dataset "nanoGPT/TinyStories"

# Pre-tokenize with 16k tokenizer
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path /root/Ava_AI/code/data/openorca-i \
  --output-path /root/Ava_AI/code/data/openorca-i-tokenized \
  --dataset-name openorca-i
```

## Datasets

| Dataset | Compressed | Uncompressed | Type | Use Case |
|---------|-----------|--------------|------|----------|
| **TinyStories** (roneneldan/TinyStories) | ~200MB | ~2GB | Stories | Quick testing ✓ |
| **OpenOrca** (Open-Orca/OpenOrca) | ~13GB | ~15GB | Instruction-following | Production |
| **C4** (allenai/c4) | 305GB+ | ~1.2TB+ | Web text | Large-scale only |

### Size Reference
```
TinyStories:    200MB  → Download in seconds, test in minutes
OpenOrca:       13GB   → Download in 10-30 min, full training in hours
C4:             305GB+ → Download in days, not recommended without partitioning
```

## Workflow

### Step 1: Download Datasets

**Recommended for most users:**
```bash
# TinyStories only (fastest, ~200MB, good for testing)
python code/scripts/1_data_download/unified_download.py --dataset "roneneldan/TinyStories"
```

**For production training:**
```bash
# OpenOrca (13GB, instruction-following data)
python code/scripts/1_data_download/unified_download.py --dataset "Open-Orca/OpenOrca"
```

**For large-scale training (requires significant storage/bandwidth):**
```bash
# C4 with partitions (limit to first 5 partitions = ~50GB)
python code/scripts/1_data_download/unified_download.py \
  --dataset "allenai/c4" \
  --max-partitions 5
```

**Download multiple datasets:**
```bash
python code/scripts/1_data_download/unified_download.py \
  --datasets roneneldan/TinyStories Open-Orca/OpenOrca
```

**Output:**
- `code/data/TinyStories/` - TinyStories dataset (~2GB uncompressed)
- `code/data/OpenOrca/` - OpenOrca dataset (~15GB uncompressed)
- `code/data/c4/` - C4 dataset (1.2TB+ uncompressed)

### Step 2: Pre-tokenize with 16k Tokenizer

```bash
# Pre-tokenize OpenOrca
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path code/data/openorca-i \
  --output-path code/data/openorca-i-tokenized \
  --dataset-name openorca-i

# Pre-tokenize C4
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path code/data/c4 \
  --output-path code/data/c4-tokenized \
  --dataset-name allenai/c4

# Pre-tokenize TinyStories
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path code/data/TinyStories \
  --output-path code/data/TinyStories-tokenized \
  --dataset-name nanoGPT/TinyStories
```

**Output:**
- `code/data/openorca-i-tokenized/` - Pre-tokenized Arrow format
- `code/data/c4-tokenized/` - Pre-tokenized Arrow format
- `code/data/TinyStories-tokenized/` - Pre-tokenized Arrow format

### Step 3: Configure Training

Update your config file to use pre-tokenized data:

```yaml
data:
  use_pretokenized: true
  pretokenized_paths:
    - /root/Ava_AI/code/data/openorca-i-tokenized
    - /root/Ava_AI/code/data/c4-tokenized
    - /root/Ava_AI/code/data/TinyStories-tokenized
```

## Features

### unified_download.py
- ✓ Download multiple datasets in parallel
- ✓ Resume incomplete downloads
- ✓ Skip already downloaded files
- ✓ Limit partitions for testing
- ✓ Progress tracking

**Options:**
```bash
--output-dir OUTPUT_DIR           # Default: /root/Ava_AI/code/data
--datasets DATASET [DATASET ...]  # Default: all three
--workers WORKERS                 # Default: 4 parallel workers
--max-partitions N                # Optional: limit partitions per dataset
--dataset SINGLE_DATASET          # Single dataset mode
```

### pretokenize_datasets.py
- ✓ Convert to pre-tokenized Arrow format
- ✓ Memory-efficient batch processing
- ✓ Support for OpenOrca, C4, TinyStories
- ✓ Custom tokenizer support
- ✓ Progress tracking

**Options:**
```bash
--dataset-path PATH       # Input dataset path (required)
--output-path PATH        # Output path (required)
--dataset-name NAME       # Dataset name (default: openorca-i)
--tokenizer-path PATH     # Tokenizer path (default: 16k tokenizer)
```

## Tokenizers Available

Located in `/root/Ava_AI/code/data/Ava_Ai/`:

- `tokenizer_16k/` - 16K vocabulary (recommended for most tasks)
- `tokenizer_v3/` - Latest tokenizer version
- `tokenizer_v2/` - Previous version
- `tokenizer_custom/` - Custom configurations

## Troubleshooting

### "Repository Not Found" Error
- Dataset doesn't exist on Hugging Face Hub
- Solution: Use available datasets (openorca-i, allenai/c4, nanoGPT/TinyStories)

### Slow Downloads
- Increase workers: `--workers 8`
- Limit partitions for testing: `--max-partitions 2`

### Authentication Errors
```bash
# Login to Hugging Face
huggingface-cli login

# Enter your token from https://huggingface.co/settings/tokens
```

### Out of Memory During Pre-tokenization
- Reduce batch size in pretokenize script
- Download smaller dataset first (TinyStories)

## Resume Downloads

All downloads are resumable - if interrupted, just run the command again:

```bash
python code/scripts/1_data_download/unified_download.py
# Will continue from where it left off
```

## Example: Full Setup

```bash
# 1. Download TinyStories (fastest for testing)
python code/scripts/1_data_download/unified_download.py --dataset "nanoGPT/TinyStories"

# 2. Pre-tokenize with 16k tokenizer
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path code/data/TinyStories \
  --output-path code/data/TinyStories-tokenized

# 3. Update config to use pre-tokenized data
# Edit code/configs/moe/large.yaml:
#   data:
#     use_pretokenized: true
#     pretokenized_paths:
#       - /root/Ava_AI/code/data/TinyStories-tokenized

# 4. Start training
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml
```

## Data Sizes (Uncompressed on Disk)

| Dataset | Compressed | Uncompressed | +Tokenization | Download Time |
|---------|-----------|--------------|---------------|----------------|
| **TinyStories** | 200MB | 2GB | 2.5GB | Seconds |
| **OpenOrca** | 13GB | 15GB | 18GB | 10-30 min |
| **C4 (1 partition)** | 10GB | 50GB | 60GB | 5-15 min |
| **C4 (5 partitions)** | 50GB | 250GB | 300GB | 30-90 min |
| **C4 (full)** | 305GB | 1.2TB+ | 1.4TB+ | **Days** |

### Recommendations

**Development / Testing:**
- Use **TinyStories** (~200MB download, ~2GB uncompressed)
- Gets you started in seconds, sufficient for model debugging

**Single GPU Training (24GB VRAM):**
- Use **TinyStories** + **OpenOrca** (~13GB)
- Total: ~15GB download, ~17GB uncompressed
- Provides ~3-5 hours of training

**Multi-GPU / Production (48GB+ VRAM):**
- Use **OpenOrca** (~13GB)
- Or C4 with **partitions 0-5** (~50GB download, ~250GB uncompressed)
- Provides ~20+ hours of training

**Large-scale / Research:**
- Full **C4** (305GB) - requires serious infrastructure

### Disk Space Required

```
Minimum (TinyStories only):     5GB
Small (TinyStories + OpenOrca): 20GB
Medium (OpenOrca):              30GB
Large (C4 - 5 partitions):      300GB
Very Large (Full C4):           1.5TB+
```
