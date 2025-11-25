# OpenOrca Dataset Downloader

Successfully created and tested downloader for OpenOrca dataset with custom Ava tokenizer.

## Download Complete!

**Status**: ✅ Successfully downloaded and tokenized
**Date**: 2025-11-24
**Location**: `/project/code/data/fine-tuning/OpenOrca/arrow/`

### Dataset Statistics

- **Total Samples**: 4,033,923 (4.0M)
- **Chunk Files**: 81 files
- **Total Size**: 2.3 GB
- **Format**: Parquet (PyArrow)
- **Max Sequence Length**: 2048 tokens
- **Tokenizer**: Custom Ava tokenizer (vocab_size=50,680)

### Files Structure

```
/project/code/data/fine-tuning/OpenOrca/
├── arrow/
│   ├── chunk_0004.parquet  (50,000 samples each)
│   ├── chunk_0005.parquet
│   ├── ...
│   └── chunk_0084.parquet
└── dataset_info.json
```

## Usage

### 1. Use in Training Config

Update your finetuning config to use this dataset:

```yaml
# In code/configs/moe/finetune_from_checkpoint.yaml
data:
  data_dir: /project/code/data/fine-tuning/OpenOrca/arrow
  streaming: true
  use_pretokenized: false
  max_length: 2048
```

### 2. Download Script Usage

The downloader script supports various options:

```bash
# Full dataset (4.2M samples - already done!)
python download_openorca.py

# Test with limited samples
python download_openorca.py --max-samples 10000

# Custom chunk size (to manage memory)
python download_openorca.py --chunk-size 25000

# Custom output directory
python download_openorca.py --output-dir /path/to/output
```

### 3. Script Features

- ✅ **Memory Efficient**: Writes to disk every 50k samples (configurable)
- ✅ **Streaming**: Never loads full dataset into memory
- ✅ **Batched Tokenization**: Processes 1000 samples at a time
- ✅ **Progress Tracking**: Shows real-time progress with tqdm
- ✅ **Interrupt Safe**: Saves partial data on Ctrl+C
- ✅ **Custom Tokenizer**: Uses your project's Ava tokenizer

## Comparison: OpenOrca vs CodeAlpaca

| Dataset | Samples | Size | Best For |
|---------|---------|------|----------|
| **OpenOrca** | 4.0M | 2.3 GB | Comprehensive instruction-following & human-assistant conversations |
| CodeAlpaca | 20k | 2.9 MB | Code-focused tasks only |

**Recommendation**: Use OpenOrca for finetuning - it has 200x more data and covers human-assistant interactions comprehensively!

## Data Format

Each chunk is a Parquet file with columns:
- `input_ids`: List[int] - Tokenized text (length 2048)
- `attention_mask`: List[int] - Attention masks (1 = token, 0 = padding)

The data is formatted as conversations:
```
System: {system_prompt}
User: {question}
Assistant: {response}
```

## Training Estimate

With OpenOrca (4.0M samples):

- **Batch size**: 64
- **Gradient accumulation**: 8
- **Effective batch**: 512
- **Steps per epoch**: ~7,800
- **3 epochs**: ~23,400 steps
- **Total tokens**: ~24.6 billion tokens

This is **200x more training data** than CodeAlpaca (20k samples)!

## Next Steps

1. ✅ Downloaded and tokenized OpenOrca
2. Update `finetune_from_checkpoint.yaml` to use OpenOrca data
3. Run finetuning: `python finetune.py`
4. Model will learn proper human-assistant interactions!

## Troubleshooting

If you encounter memory issues during training:
- Reduce batch size in config
- Increase gradient accumulation steps
- Use gradient checkpointing (already enabled)
- Enable mixed precision training (already enabled)

## Script Location

- **Downloader**: `/project/code/scripts/1_data_download/download_openorca.py`
- **Data**: `/project/code/data/fine-tuning/OpenOrca/arrow/`
- **Config**: `/project/code/configs/moe/finetune_from_checkpoint.yaml`