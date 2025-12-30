# Pre-Tokenization Guide

The improved pre-tokenizer automatically detects and tokenizes all datasets in your data folder.

## Quick Start

### Tokenize ALL datasets at once:
```bash
python code/scripts/1_data_download/pretokenize_datasets.py
```

This will:
- Find all datasets in `/root/Ava_AI/code/data/`
- Automatically detect format (parquet, json, jsonl, csv, arrow)
- Tokenize with 16k tokenizer
- Save as `-tokenized` folders

### Tokenize a single dataset:
```bash
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path /root/Ava_AI/code/data/c4 \
  --output-path /root/Ava_AI/code/data/c4-tokenized
```

## Features

✓ **Auto-detection**
  - Automatically finds all datasets
  - Detects file formats (parquet, json, jsonl, csv, arrow)
  - Intelligent text column detection

✓ **Smart text column detection**
  - Looks for: text, content, data, input, instruction, output, completion
  - Falls back to first string column if not found
  - Handles multi-column datasets

✓ **Batch processing**
  - 32 examples per batch (memory efficient)
  - Progress tracking
  - Recovers from errors

✓ **Format support**
  - Parquet (Hugging Face datasets)
  - JSON / JSONL (streaming datasets)
  - CSV (tabular data)
  - Arrow (Apache Arrow format)

## Output

Each tokenized dataset gets saved in Arrow format:
```
/root/Ava_AI/code/data/
├── c4/                      # Original
├── c4-tokenized/            # Tokenized version
│   ├── dataset_info.json
│   ├── state.json
│   └── data/
├── TinyStories/             # Original
├── TinyStories-tokenized/   # Tokenized version
├── OpenOrca/                # Original
└── OpenOrca-tokenized/      # Tokenized version
```

## Examples

### Example 1: Auto-tokenize everything
```bash
python code/scripts/1_data_download/pretokenize_datasets.py

# Output:
# ======================================================================
#  AUTO-TOKENIZATION MODE
# ======================================================================
# Found 3 dataset(s) to tokenize:
#
#   • c4
#   • TinyStories
#   • OpenOrca
#
# [Tokenizes each one...]
#
# ======================================================================
#  SUMMARY
# ======================================================================
#
#   ✓ c4: 1,234,567 examples
#   ✓ TinyStories: 50,000 examples
#   ✓ OpenOrca: 100,000 examples
#
# Successfully tokenized: 3/3
# Total examples: 1,384,567
```

### Example 2: Tokenize just C4
```bash
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path /root/Ava_AI/code/data/c4 \
  --output-path /root/Ava_AI/code/data/c4-tokenized
```

### Example 3: Use custom tokenizer
```bash
python code/scripts/1_data_download/pretokenize_datasets.py \
  --tokenizer-path /root/Ava_AI/code/data/Ava_Ai/tokenizer_v3
```

## What Gets Tokenized

The script looks for these directories automatically:
- ✓ `c4/` → creates `c4-tokenized/`
- ✓ `TinyStories/` → creates `TinyStories-tokenized/`
- ✓ `OpenOrca/` → creates `OpenOrca-tokenized/`
- ✓ `tinystories_16k/` → creates `tinystories_16k-tokenized/`
- ✓ `tinystories_mixed/` → creates `tinystories_mixed-tokenized/`

It skips:
- ✗ `Ava_Ai/` (tokenizer, not data)
- ✗ `models/` (checkpoints)
- ✗ `-tokenized` folders (already done)
- ✗ `.cache/` directories

## Format Detection

The script automatically detects:

**Parquet files** (Hugging Face datasets)
```
c4/
├── en.noblocklist/
│   ├── c4-train.00000.json.gz
│   ├── c4-train.00001.json.gz
│   └── ...
```

**JSON files**
```
OpenOrca/
├── data-00000-of-00010.json
├── data-00001-of-00010.json
└── ...
```

**JSONL files** (line-delimited JSON)
```
dataset/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

**Arrow datasets**
```
dataset/
├── dataset_info.json
└── data/
    ├── data-00000-of-00010.arrow
    └── ...
```

## Performance

Tokenization speed depends on:
- Dataset size
- System RAM
- Batch size (default: 32)
- Tokenizer vocabulary (16k = faster)

Typical speeds:
- **TinyStories** (50K examples): ~2-5 min
- **OpenOrca** (1M examples): ~30-60 min
- **C4** (10M+ examples): ~2-6 hours

## Troubleshooting

### "No datasets found"
Make sure datasets are in `/root/Ava_AI/code/data/` with proper structure.

### "Could not load dataset"
- Check format is supported (parquet, json, jsonl, csv, arrow)
- Ensure files exist in the directory
- Try single dataset mode with explicit paths

### Out of memory
- Use `--dataset-path` to tokenize one at a time
- Reduce batch size in the script (line 156)
- Delete tokenized datasets after using them

### Slow tokenization
- Normal for large datasets (C4 is 64GB+)
- Run multiple tokenizations in parallel if disk space permits
- Monitor with `du -sh` in another terminal

## Using Tokenized Data

Once tokenized, configure your training script:

```yaml
# code/configs/moe/large.yaml
data:
  use_pretokenized: true
  pretokenized_paths:
    - /root/Ava_AI/code/data/c4-tokenized
    - /root/Ava_AI/code/data/TinyStories-tokenized
    - /root/Ava_AI/code/data/OpenOrca-tokenized
```

Then start training:
```bash
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml
```

## Tips

1. **Start with TinyStories** - it's fast to tokenize and good for testing
2. **Tokenize in background** - use `nohup` or `screen` for long runs
3. **Monitor disk space** - tokenized data takes similar space to original
4. **Use multiple datasets** - mix formats and sizes for better training

## Advanced: Custom Text Columns

If the script doesn't detect your text columns, edit the detection logic:

[pretokenize_datasets.py:71](code/scripts/1_data_download/pretokenize_datasets.py#L71)

Add your column names to the list:
```python
if col in ["text", "content", "data", "input", "instruction", "output", "completion", "your_column"]:
```
