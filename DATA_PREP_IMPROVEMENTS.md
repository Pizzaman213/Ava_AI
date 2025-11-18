# Data Preparation Improvements

## Overview

The data preprocessing workflow has been significantly improved to be more user-friendly. Instead of manually specifying input/output paths, the scripts now auto-detect everything and integrate with the config system.

## Key Changes

### 1. Enhanced `pretokenize_dataset.py` - Auto-Detection Mode

**Before:** Required explicit input AND output paths
```bash
python pretokenize_dataset.py --input /path/to/input.jsonl --output /path/to/output.arrow
python pretokenize_dataset.py --input_dir /path/to/input --output_dir /path/to/output
```

**After:** Auto-detects everything with smart defaults
```bash
# Simple - auto-detect from standard locations
python pretokenize_dataset.py

# With config (recommended)
python pretokenize_dataset.py --config configs/moe/tiny_moe.yaml

# Preview mode (dry-run)
python pretokenize_dataset.py --dry-run

# Still supports explicit paths if needed
python pretokenize_dataset.py --input-dir /custom/path
```

#### Auto-Detection Priority

1. **Config-based (if `--config` provided):**
   - Reads `data.tokenizer_name` from config
   - Reads `data.max_length` from config
   - Infers input directory from `data.data_dir`

2. **Standard locations:**
   - Input: `/project/code/data/processed/` (auto-detected if files exist)
   - Output: `/project/code/data/pretokenized/` (auto-created)

3. **Command-line overrides:**
   - All flags optional with sensible defaults
   - Can override auto-detected values

#### New Features

- **`--config`**: Load tokenizer, max_length, and data locations from YAML config
- **`--dry-run`**: Preview what would be processed without creating files
- **`--force`**: Force re-tokenization of existing files
- **`--num-workers`**: Parallel processing (default: 4)
- All arguments optional - intelligently defaults to standard locations

#### Example Usage

```bash
# Recommended: Config-driven workflow
python pretokenize_dataset.py --config configs/moe/tiny_moe.yaml

# Auto-detect everything from standard locations
python pretokenize_dataset.py

# Override specific settings
python pretokenize_dataset.py --tokenizer /custom/tokenizer --max-length 512

# Preview what would happen
python pretokenize_dataset.py --dry-run
```

### 2. New `prepare_data.py` - Unified Workflow

**Purpose:** Single entry point for the complete data preparation pipeline

Handles:
1. Checking if raw data exists
2. Optionally downloading datasets (if needed)
3. Pretokenizing all JSONL files
4. Reporting statistics

```bash
# Complete workflow with config (recommended)
python prepare_data.py --config configs/moe/tiny_moe.yaml

# Preview mode
python prepare_data.py --config configs/moe/tiny_moe.yaml --dry-run

# Only pretokenize (skip download)
python prepare_data.py --no-download

# Only download (skip pretokenization)
python prepare_data.py --download-only

# Force re-download and re-tokenize
python prepare_data.py --force
```

#### Features

- **Auto-detection:** Reads input/output dirs from config or uses standard locations
- **Smart workflow:** Only downloads/pretokenizes as needed
- **Dry-run mode:** Preview what would happen
- **Final report:** Shows statistics about raw and pretokenized data
- **Helpful guidance:** Suggests next steps based on what's available

#### Example Output

```
======================================================================
Unified Data Preparation Pipeline
======================================================================

📂 Input directory:  /project/code/data/processed
📂 Output directory: /project/code/data/pretokenized

🔍 Checking data status...
   ✓ Raw data found in /project/code/data/processed
   ✗ No pretokenized data in /project/code/data/pretokenized

======================================================================
  Pretokenizing Datasets
======================================================================

🤖 Pretokenization script: /project/code/scripts/2_data_prep/pretokenize_dataset.py
📂 Input directory:  /project/code/data/processed
📂 Output directory: /project/code/data/pretokenized
👷 Workers: 4

Executing: python /project/code/scripts/2_data_prep/pretokenize_dataset.py ...
```

## Workflow Comparison

### Old Workflow
```
1. Manually download datasets
   python scripts/1_data_download/unified_download.py ...

2. Manually pretokenize
   python scripts/2_data_prep/pretokenize_dataset.py \
     --input_dir /project/code/data/processed \
     --output_dir /project/code/data/pretokenized

3. Hope directories were created, paths were correct
```

### New Workflow
```
# Option 1: Single command with config
python scripts/prepare_data.py --config configs/moe/tiny_moe.yaml

# Option 2: Just pretokenize (if data already exists)
python scripts/2_data_prep/pretokenize_dataset.py --config configs/moe/tiny_moe.yaml

# Everything auto-detected and created!
```

## Standard Directory Structure

The scripts expect (and auto-create) this structure:

```
/project/code/data/
├── processed/              # Raw JSONL files (from download script)
│   ├── dataset1.jsonl
│   ├── dataset2.jsonl
│   └── ...
└── pretokenized/           # Arrow files (output from pretokenize)
    ├── dataset1.arrow
    ├── dataset2.arrow
    └── ...

/project/code/models/tokenizer/
└── enhanced-50680/         # Default tokenizer
```

## Configuration Integration

Both scripts integrate with YAML configs:

```yaml
data:
  # Location of pretokenized data (for training)
  data_dir: /project/code/data/pretokenized

  # Tokenizer to use
  tokenizer_name: /project/code/models/tokenizer/enhanced-50680

  # Sequence length for tokenization
  max_length: 256
```

The scripts automatically read these settings instead of requiring command-line arguments.

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Required Args** | `--input_dir`, `--output_dir` | All optional |
| **Config Support** | None | Full integration with YAML configs |
| **Auto-creation** | Manual `mkdir` needed | Automatic |
| **Dry-run** | Not available | `--dry-run` for previewing |
| **Error Messages** | Generic | Helpful with suggestions |
| **Tokenizer** | CLI arg required | Auto from config or default |
| **Max length** | CLI arg required | Auto from config or default |
| **Entry point** | Two separate scripts | `prepare_data.py` unified workflow |

## Migration Guide

### If you have existing scripts:

**Old code:**
```bash
python pretokenize_dataset.py \
  --input_dir /project/code/data/processed \
  --output_dir /project/code/data/pretokenized \
  --tokenizer /project/code/models/tokenizer/enhanced-50680 \
  --max_length 2048
```

**New code:**
```bash
# Just specify config, everything else auto-detected
python pretokenize_dataset.py --config configs/moe/tiny_moe.yaml
```

### Config with data locations:

To have `pretokenize_dataset.py` auto-detect input from your config, add:

```yaml
data:
  data_dir: /project/code/data/processed  # For preprocessing
  tokenizer_name: /project/code/models/tokenizer/enhanced-50680
  max_length: 256
```

The script will:
- Read `tokenizer_name` and `max_length` from config
- Infer input from `data_dir` if it points to `processed`
- Auto-detect or create output directory

## Next Steps

1. Use `python prepare_data.py --config <your_config>` for complete data setup
2. Or use `python pretokenize_dataset.py --config <your_config>` for just tokenization
3. Check `--help` on both scripts for all options
4. Use `--dry-run` to preview without making changes

## Benefits

✓ **Less typing:** No need to specify paths when using configs
✓ **Fewer errors:** Automatic directory creation, smart defaults
✓ **Better integration:** Works seamlessly with config files
✓ **Batch processing:** Auto-finds all JSONL files and processes them
✓ **Safe:** `--dry-run` mode lets you preview before running
✓ **Smart errors:** Helpful messages suggest missing steps
✓ **Unified workflow:** Single entry point (`prepare_data.py`) handles everything
