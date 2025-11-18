# Data Preparation - Quick Start

## TL;DR

```bash
# One command does everything!
python code/scripts/prepare_data.py --config code/configs/moe/tiny_moe.yaml
```

That's it. The script will:
- ✓ Check if raw data exists
- ✓ Download if needed (with your approval)
- ✓ Pretokenize all files
- ✓ Report what was created

## Common Commands

### Option 1: Full Workflow with Config (Recommended)

```bash
python code/scripts/prepare_data.py --config code/configs/moe/tiny_moe.yaml
```

**What it does:**
- Checks for raw JSONL files in `/project/code/data/processed`
- Offers to download if missing
- Pretokenizes to `/project/code/data/pretokenized`
- Reads settings from config (tokenizer, max_length)

### Option 2: Just Pretokenize

```bash
python code/scripts/2_data_prep/pretokenize_dataset.py --config code/configs/moe/tiny_moe.yaml
```

**What it does:**
- Auto-finds JSONL files in standard location
- Reads tokenizer & max_length from config
- Creates Arrow files automatically

### Option 3: Auto-Detect Everything

```bash
python code/scripts/2_data_prep/pretokenize_dataset.py
```

**What it does:**
- Finds raw data in `/project/code/data/processed`
- Uses default tokenizer
- Outputs to `/project/code/data/pretokenized`

### Option 4: Preview Before Running

```bash
python code/scripts/prepare_data.py --config code/configs/moe/tiny_moe.yaml --dry-run
```

**What it does:**
- Shows what WOULD happen
- No files created
- Safe to check before committing

## What Gets Created

```
/project/code/data/pretokenized/
├── file1.arrow          # Ready for training!
├── file2.arrow
└── ...
```

Each `.arrow` file is optimized for fast loading during training.

## Need Help?

### "I don't have raw data"

```bash
python code/scripts/1_data_download/unified_download.py --help
```

This downloads datasets to `/project/code/data/processed`

### "I want to use different settings"

Override with command-line flags:

```bash
python code/scripts/2_data_prep/pretokenize_dataset.py \
  --config code/configs/moe/tiny_moe.yaml \
  --max-length 512 \
  --num-workers 8
```

### "I want to re-process existing files"

```bash
python code/scripts/prepare_data.py --force
```

### "I want to see what would happen first"

```bash
python code/scripts/prepare_data.py --dry-run
```

## File Locations

| What | Where | What Happens |
|------|-------|--------------|
| Config | `code/configs/moe/*.yaml` | Read settings from here |
| Raw data | `code/data/processed/*.jsonl` | Input files |
| Pretokenized | `code/data/pretokenized/*.arrow` | Output files (auto-created) |
| Tokenizer | `code/models/tokenizer/enhanced-50680` | Used for tokenization |

## Config Example

Your config file should have:

```yaml
data:
  tokenizer_name: /project/code/models/tokenizer/enhanced-50680
  max_length: 256
```

The scripts use these settings automatically.

## Success Indicators

✓ Command runs without errors
✓ Arrow files created in `/project/code/data/pretokenized/`
✓ Script says "Success!" at the end
✓ You can see file sizes with: `ls -lh /project/code/data/pretokenized/`

## Troubleshooting

**"No JSONL files found"**
→ Run download script first: `python code/scripts/1_data_download/unified_download.py`

**"Config file not found"**
→ Check path exists: `ls code/configs/moe/tiny_moe.yaml`

**"Tokenizer not found"**
→ The default tokenizer should be at: `/project/code/models/tokenizer/enhanced-50680`
→ Override with: `--tokenizer /path/to/your/tokenizer`

## Full Help

```bash
# Full options for prepare_data.py
python code/scripts/prepare_data.py --help

# Full options for pretokenize
python code/scripts/2_data_prep/pretokenize_dataset.py --help
```

## That's It!

Everything is designed to "just work" with sensible defaults. Most users will just run:

```bash
python code/scripts/prepare_data.py --config code/configs/moe/tiny_moe.yaml
```

And be done! ✨
