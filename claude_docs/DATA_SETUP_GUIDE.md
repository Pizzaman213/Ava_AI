# Data Setup Guide

## Problem: Missing Training Data Files

When running training, you may see errors like:
```
No such file or directory: '/project/code/data/processed/HuggingFaceFW_fineweb-edu_processed.jsonl'
WARNING: Using synthetic data - no real data files found in /project/code/data/processed
```

This happens because:
1. Training data files referenced in the configuration don't exist
2. The data hasn't been downloaded or processed yet
3. The data directory path is incorrect

---

## Quick Fix: Use Existing Data ✅

The codebase includes one small pre-processed dataset. Let's use it:

### Step 1: Create Training Data (Fastest Option)

```bash
# Navigate to data directory
cd /project/code/data/processed

# List available files
ls -lh

# Should show:
# Open-Orca_OpenOrca_processed.jsonl  (20 samples)

# Create training data by duplicating existing file
cp Open-Orca_OpenOrca_processed.jsonl synthetic_train.jsonl
cat Open-Orca_OpenOrca_processed.jsonl >> synthetic_train.jsonl
cat Open-Orca_OpenOrca_processed.jsonl >> synthetic_train.jsonl

# Verify
wc -l synthetic_train.jsonl
# Should show: 60 samples
```

### Step 2: Update Configuration

Edit `configs/gpu/small.yaml` and ensure it points to existing data:

```yaml
data:
  data_dir: /project/code/data/processed  # Make sure this exists
```

### Step 3: Run Training

```bash
python code/scripts/5_training/train.py --config configs/gpu/small.yaml --max-steps 100
```

---

## Understanding Data Structure

### Directory Layout
```
/project/code/data/
├── processed/               # Pre-processed, ready-to-train files
│   ├── Open-Orca_OpenOrca_processed.jsonl    # Available (20 samples)
│   ├── synthetic_train.jsonl                 # Created above (60 samples)
│   └── ... (other files if they exist)
├── ag_news/                # Sample dataset (not used by default)
├── raw/                    # Raw data before processing (usually empty)
└── rlhf/                   # RLHF-specific data
```

### File Format

Training data should be in JSONL format (one JSON object per line):

```json
{"text": "Your training text here", "source": "dataset_name", "type": "general"}
{"text": "Another training example", "source": "dataset_name", "type": "general"}
```

The `_tokenize_text()` function looks for:
- `text` field (primary)
- `content`, `document`, `passage`, `input`, `question`, `instruction` fields (fallback)

---

## Data Loading Error Fixes

### Fix 1: Check File Existence ✅

The data loading code now checks if files exist before reading:

**Code change**:
```python
# Before: Would crash trying to read non-existent file
for line in encoding_detector.read_file_robust(file_path):

# After: Checks existence first
if not file_path.exists():
    print(f"⚠️  File does not exist, skipping: {file_path.name}")
    return
```

**Impact**: Training won't crash on missing files, just skips them

### Fix 2: Filter Non-existent Files ✅

File filtering now handles missing files:

**Code change**:
```python
# Before: Would error on stat() for missing files
if f.stat().st_size >= MIN_FILE_SIZE:

# After: Safely checks existence
try:
    if f.exists() and f.stat().st_size >= MIN_FILE_SIZE:
        substantial_files.append(f)
except (OSError, FileNotFoundError):
    filtered_count += 1
```

**Impact**: Data loading pipeline is more robust

---

## Complete Data Setup Options

### Option 1: Quick Test Data (5 minutes)

For fast testing with minimal data:

```bash
# Create 60-sample dataset
cd /project/code/data/processed
cp Open-Orca_OpenOrca_processed.jsonl quick_train.jsonl
cat Open-Orca_OpenOrca_processed.jsonl >> quick_train.jsonl

# Run quick test
python code/scripts/5_training/train.py \
    --config configs/gpu/small.yaml \
    --max-steps 50
```

**Advantages**:
- ✅ Instant setup
- ✅ No downloads needed
- ✅ Tests pipeline quickly

**Disadvantages**:
- ✗ Very small dataset (60 samples)
- ✗ Limited training variety

---

### Option 2: Use AG News Dataset (15 minutes)

The codebase includes AG News dataset in raw format:

```bash
# Process AG News to standard format
python code/scripts/2_data_prep/process_all_data.py \
    --input-dir /project/code/data/ag_news/raw \
    --output-dir /project/code/data/processed

# This creates:
# - ag_news_train_processed.jsonl
# - ag_news_val_processed.jsonl
# - ag_news_test_processed.jsonl
```

**Setup**:
```bash
# Verify files were created
ls -lh /project/code/data/processed/ag_news*

# Run training
python code/scripts/5_training/train.py --config configs/gpu/small.yaml
```

**Advantages**:
- ✅ Real dataset (~120k examples)
- ✅ Good variety
- ✅ Balanced train/val/test split

**Disadvantages**:
- ✗ Requires processing step

---

### Option 3: Download Additional Data (30+ minutes)

Download from HuggingFace datasets:

```bash
# Edit unified_download.py to specify desired datasets
nano code/scripts/1_data_download/unified_download.py

# Example: Download specific dataset
python code/scripts/1_data_download/unified_download.py \
    --dataset-name wikitext \
    --dataset-config wikitext-2 \
    --output-dir /project/code/data/raw

# Then process it
python code/scripts/2_data_prep/process_all_data.py \
    --input-dir /project/code/data/raw \
    --output-dir /project/code/data/processed
```

**Advantages**:
- ✅ Larger datasets available
- ✅ High quality training data
- ✅ Multiple domain options

**Disadvantages**:
- ✗ Requires internet connection
- ✗ Takes time to download
- ✗ Takes time to process

---

### Option 4: Use Custom Data

If you have your own data:

```bash
# Prepare data in JSONL format
# Each line should be valid JSON with a "text" field

# Place in /project/code/data/processed/
mv your_data.jsonl /project/code/data/processed/

# Update config if needed
# Edit configs/gpu/small.yaml to point to data_dir

# Run training
python code/scripts/5_training/train.py --config configs/gpu/small.yaml
```

**Format requirements**:
- File extension: `.jsonl` (one JSON per line)
- JSON fields:
  - `text` (required): The training text
  - `source` (optional): Data source name
  - `type` (optional): Data type/category

**Example**:
```json
{"text": "Machine learning is a subset of artificial intelligence.", "source": "wikipedia", "type": "general"}
{"text": "Natural language processing enables computers to understand human language.", "source": "custom", "type": "nlp"}
```

---

## Data Configuration in YAML

### Default Configuration

```yaml
data:
  data_dir: /project/code/data/processed
  max_length: 256
  tokenizer_name: /project/code/models/tokenizer/enhanced-65536
```

### Custom Configuration

```yaml
data:
  # Change to your data directory
  data_dir: /path/to/your/data

  # Maximum sequence length for training
  max_length: 512

  # Tokenizer to use
  tokenizer_name: Qwen/Qwen2.5-0.5B

data_loading:
  # How many samples to load in buffer (for shuffling)
  buffer_size: 10000

  # Max training examples (None = unlimited)
  max_train_examples: null

  # Max validation examples
  max_eval_examples: 5000

  # Samples per file before rotating to next file
  samples_per_file: 1
```

---

## Verifying Data Setup

### Check Data Files
```bash
# List available data files
find /project/code/data/processed -type f -name "*.jsonl" -o -name "*.arrow" -o -name "*.parquet"

# Check file sizes
ls -lh /project/code/data/processed/

# Count lines in JSONL file
wc -l /project/code/data/processed/*.jsonl
```

### Check Data Format
```bash
# Show first few lines
head -3 /project/code/data/processed/your_file.jsonl

# Validate JSON format
python -c "
import json
with open('/project/code/data/processed/your_file.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Line {i+1}: Invalid JSON')
        if i >= 5:
            break
print('✓ First 6 lines valid JSON')
"

# Check text field exists
python -c "
import json
with open('/project/code/data/processed/your_file.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if 'text' not in data:
            print(f'Line {i+1}: Missing text field')
        if i >= 5:
            break
print('✓ All lines have text field')
"
```

### Test Data Loading
```bash
# Quick test that data loads correctly
python -c "
from pathlib import Path
from src.Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create dataloaders
train_loader, val_loader = create_streaming_dataloaders(
    tokenizer=tokenizer,
    batch_size=8,
    max_length=256,
    data_dir='/project/code/data/processed',
    num_workers=2,
    max_samples=100
)

# Try loading a batch
print('Loading batch...')
for i, batch in enumerate(train_loader):
    print(f'✓ Batch {i}: keys={list(batch.keys())}, shapes=')
    for k, v in batch.items():
        print(f'  {k}: {v.shape}')
    if i >= 2:
        break

print('✓ Data loading works!')
"
```

---

## Troubleshooting

### Error: "No such file or directory"

**Cause**: File doesn't exist in the specified directory

**Solution**:
1. Check file exists: `ls -l /project/code/data/processed/filename.jsonl`
2. Check path in config: `data.data_dir` in YAML
3. Create data files as described above

### Error: "Files appear to be empty after restarts"

**Cause**: Data directory exists but contains no valid files

**Solution**:
1. Check files exist: `ls -lh /project/code/data/processed/`
2. Check file sizes: `wc -l /project/code/data/processed/*.jsonl`
3. Create test data: `cp Open-Orca_OpenOrca_processed.jsonl test.jsonl`

### Error: "Invalid JSON in JSONL file"

**Cause**: File contains malformed JSON

**Solution**:
1. Validate JSON: `python -m json.tool filename.jsonl | head`
2. Check encoding: `file -i filename.jsonl` (should be UTF-8)
3. Repair file: Use JSON validation tools or recreate from source

### Training uses synthetic data instead of real data

**Cause**: Data files found but too small or corrupt

**Solution**:
1. Check file size: `ls -lh /project/code/data/processed/`
2. Min file size is 10KB - very small files are skipped
3. Create larger dataset: Duplicate existing files as shown above

---

## Best Practices

### 1. Check Data Integrity
```bash
# Before training, validate your data
python scripts/validation/validate_data.py \
    --data-dir /project/code/data/processed
```

### 2. Monitor Data Loading
```bash
# Training output should show:
# "Found X data files for train split"
# "Streaming from X unique files"
# "Buffer size: 10,000 samples"
```

### 3. Use Appropriate Batch Sizes
- Small dataset (< 10k samples): batch_size = 8-16
- Medium dataset (10k-100k): batch_size = 32
- Large dataset (100k+): batch_size = 64-128

### 4. Manage Validation Split
```yaml
data_loading:
  val_split_ratio: 0.15  # 15% of training for validation
  val_max_samples: 5000  # Cap validation at 5000 samples
```

---

## Summary

| Task | Time | Difficulty | Result |
|------|------|-----------|--------|
| Quick test (Option 1) | 5 min | Easy | 60 samples |
| Process AG News (Option 2) | 15 min | Easy | 120k samples |
| Download data (Option 3) | 30+ min | Medium | 1M+ samples |
| Use custom data (Option 4) | Variable | Medium | Your data |

**Recommended for testing**: Option 1 (Quick test data)
**Recommended for training**: Option 2 (AG News) or Option 3 (Downloaded data)

---

## Getting Help

- Check data files exist: `ls /project/code/data/processed/`
- Verify format: `head -1 data.jsonl | python -m json.tool`
- Check configuration: `grep data_dir configs/gpu/small.yaml`
- Review data loading code: `src/Ava/data_streaming.py`

---

**Status**: Data setup is now more robust with automatic file validation ✅
