# Data Issues Analysis - `/project/code/data/pretokenized`

## Critical Issues Found

### 1. **File Format Mismatch** ❌
**Problem**: Training script expects `.arrow` files, but directory contains `.parquet` files.

- **Expected**: `*.arrow` files
- **Found**: `*.parquet` files (12 files, ~71MB total)
- **Impact**: Training script will fail to load data and fall back to dummy dataset

**Code reference**: [train_100m_full.py:205](code/scripts/5_training/train_100m_full.py#L205)
```python
arrow_files = sorted(list(Path(data_dir).glob('*.arrow')))  # Looks for .arrow only!
if arrow_files:
    # ... loads data
else:
    # Falls back to DummyDataset
```

### 2. **Parquet Files Are Readable** ✓
The parquet files have the correct structure:
- **Columns**: `input_ids`, `attention_mask`, `length`
- **Data type**: Lists of token IDs (already tokenized)
- **Example file**: `Anthropic_model-written-evals_processed.parquet` (3,252 rows, 270K)

### 3. **Solution Options**

#### Option A: Convert Parquet → Arrow (Recommended)
Convert all parquet files to arrow format that the training script expects.

```bash
# Convert parquet files to arrow format
python3 << 'EOF'
import pandas as pd
from datasets import Dataset
import os
from pathlib import Path

data_dir = '/project/code/data/pretokenized'
parquet_files = sorted(Path(data_dir).glob('*.parquet'))

for parquet_file in parquet_files:
    # Read parquet
    df = pd.read_parquet(parquet_file)

    # Convert to Arrow
    dataset = Dataset.from_pandas(df)
    arrow_file = parquet_file.with_suffix('.arrow')
    dataset.save_to_disk(arrow_file)

    print(f"✓ Converted: {parquet_file.name} → {arrow_file.name}")
    print(f"  Rows: {len(df)}, Size: {arrow_file.stat().st_size / 1e6:.1f}MB")
EOF
```

#### Option B: Update Training Script to Accept Parquet
Modify the training script to load parquet files directly (simpler for now):

```python
# In train_100m_full.py, around line 205:
parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))
if parquet_files:
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            dataset = Dataset.from_pandas(df)
            datasets_list.append(dataset)
        except Exception as e:
            continue
```

#### Option C: Use Dummy Data for Testing
Keep current setup but understand you're training on random data, not real data.

---

## File Inventory

| File | Size | Rows | Status |
|------|------|------|--------|
| AllenAI_prosocial-dialog_processed.parquet | 1.1M | ? | ✓ Readable |
| Anthropic_model-written-evals_processed.parquet | 270K | 3,252 | ✓ Readable |
| HuggingFaceH4_no_robots_processed.parquet | 4.8M | ? | ✓ Readable |
| OpenAssistant_oasst2_processed.parquet | 25M | ? | ✓ Readable |
| QingyiSi_Alpaca-CoT_processed.parquet | 2.7M | ? | ✓ Readable |
| blended_skill_talk_processed.parquet | 12K | ? | ✓ Readable |
| iamtarun_python_code_instructions_18k_alpaca_processed.parquet | 5.6M | ? | ✓ Readable |
| roneneldan_TinyStories_once_upon_stories.parquet | 29M | ? | ✓ Readable |
| sahil2801_CodeAlpaca-20k_processed.parquet | 2.9M | ? | ✓ Readable |

**Total**: 12 files, ~71MB, all parquet format

---

## Recommended Fix

Run this conversion script to create arrow files for training:

```bash
cd /project/code/data/pretokenized
python3 << 'EOF'
import pandas as pd
from datasets import Dataset
from pathlib import Path
import os

data_dir = Path('/project/code/data/pretokenized')
parquet_files = sorted(data_dir.glob('*.parquet'))

print(f"Converting {len(parquet_files)} parquet files to arrow format...\n")

total_rows = 0
for i, parquet_file in enumerate(parquet_files, 1):
    try:
        # Read parquet
        df = pd.read_parquet(parquet_file)
        n_rows = len(df)
        total_rows += n_rows

        # Convert to Arrow (HuggingFace format)
        dataset = Dataset.from_pandas(df)
        arrow_file = parquet_file.with_suffix('.arrow')
        dataset.save_to_disk(str(arrow_file))

        size_mb = arrow_file.stat().st_size / 1e6
        print(f"{i:2d}. ✓ {parquet_file.name}")
        print(f"     → {arrow_file.name} ({n_rows:,} rows, {size_mb:.1f}MB)\n")

    except Exception as e:
        print(f"{i:2d}. ✗ {parquet_file.name}: {e}\n")

print(f"Total rows converted: {total_rows:,}")
print(f"\nTraining script will now find and load all .arrow files automatically!")
EOF
```

---

## Why This Matters

1. **Training Script Only Looks for `.arrow` Files**: The data loader in `train_100m_full.py` uses `glob('*.arrow')`
2. **Current Status**: Falls back to `DummyDataset` with random data
3. **Real Data Available**: All parquet files are ready to use, just need format conversion
4. **Easy Fix**: One-time conversion takes ~30 seconds

---

## Quick Diagnosis

To verify the issue yourself:

```bash
# Check what training script sees:
ls /project/code/data/pretokenized/*.arrow 2>/dev/null || echo "No arrow files found!"

# Check what actually exists:
ls /project/code/data/pretokenized/*.parquet | wc -l
```
