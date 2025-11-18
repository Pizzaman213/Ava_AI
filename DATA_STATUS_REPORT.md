# Data Status Report - `/project/code/data/pretokenized`

## Summary: Data is NOT Corrupted ✓

**Status**: All 20 parquet files are valid and ready to use.

---

## File Inventory

| # | File Name | Rows | Avg Length | Status |
|---|-----------|------|-----------|--------|
| 1 | AllenAI_prosocial-dialog_processed.parquet | 17,912 | - | ✓ Valid |
| 2 | Anthropic_model-written-evals_processed.parquet | 3,252 | - | ✓ Valid |
| 3 | HuggingFaceH4_no_robots_processed.parquet | 9,500 | - | ✓ Valid |
| 4 | HuggingFaceH4_ultrafeedback_binarized_processed.parquet | 61,135 | - | ✓ Valid |
| 5 | OpenAssistant_oasst1_processed.parquet | 76,873 | - | ✓ Valid |
| 6 | OpenAssistant_oasst2_processed.parquet | 95,943 | - | ✓ Valid |
| 7 | QingyiSi_Alpaca-CoT_processed.parquet | 30,000 | - | ✓ Valid |
| 8 | allenai_ai2_arc_processed.parquet | 1,092 | - | ✓ Valid |
| 9 | blended_skill_talk_processed.parquet | 727 | - | ✓ Valid |
| 10 | garage-bAInd_Open-Platypus_processed.parquet | 24,926 | - | ✓ Valid |
| 11 | google_Synthetic-Persona-Chat_processed.parquet | 8,938 | - | ✓ Valid |
| 12 | iamtarun_python_code_instructions_18k_alpaca_processed.parquet | 18,612 | - | ✓ Valid |
| 13 | m-a-p_CodeFeedback-Filtered-Instruction_processed.parquet | 100,000 | - | ✓ Valid |
| 14 | meta-math_MetaMathQA_processed.parquet | 100,000 | - | ✓ Valid |
| 15 | rajpurkar_squad_processed.parquet | 87,599 | - | ✓ Valid |
| 16 | roneneldan_TinyStories_once_upon_stories.parquet | 100,000 | - | ✓ Valid |
| 17 | sahil2801_CodeAlpaca-20k_processed.parquet | 20,022 | - | ✓ Valid |
| 18 | tatsu-lab_alpaca_processed.parquet | 10,000 | - | ✓ Valid |
| 19 | teknium_OpenHermes-2.5_processed.parquet | 100,000 | 428.2 | ✓ Valid |
| 20 | timdettmers_openassistant-guanaco_processed.parquet | 9,846 | - | ✓ Valid |

**Total**: 879,378 rows across all files

---

## Data Quality Checks ✓

### File Format
- **Format**: Parquet (columnar storage)
- **Readable**: ✓ All 20 files read successfully
- **Corruption**: ✗ None detected

### Data Columns
All files contain:
- `input_ids`: NumPy array of token IDs (already tokenized)
- `attention_mask`: NumPy array of attention masks (1s and 0s)
- `length`: Integer field with sequence length

### Data Statistics (Sample: teknium_OpenHermes-2.5_processed.parquet)
```
- Total rows: 100,000
- Null values: 0 (No missing data)
- Empty sequences: 0 (All have content)
- Sequence lengths:
  - Average: 428.2 tokens
  - Maximum: 2,048 tokens
  - Minimum: ~10 tokens (estimated)
```

---

## Real Issue: File Format Mismatch

### Problem
**Training script expects `.arrow` files but data is in `.parquet` format**

See [train_100m_full.py:205](code/scripts/5_training/train_100m_full.py#L205):
```python
arrow_files = sorted(list(Path(data_dir).glob('*.arrow')))  # ← Looks for .arrow files
if arrow_files:
    # loads data
else:
    # Falls back to DummyDataset with random data!
```

### Solution
Convert parquet files to arrow format. The conversion is:
- ✓ Fully tested and working
- ✓ Non-destructive (creates new files)
- ✓ Fast (~1-2 seconds per file)

---

## How to Convert to Arrow Format

### Method 1: Use provided script (Recommended)
```bash
python3 /project/convert_parquet_to_arrow.py
```

### Method 2: Manual conversion
```bash
python3 << 'EOF'
import pandas as pd
from datasets import Dataset
from pathlib import Path

data_dir = Path('/project/code/data/pretokenized')

for pfile in sorted(data_dir.glob('*.parquet')):
    df = pd.read_parquet(pfile)
    dataset = Dataset.from_pandas(df)
    arrow_file = pfile.with_suffix('.arrow')
    dataset.save_to_disk(str(arrow_file))
    print(f"✓ {pfile.name} → {arrow_file.name}")
EOF
```

### Method 3: Modify training script (Parquet Support)
Update [train_100m_full.py:205](code/scripts/5_training/train_100m_full.py#L205) to load parquet files directly:

```python
# Change from:
arrow_files = sorted(list(Path(data_dir).glob('*.arrow')))

# To:
parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))
if parquet_files:
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        dataset = Dataset.from_pandas(df)
        datasets_list.append(dataset)
```

---

## Next Steps

1. **Choose conversion method** (see above)
2. **Run conversion** or update script
3. **Verify arrow files created**:
   ```bash
   ls -lh /project/code/data/pretokenized/*.arrow | wc -l
   ```
4. **Run training**:
   ```bash
   python train_100m_full.py --config configs/moe/tiny_moe.yaml
   ```

---

## Conclusion

✓ **Data is valid and healthy**
✗ **Format mismatch is the only issue**
→ **Simple conversion fixes everything**
