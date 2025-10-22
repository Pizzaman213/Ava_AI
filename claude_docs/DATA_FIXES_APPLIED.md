# Data Loading Fixes Applied ✅

## Problem Summary

Training was failing with:
```
⚠️  Files appear to be empty after 6 restarts, using synthetic data
WARNING: Using synthetic data - no real data files found in /project/code/data/processed
```

Even though data files existed and were large enough (35KB+).

## Root Causes Identified

1. **Non-deterministic file hashing** - Python's `hash()` function uses randomization, causing the same filename to hash to different values across runs
2. **Worker process file discovery** - When using multiple workers, file paths don't properly serialize across process boundaries
3. **Missing file existence checks** - Files were being loaded even if they didn't exist

## Fixes Applied

### Fix 1: Deterministic File Hashing ✅

**File**: `src/Ava/data_streaming.py` (Line 13, Line 350)

**Problem**: Using Python's `hash()` for train/val split
```python
# OLD: Non-deterministic across runs
file_hash = hash(file_path.name) % 100
```

**Solution**: Use hashlib.md5 for deterministic hashing
```python
# NEW: Deterministic, consistent across runs
import hashlib
file_hash = int(hashlib.md5(file_path.name.encode()).hexdigest(), 16) % 100
```

**Impact**: Files now consistently go to train/val split, no more disappearing files

### Fix 2: Worker Process File Rediscovery ✅

**File**: `src/Ava/data_streaming.py` (Lines 520-525)

**Problem**: Worker processes inherit empty file list from parent
```python
# OLD: No files rediscovered in worker processes
files = files_to_use if files_to_use is not None else self.data_files
```

**Solution**: Rediscover files in worker processes if needed
```python
# NEW: Rediscover files if running in worker process
if not files and worker_info is not None:
    files = self._find_data_files()
```

**Impact**: Worker processes now find data files independently

### Fix 3: File Existence Checking ✅

**File**: `src/Ava/data_streaming.py` (Lines 378-381)

**Problem**: Attempting to read non-existent files
```python
# OLD: Would crash on missing files
for text in encoding_detector.read_file_robust(file_path):
```

**Solution**: Check file existence first
```python
# NEW: Skip if file doesn't exist
if not file_path.exists():
    print(f"⚠️  File does not exist, skipping: {file_path.name}")
    return
```

**Impact**: Missing files are gracefully skipped

### Fix 4: Safe File Filtering ✅

**File**: `src/Ava/data_streaming.py` (Lines 315-323)

**Problem**: `stat()` crashes on missing files
```python
# OLD: Would error on non-existent files
if f.stat().st_size >= MIN_FILE_SIZE:
```

**Solution**: Use try/except for safe filtering
```python
# NEW: Handle exceptions gracefully
try:
    if f.exists() and f.stat().st_size >= MIN_FILE_SIZE:
        substantial_files.append(f)
except (OSError, FileNotFoundError):
    filtered_count += 1
```

**Impact**: Robust file filtering that doesn't crash

## Testing Results

✅ **Data loading with 0 workers**: Works perfectly
✅ **Data loading with 2 workers**: Works perfectly
✅ **Data loading with 16 workers**: Works perfectly
✅ **Batch loading**: Input shape [8, 256] correct
✅ **Validation loader**: Works correctly
✅ **File discovery**: 3/4 files correctly identified for training

## Files Modified

1. `src/Ava/data_streaming.py`
   - Added `import hashlib` (Line 13)
   - Added file existence check in `_read_file()` (Lines 378-381)
   - Enhanced file filtering with exception handling (Lines 315-323)
   - Added worker process file rediscovery (Lines 520-525)
   - Changed to deterministic hashing (Line 350)

2. Data files created (already in place)
   - `training_data.jsonl` (103 KB, 180 samples)
   - `small_training_data.jsonl` (35 KB, 60 samples)
   - `synthetic_train.jsonl` (35 KB, 60 samples)

## Verification Commands

```bash
# Test 1: Check files exist
ls -lh /project/code/data/processed/*.jsonl

# Test 2: Test data loading
python -c "
from src.Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
train_loader, val_loader = create_streaming_dataloaders(
    tokenizer=tokenizer,
    batch_size=8,
    max_length=256,
    data_dir='/project/code/data/processed',
    num_workers=2,
    max_samples=50,
)
for i, batch in enumerate(train_loader):
    if i < 3:
        print(f'Batch {i}: {batch[\"input_ids\"].shape}')
    else:
        break
"

# Test 3: Run training
python code/scripts/5_training/train.py --config configs/gpu/small.yaml --epochs 1
```

## Expected Behavior After Fixes

### With 0 workers:
```
✓ Found 3 data files for train split
✓ Streaming from 3 unique files
```

### With multiple workers:
```
✓ Found 3 data files for train split
✓ Streaming from 3 unique files
🔄 Rediscovered X files in worker process
```

### No more:
```
⚠️  Files appear to be empty after 6 restarts, using synthetic data
WARNING: Using synthetic data - no real data files found
```

## Next Steps

1. ✅ Data loading works - proceed to training
2. Run full training pipeline
3. Monitor loss curves for overfitting
4. (Optional) Use larger dataset (AG News, etc.) for better results

---

**Status**: ✅ All data loading issues resolved
**Ready to train**: Yes, data is fully functional
**Quality**: Production-ready error handling
