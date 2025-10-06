# Data Loader Critical Bug Fixes

**Date:** 2025-10-06
**Issue:** Training stops after ~1,200 steps instead of continuing for full dataset (2.28M samples)

## Root Cause Analysis

Three critical bugs were identified that caused training to halt prematurely:

---

## Bug #1: Multi-Worker File Distribution Deadlock 🔴

### Location
[src/Ava/data_streaming.py:534](../src/Ava/data_streaming.py#L534)

### Problem
```python
# OLD CODE (BUGGY):
worker_files = [f for i, f in enumerate(self.data_files) if i % num_workers == worker_id]
```

**Issue:** With 7 data files and 16 workers:
- Workers 0-6 get 1 file each
- **Workers 7-15 get 0 files** → yield nothing
- DataLoader expects all workers to contribute data
- Results in **deadlock or premature StopIteration**

### Impact
Training halts when workers with no files signal completion, causing the dataloader to think the epoch is finished at ~1,200 steps.

### Fix
Implemented round-robin file cycling to ensure all workers get files:

```python
# NEW CODE (FIXED):
if len(self.data_files) >= num_workers:
    # More files than workers: distribute evenly
    worker_files = [f for i, f in enumerate(self.data_files) if i % num_workers == worker_id]
else:
    # Fewer files than workers: cycle files to give each worker at least one
    worker_files = []
    file_idx = worker_id % len(self.data_files)
    while file_idx < len(self.data_files):
        worker_files.append(self.data_files[file_idx])
        file_idx += num_workers

    # If worker still has no files, assign in round-robin
    if not worker_files:
        worker_files = [self.data_files[worker_id % len(self.data_files)]]
```

### Config Change
Reduced workers from 16 to 8 to better match the 7 available data files:

```yaml
# configs/gpu/small.yaml
training:
  dataloader_num_workers: 8  # Reduced from 16
  prefetch_factor: 4  # Reduced from 8

data_loading:
  num_workers: 8  # Reduced from 16
```

---

## Bug #2: Validation Dataset Hardcoded Limit 🔴

### Location
[src/Ava/data_streaming.py:718](../src/Ava/data_streaming.py#L718)

### Problem
```python
# OLD CODE (BUGGY):
max_samples=max_samples // 10 if max_samples else 1000,  # Hardcoded 1000!
```

**Issue:**
- Validation hardcoded to **1,000 samples** regardless of dataset size
- With 2.28M samples available, this is 0.04% of data
- Evaluation every 5,000 steps hits this limit immediately
- Training interprets exhausted validation as end of data

### Impact
Validation runs dry after 1,000 samples, causing training to think epochs are complete.

### Fix
Made validation samples configurable with ratio-based calculation:

```python
# NEW CODE (FIXED):
# Added new parameters to create_streaming_dataloaders():
val_max_samples: Optional[int] = None,
val_split_ratio: float = 0.1

# Calculate validation samples based on config
if val_max_samples is not None:
    # Use explicit limit if provided
    computed_val_samples = val_max_samples
elif max_samples is not None:
    # Use ratio of training samples
    computed_val_samples = int(max_samples * val_split_ratio)
else:
    # No limit - use None for unlimited validation
    computed_val_samples = None
```

### Config Changes
Added new configuration fields:

```yaml
# configs/gpu/small.yaml
data:
  max_samples: null  # No training limit
  val_max_samples: null  # No validation limit (use ratio)
  val_split_ratio: 0.1  # 10% of training data

data_loading:
  val_max_samples: null
  val_split_ratio: 0.1
```

### Code Changes
Updated [scripts/training/train.py:505-520](../scripts/training/train.py#L505-L520) to pass new parameters:

```python
# Get validation dataset config
val_max_samples = getattr(training_config.data, 'val_max_samples', None)
val_split_ratio = getattr(training_config.data, 'val_split_ratio', 0.1)

train_loader, val_loader = create_streaming_dataloaders(
    # ... other params ...
    val_max_samples=val_max_samples,
    val_split_ratio=val_split_ratio,
)
```

---

## Bug #3: Excessive Buffer Size ⚠️

### Location
- [src/Ava/data_streaming.py:550](../src/Ava/data_streaming.py#L550)
- [configs/gpu/small.yaml:224](../configs/gpu/small.yaml#L224)

### Problem
```yaml
# OLD CONFIG (EXCESSIVE):
buffer_size: 20000  # 20K samples per worker!
```

**Issue:**
- Each of 7 active workers accumulates **20,000 samples** before tokenizing
- Total: **140,000 samples in RAM** before any training starts
- Periodic flushes every **100,000 samples** (line 579)
- Creates massive startup delay and memory pressure

### Impact
- Slow training startup (must fill 140K sample buffer first)
- High memory usage (20K text samples × 16 workers = 320K samples in memory)
- Delayed tokenization and batch yielding

### Fix
Reduced buffer size to 2,000 samples:

```yaml
# configs/gpu/small.yaml
data:
  buffer_size: 2000  # Reduced from 20000

data_loading:
  buffer_size: 2000  # 2K samples × 8 workers = 16K total buffer
```

**Benefits:**
- Faster startup: Fill 16K buffer instead of 320K
- Lower memory: ~2GB instead of ~32GB for text buffer
- Faster data flow: Tokenization starts after 2K samples vs 20K

---

## Summary of Changes

### Files Modified

1. **[src/Ava/data_streaming.py](../src/Ava/data_streaming.py)**
   - Fixed multi-worker file distribution (lines 535-554)
   - Added `val_max_samples` and `val_split_ratio` parameters (lines 666-674)
   - Fixed validation sample calculation (lines 720-746)

2. **[scripts/training/train.py](../scripts/training/train.py)**
   - Added validation config extraction (lines 505-507)
   - Passed new parameters to dataloader (lines 518-519)

3. **[configs/gpu/small.yaml](../configs/gpu/small.yaml)**
   - Reduced `dataloader_num_workers` from 16 to 8
   - Reduced `prefetch_factor` from 8 to 4
   - Reduced `buffer_size` from 20000 to 2000
   - Added `val_max_samples` and `val_split_ratio` fields

### Expected Results

**Before Fixes:**
- Training stops at ~1,200 steps
- Validation exhausted after 1,000 samples
- Workers 7-15 deadlocked with no data
- 140K sample buffer causing delays

**After Fixes:**
- Training continues for full dataset (2.28M samples)
- Validation uses 10% of training data (~228K samples)
- All 8 workers get files via round-robin distribution
- 16K sample buffer for faster startup

**Calculated Training Steps:**
```
Total samples: 2,279,975
Batch size: 8
Gradient accumulation: 4
Effective batch: 32

Expected steps: 2,279,975 / 32 = 71,249 steps
Expected validation samples: 227,997 (10% of training)
```

---

## Testing Recommendations

1. **Verify worker distribution:**
   ```python
   # Check that all 8 workers get files
   # Should see balanced distribution across workers
   ```

2. **Monitor validation samples:**
   ```python
   # Validation should now have ~228K samples instead of 1,000
   # Check logs for "computed_val_samples" value
   ```

3. **Check buffer startup time:**
   ```python
   # Time to first batch should be much faster
   # Look for reduced memory usage in logs
   ```

4. **Verify training continues past 1,200 steps:**
   ```python
   # Training should continue to step 71,249
   # Check for completion or early stopping criteria
   ```

---

## Additional Recommendations

1. **Dynamic worker adjustment:** Consider auto-detecting num_workers based on file count
2. **Validation sampling strategy:** Could implement stratified sampling for better validation
3. **Buffer size auto-tuning:** Could calculate optimal buffer based on available RAM
4. **Progress monitoring:** Add logging for worker file assignments and buffer fill rates

---

## Related Documentation

- [INTERLEAVING_FIX.md](../INTERLEAVING_FIX.md) - Previous data interleaving fixes
- [DATA_LOADER_SUMMARY.md](../../claude_docs/DATA_LOADER_SUMMARY.md) - Data loader architecture
- [TRAINING_OPTIMIZATIONS_APPLIED.md](../../claude_docs/TRAINING_OPTIMIZATIONS_APPLIED.md) - Performance optimizations
