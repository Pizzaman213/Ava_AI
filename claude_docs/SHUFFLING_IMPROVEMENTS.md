# Data Shuffling Improvements

## Changes Made

### 1. Rotating File Sampling (Better Shuffling)
**File:** `code/src/Ava/data_streaming.py:543`

**Before:**
```python
samples_per_file = 500  # Read 500 samples from each file before switching
```

**After:**
```python
samples_per_file = 1  # Read 1 sample from each file before switching for better mixing
```

**Impact:**
- Takes 1 sample from File A → 1 from File B → 1 from File C → ... → rotate back to File A
- Creates much better data mixing across datasets
- Prevents long sequences from a single dataset
- More diverse batches throughout training

---

### 2. Sample-Level Worker Distribution (All Files to All Workers)
**File:** `code/src/Ava/data_streaming.py:679-711`

**Before:**
```python
# Split data files across workers to avoid duplication
worker_files = [f for i, f in enumerate(self.data_files) if i % num_workers == worker_id]
# Result: Each worker only sees 1-2 files out of 11 total
```

**After:**
```python
# All workers access all files for better data mixing
for text in self._stream_examples(files_to_use=None):
    # Worker-level sample distribution: each worker takes every Nth sample
    if sample_index % num_workers != worker_id:
        sample_index += 1
        continue
# Result: All workers access all 11 files
```

**Impact:**
- All workers can now access all 11 training files
- Workers distribute samples instead of files
- Worker N processes every Nth sample (N = num_workers)
- Maximum data diversity for all workers

---

### 3. Improved Logging
**File:** `code/src/Ava/data_streaming.py:539-540`

**Before:**
```python
print(f"  📚 Streaming from {len(set(f[0] for f in file_generators))} unique files")
# Output: "📚 Streaming from 3 unique files" (misleading - only shows per-worker count)
```

**After:**
```python
print(f"  📚 Streaming from {unique_files} unique files (all workers access all files, rotating 1 sample per file)")
# Output: "📚 Streaming from 11 unique files (all workers access all files, rotating 1 sample per file)"
```

**Impact:**
- Clear message about total files being used
- Explains the rotating sampling strategy
- No more confusion about "only 3 files"

---

## Data Flow

### Before Changes
```
12 total files
  ↓
Train/Val split (85/15)
  ↓
11 train files
  ↓
Distribute across 8 workers:
  - Worker 0: 2 files (Anthropic, Microsoft)
  - Worker 1: 2 files (HF_no_robots, CodeAlpaca)
  - Worker 2: 2 files (Ultrachat, Wikipedia)
  - Workers 3-7: 1 file each
  ↓
Each worker reads 500 samples per file before switching
  ↓
Limited data mixing, some workers see fewer datasets
```

### After Changes
```
12 total files
  ↓
Train/Val split (85/15)
  ↓
11 train files
  ↓
ALL workers access ALL 11 files
  ↓
Rotating file sampling:
  File 0 → sample 0
  File 1 → sample 1
  File 2 → sample 2
  ...
  File 10 → sample 10
  File 0 → sample 11  ← rotates back
  File 1 → sample 12
  ...
  ↓
Worker-level distribution:
  - Worker 0: samples 0, 8, 16, 24, ...
  - Worker 1: samples 1, 9, 17, 25, ...
  - Worker 2: samples 2, 10, 18, 26, ...
  - ...
  - Worker 7: samples 7, 15, 23, 31, ...
  ↓
Excellent data mixing, all datasets represented in every batch
```

## Benefits

### ✓ Better Data Mixing
- Each batch contains samples from multiple diverse datasets
- Rotating ensures uniform distribution across sources

### ✓ All Files Utilized
- All 11 training files accessible to all 8 workers
- No worker is limited to just 1-2 files

### ✓ Prevents Dataset Bias
- No long sequences from a single dataset
- Model sees variety throughout training

### ✓ Optimal Worker Utilization
- All workers process diverse data
- No worker is "stuck" with limited data sources

### ✓ Maintains Efficiency
- 50K buffer still provides excellent throughput
- Sample-level distribution adds minimal overhead

## Configuration

Current settings (in `code/scripts/5_training/train.py`):
- **Workers:** 8
- **Buffer size:** 50,000 samples
- **Files per rotation:** 1 sample (rotating mode)
- **Train/Val split:** 85/15 (file-based, deterministic)
- **Weighted mixing:** Enabled (DoReMi-style quality scores)

## Testing

Run the diagnostic:
```bash
python /project/test_fixed_distribution.py
```

Expected output:
```
✓ Worker distribution updated to sample-level
✓ Sample-level filtering implemented
✓ Logging message updated
✓ Rotating shuffle enabled (1 sample per file)
```

## Next Training Run

On the next training run, you should see:
```
📚 Streaming from 11 unique files (all workers access all files, rotating 1 sample per file)
```

Instead of:
```
📚 Streaming from 3 unique files
```

This confirms all fixes are working correctly!
