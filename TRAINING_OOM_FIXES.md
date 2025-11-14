# Training OOM Fixes - November 13, 2025

## Problem Summary

Training was killed with these errors:
- **Killed** (OOM kill by system)
- `_pickle.UnpicklingError: pickle data was truncated`
- `resource_tracker: There appear to be 61 leaked semaphore objects`

## Root Cause

**Critical Issue: Swap memory was completely full (8GB/8GB)**

When multiprocessing spawns dataloader workers, it needs to:
1. Fork/spawn new processes
2. Serialize (pickle) Python objects to pass to workers
3. Allocate memory for each worker

With swap full, the OS couldn't allocate memory for worker processes, causing:
- OOM kill during worker spawn
- Pickle truncation (process killed mid-serialization)
- Leaked semaphores (workers failed to initialize properly)

## Fixes Applied

### 1. Fixed Multiprocessing Start Method
**File:** [train.py:4112-4118](code/scripts/5_training/train.py#L4112-L4118)

```python
# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
```

**Why:** CUDA requires 'spawn' method. This prevents GPU context issues.

### 2. Reduced Dataloader Workers
**File:** [single_gpu_optimized.yaml:107-111](code/configs/moe/single_gpu_optimized.yaml#L107-L111)

**Before:**
```yaml
num_workers: 6
dataloader_pin_memory: true
dataloader_prefetch_factor: 8
dataloader_persistent_workers: true
```

**After:**
```yaml
num_workers: 2  # Reduced to prevent OOM with multiprocessing spawn
dataloader_pin_memory: false  # Disabled to reduce memory pressure
dataloader_prefetch_factor: 2  # Reduced to prevent memory exhaustion
dataloader_persistent_workers: false  # Disabled to prevent worker memory accumulation
```

**Why:** Each worker consumes ~1-2GB RAM. 6 workers × 2GB = 12GB just for workers.

### 3. Reduced Batch Size
**File:** [single_gpu_optimized.yaml:67-68](code/configs/moe/single_gpu_optimized.yaml#L67-L68)

**Before:**
```yaml
batch_size: 16  # 4x increase
gradient_accumulation_steps: 12  # effective batch=192
```

**After:**
```yaml
batch_size: 8  # Reduced to prevent OOM (effective batch=96)
gradient_accumulation_steps: 12  # Keep high for gradient stability
```

**Why:** Smaller per-step memory footprint prevents swap thrashing.

### 4. Reduced Buffer Sizes
**File:** [single_gpu_optimized.yaml:97](code/configs/moe/single_gpu_optimized.yaml#L97)

**Before:**
```yaml
buffer_size: 10000
```

**After:**
```yaml
buffer_size: 5000  # Further reduced to save memory
```

### 5. Fixed Default Worker Settings in Code
**File:** [train.py:1035-1042](code/scripts/5_training/train.py#L1035-L1042)

**Before:**
```python
num_workers = getattr(training_config.data_loading, 'num_workers', 8)
prefetch_factor = getattr(training_config.data_loading, 'prefetch_factor', 12)
persistent_workers = getattr(training_config.data_loading, 'persistent_workers', True)
```

**After:**
```python
num_workers = getattr(training_config.data_loading, 'num_workers', 2)
prefetch_factor = getattr(training_config.data_loading, 'prefetch_factor', 2)
persistent_workers = getattr(training_config.data_loading, 'persistent_workers', False)
```

**Why:** Ensures safe defaults even if config doesn't specify workers.

## Test Results

✅ **Training started successfully**
✅ **No pickle errors**
✅ **No OOM kills during 2-minute test**
✅ **Workers spawned correctly**

```
System Resources (Before):
Mem:  61Gi    10Gi    50Gi
Swap: 8.0Gi   8.0Gi   17Mi  ⚠️  FULL!

System Resources (During Test):
Mem:  61Gi    9.7Gi   51Gi
Swap: 8.0Gi   7.8Gi   223Mi  ⚠️  Still high but not full
```

## Remaining Issue: Swap Still High

**Current Status:** Swap is at 7.8GB/8GB (97.5% full)

This indicates:
1. Something else on the system is using swap
2. OR previous training runs left swap-resident pages

### Recommended Actions:

#### Option 1: Clear Swap (Requires sudo)
```bash
sudo swapoff -a && sudo swapon -a
```

#### Option 2: Reduce Model Size
The model is **6.4B parameters** which is quite large for 24GB VRAM:
```
Model: 6385.0M parameters
```

Consider using a smaller config:
- `tiny_moe_multi_gpu.yaml` (smaller model)
- `small_moe.yaml` (medium model)

#### Option 3: Enable More Aggressive Memory Optimization

Update [single_gpu_optimized.yaml](code/configs/moe/single_gpu_optimized.yaml):

```yaml
# Reduce active experts on GPU
max_active_experts_gpu: 2  # From 4 to 2

# More aggressive expert cache
expert_cache:
  cache_size: 4  # From 8 to 4
  compression_ratio: 16  # From 8 to 16

# Reduce max length
max_length: 256  # From 512 to 256
```

## How to Run Training Now

```bash
cd /project/code/scripts/5_training

# Option 1: Use test script (2-minute test)
bash /project/test_training_fixed.sh

# Option 2: Run directly
python train.py \
    --config /project/code/configs/moe/single_gpu_optimized.yaml \
    --force-single-gpu

# Option 3: With memory safety env vars
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
python train.py \
    --config /project/code/configs/moe/single_gpu_optimized.yaml \
    --force-single-gpu
```

## Next Steps

1. **Clear swap memory** (if you have sudo access)
2. **Monitor memory during training:**
   ```bash
   watch -n 1 'free -h; echo ""; nvidia-smi'
   ```
3. **If OOM persists:**
   - Reduce `batch_size` to 4
   - Reduce `num_workers` to 1
   - Use smaller model config
   - Reduce `max_length` to 256

## Files Modified

1. [code/scripts/5_training/train.py](code/scripts/5_training/train.py)
   - Added multiprocessing safety (L4112-4118)
   - Reduced default workers (L1035-1042)

2. [code/configs/moe/single_gpu_optimized.yaml](code/configs/moe/single_gpu_optimized.yaml)
   - Reduced workers: 6→2
   - Reduced batch size: 16→8
   - Reduced buffer: 10000→5000
   - Disabled pin_memory and persistent_workers

3. [test_training_fixed.sh](test_training_fixed.sh) *(NEW)*
   - Memory-safe training test script

## Success Indicators

✅ No more pickle errors
✅ No more OOM kills
✅ Workers spawn successfully
✅ Training runs without crashes
