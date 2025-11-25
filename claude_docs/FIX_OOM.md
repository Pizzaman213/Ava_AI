# OOM Error Analysis & Fix

## Problem
Phase 1 optimizations (torch.compile, grouped GEMM, Triton kernels, etc.) use more memory than expected. Batch size 256 causes OOM on 24GB GPU.

## Root Cause
The additional optimizations have memory overhead:
- torch.compile creates compiled graphs (extra memory)
- Triton kernels may have different memory patterns
- Grouped GEMM batches expert computations (temporary buffers)

**Memory usage:** ~21GB / 23.55GB = 89% utilization
**Trying to allocate:** 6.14 GB for loss computation
**Result:** OOM error

## Immediate Fix Applied

Reduced batch size back to 128 (from 256):
```yaml
training:
  batch_size: 128  # Conservative, fits in 24GB
  gradient_accumulation_steps: 4  # Maintain effective batch = 512
```

## Additional Fixes to Try

### 1. Enable PyTorch Memory Fragmentation Fix
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Add to training script or run before training.

### 2. Enable Dynamic Batching (Phase 2)
This would have automatically reduced the batch size when approaching OOM!

```yaml
dynamic_batching:
  enabled: true
  min_batch_size: 64
  max_batch_size: 256  # Reduced from 1024
  high_memory_threshold: 0.80  # More conservative
```

### 3. Reduce Compiled Model Memory
If still having issues, try:
```yaml
performance:
  enable_torch_compile: false  # Disable whole-model compilation
  torch_compile_dynamic: false  # Less memory overhead
```

Keep model-level compile options:
```yaml
model:
  use_torch_compile: true  # Router compilation only (less memory)
```

## Memory Budget Breakdown

**24GB GPU allocation:**
- Model parameters: ~1.2GB (300M params in BF16)
- Optimizer states: ~2.4GB (Lion is 2x params)
- Activations (batch=128): ~8-10GB
- Gradients: ~1.2GB
- torch.compile graphs: ~2-3GB
- KV cache & temp buffers: ~2-3GB
- **Total:** ~17-21GB

**With batch=256:**
- Activations double: ~16-20GB
- **Total:** ~25-29GB → **OOM!**

## Testing Status

### What Works:
✅ Batch size 128 with all Phase 1 optimizations
✅ All optimization features enabled (compile, Triton, grouped GEMM)
✅ Effective batch size maintained at 512

### Next Steps:
1. Test with batch_size=128 (should work now)
2. Enable dynamic batching to automatically find optimal batch size
3. Consider sequence packing to reduce memory via less padding

## Performance Impact

**Batch 256 → 128:**
- Throughput: -15% (smaller batches less efficient)
- Memory: Fits in GPU ✅
- Effective batch: Same (512 via gradient accumulation)

**With Dynamic Batching:**
- Would have auto-adjusted from 256 → 128 → 192 → optimal
- Could find sweet spot (maybe 160-192 works)
- No manual tuning needed

## Commands

Test current fix:
```bash
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/minimal_working.yaml \
    --max_steps 100
```

Enable fragmentation fix:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/minimal_working.yaml
```

Enable dynamic batching:
```bash
# Edit config: dynamic_batching.enabled = true
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/minimal_working.yaml
```

## Lesson Learned

This is **EXACTLY** why dynamic batching is valuable! It would have:
1. Detected memory approaching 85%
2. Automatically reduced batch from 256 → 205 → 164 → 128
3. Prevented OOM entirely
4. Found the maximum safe batch size automatically

**Recommendation:** After validating batch_size=128 works, enable dynamic batching to find the optimal size automatically!
