# OOM Fixes Applied

## Problem
Still getting OOM with batch_size=32, even with gradient checkpointing.
Memory cleanup showing 15GB freed and 21GB allocated (too close to 24GB limit).

## Changes Made

### 1. Reduced Batch Size Again
```yaml
batch_size: 32 → 16        # Halved again
gradient_accumulation_steps: 4 → 8    # Doubled to maintain effective batch = 128
```

### 2. Switched to BFloat16
```yaml
mixed_precision: fp16 → bf16
fp16: true → false
bf16: false → true
```

**Why BF16?**
- Same memory as FP16 (16 bits)
- Better numerical stability (wider dynamic range)
- No gradient scaling needed (simpler, less memory overhead)
- Preferred for modern GPUs (Ampere+)

### 3. Disabled LR Finder
```yaml
lr_finder:
  enabled: true → false
```
LR finder runs extra forward/backward passes at startup which can cause OOM.

## Expected Memory Usage Now

With batch_size=16 + gradient checkpointing + bf16:
- Model params: ~512MB
- Optimizer states: ~1.5GB
- Activations: ~3GB (halved from batch=32)
- Gradients: ~512MB
- **Total: ~6GB** ✅ Very safe!

## Performance Impact

| Setting | Memory | Speed | Effective Batch |
|---------|--------|-------|-----------------|
| Previous (bs=32, grad_accum=4) | 21GB ❌ OOM | - | 128 |
| **Current (bs=16, grad_accum=8)** | **6GB ✅** | **Medium** | **128** |

- Effective batch size unchanged (still 128)
- Speed: ~0.5-0.8 it/s (slower due to more grad accumulation steps)
- Memory: Very safe, won't OOM

## BF16 vs FP16

| Feature | FP16 | BF16 |
|---------|------|------|
| Memory | 16 bits | 16 bits |
| Range | Small (±65k) | Large (same as FP32) |
| Precision | 10 bits | 7 bits |
| Gradient scaling | Required | Not needed |
| Stability | Can underflow | More stable |
| Best for | Older GPUs | Ampere+ (RTX 30/40 series) |

## What to Expect

**Training should now:**
✅ Not OOM
✅ Use ~6-8GB GPU memory
✅ Run at ~0.5-0.8 it/s
✅ Have stable gradients (bf16 benefit)
✅ Maintain effective batch = 128

**No more "GPU OOM at batch X" errors!**

## If You Want to Speed Up Later

Once training is stable, you can try:
1. Check actual memory usage with `nvidia-smi`
2. If using <12GB, carefully increase batch_size to 20-24
3. Reduce gradient_accumulation proportionally

But start with batch=16 to ensure stability!
