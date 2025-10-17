# LR Finder Analysis - Why It Failed

**Date:** 2025-10-15
**Issue:** LR finder recommended 7.94e-07 (way too low)

## LR Finder Results

```
fastai:   3.65e-07  (very conservative)
valley:   1.12e-05  (balanced)
steepest: 1.59e-07  (very conservative)

Variance: 70.67x (VERY HIGH)
Recommended: 7.94e-07 (geometric mean - TOO LOW!)
```

## Why It Failed

### 1. High Variance = Unreliable Results
When methods disagree by **70x**, the LR finder results are **not trustworthy**.

**Causes of high variance:**
- Model starts from **random initialization** (loss ~32)
- Loss drops dramatically during test (32 → 5.5)
- Very noisy loss landscape early in training
- Different methods see different patterns in the noise

### 2. Geometric Mean Problem
When variance is high, geometric mean gets **pulled down** by conservative methods:

```python
geometric_mean(3.65e-07, 1.12e-05) = 7.94e-07
#               ↑ pulls down     ↑ reasonable
```

This gives an **unusably low LR** that will make training painfully slow.

## The Solution

### What To Use Instead

**Use the valley method (1.12e-05)** OR **rule-of-thumb (3e-4)**:

```yaml
# Option 1: Valley method (conservative but reliable)
learning_rate: 1.12e-05
lr_end: 3.73e-08  # 1/300 of peak

# Option 2: Rule-of-thumb (recommended for 83M params)
learning_rate: 3.0e-04
lr_end: 1.0e-06   # 1/300 of peak
```

**We're using Option 2** because:
- Standard for models this size
- Valley method might still be too conservative
- Can always reduce if training diverges

### When To Trust LR Finder

**Trust it when:**
- ✅ Variance < 10x (methods agree)
- ✅ Confidence: HIGH or MEDIUM
- ✅ Loss curve is smooth

**Don't trust it when:**
- ❌ Variance > 30x (methods disagree wildly)
- ❌ Confidence: VERY LOW
- ❌ Model from random initialization
- ❌ Loss curve very noisy

### Better Approach

**Run LR finder AFTER some training:**

```bash
# Step 1: Train with rule-of-thumb LR for 1000 steps
cd /project/code
python train.py --config configs/gpu/small.yaml --max-steps 1000

# Step 2: NOW run LR finder (loss will be more stable)
python scripts/4_Find_Lr/run_lr_finder_enhanced.py \
    --config configs/gpu/small.yaml \
    --output lr_results_after_warmup/

# Step 3: Use the (hopefully more reliable) result
# If variance is still >30x, stick with 3e-4
```

## Fix Applied

Updated [run_lr_finder_enhanced.py](code/scripts/4_Find_Lr/run_lr_finder_enhanced.py):

```python
# NEW: If variance > 30x, use valley method instead of geometric mean
if variance > 30.0:
    recommended_lr = valley_lr  # Most robust for noisy landscapes
    strategy = "valley (high variance, geometric mean unreliable)"
else:
    geometric_mean = math.sqrt(valley_lr * fastai_lr)
    recommended_lr = geometric_mean
    strategy = "geometric_mean(valley, fastai)"
```

This prevents the LR finder from recommending unusably low LRs when results are unreliable.

## Current Config Status

**✅ CORRECT:** Restored to manually-set values:

```yaml
learning_rate: 3.0e-04  # Good for 83M params
lr_end: 1.0e-06         # Proper decay (1/300 of peak)
warmup_steps: 1000      # Stable warmup
```

**❌ DO NOT use the LR finder result (7.94e-07)** - it's too low!

## Recommendation

**Just start training with current config (3e-4 LR).**

Don't run LR finder again until:
1. You've trained for 1000+ steps (more stable loss)
2. Or you see signs of instability (loss spiking, gradients exploding)

The rule-of-thumb LR (3e-4) is much more reliable than the LR finder in this case.
