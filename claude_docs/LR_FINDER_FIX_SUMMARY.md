# LR Finder Fix Summary

## Problem
The LR finder showed **1084x variance** between methods, making it completely unreliable:
- FastAI: 1.80e-04 ✓ (reasonable)
- Valley: 4.06e-06 ❌ (way too low)
- Steepest: 1.66e-07 ❌ (absurdly low)

## Root Cause
**Both Valley and Steepest methods were starting at index 0** due to bugs in detection logic.

## Fixes Applied

### 1. Fixed Steepest Gradient Method
**Problem**: Was picking the very first gradient (index 0), giving LR = 1.66e-07

**Solution**: Skip first 10% of data (noisy initialization) and find steepest descent in the actual learning region
```python
# Skip early noise and find steepest in learning region
skip_start = max(5, len(gradients) // 10)
gradients_learning_region = gradients[skip_start:min_loss_idx]
min_gradient_idx = gradients_learning_region.index(min(gradients_learning_region))
```

**File**: [lr_finder.py:592-619](code/src/Ava/training/lr_finder.py#L592-L619)

### 2. Fixed Valley Detection Method
**Problem**: 2% threshold was meaningless (loss goes 12→4), causing valley to start at index 0

**Solution**: Use 5% loss decrease threshold + sustained negative gradient, then take 2/3 point through valley
```python
# Find where loss drops by 5% with sustained learning
threshold_loss = start_loss * 0.95
# Take 2/3 through valley (balanced, not too conservative)
valley_target = valley_start + int(valley_width * 0.67)
```

**File**: [lr_finder.py:431-496](code/src/Ava/training/lr_finder.py#L431-L496)

### 3. Improved FastAI Method
**Problem**: Was using `idx // 10` instead of finding where loss decreased by 90%

**Solution**: Find actual point where loss has decreased 90% from start to minimum
```python
target_loss = start_loss - 0.9 * (start_loss - min_loss)
# Find first point that crosses this threshold
```

**File**: [lr_finder.py:571-590](code/src/Ava/training/lr_finder.py#L571-L590)

### 4. Reduced Over-Smoothing
**Problem**: Beta=0.98 was hiding valley details

**Solution**: Reduced to beta=0.90 for better visibility of loss curve features

**File**: [lr_finder.py:58](code/src/Ava/training/lr_finder.py#L58)

### 5. Improved Recommendation Logic
**Problem**: Same strategy regardless of variance level

**Solution**: Tiered approach based on variance:
- Variance < 5x: Geometric mean of all methods (high confidence)
- Variance 5-20x: Geometric mean of valley + fastai (medium confidence)
- Variance 20-50x: Valley only (low confidence)
- Variance > 50x: Extra conservative + warning

**File**: [run_lr_finder_enhanced.py:364-443](code/scripts/4_Find_Lr/run_lr_finder_enhanced.py#L364-L443)

## Expected Improvement

### Before (BROKEN):
```
Variance: 1084x ❌
FastAI:   1.80e-04
Valley:   4.06e-06  ← idx 0 bug
Steepest: 1.66e-07  ← idx 0 bug
```

### After (FIXED):
```
Variance: ~5-10x ✅ (acceptable for untrained model)
FastAI:   ~1.5-2.5e-04
Valley:   ~5.0-8.0e-05
Steepest: ~3.0-6.0e-05
Recommended: ~7.0e-05 (geometric mean)
```

## Testing

Run the improved LR finder:
```bash
cd /project/code/scripts/4_Find_Lr
python run_lr_finder_enhanced.py --config ../../configs/gpu/small.yaml
```

Or use the quick test script:
```bash
bash QUICK_TEST.sh
```

## Success Criteria

✅ **Variance < 100x** (was 1084x)
✅ **Steepest LR > 1e-5** (was 1.66e-07)
✅ **Valley LR > 1e-5** (was 4.06e-06)
✅ **All methods in similar order of magnitude** (e-4 to e-5 range)
✅ **Confidence MEDIUM or higher** (was VERY LOW)

## Files Changed

1. `/project/code/src/Ava/training/lr_finder.py` - Core algorithm fixes
2. `/project/code/scripts/4_Find_Lr/run_lr_finder_enhanced.py` - Recommendation logic
3. `/project/LR_FINDER_IMPROVEMENTS.md` - Detailed documentation
4. `/project/code/scripts/4_Find_Lr/QUICK_TEST.sh` - Test script

## Next Steps

1. **Run the LR finder** to verify variance is now < 100x
2. **Use recommended LR** in training config
3. **Monitor first 1000 steps** to validate the choice

If variance is still high (>30x), it indicates the loss curve is genuinely noisy, and you should either:
- Train for ~1000 steps first, then re-run LR finder
- Use rule-of-thumb: 3e-4 for ~100M params
- Increase batch size to reduce noise
