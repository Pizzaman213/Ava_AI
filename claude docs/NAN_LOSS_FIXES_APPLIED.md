# NaN/Inf Loss Fixes Applied

## Problem Summary

Training was experiencing **100% NaN/Inf loss** from the very first step, causing all optimizer steps to be skipped. The model was unable to learn anything.

## Root Causes Identified

1. **Learning Rate Too High**: 0.0002 (2e-4) was too aggressive for MoE models with multiple auxiliary losses
2. **Router Loss Coefficients Too High**: Load balance loss coef of 0.05 was causing loss explosions
3. **Weight Decay Too High**: 0.1 was over-regularizing the model
4. **Model Initialization Range Too Large**: 0.02 was causing initial instability
5. **Mixed Precision Issues**: BF16 can cause numerical instability with complex loss functions
6. **Too Many Auxiliary Losses**: Multiple penalty losses were compounding instability
7. **Aggressive Gradient Clipping**: max_grad_norm of 1.0 wasn't aggressive enough for early training

## Fixes Applied to `tiny_moe_ultra_low_mem.yaml`

### 1. Reduced Learning Rate
```yaml
# Before: learning_rate: 0.0002
# After:  learning_rate: 0.00005  (5e-5)
```
**Rationale**: MoE models require lower learning rates due to multiple expert gradients and auxiliary losses.

### 2. Reduced Weight Decay
```yaml
# Before: weight_decay: 0.1
# After:  weight_decay: 0.01
```
**Rationale**: High weight decay can cause gradient instability, especially early in training.

### 3. More Aggressive Gradient Clipping
```yaml
# Before: max_grad_norm: 1.0
# After:  max_grad_norm: 0.5
```
**Rationale**: Tighter clipping prevents gradient explosions during unstable early training.

### 4. Reduced Router Loss Coefficients
```yaml
# Before:
  router_z_loss_coef: 0.001
  load_balance_loss_coef: 0.05
  diversity_loss_coef: 0.001

# After:
  router_z_loss_coef: 0.0001
  load_balance_loss_coef: 0.01
  diversity_loss_coef: 0.0001
```
**Rationale**: Router auxiliary losses were contributing too much to total loss, causing instability.

### 5. Reduced Model Initialization Range
```yaml
# Before: initializer_range: 0.02
# After:  initializer_range: 0.01
```
**Rationale**: Smaller initialization prevents extreme initial activations that can cause overflow.

### 6. Disabled Adaptive Temperature and Reduced Label Smoothing
```yaml
# Before:
  adaptive_temperature: true
  label_smoothing: 0.1
  eos_penalty_weight: 0.1

# After:
  adaptive_temperature: false
  label_smoothing: 0.05
  eos_penalty_weight: 0.05
```
**Rationale**: Simpler loss configuration during initial training for stability.

### 7. Disabled Auxiliary Penalty Losses
```yaml
# Before:
  use_ngram_penalty: true
  use_immediate_repetition_detector: true
  use_diversity_loss: true

# After:
  use_ngram_penalty: false
  use_immediate_repetition_detector: false
  use_diversity_loss: false
```
**Rationale**: Multiple penalty losses compound numerical instability. Enable these after stable training.

### 8. Switched to FP32 Precision
```yaml
# Before: mixed_precision: bf16
# After:  mixed_precision: fp32
```
**Rationale**: BF16 can cause underflow/overflow with complex loss functions. Use FP32 initially, switch to BF16 after stability is confirmed.

## Next Steps

### Immediate Testing
1. Run training with the fixed configuration
2. Verify loss is finite and decreasing
3. Monitor for the first 1000 steps

### Gradual Re-enablement (After Stable Training)
Once training is stable for 5000+ steps:

1. **Switch to BF16** (if memory allows, for faster training)
   ```yaml
   mixed_precision: bf16
   ```

2. **Re-enable auxiliary losses gradually** (one at a time, monitor stability):
   ```yaml
   use_diversity_loss: true      # Enable first
   use_ngram_penalty: true       # Enable second
   use_immediate_repetition_detector: true  # Enable last
   ```

3. **Increase router loss coefficients slightly** (if expert utilization is poor):
   ```yaml
   load_balance_loss_coef: 0.02  # Gradually increase from 0.01
   ```

4. **Re-enable adaptive temperature** (for better convergence):
   ```yaml
   adaptive_temperature: true
   ```

## Testing Commands

Test with the fixed configuration:
```bash
python code/scripts/5_training/train.py --config code/configs/moe/tiny_moe_ultra_low_mem.yaml
```

Monitor the training logs for:
- Loss should be finite (not NaN/Inf)
- Loss should decrease over time
- No "CRITICAL: NaN or Inf loss detected" messages
- Gradient norms should be reasonable (< 10.0)

## Expected Behavior

**Before Fixes:**
- Loss: inf from step 1
- All optimizer steps skipped
- No learning

**After Fixes:**
- Loss: ~8-10 initially, decreasing gradually
- Normal optimizer steps
- Model learns successfully

## Configuration Files Modified

1. `/project/code/configs/moe/tiny_moe_ultra_low_mem.yaml` - Primary configuration with all stability fixes

## Monitoring Recommendations

Watch these metrics during training:
- **Loss**: Should start around 8-10 and decrease
- **Gradient Norm**: Should be < 5.0 after clipping
- **Learning Rate**: Should warm up from 0 to 5e-5 over 5000 steps
- **Expert Utilization**: Should be relatively balanced (check MoE metrics)

## Rollback Plan

If issues persist, try even more conservative settings:
```yaml
learning_rate: 0.00001  # Even lower (1e-5)
max_grad_norm: 0.3      # Even more aggressive clipping
use_deepseek_loss: false  # Fall back to simple cross-entropy
```

---

**Status**: Fixes applied and ready for testing
**Date**: 2025-11-10
**Impact**: Should eliminate NaN/Inf losses and enable stable training
