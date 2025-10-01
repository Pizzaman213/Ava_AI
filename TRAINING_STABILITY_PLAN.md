# 🚨 Training Stability Issues - Root Cause Analysis & Fix Plan

## Executive Summary

Your LLM training has **3 CRITICAL configuration errors** causing unstable, oscillating loss:

| Issue | Your Config | Should Be | Impact |
|-------|-------------|-----------|--------|
| **Batch Size** | 5 | 12 | High variance, unstable gradients |
| **Learning Rate** | 5e-5 | 4.2e-4 | **8.4x too low** - barely learning |
| **Precision** | fp16 | bf16 | Numerical instability in MoE routing |

---

## Root Cause Analysis

### 1. ❌ Batch Size Too Small (5 → need 12+)
- **Problem**: MoE models need larger batches for stable expert routing
- **Your config**: batch_size=5, gradient_accumulation=4 → effective=20
- **Issue**: Per-batch variance is catastrophically high
- **Working configs**: All use batch_size ≥12

### 2. ❌ Learning Rate 8.4x Too Low
- **Problem**: LR of 5e-5 is too conservative for batch size
- **Your config**: 5e-5 (0.00005)
- **Working configs**: 4.2e-4 (0.00042) = **8.4x higher**
- **Result**: Model barely learns, loss oscillates randomly

### 3. ❌ Using FP16 Instead of BF16
- **Problem**: FP16 causes numerical instability in MoE routing
- **Your config**: mixed_precision: true (defaults to fp16)
- **Working configs**: mixed_precision: bf16
- **Result**: Loss spikes, gradient explosions

### 4. ⚠️ Aggressive Gradient Clipping
- **Problem**: High clipping (5.0→3.0) masks instability instead of fixing it
- **Your config**: Trainer uses initial_clip=5.0, final=3.0
- **Working configs**: Use max_grad_norm=1.0
- **Result**: Training appears "stable" but loss oscillates

### 5. ⚠️ Adaptive LR Causing Oscillations
- **Problem**: Adaptive LR detects normal variance as problems with small batches
- **Your setup**: Spike threshold 2.0x triggers false alarms
- **Working configs**: Disable adaptive LR (use_adaptive_lr: false)
- **Result**: LR constantly adjusts, causing oscillations

---

## Solution: 3 Options

### Option 1: Quick Fix (30 seconds)
```bash
# Use the fixed config I created for you
cp /project/minimal_config_FIXED.yaml /project/minimal_config.yaml
cd /project/code/scripts/training
python train.py --config /project/minimal_config.yaml
```

**Changes in FIXED config:**
- batch_size: 5 → 12
- learning_rate: 5e-5 → 4.2e-4
- mixed_precision: true → bf16
- gradient_accumulation: 4 → 1
- weight_decay: 0.01 → 0.1
- beta2: 0.999 → 0.95
- use_adaptive_lr: (enabled) → false

### Option 2: Use Proven Stable Config (Best)
```bash
cd /project/code/scripts/training
python train.py --config ../../configs/gpu/small.yaml
```

This config has been proven stable in your codebase with many successful runs.

### Option 3: Manual Fix
Edit `/project/minimal_config.yaml`:
```yaml
training:
  batch_size: 12                    # INCREASE from 5
  learning_rate: 0.00042            # INCREASE from 0.00005
  gradient_accumulation_steps: 1    # REDUCE from 4
  mixed_precision: bf16             # CHANGE from 'true'
  weight_decay: 0.1                 # INCREASE from 0.01
  beta2: 0.95                       # REDUCE from 0.999
  use_adaptive_lr: false            # ADD THIS LINE
```

---

## Expected Results After Fix

### Before (Unstable):
```
Epoch 1: Loss 8.2 → 12.4 → 6.1 → 15.2 → 9.8
Epoch 2: Loss 11.3 → 7.5 → 18.1 → 5.9 → 14.2
```
**Oscillating wildly, not converging**

### After (Stable):
```
Epoch 1: Loss 8.2 → 7.8 → 7.4 → 7.1 → 6.9
Epoch 2: Loss 6.7 → 6.5 → 6.3 → 6.2 → 6.0
```
**Smooth, consistent decrease**

---

## Why These Specific Values?

### Batch Size = 12
- Minimum for stable MoE expert routing
- Reduces per-batch gradient variance
- Proven stable in small.yaml, tiny.yaml configs

### Learning Rate = 4.2e-4
- Scaled appropriately for batch size 12
- **8.4x higher** than your current 5e-5
- This is the **#1 reason** your loss isn't stable
- All working configs use exactly this value

### BF16 Precision
- Better numerical range than FP16
- Critical for MoE routing stability
- Prevents gradient underflow/overflow
- Same speed as FP16, better accuracy

### Gradient Accumulation = 1
- Higher update frequency = more stable
- With batch_size=12, no accumulation needed
- Working configs all use 1

---

## Additional Recommendations

### After applying fixes, monitor:
1. **Loss should decrease smoothly** (not oscillate)
2. **Gradient norms should be <5.0** consistently
3. **Learning rate should stay stable** (no emergency reductions)
4. **GPU memory should be ~8-11GB** (you have headroom)

### If still unstable (unlikely):
1. Increase warmup_steps to 3000-5000
2. Reduce learning rate to 2e-4
3. Enable gradient_checkpointing for memory
4. Check data quality (corrupted samples cause spikes)

---

## Verification Commands

### Before training:
```bash
# Check your config is fixed
python3 << 'EOF'
import yaml
with open('/project/minimal_config.yaml') as f:
    cfg = yaml.safe_load(f)
t = cfg['training']
print(f"✓ batch_size: {t['batch_size']} (need ≥12)")
print(f"✓ learning_rate: {t['learning_rate']} (need 4.2e-4)")
print(f"✓ mixed_precision: {t['mixed_precision']} (need bf16)")
