# Adaptive Learning Rate Fix Applied

## Date: 2025-11-10

## Problem Identified

Your training was suffering from **overly aggressive adaptive learning rate reductions** that caused the LR to drop from `1e-4` to `3.91e-7` (256x reduction) within just 1000 optimizer steps. This prevented the model from learning effectively.

### Root Causes

1. **Adaptive LR Manager too aggressive:**
   - `plateau_patience: 500` batches (only 62.5 optimizer steps with gradient_accumulation=8)
   - `plateau_factor: 0.5` (halved LR on each plateau)
   - `emergency_factor: 0.5` (halved LR on loss spikes)

2. **Loss stuck at 10.90:**
   - Model hadn't learned the task yet (perplexity ~54,000)
   - Needed stable LR to learn basic patterns
   - LR kept getting reduced before model could improve

3. **Symptoms:**
   - Gibberish generation outputs
   - No loss improvement over 1000 steps
   - LR dropped 8 times via 0.5x reductions

## Solution Applied

### Configuration Changes in `tiny_moe_ultra_low_mem.yaml`

Added a new `adaptive_lr` section with **much more patient settings**:

```yaml
adaptive_lr:
  # Plateau detection - when to reduce LR
  plateau_patience: 5000          # ↑ Increased from 500 to 5000 batches (10x more patient)
  plateau_factor: 0.7             # ↑ Reduced from 0.5 to 0.7 (30% reduction vs 50%)
  lr_check_interval: 500          # ↑ Increased from 50 to 500 (check 10x less frequently)

  # LR boundaries
  min_lr: 1e-6                    # ↑ Increased from 1e-7 (10x higher minimum)
  max_lr: 2e-4                    # Maximum LR (2x the initial LR)

  # Improvement detection
  min_improvement: 0.001          # Minimum loss improvement to count as progress
  batch_loss_window: 100          # Window size for averaging loss

  # Stability-based increases
  stability_threshold: 5          # Consecutive improvements needed before increasing LR
  increase_factor: 1.05           # Factor to increase LR when stable (5% increase)
  increase_min_gap: 1000          # Minimum steps between LR increases

  # Emergency spike handling
  divergence_threshold: 3.0       # Loss spike threshold (3x recent best)
  emergency_factor: 0.7           # ↑ Increased from 0.5 to 0.7 (less aggressive)
```

### Key Changes Summary

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| `plateau_patience` | 500 batches | 5000 batches | **10x more patient** before reducing LR |
| `plateau_factor` | 0.5 (50% cut) | 0.7 (30% cut) | **Gentler reductions** |
| `lr_check_interval` | 50 batches | 500 batches | **Check 10x less often** |
| `min_lr` | 1e-7 | 1e-6 | **10x higher floor** for LR |
| `emergency_factor` | 0.5 (50% cut) | 0.7 (30% cut) | **Less aggressive** emergency cuts |

## Expected Results

With these changes, when you restart training:

1. **LR will stay stable longer:**
   - With `plateau_patience=5000` batches (625 optimizer steps), the LR will only reduce after 625 steps of no improvement
   - Previously it was reducing every 62.5 steps!

2. **Gentler reductions:**
   - LR reductions will be 0.7x (30% cut) instead of 0.5x (50% cut)
   - This gives the model more chances to recover

3. **Higher minimum LR:**
   - LR will never drop below 1e-6 (vs 1e-7 before)
   - Ensures meaningful learning can continue

4. **Expected training progression:**
   - **Steps 0-5000:** Warmup from 1e-8 to 1e-4
   - **Steps 5000-10000:** Loss should drop from ~11 to ~9 (if learning properly)
   - **Steps 10000-20000:** Loss should drop to ~7-8 (coherent words start appearing)
   - **Steps 20000-30000:** Loss should drop to ~6-7 (coherent sentences)
   - **Steps 30000-50000:** Fine-tuning to final loss ~5-6

## How to Restart Training

Since your config has `fresh_start: true`, simply restart training:

```bash
python code/scripts/5_training/train.py \
  --config code/configs/moe/tiny_moe_ultra_low_mem.yaml
```

The training will:
- ✅ Start with fresh LR = 1e-4
- ✅ Use the new adaptive_lr settings
- ✅ Be much more patient before reducing LR
- ✅ Give the model time to learn

## Monitoring Recommendations

Watch for these signs of healthy training:

### Good signs ✅
- Loss drops below 9.0 within first 10,000 steps
- LR stays at 1e-4 for first 5,000+ steps
- Generation samples start showing real words (not gibberish) around step 15,000
- LR reductions happen slowly (every 5,000+ steps, not every 500)

### Warning signs ⚠️
- Loss still at 10+ after 10,000 steps → may need to increase initial LR
- LR dropping rapidly again → check if plateau_patience is being read correctly
- Lots of "emergency reduction" messages → model may have other issues

## Files Modified

- [tiny_moe_ultra_low_mem.yaml](code/configs/moe/tiny_moe_ultra_low_mem.yaml) - Added `adaptive_lr` section with less aggressive settings

## Technical Details

### Why the Previous Settings Were Too Aggressive

With `gradient_accumulation_steps=8`:
- Every 8 batches = 1 optimizer step
- `plateau_patience=500` batches = 62.5 optimizer steps
- At loss of 10.90 (very early in training), the model had barely started learning
- The adaptive LR manager was reducing LR every 62.5 optimizer steps when loss didn't improve
- After 8 reductions: 1e-4 → 3.9e-7 (too small to learn anything)

### Why the New Settings Are Better

With the new settings:
- `plateau_patience=5000` batches = 625 optimizer steps (10x more patient)
- `plateau_factor=0.7` = only 30% reduction (vs 50%)
- `min_lr=1e-6` = 10x higher minimum (prevents microscopic LRs)
- This gives the model ~5,000-10,000 steps at healthy LR before first reduction

## Next Steps

1. ✅ Stop current training (wasting compute at LR=3.91e-7)
2. ✅ Restart training with new config
3. Monitor loss for first 10,000 steps
4. If loss drops below 9.0 → training is working!
5. If loss stays at 10+ → may need further adjustments

## References

- Training logs showed LR at 3.91e-7 at step 1000 (optimizer_step=1000)
- Config showed initial_lr=1e-4
- Calculated ~8 halvings occurred (1e-4 * 0.5^8 ≈ 3.9e-7)
- AdaptiveLearningRateManager source: `/project/code/src/Ava/optimization/learning_rate/managers.py`
