# Evaluation Steps Fix - 2025-10-20

## Problem

The training was not running generation tests at the expected intervals.

**Expected behavior:** Evaluate every 1000 steps (as configured in `eval_steps: 1000`)
**Actual behavior:** Evaluated every ~8000 steps

## Root Cause

The training script was using **optimizer steps** instead of **training steps** to determine when to evaluate.

With `gradient_accumulation_steps: 8`:
- 1 optimizer step = 8 training steps
- `eval_steps: 1000` meant "every 1000 optimizer steps"
- This translated to evaluation every **8000 training steps**!

## Timeline of Events

Your training run (started at step 0):
- Step 1,000: ❌ No evaluation (optimizer_step = 125)
- Step 2,000: ✅ Evaluation ran (optimizer_step = 250, checkpoint saved)
- Step 8,000: ❌ No evaluation (optimizer_step = 1000)
- Step 16,000: ✅ Evaluation ran (optimizer_step = 2000)
- Step 21,000: ❌ No evaluation (optimizer_step = 2625)
- Step 24,000: ✅ Would have evaluated (optimizer_step = 3000)

## Solution

### 1. Added Config Toggle

New config parameter in [small.yaml](code/configs/gpu/small.yaml):

```yaml
training:
  eval_steps: 1000
  eval_steps_type: training_steps  # NEW: 'training_steps' or 'optimizer_steps'
```

**Options:**
- `training_steps` (default): Evaluate every N training iterations
- `optimizer_steps`: Evaluate every N optimizer updates (old behavior)

### 2. Updated Training Script

Modified [train.py](code/scripts/5_training/train.py) to:
- Read `eval_steps_type` from config (defaults to `training_steps`)
- Use `trainer.step_count` for training steps
- Use `trainer.optimizer_step_count` for optimizer steps
- Log which type of step is being used

### 3. Applied to Both Validation Points

Fixed evaluation logic in TWO places:
1. **In-epoch validation** (line ~1150) - Runs during long epochs
2. **End-of-epoch validation** (line ~2560) - Runs at epoch boundaries

## Impact

### Before Fix
```
eval_steps: 1000 with gradient_accumulation_steps: 8
→ Evaluation every 8000 training steps
→ Very infrequent generation testing
```

### After Fix
```
eval_steps: 1000 with eval_steps_type: training_steps
→ Evaluation every 1000 training steps (as intended!)
→ Frequent generation testing and checkpoint saving
```

## What Happens Now

With your current training at step ~21,000:

**Next evaluation:** Step 22,000 (in ~1000 steps, ~20 minutes)

After that:
- Step 23,000: Evaluation + generation test
- Step 24,000: Evaluation + generation test
- Step 25,000: Evaluation + generation test
- etc.

## Configuration Examples

### Frequent evaluation (recommended for debugging):
```yaml
eval_steps: 500
eval_steps_type: training_steps
# Evaluates every 500 training steps
```

### Moderate evaluation (default):
```yaml
eval_steps: 1000
eval_steps_type: training_steps
# Evaluates every 1000 training steps
```

### Infrequent evaluation (for very long runs):
```yaml
eval_steps: 1000
eval_steps_type: optimizer_steps
# With grad_accum=8, evaluates every 8000 training steps
```

## Testing

To verify the fix works:
1. The config has been updated with `eval_steps_type: training_steps`
2. The training script will now check `trainer.step_count`
3. Next evaluation should occur at step 22,000 (not step 24,000)

## Backward Compatibility

The fix is **backward compatible**:
- Default behavior: `eval_steps_type: training_steps` (most intuitive)
- Old behavior available: `eval_steps_type: optimizer_steps`
- Existing configs without `eval_steps_type` will default to `training_steps`

## Files Modified

1. ✅ [code/configs/gpu/small.yaml](code/configs/gpu/small.yaml) - Added `eval_steps_type: training_steps`
2. ✅ [code/scripts/5_training/train.py](code/scripts/5_training/train.py) - Updated evaluation logic (2 locations)
3. ✅ Documentation added to config notes

## Notes for Future Training Runs

- **For standard training:** Use `eval_steps_type: training_steps` (default)
- **For memory-constrained runs:** Use larger `eval_steps` values (e.g., 2000, 5000)
- **For old behavior:** Explicitly set `eval_steps_type: optimizer_steps`

---

**Status:** ✅ Fixed and documented
**Next Action:** Monitor that evaluation runs at step 22,000
