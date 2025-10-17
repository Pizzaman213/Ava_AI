# Training Issues Diagnosed & Fixed

**Date:** 2025-10-15
**Issue:** Model stuck at 93.8% repetition, training collapsing at step 18000+

## Problems Found

### 1. **CRITICAL: Inverted LR Schedule** ❌
```yaml
# WRONG (your config had this bug):
learning_rate: 1.5e-06  # Starting LR
lr_end: 1.0e-05         # Ending LR (HIGHER than start!)

# FIXED:
learning_rate: 3.0e-04  # Proper starting LR
lr_end: 1.0e-06         # Must be LOWER than start
```

**Impact:** The cosine scheduler was trying to increase LR instead of decrease it, causing the LR to stay frozen at ~1.58e-06. Model couldn't learn.

**Root cause:** The LR finder script was setting `lr_end` to 40% of peak LR, which is way too high. Now it sets `lr_end = peak_lr / 300`.

---

### 2. **Learning Rate Too Low** ⚠️
```yaml
# Before: 1.5e-06 (200x too low!)
# After:  3.0e-04 (proper for your model size)
```

At 1.58e-06, gradient updates were so small the model effectively stopped learning.

---

### 3. **Excessive Penalty Weights** ⚠️
Your penalties were fighting each other:

```yaml
# BEFORE (caused repetition collapse):
eos_penalty_weight: 3.0            # Prevented model from ending
eos_logit_bias: 2.0                # Forced longer sequences
repetition_penalty_weight: 1.2     # Then penalized repetition
immediate_repetition_weight: 2.0   # Creating a conflict!
ngram_penalty_weight: 2.0          # Even more penalties

# AFTER (let model learn naturally):
eos_penalty_weight: 0.0            # Disabled
eos_logit_bias: 0.0                # Disabled
repetition_penalty_weight: 1.0     # Minimal
immediate_repetition_weight: 0.5   # Very gentle
use_ngram_penalty: false           # Disabled
```

**Why this matters:** The model was penalized for ending sequences (EOS) AND penalized for repetition. It chose repetition as the lesser evil, causing 93.8% repetition rate.

---

### 4. **Short Warmup Period** ⚠️
```yaml
# Before: 200 steps
# After:  1000 steps
```

200 steps is too short for stable warmup. Model can hit instabilities early.

---

## Fixes Applied

### Automatic Fixes

1. **[small.yaml](code/configs/gpu/small.yaml)** - Manually fixed all issues above
2. **[run_lr_finder_enhanced.py](code/scripts/4_Find_Lr/run_lr_finder_enhanced.py)** - Fixed to set proper `lr_end = peak_lr / 300`
3. **[validate_config.py](code/scripts/validation/validate_config.py)** - NEW script to catch these bugs automatically

### New Tools Created

#### 1. Config Validator (Catches Bugs Automatically)
```bash
# Validate config
python code/scripts/validation/validate_config.py --config code/configs/gpu/small.yaml

# Validate + Auto-fix
python code/scripts/validation/validate_config.py --config code/configs/gpu/small.yaml --fix

# Quick script
bash VALIDATE_AND_FIX_CONFIG.sh
```

**What it checks:**
- ✅ LR schedule validity (lr_end < learning_rate)
- ✅ Excessive penalty weights
- ✅ Short warmup periods
- ✅ Invalid batch sizes
- ✅ Missing gradient clipping
- ✅ Data paths exist

#### 2. Enhanced LR Finder (Now Sets Proper lr_end)
```bash
# Find optimal LR and auto-update config
cd /project/code
python scripts/4_Find_Lr/run_lr_finder_enhanced.py --config configs/gpu/small.yaml

# Quick script (recommended)
bash /project/QUICK_FIND_LR.sh
```

**What it does:**
- Runs 3 methods (fastai, valley, steepest) for cross-validation
- Calculates geometric mean for final recommendation
- **VALIDATES lr_end < learning_rate** before saving
- Sets `lr_end = peak_lr / 300` (safe for cosine decay)
- Creates timestamped backup

---

## What To Do Now

### Option A: Clean Restart (Recommended) ✅

Your current checkpoint is trained with broken config. Starting fresh is faster:

```bash
# 1. Clean old outputs
cd /project/code
rm -rf outputs/small_enhanced/run_*

# 2. Optionally: Find optimal LR first
bash /project/QUICK_FIND_LR.sh

# 3. Start training with fixed config
python train.py --config configs/gpu/small.yaml
```

**Expected results after ~2000 steps:**
- ✅ LR decreases smoothly from 3e-4 to 1e-6
- ✅ Loss drops consistently
- ✅ Repetition rate < 20%
- ✅ Average length > 100 tokens
- ✅ No mixed precision resets

---

### Option B: Continue Training (Not Recommended)

The model has learned bad patterns (93.8% repetition). Even with fixed config, it will struggle to unlearn:

```bash
cd /project/code
python train.py --config configs/gpu/small.yaml --resume
```

**Expect:** Slow improvement, may still generate repetitive text for thousands of steps.

---

## Validation Before Every Training Run

**NEW WORKFLOW** (prevents these bugs in future):

```bash
# Step 1: Validate config
bash VALIDATE_AND_FIX_CONFIG.sh

# Step 2: (Optional) Find optimal LR
bash QUICK_FIND_LR.sh

# Step 3: Start training
cd /project/code
python train.py --config configs/gpu/small.yaml
```

---

## Key Lessons

1. **Always validate `lr_end < learning_rate`** for cosine scheduler
2. **Less is more** with penalty weights - let model learn naturally
3. **Warm up properly** - 1000+ steps for stability
4. **Monitor early** - If repetition > 30% by step 5000, something is wrong
5. **Use the validator** - Catches bugs before wasting GPU hours

---

## Files Changed

### Fixed
- [code/configs/gpu/small.yaml](code/configs/gpu/small.yaml) - Fixed LR schedule and penalties
- [code/scripts/4_Find_Lr/run_lr_finder_enhanced.py](code/scripts/4_Find_Lr/run_lr_finder_enhanced.py) - Now sets proper lr_end

### Created
- [code/scripts/validation/validate_config.py](code/scripts/validation/validate_config.py) - Config validator
- [QUICK_FIND_LR.sh](QUICK_FIND_LR.sh) - One-command LR optimization
- [VALIDATE_AND_FIX_CONFIG.sh](VALIDATE_AND_FIX_CONFIG.sh) - One-command validation

---

## Summary

**Root cause:** Inverted LR schedule (lr_end > learning_rate) + excessive penalties = training collapse

**Fix:** Proper LR schedule + minimal penalties + validation tooling

**Next step:** Clean restart with fixed config OR continue training (not recommended)

**Prevention:** Always run `bash VALIDATE_AND_FIX_CONFIG.sh` before training
