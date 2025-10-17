# Quick Start - Fixed Training

Your training was failing due to **inverted LR schedule** + **excessive penalties**.

## ✅ All Fixed! Here's What Changed:

### Before (BROKEN ❌)
```yaml
learning_rate: 1.5e-06  ← Too low
lr_end: 1.0e-05         ← HIGHER than start (BUG!)
eos_penalty_weight: 3.0
repetition_penalty_weight: 1.2
warmup_steps: 200
```
**Result:** 93.8% repetition, stuck at step 18000

### After (FIXED ✅)
```yaml
learning_rate: 3.0e-04  ← Proper starting LR
lr_end: 1.0e-06         ← LOWER than start (correct!)
eos_penalty_weight: 0.0
repetition_penalty_weight: 1.0
warmup_steps: 1000
```
**Expected:** <20% repetition, smooth training

---

## 🚀 Start Training (2 Options)

### Option A: Clean Restart (Recommended)
```bash
cd /project/code
rm -rf outputs/small_enhanced/run_*
python train.py --config configs/gpu/small.yaml
```

### Option B: Continue from Checkpoint
```bash
cd /project/code
python train.py --config configs/gpu/small.yaml --resume
```
⚠️ Not recommended - model learned bad patterns (93.8% repetition)

---

## 🔧 New Tools Available

### 1. Validate Config (Run Before Every Training!)
```bash
bash /project/VALIDATE_AND_FIX_CONFIG.sh
```

### 2. Find Optimal LR
```bash
bash /project/QUICK_FIND_LR.sh
```

---

## 📊 What To Expect

After fix, by step 5000 you should see:
- ✅ LR: ~2.8e-04 (decreasing smoothly)
- ✅ Loss: ~2.5-3.0 (dropping)
- ✅ Repetition: <20%
- ✅ Avg length: >100 tokens

If repetition > 30% by step 5000, **STOP** and run validator.

---

## 📖 Full Details
See [TRAINING_FIXES_APPLIED.md](TRAINING_FIXES_APPLIED.md) for complete analysis.
