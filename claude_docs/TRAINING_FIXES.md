# 🚨 Critical Training Fixes for Ava 100M Model

## Problem: Gibberish Generation After 100k Steps

**Root Cause:** Learning rate = 0.0001 is far too low for training from scratch

---

## ✅ FIX #1: Increase Learning Rate (CRITICAL)

### Current (BROKEN):
```bash
LR = 0.0001   # 10-100x TOO LOW
```

### Fixed (REQUIRED):
```bash
--learning-rate 0.006   # 60x higher - GPT-2 124M uses this
```

### Why:
- **0.0001 = fine-tuning rate** (for pre-trained models)
- **0.003-0.006 = pre-training rate** (for random initialization)
- Your model weights barely moved in 100k steps!

---

## ✅ FIX #2: Add LR Warmup Schedule

```bash
python train.py \
  --config configs/gpu/small.yaml \
  --learning-rate 0.006 \
  --warmup-steps 2000 \
  --lr-scheduler cosine \
  --min-lr 1e-5
```

Schedule: 0 → 0.006 (warmup) → 0.00001 (decay)

---

## ✅ FIX #3: Increase Batch Size

```bash
--batch-size 32 \
--gradient-accumulation 4   # Effective batch = 128
```

Larger batches = smoother gradients = faster learning

---

## ✅ FIX #4: Better Generation Sampling

During eval, use:
```python
temperature = 0.7   # Not 0.5 or 1.0
top_k = 40
top_p = 0.95
```

---

## 🚀 COMPLETE FIX (Copy This):

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --learning-rate 0.006 \
  --batch-size 32 \
  --gradient-accumulation 4 \
  --warmup-steps 2000 \
  --lr-scheduler cosine \
  --gradient-clip-norm 1.0 \
  --use-wandb \
  --fresh-start
```

---

## 📊 What to Expect:

| Steps | Loss | Output Quality |
|---|---|---|
| 0 | 10.5 | Random gibberish |
| 1,000 | 6.5 | Some word structure |
| 5,000 | 4.2 | Basic sentences ✓ |
| 10,000 | 3.5 | Coherent text ✓✓ |
| 100,000 | 2.5 | High quality ✓✓✓ |

With LR=0.0001: Loss stays ~10.0 forever (NO LEARNING)

---

## ✅ Verify After 1000 Steps:

```bash
# Check loss dropped
grep "step 1000" outputs/runs/*/logs/training.log
# Should show: loss = 6.5 or lower

# Test generation
python scripts/generation/generate.py --prompt "The quick brown fox" --temperature 0.7
# Should produce: actual words (may be weird but word-like)
```

If loss still ~10.0 → increase LR to 0.01
If loss = NaN → decrease LR to 0.003 and add --gradient-clip-norm 0.5
