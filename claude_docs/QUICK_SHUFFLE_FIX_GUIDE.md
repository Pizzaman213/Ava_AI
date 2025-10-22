# Quick Shuffle Fix & Loss Issue - Troubleshooting Guide

## TL;DR - What Changed?

### Three Key Changes Made:
1. **Epoch-aware random seeds** for shuffling (not fixed seed 42 every time)
2. **Smaller buffer** (10k instead of 50k) → more reshuffles → better variety
3. **Faster file rotation** (1 sample per file instead of 32) → maximum diversity

### Expected Result:
- ✅ Training loss decreases more **steadily** (not cliff-like)
- ✅ Validation loss **doesn't diverge** as much
- ✅ Model generalizes better to **new data**

---

## Symptoms Checklist: Is Your Loss Problem Fixed?

### Before Improvements ❌
```
Training loss: 3.5 → 2.1 → 1.8 → 1.5 → 1.2 (very steep cliff)
Validation loss: 3.6 → 2.8 → 3.2 → 4.1 → 5.8 (diverging!)
Problem: Model is memorizing the training shuffle order
```

### After Improvements ✅
```
Training loss: 3.5 → 2.8 → 2.4 → 2.1 → 1.9 (steady curve)
Validation loss: 3.6 → 2.9 → 2.5 → 2.2 → 2.0 (follows training)
Good: Model learning actual patterns, not shuffle artifacts
```

---

## How to Verify Improvements are Working

### Option 1: Quick Visual Check (5 minutes)
```bash
# Run 10 steps with new code
python train.py --config configs/gpu/small.yaml \
    --max-steps 10 \
    --logging-steps 1

# Look for in the output:
# ✓ "Shuffle seed = 42" (epoch 0)
# ✓ "Shuffle seed = 43" (epoch 1) - different!
# ✓ Buffer shuffle happening with different seeds
```

### Option 2: TensorBoard/Weights & Biases Check (15 minutes)
```bash
# Run with monitoring enabled
python train.py --config configs/gpu/small.yaml \
    --enable-observability \
    --wandb-project test-improvements

# Check the loss curve:
# - Should see smooth curve, not steep cliff at start
# - Validation should follow training (not diverge)
# - Per-epoch improvement should be decreasing (convergence signal)
```

### Option 3: Log Analysis (30 minutes)
```bash
# Run full training and analyze logs
python train.py --config configs/gpu/small.yaml --num-epochs 3

# Calculate epoch variance in loss:
grep "loss" training.log | awk '{print $NF}' > loss_values.txt

# If shuffling is working:
# - Loss differences between epochs should be smooth
# - No sudden jumps (which indicate memorization)
# - Each epoch loss starts where previous left off
```

---

## Configuration Troubleshooting

### Problem: Still seeing very fast loss drop?

**Check 1: Is buffer_size correct?**
```yaml
# BAD (old)
buffer_size: 50000

# GOOD (new)
buffer_size: 10000
```

**Check 2: Is samples_per_file set correctly?**
```yaml
# BAD (old)
samples_per_file: 32

# GOOD (new)
samples_per_file: 1
```

**Check 3: Verify epoch seed is being used**
Look for in logs:
```
"Shuffle seed = 42" (epoch 0)
"Shuffle seed = 43" (epoch 1)
```
If you see "Shuffle seed = 42" repeatedly → OLD CODE

---

## What Each Change Does

### Change 1: Epoch-Aware Shuffle Seed
**File**: `src/Ava/data_streaming.py` (lines 720, 761)

**Before**:
```python
random.Random(42).shuffle(buffer)  # Same every epoch!
```

**After**:
```python
buffer_seed = 42 + epoch_number + (samples_processed // buffer_size)
random.Random(buffer_seed).shuffle(buffer)  # Different each epoch
```

**Effect**:
- Prevents model from learning shuffle order patterns
- More randomness = harder for model to overfit

### Change 2: Smaller Buffer (50k → 10k)
**File**: `src/Ava/data_streaming.py` (line 822) & `configs/gpu/small.yaml` (line 252)

**Before**:
```python
buffer_size: int = 50000  # ONE shuffle per 50k samples
```

**After**:
```python
buffer_size: int = 10000  # FIVE shuffles per 50k samples
```

**Effect**:
- 5x more shuffle operations = 5x more variation
- Still efficient (no extra I/O overhead)
- Better data mixing naturally

### Change 3: Maximum Diversity (samples_per_file)
**File**: `configs/gpu/small.yaml` (line 260)

**Before**:
```yaml
samples_per_file: 32  # Read 32 from File1, then 32 from File2...
```

**After**:
```yaml
samples_per_file: 1   # Read 1 from File1, 1 from File2, 1 from File3... (round-robin)
```

**Effect**:
- Perfect interleaving of different data sources
- No clustering by source
- Natural dataset diversity

---

## Is My Loss Still Decreasing Too Fast?

### If Yes, Check These (in order):

1. **Did you update `src/Ava/data_streaming.py`?**
   ```bash
   grep "buffer_seed = 42" src/Ava/data_streaming.py
   # Should return 2 matches (lines ~720 and ~761)
   ```

2. **Did you update `configs/gpu/small.yaml`?**
   ```bash
   grep "samples_per_file: 1" configs/gpu/small.yaml
   # Should return 1 match
   ```

3. **Did you update buffer size defaults?**
   ```bash
   grep "buffer_size: int = 10000" src/Ava/data_streaming.py
   # Should return 1 match (line ~822)
   ```

4. **Are you using the right config?**
   ```bash
   # Make sure you're using updated config
   python train.py --config configs/gpu/small.yaml

   # NOT using cached/old config
   ```

5. **Clear Python cache**
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

---

## Advanced Monitoring

### Monitor Shuffle Effectiveness

Add this to your training loop:
```python
# After each epoch
if epoch % 10 == 0:
    # Check that shuffles are actually different
    epoch_seed = 42 + epoch
    epoch_seed_next = 42 + (epoch + 1)

    print(f"Epoch {epoch}: seed={epoch_seed}")
    print(f"Epoch {epoch+1}: seed={epoch_seed_next}")
    print(f"→ Seeds are different: {epoch_seed != epoch_seed_next}")
```

### Loss Curve Analysis

```python
# After each epoch, print trend
if len(epoch_losses) > 1:
    improvement = epoch_losses[-1] - epoch_losses[-2]
    improvement_rate = improvement / epoch_losses[-2]

    print(f"Epoch {epoch}: loss={epoch_losses[-1]:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement_rate*100:.1f}%)")
    print(f"  Expected: Improvement % should DECREASE over epochs")

    # Good: Improvement rate decreases (convergence)
    # Bad: Improvement rate stays same (memorization)
```

---

## Reverting Changes (if needed)

If you need to go back to old behavior:

### Revert Buffer Seed to Fixed
```python
# Line 720 & 761 in data_streaming.py
# Change from:
buffer_seed = 42 + epoch_number + (samples_processed // self.buffer_size)

# Back to:
buffer_seed = 42  # Fixed seed every time
```

### Revert Buffer Size
```yaml
# In configs/gpu/small.yaml line 252
buffer_size: 50000  # Back to original
```

### Revert samples_per_file
```yaml
# In configs/gpu/small.yaml line 260
samples_per_file: 32  # Back to original
```

---

## Getting Help

### If loss still dropping too fast:
1. Verify all 3 changes are applied
2. Check buffer_size is actually being used (not overridden)
3. Look for custom data loader that might ignore new settings
4. Check for other sources of fixed-seed shuffling

### If training is now slower:
1. This is expected with `samples_per_file: 1` (round-robin = more switching)
2. Acceptable trade-off (better quality > slight speed loss)
3. If critical for speed, try `samples_per_file: 2` (still good mixing)

### If validation diverges more:
1. This might be correct! Reveals overfitting
2. Better to see this than have it hidden
3. Add more regularization (dropout, weight decay) if needed

---

## Summary Table

| Issue | Solution | Files Changed | Expected Impact |
|-------|----------|---------------|-----------------|
| Same shuffle every epoch | Epoch-aware seed | `data_streaming.py` | Loss stabilizes |
| Large buffer = less variation | Reduce to 10k | `data_streaming.py`, `small.yaml` | More variety |
| Clustered data | 1 sample/file round-robin | `small.yaml` | Perfect mixing |
| **Result** | Combined fixes | 2 files | **Better generalization** |

---

## Next Steps

1. **Verify** the changes are applied (checklist above)
2. **Run** training with `--max-steps 100` to test
3. **Monitor** loss curve for improvements
4. **Compare** against old runs on wandb/tensorboard
5. **Document** results for your project

Good luck! 🚀
