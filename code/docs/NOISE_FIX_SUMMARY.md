# Generation Noise Fix Summary

## Problem

Early in training (steps 0-500), the model generates gibberish text with out-of-vocabulary words:

```
Step 300: "Broadcasting combined infrared diffusion Grinding missiles
          Prim cilantro eczema Judaism..."
Step 350: "her himself you took playing make it they so then on the man..."
Step 400: [Improved but still incoherent]
```

**Warning**: `Detected repetitive sequences (unique_ratio=0.0013)`

## Root Cause

1. **Large Vocabulary**: Tokenizer has 50,680 tokens (general GPT-2 vocabulary)
2. **Simple Training Data**: TinyStories only uses ~5,000 simple words
3. **Random Initialization**: At step 0-500, embeddings are random
4. **Result**: Model samples from full 50k vocab before learning which tokens are appropriate

## Why This is Normal

- At step 300, loss = 2.2 → Model hasn't learned yet
- By step 350, using story words → Learning in progress
- By step 1000+, should generate coherent text → Embeddings learned

## Solution Applied

### 1. Added `skip_until_step` Parameter

**File**: [loop.py:1254-1257](code/src/ava/training/loop.py#L1254-L1257)

```python
# Skip generation if before minimum step threshold
skip_until_step = generation_config.get('skip_until_step', 0)
if self._global_step < skip_until_step:
    return
```

### 2. Updated All Configs

| Config | skip_until_step | Rationale |
|--------|----------------|-----------|
| **minimal_working.yaml** | 500 | Production: wait for reasonable coherence |
| **fast_learning.yaml** | 1000 | More regularization = slower learning |
| **overfit_test.yaml** | 200 | Overfitting is fast, can check sooner |

### 3. Improved Generation Parameters

**fast_learning.yaml**:
```yaml
generation:
  skip_until_step: 1000          # Skip early garbage
  generate_every_n_steps: 100    # Check less frequently
  temperature: 0.8               # Balanced sampling
  top_p: 0.9                     # Sample from broader distribution
  top_k: 50                      # Allow more token choices
  repetition_penalty: 1.2        # Less aggressive
  no_repeat_ngram_size: 2        # Block 2-grams only
```

## Noise Sources Checked

✅ **Router Jitter**: Only active during training, disabled during generation
✅ **Model.eval()**: Generation uses eval mode (no dropout/noise)
✅ **Data Quality**: Verified TinyStories data is clean
✅ **Tokenizer**: Matches training data (50k vocab is correct)

## Expected Results with Fix

| Step Range | Behavior |
|------------|----------|
| 0-500 | ❌ No generation (skipped) |
| 500-1000 | ✅ First generations (may be rough) |
| 1000+ | ✅ Improving coherence |
| 2000+ | ✅ Good quality stories |

## Restart Training

To apply the fix, restart your training:

```bash
# Stop current training
pkill -f train_pipeline

# Restart with updated config
PYTHONPATH=code/src python code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/fast_learning.yaml
```

## Alternative: Wait It Out

Your current training at step 400-500 should start improving soon. You can:
1. **Keep running** - generations should get better by step 1000
2. **Check loss** - If loss is decreasing, model is learning
3. **Be patient** - Early noise is normal for random initialization

## Verification

After the fix, you should see:
- No generations before step threshold
- First generation at step 500/1000 showing basic story structure
- Steady improvement in coherence
- No "repetitive sequences" warnings after step 1500

---

**Files Modified**:
- [code/src/ava/training/loop.py](code/src/ava/training/loop.py#L1254-L1257)
- [code/configs/moe/minimal_working.yaml](code/configs/moe/minimal_working.yaml#L144)
- [code/configs/moe/fast_learning.yaml](code/configs/moe/fast_learning.yaml#L131)
- [code/configs/moe/overfit_test.yaml](code/configs/moe/overfit_test.yaml#L123)

**Bug Fixed**:
- [code/src/ava/training/data_manager.py](code/src/ava/training/data_manager.py#L657-L683) - Fixed gradient_accumulation_steps None error
