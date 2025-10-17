# Repetition Issue Fix Guide

## Problem
Model generates 93.1% repetitive output: "time time time time time..."

## Root Causes

### 1. **Insufficient Repetition Penalty During Generation**
Current config has `repetition_penalty_weight: 1.0` but this may not be passed correctly to the generator during validation.

### 2. **EOS Token Suppression**
Config has:
```yaml
eos_penalty_weight: 0.0
eos_logit_bias: 0.0
```
This means EOS token is not being discouraged, which can lead to premature stopping or repetitive patterns.

### 3. **Low Temperature/Diversity**
The validation generation likely uses default low temperature which encourages repetition.

## Solutions

### Quick Fix #1: Update Generation Parameters in Config

Add to your `small.yaml`:

```yaml
generation:
  # Strong repetition penalties
  repetition_penalty: 1.5          # Penalize repeated tokens
  eos_penalty: 2.0                 # Discourage premature EOS
  min_length: 30                   # Minimum generation length

  # Increase diversity
  temperature: 0.9                 # Higher temperature for diversity
  top_k: 100                       # More token options
  top_p: 0.95                      # Wider nucleus sampling

  # N-gram blocking to prevent "time time time"
  use_ngram_blocking: true
  ngram_size: 2                    # Block 2-gram repetitions

  # Sampling instead of greedy
  do_sample: true
```

### Quick Fix #2: Increase Repetition Penalty Weight in Training

Update lines 36-38 in `small.yaml`:

```yaml
training:
  repetition_penalty_weight: 2.0       # Increase from 1.0
  immediate_repetition_weight: 1.5     # Increase from 0.5
  eos_penalty_weight: 0.3              # Add small EOS encouragement
```

### Quick Fix #3: Enable N-gram Penalty Loss

Update lines 153-156:

```yaml
enhanced_features:
  losses:
    use_ngram_penalty: true              # Enable!
    ngram_size: 2                        # Catch "time time"
    ngram_penalty_weight: 0.5            # Moderate penalty
    use_immediate_repetition_detector: true   # Enable!
    immediate_repetition_weight: 1.0     # Increase from 0.0
```

## Priority Actions

### IMMEDIATE (Do this now):

1. **Stop current training** (it's learning bad patterns)
   ```bash
   pkill -f "train.py"
   ```

2. **Apply all three quick fixes above to config**

3. **Restart from checkpoint 14000**:
   ```bash
   # Edit config to add:
   run_management:
     resume_from_checkpoint: outputs/small_enhanced/checkpoint-14000
   ```

4. **Test generation before continuing**:
   ```bash
   python code/scripts/validation/test_checkpoint_improved.py \
     --checkpoint outputs/small_enhanced/checkpoint-14000 \
     --num-samples 10 \
     --temperature 0.9 \
     --repetition-penalty 1.5
   ```

### Data Leakage Issue

The warning "Val loss < Train loss" suggests:
- Validation set may overlap with training
- Or validation is easier than training

**Fix**: Check your data split in `/project/code/data/processed/`

## Expected Results After Fix

- Repetition rate should drop from 93.1% to < 20%
- Generation should be diverse and coherent
- Validation loss should normalize relative to train loss

## Monitor These Metrics

After restarting:
1. Watch repetition % at next validation (step 15000)
2. Check sample outputs for diversity
3. Verify val/train loss ratio stabilizes around 0.9-1.1
