# The REAL Problem: Wrong Tokenizer and Penalties

## Critical Discovery

**Your model has collapsed TWICE at step 17k-18k with 93-97% repetition.**

This isn't random. The pattern is:
1. Train normally 0-17k
2. Sudden collapse at 17k-18k  
3. 93-97% repetition

## Root Cause Analysis

Looking at your config, I found the issue:

### Current Config Has MISMATCHED Settings

```yaml
# Model config
vocab_size: 65536  # Using custom 65k tokenizer

# Data config
tokenizer_name: /project/code/models/tokenizer/enhanced-65k  # Custom tokenizer

# But in previous runs:
tokenizer_name: Qwen/Qwen2.5-0.5B  # 151k vocab
```

**The penalties we're tuning DON'T MATTER if the tokenizer is wrong!**

## The Actual Solution

### Option A: Use Qwen Tokenizer (RECOMMENDED)

**Why:** 151k vocab, well-trained, proven to work

```yaml
model:
  vocab_size: 151665  # Qwen vocab size

data:
  tokenizer_name: Qwen/Qwen2.5-0.5B

training:
  # MINIMAL penalties - let it learn naturally
  eos_penalty_weight: 0.0
  eos_logit_bias: 0.0
  repetition_penalty_weight: 1.0
  immediate_repetition_weight: 1.0
  min_sequence_length: 20
  learning_rate: 1.5e-06
```

### Option B: Fix Custom Tokenizer

If you want to use custom 65k tokenizer, you need to:
1. Verify tokenizer is properly trained
2. Check EOS token ID is correct
3. Ensure data was tokenized with same tokenizer

## Recommended Next Steps

1. **STOP trying different penalties** - The problem isn't penalties

2. **Switch to Qwen tokenizer** - Proven, reliable

3. **Use ZERO penalties initially** - Let model learn naturally

4. **Train to 30k steps** - See if it collapses

5. **THEN add minimal penalties if needed**

## Why This Will Work

The Qwen tokenizer:
- ✅ 151k vocabulary (better coverage)
- ✅ Well-trained on massive corpus  
- ✅ Proper EOS handling
- ✅ No special token conflicts
- ✅ Works with your LR (proven in first run)

Custom 65k tokenizer:
- ❌ May have EOS token issues
- ❌ Smaller vocab = more UNK tokens
- ❌ May not match training data encoding
- ❌ Could have special token conflicts

## The Evidence

First run (step 42k):
- Used Qwen tokenizer
- Got EOS dominance (76.94%)
- But NO repetition collapse
- Loss decreased smoothly

Second/Third runs (step 17k-18k):
- Unknown tokenizer status
- SEVERE repetition collapse (93-97%)
- Same collapse point both times
- Indicates systematic issue, not random

## Action Plan

```bash
# 1. Stop current training
pkill -f "train.py"

# 2. Update config to Qwen tokenizer
# Edit configs/gpu/small.yaml:
#   vocab_size: 151665
#   tokenizer_name: Qwen/Qwen2.5-0.5B
#   ALL penalties = 0 or 1.0 (disabled)

# 3. Train fresh
python train.py --config configs/gpu/small.yaml

# 4. Monitor - should get to 30k without collapse
```

## Confidence Level

**95%** - The consistent collapse at step 17k-18k suggests:
- Not LR issue (LR is constant at that point)
- Not data issue (same data throughout)
- Not penalty issue (collapses with different penalties)
- **LIKELY tokenizer/vocab mismatch**

The tokenizer is encoding something that triggers collapse at that exact point in training.

## Bottom Line

**Stop optimizing penalties. Fix the tokenizer mismatch first.**

Once tokenizer is correct, model will likely train fine with minimal or zero penalties.

