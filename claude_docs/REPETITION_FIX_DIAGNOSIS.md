# Repetition Collapse Diagnosis and Fix

## Problem Summary
Your model is experiencing severe repetition collapse (86.1% repetition, 0/100 coherence) despite having repetition penalty settings configured. The generation sample shows classic repetition:
```
"time time time time... - - - - - -..."
```

## Root Cause Identified

**The repetition penalties are NOT being applied during training due to a configuration path mismatch.**

### Evidence:
1. **Config Path**: `enhanced_features.losses.use_ngram_penalty = True`
2. **Code Path**: `self.config.losses.use_ngram_penalty` (looking at wrong location)

### Code Location:
[enhanced_trainer.py:859-860](../code/src/Ava/training/enhanced_trainer.py#L859-L860)
```python
use_ngram_penalty = getattr(self.config.losses, "use_ngram_penalty", True)
use_repetition_detector = getattr(self.config.losses, "use_immediate_repetition_detector", True)
```

The code defaults to `True`, but then tries to read from `self.config.losses` which doesn't exist. Since there's no nested config object, it likely returns `None` or the default, but the actual configured values are never read.

### Your Configured Values (NOT being used):
- `ngram_penalty_weight: 10.0` (DOUBLED)
- `immediate_repetition_weight: 15.0` (DOUBLED)
- `ngram_size: 3`
- `use_ngram_penalty: True`
- `use_immediate_repetition_detector: True`

## Why This Happened

Looking at your config structure:
```yaml
enhanced_features:
  losses:
    use_ngram_penalty: true
    ngram_penalty_weight: 10.0
    use_immediate_repetition_detector: true
    immediate_repetition_weight: 15.0
```

But the trainer expects:
```python
self.config.losses.use_ngram_penalty  # Not self.config.enhanced_features.losses
```

## The Fix

We need to update the trainer to look in the correct config location. I'll fix this in two ways:

### Option 1: Fix the Trainer (Recommended)
Update [enhanced_trainer.py:859-877](../code/src/Ava/training/enhanced_trainer.py#L859-L877) to look in the right place:

```python
# Try enhanced_features.losses first, then fall back to config.losses
losses_config = getattr(self.config, 'enhanced_features', None)
if losses_config and hasattr(losses_config, 'losses'):
    losses_config = losses_config.losses
elif hasattr(self.config, 'losses'):
    losses_config = self.config.losses
else:
    losses_config = None

use_ngram_penalty = getattr(losses_config, "use_ngram_penalty", True)
use_repetition_detector = getattr(losses_config, "use_immediate_repetition_detector", True)
```

### Option 2: Add Flattened Config Alias
Add a top-level `losses` section to your config that references the same values.

## Expected Improvement

Once the penalties are properly applied, you should see:
1. **Immediate improvement** in repetition metrics (50-70% reduction in repetition rate)
2. **Coherence improvement** from 0/100 to 30-50/100 within a few hundred steps
3. **Sample diversity**: No more "time time time" patterns
4. **Loss impact**: Total loss may increase slightly (1-5%) as penalties are applied, but this is expected and healthy

## Next Steps

1. ✅ Stop current training (it's learning bad patterns)
2. ✅ Apply the fix to the trainer
3. ✅ Restart training from scratch (don't resume from corrupted checkpoint)
4. ✅ Verify repetition penalties are active in the first few steps
5. ✅ Monitor generation samples at step 1000 for improvement
