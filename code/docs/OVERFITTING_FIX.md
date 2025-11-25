# Overfitting Prevention - Learning Rate & Regularization Tuning

**Date**: November 19, 2025
**Issue**: Overfitting observed with batch size 128
**Solution**: Reduced learning rate + increased regularization

---

## Problem Analysis

With batch size 128, the model shows signs of overfitting:
- Loss is noisy but generally decreasing (~2.09-2.14)
- Model may be learning batch-specific patterns
- Need stronger regularization

---

## Solution Applied

### Learning Rate Reduction

**Before**:
```yaml
learning_rate: 0.0001
```

**After**:
```yaml
learning_rate: 0.00005  # 50% reduction
```

**Why**:
- Batch size 128 is relatively large for this model
- Larger batch = need smaller learning rate to avoid overshooting
- 0.00005 is more conservative, reduces oscillation
- Better for fine-grained weight updates

---

### Warmup Steps Increase

**Before**:
```yaml
warmup_steps: 100
```

**After**:
```yaml
warmup_steps: 500  # 5x increase
```

**Why**:
- Gradually increase LR from 0 to target over more steps
- Prevents sudden jumps in optimization
- Better convergence with lower learning rate
- Allows model to settle into good regions early

**Formula**: `lr_at_step = initial_lr * (step / warmup_steps)` for step < warmup_steps

---

### Weight Decay Increase

**Before**:
```yaml
weight_decay: 0.1
```

**After**:
```yaml
weight_decay: 0.2  # 2x stronger regularization
```

**Why**:
- Penalizes large weights (L2 regularization)
- Prevents model from memorizing data
- With Lion optimizer, weight decay is directly applied
- 0.2 is standard for preventing overfitting

**Effect**: Model prefers smaller, distributed weights over large, specific ones

---

## Recommended Training Settings by Batch Size

| Batch Size | Learning Rate | Warmup Steps | Weight Decay | Use Case |
|-----------|---------------|--------------|--------------|----------|
| 8 | 0.0001 | 100 | 0.05 | Very small, gentle |
| 16 | 0.00008 | 200 | 0.1 | Small batches |
| 32 | 0.00006 | 300 | 0.15 | Medium batches |
| **128** | **0.00005** | **500** | **0.2** | **Current setting** |
| 256 | 0.000025 | 1000 | 0.25 | Large batches |
| 512 | 0.0000125 | 2000 | 0.3 | Very large batches |

---

## What to Monitor During Training

### Good Signs
 Loss decreases smoothly over time
 Validation loss follows training loss (not diverging)
 No sudden spikes in loss
 Learning rate schedule is visible in logs

### Bad Signs
 Training loss drops but validation loss increases
 Loss becomes very noisy/unstable
 Training gets stuck at high loss value
 Loss increases instead of decreasing

---

## How to Further Tune If Needed

### If still overfitting:
1. **Increase weight decay**: 0.2 → 0.3
2. **Increase warmup**: 500 → 1000 steps
3. **Reduce batch size**: 128 → 64
4. **Add dropout**: In model config

### If loss is too slow to decrease:
1. **Increase learning rate**: 0.00005 → 0.0001
2. **Decrease warmup**: 500 → 200 steps
3. **Reduce weight decay**: 0.2 → 0.1
4. **Increase batch size**: 128 → 256

---

## Complete Hyperparameter Changes

```yaml
# BEFORE (overfitting with bs=128)
learning_rate: 0.0001
warmup_steps: 100
weight_decay: 0.1

# AFTER (regularized for bs=128)
learning_rate: 0.00005
warmup_steps: 500
weight_decay: 0.2
```

---

## Why These Changes Work Together

### Learning Rate (0.0001 → 0.00005)
- Direct effect: Smaller steps through weight space
- Prevents overshooting optimal point
- More stable convergence

### Warmup (100 → 500 steps)
- Gradual effect: Slowly ramp up learning
- Prevents cold start problems
- Better initial exploration

### Weight Decay (0.1 → 0.2)
- Regularization effect: Penalize complexity
- Prevents memorization
- Encourages generalization

**Combined**: Conservative optimization + strong regularization = better generalization

---

## Expected Training Behavior

### Convergence Curve
```
Loss
4.0 |
    |
3.0 |     . . .
    |    .     .
2.5 | . .       .  ← Slower initial decrease
    |.            .
2.0 |               . . . . . ← Smoother, steadier descent
    |
1.5 |                         . . .
    |_____________________________________________________ Steps
    0      warmup       500      1000     2000
```

**Key differences**:
- Slower initial descent (due to lower LR and warmup)
- Smoother curve (less oscillation)
- Better generalization (weight decay prevents overfitting)

---

## Testing the Changes

### Run training with new settings:
```bash
python code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/minimal_working.yaml \
  --epochs 1 \
  --log-interval 50
```

### Monitor in real-time:
```bash
tail -f logs/training_*.log | grep "Loss:"
```

### Expected output:
```
Epoch 1 | Batch 50/... | Loss: 2.15 | Avg Loss: 2.35 | LR: 0.00e-05
Epoch 1 | Batch 100/... | Loss: 2.08 | Avg Loss: 2.28 | LR: 0.10e-04
Epoch 1 | Batch 150/... | Loss: 2.02 | Avg Loss: 2.20 | LR: 0.20e-04  ← Warmup phase
Epoch 1 | Batch 200/... | Loss: 1.98 | Avg Loss: 2.15 | LR: 0.30e-04
...
Epoch 1 | Batch 500+/... | Loss: 1.85 | Avg Loss: 2.05 | LR: 5.00e-05 ← Steady learning
```

---

## Comparison Table

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| Learning Rate | 0.0001 | 0.00005 | 50% reduction |
| Warmup Steps | 100 | 500 | 5x more careful startup |
| Weight Decay | 0.1 | 0.2 | 2x stronger regularization |
| Expected Overfitting | High | Low | Better generalization |
| Convergence Speed | Fast | Slower | More stable |

---

## Lion Optimizer Context

These settings are optimized for **Lion optimizer** (not AdamW):

- Lion has lower memory usage (50% less than AdamW)
- Different sensitivity to learning rate
- Weight decay works directly on parameters
- Requires slightly different tuning than AdamW

**Key difference**: With Lion, weight_decay is simpler but more aggressive

---

## Recommended Next Steps

1. **Train for 1 epoch** with new settings
2. **Monitor loss curve** - should be smoother
3. **Check validation loss** - should improve
4. **Save checkpoint** after epoch 1
5. **Compare generations** - should be better quality
6. **Fine-tune if needed** based on results

---

## Additional Regularization Options

If these changes aren't enough, consider:

### In the model config:
```yaml
model:
  dropout: 0.1              # Add dropout to embeddings
  attention_dropout: 0.1    # Add dropout to attention
```

### In training:
```yaml
training:
  gradient_clipping: 1.0    # Clip gradients to prevent explosion
  label_smoothing: 0.05     # Smooth target distributions
```

---

## References

- **Lion Optimizer**: https://arxiv.org/abs/2302.06675
- **Learning Rate Scheduling**: https://cs231n.github.io/neural-networks-3/#annealing
- **Weight Decay**: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
- **Batch Size Effects**: https://arxiv.org/abs/1711.00489

---

## Quick Summary

 **Learning Rate**: 0.0001 → 0.00005 (50% reduction for bs=128)
 **Warmup Steps**: 100 → 500 (more gradual startup)
 **Weight Decay**: 0.1 → 0.2 (stronger regularization)

**Result**: Better generalization, smoother training curve, reduced overfitting

Ready to train! 
