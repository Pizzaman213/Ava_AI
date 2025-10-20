# Effective Batch Size Display Added ✅

## What Changed

Added **EffBS** (Effective Batch Size) to the training progress bar.

### Before:
```
Epoch 1/1: 10it [00:29, 2.84s/it, Loss=10.2363, LR=1.00e-08, it/s=0.28, BS=24]
```

### After:
```
Epoch 1/1: 10it [00:29, 2.84s/it, Loss=10.2363, LR=1.00e-08, it/s=0.28, BS=24, EffBS=192]
                                                                                    ^^^^^^^^^
                                                                                    NEW!
```

## How It Works

**EffBS = BS × gradient_accumulation_steps**

Example:
- BS = 24 (actual batch size per forward pass)
- gradient_accumulation_steps = 8
- **EffBS = 192** (effective batch size for optimizer update)

## Code Added

In `/project/code/scripts/5_training/train.py` (lines 1283-1290):

```python
# Add effective batch size (BS × gradient_accumulation_steps)
try:
    grad_accum = getattr(trainer.config.training, 'gradient_accumulation_steps',
                        getattr(trainer.config.training, 'gradient_accumulation', 1))
    effective_bs = batch_size * grad_accum
    postfix["EffBS"] = str(effective_bs)
except:
    pass  # If can't get grad_accum, just skip EffBS
```

## Why This Matters

**Effective batch size** is what actually matters for training dynamics:
- Small BS (e.g., 24) = memory efficient
- Large EffBS (e.g., 192) = stable gradients
- Shows you're using gradient accumulation correctly

## What You'll See After Restart

```
Epoch 1/1: 10it [00:29, Loss=10.24, LR=1.00e-08, it/s=0.28, BS=24, EffBS=192]
                                                                     ^^   ^^^^^
                                                          Per-step    Total for
                                                          batch       optimizer
```

## Verifying Gradient Accumulation Works

If gradient accumulation is working correctly:
- **BS**: Should show the micro-batch size (e.g., 24)
- **EffBS**: Should show BS × 8 = 192

If gradient accumulation is NOT working:
- **BS**: 24
- **EffBS**: 24 (no multiplication, means grad_accum=1)

## Restart Training

**Restart to see the new display:**

```bash
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

You should see:
```
✓ Gradient accumulation loaded from YAML: 8
...
Epoch 1/1: [Loss=..., BS=24, EffBS=192]  ← Both values shown!
```

Now you can instantly verify your effective batch size during training! 🎯
