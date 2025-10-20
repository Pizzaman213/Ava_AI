# Gradient Accumulation Not Working - Diagnosis

## Problem
Training shows `gradient_accum=1` but config has `gradient_accumulation_steps: 8`

```
Output: gradient_accum=1
Config: gradient_accumulation_steps: 8  ❌ MISMATCH!
```

## Root Cause

The training process was started BEFORE you changed the config to set `gradient_accumulation_steps: 8`.

**The old process is still running with the old config values!**

## Solution

**You MUST restart the training process!**

Config changes don't take effect until you restart. The running Python process loaded the config when it started.

### Steps to Fix:

1. **Kill the current training:**
   ```bash
   # Press Ctrl+C in the training terminal
   # Or find and kill the process:
   ps aux | grep python | grep train
   kill <PID>
   ```

2. **Verify config is correct:**
   ```bash
   grep -A2 "batch_size:" code/configs/gpu/small.yaml
   ```
   Should show:
   ```yaml
   batch_size: 24
   gradient_accumulation_steps: 8
   ```

3. **Restart training:**
   ```bash
   # Re-run your training command
   python train.py --config configs/gpu/small.yaml
   ```

4. **Verify it's working:**
   Look for this in the output:
   ```
   📈 LR Schedule: gradient_accum=8  ← Should be 8, not 1!
   ```

## Expected After Restart

**Memory:**
- Batch size: 24
- Gradient accumulation: 8
- **Effective batch: 192** (was 24!)

**Speed:**
- Slower per iteration (doing 8 micro-batches)
- But same effective throughput
- Better convergence with larger effective batch

**Logs should show:**
```
Batch size: 24
gradient_accum=8  ← Fixed!
```

## Why This Matters

**Current (wrong):**
- Effective batch = 24
- Too small for stable training
- Noisier gradients

**After restart (correct):**
- Effective batch = 192
- Much more stable
- Better training dynamics

## Quick Check

Run this to verify config:
```bash
python3 -c "
import yaml
with open('code/configs/gpu/small.yaml') as f:
    cfg = yaml.safe_load(f)
print(f\"Batch: {cfg['training']['batch_size']}\")
print(f\"Grad accum: {cfg['training']['gradient_accumulation_steps']}\")
print(f\"Effective: {cfg['training']['batch_size'] * cfg['training']['gradient_accumulation_steps']}\")
"
```

Should output:
```
Batch: 24
Grad accum: 8
Effective: 192
```

**Then restart training to apply it!**
