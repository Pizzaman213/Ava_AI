# Training Stability Fix & Recovery Plan

## Problem Diagnosis

Your training experienced a **severe loss spike at ~5,500 steps** (jumping from ~4.3 to ~5.8), indicating training instability rather than a plateau. The model never recovered to its pre-spike performance.

### Root Causes Identified

1. **Learning Rate Too Aggressive**
   - Base LR: 0.00008 (too high after spike recovery)
   - Cosine with restarts + adaptive LR = automatic LR increases triggering more spikes
   - `plateau_factor: 1.15` = boosting LR by 15% during unstable periods

2. **Batch Size Mismatch & Too Small**
   - Config batch_size: 64, gradient_accumulation: 2 (effective: 128)
   - DeepSpeed: micro_batch: 64, train_batch: 128, grad_accum: 2 (MISMATCH!)
   - Small batches = noisier gradients = instability

3. **Excessive Noise**
   - Dropout: 0.12 (too high during unstable training)
   - Weight decay: 0.08 (adding regularization noise)

4. **Weak Gradient Control**
   - max_gradient_norm: 2.0 (too loose to catch spikes early)
   - explosion_threshold: 3.0 (too high)

---

## Changes Applied

### ✅ Core Training Parameters ([small.yaml:28-48](code/configs/gpu/small.yaml#L28-L48))

```yaml
# BEFORE → AFTER
batch_size: 64 → 128                          # 2x larger for smoother gradients
gradient_accumulation_steps: 2 → 1            # Simplified (same effective batch per step)
learning_rate: 0.00008 → 0.00004             # 50% reduction for stability
lr_scheduler_type: cosine_with_restarts → cosine  # Removed restarts (spike trigger)
num_cycles: 2 → 1                            # Single smooth decay
lr_end: 0.00001 → 0.000005                   # Gentler end (50% of original)
weight_decay: 0.08 → 0.05                    # Reduced regularization
max_gradient_norm: 2.0 → 1.0                 # Tighter clipping
warmup_steps: 1000 → 2000                    # Longer warmup
```

### ✅ Dropout Reduction ([small.yaml:27,17-18](code/configs/gpu/small.yaml#L27))

```yaml
dropout: 0.12 → 0.08                         # -33% noise
attention_dropout: 0.12 → 0.08
hidden_dropout: 0.12 → 0.08
```

### ✅ Adaptive LR Disabled ([small.yaml:79-80](code/configs/gpu/small.yaml#L79-L80))

```yaml
adaptive_lr:
  enabled: true → false                      # Turn off until loss stable < 4.0
```

**Reasoning:** Adaptive LR was boosting learning rate during unstable periods, making spikes worse.

### ✅ Enhanced Gradient Health ([small.yaml:112-120](code/configs/gpu/small.yaml#L112-L120))

```yaml
gradient_health:
  initial_clip_value: 2.0 → 1.0             # Tighter from start
  final_clip_value: 2.0 → 1.0
  warmup_steps: 1000 → 2000                 # Longer warmup
  explosion_threshold: 3.0 → 2.0            # Lower threshold to catch spikes earlier
  auto_reduce_lr: true                      # Automatically cut LR on explosion
  lr_reduction_factor: 0.5                  # Halve LR if explosion detected
```

### ✅ Monitoring Improvements ([small.yaml:62,67-68](code/configs/gpu/small.yaml#L62))

```yaml
eval_steps: 500 → 250                       # 2x more frequent monitoring
early_stopping_patience: 5 → 8              # More patient (allow recovery)
early_stopping_threshold: 0.005 → 0.002     # More sensitive to improvements
```

### ✅ DeepSpeed Configuration Sync

**File:** [deepspeed_config.json](code/configs/deepspeed_config.json)

```json
"train_batch_size": 128 (unchanged)
"train_micro_batch_size_per_gpu": 64 → 128  # Match batch_size
"gradient_accumulation_steps": 2 → 1        # Match training config
"lr": 0.0004 → 0.00004                      # Match training LR
"gradient_clipping": 10.0 → 1.0             # Match training clip
"warmup_num_steps": 3000 → 2000             # Match training warmup
```

**File:** [small.yaml:190-199](code/configs/gpu/small.yaml#L190-L199)

```yaml
deepspeed:
  train_batch_size: 128 (unchanged)
  micro_batch_size: 64 → 128                # Now consistent!
  gradient_accumulation_steps: 2 → 1        # Now consistent!
```

---

## Expected Results

### Immediate (Steps 6k-8k)
- **No more loss spikes** - should be rock solid
- Smooth training with gradual descent
- Gradient norms staying below 1.0

### Short-term (Steps 8k-10k)
- Loss descent from ~4.8 to **3.5-4.0**
- Perplexity dropping from ~12.2 to **~7-8**
- Stable, monotonic improvement

### Mid-term (Steps 10k-15k)
- Once loss stable < 4.0, can **re-enable adaptive_lr** if needed
- Target loss: **2.5-3.0** (realistic for 100M MoE)
- Target perplexity: **5-7**

---

## Recovery Instructions

### Option 1: Resume from Pre-Spike Checkpoint (RECOMMENDED)

This is the cleanest approach - go back to before the spike occurred.

```bash
# 1. Find checkpoints around step 5000 (before spike at 5500)
ls -lh /project/code/outputs/small_enhanced/checkpoint-*

# 2. Resume from the best checkpoint before the spike
# Update small.yaml:
run_management:
  resume_from_checkpoint: /project/code/outputs/small_enhanced/checkpoint-5000

# 3. Start training with new stable config
python code/scripts/training/train.py \
  --config code/configs/gpu/small.yaml
```

**Why this works:** You'll resume from when the model was learning well (loss ~4.3), and the new config will prevent the spike from happening again.

### Option 2: Continue from Current Checkpoint

If you want to continue from your current position (~6.5k steps, loss ~4.8):

```bash
# 1. Find latest checkpoint
ls -lh /project/code/outputs/small_enhanced/checkpoint-* | tail -5

# 2. Resume from latest
# Update small.yaml:
run_management:
  resume_from_checkpoint: /project/code/outputs/small_enhanced/checkpoint-latest

# 3. Start training
python code/scripts/training/train.py \
  --config code/configs/gpu/small.yaml
```

**Trade-off:** You keep the progress, but you're starting from a worse loss position (4.8 vs 4.3). Will take longer to get back on track.

### Option 3: Fresh Start (NOT RECOMMENDED)

Only if checkpoints are corrupted or you want a clean experiment:

```bash
# Clear checkpoint reference
run_management:
  resume_from_checkpoint: null

# Start fresh
python code/scripts/training/train.py \
  --config code/configs/gpu/small.yaml
```

**Downside:** Lose all progress from first 6.5k steps.

---

## Monitoring Checklist

After resuming training, monitor these metrics closely in the first 1000 steps:

### Critical Stability Indicators

1. **Loss Trend** (check every 250 steps via W&B)
   - ✅ Should decrease smoothly, no jumps > 0.1
   - ❌ If any spike > 0.5, stop immediately

2. **Gradient Norms** (check logs every 100 steps)
   - ✅ Should stay < 1.0 (clipping at 1.0)
   - ❌ If consistently hitting 1.0, LR may still be too high

3. **Learning Rate Schedule** (check W&B)
   - ✅ Should start at ~0.00004 after warmup
   - ✅ Smoothly decay following cosine curve
   - ❌ No sudden jumps (adaptive LR is off)

4. **GPU Memory** (check logs)
   - Batch size 128 → may use ~1-2GB more VRAM
   - If OOM, reduce batch_size to 96 (keep grad_accum at 1)

### Performance Indicators

5. **Steps per Second**
   - Expect ~10-20% slower (larger batches)
   - Trade-off: fewer total steps needed (better gradient estimates)

6. **MoE Metrics** (every 250 steps)
   - Expert utilization: should be balanced (all experts 10-15%)
   - Router entropy: should stay > 1.5 (good diversity)
   - Load balance loss: should stay < 0.05

---

## Troubleshooting

### If loss is still unstable after 1000 steps:

```yaml
# Further reduce LR
learning_rate: 0.00004 → 0.00002

# Tighten gradient clipping even more
max_gradient_norm: 1.0 → 0.5
gradient_health:
  explosion_threshold: 2.0 → 1.5
```

### If GPU OOM with batch_size=128:

```yaml
# Reduce batch size slightly
batch_size: 128 → 96
gradient_accumulation_steps: 1 (keep same)

# Update deepspeed section too:
deepspeed:
  train_batch_size: 96
  micro_batch_size: 96
  gradient_accumulation_steps: 1
```

### If loss decreases too slowly:

```yaml
# After 10k steps, if loss is stable and below 4.0:
adaptive_lr:
  enabled: false → true

# This will allow careful LR increases when safe
```

### If training speed is too slow:

```yaml
# Keep stability fixes, but optimize dataloading:
dataloader_num_workers: 6 → 8
data:
  prefetch_factor: 4 → 8
  persistent_workers: true (already set)
```

---

## Key Insights

1. **Your model WAS learning fine until 5.5k steps** - the architecture is good
2. **The spike suggests LR restarts or adaptive LR increases triggered it**
3. **Larger batches (128 vs 64) = smoother, more reliable gradients**
4. **Disabling adaptive LR temporarily is critical** - it was helping during plateau, but hurting during spikes

## Success Criteria

After 1000 steps of resumed training, you should see:

- ✅ Zero loss spikes (no jumps > 0.1)
- ✅ Loss below 4.5 (if resuming from step 5k)
- ✅ Loss below 4.7 (if resuming from current position)
- ✅ Gradient norms consistently < 1.0
- ✅ Smooth, monotonic improvement in validation loss

If you see these signs, continue training to 15k-20k steps total.

---

## Next Steps After Stability Achieved

Once loss is stable below 4.0 for 3k+ steps:

1. **Re-enable adaptive LR carefully:**
   ```yaml
   adaptive_lr:
     enabled: true
     plateau_factor: 1.05  # More conservative (was 1.15)
     max_lr: 0.00006      # Lower ceiling (was 0.00012)
   ```

2. **Consider slight LR warmup if plateau persists:**
   ```yaml
   lr_scheduler_type: cosine_with_restarts
   num_cycles: 1  # Single restart at 50% of training
   ```

3. **Monitor for 2k steps before fully trusting the new settings**

---

## Files Modified

1. [code/configs/gpu/small.yaml](code/configs/gpu/small.yaml) - Main training config
2. [code/configs/deepspeed_config.json](code/configs/deepspeed_config.json) - DeepSpeed settings

## Rollback Instructions

If you need to revert to previous config:

```bash
git diff HEAD code/configs/gpu/small.yaml
git checkout HEAD -- code/configs/gpu/small.yaml
git checkout HEAD -- code/configs/deepspeed_config.json
```

---

**Generated:** 2025-11-07
**Issue:** Loss spike at 5.5k steps (4.3 → 5.8)
**Solution:** Stability-first config with larger batches, lower LR, disabled adaptive LR
**Expected Outcome:** Smooth descent to loss 2.5-3.0 by 15k steps
