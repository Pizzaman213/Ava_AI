# Learning Rate Schedule Analysis & Recommendations

## Current Configuration Issues

### Your Current Schedule
```yaml
optimizer:
  learning_rate: 6.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.01

schedule:
  warmup_steps: 2000          # Only 1.2% of training (2k / 163k = 1.2%)
  scheduler_type: cosine
  min_lr: 2.0e-6              # Decays to 0.00003x of peak (very aggressive)
```

### Problems Identified

1. **Warmup Too Short**
   - Current: 2000 steps (1.2% of training)
   - Recommended: 5000-10000 steps (3-6% of training)
   - **Issue**: Model may not stabilize before aggressive learning begins

2. **Cosine Decay Behavior**
   ```
   Step 163,500 / 312,500 (52% progress):
   - Expected LR: ~4.5e-5 (25% decay from peak)
   - Actual LR: 6.03e-5 (0.5% decay)
   ```
   - **Issue**: Cosine decay is very slow in middle, fast at end
   - Most decay happens in final 25% of training

3. **Loss Plateau at 2.9**
   - Loss stuck around 2.9 for thousands of steps
   - **Possible causes**:
     - LR too low to escape local minimum (unlikely at 6e-5)
     - Model capacity limit reached (unlikely for 446M params)
     - Data quality/diversity issues (more likely)
     - Need learning rate warm restart

## Recommended LR Schedule Options

### Option 1: **Longer Warmup + Slower Decay** (Conservative, Recommended)

```yaml
training:
  optimizer:
    learning_rate: 6.0e-5      # Keep current (good value)
    betas: [0.9, 0.95]
    weight_decay: 0.01

  schedule:
    warmup_steps: 5000         # Increased from 2000 (3.2% of 156k steps)
    scheduler_type: cosine
    min_lr: 1.0e-5             # Gentler decay (6e-5 → 1e-5 = 6x, not 30x)
```

**Benefits:**
- ✅ Longer stabilization period
- ✅ More gradual LR decay
- ✅ Better for pre-training (less aggressive)

**Trade-offs:**
- Slower initial convergence
- May need more epochs

---

### Option 2: **Cosine with Warm Restarts** (Aggressive, Best for Escaping Plateaus)

```yaml
training:
  optimizer:
    learning_rate: 6.0e-5
    betas: [0.9, 0.95]
    weight_decay: 0.01

  schedule:
    warmup_steps: 5000
    scheduler_type: cosine_with_restarts
    num_cycles: 2              # 2 cycles over 3 epochs
    min_lr: 1.0e-5
```

**Benefits:**
- ✅ **Escapes plateaus** via periodic LR spikes
- ✅ Explores multiple local minima
- ✅ Often finds better solutions than pure cosine

**Behavior:**
```
LR over time:
Step 0     → 0      (start)
Step 5000  → 6e-5   (warmup complete)
Step 78k   → 1e-5   (end cycle 1)
Step 79k   → 6e-5   (restart!)
Step 156k  → 1e-5   (end cycle 2)
```

**Trade-offs:**
- Can cause temporary loss spikes during restarts
- Requires tuning num_cycles

---

### Option 3: **Linear Warmup + Polynomial Decay** (Stable, Predictable)

```yaml
training:
  optimizer:
    learning_rate: 6.0e-5
    betas: [0.9, 0.95]
    weight_decay: 0.01

  schedule:
    warmup_steps: 5000
    scheduler_type: polynomial
    power: 1.5                 # Between linear (1.0) and quadratic (2.0)
    min_lr: 5.0e-6             # 12x decay (gentler than cosine)
```

**Benefits:**
- ✅ More predictable decay than cosine
- ✅ Earlier decay helps convergence
- ✅ Smoother than cosine (no slow middle phase)

**Trade-offs:**
- Less commonly used (fewer reference implementations)

---

### Option 4: **Inverse Square Root (Transformer Standard)**

```yaml
training:
  optimizer:
    learning_rate: 6.0e-5
    betas: [0.9, 0.95]
    weight_decay: 0.01

  schedule:
    warmup_steps: 8000         # Longer warmup for inv_sqrt
    scheduler_type: inverse_sqrt
```

**Benefits:**
- ✅ Used in original Transformer paper
- ✅ Gentle, continuous decay
- ✅ Works well for very long training runs

**Behavior:**
```
LR = peak_lr / sqrt(max(step, warmup_steps))

Step 8k   → 6.0e-5 (peak)
Step 32k  → 3.0e-5 (1/2)
Step 128k → 1.5e-5 (1/4)
```

**Trade-offs:**
- Never reaches zero (keeps learning forever)
- Slower decay than cosine

---

## My Recommendation: **Option 1 or 2**

### For Your Current Situation (Loss Plateau):

**Start with Option 2 (Cosine with Restarts):**

```yaml
training:
  optimizer:
    learning_rate: 6.0e-5
    betas: [0.9, 0.95]
    weight_decay: 0.01

  schedule:
    warmup_steps: 5000         # Longer stabilization
    scheduler_type: cosine_with_restarts
    num_cycles: 2              # 2 restarts over 3 epochs
    min_lr: 1.0e-5             # Gentler min LR
```

**Why?** The warm restarts will help you escape the 2.9 loss plateau by periodically boosting LR and exploring new regions of the loss landscape.

---

## Alternative: Keep Current Schedule, Adjust LR Value

If you don't want to change the schedule, try increasing the base LR:

```yaml
training:
  optimizer:
    learning_rate: 8.0e-5      # Increased from 6e-5 (33% higher)
    # OR
    learning_rate: 1.0e-4      # More aggressive (67% higher)
```

**When to use:**
- Loss plateau is due to LR being too low
- Quick test without changing schedule

**Risks:**
- May cause instability/divergence
- Monitor loss carefully for first 1000 steps

---

## Diagnostic: Check if LR is the Issue

Run this to see if your loss is actually plateauing or slowly decreasing:

```bash
# Extract loss from logs
grep -oP "Loss: \K[\d.]+" /path/to/training.log | tail -100 | \
  awk '{s+=$1; c++} END {print "Recent avg loss:", s/c}'

# Compare to earlier training
grep -oP "Loss: \K[\d.]+" /path/to/training.log | head -100 | \
  awk '{s+=$1; c++} END {print "Early avg loss:", s/c}'
```

**If loss is truly flat**, try:
1. Warm restarts (Option 2)
2. Higher base LR (8e-5 or 1e-4)
3. Check data quality/diversity

**If loss is slowly decreasing**, keep current schedule but consider longer warmup.

---

## Summary Table

| Schedule | Warmup | Behavior | Best For | Escape Plateau? |
|----------|--------|----------|----------|-----------------|
| **Cosine (current)** | 2000 | Slow middle, fast end | Stable training | ❌ No |
| **Option 1: Cosine + Longer Warmup** | 5000 | Gentler decay | Conservative | ⚠️ Maybe |
| **Option 2: Cosine Restarts** | 5000 | Periodic LR spikes | Plateaus | ✅ Yes |
| **Option 3: Polynomial** | 5000 | Predictable decay | General training | ⚠️ Maybe |
| **Option 4: Inverse Sqrt** | 8000 | Continuous gentle | Long runs | ⚠️ Maybe |

---

## Quick Fix: Update Your Config

Replace the `schedule` section in `code/configs/moe/large.yaml`:

```yaml
  schedule:
    num_epochs: 3
    max_steps: null
    warmup_steps: 5000              # Increased from 2000
    scheduler_type: cosine_with_restarts
    num_cycles: 2                   # Add warm restarts
    min_lr: 1.0e-5                  # Gentler decay (was 2e-6)
```

Then restart training from your latest checkpoint:

```bash
python code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/large.yaml \
  --resume code/outputs/pretraining/*/checkpoints/checkpoint_step_163500.pt
```

The first warm restart will happen around step 230k, which may help escape the plateau.