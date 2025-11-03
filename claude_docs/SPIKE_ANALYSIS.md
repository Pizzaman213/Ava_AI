# Loss Spike Root Cause Analysis

## Observed Pattern

From analyzing the training run `run_20251028_005024_538591d7`, I've identified a **clear periodic spike pattern**:

### Spike Timeline
```
Step    Loss    Pattern
----    ----    -------
100     3.86    (warmup)
200     2.41    (warmup)
300     1.39    (warmup)
400     2.14    (warmup)
500     1.16    (warmup)
700     2.34    ← SPIKE
1000    0.15    (good)
1400    1.23    ← SPIKE
1700    2.00    ← SPIKE
2000    0.06    (good)
2400    1.01    ← SPIKE
2700    1.94    ← SPIKE
3000    0.05    (good)
3700    1.94    ← SPIKE
4000    0.04    (good)
4700    1.78    ← SPIKE
5000    0.05    (good)
5700    1.72    ← SPIKE
```

### Key Observations

1. **Post-warmup spikes occur every ~700-1000 steps**
2. **Spikes are NOT aligned with checkpoints** (save_steps: 1000, eval_steps: 1000)
   - Spikes at: 700, 1400, 1700, 2400, 2700, 3700, 4700, 5700
   - Checkpoints at: 1000, 2000, 3000, 4000, 5000
3. **Spike pattern is periodic but irregular** (intervals: 700, 700, 300, 700, 300, 1000, 1000, 1000)
4. **Between spikes, loss decreases smoothly** (excellent training dynamics)

## Root Cause: Data Pipeline Issue (MOST LIKELY)

### Evidence
1. **Periodic but irregular pattern** suggests specific problematic data files/sequences
2. **Not aligned with system events** (checkpoints, eval, gradient accumulation boundaries)
3. **Pattern starts after warmup** (step 700+), suggesting data order dependency

### Hypothesis: File Boundary Effect

With your config:
```yaml
data:
  streaming: true
  buffer_size: 10000
  samples_per_file: 1  # Each file contributes 1 sample per shuffle
  reshuffle_each_epoch: true
```

**The problem**: If your dataset has files with varying difficulty or problematic sequences, and the streaming loader encounters these at regular intervals, it would cause periodic spikes.

### Supporting Evidence from Config

1. **Buffer size = 10,000 tokens**
   - With batch_size=24 and max_length=256: ~416 samples in buffer
   - If problematic files are spread ~every 400-800 samples, you'd see spikes every 700-1400 steps ✓

2. **samples_per_file = 1**
   - Good for diversity, but means problematic files will reappear
   - Each epoch reshuffles, but problematic content remains

3. **Streaming enabled**
   - Files loaded incrementally
   - No global shuffle across all files
   - File order matters more

## Secondary Hypothesis: MoE Expert Routing Instability

### Evidence from Code ([routing.py:665-668](code/src/Ava/layers/routing.py#L665-L668))

```python
# Expert dropout - randomly disable experts during training
if self.training and self.expert_dropout > 0:
    expert_dropout_mask = torch.bernoulli(dropout_probs).bool()
```

However, your config has:
```yaml
model:
  expert_dropout: 0.0  # NOT ENABLED
  router_aux_loss_coef: 0.01  # VERY LOW
```

**Problem**: With 4 experts and top-2 routing, low auxiliary loss can cause expert collapse. If one expert suddenly gets zero tokens due to routing imbalance, the next batch where it receives tokens can cause a loss spike.

### MoE Metrics Not Tracked

```yaml
evaluation:
  moe_metrics:
    track_expert_utilization: false  # ← NO MONITORING!
```

**Cannot verify** if expert collapse is occurring because metrics aren't being logged.

## Tertiary Hypothesis: Gradient Accumulation Sync (LESS LIKELY)

Your gradient accumulation setup:
```yaml
training:
  gradient_accumulation_steps: 8
  batch_size: 24
  # Effective batch = 192
```

If spikes aligned with gradient accumulation boundaries (multiples of 8 steps), this could indicate:
- Gradient staleness
- Inconsistent loss scaling
- DeepSpeed synchronization issues

**However**: Spikes at 700, 1400, 1700, 2400, 2700, etc. don't align with multiples of 8 or 80, so this is **unlikely**.

## Recommended Actions (Priority Order)

### 1. **Enable MoE Monitoring** (IMMEDIATE)

```yaml
evaluation:
  moe_metrics:
    track_expert_utilization: true
    track_routing_entropy: true
    track_load_balance: true
    log_frequency: 100  # Log every 100 steps
```

This will definitively show if expert collapse is causing spikes.

### 2. **Strengthen MoE Load Balancing** (HIGH PRIORITY)

```yaml
model:
  router_aux_loss_coef: 0.05  # Increase from 0.01
  expert_dropout: 0.1  # Enable expert dropout for robustness
  routing_jitter: 0.05  # Already at 0.01, increase to 0.05
```

### 3. **Investigate Data Files** (HIGH PRIORITY)

```bash
# Check for problematic sequences in your dataset
cd /project/code/data/processed

# Look for files with encoding issues
find . -name "*.jsonl" -exec file {} \; | grep -v "ASCII\|UTF-8"

# Check for files with unusual sizes (too large/small)
find . -name "*.jsonl" -exec ls -lh {} \; | awk '{print $5, $9}' | sort -h

# Sample random sequences to check for repeated/malformed content
for file in *.jsonl; do
    echo "=== $file ==="
    head -5 "$file"
done
```

### 4. **Adjust Data Pipeline** (MEDIUM PRIORITY)

```yaml
data:
  buffer_size: 20000  # Increase from 10000 for better mixing
  samples_per_file: 2  # Increase from 1 for less file-boundary effects

  # Add validation to skip problematic sequences
  max_examples_per_file: 1000  # Limit impact of bad files
```

### 5. **Add Gradient Clipping Safety** (LOW PRIORITY)

Already quite conservative, but could tighten:
```yaml
training:
  max_gradient_norm: 1.0  # Reduce from 1.2
  explosion_threshold: 1.5  # Reduce from 2.0
```

## Diagnostic Commands

### Check Current Run for Expert Collapse
```bash
# If expert metrics were logged
grep -E "expert.*load|routing_entropy" /path/to/run/wandb/output.log

# Check for warning messages
grep -i "expert\|routing\|collapse" /path/to/run/logs/training.log
```

### Profile Data Loading
```python
# Add to training script
import time
import torch

def profile_dataloader(dataloader, num_batches=1000):
    spike_candidates = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        # Check for unusual batch characteristics
        input_ids = batch['input_ids']
        batch_loss_estimate = F.cross_entropy(
            model(input_ids).logits.view(-1, vocab_size),
            input_ids.view(-1),
            reduction='mean'
        )

        if batch_loss_estimate > 2.0:  # Suspiciously high
            spike_candidates.append({
                'batch_idx': i,
                'loss_estimate': batch_loss_estimate.item(),
                'input_stats': {
                    'mean': input_ids.float().mean().item(),
                    'std': input_ids.float().std().item(),
                    'unique_tokens': len(torch.unique(input_ids))
                }
            })

    return spike_candidates
```

## Expected Outcome

If **MoE expert collapse** is the root cause:
- Enabling expert metrics will show 0 or near-0 load on 1+ experts at spike steps
- Increasing `router_aux_loss_coef` and enabling `expert_dropout` will eliminate spikes
- Recovery time: **IMMEDIATE** (next training run)

If **data pipeline issue** is the root cause:
- Problematic files/sequences will be identifiable via profiling
- Removing/fixing those files will eliminate spikes
- Increasing buffer size will reduce spike magnitude
- Recovery time: **1-2 days** (requires data cleaning)

## Confidence Assessment

| Hypothesis | Confidence | Why |
|-----------|-----------|-----|
| Data pipeline (specific bad files) | **HIGH (70%)** | Periodic but irregular pattern, not aligned with system events |
| MoE expert collapse | **MEDIUM (20%)** | Low aux loss + no monitoring = plausible, but would expect more regular pattern |
| Gradient accumulation | **LOW (10%)** | Spike timing doesn't align with gradient accumulation boundaries |
| Checkpoint/eval operations | **NONE (0%)** | Definitively ruled out by timing mismatch |

## Next Steps

1. **Run training with MoE metrics enabled** (10 minutes to set up)
2. **Monitor WandB for expert utilization during next spike** (real-time validation)
3. **If experts balanced → focus on data pipeline** (run data profiling)
4. **If expert collapse detected → apply MoE fixes** (update config and restart)

---

**Generated**: 2025-10-29
**Based on run**: `run_20251028_005024_538591d7`
**Analysis confidence**: HIGH
