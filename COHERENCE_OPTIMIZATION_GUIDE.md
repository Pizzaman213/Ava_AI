# Coherence Evaluation Optimization Guide

## Current Bottleneck Analysis

Your training shows: `Logged 81 generation(s) to WandB table at step 162031`

This is **cumulative** (all generations across steps), not per-evaluation. However, coherence measurement is still expensive:

### Performance Impact Per Evaluation
- **Generation**: 3 samples × 128 tokens = 384 forward passes
- **Perplexity**: 1 full forward pass on generated samples
- **Flow score**: 1 full forward pass + hidden state extraction
- **Topic consistency**: Uses cached hidden states (no extra pass)
- **Repetition**: CPU-based n-gram counting

**Total**: ~5-7 forward passes + generation overhead every 2000 steps

## Quick Wins (Immediate Speedup)

### 1. Use Fast Coherence Measurer (Already Implemented)
Your code has `FastCoherenceMeasurer` but it may not be enabled. Verify in [generation.py:511](code/src/ava/training/generation.py#L511):

```python
# Should be: use_fast=True (default)
metrics = self._generation_manager.measure_coherence(
    model,
    global_step,
    coherence_config,
    use_fast=True,      # ✅ Single forward pass optimization
    use_bf16=True,      # ✅ 2x memory reduction
    micro_batch_size=16 # ✅ Larger batches = faster
)
```

**Speedup**: 3-5x faster

### 2. Increase Evaluation Interval
Current: Every 2000 steps
Recommended: Every 5000-10000 steps

```yaml
# In code/configs/moe/large.yaml
training:
  coherence:
    enabled: true
    eval_every_n_steps: 5000  # Changed from 2000
```

**Speedup**: 2.5x less frequent = 2.5x less overhead

### 3. Reduce Sample Count (If Quality Allows)
Current: 3 samples
Recommended: 2 samples (still statistically meaningful)

```yaml
training:
  coherence:
    num_samples: 2  # Changed from 3
```

**Speedup**: 1.5x fewer generations

### 4. Disable Console Logging
```yaml
training:
  coherence:
    log_to_console: false  # Changed from true
    log_to_wandb: true     # Keep WandB for tracking
```

**Speedup**: Reduces I/O overhead

### 5. Skip Coherence During Active Training
Only measure at validation steps:

```python
# In code/src/ava/training/loop.py around line 741
if coherence_config and (self._global_step % self._eval_steps == 0):
    # Only measure coherence at validation steps
    self._maybe_measure_coherence(model, coherence_config)
```

**Speedup**: 20x less frequent (if eval_steps=1000, coherence=2000, only every 2000)

## Advanced Optimizations

### 6. Async Coherence Measurement (Recommended)
Move coherence to background thread (like async generation):

```python
# Pseudo-code for implementation
def measure_coherence_async(self, model, step, config):
    """Non-blocking coherence measurement."""
    if self._coherence_future and not self._coherence_future.done():
        return None  # Skip if previous measurement still running

    # Capture model snapshot
    state_dict_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def _run_coherence():
        cpu_model = reconstruct_model(state_dict_cpu)
        return self.measure_coherence(cpu_model, step, config)

    self._coherence_future = self.executor.submit(_run_coherence)
    return self._coherence_future
```

**Speedup**: ~Zero training overhead (runs in parallel)

### 7. Cache Generated Samples
Generate once, measure multiple times:

```python
# Generate samples
input_ids, texts = measurer._generate_samples()

# Reuse for multiple metrics
perplexity = measurer._compute_perplexity(input_ids)
repetition = measurer._compute_repetition_score(input_ids)
# ... etc
```

**Speedup**: Already implemented in FastCoherenceMeasurer

### 8. Use Gradient Checkpointing for Coherence
Enable for coherence measurement only (not main training):

```yaml
model:
  gradient_checkpointing: true  # Already enabled in your config
```

**Benefit**: 40-60% less memory = can use larger micro_batch_size

## Recommended Configuration

```yaml
# code/configs/moe/large.yaml
training:
  # Reduce coherence frequency
  coherence:
    enabled: true
    eval_every_n_steps: 5000      # ⚡ 2.5x less frequent
    num_samples: 2                # ⚡ 1.5x fewer samples
    max_generation_length: 100    # ⚡ Slightly shorter (128→100)
    temperature: 0.5              # Already optimal
    top_p: 0.9                    # Already optimal
    top_k: 25                     # Already optimal
    log_to_wandb: true            # Keep for tracking
    log_to_console: false         # ⚡ Disable console spam

    # NEW: Fast measurer settings (add these)
    use_fast: true                # ⚡ Single forward pass
    use_bf16: true                # ⚡ Half memory
    micro_batch_size: 16          # ⚡ Larger batches (adjust based on VRAM)
```

## Expected Speedup

| Optimization | Individual Speedup | Cumulative |
|-------------|-------------------|------------|
| Base (current) | 1.0x | 1.0x |
| + FastCoherenceMeasurer | 3-5x | 3-5x |
| + Less frequent (5000 steps) | 2.5x | 7.5-12.5x |
| + Fewer samples (2) | 1.5x | 11-19x |
| + Disable console log | 1.1x | 12-21x |
| + Async measurement | ∞ (non-blocking) | **~Zero overhead** |

**Total**: Coherence overhead reduced from ~5-10 seconds to <1 second, or **zero** with async.

## Implementation Priority

1. **Immediate** (config changes only):
   - Increase `eval_every_n_steps: 5000`
   - Reduce `num_samples: 2`
   - Disable `log_to_console: false`
   - Add `use_fast: true, use_bf16: true, micro_batch_size: 16`

2. **Quick Win** (1-line code change):
   - Only measure at validation steps (link coherence to eval_steps)

3. **Best Long-term** (30 min implementation):
   - Async coherence measurement (background thread)

## Diagnostic Commands

### Check current coherence timing
```bash
# During training, watch for coherence messages
grep "Coherence" code/outputs/pretraining/*/logs/training.log

# Check WandB for coherence step timings
# Look for gaps in "Step X/Y" logs around coherence steps
```

### Verify FastCoherenceMeasurer is used
```python
# In code/src/ava/training/generation.py:511
# Should see use_fast=True when calling measure_coherence
```

### Profile coherence overhead
```python
import time
start = time.time()
metrics = gen_mgr.measure_coherence(model, step, config)
elapsed = time.time() - start
print(f"Coherence took {elapsed:.2f}s")
```

## Alternative: Disable Coherence Entirely

If coherence metrics aren't critical for your training:

```yaml
training:
  coherence:
    enabled: false  # ⚡ Complete elimination of overhead
```

You can always re-enable later for final model evaluation.

## Related Files

- Coherence implementation: [code/src/ava/eval/coherence.py](code/src/ava/eval/coherence.py)
- Generation manager: [code/src/ava/training/generation.py](code/src/ava/training/generation.py)
- Training loop: [code/src/ava/training/loop.py](code/src/ava/training/loop.py#L741)
- Config reference: [code/configs/moe/large.yaml](code/configs/moe/large.yaml#L156-L165)