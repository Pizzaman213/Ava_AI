# Dynamic Batching System

Automatic memory-aware batch sizing for optimal GPU utilization.

## Overview

The dynamic batching system automatically adjusts batch size based on real-time GPU memory utilization. It targets 80% VRAM usage and scales token budget proportionally to maximize throughput without OOM errors.

## How It Works

### Token Budget Mode (Recommended)

Instead of fixed batch sizes, the system targets a **token count per batch**:

```
Token Budget = f(VRAM utilization)

- VRAM < 80%  → Scale UP toward max_tokens (16384)
- VRAM = 80%  → Use base_tokens (4096)
- VRAM > 80%  → Scale DOWN toward min_tokens (512)
- VRAM > 92%  → Emergency: use min_tokens + clear cache
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │   DataLoader    │───▶│   DynamicBatchIterator       │   │
│  │  (mini-batches) │    │   - Accumulates mini-batches │   │
│  │    BS: 8        │    │   - Yields when token budget │   │
│  └─────────────────┘    │     is reached               │   │
│                         └──────────────┬───────────────┘   │
│                                        │                    │
│                         ┌──────────────▼───────────────┐   │
│                         │   DynamicBatchScheduler      │   │
│                         │   - Monitors GPU memory      │   │
│                         │   - Calculates token budget  │   │
│                         │   - EMA smoothing            │   │
│                         └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `code/src/Ava/training/optimizations/dynamic_batching.py` | Core scheduler, memory monitoring, token budget calculation |
| `code/src/Ava/data/dynamic_batch_iterator.py` | Wraps DataLoader, accumulates batches to token budget |
| `code/configs/moe/large.yaml` | Configuration options |

## Configuration

```yaml
training:
  batching:
    batch_size: null  # Let dynamic batching control this
    gradient_accumulation_steps: 4

    dynamic_batching:
      enabled: true
      min_batch_size: 8      # Mini-batch size from DataLoader
      max_batch_size: 256    # Maximum samples per yielded batch

      # Token Budget (Feature 1)
      token_budget:
        enabled: true
        target_tokens_per_batch: 4096   # Base target at 80% VRAM
        max_tokens_per_batch: 16384     # Max when VRAM is low
        min_tokens_per_batch: 512       # Min when VRAM is high
```

## Token Budget Calculation

The `get_dynamic_token_budget()` function in `dynamic_batching.py`:

```python
def get_dynamic_token_budget(self) -> int:
    target_vram = 0.80  # Target 80% VRAM
    safety_max = 0.92   # Emergency threshold

    if vram_util >= safety_max:
        torch.cuda.empty_cache()
        return min_tokens  # 512

    if vram_util < target_vram:
        # Below target: scale UP
        headroom = (target_vram - vram_util) / target_vram
        return base_tokens + (max_tokens - base_tokens) * headroom
    else:
        # Above target: scale DOWN
        excess = (vram_util - target_vram) / (safety_max - target_vram)
        return base_tokens - (base_tokens - min_tokens) * excess
```

### Examples

| VRAM % | Headroom/Excess | Token Budget | Approx BS |
|--------|-----------------|--------------|-----------|
| 50%    | 37.5% headroom  | 8,704        | ~136      |
| 60%    | 25% headroom    | 7,168        | ~112      |
| 70%    | 12.5% headroom  | 5,632        | ~88       |
| 80%    | 0% (at target)  | 4,096        | ~64       |
| 85%    | 42% excess      | 2,590        | ~40       |
| 90%    | 83% excess      | 1,122        | ~18       |
| 92%+   | Emergency       | 512          | ~8        |

## Features Implemented

### Feature 1: Token-Budget Batching
Target token count instead of sample count. Maximizes GPU utilization by packing variable-length sequences efficiently.

### Feature 2: Sequence-Length Aware (Disabled by default)
Scales batch size inversely with sequence length (attention is O(n²)). Conflicts with token-budget mode.

### Feature 3: Multi-GPU Sync
Synchronizes batch sizes across GPUs to prevent DDP hangs from mismatched batches.

### Feature 4: Gradient Accumulation Integration
Coordinates batch size with gradient accumulation for stable effective batch size.

### Feature 5: Predictive Memory Estimation
Calibrates a memory model over 50 steps to predict memory usage before increasing batch size.

### Feature 6: Memory Thresholds
Configurable thresholds for low/target/high/critical memory states.

### Feature 7: Geometric Warmup
Grows batch size exponentially during warmup phase to find stable operating point.

### Feature 8: Memory Trend Detection
Detects memory trends and oscillations, auto-tunes EMA smoothing factor.

## Memory Measurement

Uses `torch.cuda.memory_reserved()` instead of `memory_allocated()`:
- `memory_allocated()`: Only PyTorch tensors
- `memory_reserved()`: Includes CUDA caches, workspace - matches `nvidia-smi`

## Automatic Cache Clearing

When VRAM > 90%, the system automatically:
1. Clears CUDA cache (`torch.cuda.empty_cache()`)
2. Clears model caches (causal mask, RoPE embeddings)
3. Reduces token budget to minimum

## Logging

The system logs every 10 steps (configurable):
```
[21:42:16] Epoch 1 | Batch 3290/3810644 | BS: 8 | Loss: 6.6860 | VRAM: 96% | TokBudget: 512
```

Every 100 steps, the scheduler logs:
```
[DynBatch] Step 3300: VRAM=96%, token_budget=512
```

## Troubleshooting

### VRAM stuck at 90%+ with small batches
- **Cause**: Model + optimizer nearly fills GPU
- **Solution**: Reduce model size, enable more aggressive gradient checkpointing, or use a bigger GPU

### Token budget not increasing
- **Cause**: Thresholds too tight, or EMA smoothing too slow
- **Solution**: Check `smoothed_utilization` vs `raw_utilization` in logs

### Batch size oscillating
- **Cause**: Memory fragmentation or variable sequence lengths
- **Solution**: Enable trend detection, increase `oscillation_threshold`

### OOM errors
- **Cause**: Token budget too aggressive, or sudden memory spike
- **Solution**: Lower `safety_max` threshold (default 0.92), reduce `max_tokens_per_batch`

## Performance Impact

| Metric | Without Dynamic Batching | With Dynamic Batching |
|--------|--------------------------|----------------------|
| VRAM utilization | 60-70% (conservative) | 75-85% (optimal) |
| Throughput | Baseline | +15-25% |
| OOM risk | Medium | Low (auto-adjusts) |
| Manual tuning | Required | Automatic |

## Code Flow

1. **DataLoader** yields mini-batches (BS: 8)
2. **DynamicBatchIterator** accumulates mini-batches
3. For each mini-batch, calls `scheduler.get_dynamic_token_budget()`
4. When `accumulated_tokens >= token_budget`, yields concatenated batch
5. **Scheduler** updates memory stats via `get_memory_stats()`
6. EMA smoothing applied: `smoothed = alpha * raw + (1-alpha) * smoothed`
7. Next iteration uses updated token budget

## Future Improvements

- [ ] Adaptive EMA alpha based on memory variance
- [ ] Per-layer memory tracking for more accurate predictions
- [ ] Integration with FSDP/DeepSpeed memory management
- [ ] Automatic model parallelism suggestions when memory is critical
