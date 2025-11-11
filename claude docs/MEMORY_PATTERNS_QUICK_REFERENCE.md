# CPU Memory Patterns - Quick Reference

## File Locations & Line Numbers

### Core Memory Management Files

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| **Memory Monitor** | `/project/code/src/Ava/distributed/memory_monitor.py` | 67-150 | Real-time GPU/CPU memory tracking with OOM prediction |
| **GPU Memory Manager** | `/project/code/src/Ava/utils/gpu_memory.py` | 25-150 | Memory cleanup & defragmentation utilities |
| **Data Loader** | `/project/code/src/Ava/data/dataloader.py` | 1-1300 | Streaming data loading with dynamic batching |
| **Trainer Core** | `/project/code/src/Ava/training/core/trainer.py` | 2600-3100 | Training loop with gradient accumulation |
| **Expert Offloading** | `/project/code/src/Ava/layers/offloaded_experts.py` | 171-826 | CPU offloading with async prefetch |

---

## Memory Usage Patterns

### Data Loading
```
Buffer size:        15,000 samples × ~2KB = 30MB per worker
Buffer Memory:      6 workers × 30MB = 180MB
Prefetch batches:   6 workers × 2 batches = 12 prefetched
Prefetch Memory:    12 × 32 batch × 2048 tokens × 8 bytes ≈ 6GB
Bucketing overhead: Dict storage + statistics ~50MB per worker

TOTAL DATA LOADER RAM: ~6.5GB (with defaults, long sequences)
OPTIMIZED:            ~2GB (with adaptive prefetch=1, short sequences)
```

### Gradient Accumulation
```
Micro-steps per optimizer step: gradient_accumulation_steps (typically 1-4)
Gradient buffer growth:         Scales linearly with accumulation steps
Activation memory:              ~25% reduction with gradient checkpointing
DeepSpeed ZeRO overhead:        Distributed gradient tracking ~5-10% memory

EXAMPLE: batch=16, grad_accum=4, seq_len=2048
  Total micro-batches in flight: 4 × 16 = 64 effective
  Activation memory: ~8GB (24GB GPU / 3)
  With checkpointing: ~6GB (25% saved)
```

### Expert Offloading
```
Expert size (100M params):    400MB (FP32), 50MB (INT8)
32 experts on CPU:            12.8GB pinned RAM (FP32)
Active on GPU at once:        4 experts (LRU cache)
Transfer rate (pinned):       10-20 GB/s
Transfer time per expert:     27ms (FP32) → 6.75ms (INT8)
Prefetch lookahead:           1-3 experts (adjusts adaptively)
Expected prefetch hit rate:   >70% with adaptive depth

BOTTLENECK: 4 experts × 27ms = 108ms transfer per forward pass
OPTIMIZED:  With INT8: 4 × 6.75ms = 27ms, or use predictive caching
```

---

## Memory Thresholds & Limits

### Memory Monitor Configuration
```python
target_utilization:    85%   (desired GPU memory usage)
warning_threshold:     99%   (allow maximum GPU utilization)
critical_threshold:    99%   (batch size reduction trigger)
emergency_threshold:   99%   (emergency cleanup trigger)
memory_headroom:       1GB   (reserved for safety)
```

**Note**: Thresholds are very aggressive (99%) to maximize GPU utilization, but only 1GB headroom may be insufficient for large models.

### GPU Cache Clearing
```python
- Called before validation (clears fragmentation)
- Called during OOM recovery (aggressive mode)
- Called after validation (prevents cascade effect)
- Time cost: 100-500ms depending on memory fragmentation
```

---

## Dynamic Adaptations

### Prefetch Factor (Auto-Adjusted)
```
Formula: prefetch_factor = max(1, min(4, 2048 / max_length))

seq_length ≤ 512   → prefetch=4  (full prefetch)
seq_length = 1024  → prefetch=2  (balanced)
seq_length = 2048  → prefetch=1  (minimal)
seq_length ≥ 4096  → prefetch=1  (minimal, saves 5-10GB)

Memory saved: (original - new) × workers × batch × length × 2 bytes
Example: 4→1, 6 workers, batch=32, len=4096
  Saves: (4-1) × 6 × 32 × 4096 × 2 = 4.7GB
```

### Adaptive Prefetch Depth (Expert Offloading)
```
Hit rate < 70%  → increase depth (need more lookahead)
Hit rate > 90%  → decrease depth (reduce overhead)
Adjustment frequency: every 100 accesses
Depth range: 1-5 experts ahead

Tracks: _prefetch_hit_count, _prefetch_miss_count
Adjusts: _current_prefetch_depth dynamically
```

### Predictive Expert Prefetch
```
Tracks access patterns: which experts follow which
History size: last 1000 expert transitions
Prediction k: top 3 likely next experts
Use case: detect training loop patterns, prefetch predictively

Example: If expert[0] → expert[2] → expert[4] pattern detected,
prefetch expert[2] + [4] before expert[0] is processed
```

---

## Bottleneck Summary

| Bottleneck | Location | Severity | Latency | Mitigation |
|-----------|----------|----------|---------|-----------|
| CPU buffer accumulation | `dataloader.py:931` | Medium | ~50ms | Streaming tokenization |
| DataLoader prefetch | `dataloader.py:1120` | High | N/A (memory) | Dynamic prefetch factor |
| Expert transfer latency | `offloaded_experts.py:521` | Medium | 27-108ms | INT8 quantization (4x faster) |
| Sync cache clearing | `trainer.py:2281` | Low | 100-500ms | Async cleanup thread |
| Reactive OOM handling | `trainer.py:1771` | High | Restart cost | Proactive batch reduction |

---

## Configuration Recommendations

### For 24GB GPU (Small MoE)
```yaml
batch_size: 16
gradient_accumulation: 4
num_workers: 4                    # Reduced from 6
buffer_size: 10000               # Reduced from 15000
prefetch_factor: 2               # Auto-adjusts dynamically
use_lora_experts: true           # 40-60% memory savings
use_expert_offloading: false     # Not needed with LoRA
gradient_checkpointing: true     # 25% activation memory
```

### For 40GB GPU+ (Large MoE)
```yaml
batch_size: 32
gradient_accumulation: 2
num_workers: 8
buffer_size: 20000
prefetch_factor: 4               # Full prefetch
use_lora_experts: true
use_expert_offloading: true      # 75-87% expert memory
use_expert_quantization: false   # INT8 if >1B experts
gradient_checkpointing: true
```

### For Memory-Constrained Systems
```yaml
batch_size: 8
gradient_accumulation: 8
num_workers: 2                    # Minimize worker overhead
buffer_size: 5000                # Aggressive buffering
prefetch_factor: 1               # Minimal prefetch
use_lora_experts: true
use_expert_offloading: true
use_expert_quantization: true    # INT8 for 75% memory
gradient_checkpointing: true
offload_optimizer: true          # CPU optimizer state
```

---

## Key Code Patterns

### Memory-Aware Batch Processing
```python
# Dynamic token batching (15-20% less padding)
max_tokens = 8192
batch = []
current_tokens = 0

# Results in variable batch sizes but fixed token count
# Benefits: reduced padding waste, efficient GPU utilization
```

### Length-Based Bucketing
```python
# Group sequences by length (minimize padding within batch)
bucket_boundaries = [64, 128, 256, 512, 1024, 2048, 4096]
max_bucket_size = 200
# Groups similar lengths together → less padding waste
```

### Expert LRU Cache
```python
# Keep only max_active_experts on GPU
_training_cache: Dict[int, nn.Module] = {}
_cache_access_order: List[int] = []

# LRU eviction: pop(0) removes least recently used
# Prevents OOM from unbounded cache growth
```

### Async Expert Prefetch
```python
# Non-blocking H2D transfer with CUDA streams
with torch.cuda.stream(prefetch_stream):
    param.data = param.data.to(device, non_blocking=True)

# Computation can overlap with transfer
# Requires stream sync before expert use
```

---

## Memory Profiling Commands

```python
# Current GPU memory
allocated = torch.cuda.memory_allocated() / 1e9  # GB
reserved = torch.cuda.memory_reserved() / 1e9    # GB
utilization = allocated / reserved * 100         # %

# Clear cache (blocking)
torch.cuda.empty_cache()

# Reset peak memory tracking
torch.cuda.reset_peak_memory_stats()

# Get peak memory used in current process
peak = torch.cuda.max_memory_allocated() / 1e9   # GB
```

---

## Performance Impact Summary

| Optimization | Memory Savings | Speed Impact | Complexity |
|---|---|---|---|
| Dynamic token batching | 15-20% | +5-10% | Low |
| Length bucketing | 5-10% | +3-5% | Low |
| Gradient checkpointing | 25% | -8% (recompute) | Medium |
| LoRA experts | 40-60% | -5% (smaller params) | Medium |
| Expert offloading | 75-87% | -20-40% (transfers) | High |
| INT8 quantization | 75% | -10-15% | High |
| Adaptive prefetch | 5-10% (indirect) | +15-35% (better prefetch) | Medium |
| Memory pinning | 0% (memory) | +20% (faster H2D) | Low |

---

## Debug/Monitoring Points

### Memory Leaks
- Check `buffer = []` accumulation in dataloader
- Verify expert cache size bounded by `_max_cache_size`
- Monitor `_training_cache` not growing unbounded

### Transfer Bottlenecks
- Profile expert transfer latency with `torch.cuda.Event()`
- Check prefetch hit/miss rate: `_prefetch_hit_count / (_prefetch_hit_count + _prefetch_miss_count)`
- Verify adaptive prefetch depth increasing when hit_rate < 70%

### Cache Fragmentation
- Monitor `torch.cuda.memory_reserved() / torch.cuda.memory_allocated()` ratio
- High ratio (>2.0) indicates fragmentation
- Trigger `torch.cuda.empty_cache()` when >95% of reserved is allocated

---

## References in Code

**Memory Monitor Initialization**:
- `trainer.py:369` - MemoryMonitor setup
- `trainer.py:318-360` - Memory config loading

**Gradient Accumulation**:
- `trainer.py:2652-2668` - Gradient accumulation setup
- `trainer.py:3009-3019` - Optimizer step with cache clearing

**Expert Offloading**:
- `offloaded_experts.py:501-522` - Multi-stage async prefetch
- `offloaded_experts.py:670-686` - Adaptive prefetch depth adjustment

**DataLoader Memory**:
- `dataloader.py:1177-1189` - Dynamic prefetch factor calculation
- `dataloader.py:862-900` - Efficient collate function with pre-allocated tensors

