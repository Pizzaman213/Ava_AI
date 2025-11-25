# Ava Pipeline Optimization Configuration Guide

**Version**: 1.0.0
**Date**: 2025-11-08
**Target**: Ava MoE Training Pipeline

---

##  Overview

This guide explains how to configure all optimization settings in the Ava pipeline. All optimizations implemented in Phases 1-3 are now fully configurable through YAML config files.

---

##  Quick Start

All optimization settings are in the `optimizations` section of your config file:

```yaml
optimizations:
  gradient_health_monitoring: false
  torchinductor_autotune: 1
  memory_cleanup_thresholds:
    warning: 0.990
    critical: 0.995
    emergency: 0.999
  expert_prefetch:
    enabled: true
    lookahead: 2
  # ... more settings
```

---

##  Complete Configuration Reference

### Phase 1: Quick Wins

#### 1.1 Gradient Health Monitoring
**Impact**: 5-8% speedup when disabled
**Default**: `false`

```yaml
optimizations:
  gradient_health_monitoring: false  # Set to true to enable monitoring
```

**Also configure in**:
```yaml
gradient_health:
  enabled: false  # Must match optimizations.gradient_health_monitoring
```

#### 1.2 TorchInductor Auto-Tuning
**Impact**: 10-15% speedup
**Default**: `1`

```yaml
optimizations:
  torchinductor_autotune: 1  # 0=off, 1=basic, 2=aggressive

# Also set in performance section:
performance:
  torchinductor_max_autotune: 1
```

**Options**:
- `0`: Disabled (no auto-tuning)
- `1`: Basic auto-tuning (recommended, ~30s warmup)
- `2`: Aggressive auto-tuning (~1-2min warmup, slightly better performance)

#### 1.3 Memory Cleanup Thresholds
**Impact**: 5-10% speedup (50% less cleanup overhead)
**Defaults**: See below

```yaml
optimizations:
  memory_cleanup_thresholds:
    warning: 0.990   # Trigger warning at 99.0% GPU memory
    critical: 0.995  # Trigger critical cleanup at 99.5%
    emergency: 0.999 # Trigger emergency cleanup at 99.9%
```

**Tuning Guidelines**:
- **Conservative** (more frequent cleanups, safer for unstable training):
  ```yaml
  warning: 0.985
  critical: 0.990
  emergency: 0.995
  ```

- **Balanced** (recommended, default):
  ```yaml
  warning: 0.990
  critical: 0.995
  emergency: 0.999
  ```

- **Aggressive** (fewer cleanups, maximum speed):
  ```yaml
  warning: 0.995
  critical: 0.998
  emergency: 0.999
  ```

---

### Phase 2: Expert Offloading Optimizations

#### 2.1 Multi-Stage Expert Prefetch
**Impact**: 25-35% speedup with CPU offloading
**Defaults**: See below

```yaml
optimizations:
  expert_prefetch:
    enabled: true              # Enable multi-stage async prefetch
    lookahead: 2               # Number of experts to prefetch ahead (1-3)
    use_multiple_streams: true # Use multiple CUDA streams
```

**Tuning Guidelines**:

**For CPU Offloading Scenarios**:
```yaml
expert_prefetch:
  enabled: true   # CRITICAL for offloading performance
  lookahead: 2    # 2-3 for balanced systems
  use_multiple_streams: true
```

**For All-GPU Scenarios**:
```yaml
expert_prefetch:
  enabled: false  # Not needed without offloading
  lookahead: 0
  use_multiple_streams: false
```

**Low Memory Systems**:
```yaml
expert_prefetch:
  enabled: true
  lookahead: 1    # Reduce lookahead to save memory
  use_multiple_streams: false
```

#### 2.2 Expert Cache Management
**Impact**: Prevents OOM, stable memory usage
**Defaults**: Both `true`

```yaml
optimizations:
  expert_cache:
    auto_limit: true           # Automatically limit cache size
    use_lru_eviction: true     # LRU eviction when cache is full
```

**When to disable**:
- `auto_limit: false` - If you want manual cache management
- `use_lru_eviction: false` - If you prefer FIFO or no eviction

**Recommended**: Keep both enabled for stability

#### 2.3 Async Checkpoint Saving
**Impact**: Saves 20-30 seconds per checkpoint
**Default**: `true`

```yaml
optimizations:
  checkpoint:
    async_saving: true  # Save checkpoints in background thread
```

**Usage in code**:
```python
# Async (non-blocking, returns immediately)
trainer.save_checkpoint_async(checkpoint_dir, tag="step_1000")

# Sync (blocks until complete)
trainer.save_checkpoint(checkpoint_dir, tag="step_1000")

# Before program exits, wait for async checkpoint
trainer._wait_for_checkpoint()
```

**When to disable**:
- Debugging checkpoint issues
- Need guaranteed checkpoint completion before proceeding
- Distributed training with strict synchronization requirements

#### 2.4 GPU Memory Cleanup
**Impact**: Saves ~2.5 seconds per cleanup
**Defaults**: Both `true`

```yaml
optimizations:
  memory_cleanup:
    fast_mode: true      # Reduced cleanup rounds (2 vs 5, 1 vs 3)
    remove_sleep: true   # Remove sleep between cleanup rounds
```

**Conservative mode** (for unstable training):
```yaml
memory_cleanup:
  fast_mode: false
  remove_sleep: false
```

---

### Phase 2: Data Loading Optimizations

#### 2.5 Adaptive File Reading
**Impact**: 20-30% faster I/O for mixed file sizes
**Defaults**: See below

```yaml
optimizations:
  dataloader:
    adaptive_file_reading: true    # Adjust samples per file based on size
    adaptive_multipliers:
      large_files: 4      # Multiplier for files >10MB
      medium_files: 2     # Multiplier for files >1MB
      small_files: 1      # Multiplier for files <1MB
```

**Tuning Guidelines**:

**For Large Files (>10MB average)**:
```yaml
adaptive_multipliers:
  large_files: 8   # Read even more to reduce I/O
  medium_files: 4
  small_files: 2
```

**For Small Files (<1MB average)**:
```yaml
adaptive_multipliers:
  large_files: 2   # Maintain more diversity
  medium_files: 2
  small_files: 1
```

**Uniform File Sizes** (disable adaptive):
```yaml
dataloader:
  adaptive_file_reading: false
```

---

### Phase 2: Gradient Checkpointing

#### 2.6 Selective Gradient Checkpointing
**Impact**: 25% memory savings with 8% slowdown
**Defaults**: Selective enabled

```yaml
optimizations:
  gradient_checkpointing:
    selective: true            # Only checkpoint MoE/FFN layers
    checkpoint_attention: false # Set to true for full checkpointing
```

**Configuration Options**:

**Selective (Recommended)**:
```yaml
gradient_checkpointing:
  selective: true
  checkpoint_attention: false
```
- **Memory**: 25% savings
- **Speed**: 8% slowdown
- **Best for**: Balanced training with MoE models

**Full Checkpointing**:
```yaml
gradient_checkpointing:
  selective: false
  checkpoint_attention: true
```
- **Memory**: 30% savings
- **Speed**: 15% slowdown
- **Best for**: Maximum memory savings

**No Checkpointing** (also disable in training section):
```yaml
training:
  gradient_checkpointing: false

optimizations:
  gradient_checkpointing:
    selective: false
    checkpoint_attention: false
```
- **Memory**: 0% savings
- **Speed**: Fastest
- **Best for**: GPUs with plenty of memory

---

### Router Optimizations

#### 2.7 Router Cache and Compilation
**Impact**: 15-20% faster eval, 20-30% faster routing
**Defaults**: All `true`, mode `"default"`

```yaml
optimizations:
  router:
    cache_hash_on_gpu: true    # Compute cache hash on GPU (no CPU sync)
    compile_routers: true      # torch.compile routers
    compile_mode: "default"    # "default" or "reduce-overhead"
    compile_dynamic: true      # Handle variable sequence lengths
```

**Tuning Guidelines**:

**For Fixed Batch/Sequence Sizes**:
```yaml
router:
  cache_hash_on_gpu: true
  compile_routers: true
  compile_mode: "reduce-overhead"  # More aggressive optimization
  compile_dynamic: false           # Disable dynamic shapes
```

**For Variable Sizes (Recommended)**:
```yaml
router:
  cache_hash_on_gpu: true
  compile_routers: true
  compile_mode: "default"  # Better for variable inputs
  compile_dynamic: true
```

**Debugging / Development**:
```yaml
router:
  cache_hash_on_gpu: false  # Easier to debug
  compile_routers: false    # Faster startup, easier debugging
  compile_mode: "default"
  compile_dynamic: true
```

---

##  Preset Configurations

### Maximum Speed (Production)
```yaml
optimizations:
  gradient_health_monitoring: false
  torchinductor_autotune: 1
  memory_cleanup_thresholds:
    warning: 0.995
    critical: 0.998
    emergency: 0.999
  expert_prefetch:
    enabled: true
    lookahead: 3
    use_multiple_streams: true
  expert_cache:
    auto_limit: true
    use_lru_eviction: true
  checkpoint:
    async_saving: true
  memory_cleanup:
    fast_mode: true
    remove_sleep: true
  dataloader:
    adaptive_file_reading: true
    adaptive_multipliers:
      large_files: 8
      medium_files: 4
      small_files: 2
  gradient_checkpointing:
    selective: true
    checkpoint_attention: false
  router:
    cache_hash_on_gpu: true
    compile_routers: true
    compile_mode: "default"
    compile_dynamic: true
```

### Maximum Stability (Development/Debugging)
```yaml
optimizations:
  gradient_health_monitoring: true  # Enable monitoring
  torchinductor_autotune: 0         # Disable for faster startup
  memory_cleanup_thresholds:
    warning: 0.985
    critical: 0.990
    emergency: 0.995
  expert_prefetch:
    enabled: false  # Simplify for debugging
    lookahead: 0
    use_multiple_streams: false
  expert_cache:
    auto_limit: true
    use_lru_eviction: true
  checkpoint:
    async_saving: false  # Synchronous for debugging
  memory_cleanup:
    fast_mode: false
    remove_sleep: false
  dataloader:
    adaptive_file_reading: false
  gradient_checkpointing:
    selective: false
    checkpoint_attention: false
  router:
    cache_hash_on_gpu: false
    compile_routers: false
    compile_mode: "default"
    compile_dynamic: true
```

### Ultra-Low Memory
```yaml
optimizations:
  gradient_health_monitoring: false
  torchinductor_autotune: 1
  memory_cleanup_thresholds:
    warning: 0.980   # More aggressive cleanup
    critical: 0.985
    emergency: 0.990
  expert_prefetch:
    enabled: true
    lookahead: 1     # Minimal lookahead
    use_multiple_streams: false
  expert_cache:
    auto_limit: true  # CRITICAL
    use_lru_eviction: true
  checkpoint:
    async_saving: true
  memory_cleanup:
    fast_mode: false  # More thorough cleanup
    remove_sleep: false
  dataloader:
    adaptive_file_reading: true
    adaptive_multipliers:
      large_files: 2  # Smaller multipliers
      medium_files: 1
      small_files: 1
  gradient_checkpointing:
    selective: true    # Save memory
    checkpoint_attention: true  # Checkpoint everything
  router:
    cache_hash_on_gpu: true
    compile_routers: true
    compile_mode: "default"
    compile_dynamic: true
```

---

##  Configuration Impact Matrix

| Setting | Speed Impact | Memory Impact | Stability | Complexity |
|---------|-------------|---------------|-----------|------------|
| `gradient_health_monitoring: false` | +5-8% | 0% | Slightly lower | Low |
| `torchinductor_autotune: 1` | +10-15% | 0% | High | Low |
| `memory_cleanup_thresholds` (optimized) | +5-10% | 0% | Medium | Low |
| `expert_prefetch.enabled: true` | +25-35%* | 0% | High | Medium |
| `expert_cache.auto_limit: true` | Stability | Prevents OOM | Very High | Low |
| `checkpoint.async_saving: true` | +20-30s saved | 0% | High | Medium |
| `memory_cleanup.fast_mode: true` | +2.5s saved | 0% | Medium | Low |
| `dataloader.adaptive_file_reading: true` | +20-30%** | 0% | High | Low |
| `gradient_checkpointing.selective: true` | -8% | +25% saved | High | Medium |
| `router.cache_hash_on_gpu: true` | +15-20%*** | 0% | High | Low |
| `router.compile_routers: true` | +20-30% | 0% | High | Medium |

\* Only with CPU offloading
\** Only with mixed file sizes
\*** Only during eval/inference

---

##  Monitoring & Validation

### Verify Optimizations Are Active

```python
# Check if optimizations are loaded
import yaml
with open('configs/moe/small_moe.yaml') as f:
    config = yaml.safe_load(f)
    print("Optimizations:", config.get('optimizations', {}))
```

### Monitor Performance Metrics

```python
# In your training loop
import time

# Track throughput
start_time = time.time()
# ... training step ...
step_time = time.time() - start_time
samples_per_sec = batch_size / step_time

# Track memory
allocated_gb = torch.cuda.memory_allocated() / 1e9
reserved_gb = torch.cuda.memory_reserved() / 1e9
utilization = allocated_gb / total_gpu_memory

# Track I/O
data_load_time = data_end - data_start
compute_time = compute_end - compute_start
io_ratio = data_load_time / (data_load_time + compute_time)

print(f"Throughput: {samples_per_sec:.2f} samples/sec")
print(f"Memory: {allocated_gb:.2f}GB / {total_gpu_memory:.2f}GB ({utilization*100:.1f}%)")
print(f"I/O ratio: {io_ratio*100:.1f}%")
```

### Expected Performance Improvements

With all optimizations enabled (Maximum Speed preset):
- **Training speed**: 2.5-3x faster
- **Memory cleanups**: 50% less frequent
- **Checkpoint time**: Non-blocking (<100ms return time)
- **I/O throughput**: 20-30% faster
- **Eval/inference**: 15-20% faster

---

##  Troubleshooting

### Issue: OOM Errors After Enabling Optimizations

**Solution**: Adjust memory cleanup thresholds
```yaml
memory_cleanup_thresholds:
  warning: 0.980   # More aggressive
  critical: 0.985
  emergency: 0.990
```

And/or enable aggressive cleanup:
```yaml
memory_cleanup:
  fast_mode: false
  remove_sleep: false
```

### Issue: Slow First Training Step

**Cause**: TorchInductor auto-tuning warmup

**Solution**: This is normal. First step takes ~30s-2min for compilation, then speeds up significantly.

To reduce warmup time:
```yaml
torchinductor_autotune: 0  # Disable auto-tuning
```

### Issue: Checkpoint Corruption with Async Saving

**Solution**: Ensure checkpoint completes before program exit:
```python
# At end of training
trainer._wait_for_checkpoint()
```

Or disable async saving:
```yaml
checkpoint:
  async_saving: false
```

### Issue: Slower Performance with Optimizations

**Check**:
1. Verify optimizations are actually loaded (see Monitoring section above)
2. Check if GPU supports BF16 (`mixed_precision: bf16`)
3. Ensure TorchInductor warmup has completed (first few steps are slow)
4. Verify expert offloading is enabled if using prefetch

**Try**:
```yaml
# Disable individual optimizations to isolate issue
expert_prefetch:
  enabled: false  # Test without prefetch
router:
  compile_routers: false  # Test without compilation
```

---

##  Related Documentation

- [ALL_PHASES_OPTIMIZATIONS_COMPLETE.md](ALL_PHASES_OPTIMIZATIONS_COMPLETE.md) - Complete optimization details
- [PHASE1_OPTIMIZATIONS_APPLIED.md](PHASE1_OPTIMIZATIONS_APPLIED.md) - Phase 1 details
- [07_CONFIGURATION_SYSTEM.md](07_CONFIGURATION_SYSTEM.md) - General configuration guide
- [03_MEMORY_OPTIMIZATION.md](03_MEMORY_OPTIMIZATION.md) - Memory optimization strategies
- [MOE_OPTIMIZATION_GUIDE.md](MOE_OPTIMIZATION_GUIDE.md) - MoE-specific optimizations

---

##  Quick Checklist

Before training, verify:
- [ ] `optimizations` section exists in your config
- [ ] `gradient_health.enabled` matches `optimizations.gradient_health_monitoring`
- [ ] `performance.torchinductor_max_autotune` matches `optimizations.torchinductor_autotune`
- [ ] Expert prefetch enabled if using CPU offloading
- [ ] Memory thresholds appropriate for your GPU
- [ ] Async checkpointing enabled for large models
- [ ] Adaptive file reading enabled for datasets with mixed file sizes

---

**Last Updated**: 2025-11-08
**Version**: 1.0.0
**Maintained by**: Ava Development Team
