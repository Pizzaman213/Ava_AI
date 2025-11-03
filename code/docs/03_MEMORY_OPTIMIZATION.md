# Memory Optimization Guide

Comprehensive guide to memory management and optimization in the Ava MoE++ Training Framework.

## Table of Contents

1. [Overview](#overview)
2. [Memory Optimizations Enabled](#memory-optimizations-enabled)
3. [Configuration Guide](#configuration-guide)
4. [Memory Monitoring](#memory-monitoring)
5. [Troubleshooting OOM Errors](#troubleshooting-oom-errors)
6. [Advanced Optimization Techniques](#advanced-optimization-techniques)
7. [Performance Trade-offs](#performance-trade-offs)
8. [GPU-Specific Recommendations](#gpu-specific-recommendations)

---

## Overview

The Ava training framework implements multiple memory optimization techniques to enable training large models on consumer and professional GPUs. With proper configuration, you can achieve **2-3x memory reduction** compared to naive implementations.

### Memory Savings Summary

| Optimization | Memory Savings | Enabled By Default |
|--------------|----------------|-------------------|
| Gradient Checkpointing | 60-80% | ✅ Yes |
| Flash Attention | 50-70% | ✅ Yes |
| Mixed Precision (FP16/BF16) | 50% | ✅ Yes |
| DeepSpeed ZeRO Stage 1 | 1.5x | ✅ Yes |
| DeepSpeed ZeRO Stage 2 | 2x | ⏸️ Optional |
| DeepSpeed ZeRO Stage 3 | 4x | ⏸️ Optional |
| Activation Checkpointing | 30-50% | ✅ Yes |
| CPU Offloading | 2-8x | ❌ No |

**Combined Effective Savings**: ~70-80% total memory reduction with default settings.

---

## Memory Optimizations Enabled

### 1. Gradient Checkpointing

**What it does**: Instead of storing all intermediate activations during the forward pass, gradient checkpointing only stores activations at certain layers (checkpoints). During the backward pass, it recomputes the missing activations on-the-fly.

**Memory Savings**: 60-80% of activation memory
**Speed Impact**: 10-20% slower (recomputation overhead)
**Trade-off**: Worth it! Enables 2-3x larger batch sizes, offsetting the slowdown.

**Configuration**:
```yaml
# config.yaml
output:
  gradient_checkpointing: true  # Enable in output section

# AND/OR in model section
model:
  deepspeed_activation_checkpointing: true

# AND/OR in DeepSpeed config
training:
  deepspeed:
    activation_checkpointing: true
```

**When to use**:
- ✅ **Always** for models >500M parameters
- ✅ When batch size is memory-limited
- ✅ For long sequences (>512 tokens)
- ❌ For tiny models (<100M params) where overhead matters

### 2. Flash Attention

**What it does**: Implements memory-efficient attention using tiling and recomputation. Reduces attention memory complexity from O(n²) to O(n).

**Memory Savings**: 50-70% of attention memory
**Speed Impact**: 2-4x **faster** than standard attention!
**Trade-off**: Win-win! Saves memory AND speeds up training.

**Configuration**:
```yaml
# config.yaml
model:
  use_flash_attention: true  # Enabled by default
```

**Requirements**:
- CUDA GPU with compute capability ≥7.5 (V100, A100, RTX 20xx+)
- Install `flash-attn` package: `pip install flash-attn`

**When to use**:
- ✅ **Always** if your GPU supports it
- ✅ Critical for long sequences (>1024 tokens)
- ❌ Only if flash-attn is not available on your system

### 3. Mixed Precision Training

**What it does**: Uses FP16 (float16) or BF16 (bfloat16) instead of FP32 for parameters, activations, and gradients.

**Memory Savings**: 50% for model weights and activations
**Speed Impact**: 2-3x faster on modern GPUs (Tensor Cores)
**Trade-off**: Minimal precision loss with proper loss scaling.

**Configuration**:
```yaml
# config.yaml
training:
  mixed_precision: fp16  # or bf16 for A100/H100
  fp16: true
  bf16: false  # Use bf16 instead for better numerical stability on A100+
```

**When to use**:
- ✅ **Always** on GPUs with Tensor Cores (V100, A100, RTX 20xx+)
- ✅ Use FP16 for most GPUs
- ✅ Use BF16 for A100/H100 (better numerical range)
- ❌ Use FP32 only for debugging numerical issues

### 4. DeepSpeed ZeRO

**What it does**: Partitions optimizer states, gradients, and optionally parameters across GPUs.

**Stages**:
- **Stage 1**: Partition optimizer states (1.5x memory reduction)
- **Stage 2**: Partition optimizer + gradients (2x memory reduction)
- **Stage 3**: Partition optimizer + gradients + parameters (4x memory reduction)

**Configuration**:
```yaml
# config.yaml
training:
  deepspeed:
    use_deepspeed: true
    zero_stage: 1  # Start with Stage 1, increase if needed
    cpu_offload: false  # Enable for extreme memory savings
    nvme_offload: false  # Offload to SSD (slower but huge memory)
```

**When to use**:
- Stage 1: ✅ Default for all multi-GPU training
- Stage 2: ✅ When Stage 1 still OOMs
- Stage 3: ✅ For very large models (>7B params) or limited GPU memory
- CPU offload: ⚠️ Only as last resort (10-20x slower)

### 5. Data Pipeline Optimization

**What it does**: Reduces memory used by data prefetching and buffering.

**Memory Savings**: 50% data pipeline memory
**Speed Impact**: Minimal (still prefetches 2 batches ahead)

**Configuration**:
```yaml
# config.yaml
data:
  prefetch_factor: 2  # Reduced from 4 (default)
  num_workers: 8  # Balance CPU cores vs memory
  buffer_size: 30000  # Reduce for long sequences
  persistent_workers: true  # Keep workers alive (saves startup time)
```

**Optimization Formula**:
```
Data Memory = num_workers × prefetch_factor × batch_size × seq_length × 2 bytes

Example (seq_length=256):
8 workers × 2 prefetch × 12 batch × 256 tokens × 2 bytes = ~1.5 GB

Example (seq_length=2048):
8 workers × 2 prefetch × 12 batch × 2048 tokens × 2 bytes = ~12 GB
```

**Recommendations**:
- Reduce `prefetch_factor` to 1 for very long sequences (>2048)
- Reduce `num_workers` if CPU memory is constrained
- Reduce `buffer_size` for streaming datasets with long sequences

### 6. Memory Thresholds

**What it does**: Proactively manages memory to prevent OOM crashes.

**Configuration**:
```yaml
# config.yaml
training:
  memory:
    target_utilization: 0.90  # Target 90% GPU memory usage
    warning_threshold: 0.92  # Warn at 92%
    critical_threshold: 0.95  # Reduce batch size at 95%
    emergency_threshold: 0.97  # Aggressive cleanup at 97%
    silent_mode: true  # Suppress frequent warnings
```

**Aggressive vs Conservative**:

| Profile | Target | Warning | Critical | Emergency | Use Case |
|---------|--------|---------|----------|-----------|----------|
| **Aggressive** | 0.92 | 0.94 | 0.96 | 0.98 | Max performance, stable model |
| **Balanced** (default) | 0.90 | 0.92 | 0.95 | 0.97 | Production training |
| **Conservative** | 0.85 | 0.88 | 0.92 | 0.95 | Experimental, unstable model |

---

## Configuration Guide

### Quick Start: Optimal Settings

For most users, these settings provide the best balance of memory efficiency and speed:

```yaml
# config.yaml - Optimal Memory Settings

model:
  use_flash_attention: true
  deepspeed_activation_checkpointing: true

training:
  mixed_precision: fp16  # or bf16 for A100/H100
  gradient_checkpointing: true

  deepspeed:
    use_deepspeed: true
    zero_stage: 1  # Increase to 2 or 3 if still OOM
    activation_checkpointing: true

  memory:
    target_utilization: 0.90
    warning_threshold: 0.92
    critical_threshold: 0.95
    emergency_threshold: 0.97

data:
  prefetch_factor: 2
  num_workers: 8
  buffer_size: 30000  # Reduce for long sequences

output:
  gradient_checkpointing: true
```

### GPU-Specific Configurations

#### RTX 3090/4090 (24GB)
```yaml
training:
  batch_size: 12
  gradient_accumulation_steps: 4
  mixed_precision: fp16

  memory:
    target_utilization: 0.90
    pool_size_gb: 24

model:
  use_flash_attention: true
```

#### A100 80GB
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  mixed_precision: bf16  # BF16 is better on A100

  memory:
    target_utilization: 0.92
    pool_size_gb: 80

model:
  use_flash_attention: true
```

#### H100 80GB
```yaml
training:
  batch_size: 6
  gradient_accumulation_steps: 8
  mixed_precision: bf16  # BF16 + FP8 support

  memory:
    target_utilization: 0.92
    pool_size_gb: 80

model:
  use_flash_attention: true
```

---

## Memory Monitoring

### Using the Memory Monitor

The framework includes comprehensive memory monitoring in [`code/src/Ava/training/memory/memory_monitor.py`](../src/Ava/training/memory/memory_monitor.py).

#### Basic Usage

```python
from Ava.training.memory import MemoryMonitor

# Initialize monitor
monitor = MemoryMonitor(
    target_utilization=0.90,
    warning_threshold=0.92,
    critical_threshold=0.95,
    emergency_threshold=0.97,
    silent_mode=False
)

# Check memory health
health = monitor.check_memory_health(batch_size=12)
print(f"Status: {health['status']}")  # normal, warning, critical, emergency
print(f"GPU Usage: {health['gpu_utilization']:.1%}")
print(f"Recommendations: {health['recommendations']}")

# Get detailed breakdown
breakdown = monitor.get_detailed_memory_breakdown()
print(f"Allocated: {breakdown['total']['allocated_gb']:.2f} GB")
print(f"Reserved: {breakdown['total']['reserved_gb']:.2f} GB")
print(f"Fragmentation: {breakdown['fragmentation']['ratio']:.1%}")

# Estimate activation memory
estimates = monitor.estimate_activation_memory(
    batch_size=12,
    sequence_length=256,
    hidden_size=512,
    num_layers=14,
    num_attention_heads=8,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)
print(f"Activation Memory: {estimates['effective_memory_gb']:.2f} GB")
print(f"Savings from checkpointing: {estimates['savings_gb']:.2f} GB")

# Track enabled optimizations
optimizations = monitor.track_memory_optimizations(config)
for opt_name, opt_info in optimizations.items():
    print(f"{opt_name}: {opt_info['enabled']} - saves {opt_info['memory_savings']}")
```

### Memory Dashboard

During training, the framework automatically logs memory statistics:

```
[Step 100] Memory: 18.5/24.0 GB (77%), Status: normal
[Step 200] Memory: 21.2/24.0 GB (88%), Status: normal
[Step 300] Memory: 22.8/24.0 GB (95%), Status: critical - reducing batch size
[Step 400] Memory: 20.1/24.0 GB (84%), Status: normal - batch size restored
```

---

## Troubleshooting OOM Errors

### Step-by-Step OOM Resolution

If you encounter "CUDA out of memory" errors:

#### 1. Enable All Standard Optimizations
```yaml
# Ensure these are all enabled:
gradient_checkpointing: true
use_flash_attention: true
mixed_precision: fp16  # or bf16
activation_checkpointing: true
prefetch_factor: 2
```

#### 2. Reduce Batch Size
```yaml
training:
  batch_size: 12  # Try 8, 6, 4, 2
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

#### 3. Reduce Sequence Length
```yaml
data:
  max_length: 256  # Try 128, 64 for initial testing
```

#### 4. Enable DeepSpeed ZeRO Stage 2/3
```yaml
training:
  deepspeed:
    zero_stage: 2  # or 3 for extreme cases
```

#### 5. Reduce Model Size
```yaml
model:
  num_layers: 14  # Try 12, 10, 8
  hidden_size: 512  # Try 384, 256
  num_experts: 8  # Try 4
```

#### 6. Last Resort: CPU Offloading
```yaml
training:
  deepspeed:
    zero_stage: 3
    cpu_offload: true  # Slow but prevents OOM

  memory:
    enable_cpu_offload: true
```

### Common OOM Causes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM at step 0 | Model too large for GPU | Reduce model size or enable ZeRO Stage 3 |
| OOM after N steps | Memory leak or fragmentation | Enable `clear_cache_frequency`, reduce batch size |
| OOM during eval | Eval batch too large | Reduce `eval_batch_size` |
| OOM with long sequences | Attention memory O(n²) | Enable flash attention, reduce `max_length` |
| Intermittent OOM | Memory fragmentation | Restart training, enable `clear_cache_frequency: 50` |

---

## Advanced Optimization Techniques

### 1. Dynamic Batch Sizing

Automatically adjust batch size based on available memory:

```yaml
training:
  dynamic_batching:
    enabled: true
    min_batch_size: 4
    max_batch_size: 16
    target_memory_utilization: 0.90
    adjustment_frequency: 100
    smooth_transitions: true
```

### 2. Gradient Accumulation Strategies

Balance memory and optimization stability:

```yaml
# Small GPU (12-16 GB): Use small batch + high accumulation
training:
  batch_size: 4
  gradient_accumulation_steps: 16  # Effective batch: 64

# Large GPU (40-80 GB): Use larger batch + low accumulation
training:
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective batch: 32
```

### 3. Activation Recomputation

Fine-tune which layers recompute activations:

```python
# In model code
model.gradient_checkpointing_enable()

# Or use selective checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_func=custom_checkpoint_fn)
```

### 4. Memory-Efficient Optimizers

Use optimizers with lower memory footprint:

```yaml
# AdamW: 2x memory (stores momentum + variance)
# Standard choice

# Adafactor: 1x memory (factorized second moments)
# Lower memory but may converge slower

# 8-bit Adam: 0.5x memory (quantized optimizer states)
training:
  optimizer: adamw_8bit  # Requires bitsandbytes package
```

### 5. Sparse Attention

For very long sequences (>2048), use sparse attention patterns:

```yaml
model:
  use_flash_attention: true
  attention_type: flash  # or 'sparse', 'sliding_window'
  sparse_block_size: 64  # For sparse attention
  num_local_blocks: 4  # Sliding window size
```

---

## Performance Trade-offs

### Speed vs Memory Table

| Optimization | Memory Saved | Speed Impact | Recommended |
|--------------|--------------|--------------|-------------|
| Gradient Checkpointing | ⬇️⬇️⬇️ 60-80% | ⬇️ -10 to -20% | ✅ Yes |
| Flash Attention | ⬇️⬇️ 50-70% | ⬆️ +200% to +400% | ✅ Yes |
| Mixed Precision FP16 | ⬇️⬇️ 50% | ⬆️ +200% | ✅ Yes |
| DeepSpeed ZeRO-1 | ⬇️ 33% | ≈ 0% | ✅ Yes |
| DeepSpeed ZeRO-2 | ⬇️⬇️ 50% | ⬇️ -5% | ✅ Multi-GPU |
| DeepSpeed ZeRO-3 | ⬇️⬇️⬇️ 75% | ⬇️ -15% | ⚠️ If needed |
| CPU Offload | ⬇️⬇️⬇️⬇️ 90% | ⬇️⬇️⬇️ -1000% | ❌ Last resort |
| Reduced Batch Size | ⬇️ Variable | ⬇️ Variable | ⚠️ As needed |

**Legend**: ⬆️ Faster, ⬇️ Slower/Less, ≈ No change

### Optimization Priority

1. **Enable Flash Attention** (saves memory AND speeds up)
2. **Enable Mixed Precision** (saves memory AND speeds up)
3. **Enable Gradient Checkpointing** (big memory save, small speed cost)
4. **Optimize Data Pipeline** (free memory save)
5. **Use DeepSpeed ZeRO-1** (moderate save, minimal cost)
6. **Increase Memory Thresholds** (free optimization)
7. **Use DeepSpeed ZeRO-2** (if still OOM)
8. **Reduce Batch Size** (last resort)

---

## GPU-Specific Recommendations

### Consumer GPUs

#### RTX 3090 Ti (24GB)
```yaml
training:
  batch_size: 12
  gradient_accumulation_steps: 4
  mixed_precision: fp16

model:
  hidden_size: 512
  num_layers: 14
  use_flash_attention: true

data:
  max_length: 256
  prefetch_factor: 2
```

**Expected Performance**:
- Model Size: ~100M parameters
- Training Speed: ~180 tokens/sec
- Memory Usage: ~21-22 GB (88-92%)

#### RTX 4090 (24GB)
```yaml
# Same as 3090 but:
training:
  batch_size: 16  # Ada architecture more efficient

data:
  max_length: 512  # Can handle longer sequences
```

### Professional GPUs

#### A100 40GB
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  mixed_precision: bf16  # Better on A100

model:
  hidden_size: 768
  num_layers: 16
  use_flash_attention: true

data:
  max_length: 512
```

#### A100 80GB
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  mixed_precision: bf16

model:
  hidden_size: 1024  # Can train larger models
  num_layers: 24
  use_flash_attention: true

data:
  max_length: 2048  # Long context support
```

#### H100 80GB/SXM
```yaml
training:
  batch_size: 6
  gradient_accumulation_steps: 8
  mixed_precision: bf16

model:
  hidden_size: 1024
  num_layers: 32  # H100 can handle even larger
  use_flash_attention: true

data:
  max_length: 4096  # Very long context

# H100-specific optimizations
performance:
  torch_compile:
    enabled: true
    mode: max-autotune  # H100 benefits greatly
  float32_matmul_precision: high  # Use TF32
```

---

## Appendix: Memory Estimation Formulas

### Model Parameters
```
params_memory = num_parameters × bytes_per_param
- FP32: 4 bytes
- FP16/BF16: 2 bytes
- INT8: 1 byte

Example (100M params, FP16):
100M × 2 bytes = 200 MB
```

### Gradients
```
gradient_memory = num_parameters × bytes_per_param
- Same size as parameters

Example (100M params, FP16):
100M × 2 bytes = 200 MB
```

### Optimizer States
```
optimizer_memory = num_parameters × optimizer_multiplier

AdamW (momentum + variance):
100M × 8 bytes = 800 MB (FP32 states)

Adafactor (factorized):
100M × 4 bytes = 400 MB
```

### Activations (without checkpointing)
```
activation_memory = batch_size × sequence_length × hidden_size × num_layers × 4 × bytes_per_activation

Example (batch=12, seq=256, hidden=512, layers=14, FP16):
12 × 256 × 512 × 14 × 4 × 2 = 443 MB per layer
Total: ~6 GB

With gradient checkpointing (save 80%):
~1.2 GB
```

### Attention Memory
```
# Standard Attention: O(batch × seq² × heads)
attention_memory = batch_size × num_heads × sequence_length² × 2 bytes

Example (batch=12, heads=8, seq=256, FP16):
12 × 8 × 256² × 2 = 100 MB per layer

# Flash Attention: O(batch × seq)
flash_attention_memory = batch_size × sequence_length × hidden_size × 2

Example (batch=12, seq=256, hidden=512, FP16):
12 × 256 × 512 × 2 = 3 MB per layer (97% savings!)
```

### Total Training Memory
```
total_memory = (
    model_params
    + gradients
    + optimizer_states
    + activations
    + attention
    + framework_overhead
)

Typical breakdown:
- Parameters: 10%
- Gradients: 10%
- Optimizer: 40%
- Activations: 30%
- Attention: 5%
- Overhead: 5%
```

---

## Summary

**Key Takeaways**:

1. ✅ **Enable gradient checkpointing** - saves 60-80% activation memory
2. ✅ **Enable flash attention** - saves 50-70% attention memory AND speeds up training
3. ✅ **Use mixed precision** - saves 50% parameter memory AND speeds up training
4. ✅ **Optimize data pipeline** - reduce prefetch_factor to 2
5. ✅ **Monitor memory proactively** - use MemoryMonitor for real-time tracking
6. ⚠️ **Increase memory thresholds** - target 90% instead of 75% utilization
7. ⚠️ **Use DeepSpeed ZeRO** - Stage 1 default, Stage 2/3 if needed

**Expected Results**:
- **2-3x memory reduction** from default settings
- **2-3x larger batch sizes** possible on same hardware
- **Faster training** due to Flash Attention and mixed precision
- **More stable training** with proactive memory management

For more information, see:
- [Architecture Guide](./01_ARCHITECTURE.md) - System design and components
- [Training Guide](./02_TRAINING_GUIDE.md) - Complete training walkthrough
- [Memory Monitor Source](../src/Ava/training/memory/memory_monitor.py) - Implementation details
