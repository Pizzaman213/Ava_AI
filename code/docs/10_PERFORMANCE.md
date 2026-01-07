# Performance Tuning Guide

Optimization strategies to maximize training throughput and efficiency.

## Performance Overview

Key metrics to optimize:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| GPU Utilization | >90% | `nvidia-smi` |
| Memory Usage | 80-95% | `nvidia-smi` |
| Throughput | tokens/sec | Training logs |
| Time per Step | minimize | Training logs |

## Quick Wins

### 1. Enable TF32 (Ampere+ GPUs)

```yaml
performance:
  enable_tf32: true
```

**Impact**: 2-3x faster matrix operations with minimal precision loss.

### 2. Enable Flash Attention

```yaml
model:
  use_flash_attention: true
```

**Impact**: 2-4x faster attention, O(N) memory vs O(N²).

### 3. Use BF16 Mixed Precision

```yaml
hardware:
  mixed_precision: 'bf16'
```

**Impact**: 2x memory reduction, faster compute.

### 4. Enable cuDNN Benchmark

```yaml
performance:
  enable_cudnn_benchmark: true
```

**Impact**: Auto-selects fastest convolution algorithms.

---

## Data Loading Optimization

### Parallel Workers

```yaml
data:
  num_workers: 8          # Match CPU cores
  prefetch_factor: 4      # Batches ahead to load
  persistent_workers: true  # Don't restart workers
  dataloader_pin_memory: true
```

### Pre-tokenized Data

```bash
# Pre-tokenize once, load fast forever
python code/scripts/1_data_download/build_pretokenized_data.py \
    --output data/pretokenized
```

```yaml
data:
  use_pretokenized: true
  data_dir: "data/pretokenized"
```

### Sequence Packing

Eliminate padding waste:

```yaml
data:
  use_sequence_packing: true
  packing_strategy: 'greedy'
```

**Impact**: Up to 30-50% throughput improvement on variable-length data.

### Memory-Mapped Loading

```yaml
data:
  use_mmap: true
```

**Impact**: Faster initial loading, lower memory footprint.

---

## Model Optimization

### Triton Kernels

```yaml
model:
  use_triton_kernels: true
  use_grouped_gemm: true
```

### Torch Compile

```yaml
model:
  use_torch_compile: true
hardware:
  compile: true
```

**Note**: First epoch slower due to compilation.

### Expert Optimization

```yaml
kernel_optimization:
  use_fused_softmax_topk: true
  use_fused_activations: true
  use_vectorized_capacity: true
```

---

## Memory Optimization

### Gradient Checkpointing

Trade compute for memory:

```yaml
model:
  gradient_checkpointing: true
```

**Impact**: ~60% memory reduction, ~30% slower.

### Expert Offloading

```yaml
moe_memory_optimization:
  use_expert_offloading: true
  max_active_experts_gpu: 4
  offload_async_transfers: true
  offload_pin_memory: true
```

### KV Cache Quantization

```yaml
model:
  quantize_kv_cache: true
```

### Gradient Accumulation

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 8
  # Effective batch: 16 × 8 = 128
```

---

## Training Optimization

### Optimizer Selection

**AdamW** (default): Good convergence, high memory
**8-bit Adam**: 2x optimizer memory reduction
**Lion**: Lower memory, competitive performance

```yaml
optimizer:
  name: 'adamw'
  # or: 'adam8bit', 'lion'
```

### Learning Rate

Use LR finder:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml \
    --run-lr-finder \
    --lr-finder-use-suggested
```

### Warmup Strategy

```yaml
training:
  warmup_steps: 2000
  adaptive_lr:
    scheduler: 'cosine'
    min_lr_ratio: 0.1
```

---

## Distributed Optimization

### Gradient Compression

Reduce communication bandwidth:

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2
  overlap_comm: true
  reduce_bucket_size: 1000000000  # 1GB buckets
```

### Expert Parallelism

```yaml
model:
  expert_parallel_size: 4  # Distribute experts across GPUs
```

### Load Balancing

```yaml
hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'adaptive'
```

---

## Profiling

### Built-in Profiling

```yaml
dev_log:
  enabled: true
  show_step_breakdown: true
  report_interval: 100
```

### Nsight Systems

```bash
nsys profile -o ava_profile \
    python code/scripts/5_training/train_pipeline.py \
        --config config.yaml
```

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    # Training step
    loss = model(batch)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Configuration Profiles

### Maximum Throughput

```yaml
model:
  use_flash_attention: true
  use_triton_kernels: true
  use_grouped_gemm: true
  gradient_checkpointing: false  # Trade memory for speed

hardware:
  mixed_precision: 'bf16'

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
  ultra_fast_mode: true

data:
  num_workers: 8
  persistent_workers: true
  use_sequence_packing: true
```

### Maximum Memory Efficiency

```yaml
model:
  gradient_checkpointing: true
  quantize_kv_cache: true
  use_flash_attention: true

moe_memory_optimization:
  use_expert_offloading: true
  max_active_experts_gpu: 2
  use_lora_experts: true
  lora_rank: 4

hardware:
  mixed_precision: 'bf16'

training:
  gradient_accumulation_steps: 16
```

### Balanced (Recommended)

```yaml
model:
  use_flash_attention: true
  gradient_checkpointing: true
  use_triton_kernels: true

hardware:
  mixed_precision: 'bf16'

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true

data:
  num_workers: 4
  persistent_workers: true
  use_sequence_packing: true

training:
  gradient_accumulation_steps: 4
```

---

## Benchmarks

### Expected Performance

| Config | GPU | Batch | Seq Len | Tokens/sec |
|--------|-----|-------|---------|------------|
| minimal | A100 40GB | 64 | 512 | ~50K |
| large | A100 40GB | 32 | 512 | ~30K |
| large | A100 80GB | 64 | 1024 | ~40K |
| large (4x) | 4×A100 | 128 | 512 | ~100K |

### Bottleneck Analysis

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low GPU util | Data loading | More workers, prefetch |
| High GPU util, slow | Memory bound | Reduce batch, enable checkpoint |
| Variable step time | GC pauses | Reduce object creation |
| First epoch slow | Compilation | Expected with torch.compile |

---

## Hardware Recommendations

### Single GPU

| GPU | VRAM | Best Config |
|-----|------|-------------|
| RTX 3090 | 24GB | minimal + bf16 |
| RTX 4090 | 24GB | large + bf16 + Flash |
| A100 40GB | 40GB | large + Flash |
| A100 80GB | 80GB | large + large batch |
| H100 | 80GB | large + FP8 |

### Multi-GPU

| GPUs | Strategy | Config |
|------|----------|--------|
| 2-4 | DDP | large_Multy |
| 4-8 | DeepSpeed ZeRO-2 | + deepspeed_zero2 |
| 8+ | DeepSpeed ZeRO-3 | + cpu_offload |

---

## Common Pitfalls

### 1. Too Many Workers

```yaml
# Bad: More workers than CPU cores
data:
  num_workers: 32

# Good: Match CPU cores
data:
  num_workers: 8
```

### 2. Ignoring Memory Headroom

```yaml
# Bad: Using 100% GPU memory
training:
  batch_size: 128  # Causes OOM on long sequences

# Good: Leave headroom
training:
  batch_size: 96
```

### 3. Not Using Flash Attention

```yaml
# Bad: Standard attention
model:
  use_flash_attention: false

# Good: Always use on supported hardware
model:
  use_flash_attention: true
```

### 4. Synchronous Checkpointing

```yaml
# Bad: Blocks training
output:
  save_every: 100

# Good: Async saves
optimizations:
  checkpoint:
    async_saving: true
```

---

## Monitoring

### WandB Dashboard

```yaml
wandb:
  use_wandb: true
  wandb_log_freq: 10
```

Track:
- `train/throughput` - Tokens per second
- `system/gpu_utilization` - GPU usage
- `system/gpu_memory` - Memory usage
- `train/step_time` - Time per step

### Console Metrics

```yaml
logging:
  metrics_log_freq: 100
  health_summary_freq: 500
  enable_timing_breakdown: true
```

---

## Next Steps

- [Training Guide](./03_TRAINING_GUIDE.md) - Training workflow
- [Distributed Training](./06_DISTRIBUTED.md) - Multi-GPU setup
- [Troubleshooting](./09_TROUBLESHOOTING.md) - Common issues
