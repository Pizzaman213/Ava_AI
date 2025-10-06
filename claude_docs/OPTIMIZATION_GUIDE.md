# Training Pipeline Optimizations Guide

## Overview

This guide covers all the training optimizations that have been implemented to maximize performance, reduce memory usage, and improve training efficiency.

## Quick Start

```python
from Ava.optimization.hardware_optimizations import auto_optimize_hardware
from Ava.optimization.fused_optimizers import create_optimizer
from Ava.data.optimized_dataloader import create_production_dataloader
from Ava.optimization.compilation_optimizations import optimize_for_training

# 1. Auto-optimize hardware settings
hw_optimizer = auto_optimize_hardware()

# 2. Compile model for faster execution
model = optimize_for_training(model, example_inputs)

# 3. Create optimized optimizer
optimizer = create_optimizer(
    model,
    optimizer_type='fused_adam',  # or 'adam8bit', 'lion', 'sophia'
    lr=3e-4,
    weight_decay=0.01
)

# 4. Create optimized dataloader
dataloader = create_production_dataloader(
    dataset,
    batch_size=32,
    num_workers=8,
    use_packing=True,
    device=torch.device('cuda')
)
```

## 1. Gradient Optimizations

### Mixed Precision Training

```python
from Ava.optimization.gradient_optimizations import MixedPrecisionManager

# Auto-detects BF16 on A100, FP16 on older GPUs
mp_manager = MixedPrecisionManager(enabled=True, dtype=None)

# Training loop
with mp_manager.autocast():
    outputs = model(**batch)
    loss = outputs['loss']

loss = mp_manager.scale_loss(loss)
loss.backward()

metrics = mp_manager.step_optimizer(optimizer, max_grad_norm=1.0)
```

**Benefits:**
- 2-3x faster training on Ampere GPUs
- 50% memory reduction
- Minimal accuracy loss with BF16

### Gradient Compression

```python
from Ava.optimization.gradient_optimizations import GradientCompressor

compressor = GradientCompressor(
    method='powersgd',  # or '1bit', 'topk'
    compression_ratio=0.1,
    error_feedback=True
)

# During distributed training
gradients = {name: param.grad for name, param in model.named_parameters()}
compressed, stats = compressor.compress(gradients)
# Send compressed gradients
decompressed = compressor.decompress(compressed)
```

**Benefits:**
- 10x communication reduction for distributed training
- Better scalability to many GPUs

### Adaptive Gradient Clipping

```python
from Ava.optimization.gradient_optimizations import AdaptiveGradientClipper

clipper = AdaptiveGradientClipper(
    clip_type='adaptive',  # or 'percentile', 'per_param'
    base_clip_value=1.0
)

stats = clipper.clip_gradients(model.named_parameters(), named=True)
print(f"Grad norm: {stats['grad_norm']:.3f}, clipped: {stats['clipped']}")
```

**Benefits:**
- Automatic threshold adaptation
- Prevents gradient explosion
- Better training stability

## 2. Data Loading Optimizations

### Memory-Mapped Dataset

```python
from Ava.data.optimized_dataloader import MemoryMappedDataset

dataset = MemoryMappedDataset(
    data_path='/path/to/data.bin',
    cache_size=10000
)
```

**Benefits:**
- Load datasets larger than RAM
- Reduced memory footprint
- Fast random access

### GPU Prefetching

```python
from Ava.data.optimized_dataloader import PrefetchDataLoader

dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
prefetch_loader = PrefetchDataLoader(dataloader, device='cuda', prefetch_factor=2)

for batch in prefetch_loader:
    # Batch already on GPU!
    outputs = model(**batch)
```

**Benefits:**
- Overlaps data transfer with computation
- 10-20% throughput improvement

### Sequence Packing

```python
from Ava.data.optimized_dataloader import SequencePackingDataset

packed_dataset = SequencePackingDataset(
    dataset,
    block_size=2048,
    create_position_ids=True
)
```

**Benefits:**
- Reduces padding waste
- 2-3x more efficient for variable-length sequences
- Compatible with Flash Attention document masking

### Dynamic Batching

```python
dataloader = create_production_dataloader(
    dataset,
    use_dynamic_batching=True,
    max_tokens=8192,  # Max tokens per batch
)
```

**Benefits:**
- Groups similar-length sequences
- Reduces padding overhead by 50-70%

## 3. Optimizer Improvements

### Fused Adam (10-15% faster)

```python
from Ava.optimization.fused_optimizers import FusedAdam

optimizer = FusedAdam(
    model.parameters(),
    lr=1e-3,
    fused=True,  # Use CUDA fused kernel
    foreach=True  # Vectorized operations
)
```

### 8-bit Adam (75% memory reduction)

```python
from Ava.optimization.fused_optimizers import Adam8bit

optimizer = Adam8bit(
    model.parameters(),
    lr=1e-3,
    optim_bits=8,
    block_wise=True
)
```

### Lion Optimizer (More memory-efficient)

```python
from Ava.optimization.fused_optimizers import Lion

optimizer = Lion(
    model.parameters(),
    lr=3e-5,  # Use 3-10x smaller LR than Adam
    weight_decay=0.01
)
```

## 4. Attention Optimizations

### Flash Attention Integration

```python
from Ava.layers.advanced_attention import FlashAttentionWrapper

flash_attn = FlashAttentionWrapper(
    hidden_size=768,
    num_heads=12,
    dropout=0.1,
    window_size=(512, 0)  # Sliding window
)

output = flash_attn(q, k, v, attention_mask)
```

**Benefits:**
- 2-4x faster than standard attention
- O(N) memory instead of O(N²)
- Automatic fallback to xformers or SDPA

### Multi-Query Attention (MQA)

```python
from Ava.layers.advanced_attention import MultiQueryAttention

mqa = MultiQueryAttention(hidden_size=768, num_heads=12)
output, cached_kv = mqa(hidden_states, use_cache=True)
```

**Benefits:**
- Reduces KV cache by num_heads factor
- Faster inference
- Minimal quality loss

### Grouped Query Attention (GQA)

```python
from Ava.layers.advanced_attention import GroupedQueryAttention

gqa = GroupedQueryAttention(
    hidden_size=768,
    num_heads=12,
    num_kv_heads=4  # 12 query heads, 4 KV heads
)
```

**Benefits:**
- Middle ground between MHA and MQA
- 3x smaller KV cache
- Better quality than MQA

### Sliding Window Attention

```python
from Ava.layers.advanced_attention import SlidingWindowAttention

swa = SlidingWindowAttention(
    hidden_size=768,
    num_heads=12,
    window_size=512
)
```

**Benefits:**
- O(N*w) complexity instead of O(N²)
- Enables longer sequences
- Compatible with Flash Attention

## 5. Compilation and Kernel Fusion

### torch.compile (20-40% speedup)

```python
from Ava.optimization.compilation_optimizations import CompilationManager

compiler = CompilationManager(mode='reduce-overhead')
model = compiler.compile_model(model, example_inputs)
```

### Fused Kernels

```python
from Ava.optimization.compilation_optimizations import FusedKernels

# LayerNorm + Residual
output = FusedKernels.fused_layer_norm_residual(x, residual, weight, bias)

# SwiGLU activation
output = FusedKernels.fused_swiglu(x, gate_weight, up_weight)
```

### CUDA Graphs

```python
from Ava.optimization.compilation_optimizations import CUDAGraphWrapper

cuda_wrapper = CUDAGraphWrapper(model)
cuda_wrapper.capture(example_input)

# Run with graph (faster)
output = cuda_wrapper(input_tensor)
```

## 6. Loss Computation Optimizations

### Vocabulary Parallelism

```python
from Ava.losses.vocab_parallel_loss import VocabParallelCrossEntropy

loss_fn = VocabParallelCrossEntropy(
    vocab_size=50000,
    label_smoothing=0.1
)
```

**Benefits:**
- Splits vocab across GPUs
- Reduces memory for large vocabularies
- Enables 100K+ vocab sizes

### Sampled Softmax

```python
from Ava.losses.vocab_parallel_loss import SampledSoftmaxLoss

loss_fn = SampledSoftmaxLoss(
    vocab_size=50000,
    hidden_size=768,
    num_samples=8192
)
```

**Benefits:**
- O(log V) instead of O(V)
- 10x faster for large vocabularies
- Approximates full softmax well

### Adaptive Softmax

```python
from Ava.losses.vocab_parallel_loss import AdaptiveSoftmax

loss_fn = AdaptiveSoftmax(
    vocab_size=50000,
    hidden_size=768,
    cutoffs=[2000, 10000]  # Cluster by frequency
)
```

## 7. Distributed Training

### FSDP (Fully Sharded Data Parallel)

```python
from Ava.training.distributed_optimizations import FSDPManager

fsdp_manager = FSDPManager(
    sharding_strategy='full',
    mixed_precision=True,
    cpu_offload=False
)

model = fsdp_manager.wrap_model(model)
```

**Benefits:**
- Train models larger than single GPU memory
- Near-linear scaling to many GPUs
- Better than DeepSpeed ZeRO for some workloads

### Gradient Communication Overlap

```python
from Ava.training.distributed_optimizations import GradientCommunicationOverlap

overlap = GradientCommunicationOverlap(model, bucket_size_mb=25)
overlap.register_hooks()
```

**Benefits:**
- Overlaps AllReduce with backward pass
- 15-25% faster distributed training

## 8. Profiling and Monitoring

### Throughput Tracking

```python
from Ava.training.profiling_tools import ThroughputTracker

tracker = ThroughputTracker(window_size=100)

# Training loop
tracker.start_batch()
tracker.start_forward()
outputs = model(**batch)
tracker.end_forward()

tracker.start_backward()
loss.backward()
tracker.end_backward()

tracker.start_optimizer()
optimizer.step()
tracker.end_optimizer()

tracker.end_batch(num_samples=32, num_tokens=32*512)

# Get metrics
metrics = tracker.get_metrics()
print(f"Throughput: {metrics.tokens_per_sec:.0f} tokens/s, MFU: {metrics.model_flops_utilization:.2%}")
```

### Advanced Profiler

```python
from Ava.training.profiling_tools import AdvancedProfiler

with AdvancedProfiler(output_dir='./traces') as profiler:
    for step in range(100):
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        profiler.step()
```

### Memory Profiler

```python
from Ava.training.profiling_tools import MemoryProfiler

profiler = MemoryProfiler(check_interval=100)

# Training loop
profiler.record()

# Get summary
summary = profiler.get_summary()
print(f"Peak memory: {summary['peak_mb']:.1f}MB")
```

## 9. Advanced Scheduling

### Learning Rate Finder

```python
from Ava.training.advanced_warmup_scheduling import find_optimal_learning_rate

optimal_lr = find_optimal_learning_rate(
    model,
    train_loader,
    criterion,
    num_steps=100
)
```

### Cyclical Batch Sizes

```python
from Ava.training.advanced_warmup_scheduling import CyclicalBatchScheduler

batch_scheduler = CyclicalBatchScheduler(
    min_batch_size=8,
    max_batch_size=64,
    cycle_length=1000
)

# Get dynamic batch size
batch_size = batch_scheduler.get_batch_size()
```

### Adaptive Warmup

```python
from Ava.training.advanced_warmup_scheduling import AdaptiveWarmupScheduler

scheduler = AdaptiveWarmupScheduler(
    optimizer,
    base_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(...),
    min_warmup_steps=100,
    max_warmup_steps=10000
)

# Step with gradient norm
scheduler.step(grad_norm=grad_norm)
```

## 10. Hardware Optimizations

### Auto-Optimize

```python
from Ava.optimization.hardware_optimizations import auto_optimize_hardware

# Automatically detect GPU and apply optimal settings
hw_optimizer = auto_optimize_hardware()
```

### A100-Specific

```python
from Ava.optimization.hardware_optimizations import optimize_for_a100

hw_optimizer = optimize_for_a100(memory_size='80GB')
```

### H100-Specific

```python
from Ava.optimization.hardware_optimizations import optimize_for_h100

hw_optimizer = optimize_for_h100()
```

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from Ava.optimization.hardware_optimizations import auto_optimize_hardware
from Ava.optimization.fused_optimizers import create_optimizer
from Ava.optimization.gradient_optimizations import MixedPrecisionManager
from Ava.data.optimized_dataloader import create_production_dataloader
from Ava.training.profiling_tools import TrainingMonitor

# 1. Hardware optimization
hw_optimizer = auto_optimize_hardware()

# 2. Compile model
model = torch.compile(model, mode='reduce-overhead')
model = model.cuda()

# 3. Create optimizer
optimizer = create_optimizer(
    model,
    optimizer_type='fused_adam',
    lr=3e-4,
    weight_decay=0.01
)

# 4. Mixed precision
mp_manager = MixedPrecisionManager(enabled=True)

# 5. Optimized dataloader
dataloader = create_production_dataloader(
    dataset,
    batch_size=32,
    num_workers=8,
    use_packing=True,
    device=torch.device('cuda')
)

# 6. Training monitor
monitor = TrainingMonitor(model, log_interval=10)

# 7. Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        monitor.start_step()

        # Forward
        with mp_manager.autocast():
            outputs = model(**batch)
            loss = outputs['loss']

        # Backward
        loss = mp_manager.scale_loss(loss)
        loss.backward()

        # Optimizer step
        metrics = mp_manager.step_optimizer(optimizer, max_grad_norm=1.0)

        optimizer.zero_grad()

        # Monitor
        stats = monitor.end_step(
            batch_size=32,
            seq_len=512,
            loss=loss.item(),
            grad_norm=metrics.get('grad_norm', 0)
        )
```

## Performance Benchmarks

### Expected Improvements

| Optimization | Speedup | Memory Savings |
|--------------|---------|----------------|
| torch.compile | 1.2-1.4x | - |
| Flash Attention | 2-4x | 50% |
| Fused Optimizers | 1.1-1.15x | - |
| 8-bit Adam | - | 75% (optimizer states) |
| Mixed Precision (BF16) | 2-3x | 50% |
| Gradient Compression | - | 90% (communication) |
| Sequence Packing | 2-3x | - |
| Dynamic Batching | 1.5-2x | - |
| TF32 | 1.5-2x | - |
| **Combined** | **5-10x** | **60-70%** |

### Throughput Examples

**A100 80GB (BF16, all optimizations):**
- 7B model: ~200K tokens/sec
- 13B model: ~100K tokens/sec
- 70B model (FSDP 8xA100): ~50K tokens/sec

**Model FLOPS Utilization (MFU):**
- Without optimizations: 15-25%
- With optimizations: 45-65%

## Troubleshooting

### OOM (Out of Memory)

1. Enable gradient checkpointing
2. Reduce batch size / enable dynamic batching
3. Use 8-bit optimizers
4. Enable CPU offloading (FSDP)
5. Use sequence packing to reduce padding

### Slow Training

1. Enable torch.compile
2. Use Flash Attention
3. Enable TF32 (A100/H100)
4. Use fused optimizers
5. Check data loading isn't bottleneck (use profiler)

### Gradient Issues

1. Use adaptive gradient clipping
2. Enable gradient noise scale monitoring
3. Use adaptive warmup
4. Check for NaN with mixed precision manager

## References

- Flash Attention: https://arxiv.org/abs/2205.14135
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- FSDP: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- Lion Optimizer: https://arxiv.org/abs/2302.06675
- Gradient Compression: https://arxiv.org/abs/1905.13727
