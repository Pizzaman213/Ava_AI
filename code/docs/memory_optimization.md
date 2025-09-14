# Memory Optimization Guide

## Recent Updates (August 2024)

### 🚀 Data Streaming Now Default
- **Problem Solved**: Loading 70GB+ datasets into RAM causing crashes
- **Solution**: All configs now use streaming by default
- **Result**: Memory usage reduced from 70GB to 2-4GB

### 📊 Enhanced Training Progress
- Shows current batch / total batches
- Displays percentage completion and ETA
- Memory-efficient progress tracking

### 🔧 Quick Fixes for Memory Issues
```bash
# If still having memory issues:
python scripts/train.py --config configs/mps/ultra_tiny.yaml \
    --batch-size 4 \
    --gradient-checkpointing \
    --max-length 256
```

## Table of Contents
- [Overview](#overview)
- [ZeRO Optimization](#zero-optimization)
- [Activation Checkpointing](#activation-checkpointing)
- [CPU/NVMe Offloading](#cpunvme-offloading)
- [Mixed Precision Training](#mixed-precision-training)
- [Memory-Efficient Data Loading](#memory-efficient-data-loading)
- [Model Sharding](#model-sharding)
- [Memory Profiling](#memory-profiling)
- [Best Practices](#best-practices)

## Overview

Training large MoE models requires careful memory management. This guide covers techniques to reduce memory usage by up to 10x while maintaining training efficiency.

### Memory Breakdown

For a typical MoE model:

| Component | Memory Usage | Optimization Potential |
|-----------|--------------|----------------------|
| Model Parameters | 2-4 bytes/param | ZeRO-3, Offloading |
| Optimizer States | 8-12 bytes/param | ZeRO-2/3, CPU offload |
| Activations | O(batch × seq × layers) | Checkpointing, Compression |
| Gradients | 2-4 bytes/param | ZeRO-2/3, Mixed precision |
| Expert Parameters | 2-4 bytes/param × experts | Expert offloading |

## ZeRO Optimization

### ZeRO Stage Configuration

```python
# ZeRO Stage 1: Optimizer State Partitioning
zero_stage_1_config = {
    "stage": 1,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
}

# ZeRO Stage 2: Optimizer State + Gradient Partitioning
zero_stage_2_config = {
    "stage": 2,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
    },
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8,
    "overlap_comm": True
}

# ZeRO Stage 3: Full Partitioning (Params + Grads + Optimizer)
zero_stage_3_config = {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True,
        "buffer_count": 4,
        "fast_init": True
    },
    "offload_param": {
        "device": "cpu",
        "pin_memory": True,
        "buffer_count": 5,
        "buffer_size": 1e8,
        "max_in_cpu": 1e9
    },
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": True
}
```

### ZeRO-Infinity (NVMe Offloading)

```python
# ZeRO-Infinity configuration
zero_infinity_config = {
    "stage": 3,
    "offload_optimizer": {
        "device": "nvme",
        "nvme_path": "/nvme_raid/offload",
        "pin_memory": True,
        "buffer_count": 4,
        "fast_init": True
    },
    "offload_param": {
        "device": "nvme",
        "nvme_path": "/nvme_raid/offload",
        "pin_memory": True,
        "buffer_count": 5,
        "buffer_size": 1e8,
        "max_in_cpu": 1e9
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },
    "infinity_activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
}
```

### Expert-Aware ZeRO

```python
from moe_llm.optimization import ExpertAwareZeRO

# Configure expert-specific partitioning
expert_zero = ExpertAwareZeRO(
    num_experts=64,
    experts_per_gpu=8,
    expert_parallel_size=8,
    hierarchical_partitioning=True
)

# Apply to model
model = expert_zero.configure_model(model)

# Expert placement strategy
placement_config = {
    "strategy": "balanced",  # or "frequency_based", "locality_aware"
    "cpu_offload_threshold": 0.1,  # Offload rarely used experts
    "dynamic_placement": True
}
```

## Activation Checkpointing

### Basic Checkpointing

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Configure checkpointing
checkpoint_config = {
    "use_checkpoint": True,
    "checkpoint_num_layers": 4,  # Checkpoint every 4 layers
    "checkpoint_attention": True,
    "checkpoint_mlp": True,
    "checkpoint_moe": True
}

# Apply configuration
from moe_llm.optimization import configure_checkpointing
configure_checkpointing(model, checkpoint_config)
```

### Selective Checkpointing

```python
# Custom checkpoint policy
def selective_checkpoint_policy(module, layer_idx):
    # Checkpoint memory-intensive layers
    if isinstance(module, MoELayer):
        return layer_idx % 2 == 0  # Every other MoE layer
    elif isinstance(module, MultiQueryAttention):
        return layer_idx > 12  # Later attention layers
    return False

model.set_checkpoint_policy(selective_checkpoint_policy)
```

### Activation Compression

```python
from moe_llm.optimization import ActivationCompression

# Configure compression
compression_config = {
    "algorithm": "quantization",  # or "sparsity", "mixed"
    "bits": 8,  # 8-bit quantization
    "threshold": 0.01,  # Sparsity threshold
    "layers_to_compress": list(range(12, 24))  # Compress later layers
}

compression = ActivationCompression(compression_config)
model = compression.wrap_model(model)

# Monitor compression ratio
compression_stats = compression.get_stats()
print(f"Compression ratio: {compression_stats['ratio']:.2f}x")
print(f"Memory saved: {compression_stats['memory_saved_gb']:.1f} GB")
```

### NEW: Memory-Efficient MoE Layer

```python
# Enable per-expert gradient checkpointing (40-60% memory reduction)
config = MoEConfig(
    expert_gradient_checkpointing=True,
    checkpoint_every_n_experts=2,  # Checkpoint every 2nd expert
    parallel_expert_processing=True  # Also improves memory locality
)

# The model automatically applies checkpointing during forward pass
# This trades ~5% compute time for significant memory savings
```

### NEW: Memory-Efficient Attention Variants

```python
# 1. Linear Attention - O(n) memory instead of O(n²)
config.attention_variant = "linear"
# Handles 100k+ sequences with constant memory usage

# 2. Sliding Window - Fixed memory regardless of sequence length
config.attention_variant = "sliding_window"
config.sliding_window_size = 512
# Memory usage: O(n × w) where w is window size

# 3. Streaming Attention - Bounded memory for infinite sequences
config.attention_variant = "streaming"
config.num_sink_tokens = 4
config.recent_window_size = 1024
# Fixed memory usage regardless of total sequence length

# 4. Sparse Attention - Reduces memory for long sequences
config.attention_variant = "sparse"
config.sparse_local_window_size = 256
# Memory usage: O(n × log(n)) instead of O(n²)
```

### NEW: Adaptive Capacity for Memory Efficiency

```python
# Dynamic expert capacity prevents memory waste
config.use_adaptive_capacity = True
config.capacity_warmup_steps = 1000

# The model predicts required capacity per expert
# and allocates memory dynamically, reducing waste by 20-40%
```

## CPU/NVMe Offloading

### Optimizer Offloading

```python
# CPU offloading configuration
cpu_adam_config = {
    "optimizer": "CPUAdam",
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "cpu_offload": True,
    "pin_memory": True
}

# Create CPU optimizer
from moe_llm.optimization import CPUAdam
optimizer = CPUAdam(model.parameters(), **cpu_adam_config)
```

### Parameter Offloading

```python
from moe_llm.optimization import ParameterOffloadEngine

# Configure offloading
offload_engine = ParameterOffloadEngine(
    device="cuda",
    offload_device="cpu",
    buffer_size=1e9,  # 1GB buffer
    pin_memory=True,
    num_buffers=4,
    prefetch_ahead=2
)

# Wrap model
offloaded_model = offload_engine.wrap_model(model)

# Manual prefetching for specific layers
for layer_idx in range(num_layers):
    offload_engine.prefetch_layer(layer_idx + 2)
    output = offloaded_model.layers[layer_idx](input)
    offload_engine.offload_layer(layer_idx - 2)
```

### NVMe Optimization

```python
# Setup NVMe array
nvme_config = {
    "nvme_path": "/nvme_raid",
    "num_threads": 4,
    "block_size": 1048576,  # 1MB blocks
    "queue_depth": 16,
    "direct_io": True
}

# Create NVMe-backed parameter storage
from moe_llm.optimization import NVMeParameterStore

param_store = NVMeParameterStore(
    model_config=model.config,
    nvme_config=nvme_config,
    compression="lz4"  # Fast compression
)

# Offload experts to NVMe
for expert_id in rarely_used_experts:
    param_store.offload_expert(model, expert_id)
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

# Training loop with AMP
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### BFloat16 Training

```python
# Configure BF16
bf16_config = {
    "enabled": True,
    "loss_scale": 1.0,  # BF16 doesn't need loss scaling
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
}

# Convert model to BF16
model = model.to(dtype=torch.bfloat16)

# Mixed precision optimizer
from moe_llm.optimization import MixedPrecisionOptimizer

mp_optimizer = MixedPrecisionOptimizer(
    optimizer,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    dynamic_loss_args={}
)
```

### Precision-Aware Expert Routing

```python
# Use lower precision for routing
class MixedPrecisionRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            dtype=torch.float16  # FP16 for routing
        )
    
    def forward(self, x):
        # Cast to FP16 for routing
        x_fp16 = x.to(dtype=torch.float16)
        scores = self.router(x_fp16)
        
        # Cast back for softmax stability
        scores = scores.to(dtype=torch.float32)
        return F.softmax(scores, dim=-1)
```

## Memory-Efficient Data Loading

### Memory-Mapped Datasets

```python
from moe_llm.data import MemoryMappedDataset, DatasetBuilder

# Build memory-mapped dataset
builder = DatasetBuilder(
    tokenizer=tokenizer,
    max_length=2048,
    dtype=np.uint16  # Save memory with uint16
)

builder.build_from_jsonl(
    input_files=["data/train_*.jsonl"],
    output_prefix="data/train_mmap",
    num_workers=32
)

# Load dataset
dataset = MemoryMappedDataset(
    data_file="data/train_mmap.bin",
    index_file="data/train_mmap.idx",
    sequence_length=2048
)
```

### Streaming DataLoader

```python
from moe_llm.data import StreamingDataLoader

# Configure streaming
dataloader = StreamingDataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    # Memory optimizations
    copy_before_send=False,  # Avoid copies
    share_memory=True,  # Share between workers
    buffer_size=1000  # Small buffer
)
```

### Dynamic Batching

```python
from moe_llm.data import DynamicBatchSampler

# Create dynamic batch sampler
sampler = DynamicBatchSampler(
    dataset,
    max_tokens=50000,  # Max tokens per batch
    max_sequences=64,  # Max sequences per batch
    length_bucket_width=100,  # Bucket by length
    shuffle_buffer_size=10000
)

dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=dynamic_collate_fn
)
```

## Model Sharding

### Tensor Parallelism

```python
from moe_llm.distributed import TensorParallelism

# Configure tensor parallelism
tp_config = {
    "tensor_parallel_size": 4,
    "sequence_parallel": True,
    "async_tensor_parallel": True
}

# Apply to model
tp = TensorParallelism(tp_config)
model = tp.parallelize_model(model)

# Memory-efficient attention with TP
class TPAttention(nn.Module):
    def __init__(self, config, tp_size):
        self.tp_size = tp_size
        self.local_heads = config.num_heads // tp_size
        self.head_dim = config.hidden_size // config.num_heads
        
        # Shard QKV projections
        self.q_proj = ColumnParallelLinear(...)
        self.k_proj = ColumnParallelLinear(...)
        self.v_proj = ColumnParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
```

### Pipeline Parallelism

```python
from moe_llm.distributed import PipelineParallelism

# Configure pipeline parallelism
pp_config = {
    "pipeline_parallel_size": 4,
    "micro_batch_size": 4,
    "pipeline_schedule": "1f1b",  # 1-forward-1-backward
    "activation_checkpoint_interval": 1
}

# Create pipeline
pp = PipelineParallelism(pp_config)
model = pp.partition_model(model, balance_method="parameter_count")
```

### Expert Parallelism

```python
from moe_llm.distributed import ExpertParallelism

# Configure expert parallelism
ep_config = {
    "expert_parallel_size": 8,
    "expert_slicing": "vertical",  # or "horizontal"
    "load_balance": True,
    "capacity_factor": 1.25
}

# Apply expert parallelism
ep = ExpertParallelism(ep_config)
model = ep.parallelize_experts(model)
```

## Memory Profiling

### PyTorch Profiler

```python
import torch.profiler

# Profile memory usage
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    # Training step
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Analyze results
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))

# Export to Chrome tracing
prof.export_chrome_trace("memory_trace.json")
```

### Custom Memory Monitor

```python
from moe_llm.utils import MemoryMonitor

# Initialize monitor
monitor = MemoryMonitor(
    log_interval=100,
    track_peak=True,
    breakdown_by_layer=True
)

# Monitor training
with monitor:
    for step, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            stats = monitor.get_stats()
            print(f"Step {step}:")
            print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"  Reserved: {stats['reserved_gb']:.2f} GB")
            print(f"  Peak: {stats['peak_gb']:.2f} GB")
```

### Memory Leak Detection

```python
from moe_llm.utils import MemoryLeakDetector

# Setup leak detector
detector = MemoryLeakDetector(
    threshold_gb=0.1,  # Alert if memory grows > 100MB
    check_interval=1000  # Check every 1000 steps
)

# Monitor for leaks
for step in range(num_steps):
    # Training step
    train_step(model, batch)
    
    # Check for leaks
    if detector.check():
        print(f"Warning: Potential memory leak detected at step {step}")
        detector.dump_snapshot(f"leak_snapshot_{step}.pkl")
```

## Best Practices

### 1. Memory Planning

```python
# Calculate memory requirements
from moe_llm.utils import estimate_memory

memory_estimate = estimate_memory(
    model_config=config,
    batch_size=32,
    sequence_length=2048,
    optimizer="adamw",
    mixed_precision="bf16",
    zero_stage=3,
    activation_checkpointing=True
)

print(f"Estimated memory per GPU: {memory_estimate['per_gpu_gb']:.1f} GB")
print(f"Recommended GPUs: {memory_estimate['recommended_gpus']}")
```

### 2. Optimization Priority

1. **Enable mixed precision** (BF16/FP16) - 2x memory reduction
2. **Use ZeRO-3** - Divides memory by number of GPUs
3. **Enable gradient checkpointing** - Trade compute for memory
4. **Offload to CPU/NVMe** - For extremely large models
5. **Optimize batch size** - Use gradient accumulation

### 3. Memory-Efficient Training Loop

```python
def memory_efficient_train_step(model, batch, optimizer, scaler):
    # Clear gradients
    optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    # Forward pass with autocast
    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Free intermediate activations
    if hasattr(outputs, "clear"):
        outputs.clear()
    
    # Gradient accumulation
    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Clear cache periodically
        if (step + 1) % clear_cache_interval == 0:
            torch.cuda.empty_cache()
    
    return loss.item() * gradient_accumulation_steps
```

### 4. Expert-Specific Optimizations

```python
# Optimize expert memory usage
from moe_llm.optimization import ExpertMemoryOptimizer

expert_optimizer = ExpertMemoryOptimizer(
    model=model,
    config={
        "expert_dropout": 0.1,  # Randomly drop experts
        "expert_capacity_factor": 1.25,  # Limit tokens per expert
        "shared_expert_params": True,  # Share some parameters
        "expert_pruning_threshold": 0.01,  # Prune rarely used experts
        "dynamic_expert_allocation": True  # Allocate based on usage
    }
)

optimized_model = expert_optimizer.optimize()
```

### 5. Monitoring and Debugging

```python
# Comprehensive memory tracking
from moe_llm.utils import MemoryTracker

tracker = MemoryTracker(
    track_activations=True,
    track_parameters=True,
    track_gradients=True,
    track_optimizer_states=True,
    log_to_tensorboard=True
)

# Track memory throughout training
with tracker:
    train_model(model, dataloader, optimizer)

# Generate report
report = tracker.generate_report()
report.save("memory_analysis.html")
```

For more optimization techniques, see:
- [Training Guide](training.md)
- [Performance Tuning](performance_tuning.md)
- [Distributed Training](distributed_training.md)