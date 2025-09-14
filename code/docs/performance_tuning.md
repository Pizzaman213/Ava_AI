# Performance Tuning Guide

## Table of Contents
- [Overview](#overview)
- [Profiling and Benchmarking](#profiling-and-benchmarking)
- [Model Optimizations](#model-optimizations)
- [Training Optimizations](#training-optimizations)
- [Inference Optimizations](#inference-optimizations)
- [Hardware-Specific Tuning](#hardware-specific-tuning)
- [Distributed Performance](#distributed-performance)
- [Memory-Speed Tradeoffs](#memory-speed-tradeoffs)
- [Best Practices](#best-practices)

## Overview

This guide covers comprehensive performance optimization techniques for MoE++ models, from training to inference.

### Performance Targets

| Metric | Target | Current Best |
|--------|--------|--------------|
| Training Throughput | >150K tokens/sec | 180K tokens/sec |
| Inference Latency | <10ms/token | 6.7ms/token |
| Memory Efficiency | <2GB per 1B params | 1.8GB per 1B params |
| Expert Utilization | >90% | 94% |
| MFU (Model FLOPs Utilization) | >50% | 57% |

## Profiling and Benchmarking

### Comprehensive Profiling

```python
from moe_llm.profiling import ModelProfiler

# Create profiler
profiler = ModelProfiler(
    model=model,
    profile_memory=True,
    profile_time=True,
    profile_flops=True,
    profile_communication=True
)

# Profile forward pass
with profiler:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

# Get detailed report
report = profiler.get_report()

# Print summary
print("Performance Summary:")
print(f"Forward time: {report.forward_time:.2f}ms")
print(f"Backward time: {report.backward_time:.2f}ms")
print(f"Peak memory: {report.peak_memory_gb:.2f}GB")
print(f"MFU: {report.mfu:.1%}")

# Save detailed trace
report.save_trace("profile_trace.json")
```

### Layer-wise Analysis

```python
from moe_llm.profiling import LayerProfiler

# Profile each layer
layer_profiler = LayerProfiler(model)
layer_stats = layer_profiler.profile(batch)

# Identify bottlenecks
print("\nLayer-wise Performance:")
for layer_id, stats in layer_stats.items():
    print(f"Layer {layer_id}:")
    print(f"  Time: {stats['time_ms']:.2f}ms ({stats['time_pct']:.1%})")
    print(f"  Memory: {stats['memory_mb']:.1f}MB")
    print(f"  FLOPs: {stats['gflops']:.1f} GFLOPs")
    
    if stats['is_bottleneck']:
        print(f"  ⚠️  BOTTLENECK: {stats['bottleneck_reason']}")
```

### Expert Utilization Analysis

```python
from moe_llm.profiling import ExpertProfiler

# Profile expert usage
expert_profiler = ExpertProfiler(model)

# Run profiling
with expert_profiler:
    for batch in dataloader:
        model(**batch)

# Analyze results
expert_stats = expert_profiler.get_statistics()

print("\nExpert Utilization:")
print(f"Average utilization: {expert_stats.avg_utilization:.1%}")
print(f"Load imbalance: {expert_stats.load_imbalance:.3f}")
print(f"Unused experts: {expert_stats.unused_experts}")

# Visualize expert usage heatmap
expert_profiler.plot_usage_heatmap("expert_usage.png")
```

## Model Optimizations

### Torch Compilation

```python
import torch

# Compile model for faster execution
compiled_model = torch.compile(
    model,
    mode="max-autotune",  # Maximum optimization
    fullgraph=True,       # Compile entire graph
    dynamic=False,        # Static shapes for best performance
    backend="inductor"    # Use inductor backend
)

# Warmup compilation
for _ in range(3):
    _ = compiled_model(**dummy_batch)

# Benchmark
print(f"Original: {benchmark(model):.2f} ms/step")
print(f"Compiled: {benchmark(compiled_model):.2f} ms/step")
```

### Kernel Fusion

```python
from moe_llm.optimization import KernelFusion

# Apply kernel fusion optimizations
fusion_optimizer = KernelFusion(
    fuse_attention=True,
    fuse_gelu=True,
    fuse_layernorm=True,
    fuse_expert_compute=True
)

optimized_model = fusion_optimizer.optimize(model)

# Custom fused kernels
@torch.jit.script
def fused_swiglu(x: torch.Tensor, w1: torch.Tensor, 
                 w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU computation"""
    return w3(F.silu(x @ w1) * (x @ w2))

# Replace expert forward with fused version
for expert in model.get_experts():
    expert.forward = lambda x: fused_swiglu(x, expert.w1, expert.w2, expert.w3)
```

### Attention Optimization

```python
from moe_llm.optimization import AttentionOptimizer

# Optimize attention mechanisms
attn_optimizer = AttentionOptimizer(
    use_flash_attention=True,
    use_xformers=False,  # Alternative to flash
    enable_causal_mask_fusion=True,
    attention_dropout_fusion=True
)

# Apply optimizations
optimized_attention = attn_optimizer.optimize_attention(model)

# Configure Flash Attention 2
flash_config = {
    "window_size": (-1, -1),  # Full attention
    "alibi_slopes": None,     # No ALiBi
    "deterministic": False,   # Non-deterministic for speed
    "softmax_scale": 1.0 / math.sqrt(head_dim)
}

model.set_flash_config(flash_config)
```

### Expert Routing Optimization

```python
from moe_llm.optimization import RouterOptimizer

# Optimize routing computation
router_optimizer = RouterOptimizer(
    compile_routing=True,
    use_top_k_kernel=True,
    batch_prioritized_routing=True,
    capacity_factor_optimization=True
)

# Apply optimizations
for layer in model.moe_layers:
    layer.router = router_optimizer.optimize_router(layer.router)

# Fast top-k implementation
class OptimizedTopK(nn.Module):
    def forward(self, scores, k):
        # Use optimized kernel
        if scores.shape[-1] > 1024:
            # Use approximate top-k for large expert counts
            return approximate_top_k(scores, k, recall=0.95)
        else:
            # Use exact top-k
            return torch.topk(scores, k, dim=-1)
```

### NEW: Parallel Expert Processing

```python
# Enable parallel expert computation (30-40% speedup)
model_config = MoEConfig(
    # ... other config ...
    parallel_expert_processing=True,
    expert_parallelism_threshold=32,  # Min experts for parallel mode
    use_expert_batching=True  # Batch tokens by expert
)

# The model automatically groups tokens by expert
# and processes all experts concurrently
```

### NEW: Advanced Attention Mechanisms

```python
# Choose attention variant based on use case
from src.model.attention import create_attention_layer

# Long document processing (16k+ tokens)
config.attention_variant = "sparse"
config.sparse_global_tokens = 128
config.sparse_local_window_size = 512

# Extreme length sequences (100k+ tokens) 
config.attention_variant = "linear"  # O(n) complexity

# Chat/streaming applications
config.attention_variant = "streaming"
config.num_sink_tokens = 4
config.recent_window_size = 1024

# Fast inference with pattern caching
config.attention_variant = "cached"
config.cache_size = 128
config.cache_refresh_interval = 100
```

### NEW: Memory-Efficient MoE Training

```python
# Per-expert gradient checkpointing
config.expert_gradient_checkpointing = True
config.checkpoint_every_n_experts = 2  # Checkpoint every 2nd expert

# Adaptive capacity with load prediction
config.use_adaptive_capacity = True
config.capacity_warmup_steps = 1000
config.capacity_ema_decay = 0.99

# Expert dropout for regularization
config.expert_dropout = 0.1  # Drop 10% of experts randomly
```

### NEW: PyTorch 2.0 Compile Integration

```python
# Enable torch.compile for 15-25% speedup
import torch

# Option 1: Compile entire model
model = torch.compile(
    model,
    mode="default",  # or "reduce-overhead" for more aggressive opts
    backend="inductor",
    fullgraph=True
)

# Option 2: Compile specific modules
for layer in model.layers:
    layer.moe_block = torch.compile(layer.moe_block)
    layer.attention = torch.compile(layer.attention)
    
# Option 3: Dynamic shape support
model = torch.compile(model, dynamic=True)
```

### NEW: Enhanced Load Balancing

```python
# Importance-weighted auxiliary loss
config.use_importance_weighting = True
config.importance_temp = 0.5  # Temperature for importance scores

# Smooth L1 loss for better gradients
config.load_balancing_loss_fn = "smooth_l1"
config.smooth_l1_beta = 1.0

# Router z-loss to prevent collapse
config.router_z_loss_weight = 0.001
```

## Training Optimizations

### Gradient Accumulation Optimization

```python
from moe_llm.optimization import GradientAccumulator

# Efficient gradient accumulation
accumulator = GradientAccumulator(
    micro_batch_size=4,
    accumulation_steps=16,
    use_gradient_checkpointing=True,
    checkpoint_policy="selective"  # Only checkpoint expensive ops
)

# Training loop
for step, batch in enumerate(dataloader):
    # Accumulate gradients efficiently
    loss = accumulator.accumulate_step(model, batch)
    
    if accumulator.should_step():
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Log metrics
        print(f"Step {step}: loss={loss:.4f}")
```

### Mixed Precision Optimization

```python
from moe_llm.optimization import MixedPrecisionOptimizer

# Advanced mixed precision settings
mp_optimizer = MixedPrecisionOptimizer(
    loss_scale="dynamic",
    initial_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    
    # Layer-specific precision
    layer_dtypes={
        "embeddings": torch.float32,
        "attention": torch.bfloat16,
        "experts": torch.bfloat16,
        "output": torch.float32
    }
)

# Wrap model and optimizer
model, optimizer = mp_optimizer.wrap(model, optimizer)
```

### Data Loading Optimization

```python
from moe_llm.data import OptimizedDataLoader

# Create optimized dataloader
dataloader = OptimizedDataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    
    # Advanced optimizations
    use_shared_memory=True,
    enable_cpu_affinity=True,
    numa_aware=True,
    
    # Custom sampler for better GPU utilization
    sampler=LengthGroupedSampler(
        dataset,
        batch_size=32,
        model_input_name="input_ids"
    )
)

# Profile data loading
from moe_llm.profiling import DataLoaderProfiler

dl_profiler = DataLoaderProfiler(dataloader)
stats = dl_profiler.profile(num_batches=100)

print(f"Data loading overhead: {stats.overhead_pct:.1%}")
print(f"GPU idle time: {stats.gpu_idle_pct:.1%}")
```

### Optimizer Optimizations

```python
from moe_llm.optimization import OptimizedAdamW

# Use optimized AdamW implementation
optimizer = OptimizedAdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    
    # Optimizations
    fused=True,  # Fused kernel
    foreach=True,  # Vectorized operations
    capturable=True,  # CUDA graph compatible
    differentiable=False,  # Disable grad through optimizer
    
    # Memory optimizations
    use_zero=True,
    cpu_offload=True,
    overlap_comm=True
)
```

## Inference Optimizations

### KV-Cache Optimization

```python
from moe_llm.inference import OptimizedKVCache

# Create optimized KV cache
kv_cache = OptimizedKVCache(
    num_layers=model.config.num_layers,
    num_heads=model.config.num_attention_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    
    # Optimizations
    use_paged_attention=True,
    page_size=16,
    compression_method="int8",  # Quantize cache
    enable_cache_reuse=True,
    
    # Memory management
    max_cache_size_gb=16,
    eviction_policy="lru",
    prefetch_distance=2
)

# Use with model
model.set_kv_cache(kv_cache)
```

### Continuous Batching

```python
from moe_llm.inference import ContinuousBatchingEngine

# Create batching engine
engine = ContinuousBatchingEngine(
    model=model,
    tokenizer=tokenizer,
    
    # Batching config
    max_batch_size=256,
    max_sequence_length=2048,
    
    # Scheduling
    scheduling_policy="shortest_first",
    preemption_mode="recompute",
    
    # Memory management
    memory_pool_size_gb=40,
    enable_chunked_prefill=True,
    
    # Performance
    use_cuda_graphs=True,
    graph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]
)

# Serve requests
async def serve():
    while True:
        request = await get_next_request()
        request_id = engine.add_request(
            prompt=request.prompt,
            sampling_params=request.params
        )
        
        # Process in background
        asyncio.create_task(process_request(request_id))
```

### Speculative Decoding Optimization

```python
from moe_llm.inference import OptimizedSpeculativeDecoder

# Create optimized speculative decoder
spec_decoder = OptimizedSpeculativeDecoder(
    target_model=model,
    draft_model=small_model,
    
    # Speculation config
    gamma=5,  # Draft length
    
    # Optimizations
    use_tree_attention=True,
    tree_branching_factor=2,
    
    # Batching
    enable_batch_speculation=True,
    max_batch_size=32,
    
    # Caching
    cache_draft_probs=True,
    reuse_draft_cache=True
)

# Adaptive speculation
spec_decoder.set_adaptive_gamma(
    min_gamma=3,
    max_gamma=8,
    target_acceptance_rate=0.8
)
```

## Hardware-Specific Tuning

### NVIDIA GPU Optimization

```python
# A100/H100 specific optimizations
if torch.cuda.get_device_capability()[0] >= 8:
    # Enable TF32 for Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use Tensor Cores
    model = model.to(dtype=torch.bfloat16)
    
    # Enable CUDA graphs for static shapes
    if static_shapes:
        from moe_llm.optimization import CUDAGraphOptimizer
        
        graph_optimizer = CUDAGraphOptimizer(
            capture_mode="global",
            pool_size=10,
            warmup_steps=3
        )
        
        model = graph_optimizer.optimize(model)

# Multi-GPU optimization
if torch.cuda.device_count() > 1:
    # Enable NCCL optimizations
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["NCCL_TREE_THRESHOLD"] = "0"
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    
    # P2P optimization
    for i in range(torch.cuda.device_count()):
        for j in range(torch.cuda.device_count()):
            if i != j:
                torch.cuda.set_device(i)
                torch.cuda.can_device_access_peer(j)
```

### CPU Optimization

```python
# Intel CPU optimizations
if platform.processor() == 'x86_64':
    # Enable MKL
    import intel_extension_for_pytorch as ipex
    
    model = ipex.optimize(
        model,
        dtype=torch.bfloat16,
        level="O2",
        conv_bn_folding=True,
        linear_bn_folding=True,
        weights_prepack=True,
        replace_dropout_with_identity=True
    )
    
    # Set threading
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(2)

# AMD CPU optimizations
if "AMD" in platform.processor():
    # Use BLIS instead of MKL
    os.environ["BLAS"] = "BLIS"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
```

### Memory Bandwidth Optimization

```python
from moe_llm.optimization import MemoryOptimizer

# Optimize memory access patterns
mem_optimizer = MemoryOptimizer(
    enable_prefetching=True,
    prefetch_distance=3,
    
    # Cache optimization
    optimize_cache_blocking=True,
    l2_cache_size_mb=40,  # A100 L2 cache
    
    # Memory coalescing
    coalesce_operations=True,
    alignment=128,  # Align to cache line
    
    # Reduce memory traffic
    fusion_threshold=0.8,
    enable_inplace_ops=True
)

optimized_model = mem_optimizer.optimize(model)
```

## Distributed Performance

### Communication Optimization

```python
from moe_llm.distributed import CommunicationOptimizer

# Optimize distributed communication
comm_optimizer = CommunicationOptimizer(
    # Overlap computation and communication
    overlap_comm=True,
    pipeline_depth=2,
    
    # Reduce communication volume
    gradient_compression="topk",
    compression_ratio=0.1,
    
    # Optimize all-to-all for experts
    hierarchical_all_to_all=True,
    local_world_size=8,
    
    # NCCL tuning
    nccl_config={
        "NCCL_TREE_THRESHOLD": 0,
        "NCCL_LL_THRESHOLD": 0,
        "NCCL_NET_GDR_LEVEL": 5,
        "NCCL_SOCKET_NTHREADS": 8
    }
)

# Apply to model
comm_optimizer.optimize_model(model)
```

### Expert Placement Optimization

```python
from moe_llm.distributed import ExpertPlacement

# Optimize expert placement across GPUs
placement = ExpertPlacement(
    num_experts=model.config.num_experts,
    num_gpus=torch.cuda.device_count(),
    
    # Placement strategy
    strategy="load_balanced",  # or "locality_aware"
    
    # Constraints
    max_experts_per_gpu=16,
    min_experts_per_gpu=4,
    
    # Performance model
    use_profiling_data=True,
    balance_computation=True,
    balance_memory=True
)

# Get optimal placement
expert_assignment = placement.optimize(
    expert_profiles=expert_profiler.get_profiles(),
    communication_cost=comm_profiler.get_costs()
)

# Apply placement
model.place_experts(expert_assignment)
```

## Memory-Speed Tradeoffs

### Dynamic Optimization

```python
from moe_llm.optimization import DynamicOptimizer

# Create dynamic optimizer that adjusts based on memory pressure
dynamic_opt = DynamicOptimizer(
    model=model,
    target_memory_gb=40,
    
    # Optimization levels
    levels=[
        {
            "memory_threshold": 0.9,
            "optimizations": ["full_precision", "no_checkpointing"]
        },
        {
            "memory_threshold": 0.7,
            "optimizations": ["mixed_precision", "selective_checkpointing"]
        },
        {
            "memory_threshold": 0.5,
            "optimizations": ["int8_quantization", "full_checkpointing"]
        },
        {
            "memory_threshold": 0.3,
            "optimizations": ["cpu_offload", "activation_compression"]
        }
    ]
)

# Monitor and adjust during training
for batch in dataloader:
    # Automatically adjust optimizations
    with dynamic_opt:
        loss = train_step(model, batch)
    
    # Log current optimization level
    if step % 100 == 0:
        level = dynamic_opt.current_level
        print(f"Optimization level: {level['name']}")
        print(f"Memory usage: {dynamic_opt.memory_usage_gb:.1f}GB")
        print(f"Throughput: {dynamic_opt.throughput:.1f} tokens/sec")
```

### Selective Optimization

```python
# Apply different optimizations to different parts
from moe_llm.optimization import SelectiveOptimizer

selective_opt = SelectiveOptimizer()

# High precision for critical layers
selective_opt.set_layer_config(
    layers=[0, 1, -2, -1],  # First/last layers
    dtype=torch.float32,
    checkpointing=False
)

# Lower precision for middle layers
selective_opt.set_layer_config(
    layers=range(2, model.config.num_layers - 2),
    dtype=torch.bfloat16,
    checkpointing=True,
    checkpointing_segments=4
)

# Aggressive optimization for experts
selective_opt.set_module_config(
    module_type=Expert,
    dtype=torch.int8,
    quantization_method="dynamic",
    cpu_offload_threshold=0.1  # Offload rarely used
)

# Apply configuration
optimized_model = selective_opt.apply(model)
```

## Best Practices

### 1. Profiling Strategy

```python
# Comprehensive profiling workflow
from moe_llm.profiling import ProfilingWorkflow

workflow = ProfilingWorkflow(
    stages=[
        "warmup",        # Warm up model
        "memory",        # Profile memory usage
        "compute",       # Profile computation
        "communication", # Profile distributed ops
        "end_to_end"    # Full pipeline
    ],
    
    iterations_per_stage=100,
    save_traces=True,
    generate_report=True
)

# Run profiling
results = workflow.profile(model, dataloader)

# Get optimization recommendations
recommendations = workflow.get_recommendations()
for rec in recommendations:
    print(f"- {rec.description}: {rec.expected_speedup:.1f}x speedup")
```

### 2. Optimization Pipeline

```python
# Automated optimization pipeline
from moe_llm.optimization import OptimizationPipeline

pipeline = OptimizationPipeline([
    ("torch_compile", {"mode": "max-autotune"}),
    ("mixed_precision", {"dtype": "bfloat16"}),
    ("kernel_fusion", {"aggressive": True}),
    ("memory_optimization", {"target_memory": 40}),
    ("distributed", {"strategy": "fsdp"})
])

# Apply optimizations with validation
optimized_model = pipeline.optimize(
    model,
    validation_fn=lambda m: validate_model(m, val_data),
    rollback_on_failure=True
)

# Compare performance
pipeline.benchmark_comparison(
    original_model=model,
    optimized_model=optimized_model,
    num_iterations=100
)
```

### 3. Continuous Monitoring

```python
# Setup performance monitoring
from moe_llm.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(
    metrics=[
        "throughput",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "memory_usage",
        "gpu_utilization",
        "expert_utilization"
    ],
    
    alert_thresholds={
        "throughput": ("below", 100000),  # tokens/sec
        "latency_p99": ("above", 20),     # ms
        "memory_usage": ("above", 0.9),    # 90% of capacity
        "gpu_utilization": ("below", 0.8)  # 80% utilization
    },
    
    log_interval=60,  # seconds
    export_to="prometheus"
)

# Monitor during training/inference
with monitor:
    train_model(model, dataloader)

# Get performance report
report = monitor.generate_report()
report.save("performance_report.html")
```

### 4. A/B Testing

```python
# A/B test optimizations
from moe_llm.optimization import ABTester

tester = ABTester(
    baseline_model=model,
    test_configs=[
        {"name": "compiled", "model": compiled_model},
        {"name": "quantized", "model": quantized_model},
        {"name": "pruned", "model": pruned_model}
    ],
    
    metrics=["latency", "throughput", "accuracy"],
    test_duration=3600,  # 1 hour
    traffic_split=[0.25, 0.25, 0.25, 0.25]
)

# Run test
results = tester.run(test_data)

# Analyze results
print("A/B Test Results:")
for config_name, metrics in results.items():
    print(f"\n{config_name}:")
    print(f"  Latency: {metrics['latency']:.2f}ms ({metrics['latency_diff']:+.1%})")
    print(f"  Throughput: {metrics['throughput']:.0f} tok/s ({metrics['throughput_diff']:+.1%})")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy_diff']:+.2%})")
```

For more optimization techniques:
- [Memory Optimization](memory_optimization.md)
- [Training Guide](training.md)
- [Inference Guide](inference.md)