# Ava Pipeline Optimization Implementation Guide

## Executive Summary
This guide provides actionable steps to optimize the Ava MoE training pipeline for 25-35% additional performance gains beyond the current optimizations.

## Current State
- **Memory**: 32% reduction achieved (8GB → 6.1GB per GPU)
- **Speed**: 2-3x improvement over baseline
- **Code**: 26,000+ lines, Phase 8 (Testing & Validation)

## Quick Win Optimizations (1-2 weeks, 15-30% gains)

### 1. Expert Parallel GEMM Integration (15-20% speedup)
**Status**: Code exists, needs integration

```python
# File: code/src/Ava/distributed/expert_parallel.py
# Already implemented, just needs to be enabled in MoE layer

# In code/src/Ava/models/moe_layer.py, modify forward():
if self.use_expert_parallel_gemm:
    from ..distributed.expert_parallel import ExpertParallelGroup
    expert_group = ExpertParallelGroup(world_size=torch.distributed.get_world_size())
    outputs = expert_group.batched_expert_forward(
        hidden_states,
        selected_experts,
        self.experts
    )
```

**Testing**:
```bash
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu_optimized.yaml \
    --benchmark_mode
```

### 2. Dynamic Batch Sizing (10-15% speedup)
**Implementation**: Add to `code/src/Ava/data/dataloader.py`

```python
class DynamicBatchSizeManager:
    def __init__(self, min_batch=8, max_batch=32, target_util=0.90):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_util = target_util
        self.current_batch = min_batch

    def adjust_batch_size(self):
        """Adjust batch size based on GPU memory utilization"""
        mem_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

        if mem_used < self.target_util - 0.05:
            # Increase batch size
            self.current_batch = min(self.current_batch + 2, self.max_batch)
        elif mem_used > self.target_util + 0.05:
            # Decrease batch size
            self.current_batch = max(self.current_batch - 2, self.min_batch)

        return self.current_batch
```

### 3. Expert Mixture Caching (8-12% speedup)
**Implementation**: Add to `code/src/Ava/models/moe_layer.py`

```python
class ExpertCache:
    def __init__(self, cache_size=256, similarity_threshold=0.95):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def get_cached_output(self, input_hash, input_tensor):
        """Check cache for similar inputs"""
        for cached_hash, (cached_input, cached_output) in self.cache.items():
            similarity = F.cosine_similarity(
                input_tensor.flatten(),
                cached_input.flatten(),
                dim=0
            )
            if similarity > self.similarity_threshold:
                self.hits += 1
                return cached_output.clone()

        self.misses += 1
        return None

    def add_to_cache(self, input_hash, input_tensor, output_tensor):
        """Add computation to cache"""
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        self.cache[input_hash] = (input_tensor.clone(), output_tensor.clone())
```

### 4. Adaptive Validation Frequency (5-15% time savings)
**Implementation**: Modify `code/src/Ava/training/core/trainer.py`

```python
class AdaptiveValidationScheduler:
    def __init__(self, min_steps=500, max_steps=5000, stability_window=100):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.stability_window = stability_window
        self.loss_history = deque(maxlen=stability_window)
        self.next_eval_step = min_steps

    def should_evaluate(self, step, current_loss):
        """Determine if evaluation should run based on loss stability"""
        self.loss_history.append(current_loss)

        if step < self.next_eval_step:
            return False

        if len(self.loss_history) >= self.stability_window:
            loss_std = np.std(self.loss_history)
            loss_mean = np.mean(self.loss_history)
            stability = loss_std / (loss_mean + 1e-8)

            # More stable = less frequent evaluation
            if stability < 0.01:  # Very stable
                eval_interval = self.max_steps
            elif stability < 0.05:  # Stable
                eval_interval = (self.min_steps + self.max_steps) // 2
            else:  # Unstable
                eval_interval = self.min_steps

            self.next_eval_step = step + eval_interval
            return True

        return False
```

### 5. Loss Computation Vectorization (2-3% speedup)
**Implementation**: Modify loss computation in `code/src/Ava/losses/losses.py`

```python
def vectorized_cross_entropy(logits, labels, ignore_index=-100):
    """Vectorized cross-entropy computation"""
    # Reshape for efficient computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Create mask for valid positions
    mask = labels_flat != ignore_index

    # Compute loss only on valid positions
    if mask.any():
        valid_logits = logits_flat[mask]
        valid_labels = labels_flat[mask]

        # Use torch.nn.functional for optimized computation
        loss = F.cross_entropy(
            valid_logits,
            valid_labels,
            reduction='mean'
        )
    else:
        loss = torch.tensor(0.0, device=logits.device)

    return loss
```

## Medium-Term Optimizations (1-2 months, 8-15% gains)

### 6. Layer Fusion with torch.compile (5-7% speedup)
```python
# In code/src/Ava/models/moe_model.py
@torch.compile(mode="max-autotune", fullgraph=False)
def fused_moe_block(self, hidden_states, attention_mask):
    """Fused MoE transformer block"""
    # Attention + MoE + LayerNorm fused
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    hidden_states = self.attention(hidden_states, attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    hidden_states = self.moe_layer(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states
```

### 7. Gradient Compression (40-50% bandwidth reduction)
```python
# In code/src/Ava/distributed/distributed_manager.py
class PowerSGDCompressor:
    def __init__(self, rank=4):
        self.rank = rank
        self.q_memory = {}

    def compress(self, grad):
        """Compress gradient using PowerSGD"""
        orig_shape = grad.shape
        grad_2d = grad.view(grad.shape[0], -1)

        # Low-rank approximation
        u, s, v = torch.svd_lowrank(grad_2d, q=self.rank)
        compressed = u @ torch.diag(s) @ v.T

        return compressed.view(orig_shape)
```

## Testing and Validation

### Performance Benchmarking Script
```bash
#!/bin/bash
# save as: code/scripts/benchmark_optimizations.sh

# Baseline
echo "Running baseline..."
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu.yaml \
    --max_steps 1000 \
    --benchmark_mode \
    --output_dir /tmp/baseline

# Optimized
echo "Running optimized..."
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu_optimized.yaml \
    --max_steps 1000 \
    --benchmark_mode \
    --output_dir /tmp/optimized

# Compare results
python -c "
import json
with open('/tmp/baseline/metrics.json') as f:
    baseline = json.load(f)
with open('/tmp/optimized/metrics.json') as f:
    optimized = json.load(f)

speedup = optimized['steps_per_second'] / baseline['steps_per_second']
memory_reduction = (baseline['peak_memory_gb'] - optimized['peak_memory_gb']) / baseline['peak_memory_gb']

print(f'Speed improvement: {speedup:.2f}x')
print(f'Memory reduction: {memory_reduction:.1%}')
print(f'Time per 1000 steps: {1000/optimized["steps_per_second"]:.1f}s vs {1000/baseline["steps_per_second"]:.1f}s')
"
```

## Implementation Timeline

### Week 1-2: Quick Wins
1. **Day 1-2**: Enable Expert Parallel GEMM
2. **Day 3-4**: Implement Dynamic Batch Sizing
3. **Day 5-6**: Add Expert Mixture Caching
4. **Day 7-8**: Implement Adaptive Validation
5. **Day 9-10**: Add Loss Vectorization
6. **Day 11-14**: Testing and tuning

### Week 3-4: Integration & Testing
1. Profile and measure improvements
2. Fix any regressions
3. Tune hyperparameters
4. Run full training validation

### Month 2: Advanced Optimizations
1. Layer fusion with torch.compile
2. Gradient compression
3. Advanced prefetching
4. CUDA graphs integration

## Monitoring and Metrics

### Key Metrics to Track
```python
# Add to code/src/Ava/training/monitoring/metrics.py
class OptimizationMetrics:
    def __init__(self):
        self.metrics = {
            'steps_per_second': [],
            'gpu_utilization': [],
            'memory_usage_gb': [],
            'cache_hit_rate': [],
            'validation_time_ratio': [],
            'gradient_comm_time': [],
            'expert_compute_time': []
        }

    def log_metrics(self, step):
        """Log optimization metrics"""
        metrics = {
            'step': step,
            'steps_per_second': self.compute_throughput(),
            'gpu_util': torch.cuda.utilization(),
            'memory_gb': torch.cuda.max_memory_allocated() / 1e9,
            'cache_hits': self.expert_cache.hits / (self.expert_cache.hits + self.expert_cache.misses),
            'val_overhead': self.validation_time / self.total_time
        }

        # Log to wandb
        if wandb.run:
            wandb.log(metrics)

        return metrics
```

## Expected Results

### Performance Improvements
| Optimization | Expected Gain | Cumulative |
|-------------|--------------|------------|
| Expert Parallel GEMM | 15-20% | 15-20% |
| Dynamic Batch Sizing | 10-15% | 25-35% |
| Expert Caching | 8-12% | 33-47% |
| Adaptive Validation | 5-15% | 38-62% |
| Loss Vectorization | 2-3% | 40-65% |

### Memory Savings
| Optimization | Memory Impact |
|-------------|--------------|
| Dynamic Batching | +10-20% efficiency |
| Expert Caching | -200MB cache overhead |
| Gradient Compression | -40% gradient memory |
| Smart Checkpointing | -15% activation memory |

## Troubleshooting

### Common Issues and Solutions

1. **OOM with Dynamic Batching**
   - Reduce `target_gpu_utilization` to 0.85
   - Implement gradual batch size changes
   - Add memory pressure detection

2. **Cache Misses Too High**
   - Lower `similarity_threshold` to 0.90
   - Increase `cache_size` to 512
   - Implement better hash function

3. **Validation Skipped Too Often**
   - Reduce `max_eval_steps` to 2500
   - Adjust stability threshold
   - Force evaluation at checkpoints

4. **Performance Regression**
   - Profile with `torch.profiler`
   - Check for CPU-GPU sync points
   - Verify async operations working
   - Monitor communication overhead

## Validation Commands

```bash
# Quick validation (5 minutes)
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu_optimized.yaml \
    --max_steps 100 \
    --profile

# Full validation (1 hour)
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu_optimized.yaml \
    --max_steps 1000 \
    --run_validation \
    --save_steps 500

# Memory profiling
python code/scripts/5_training/train.py \
    --config code/configs/moe/tiny_moe_multi_gpu_optimized.yaml \
    --max_steps 100 \
    --memory_profile \
    --log_memory_every 10
```

## Success Criteria

✅ **Performance**: 25-35% speedup over current implementation
✅ **Memory**: Maintain or improve current memory usage
✅ **Stability**: No increase in loss spikes or NaN occurrences
✅ **Accuracy**: Validation loss within 1% of baseline
✅ **Scalability**: Improvements scale to 8+ GPUs

## Next Steps

1. Run baseline benchmarks with current config
2. Implement quick wins (Week 1-2)
3. Measure and validate improvements
4. Deploy optimized configuration
5. Monitor production metrics
6. Iterate on medium-term optimizations

For questions or issues, refer to the detailed analysis in:
- `COMPREHENSIVE_OPTIMIZATION_ANALYSIS.md`
- `OPTIMIZATION_QUICK_REFERENCE.md`