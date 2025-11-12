# MoE Optimization Guide - Performance & Monitoring

This guide covers the performance optimizations and monitoring tools added for Sparse Mixture of Experts (MoE) models with DeepSpeed integration.

## Table of Contents

1. [Performance Optimizations](#performance-optimizations)
2. [Expert Load Balance Monitoring](#expert-load-balance-monitoring)
3. [Profiling Tools](#profiling-tools)
4. [DeepSpeed Configuration](#deepspeed-configuration)
5. [Troubleshooting](#troubleshooting)

---

## Performance Optimizations

### Automatic TF32 Acceleration (Already Integrated)

The training script **automatically enables** TF32 (TensorFloat-32) optimizations on supported GPUs (Ampere and newer: RTX 30xx, RTX 40xx, A100, H100).

**Location**: `code/scripts/5_training/train.py` lines 2274-2287

**What it does:**
```python
torch.set_float32_matmul_precision('high')            # Use TF32 for matmul
torch.backends.cuda.matmul.fp32_precision = 'tf32'   # Enable TF32 matmul (PyTorch 2.9+)
torch.backends.cudnn.conv.fp32_precision = 'tf32'    # Enable TF32 in cuDNN (PyTorch 2.9+)
torch.backends.cudnn.benchmark = True                 # Auto-tune kernels
```

**Performance impact:**
- 8x faster matrix multiplication on Ampere GPUs
- No accuracy loss (maintains float32 range with reduced precision)
- Automatically disabled on older GPUs

**Configuration:**
You can configure this via config YAML:
```yaml
performance:
  float32_matmul_precision: 'high'  # Options: 'highest', 'high', 'medium'
```

---

## Expert Load Balance Monitoring

### Overview

The `MoELoadBalanceMonitor` tracks expert utilization and provides **automatic suggestions** when load imbalance is detected.

### Features

- **Real-time monitoring** of expert usage distribution
- **Balance score** calculation (entropy-based, 0.0 to 1.0)
- **Automatic warnings** when imbalance is detected
- **Actionable suggestions** for `load_balance_loss_coef` adjustments

### Usage

#### Standalone Usage

```python
from src.Ava.training.monitoring import MoELoadBalanceMonitor

# Initialize monitor
monitor = MoELoadBalanceMonitor(
    num_experts=16,
    current_load_balance_coef=0.01,
    check_interval=100,          # Check every 100 steps
    target_balance_score=0.7,    # Target 70% balance
)

# During training loop
for batch in dataloader:
    outputs = model(input_ids, attention_mask)

    # Update monitor with expert indices
    stats = monitor.update(
        expert_indices=outputs.expert_indices,  # [num_tokens, k]
        expert_weights=outputs.expert_weights,   # [num_tokens, k]
    )

    # Check for suggestions
    if stats and stats.suggested_load_balance_coef:
        print(f"Suggestion: Increase coef to {stats.suggested_load_balance_coef}")
        # Optionally update config and restart training
```

#### Integration with Trainer

The monitor can be integrated into the training loop to track expert usage:

```python
# In your training script
moe_monitor = MoELoadBalanceMonitor(
    num_experts=config['model']['num_experts'],
    current_load_balance_coef=config['model']['load_balance_loss_coef'],
    logger=logger,
)

# In training loop
for batch in dataloader:
    outputs = model(input_ids, attention_mask)

    # Extract MoE metrics if available
    if hasattr(outputs, 'moe_metrics'):
        stats = moe_monitor.update(moe_metrics=outputs.moe_metrics)
```

### Metrics Explained

#### Balance Score
- **Range**: 0.0 (worst) to 1.0 (perfect balance)
- **Calculation**: Entropy-based normalization
- **Target**: 0.6-0.9 (typical for well-balanced MoE)
- **Interpretation**:
  - `> 0.8`: Excellent balance
  - `0.6-0.8`: Good balance
  - `0.4-0.6`: Moderate imbalance (action recommended)
  - `< 0.4`: Severe imbalance (expert collapse)

#### Coefficient of Variation (CV)
- **Formula**: `std(usage) / mean(usage)`
- **Target**: < 0.5
- **High CV** (> 1.0): Large variation in expert usage

#### Max/Min Ratio
- **Formula**: `max(usage) / min(usage)`
- **Target**: < 5x
- **Interpretation**:
  - `< 3x`: Excellent
  - `3-5x`: Good
  - `5-10x`: Moderate imbalance
  - `> 10x`: Severe (expert collapse)

### Automatic Suggestions

The monitor provides graded suggestions based on imbalance severity:

| Balance Deficit | Severity | Multiplier | Example |
|----------------|----------|------------|---------|
| 0.05-0.15 | Mild | 1.25x | 0.01 → 0.0125 |
| 0.15-0.30 | Moderate | 1.5x | 0.01 → 0.015 |
| > 0.30 | Severe | 2.0x | 0.01 → 0.02 |

**Maximum cap**: 0.1 (prevents over-regularization)

---

## Profiling Tools

### Training Profiler

Profile your training loop to identify performance bottlenecks.

**Location**: `code/scripts/testing/profile_training.py`

### Usage

```bash
# Basic profiling (3 steps)
python code/scripts/testing/profile_training.py \
    --config code/configs/moe/small_moe.yaml

# Profile more steps
python code/scripts/testing/profile_training.py \
    --config code/configs/moe/deepseek_style.yaml \
    --steps 10

# Custom output directory
python code/scripts/testing/profile_training.py \
    --config code/configs/moe/small_moe.yaml \
    --output-dir /path/to/output
```

### Output

The profiler generates:

1. **Console summary** with top bottlenecks
2. **Chrome trace** (`trace.json`) - viewable at `chrome://tracing`
3. **Stack traces** for detailed analysis
4. **JSON summary** with key metrics

### Interpreting Results

#### Key Operations to Monitor

| Operation | Expected % | Issue if Higher |
|-----------|-----------|-----------------|
| `forward` | 40-50% | Model too complex |
| `backward` | 30-40% | Inefficient gradients |
| `optimizer_step` | 5-10% | Too many parameters |
| `data_transfer` | < 5% | Slow data pipeline |
| `moe_routing` | 5-10% | Router bottleneck |
| `expert_computation` | 30-40% | Expert efficiency |

#### Example Output

```
🔥 Top 10 Operations by CUDA Time:
-------------------------------------------------------------------------------------------------
  Name                           Self CPU    Self CUDA    CPU total    CUDA total  # Calls
-------------------------------------------------------------------------------------------------
  forward                        12.34ms     123.45ms     234.56ms     345.67ms    3
  MoELayer::routing              5.67ms      56.78ms      78.90ms      89.01ms     12
  expert_computation             23.45ms     234.56ms     234.56ms     234.56ms    48
  backward                       8.90ms      89.01ms      123.45ms     234.56ms    3
  optimizer_step                 2.34ms      23.45ms      34.56ms      45.67ms     3
```

#### Common Bottlenecks

1. **Data Transfer > 10%**
   - Solution: Increase `num_workers`, enable `pin_memory`
   - Config: `data.num_workers: 8`, `dataloader_pin_memory: true`

2. **MoE Routing > 15%**
   - Solution: Enable Triton kernels, reduce `num_experts`
   - Config: `use_triton_kernels: true`

3. **Expert Computation > 50%**
   - Solution: Enable grouped GEMM, reduce expert size
   - Config: `use_grouped_gemm: true`

---

## DeepSpeed Configuration

### Recommended Settings for MoE

```yaml
deepspeed:
  enabled: true
  zero_stage: 2          # Use ZeRO-2 for MoE (NOT ZeRO-3)
  precision: "bf16"      # BF16 for stability

  # ZeRO optimization settings
  zero_optimization:
    stage: 2
    allgather_partitions: true
    reduce_scatter: true
    overlap_comm: true
    contiguous_gradients: true

  # BF16 settings
  bf16:
    enabled: true

  # Gradient clipping (important for MoE)
  gradient_clipping: 1.0

  # Batch size (CRITICAL for initialization)
  train_batch_size: 256      # Must match formula
  micro_batch_size: 64       # Per-GPU batch size
  gradient_accumulation_steps: 4
```

### Important Notes

1. **ZeRO-3 Compatibility**: ZeRO-3 can have issues with sparse MoE gradients. Use ZeRO-2 instead.

2. **Batch Size Formula**: `train_batch_size = micro_batch_size × gradient_accumulation_steps × num_gpus`

3. **CPU Offload**: Disable for MoE (causes issues with sparse gradients)
   ```yaml
   cpu_offload: false
   cpu_offload_params: false
   ```

4. **Expert Parallelism** (Multi-GPU):
   ```yaml
   moe:
     enabled: true
     moe_experts: 16
     moe_top_k: 2
     moe_min_capacity: 4
     moe_normalize_expert_weights: true
   ```

### Performance Tuning

#### Memory Optimization
```yaml
model:
  gradient_checkpointing: true   # Saves memory
  use_grouped_gemm: true         # 5-10x faster expert computation
```

#### Compute Optimization
```yaml
model:
  use_triton_kernels: true       # Faster routing
  use_torch_compile: true        # 10-20% overall speedup
```

#### Load Balancing
```yaml
model:
  router_z_loss_coef: 0.001      # Router regularization
  load_balance_loss_coef: 0.02   # Increase if imbalanced
  diversity_loss_coef: 0.002     # Encourage diversity
  capacity_factor: 1.25          # Increase if tokens dropped
```

---

## Troubleshooting

### Expert Collapse (All tokens → same experts)

**Symptoms:**
- Balance score < 0.4
- Max/Min ratio > 10x
- Training loss not improving

**Solutions:**
1. Increase `load_balance_loss_coef` from 0.01 → 0.03-0.05
2. Increase `router_jitter_noise` from 0.01 → 0.02
3. Add `diversity_loss_coef: 0.002` if not present
4. Reduce learning rate temporarily

### OOM Errors

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes mid-batch

**Solutions:**
1. Enable gradient checkpointing: `gradient_checkpointing: true`
2. Reduce batch size: `batch_size: 32` → `16`
3. Increase gradient accumulation: `gradient_accumulation_steps: 8`
4. Reduce model size: `num_experts: 16` → `8`

### Slow Training

**Symptoms:**
- Tokens/sec < 1000
- GPU utilization < 80%

**Solutions:**
1. Enable all optimizations:
   ```yaml
   use_grouped_gemm: true
   use_triton_kernels: true
   use_torch_compile: true
   ```
2. Increase batch size (if memory allows)
3. Enable mixed precision: `mixed_precision: "bf16"`
4. Profile to identify bottleneck (see Profiling Tools above)

### DeepSpeed + MoE Issues

**Symptoms:**
- `AssertionError: ZeRO-3 not compatible with MoE`
- Gradients are None for some experts

**Solutions:**
1. Use ZeRO-2 instead of ZeRO-3:
   ```yaml
   deepspeed:
     zero_stage: 2
   ```
2. Disable CPU offload:
   ```yaml
   cpu_offload: false
   cpu_offload_params: false
   ```
3. Ensure batch size formula is correct

### Token Dropping

**Symptoms:**
- Warning: "Tokens dropped due to capacity"
- Training unstable

**Solutions:**
1. Increase `capacity_factor` from 1.25 → 1.5 or 2.0
2. Reduce `num_experts_per_token` if very high
3. Monitor with `expert_capacity_histogram` metric

---

## Quick Reference

### Config Template for Optimal Performance

```yaml
model:
  # MoE settings
  num_experts: 16
  num_experts_per_token: 2
  router_type: 'deepseek'        # or 'mixtral'
  use_shared_expert: true        # DeepSeek style
  shared_expert_weight: 0.5

  # Performance
  use_grouped_gemm: true
  use_triton_kernels: true
  use_torch_compile: true
  gradient_checkpointing: true   # If memory limited

  # Load balancing
  capacity_factor: 1.25
  router_z_loss_coef: 0.001
  load_balance_loss_coef: 0.02   # Adjust based on monitoring
  diversity_loss_coef: 0.002
  router_jitter_noise: 0.01

training:
  batch_size: 64
  gradient_accumulation_steps: 4
  learning_rate: 1.5e-4
  weight_decay: 0.1

hardware:
  mixed_precision: "bf16"
  compile: true

deepspeed:
  enabled: true
  zero_stage: 2                  # NOT 3 for MoE
  precision: "bf16"
  gradient_clipping: 1.0
```

### Monitoring Checklist

- [ ] Balance score > 0.6
- [ ] Max/Min ratio < 5x
- [ ] GPU utilization > 80%
- [ ] No token dropping warnings
- [ ] Loss decreasing steadily
- [ ] TF32 enabled (check logs)

---

## Additional Resources

- **Profiling**: `python code/scripts/testing/profile_training.py --help`
- **MoE Architecture**: `code/docs/SPARSE_MOE_GUIDE.md`
- **Training Examples**: `code/configs/moe/*.yaml`

For issues or questions, check the training logs or run the profiler to identify bottlenecks.
