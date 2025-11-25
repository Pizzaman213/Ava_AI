# Quick Start: Memory Optimizations

## TL;DR - Use These Now!

### 1. GaLore Optimizer (50-65% gradient memory savings)

```python
from Ava.optimization.optimizers import create_galore_optimizer

optimizer = create_galore_optimizer(
    model,
    optimizer_type='adamw',  # or 'lion'
    lr=1e-3,
    rank=128
)
```

### 2. Memory Dashboard (track and optimize memory)

```python
from Ava.training.monitoring import profile_training

dashboard = profile_training(model)
# ... training loop ...
dashboard.save_report('memory_report.html')
```

### 3. Predictive Expert Prefetching (automatic for MoE)

```python
# Already enabled! Just check stats:
stats = expert_layer.get_prediction_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

---

## Configuration File Approach

Use the example config:

```bash
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/memory/galore_memory_efficient.yaml
```

---

## When to Use What

### GaLore Optimizer
✅ Use when:
- Training large models (>1B params)
- GPU memory is limited
- Want to increase batch size

❌ Skip when:
- Training small models (<100M params)
- Memory is abundant
- Speed is critical (adds 10-15% overhead)

### Memory Dashboard
✅ Use when:
- First time training a model
- Experiencing OOM errors
- Optimizing memory usage
- Need to justify hardware purchases

❌ Skip when:
- Memory is not a concern
- Training is stable
- Production deployment (minimal overhead but unnecessary)

### Predictive Prefetching
✅ Use when:
- Training MoE models with CPU offloading
- Have >8 experts
- Expert transfer is a bottleneck

❌ Skip when:
- Not using MoE
- All experts fit on GPU
- <8 experts (limited benefit)

---

## Memory Optimization Checklist

Use this checklist to maximize memory efficiency:

### Priority 1: Essential (Always Use)
- [ ] Mixed precision (FP16/BF16) → 50% model memory
- [ ] Gradient checkpointing → 60-80% activation memory
- [ ] Flash Attention → 50-70% attention memory

### Priority 2: High Impact
- [ ] 8-bit optimizer (Lion8bit) → 87.5% optimizer memory
- [ ] GaLore (if memory-constrained) → 50-65% gradient memory

### Priority 3: For Specific Cases
- [ ] DeepSpeed ZeRO-3 (multi-GPU) → 4x model memory
- [ ] MoE LoRA experts (MoE models) → 80-96% expert memory
- [ ] CPU offloading (very large models) → 2-8x memory

### Monitoring
- [ ] Memory Dashboard → Track and optimize

---

## Expected Memory Savings

### Small Model (100M params)
```
Baseline:              ~2-3 GB
+ Mixed precision:     ~1-1.5 GB    (50% saved)
+ Grad checkpoint:     ~0.6-0.9 GB  (33% saved)
+ 8-bit optimizer:     ~0.4-0.6 GB  (33% saved)
Final:                 ~0.4-0.6 GB

Total Reduction: 80-86%
```

### Medium Model (1B params)
```
Baseline:              ~18-20 GB
+ Mixed precision:     ~10-11 GB    (45% saved)
+ Grad checkpoint:     ~6-7 GB      (36% saved)
+ GaLore:              ~5-6 GB      (17% saved)
+ 8-bit optimizer:     ~3-4 GB      (33% saved)
Final:                 ~3-4 GB

Total Reduction: 78-83%
```

### Large Model (7B params)
```
Baseline:              ~120-140 GB
+ Mixed precision:     ~65-75 GB     (46% saved)
+ Grad checkpoint:     ~35-45 GB     (38% saved)
+ GaLore:              ~28-36 GB     (20% saved)
+ DeepSpeed ZeRO-3:    ~10-15 GB/GPU (60-70% saved per GPU)
Final:                 ~10-15 GB per GPU (4 GPUs)

Total Reduction: 86-90%
```

---

## Troubleshooting

### "Out of memory" Error

1. **Check current usage**:
   ```python
   from Ava.training.monitoring import MemoryDashboard

   dashboard = MemoryDashboard(model)
   dashboard.start_profiling()
   dashboard.print_summary()
   dashboard.print_recommendations()  # Follow these!
   ```

2. **Quick fixes** (in order of impact):
   - Reduce batch size by 50%
   - Enable gradient checkpointing
   - Use GaLore optimizer
   - Enable DeepSpeed ZeRO-3
   - Add CPU offloading

### "GaLore training diverges"

1. **Increase rank**: Try `rank=256` instead of `rank=128`
2. **Decrease LR**: Reduce by 20-30%
3. **Update more frequently**: Try `update_proj_gap=100`

### "Predictive prefetch not helping"

1. **Check hit rate**: Should be >40% after warmup
   ```python
   stats = experts.get_prediction_stats()
   print(stats)
   ```

2. **Increase predictions**: Try `prefetch_lookahead=5`
3. **Wait for warmup**: Need ~100 steps for transition matrix

---

## One-Line Improvements

Copy-paste these into your training script:

```python
# 50-65% gradient memory reduction
optimizer = create_galore_optimizer(model, 'adamw', lr=1e-3, rank=128)

# Real-time memory monitoring
dashboard = profile_training(model)

# Track memory at each step
dashboard.record_step(step=i, loss=loss.item())

# Get expert prediction stats (MoE only)
stats = expert_layer.get_prediction_stats()
```

---

## Full Example: Maximum Memory Efficiency

```python
import torch
from Ava.models.moe_model import MoEModel
from Ava.optimization.optimizers import create_galore_optimizer
from Ava.training.monitoring import profile_training

# 1. Enable all PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# 2. Create memory-efficient model
model = MoEModel(
    num_experts=32,
    hidden_size=1024,
    use_lora=True,              # 80-96% expert memory
    lora_rank=8,
    expert_type='cpu_offloaded',  # Offload experts to CPU
    prefetch_lookahead=3,       # Predictive prefetching
    use_gradient_checkpointing=True  # 60-80% activation memory
).half()  # FP16 for 50% model memory

# 3. Use GaLore optimizer
optimizer = create_galore_optimizer(
    model,
    optimizer_type='lion',  # Most memory-efficient
    lr=1e-4,
    rank=128,              # 50-65% gradient memory
    weight_decay=0.01
)

# 4. Enable memory monitoring
dashboard = profile_training(model)

# 5. Training loop
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    output = model(batch)
    loss = compute_loss(output)

    loss.backward()
    optimizer.step()

    # Monitor memory
    dashboard.record_step(step=step, loss=loss.item())

# 6. Get recommendations
dashboard.save_report('memory_report.html')
dashboard.print_recommendations()
```

---

## Quick Config Template

```yaml
# configs/memory/my_memory_efficient_config.yaml

optimizer:
  type: "galore_lion"  # Maximum memory efficiency
  lr: 1.0e-4
  galore_params:
    rank: 128
    update_proj_gap: 200

training:
  batch_size: 16
  gradient_checkpointing: true
  mixed_precision: "bf16"
  flash_attention: true

model:
  moe:
    num_experts: 32
    use_lora: true
    lora_rank: 8
    cpu_offloading:
      enabled: true
      prefetch_lookahead: 3

monitoring:
  memory_dashboard:
    enabled: true
    sample_interval: 10
```

---

## Results Snapshot

With all optimizations:

| Model Size | Baseline | Optimized | Reduction |
|------------|----------|-----------|-----------|
| 100M | 2-3 GB | 0.4-0.6 GB | 80-86% |
| 1B | 18-20 GB | 3-4 GB | 78-83% |
| 7B | 120-140 GB | 10-15 GB/GPU | 86-90% |

**This means you can:**
- Train 1B models on consumer GPUs (RTX 3090)
- Train 7B models on 4x A100 instead of 8x
- Increase batch size 2-3x for faster training

---

## Need More Help?

📚 Full Documentation: [`code/docs/MEMORY_EFFICIENCY_ENHANCEMENTS.md`](code/docs/MEMORY_EFFICIENCY_ENHANCEMENTS.md)

📊 Summary: [`MEMORY_ENHANCEMENTS_SUMMARY.md`](MEMORY_ENHANCEMENTS_SUMMARY.md)

⚙️ Config Examples: [`code/configs/memory/`](code/configs/memory/)

🧪 Tests: [`code/scripts/test_galore_optimizer.py`](code/scripts/test_galore_optimizer.py)

---

**Start with GaLore and Memory Dashboard - you'll see immediate benefits! 🚀**
