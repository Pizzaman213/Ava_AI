# Quick Start: MoE Optimizations & Monitoring

This guide gets you started with the new MoE performance optimizations and monitoring tools in under 5 minutes.

## ✅ What's Already Working

### 1. TF32 Acceleration (Automatic)

**Status**: ✅ Already enabled in training script

Your training script **automatically** enables TF32 on supported GPUs:
- RTX 30xx/40xx series (Ampere/Ada)
- A100, H100 (datacenter)

**Performance**: 8x faster matrix multiplication, no code changes needed!

Check your logs for confirmation:
```
✅ TF32 and CuDNN benchmark optimizations applied
```

## 🆕 New Features

### 2. Expert Load Balance Monitor

**What it does**: Tracks expert usage and warns you if experts are imbalanced

**Quick test**:
```bash
python code/scripts/examples/monitor_moe_balance.py
```

Expected output:
```
⚠️  Expert Load Imbalance Detected (Step 10):
   Balance Score: 0.470 (target: 0.700)
   💡 Suggestion: Increase load_balance_loss_coef from 0.0100 to 0.0150
```

### 3. Training Profiler

**What it does**: Identifies performance bottlenecks in your training loop

**Quick test**:
```bash
python code/scripts/testing/profile_training.py \
    --config code/configs/moe/small_moe.yaml \
    --steps 3
```

**Output**:
- Console summary of bottlenecks
- `outputs/profiling/trace.json` (view in Chrome at `chrome://tracing`)
- Performance recommendations

## 📋 Common Use Cases

### Use Case 1: Check if TF32 is Enabled

```bash
# Run training and check logs
python code/scripts/5_training/train.py \
    --config code/configs/moe/small_moe.yaml

# Look for this line in output:
# ✅ TF32 and CuDNN benchmark optimizations applied
```

### Use Case 2: My Experts Are Imbalanced

**Symptoms**:
- Some experts never used
- Training loss not improving
- Warnings in logs about expert collapse

**Solution**:
1. Check current balance score (see logs)
2. Increase `load_balance_loss_coef` in your config:

```yaml
# Before
model:
  load_balance_loss_coef: 0.01

# After (for moderate imbalance)
model:
  load_balance_loss_coef: 0.03

# After (for severe imbalance)
model:
  load_balance_loss_coef: 0.05
```

3. Restart training

### Use Case 3: Training is Slow

**Step 1**: Profile to find the bottleneck
```bash
python code/scripts/testing/profile_training.py \
    --config code/configs/moe/your_config.yaml
```

**Step 2**: Check the output

If **Data Transfer** is slow (> 10%):
```yaml
data:
  num_workers: 8              # Increase
  dataloader_pin_memory: true # Enable
```

If **MoE Routing** is slow (> 15%):
```yaml
model:
  use_triton_kernels: true    # Enable
  use_grouped_gemm: true      # Enable
```

If **Expert Computation** is slow (> 50%):
```yaml
model:
  use_grouped_gemm: true      # Critical!
  use_torch_compile: true     # 10-20% speedup
```

### Use Case 4: Out of Memory (OOM)

**Quick fixes**:
```yaml
training:
  batch_size: 32              # Reduce from 64
  gradient_accumulation_steps: 8  # Increase from 4

model:
  gradient_checkpointing: true    # Enable
  num_experts: 8                  # Reduce from 16
```

## 🔧 Configuration Templates

### Optimal for Single GPU (24GB VRAM)

```yaml
# code/configs/moe/small_moe.yaml (already configured)
model:
  num_experts: 4
  hidden_size: 256
  num_layers: 4
  use_grouped_gemm: true
  gradient_checkpointing: false  # Not needed for small model

training:
  batch_size: 16
  gradient_accumulation_steps: 1

deepspeed:
  enabled: false  # Not needed for small model
```

### Optimal for Multi-GPU (8x A100)

```yaml
# code/configs/moe/deepseek_style.yaml (reference)
model:
  num_experts: 16
  hidden_size: 4096
  num_layers: 32
  use_grouped_gemm: true
  use_triton_kernels: true
  gradient_checkpointing: true

training:
  batch_size: 64
  gradient_accumulation_steps: 4

deepspeed:
  enabled: true
  zero_stage: 2               # NOT 3 for MoE!
  precision: "bf16"
```

## 📊 Monitoring Checklist

Before each training run, verify:

- [ ] **TF32 enabled**: Check for log message
- [ ] **Balance score > 0.6**: Monitor during training
- [ ] **GPU utilization > 80%**: Use `nvidia-smi`
- [ ] **No OOM errors**: Reduce batch size if needed
- [ ] **Tokens/sec > 1000**: Profile if slower

## 🐛 Troubleshooting

### Problem: "DeepSpeed ZeRO incompatible with sparse MoE"

**Solution**: Use ZeRO-2 instead of ZeRO-3
```yaml
deepspeed:
  zero_stage: 2  # NOT 3
```

### Problem: "Balance score = 0.2" (severe imbalance)

**Solution**: Increase load balance coefficient
```yaml
model:
  load_balance_loss_coef: 0.05  # From 0.01
```

### Problem: Training is 10x slower than expected

**Solution**: Enable all optimizations
```yaml
model:
  use_grouped_gemm: true     # 5-10x faster!
  use_triton_kernels: true   # Faster routing
  use_torch_compile: true    # 10-20% faster
```

## 📚 Full Documentation

- **Comprehensive Guide**: [MOE_OPTIMIZATION_GUIDE.md](MOE_OPTIMIZATION_GUIDE.md)
- **MoE Architecture**: [SPARSE_MOE_GUIDE.md](SPARSE_MOE_GUIDE.md)
- **Example Configs**: `code/configs/moe/*.yaml`

## 🚀 Next Steps

1. **Test the monitor**: `python code/scripts/examples/monitor_moe_balance.py`
2. **Profile your training**: `python code/scripts/testing/profile_training.py --config your_config.yaml`
3. **Review your config** against the templates above
4. **Start training** and monitor the logs!

## 💡 Pro Tips

1. **Always start small**: Test with `small_moe.yaml` before scaling up
2. **Profile first**: Don't guess at bottlenecks, measure them
3. **Monitor balance**: Check balance score every 1000 steps
4. **Enable all optimizations**: `use_grouped_gemm`, `use_triton_kernels`, `use_torch_compile`
5. **Use BF16**: More stable than FP16 for MoE models

---

**Questions?** Check the full guide at [MOE_OPTIMIZATION_GUIDE.md](MOE_OPTIMIZATION_GUIDE.md)
