# Memory Optimization Reference Guide

This guide explains when and how to use each memory optimization technique in the Ava MoE training pipeline.

## Available Optimization Techniques

### 1. LoRA (Low-Rank Adaptation)
**Implementation:** [lora_experts.py](../src/Ava/layers/lora_experts.py)

**What it does:**
- Shares a base weight matrix across all experts
- Adds small per-expert low-rank delta matrices (ΔW = B @ A)
- Formula: `W_expert = W_base + (lora_alpha / rank) * (B @ A)`

**Memory savings:** 50-86% depending on rank
- Rank 4: ~86% savings
- Rank 8: ~75% savings
- Rank 16: ~50% savings

**Configuration:**
```yaml
moe_memory_optimization:
  use_lora_experts: true
  lora_rank: 4        # Lower = more savings, less capacity
  lora_alpha: 8       # Scaling factor (typically 2x rank)
  freeze_lora_base: false  # Whether to freeze shared base weights
```

**When to use:**
- ✅ Training with limited GPU memory
- ✅ When experts should share common patterns
- ✅ Fast iteration during development
- ❌ When you need full parameter capacity per expert

**Training impact:**
- Speed: ~10-20% faster (fewer parameters to update)
- Quality: Slight reduction in capacity (mitigated by proper rank selection)

---

### 2. Expert Offloading
**Implementation:** [offloaded_experts.py](../src/Ava/layers/offloaded_experts.py)

**What it does:**
- Stores all experts on CPU memory
- Keeps only active experts on GPU (LRU cache)
- Async prefetching and batched transfers for efficiency

**Memory savings:** 75-87% GPU memory
- 2/8 active experts: ~75% savings
- 1/8 active experts: ~87% savings

**Configuration:**
```yaml
moe_memory_optimization:
  use_expert_offloading: true
  max_active_experts_gpu: 2          # Number of experts cached on GPU
  offload_prefetch_lookahead: 1      # Prefetch next N experts
  offload_eviction_policy: lru       # or 'lfu', 'fifo'
  offload_pin_memory: true          # Use pinned memory for faster transfers
  offload_async_transfers: true     # Async CPU↔GPU transfers
```

**When to use:**
- ✅ Large models that don't fit in GPU memory
- ✅ Many experts (8+)
- ✅ When CPU→GPU transfer speed is acceptable
- ❌ When training speed is critical (adds ~30-50% overhead)

**Training impact:**
- Speed: ~30-50% slower (due to transfers)
- Quality: No impact on model capacity
- Hardware: Requires fast CPU-GPU interconnect (PCIe 4.0+)

---

### 3. Expert Quantization
**Implementation:** [quantized_lora_experts.py](../src/Ava/layers/quantized_lora_experts.py)

**What it does:**
- Quantizes expert weights to INT8 or INT4
- Per-channel quantization for better accuracy
- Automatic dequantization on GPU transfer

**Memory savings:** 75% (INT8) or 87.5% (INT4)

**Configuration:**
```yaml
moe_memory_optimization:
  use_expert_quantization: true
  expert_quantization_bits: 8     # 8 or 4
```

**When to use:**
- ✅ Extreme memory constraints
- ✅ Combined with offloading (quantized storage on CPU)
- ✅ When slight accuracy loss is acceptable
- ❌ When FP16 precision is critical
- ❌ When quantization overhead outweighs benefits

**Training impact:**
- Speed: ~20-30% slower (quantize/dequantize overhead)
- Quality: Minimal impact with INT8, ~1-2% degradation with INT4
- Note: Gradients are always computed in full precision

---

## Combination Strategies

### Strategy 1: LoRA Only
**Memory savings:** ~86% (rank=4)
**Speed impact:** +10-20% faster
**Use case:** Fast development, moderate memory constraints

```yaml
moe_memory_optimization:
  use_lora_experts: true
  lora_rank: 4
  lora_alpha: 8
  use_expert_offloading: false
  use_expert_quantization: false
```

---

### Strategy 2: LoRA + Offloading
**Memory savings:** ~96% GPU memory
**Speed impact:** -30-50% slower
**Use case:** Large models, limited GPU memory, acceptable training time

```yaml
moe_memory_optimization:
  use_lora_experts: true
  lora_rank: 4
  lora_alpha: 8
  use_expert_offloading: true
  max_active_experts_gpu: 2
  offload_pin_memory: true
  offload_async_transfers: true
  use_expert_quantization: false
```

**Example config:** [tiny_moe_ultra_low_mem.yaml](../configs/old/tiny_moe_ultra_low_mem.yaml) (before fix)

---

### Strategy 3: LoRA + Offloading + Quantization (MAXIMUM SAVINGS)
**Memory savings:** ~99% GPU memory
**Speed impact:** -40-60% slower
**Use case:** Extreme memory constraints, research, multi-task training

```yaml
moe_memory_optimization:
  use_lora_experts: true
  lora_rank: 4
  lora_alpha: 8
  use_expert_offloading: true
  max_active_experts_gpu: 2
  offload_pin_memory: true
  offload_async_transfers: true
  use_expert_quantization: true
  expert_quantization_bits: 8
```

**Example config:** [tiny_moe_ultra_low_mem.yaml](../configs/old/tiny_moe_ultra_low_mem.yaml) (after fix)

**Memory breakdown for 8 experts:**
- Standard: ~8MB GPU memory
- LoRA only: ~1.08MB (86% savings)
- LoRA + Offload: ~0.27MB (96% savings)
- **LoRA + Offload + Quant: ~0.08MB (99% savings)** ⭐

---

### Strategy 4: Offloading + Quantization (No LoRA)
**Memory savings:** ~93% GPU memory
**Speed impact:** -40-50% slower
**Use case:** Need full expert capacity, extreme memory constraints

```yaml
moe_memory_optimization:
  use_lora_experts: false
  use_expert_offloading: true
  max_active_experts_gpu: 2
  use_expert_quantization: true
  expert_quantization_bits: 8
```

---

## Decision Matrix

| Constraint | Recommended Strategy |
|------------|---------------------|
| **Limited GPU memory (4-8GB)** | LoRA + Offloading + Quantization |
| **Fast iteration needed** | LoRA only |
| **Maximum model quality** | Standard (no optimizations) |
| **Many experts (16+)** | LoRA + Offloading |
| **Research/experimentation** | LoRA only |
| **Production training** | LoRA + Offloading (good balance) |

---

## Performance Considerations

### Training Speed vs Memory Tradeoff

| Configuration | GPU Memory | Training Speed | Model Quality |
|--------------|------------|----------------|---------------|
| Standard | 100% | 100% (baseline) | 100% |
| LoRA (r=4) | ~14% | 110-120% ⬆️ | ~98% |
| LoRA + Offload | ~4% | 50-70% ⬇️ | ~98% |
| LoRA + Offload + Quant | ~1% | 40-60% ⬇️ | ~96-97% |

### Hardware Requirements

**For Offloading:**
- Fast CPU-GPU interconnect (PCIe 4.0+ recommended)
- Sufficient CPU RAM (2-4x GPU memory)
- Pinned memory support

**For Quantization:**
- No special hardware requirements
- INT8 support on modern GPUs (Ampere+) helps slightly

---

## Troubleshooting

### Issue: Slow training with offloading
**Solutions:**
1. Increase `max_active_experts_gpu` if you have GPU memory
2. Enable `offload_async_transfers: true`
3. Enable `offload_pin_memory: true`
4. Increase `offload_prefetch_lookahead` to 2-3

### Issue: Out of memory even with all optimizations
**Solutions:**
1. Reduce `max_active_experts_gpu` to 1
2. Lower `batch_size` and increase `gradient_accumulation_steps`
3. Enable gradient checkpointing: `gradient_checkpointing: true`
4. Use INT4 quantization instead of INT8

### Issue: Quality degradation with quantization
**Solutions:**
1. Use INT8 instead of INT4
2. Increase LoRA rank (4→8)
3. Unfreeze base weights: `freeze_lora_base: false`
4. Add quality monitoring during training

---

## Implementation Details

### Code Architecture

The three optimizations are implemented in separate expert classes:
- `Expert` - Standard expert (no optimizations)
- `LoRAExpert` - LoRA only
- `QuantizedExpert` - Quantization only
- `QuantizedLoRAExpert` - LoRA + Quantization

Offloading is handled by `CPUOffloadedExpertGroup`, which wraps any expert type.

### Integration Points

1. **MoE Layer:** [moe_layer.py:197-218](../src/Ava/models/moe_layer.py#L197-L218)
   - Selects appropriate expert type based on config flags
   - Creates `CPUOffloadedExpertGroup` when offloading/quantization enabled

2. **Trainer:** [trainer.py](../src/Ava/training/core/trainer.py)
   - Device management handled automatically via PyTorch hooks
   - No special training logic needed

3. **Config:** [training_config.py](../src/Ava/config/training_config.py)
   - All optimization flags defined in `moe_memory_optimization` section

---

## References

- **LoRA Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Expert Offloading:** Inspired by DeepSpeed CPU offloading
- **Quantization:** Per-channel INT8 quantization with automatic dequantization

---

## Example Configs

| Config | Features | GPU Memory | Use Case |
|--------|----------|------------|----------|
| [tiny_moe_ultra_low_mem.yaml](../configs/old/tiny_moe_ultra_low_mem.yaml) | All 3 | ~0.08MB | Extreme savings |
| [tiny_moe_test.yaml](../configs/old/old/tiny_moe_test.yaml) | LoRA only | ~1MB | Fast iteration |
| [small_moe_lora.yaml](../configs/old/old/small_moe_lora.yaml) | LoRA only | ~4MB | Development |

---

## Quick Reference Commands

**Check current GPU memory usage:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Monitor training memory:**
```bash
watch -n 1 nvidia-smi
```

**Test config before full training:**
```bash
python code/scripts/5_training/train.py --config path/to/config.yaml --max_steps 100
```

---

**Last Updated:** 2025-11-12
**Maintainer:** Ava Training Pipeline Team
