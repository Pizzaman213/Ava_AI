# Model Upgraded: 100M → 233M Parameters

**Date**: 2025-10-16
**Config**: [small.yaml](../code/configs/gpu/small.yaml)

## Summary

✅ **Successfully upgraded model from 100M to 233M parameters** (2.3x larger)

## Configuration Changes

| Parameter | Old (100M) | New (233M) | Change |
|-----------|------------|------------|--------|
| **hidden_size** | 512 | 640 | +25% |
| **num_layers** | 6 | 10 | +67% |
| **num_attention_heads** | 8 | 10 | +25% |
| **intermediate_size** | 2048 | 2560 | +25% |
| **num_experts** | 2 | 4 | +100% |
| **vocab_size** | 65,536 | 65,536 | (unchanged) |

## Parameter Breakdown

### Old Model (100M)
```
Embeddings:      34.6M (34.7%)
Transformer:     31.5M (31.6%)
Output:          33.6M (33.7%)
─────────────────────────────
TOTAL:           99.6M
```

### New Model (233M)
```
Embeddings:      43.3M (18.6%)
Transformer:    147.5M (63.4%)
Output:          41.9M (18.0%)
─────────────────────────────
TOTAL:          232.7M
```

## Architecture Improvements

### 1. **More Depth** (10 layers vs 6)
- Better sequential reasoning
- Longer information propagation
- More complex transformations

### 2. **More Width** (640 hidden vs 512)
- Richer representations
- More expressive embeddings
- Better feature learning

### 3. **More Experts** (4 vs 2)
- Greater specialization
- More diverse knowledge
- Better capacity utilization

## Memory Requirements

| Metric | Size | % of 24GB |
|--------|------|-----------|
| Model weights (BF16) | 0.43 GB | 1.8% |
| Optimizer states | 1.73 GB | 7.2% |
| **Total (approx)** | **2.17 GB** | **9.0%** |

✓ **Fits comfortably on RTX 3090 Ti with plenty of room for batch size**

## Expected Benefits

### Better Language Understanding
- **100M**: Basic patterns, simple coherence
- **233M**: Complex patterns, better coherence, nuanced understanding

### Reduced Repetition
- More capacity to learn diverse expressions
- Better long-range dependencies (10 layers vs 6)
- More expert specialization (4 experts vs 2)

### Improved Generation Quality
With clean data + repetition penalties + larger model:
- **Target repetition**: 20-25% (vs current 59%)
- **Target coherence**: 60-75/100 (vs current 15/100)
- **Better creativity and diversity**

## Training Impact

### Speed
- **100M**: ~1.58 it/s (16 batch size)
- **233M**: ~1.1-1.3 it/s (estimate, 16 batch size)
- **Impact**: ~20-30% slower (still very fast)

### Memory
- More room for larger batch sizes if needed
- Can still use gradient accumulation effectively

### Convergence
- May need slightly more steps to converge
- But should achieve better final performance

## Next Steps

### Before Training

1. **✅ Model upgraded to 233M**
2. **⏳ Clean training data** (remove conversational artifacts)
   - Remove: OpenOrca, Anthropic RLHF, UltraChat
   - Keep: Wikipedia, C4, fineweb-edu, orca-math
3. **⏳ Stop current training** (learning from bad data)

### Start Fresh Training

With clean data + 233M model + repetition penalties:

```bash
# 1. Clean data first
./CLEAN_DATA_NOW.sh

# 2. Start training
cd /project/code
python3 -m Ava.training.run_manager --config configs/gpu/small.yaml --mode train
```

### Expected Timeline (233M model)

- **Step 1000**: Repetition 40-50%, Coherence 25-35/100
- **Step 5000**: Repetition 25-35%, Coherence 45-60/100
- **Step 10000**: Repetition 20-25%, Coherence 60-75/100 ✓

## Why This Size?

### Design Philosophy

**Balanced approach:**
- ✓ Not too small (100M struggles with complexity)
- ✓ Not too large (1B+ slow to iterate)
- ✓ Sweet spot for fast experimentation
- ✓ Good enough for coherent story generation

### Comparable Models

- GPT-2 Small: 117M (similar size)
- GPT-2 Medium: 345M (larger)
- Our model: 233M (between small and medium)

### Advantages Over 100M

1. **Better coherence** - more layers for reasoning
2. **Richer vocabulary** - wider embeddings
3. **More specialization** - 4 experts vs 2
4. **Still fast** - trains quickly on RTX 3090 Ti

## Configuration File

Updated configuration saved in: [configs/gpu/small.yaml](../code/configs/gpu/small.yaml)

```yaml
model:
  hidden_size: 640
  num_layers: 10
  num_attention_heads: 10
  intermediate_size: 2560
  num_experts: 4
  # ... rest of config
```

## Summary

🎉 **Model successfully upgraded to 233M parameters!**

- **2.3x larger** than before
- **Still efficient** (only 9% of GPU memory)
- **Better capacity** for language understanding
- **Ready to train** once data is cleaned

The combination of:
1. ✅ Larger model (233M)
2. ✅ Repetition penalties (fixed)
3. ⏳ Clean data (pending)

Should result in **significantly better coherence and reduced repetition**!
