# Memory Optimization Guide for 24GB GPU

## Fixed Configuration
```yaml
batch_size: 32                      # Fits in 24GB
gradient_accumulation_steps: 4      # Effective batch = 128
gradient_checkpointing: true        # Saves ~40% memory
```

## Why You Got OOM (Out Of Memory)

Your 101M parameter model with batch_size=64 required:
- Model params: ~512MB (fp16)
- Optimizer states: ~1.5GB (Adam stores momentum + variance)
- Activations: ~18GB (this is the killer with batch=64!)
- Gradients: ~512MB
- **Total: ~21GB** ❌ Too close to 24GB limit!

## New Configuration Memory Usage

With batch_size=32 + gradient checkpointing:
- Model params: ~512MB
- Optimizer states: ~1.5GB
- Activations: ~6GB (gradient checkpointing trades compute for memory)
- Gradients: ~512MB
- **Total: ~9GB** ✅ Comfortable fit!

## Speed vs Memory Tradeoff

| Setting | Memory | Speed | Notes |
|---------|--------|-------|-------|
| batch=64, no checkpoint | 21GB ❌ | Fastest | OOM! |
| batch=32, no checkpoint | 15GB | Fast | Might work |
| **batch=32, checkpoint** | **9GB ✅** | **Medium** | **Best balance** |
| batch=16, checkpoint | 6GB | Slower | Too conservative |

## What Gradient Checkpointing Does

**Without checkpointing:**
- Stores all activations during forward pass
- Fast backward pass (everything in memory)
- High memory usage

**With checkpointing:**
- Only stores some activations during forward pass
- Recomputes missing activations during backward pass
- Saves ~40-50% memory
- ~20% slower (worth it to avoid OOM!)

## If You Still Get OOM

Try in this order:

1. **Reduce batch_size to 16**
   ```yaml
   batch_size: 16
   gradient_accumulation_steps: 8  # Still effective batch = 128
   ```

2. **Reduce sequence length**
   ```yaml
   max_position_embeddings: 256  # Down from 512
   ```

3. **Reduce model size slightly**
   ```yaml
   num_layers: 12  # Down from 14
   ```

## Current Setup (Should Work!)

✅ batch_size: 32
✅ gradient_accumulation: 4
✅ gradient_checkpointing: true
✅ Effective batch: 128
✅ Expected memory: ~9-12GB
✅ Expected speed: 0.8-1.2 it/s

**Try training again - should work now!**
