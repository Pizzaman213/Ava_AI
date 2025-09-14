# Training with Advanced Attention Mechanisms

This guide explains how to use the various attention improvements with the `train.py` script.

## Quick Start

### 1. Standard Attention (Baseline)
```bash
python scripts/train.py --config configs/gpu/medium.yaml
```

### 2. Sliding Window Attention (Memory Efficient)
```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sliding_window \
    --sliding-window-size 256
```

### 3. Sparse Attention (Long Documents)
```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --sparse-global-tokens 128 \
    --sparse-random-blocks 5 \
    --sliding-window-size 512
```

### 4. Streaming Attention (Chat/Conversational)
```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant streaming \
    --num-sink-tokens 8
```

### 5. Linear Attention (Extreme Length)
```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant linear \
    --linear-attention-feature elu
```

### 6. ALiBi Attention (Length Extrapolation)
```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant alibi
```

## Command Line Options

### Attention Type Selection
- `--attention-variant`: Choose attention mechanism
  - `standard`: Standard multi-query attention (default)
  - `sliding_window`: Local window attention
  - `sparse`: BigBird-style sparse attention  
  - `streaming`: Attention with sink tokens
  - `alibi`: Attention with linear biases
  - `cached`: Pattern caching attention
  - `linear`: O(n) linear attention

### Attention-Specific Parameters

#### Sliding Window
- `--sliding-window-size`: Window size (default: 256)

#### Sparse Attention
- `--sparse-global-tokens`: Number of global tokens (default: 64)
- `--sparse-random-blocks`: Random attention blocks (default: 3)
- `--sliding-window-size`: Local window size (default: 256)

#### Streaming Attention
- `--num-sink-tokens`: Number of sink tokens (default: 4)

#### Linear Attention
- `--linear-attention-feature`: Feature map type (`elu`, `relu`, `none`)

#### Position Embeddings
- `--use-xpos`: Use xPos rotary embeddings for better extrapolation

## Pre-configured Examples

### Long Context Processing (Papers, Books)
```bash
python scripts/train.py --config configs/attention_examples/long_context.yaml
```
- Uses sparse attention with 128 global tokens
- 32k+ context length
- Memory-efficient MoE layer
- xPos embeddings for extrapolation

### Chat/Streaming Model
```bash
python scripts/train.py --config configs/attention_examples/chat_streaming.yaml
```
- Streaming attention with 8 sink tokens
- Optimized for interactive response
- Maintains conversation context

### Extreme Length (100k+ tokens)
```bash
python scripts/train.py --config configs/attention_examples/extreme_length.yaml
```
- Linear attention for O(n) complexity
- 128k context length
- Maximum memory optimizations
- NVMe offloading support

### Fast Inference
```bash
python scripts/train.py --config configs/attention_examples/efficient_inference.yaml
```
- Sliding window attention
- Optimized for speed
- Attention pattern caching
- Compiled with torch.compile

## Combining Features

You can combine attention variants with other optimizations:

```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --use-xpos \
    --mixed-precision bf16 \
    --gradient-checkpointing \
    --zero-stage 3 \
    --wandb
```

## Performance Guidelines

### Memory Usage vs Speed

| Attention Type | Memory | Speed | Quality | Best For |
|---------------|--------|--------|---------|----------|
| Standard | High | Fast* | 100% | Short sequences (<8k) |
| Sliding Window | Low | Fast | 95% | Local dependencies |
| Sparse | Medium | Medium | 97% | Long documents |
| Linear | Very Low | Very Fast | 85-90% | Extreme lengths |
| Streaming | Medium | Fast | 98% | Continuous generation |
| ALiBi | High | Fast | 100% | Length generalization |

*With Flash Attention 2

### Recommended Settings by Task

#### Research Papers / Academic Texts
```bash
--attention-variant sparse \
--sparse-global-tokens 128 \
--sparse-random-blocks 5 \
--use-xpos
```

#### Code Generation
```bash
--attention-variant sliding_window \
--sliding-window-size 1024 \
--attention-variant cached \
--attention-cache-size 64
```

#### Chat/Assistant Models
```bash
--attention-variant streaming \
--num-sink-tokens 8 \
--use-xpos
```

#### Book/Novel Processing
```bash
--attention-variant sparse \
--sparse-global-tokens 256 \
--max-position-embeddings 65536 \
--use-xpos
```

## Distributed Training

All attention variants support distributed training:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/gpu/large.yaml \
    --attention-variant sparse \
    --distributed

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=0 --master_addr=$MASTER_ADDR \
    scripts/train.py \
    --config configs/gpu/xlarge.yaml \
    --attention-variant linear \
    --distributed
```

## Troubleshooting

### Out of Memory Errors
1. Switch to sliding window or linear attention
2. Reduce `--batch-size`
3. Enable `--gradient-checkpointing`
4. Increase `--gradient-accumulation-steps`
5. Use higher `--zero-stage` (2 or 3)

### Slow Training
1. Ensure Flash Attention 2 is installed
2. Use `--mixed-precision bf16`
3. Try `--torch-compile-mode max-autotune`
4. Reduce attention complexity (smaller windows)

### Poor Quality with Linear Attention
1. Try different feature maps (`relu` vs `elu`)
2. Increase model size to compensate
3. Consider sparse attention instead

### Length Extrapolation Issues
1. Enable `--use-xpos`
2. Try ALiBi attention variant
3. Increase `--rope-theta` in config

## Monitoring

Use Weights & Biases to track attention-specific metrics:

```bash
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --wandb \
    --wandb-project attention-experiments
```

This will log:
- Attention pattern sparsity
- Cache hit rates (for cached attention)
- Expert utilization with different attention types
- Memory usage per attention variant