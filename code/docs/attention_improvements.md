# Advanced Attention Mechanisms

This document describes all the attention improvements implemented in `src/model/attention.py`.

## 1. Sliding Window Attention

**Purpose**: Reduces memory and computation from O(n²) to O(n×w) where w is window size.

**Best for**: 
- Sequences where local context is most important
- Models with limited memory
- Real-time applications requiring fast inference

**Usage**:
```python
attn = SlidingWindowAttention(config, window_size=256)
```

## 2. Sparse Attention (BigBird/Longformer Style)

**Purpose**: Enables processing of very long sequences (10k-100k tokens) efficiently.

**Features**:
- Local sliding window attention
- Global tokens that attend to all positions  
- Random attention blocks for information flow

**Best for**:
- Long documents (papers, books, code files)
- Tasks requiring both local and global context

**Usage**:
```python
attn = SparseAttention(
    config,
    window_size=256,
    num_global_tokens=64,
    num_random_blocks=3
)
```

## 3. Streaming Attention with Sinks

**Purpose**: Prevents attention score degradation during continuous generation.

**Features**:
- First few tokens act as "sinks" that accumulate attention
- Learnable importance weights for sink tokens
- Maintains model quality during long generations

**Best for**:
- Chat models and conversational AI
- Continuous text generation
- Streaming applications

**Usage**:
```python
attn = StreamingAttentionWithSinks(config, num_sink_tokens=4)
```

## 4. ALiBi (Attention with Linear Biases)

**Purpose**: Replaces position embeddings with learned biases for better extrapolation.

**Features**:
- No position embeddings needed
- Superior length extrapolation
- Can handle sequences much longer than training length

**Best for**:
- Models that need to generalize to longer sequences
- When training data has varied sequence lengths

**Usage**:
```python
attn = ALiBiAttention(config)
```

## 5. xPos Rotary Embeddings

**Purpose**: Improved RoPE that extrapolates better to longer sequences.

**Features**:
- Combines RoPE with exponential decay
- Stable attention scores at extreme lengths
- Drop-in replacement for standard RoPE

**Usage**:
```python
xpos = xPosRotaryEmbedding(
    dim=64,
    max_position_embeddings=8192,
    scale_base=512
)
```

## 6. Linear Attention

**Purpose**: O(n) complexity attention using kernel trick.

**Features**:
- Scales linearly with sequence length
- Trades some quality for extreme efficiency
- Multiple feature map options (ELU, ReLU)

**Best for**:
- Extremely long sequences (100k+ tokens)
- Real-time applications
- Memory-constrained environments

**Usage**:
```python
attn = LinearAttention(config, feature_map='elu')
```

## 7. Cached Attention

**Purpose**: Reuses attention patterns for similar queries.

**Features**:
- Caches computed attention patterns
- Adds slight noise for diversity
- Tracks cache hit rate

**Best for**:
- Repetitive tasks
- Batch processing of similar inputs
- Speeding up inference

**Usage**:
```python
attn = CachedAttention(config, cache_size=32)
```

## 8. Ring Attention

**Purpose**: Distributes attention computation across multiple devices.

**Features**:
- Splits sequence across devices in ring topology
- Enables processing of million+ token sequences
- (Implementation framework provided)

**Best for**:
- Extremely long sequences beyond single GPU memory
- Multi-GPU setups

## Performance Comparison

| Attention Type | Complexity | Memory | Quality | Best Sequence Length |
|----------------|------------|--------|---------|---------------------|
| Standard MHA   | O(n²)      | O(n²)  | 100%    | < 8k                |
| Sliding Window | O(n×w)     | O(n×w) | 95%     | < 32k               |
| Sparse         | O(n×√n)    | O(n×√n)| 97%     | < 100k              |
| Linear         | O(n)       | O(n)   | 85-90%  | > 100k              |
| ALiBi          | O(n²)      | O(n²)  | 100%    | Any (extrapolates)  |

## Combining Techniques

You can combine multiple techniques for optimal performance:

```python
# Example: Long document processing with efficiency
# 1. Use sparse attention for long-range modeling
# 2. Add ALiBi for position encoding
# 3. Enable Flash Attention for speed
config = MoEConfig(
    attention_variant='sparse',
    sparse_window_size=512,
    sparse_global_tokens=128,
    use_flash_attn=True,
    position_encoding='alibi'
)
```

## Recommendations by Use Case

### Chat/Conversational AI
- **Primary**: StreamingAttentionWithSinks
- **Position**: ALiBi or standard RoPE
- **Optimization**: Flash Attention 2

### Long Document Processing
- **Primary**: SparseAttention
- **Position**: ALiBi for extrapolation
- **Optimization**: Flash Attention 2

### Code Generation
- **Primary**: SlidingWindowAttention (1024 window)
- **Secondary**: Sparse global tokens for imports/definitions
- **Position**: xPos for long files

### Scientific Papers
- **Primary**: SparseAttention with 128 global tokens
- **Position**: ALiBi
- **Cache**: Enable for bibliography processing

### Real-time Applications
- **Primary**: LinearAttention or SlidingWindow
- **Position**: None (for speed)
- **Optimization**: Compile with TorchScript