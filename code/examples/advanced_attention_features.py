#!/usr/bin/env python3
"""
Example showcasing all the advanced attention features implemented
"""

import torch
import torch.nn as nn
from src.model.attention import (
    MultiQueryAttention, 
    SlidingWindowAttention,
    SparseAttention,
    StreamingAttentionWithSinks,
    ALiBiAttention,
    CachedAttention,
    LinearAttention,
    xPosRotaryEmbedding,
    create_attention_layer
)
from src.model.moe_transformer import MoEConfig

def benchmark_attention(attention_layer, seq_length=2048, batch_size=4, hidden_size=768):
    """Benchmark an attention layer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention_layer = attention_layer.to(device)
    
    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=device)
    
    # Warmup
    for _ in range(3):
        _ = attention_layer(hidden_states)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = attention_layer(hidden_states)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / 10
    
    return avg_time * 1000  # Convert to milliseconds

def main():
    # Configuration
    config = MoEConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,  # GQA with 3:1 ratio
        attention_dropout=0.1,
        use_flash_attn=True
    )
    
    print("Advanced Attention Features Demonstration")
    print("=" * 50)
    
    # 1. Standard Multi-Query Attention (baseline)
    print("\n1. Multi-Query Attention (MQA/GQA)")
    mqa = MultiQueryAttention(config)
    print(f"   - Heads: {config.num_attention_heads}")
    print(f"   - KV heads: {config.num_key_value_heads} (compression ratio {config.num_attention_heads // config.num_key_value_heads}:1)")
    print(f"   - Flash Attention: {'Enabled' if config.use_flash_attn else 'Disabled'}")
    
    # 2. Sliding Window Attention
    print("\n2. Sliding Window Attention")
    window_size = 256
    sliding_attn = SlidingWindowAttention(config, window_size=window_size)
    print(f"   - Window size: {window_size}")
    print(f"   - Memory complexity: O(n × {window_size}) vs O(n²)")
    print(f"   - Best for: Long sequences with local dependencies")
    
    # 3. Sparse Attention (BigBird style)
    print("\n3. Sparse Attention (BigBird/Longformer style)")
    sparse_attn = SparseAttention(config, window_size=256, num_global_tokens=64, num_random_blocks=3)
    print(f"   - Local window: 256 tokens")
    print(f"   - Global tokens: 64 (attend to all positions)")
    print(f"   - Random blocks: 3 per query block")
    print(f"   - Best for: Very long documents (10k+ tokens)")
    
    # 4. Streaming Attention with Sinks
    print("\n4. Streaming Attention with Sink Tokens")
    streaming_attn = StreamingAttentionWithSinks(config, num_sink_tokens=4)
    print(f"   - Sink tokens: 4 (always attended)")
    print(f"   - Best for: Continuous generation, chat models")
    print(f"   - Prevents attention degradation in long conversations")
    
    # 5. ALiBi (Attention with Linear Biases)
    print("\n5. ALiBi - Attention with Linear Biases")
    alibi_attn = ALiBiAttention(config)
    print(f"   - No position embeddings needed")
    print(f"   - Better length extrapolation")
    print(f"   - Can handle sequences longer than training")
    
    # 6. Cached Attention
    print("\n6. Cached Attention (Pattern Reuse)")
    cached_attn = CachedAttention(config, cache_size=32)
    print(f"   - Cache size: 32 patterns")
    print(f"   - Best for: Repetitive tasks, similar queries")
    print(f"   - Speeds up inference for repeated patterns")
    
    # 7. Linear Attention
    print("\n7. Linear Attention (O(n) complexity)")
    linear_attn = LinearAttention(config, feature_map='elu')
    print(f"   - Feature map: ELU")
    print(f"   - Complexity: O(n × d²) vs O(n²× d)")
    print(f"   - Best for: Extremely long sequences (100k+)")
    
    # 8. xPos Rotary Embeddings
    print("\n8. xPos - Extrapolatable Position Embeddings")
    xpos = xPosRotaryEmbedding(dim=64, max_position_embeddings=8192, scale_base=512)
    print(f"   - Max trained position: 8192")
    print(f"   - Can extrapolate to: 32k+ tokens")
    print(f"   - Better stability than standard RoPE")
    
    # Performance Comparison
    print("\n\nPerformance Comparison (2048 sequence length)")
    print("-" * 50)
    
    attention_types = {
        'Standard MQA': MultiQueryAttention(config),
        'Sliding Window': SlidingWindowAttention(config, window_size=256),
        'Sparse (BigBird)': SparseAttention(config),
        'Linear Attention': LinearAttention(config),
        'ALiBi': ALiBiAttention(config),
    }
    
    for name, attn_layer in attention_types.items():
        time_ms = benchmark_attention(attn_layer, seq_length=2048)
        print(f"{name:20s}: {time_ms:6.2f} ms/forward")
    
    # Usage Examples
    print("\n\nUsage Examples:")
    print("-" * 50)
    
    print("\n1. For Chat/Conversation Models:")
    print("   config.attention_variant = 'streaming'")
    print("   config.num_sink_tokens = 4")
    print("   attn = create_attention_layer(config)")
    
    print("\n2. For Long Document Processing (papers, books):")
    print("   config.attention_variant = 'sparse'")
    print("   config.sparse_window_size = 512")
    print("   config.sparse_global_tokens = 128")
    print("   attn = create_attention_layer(config)")
    
    print("\n3. For Extreme Length (100k+ tokens):")
    print("   config.attention_variant = 'linear'")
    print("   config.linear_attention_feature_map = 'elu'")
    print("   attn = create_attention_layer(config)")
    
    print("\n4. For Length Extrapolation:")
    print("   # Use ALiBi or xPos")
    print("   config.attention_variant = 'alibi'")
    print("   # Or use xPos with standard attention")
    print("   config.use_xpos = True")
    
    print("\n\nOptimization Tips:")
    print("-" * 50)
    print("1. Enable Flash Attention 2 for 2-4x speedup on standard attention")
    print("2. Use sliding window for sequences where long-range deps aren't critical")
    print("3. Sparse attention can handle 10x longer sequences with minimal quality loss")
    print("4. Linear attention trades some quality for O(n) complexity")
    print("5. Combine techniques: Sliding window + global tokens + caching")

if __name__ == "__main__":
    main()