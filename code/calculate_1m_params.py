#!/usr/bin/env python3
"""Calculate model architecture for ~1M parameters."""

def calculate_params(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, num_experts, experts_per_token):
    """Calculate total parameters for given architecture."""

    # Token embeddings
    token_emb = vocab_size * hidden_size

    # LM head (not tied)
    lm_head = vocab_size * hidden_size

    # Per layer calculations
    # Layer norm (2 per layer: pre-attention and pre-FFN)
    layer_norm_params = 2 * hidden_size * 2  # (weight + bias) × 2 norms

    # Attention (Q, K, V, O projections)
    attention_params = 4 * (hidden_size * hidden_size + hidden_size)  # 4 projections with bias

    # MoE router
    router_params = hidden_size * num_experts + num_experts  # Linear projection + bias

    # MoE experts (each expert has 2 linear layers: up and down)
    expert_params = num_experts * (
        (hidden_size * intermediate_size + intermediate_size) +  # Up projection
        (intermediate_size * hidden_size + hidden_size)          # Down projection
    )

    # Total per layer
    per_layer = layer_norm_params + attention_params + router_params + expert_params

    # All layers
    all_layers = per_layer * num_layers

    # Final layer norm
    final_ln = 2 * hidden_size

    # Total
    total = token_emb + lm_head + all_layers + final_ln

    return {
        'token_emb': token_emb,
        'lm_head': lm_head,
        'layers': all_layers,
        'per_layer': per_layer,
        'final_ln': final_ln,
        'total': total
    }

print("="*80)
print("SEARCHING FOR ~1M PARAMETER ARCHITECTURE")
print("="*80)

vocab_size = 500
num_experts = 4
experts_per_token = 2

# Try different configurations
configs = [
    # (hidden, layers, heads, intermediate)
    (128, 4, 4, 256),   # Current scaled up
    (128, 5, 4, 256),   # More layers
    (128, 6, 4, 256),   # Even more layers
    (160, 4, 4, 320),   # Larger hidden
    (192, 3, 4, 384),   # Larger hidden, fewer layers
    (192, 4, 4, 384),   # Larger hidden, more layers
    (256, 2, 4, 512),   # Very large hidden, few layers
    (256, 3, 4, 512),   # Very large hidden
]

results = []
for hidden, layers, heads, intermediate in configs:
    if hidden % heads != 0:
        continue  # Skip invalid configs

    params = calculate_params(vocab_size, hidden, layers, heads, intermediate, num_experts, experts_per_token)
    results.append((hidden, layers, heads, intermediate, params['total']))

    print(f"\nConfig: hidden={hidden}, layers={layers}, heads={heads}, intermediate={intermediate}")
    print(f"  Token embeddings: {params['token_emb']:,}")
    print(f"  LM head: {params['lm_head']:,}")
    print(f"  Per layer: {params['per_layer']:,}")
    print(f"  All layers: {params['layers']:,}")
    print(f"  Total: {params['total']:,} ({params['total']/1000:.1f}K)")

    diff = abs(params['total'] - 1_000_000)
    print(f"  Difference from 1M: {diff:,} ({diff/1000:.1f}K)")

print("\n" + "="*80)
print("CLOSEST TO 1M PARAMETERS:")
print("="*80)

# Sort by closeness to 1M
results.sort(key=lambda x: abs(x[4] - 1_000_000))

for i, (hidden, layers, heads, intermediate, total) in enumerate(results[:3]):
    print(f"\n#{i+1}: hidden={hidden}, layers={layers}, heads={heads}, intermediate={intermediate}")
    print(f"     Total: {total:,} ({total/1000:.1f}K)")
    print(f"     Head dim: {hidden//heads}")
