#!/usr/bin/env python3
"""Find exact 1M parameter architecture."""

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

    return total

print("="*80)
print("FINDING EXACT 1M PARAMETER ARCHITECTURE")
print("="*80)

vocab_size = 500
target = 1_000_000

# Try with fewer experts and layers
configs = [
    # (hidden, layers, heads, intermediate, experts)
    (96, 4, 4, 192, 4),
    (96, 5, 4, 192, 4),
    (112, 3, 4, 224, 4),
    (112, 4, 4, 224, 4),
    (128, 3, 4, 256, 4),
    (128, 2, 4, 256, 4),
    (128, 3, 4, 256, 2),  # Fewer experts
    (128, 4, 4, 256, 2),  # Fewer experts
    (96, 6, 4, 192, 2),
    (112, 5, 4, 224, 2),
]

results = []
for hidden, layers, heads, intermediate, experts in configs:
    if hidden % heads != 0:
        continue  # Skip invalid configs

    total = calculate_params(vocab_size, hidden, layers, heads, intermediate, experts, 2)
    results.append({
        'hidden': hidden,
        'layers': layers,
        'heads': heads,
        'intermediate': intermediate,
        'experts': experts,
        'total': total,
        'diff': abs(total - target)
    })

# Sort by closeness to 1M
results.sort(key=lambda x: x['diff'])

print(f"\nTarget: 1,000,000 parameters")
print(f"\nTop 5 closest configurations:\n")

for i, r in enumerate(results[:5]):
    print(f"#{i+1}: hidden={r['hidden']}, layers={r['layers']}, heads={r['heads']}, "
          f"intermediate={r['intermediate']}, experts={r['experts']}")
    print(f"     Total: {r['total']:,} ({r['total']/1000:.1f}K)")
    print(f"     Head dim: {r['hidden']//r['heads']}")
    print(f"     Difference: {r['diff']:,} ({r['diff']/10000:.1f}%)")
    print()

# Pick the closest
best = results[0]
print("="*80)
print("RECOMMENDED CONFIGURATION:")
print("="*80)
print(f"hidden_size: {best['hidden']}")
print(f"num_layers: {best['layers']}")
print(f"num_attention_heads: {best['heads']}")
print(f"intermediate_size: {best['intermediate']}")
print(f"num_experts: {best['experts']}")
print(f"num_experts_per_token: 2")
print(f"\nTotal parameters: {best['total']:,} ({best['total']/1000:.1f}K)")
print(f"Difference from 1M: {best['diff']:,} ({best['diff']/1000:.1f}K, {best['diff']/10000:.2f}%)")
