#!/usr/bin/env python3
"""
Test script to verify forward pass optimizations don't break functionality.

This script performs basic smoke tests to ensure the optimized forward pass
produces valid outputs and gradients.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))

def test_grouped_gemm():
    """Test optimized grouped GEMM expert computation."""
    print("Testing Grouped GEMM Expert Computation...")

    from Ava.layers.experts import ExpertParallelGroup

    # Create expert group
    experts = ExpertParallelGroup(
        num_experts=8,
        hidden_size=512,
        intermediate_size=2048,
        activation='swiglu',
        dropout=0.0,
        use_bias=False,
        dtype=torch.float32
    )

    # Test input
    batch_size = 4
    seq_len = 32
    num_tokens = batch_size * seq_len
    k = 2  # experts per token

    hidden_states = torch.randn(num_tokens, 512)
    expert_indices = torch.randint(0, 8, (num_tokens, k))
    expert_weights = torch.softmax(torch.randn(num_tokens, k), dim=-1)

    # Forward pass
    try:
        output = experts(hidden_states, expert_indices, expert_weights)

        # Verify output shape
        assert output.shape == (num_tokens, k, 512), f"Wrong shape: {output.shape}"

        # Verify no NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

        # Verify backward pass
        loss = output.sum()
        loss.backward()

        assert experts.down_weights.grad is not None, "Gradients not computed"
        assert torch.isfinite(experts.down_weights.grad).all(), "Gradients contain NaN or Inf"

        print("✓ Grouped GEMM test passed")
        return True
    except Exception as e:
        print(f"✗ Grouped GEMM test failed: {e}")
        return False


def test_rope_optimization():
    """Test optimized RoPE implementation."""
    print("Testing Optimized RoPE...")

    from Ava.models.moe_model import apply_rotary_pos_emb, RoPEPositionalEmbedding

    # Create RoPE
    head_dim = 64
    rope = RoPEPositionalEmbedding(head_dim)

    # Test input
    batch_size = 2
    num_heads = 8
    seq_len = 128

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Get cos/sin
    cos, sin = rope(seq_len, q.device)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Apply RoPE
    try:
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Verify output shape
        assert q_embed.shape == q.shape, f"Wrong Q shape: {q_embed.shape}"
        assert k_embed.shape == k.shape, f"Wrong K shape: {k_embed.shape}"

        # Verify no NaN or Inf
        assert torch.isfinite(q_embed).all(), "Q output contains NaN or Inf"
        assert torch.isfinite(k_embed).all(), "K output contains NaN or Inf"

        print("✓ RoPE optimization test passed")
        return True
    except Exception as e:
        print(f"✗ RoPE optimization test failed: {e}")
        return False


def test_bincount_replacement():
    """Test F.one_hot → bincount replacement."""
    print("Testing bincount replacement...")

    import torch.nn.functional as F

    num_experts = 16
    num_tokens = 128
    k = 2

    expert_indices = torch.randint(0, num_experts, (num_tokens, k))

    # Old approach
    expert_mask = F.one_hot(expert_indices, num_classes=num_experts).float()
    tokens_per_expert_old = expert_mask.sum(dim=(0, 1))

    # New approach (optimized)
    tokens_per_expert_new = torch.bincount(
        expert_indices.flatten(),
        minlength=num_experts
    ).float()

    # Verify equivalence
    try:
        assert torch.allclose(tokens_per_expert_old, tokens_per_expert_new, rtol=1e-5), \
            "Bincount result differs from one_hot"

        print("✓ Bincount replacement test passed")
        return True
    except Exception as e:
        print(f"✗ Bincount replacement test failed: {e}")
        return False


def test_moe_layer_forward():
    """Test complete MoE layer forward pass."""
    print("Testing MoE Layer Forward Pass...")

    from Ava.models.moe_layer import SparseMoELayer

    # Create MoE layer
    moe = SparseMoELayer(
        hidden_size=512,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_token=2,
        router_type='mixtral',
        use_grouped_gemm=True,
        use_triton_kernels=False,  # Disable Triton for CPU testing
        dtype=torch.float32
    )

    # Test input
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 512)

    # Forward pass
    try:
        output, aux_loss, metrics = moe(hidden_states, training=True)

        # Verify output shape
        assert output.shape == hidden_states.shape, f"Wrong shape: {output.shape}"

        # Verify no NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        assert torch.isfinite(aux_loss).all(), "Aux loss contains NaN or Inf"

        # Verify backward pass
        total_loss = output.sum() + aux_loss
        total_loss.backward()

        assert moe.router.gate.weight.grad is not None, "Router gradients not computed"
        assert torch.isfinite(moe.router.gate.weight.grad).all(), "Router gradients contain NaN or Inf"

        print("✓ MoE Layer forward pass test passed")
        return True
    except Exception as e:
        print(f"✗ MoE Layer forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_mask_caching():
    """Test attention mask caching."""
    print("Testing Attention Mask Caching...")

    from Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig

    # Create model
    config = EnhancedMoEConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        num_experts=4,
        num_experts_per_token=2
    )
    model = EnhancedMoEModel(config)

    # First forward pass - should create cache
    input_ids = torch.randint(0, 1000, (2, 64))
    _ = model(input_ids)

    cache_size_1 = len(model._causal_mask_cache)

    # Second forward pass with same seq_len - should use cache
    input_ids = torch.randint(0, 1000, (2, 64))
    _ = model(input_ids)

    cache_size_2 = len(model._causal_mask_cache)

    try:
        assert cache_size_1 == 1, "Cache should have 1 entry after first pass"
        assert cache_size_2 == 1, "Cache should still have 1 entry (reused)"

        print("✓ Attention mask caching test passed")
        return True
    except Exception as e:
        print(f"✗ Attention mask caching test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Forward Pass Optimization Tests")
    print("=" * 60)

    tests = [
        test_bincount_replacement,
        test_rope_optimization,
        test_grouped_gemm,
        test_attention_mask_caching,
        test_moe_layer_forward,
    ]

    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)

    print()
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed! Optimizations are working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
