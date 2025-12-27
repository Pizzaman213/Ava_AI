"""
Test suite for router bug fixes.

Tests:
1. DeepSeek router returns only routed expert indices (not combined with shared)
2. NaN/Inf handling in router logits
3. Weight normalization edge cases
4. Shared expert is applied separately in DeepSeek routing
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ava.nn.routing import MixtralRouter, DeepSeekRouter
from ava.models.moe_layer import SparseMoELayer


def test_deepseek_router_indices():
    """
    Test that DeepSeekRouter returns only routed expert indices,
    not combined with shared expert indices.
    """
    print("=" * 60)
    print("Test 1: DeepSeek Router Returns Only Routed Indices")
    print("=" * 60)

    hidden_size = 128
    num_routed_experts = 7  # DeepSeek: 8 total - 1 shared = 7 routed
    num_selected = 2
    num_shared = 1

    router = DeepSeekRouter(
        hidden_size=hidden_size,
        num_experts=num_routed_experts,
        num_selected_experts=num_selected,
        num_shared_experts=num_shared,
    )

    # Test input
    x = torch.randn(32, hidden_size)

    # Forward pass
    indices, weights, aux_loss, metrics = router(x, training=True)

    # Verify shapes
    assert indices.shape == (32, num_selected), \
        f"Expected indices shape (32, {num_selected}), got {indices.shape}"
    assert weights.shape == (32, num_selected), \
        f"Expected weights shape (32, {num_selected}), got {weights.shape}"

    # Verify indices are in valid range [0, num_routed_experts-1]
    assert indices.min() >= 0, f"Indices contain negative values: {indices.min()}"
    assert indices.max() < num_routed_experts, \
        f"Indices exceed routed expert count: max={indices.max()}, num_experts={num_routed_experts}"

    # Verify weights sum to 1 per token
    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"

    print(f"✓ Indices shape: {indices.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Index range: [{indices.min()}, {indices.max()}]")
    print(f"✓ Weight sum per token: ~1.0")
    print(f"✓ Shared expert weight in metrics: {metrics.get('shared_expert_weight', 'N/A')}")
    print()


def test_nan_inf_handling():
    """
    Test that routers handle NaN and Inf in logits gracefully.
    """
    print("=" * 60)
    print("Test 2: NaN/Inf Handling in Router Logits")
    print("=" * 60)

    hidden_size = 128
    num_experts = 8
    num_selected = 2

    # Test Mixtral router
    router = MixtralRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_selected_experts=num_selected,
    )

    # Inject NaN into router weights to simulate gradient issues
    with torch.no_grad():
        router.gate.weight[0, :] = float('nan')
        router.gate.weight[1, :] = float('inf')

    # Test input
    x = torch.randn(16, hidden_size)

    # Forward pass should handle NaN/Inf gracefully with warning
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        indices, weights, aux_loss, metrics = router(x, training=True)

        # Check that warning was issued
        assert len(w) > 0, "Expected warning for NaN/Inf in logits"
        assert "NaN or Inf" in str(w[0].message), f"Unexpected warning: {w[0].message}"

    # Verify output is still valid
    assert not torch.isnan(indices).any(), "Indices contain NaN"
    assert not torch.isnan(weights).any(), "Weights contain NaN"
    assert not torch.isinf(weights).any(), "Weights contain Inf"
    assert indices.shape == (16, num_selected)
    assert weights.shape == (16, num_selected)

    print(f"✓ NaN/Inf warning issued correctly")
    print(f"✓ Indices valid: shape={indices.shape}, no NaN/Inf")
    print(f"✓ Weights valid: shape={weights.shape}, no NaN/Inf")
    print()


def test_weight_normalization_edge_cases():
    """
    Test weight normalization handles edge cases (zero sum, very small values).
    """
    print("=" * 60)
    print("Test 3: Weight Normalization Edge Cases")
    print("=" * 60)

    hidden_size = 128
    num_experts = 8
    num_selected = 2

    router = DeepSeekRouter(
        hidden_size=hidden_size,
        num_experts=num_experts - 1,  # 7 routed
        num_selected_experts=num_selected,
        num_shared_experts=1,
    )

    # Create input that might lead to very small weights
    x = torch.randn(16, hidden_size) * 0.001  # Very small values

    # Forward pass
    indices, weights, aux_loss, metrics = router(x, training=True)

    # Verify weights are normalized properly
    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"

    # Verify no NaN or Inf in weights
    assert not torch.isnan(weights).any(), "Weights contain NaN"
    assert not torch.isinf(weights).any(), "Weights contain Inf"

    print(f"✓ Weight normalization handles small inputs")
    print(f"✓ Weight sums: {weight_sums.mean():.6f} ± {weight_sums.std():.6f}")
    print()


def test_deepseek_moe_layer():
    """
    Test full DeepSeek MoE layer with shared expert applied separately.
    """
    print("=" * 60)
    print("Test 4: DeepSeek MoE Layer with Shared Expert")
    print("=" * 60)

    hidden_size = 128
    intermediate_size = 512
    num_experts = 8
    num_experts_per_token = 2

    moe_layer = SparseMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        router_type='deepseek',
        use_shared_expert=False,  # Should be handled automatically for deepseek
    )

    # Test input
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output, aux_loss, metrics = moe_layer(x, training=True)

    # Verify output shape
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

    # Verify no NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Verify aux loss is valid
    assert not torch.isnan(aux_loss), "Aux loss is NaN"
    assert aux_loss >= 0, f"Aux loss should be non-negative: {aux_loss}"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output valid (no NaN/Inf)")
    print(f"✓ Aux loss: {aux_loss:.6f}")
    print(f"✓ Metrics: {list(metrics.keys())}")
    print()


def test_mixtral_router():
    """
    Test Mixtral router normal operation (separate test from NaN/Inf test).
    """
    print("=" * 60)
    print("Test 5: Mixtral Router Normal Operation")
    print("=" * 60)

    hidden_size = 128
    num_experts = 8
    num_selected = 2

    # Create FRESH router instance (not reusing one with injected NaN)
    router = MixtralRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_selected_experts=num_selected,
    )

    # Normal input
    x = torch.randn(32, hidden_size)
    indices, weights, aux_loss, metrics = router(x, training=True)

    # Debug: Check for NaN/Inf in router
    print(f"  Router gate weight stats: min={router.gate.weight.min():.4f}, max={router.gate.weight.max():.4f}")
    print(f"  Any NaN in weights? {torch.isnan(router.gate.weight).any()}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  First few weight rows: {weights[:3]}")

    # Verify normal operation
    assert indices.shape == (32, num_selected), f"Shape: {indices.shape}"
    assert weights.shape == (32, num_selected), f"Shape: {weights.shape}"
    assert indices.min() >= 0, f"Min index: {indices.min()}"
    assert indices.max() < num_experts, f"Max index: {indices.max()}"

    weight_sums = weights.sum(dim=1)
    print(f"  Weight sums (first 5): {weight_sums[:5]}")
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weight sums: {weight_sums}"

    print(f"✓ Mixtral router works correctly")
    print(f"✓ Index range: [{indices.min()}, {indices.max()}]")
    print(f"✓ Weight sums: ~1.0 (mean={weight_sums.mean():.6f})")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ROUTER BUG FIX TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_deepseek_router_indices()
        test_nan_inf_handling()
        test_weight_normalization_edge_cases()
        test_deepseek_moe_layer()
        test_mixtral_router()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise
