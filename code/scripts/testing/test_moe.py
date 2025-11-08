#!/usr/bin/env python3
"""
Comprehensive Test Suite for Sparse MoE Implementation

Tests all components of the Sparse Mixture of Experts implementation:
- SparseMoELayer
- MixtralRouter and DeepSeekRouter
- ExpertParallelGroup
- Auxiliary losses
- Integration with model training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
from typing import Dict, Any

from src.Ava.models.moe_layer import SparseMoELayer
from src.Ava.layers.routing import MixtralRouter, DeepSeekRouter
from src.Ava.layers.experts import ExpertParallelGroup, SharedExpertLayer, HighPerformanceExpert
from src.Ava.models.moe_model import OptimizedMoETransformer, OptimizedMoEConfig


def test_high_performance_expert():
    """Test HighPerformanceExpert with different activations."""
    print("Testing HighPerformanceExpert...")

    batch_size, seq_len, hidden_size = 4, 32, 512
    intermediate_size = 2048

    for activation in ['swiglu', 'geglu', 'gelu']:
        expert = HighPerformanceExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation
        )

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = expert(x)

        assert output.shape == (batch_size, seq_len, hidden_size), \
            f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert torch.isfinite(output).all(), "Output contains Inf"

        print(f"  ✓ {activation} activation: output shape {output.shape}")

    print("  ✓ All activation functions working correctly\n")


def test_expert_parallel_group():
    """Test ExpertParallelGroup with grouped GEMM."""
    print("Testing ExpertParallelGroup...")

    num_experts = 8
    num_tokens = 64
    k = 2
    hidden_size = 512
    intermediate_size = 2048

    experts = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu'
    )

    # Create dummy routing
    hidden_states = torch.randn(num_tokens, hidden_size)
    expert_indices = torch.randint(0, num_experts, (num_tokens, k))
    expert_weights = F.softmax(torch.randn(num_tokens, k), dim=-1)

    # Forward pass
    output = experts(hidden_states, expert_indices, expert_weights)

    assert output.shape == (num_tokens, k, hidden_size), \
        f"Expected shape {(num_tokens, k, hidden_size)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    print(f"  ✓ Grouped GEMM output shape: {output.shape}")
    print(f"  ✓ Expert utilization: {expert_indices.unique().numel()}/{num_experts} experts used\n")


def test_mixtral_router():
    """Test MixtralRouter with top-K selection."""
    print("Testing MixtralRouter...")

    hidden_size = 512
    num_experts = 16
    num_selected = 2
    num_tokens = 128

    router = MixtralRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_selected_experts=num_selected,
        router_z_loss_coef=0.001,
        load_balance_loss_coef=0.01
    )

    hidden_states = torch.randn(num_tokens, hidden_size)

    # Training mode
    expert_indices, expert_weights, aux_loss, metrics = router(hidden_states, training=True)

    assert expert_indices.shape == (num_tokens, num_selected)
    assert expert_weights.shape == (num_tokens, num_selected)
    assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-5), \
        "Routing weights should sum to 1"
    assert aux_loss.item() >= 0, "Auxiliary loss should be non-negative"

    print(f"  ✓ Expert indices shape: {expert_indices.shape}")
    print(f"  ✓ Expert weights shape: {expert_weights.shape}")
    print(f"  ✓ Weights sum to 1: {expert_weights.sum(dim=-1).mean().item():.6f}")
    print(f"  ✓ Auxiliary loss: {aux_loss.item():.6f}")
    print(f"  ✓ Balance score: {metrics['balance_score'].item():.4f}")
    print(f"  ✓ Routing entropy: {metrics['routing_entropy'].item():.4f}\n")


def test_deepseek_router():
    """Test DeepSeekRouter with shared experts."""
    print("Testing DeepSeekRouter...")

    hidden_size = 512
    num_experts = 15  # 15 routed + 1 shared = 16 total
    num_selected = 2
    num_shared = 1
    num_tokens = 128

    router = DeepSeekRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_selected_experts=num_selected,
        num_shared_experts=num_shared,
        shared_expert_weight=0.5
    )

    hidden_states = torch.randn(num_tokens, hidden_size)
    expert_indices, expert_weights, aux_loss, metrics = router(hidden_states, training=True)

    expected_k = num_shared + num_selected
    assert expert_indices.shape == (num_tokens, expected_k)
    assert expert_weights.shape == (num_tokens, expected_k)

    # Check shared expert is always included (index 0)
    assert (expert_indices[:, 0] == 0).all(), "First expert should always be shared expert (index 0)"

    print(f"  ✓ Combined indices shape: {expert_indices.shape} (shared + routed)")
    print(f"  ✓ Shared expert weight: {metrics['shared_expert_weight'].item()}")
    print(f"  ✓ Auxiliary loss: {aux_loss.item():.6f}\n")


def test_sparse_moe_layer():
    """Test complete SparseMoELayer."""
    print("Testing SparseMoELayer...")

    batch_size, seq_len = 2, 64
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    num_experts_per_token = 2

    moe_layer = SparseMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        router_type='mixtral',
        use_grouped_gemm=True,
        router_z_loss_coef=0.001,
        load_balance_loss_coef=0.01,
        diversity_loss_coef=0.001
    )

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output, aux_loss, metrics = moe_layer(hidden_states, training=True)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert aux_loss.item() >= 0

    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Total auxiliary loss: {metrics['aux_loss_total'].item():.6f}")
    print(f"  ✓   - Router loss: {metrics['aux_loss_routing'].item():.6f}")
    print(f"  ✓   - Diversity loss: {metrics['aux_loss_diversity'].item():.6f}")
    print(f"  ✓   - Expert dropout loss: {metrics['aux_loss_expert_dropout'].item():.6f}")
    print(f"  ✓ Expert utilization: {metrics['expert_utilization']}")
    print(f"  ✓ Balance score: {metrics['balance_score'].item():.4f}\n")


def test_sparse_moe_backward():
    """Test backward pass and gradient flow."""
    print("Testing SparseMoELayer backward pass...")

    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8

    moe_layer = SparseMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=2,
        router_type='mixtral'
    )

    hidden_states = torch.randn(4, 32, hidden_size, requires_grad=True)

    # Forward + backward
    output, aux_loss, metrics = moe_layer(hidden_states, training=True)
    loss = output.sum() + aux_loss
    loss.backward()

    # Check gradients exist
    assert hidden_states.grad is not None, "Input gradients should exist"
    assert not torch.isnan(hidden_states.grad).any(), "Gradients contain NaN"

    # Check parameter gradients
    for name, param in moe_layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

    print(f"  ✓ Backward pass successful")
    print(f"  ✓ Input gradient shape: {hidden_states.grad.shape}")
    print(f"  ✓ All parameter gradients computed\n")


def test_optimized_moe_transformer():
    """Test full OptimizedMoETransformer model."""
    print("Testing OptimizedMoETransformer...")

    config = OptimizedMoEConfig(
        vocab_size=1000,
        hidden_size=512,
        num_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_token=2,
        router_type='mixtral',
        max_position_embeddings=512
    )

    model = OptimizedMoETransformer(config)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass with loss
    output = model(input_ids, labels=labels, return_dict=True)

    assert 'loss' in output
    assert 'logits' in output
    assert 'aux_info' in output
    assert output['logits'].shape == (batch_size, seq_len, config.vocab_size)

    # Backward pass
    loss = output['loss']
    loss.backward()

    print(f"  ✓ Model forward pass successful")
    print(f"  ✓ Output logits shape: {output['logits'].shape}")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Number of layers with MoE: {len(output['aux_info'])}")
    print(f"  ✓ Backward pass successful\n")


def test_expert_usage_tracking():
    """Test expert usage statistics tracking."""
    print("Testing expert usage tracking...")

    moe_layer = SparseMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts=8,
        num_experts_per_token=2,
        router_type='mixtral'
    )

    # Reset counters
    moe_layer.reset_expert_counts()

    # Run multiple forward passes
    for _ in range(10):
        hidden_states = torch.randn(32, 16, 256)
        _, _, _ = moe_layer(hidden_states, training=True)

    # Get usage stats
    stats = moe_layer.get_expert_usage_stats()

    assert 'expert_usage_total' in stats
    assert 'expert_usage_normalized' in stats
    assert stats['expert_usage_total'].sum() > 0

    print(f"  ✓ Expert usage tracking working")
    print(f"  ✓ Total expert calls: {stats['expert_usage_total'].sum().item():.0f}")
    print(f"  ✓ Usage distribution: {stats['expert_usage_normalized']}\n")


def test_load_balancing_loss():
    """Test load balancing loss computation."""
    print("Testing load balancing loss...")

    # Create unbalanced routing (all tokens to first expert)
    router = MixtralRouter(
        hidden_size=256,
        num_experts=8,
        num_selected_experts=2,
        load_balance_loss_coef=0.01
    )

    # Bias towards first expert
    hidden_states = torch.randn(100, 256)
    hidden_states[:, 0] = 10.0  # Strong signal for first dimension

    _, _, aux_loss, metrics = router(hidden_states, training=True)

    # Loss should be positive (indicating imbalance)
    assert aux_loss.item() > 0, "Load balancing loss should be positive for imbalanced routing"

    print(f"  ✓ Load balancing loss: {aux_loss.item():.6f}")
    print(f"  ✓ Balance score: {metrics['balance_score'].item():.4f} (1.0 = perfect)")
    print(f"  ✓ Expert utilization variance: {metrics['expert_utilization'].std().item():.2f}\n")


def test_different_batch_sizes():
    """Test MoE with different batch sizes."""
    print("Testing different batch sizes...")

    moe_layer = SparseMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts=8,
        num_experts_per_token=2,
        router_type='mixtral'
    )

    batch_configs = [
        (1, 16),   # Small batch
        (8, 64),   # Medium batch
        (32, 128), # Large batch
    ]

    for batch_size, seq_len in batch_configs:
        hidden_states = torch.randn(batch_size, seq_len, 256)
        output, aux_loss, metrics = moe_layer(hidden_states, training=True)

        assert output.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(output).any()

        print(f"  ✓ Batch ({batch_size}, {seq_len}): output shape {output.shape}, loss {aux_loss.item():.6f}")

    print()


def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("SPARSE MoE COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        test_high_performance_expert,
        test_expert_parallel_group,
        test_mixtral_router,
        test_deepseek_router,
        test_sparse_moe_layer,
        test_sparse_moe_backward,
        test_optimized_moe_transformer,
        test_expert_usage_tracking,
        test_load_balancing_loss,
        test_different_batch_sizes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✓ All tests passed! Sparse MoE implementation is working correctly.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
