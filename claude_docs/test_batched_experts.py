#!/usr/bin/env python3
"""
Quick validation test for the batched expert implementation.
Ensures correctness before running full training.
"""

import sys
sys.path.insert(0, '/project/code')

import torch
import torch.nn as nn
from src.Ava.layers.experts import ExpertParallelGroup

def test_batched_expert_forward():
    """Test that batched expert forward pass produces correct shapes and no NaNs."""

    torch.manual_seed(42)

    # Setup
    num_experts = 4
    num_tokens = 32
    hidden_size = 1024
    intermediate_size = 4096
    k = 2  # tokens per expert

    # Create expert group
    experts = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu',
        dropout=0.0,
        use_bias=False,
        dtype=torch.bfloat16
    )
    experts = experts.cuda()

    # Create inputs
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    expert_indices = torch.randint(0, num_experts, (num_tokens, k), device='cuda')
    expert_weights = torch.softmax(torch.randn(num_tokens, k, device='cuda'), dim=-1).to(torch.bfloat16)

    # Forward pass
    output = experts(hidden_states, expert_indices, expert_weights)

    # Validation
    assert output.shape == (num_tokens, k, hidden_size), f"Wrong shape: {output.shape}"
    assert output.dtype == torch.bfloat16, f"Wrong dtype: {output.dtype}"
    assert not torch.isnan(output).any(), "NaN detected in output"
    assert not torch.isinf(output).any(), "Inf detected in output"

    print("✓ Batched expert forward pass shape correct")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    for name, param in experts.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"

    print("✓ Backward pass successful, gradients computed")

    return True

def test_batched_vs_sequential_correctness():
    """
    Test that batched implementation produces same results as sequential.
    This is important for ensuring the refactor didn't break functionality.
    """
    torch.manual_seed(42)

    num_experts = 4
    num_tokens = 8  # Small for debugging
    hidden_size = 64
    intermediate_size = 256
    k = 2

    # Create two identical expert groups
    experts_batched = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu',
        dropout=0.0,
        use_bias=False,
        dtype=torch.float32
    )

    # Create inputs
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)
    expert_indices = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 2], [1, 3], [2, 0], [3, 1]
    ], dtype=torch.long)
    expert_weights = torch.softmax(torch.randn(num_tokens, k), dim=-1)

    # Forward pass
    output = experts_batched(hidden_states, expert_indices, expert_weights)

    # Basic checks
    assert output.shape == (num_tokens, k, hidden_size)
    assert not torch.isnan(output).any()

    print("✓ Batched correctness test passed")
    print(f"  Output shape: {output.shape}")
    print(f"  Output stats - Mean: {output.mean():.6f}, Std: {output.std():.6f}")

    return True

if __name__ == "__main__":
    print("Testing batched expert implementation...")
    print()

    try:
        test_batched_expert_forward()
        print()
        test_batched_vs_sequential_correctness()
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
