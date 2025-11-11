#!/usr/bin/env python3
"""Test CPU offloading without CUDA dependencies."""

import torch
import torch.nn as nn
from code.src.Ava.layers.offloaded_experts import CPUOffloadedExpertGroup

def test_cpu_offloading():
    """Test that CPU offloading works without CUDA."""
    print("Testing CPU offloading without CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create a small expert group
    experts = CPUOffloadedExpertGroup(
        num_experts=4,
        hidden_size=256,
        intermediate_size=512,
        max_active_experts=2,
        use_lora=True,
        lora_rank=8,
        pin_memory=True,  # Should be safely ignored on CPU-only systems
        prefetch_lookahead=2,  # Should not create CUDA streams on CPU
    )

    print(f"✓ Created expert group with {len(experts.experts)} experts")
    print(f"✓ Prefetch streams: {len(experts._prefetch_streams)}")

    # Create test input
    batch_size = 4
    seq_len = 16
    num_tokens = batch_size * seq_len
    hidden_size = 256
    k = 2  # top-k experts

    hidden_states = torch.randn(num_tokens, hidden_size)
    expert_indices = torch.randint(0, 4, (num_tokens, k))
    expert_weights = torch.softmax(torch.randn(num_tokens, k), dim=-1)

    print(f"✓ Created test inputs: {hidden_states.shape}, indices: {expert_indices.shape}")

    # Test forward pass
    print("Running forward pass...")
    output = experts.forward(
        hidden_states=hidden_states,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
    )

    print(f"✓ Forward pass successful! Output shape: {output.shape}")
    print(f"✓ Output device: {output.device}")

    # Verify output shape
    assert output.shape == (num_tokens, k, hidden_size), f"Expected shape {(num_tokens, k, hidden_size)}, got {output.shape}"

    # Test batched forward
    print("\nTesting batched forward...")
    hidden_states_batched = torch.randn(batch_size, seq_len, hidden_size)
    expert_indices_batched = torch.randint(0, 4, (batch_size, seq_len, k))
    expert_weights_batched = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)

    output_batched = experts.forward_batched(
        hidden_states=hidden_states_batched,
        expert_indices=expert_indices_batched,
        expert_weights=expert_weights_batched,
        batch_size=2,
    )

    print(f"✓ Batched forward pass successful! Output shape: {output_batched.shape}")

    # Get memory stats
    stats = experts.get_memory_stats()
    print(f"\n📊 Memory statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ All tests passed! CPU offloading works without CUDA.")
    return True

if __name__ == "__main__":
    try:
        test_cpu_offloading()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
