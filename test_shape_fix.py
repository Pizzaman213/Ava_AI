#!/usr/bin/env python3
"""Test the shape handling fix in offloaded_experts.py"""

import torch
import sys
sys.path.insert(0, '/project/code/src')

from Ava.layers.offloaded_experts import CPUOffloadedExpertGroup

print("Testing shape handling in CPUOffloadedExpertGroup...")

# Create a simple expert group
num_experts = 4
hidden_size = 128
expert_group = CPUOffloadedExpertGroup(
    num_experts=num_experts,
    hidden_size=hidden_size,
    intermediate_size=256
)
# Enable batched processing
expert_group.use_batched_processing = True

print(f"\n✓ Created expert group with {num_experts} experts, hidden_size={hidden_size}")

# Test 2D input (the case that was failing)
print("\n1. Testing 2D input [num_tokens, k]...")
num_tokens = 16
top_k = 2
hidden_states_2d = torch.randn(num_tokens, hidden_size)
expert_indices_2d = torch.randint(0, num_experts, (num_tokens, top_k))
expert_weights_2d = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

print(f"   Input shapes: hidden_states={hidden_states_2d.shape}, expert_indices={expert_indices_2d.shape}")

try:
    output_2d = expert_group.forward_batched(
        hidden_states_2d,
        expert_indices_2d,
        expert_weights_2d
    )
    print(f"   ✓ Output shape: {output_2d.shape}")
    expected_shape = (num_tokens, top_k, hidden_size)
    assert output_2d.shape == expected_shape, f"Expected {expected_shape}, got {output_2d.shape}"
    print(f"   ✓ Shape matches expected: {expected_shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3D input (should still work)
print("\n2. Testing 3D input [batch, seq_len, k]...")
batch = 4
seq_len = 8
hidden_states_3d = torch.randn(batch, seq_len, hidden_size)
expert_indices_3d = torch.randint(0, num_experts, (batch, seq_len, top_k))
expert_weights_3d = torch.softmax(torch.randn(batch, seq_len, top_k), dim=-1)

print(f"   Input shapes: hidden_states={hidden_states_3d.shape}, expert_indices={expert_indices_3d.shape}")

try:
    output_3d = expert_group.forward_batched(
        hidden_states_3d,
        expert_indices_3d,
        expert_weights_3d
    )
    print(f"   ✓ Output shape: {output_3d.shape}")
    expected_shape = (batch, seq_len, top_k, hidden_size)
    assert output_3d.shape == expected_shape, f"Expected {expected_shape}, got {output_3d.shape}"
    print(f"   ✓ Shape matches expected: {expected_shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All shape handling tests passed!")
print("The fix correctly handles both 2D and 3D input shapes.")
