#!/usr/bin/env python3
"""Test per-expert processing approach for expert layer."""

import torch
import sys
import time

# Add src to path
sys.path.insert(0, '/project/code/src')

from Ava.layers.experts import ExpertParallelGroup

def test_per_expert_processing():
    """Test the per-expert processing implementation."""
    print("Testing per-expert processing approach...")
    print("=" * 60)

    # Realistic configuration
    batch_size = 32
    seq_len = 256
    hidden_size = 1536
    intermediate_size = 6144
    num_experts = 8
    k = 2  # experts per token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Intermediate size: {intermediate_size}")
    print(f"Number of experts: {num_experts}")
    print(f"Experts per token (k): {k}")
    print()

    # Create expert layer with SwiGLU activation
    expert_layer = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu',
        dropout=0.0
    ).to(device)

    # Create test inputs
    num_tokens = batch_size * seq_len
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    expert_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
    expert_weights = torch.rand(num_tokens, k, device=device)

    print(f"Input tensor size: {hidden_states.numel() * 4 / 1e9:.2f} GB")
    print()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = expert_layer(hidden_states, expert_indices, expert_weights)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print("Warmup complete")
    print()

    # Time the forward pass
    print("Running timed forward pass...")
    start_time = time.time()

    with torch.no_grad():
        output = expert_layer(hidden_states, expert_indices, expert_weights)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time

    print(f"✓ Forward pass completed in {elapsed:.3f} seconds")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected shape: ({num_tokens}, {k}, {hidden_size})")

    # Verify output
    assert output.shape == (num_tokens, k, hidden_size), f"Wrong shape: {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"✓ Peak GPU memory: {memory_allocated:.2f} GB")

    print()
    print("=" * 60)
    print("✓ Per-expert processing test PASSED!")
    print(f"✓ Speed: {elapsed:.3f}s per forward pass")
    print(f"✓ This is {'FAST' if elapsed < 0.5 else 'SLOW'} for training")

    return True

if __name__ == '__main__':
    try:
        test_per_expert_processing()
    except Exception as e:
        print(f"❌ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
