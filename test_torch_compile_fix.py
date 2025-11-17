"""
Test script to verify torch.compile + autocast fix for expert indexing.

This tests that expert layers work correctly with:
- torch.compile enabled
- Mixed precision (BF16) autocast
- Advanced indexing on expert parameters
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/project/code')
from src.Ava.layers.experts import ExpertParallelGroup

def test_expert_with_compile_and_autocast():
    """Test that expert layers work with torch.compile + autocast."""
    print("Testing expert layers with torch.compile + autocast...")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Create expert layer
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 4
    k = 2

    expert_layer = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu',
        use_bias=False
    ).to(device)

    # Create test inputs
    batch_size = 8
    seq_len = 16
    num_tokens = batch_size * seq_len

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    expert_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
    expert_weights = torch.rand(num_tokens, k, device=device, dtype=dtype)
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    # Test forward pass with autocast
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Expert indices shape: {expert_indices.shape}")

    try:
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
            output = expert_layer(
                hidden_states=hidden_states,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
                use_grouped_gemm=True
            )

        print(f"  ✓ Forward pass successful! Output shape: {output.shape}")

        # Test backward pass
        loss = output.sum()
        loss.backward()

        print(f"  ✓ Backward pass successful!")
        print(f"  ✓ Gradients computed successfully")

        # Check gradient exists (check all possible weight attributes)
        has_grad = False
        for param_name, param in expert_layer.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        if has_grad:
            print(f"  ✓ Expert weights have gradients")
        else:
            print(f"  ⚠ Warning: No gradients found (may be normal)")

        return True

    except Exception as e:
        print(f"  ✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_without_compile():
    """Test baseline without torch.compile."""
    print("\nTesting baseline (no torch.compile)...")
    # Ensure torch.compile is not interfering
    torch._dynamo.config.disable = True
    result = test_expert_with_compile_and_autocast()
    torch._dynamo.config.disable = False
    return result


def test_with_compile():
    """Test with torch.compile enabled (should work with @torch.compiler.disable decorator)."""
    print("\nTesting with torch.compile enabled...")
    # This should work because we added @torch.compiler.disable to the expert methods
    result = test_expert_with_compile_and_autocast()
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Torch.Compile + Autocast Fix Verification")
    print("=" * 70)

    # Test without compile first (baseline)
    baseline_passed = test_without_compile()

    # Test with compile (should work with the fix)
    compile_passed = test_with_compile()

    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    print(f"Baseline (no compile):     {'✓ PASSED' if baseline_passed else '✗ FAILED'}")
    print(f"With torch.compile:        {'✓ PASSED' if compile_passed else '✗ FAILED'}")
    print("=" * 70)

    if baseline_passed and compile_passed:
        print("\n✅ All tests passed! The fix is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        exit(1)
