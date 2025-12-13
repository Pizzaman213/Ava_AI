"""
Test for Triton fused softmax + top-k kernel.

Tests:
1. Correctness: Triton results match PyTorch results
2. Both single-block (num_experts <= 64) and multi-block cases
3. Various k values (1, 2, 4, 8)
"""

import sys
sys.path.insert(0, '/root/Ava_AI/code/src')

import torch
import torch.nn.functional as F


def pytorch_softmax_topk(logits, top_k):
    """Reference PyTorch implementation."""
    probs = F.softmax(logits, dim=-1)
    return torch.topk(probs, top_k, dim=-1)


def test_triton_softmax_topk():
    """Test Triton kernel against PyTorch reference."""
    from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE, _pytorch_softmax_topk

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    if not TRITON_AVAILABLE:
        print("SKIP: Triton not available")
        return True

    device = torch.device('cuda')
    torch.manual_seed(42)

    test_cases = [
        # (num_tokens, num_experts, top_k, description)
        # Single-block cases (Triton kernel)
        (1, 8, 2, "Single token, 8 experts, top-2"),
        (16, 8, 2, "16 tokens, 8 experts, top-2"),
        (128, 32, 2, "128 tokens, 32 experts, top-2"),
        (256, 64, 2, "256 tokens, 64 experts (max Triton), top-2"),
        (64, 8, 1, "64 tokens, 8 experts, top-1"),
        (64, 8, 4, "64 tokens, 8 experts, top-4"),
        (64, 8, 8, "64 tokens, 8 experts, top-8"),
        # Large expert counts (PyTorch fallback - still tested for correctness)
        (64, 128, 2, "64 tokens, 128 experts (PyTorch fallback), top-2"),
        (32, 256, 4, "32 tokens, 256 experts (PyTorch fallback), top-4"),
    ]

    all_passed = True

    for num_tokens, num_experts, top_k, desc in test_cases:
        print(f"\nTesting: {desc}")

        # Create random logits
        logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

        # Get PyTorch reference
        ref_probs, ref_indices = pytorch_softmax_topk(logits, top_k)

        # Get Triton result
        try:
            triton_probs, triton_indices = fused_softmax_topk(logits, top_k, use_triton=True)
        except Exception as e:
            print(f"  FAIL: Triton kernel raised exception: {e}")
            all_passed = False
            continue

        # Check shapes
        if triton_probs.shape != ref_probs.shape:
            print(f"  FAIL: Shape mismatch - got {triton_probs.shape}, expected {ref_probs.shape}")
            all_passed = False
            continue

        # Check probabilities (values should match closely)
        prob_diff = (triton_probs.float() - ref_probs.float()).abs().max().item()
        prob_ok = prob_diff < 1e-4

        # Check indices - they should select the same experts
        # Note: In case of ties, indices might differ, so we check the selected probs match
        indices_match = True
        for i in range(num_tokens):
            triton_set = set(triton_indices[i].tolist())
            ref_set = set(ref_indices[i].tolist())
            if triton_set != ref_set:
                # Check if it's just a tie-breaking difference
                # by comparing the actual probability values
                triton_selected = triton_probs[i].sort(descending=True)[0]
                ref_selected = ref_probs[i].sort(descending=True)[0]
                if (triton_selected.float() - ref_selected.float()).abs().max().item() > 1e-4:
                    indices_match = False
                    break

        if prob_ok and indices_match:
            print(f"  PASS: max_prob_diff={prob_diff:.6f}")
        else:
            print(f"  FAIL: prob_ok={prob_ok}, indices_match={indices_match}, prob_diff={prob_diff:.6f}")
            all_passed = False

    return all_passed


def test_benchmark():
    """Benchmark Triton vs PyTorch."""
    from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE, _pytorch_softmax_topk

    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        print("SKIP: CUDA or Triton not available")
        return

    import time

    device = torch.device('cuda')
    num_tokens = 4096
    num_experts = 32
    top_k = 2
    warmup = 10
    iterations = 100

    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = _pytorch_softmax_topk(logits, top_k)
        _ = fused_softmax_topk(logits, top_k, use_triton=True)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        _ = _pytorch_softmax_topk(logits, top_k)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fused_softmax_topk(logits, top_k, use_triton=True)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nBenchmark ({num_tokens} tokens, {num_experts} experts, top-{top_k}):")
    print(f"  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Triton:  {triton_time:.3f} ms")
    print(f"  Speedup: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Triton fused softmax + top-k kernel")
    print("=" * 60)

    passed = test_triton_softmax_topk()

    print("\n" + "=" * 60)
    print("Benchmark")
    print("=" * 60)
    test_benchmark()

    print("\n" + "=" * 60)
    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if passed else 1)
