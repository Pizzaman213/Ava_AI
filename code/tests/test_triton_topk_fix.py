"""
Test for the fixed Triton fused_softmax_topk kernel.

This test verifies that the streaming top-k algorithm correctly handles
cases where num_experts > BLOCK_SIZE_EXPERT (the bug that caused
"CUDA error: illegal memory access" during training).
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/root/Ava_AI/code/src')


def test_fused_softmax_topk():
    """Test fused_softmax_topk against PyTorch reference."""
    from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        print("SKIP: Triton not available")
        return True

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    device = torch.device('cuda')

    test_cases = [
        # (num_tokens, num_experts, top_k, description)
        (16, 8, 2, "Small: E < BLOCK_SIZE_EXPERT"),
        (32, 32, 2, "Medium: E == BLOCK_SIZE_EXPERT (32)"),
        (64, 64, 2, "Edge: E == BLOCK_SIZE_EXPERT (64)"),
        (128, 128, 2, "BUG CASE: E > BLOCK_SIZE_EXPERT (128 > 64)"),
        (256, 256, 2, "Large: E >> BLOCK_SIZE_EXPERT (256 > 64)"),
        (64, 128, 4, "Multi-expert: k=4, E > BLOCK_SIZE_EXPERT"),
        (32, 200, 2, "Real config: 200 experts (like some production models)"),
        (1024, 32, 2, "Many tokens: typical training batch"),
        (2048, 8, 2, "Very many tokens: large batch, few experts"),
    ]

    all_passed = True

    for num_tokens, num_experts, top_k, desc in test_cases:
        print(f"\nTest: {desc}")
        print(f"  num_tokens={num_tokens}, num_experts={num_experts}, top_k={top_k}")

        # Create random logits
        torch.manual_seed(42)
        logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

        # PyTorch reference
        probs_ref = F.softmax(logits, dim=-1)
        topk_probs_ref, topk_indices_ref = torch.topk(probs_ref, top_k, dim=-1)

        # Triton implementation
        try:
            topk_probs_triton, topk_indices_triton = fused_softmax_topk(
                logits, top_k, use_triton=True
            )
        except Exception as e:
            print(f"  FAIL: Triton kernel raised exception: {e}")
            all_passed = False
            continue

        # Check shapes
        if topk_probs_triton.shape != topk_probs_ref.shape:
            print(f"  FAIL: Shape mismatch - Triton {topk_probs_triton.shape} vs Ref {topk_probs_ref.shape}")
            all_passed = False
            continue

        # Check indices are valid (within bounds)
        max_idx = topk_indices_triton.max().item()
        min_idx = topk_indices_triton.min().item()
        if max_idx >= num_experts or min_idx < 0:
            print(f"  FAIL: Invalid indices - min={min_idx}, max={max_idx}, num_experts={num_experts}")
            all_passed = False
            continue

        # Check that Triton found the SAME top-k experts as PyTorch
        # (The ordering might differ if there are ties, so we check sets)
        indices_match = True
        for t in range(min(10, num_tokens)):  # Check first 10 tokens
            triton_set = set(topk_indices_triton[t].tolist())
            ref_set = set(topk_indices_ref[t].tolist())
            if triton_set != ref_set:
                # Check if it's a near-tie case (probs are very close)
                triton_probs = [probs_ref[t, i].item() for i in triton_set]
                ref_probs = [probs_ref[t, i].item() for i in ref_set]
                if not all(abs(tp - rp) < 0.01 for tp, rp in zip(sorted(triton_probs, reverse=True), sorted(ref_probs, reverse=True))):
                    indices_match = False
                    print(f"  Token {t}: Triton chose {triton_set} (probs {triton_probs})")
                    print(f"            Ref chose {ref_set} (probs {ref_probs})")

        if not indices_match:
            print(f"  FAIL: Different experts selected")
            all_passed = False
            continue

        # Check probabilities are close (allowing for numerical precision)
        prob_diff = (topk_probs_triton - topk_probs_ref).abs().max().item()
        if prob_diff > 0.01:  # 1% tolerance
            print(f"  FAIL: Probabilities differ by {prob_diff:.6f}")
            all_passed = False
            continue

        print(f"  PASS: Indices valid, probs match (max diff: {prob_diff:.6f})")

    return all_passed


def test_streaming_correctness():
    """Specifically test the streaming algorithm produces correct global indices."""
    from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return True

    device = torch.device('cuda')

    print("\n=== Streaming Correctness Test ===")
    print("Testing that indices are GLOBAL (not block-local)")

    # Create logits where the top experts are in later blocks
    # BLOCK_SIZE_EXPERT = 64, so experts 0-63 are block 0, 64-127 are block 1, etc.
    num_tokens = 10
    num_experts = 128  # 2 blocks
    top_k = 2

    # Make expert 100 and 120 the clear winners (both in block 1)
    logits = torch.zeros(num_tokens, num_experts, device=device, dtype=torch.float32)
    logits[:, 100] = 10.0  # Expert 100 should win
    logits[:, 120] = 9.0   # Expert 120 should be second
    logits[:, 0] = 1.0     # Expert 0 (block 0) should NOT be selected

    # Add small noise to other experts
    logits += torch.randn_like(logits) * 0.01

    topk_probs, topk_indices = fused_softmax_topk(logits, top_k, use_triton=True)

    print(f"  Top-k indices (first 3 tokens): {topk_indices[:3].tolist()}")

    # Verify all tokens selected experts 100 and 120
    expected_indices = {100, 120}
    all_correct = True
    for t in range(num_tokens):
        actual = set(topk_indices[t].tolist())
        if actual != expected_indices:
            print(f"  Token {t}: Expected {expected_indices}, got {actual}")
            all_correct = False

    if all_correct:
        print("  PASS: Global indices correctly returned from block 1")
    else:
        print("  FAIL: Wrong indices returned")

    return all_correct


def test_no_cuda_errors():
    """Run many iterations to check for intermittent CUDA errors."""
    from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return True

    device = torch.device('cuda')

    print("\n=== Stress Test (checking for CUDA errors) ===")

    num_iterations = 100
    errors = 0

    for i in range(num_iterations):
        # Random problem sizes
        num_tokens = torch.randint(16, 512, (1,)).item()
        num_experts = torch.randint(8, 256, (1,)).item()
        top_k = min(torch.randint(1, 9, (1,)).item(), num_experts)

        logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

        try:
            topk_probs, topk_indices = fused_softmax_topk(logits, top_k, use_triton=True)

            # Force sync to catch async errors
            torch.cuda.synchronize()

            # Basic sanity check
            assert topk_indices.max() < num_experts, f"Index out of bounds: {topk_indices.max()} >= {num_experts}"
            assert topk_indices.min() >= 0, f"Negative index: {topk_indices.min()}"

        except Exception as e:
            print(f"  Iteration {i}: ERROR with num_tokens={num_tokens}, num_experts={num_experts}, top_k={top_k}")
            print(f"    {type(e).__name__}: {e}")
            errors += 1

    if errors == 0:
        print(f"  PASS: {num_iterations} iterations completed without errors")
    else:
        print(f"  FAIL: {errors}/{num_iterations} iterations had errors")

    return errors == 0


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Triton fused_softmax_topk kernel fix")
    print("=" * 60)

    results = []

    results.append(("Basic functionality", test_fused_softmax_topk()))
    results.append(("Streaming correctness", test_streaming_correctness()))
    results.append(("Stress test", test_no_cuda_errors()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
