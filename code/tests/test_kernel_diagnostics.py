#!/usr/bin/env python3
"""
Kernel Diagnostics Test Script

This script diagnoses kernel activation issues and profiles different dispatch strategies.
Use this to verify that Triton kernels are being activated and to compare performance.

Usage:
    python code/tests/test_kernel_diagnostics.py
    python code/tests/test_kernel_diagnostics.py --profile  # With Nsight profiling
    python code/tests/test_kernel_diagnostics.py --verbose  # Enable kernel logging

Key diagnostics:
1. Triton availability and version
2. Kernel path activation (which code paths are taken)
3. D2D memory copy estimation
4. Sync overhead estimation
5. Performance comparison of different dispatch strategies
"""

import torch
import torch.nn as nn
import time
import argparse
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "code" / "src"))

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_triton_availability():
    """Check if Triton is available and get version info."""
    print_section("TRITON AVAILABILITY CHECK")

    try:
        import triton
        print(f"✓ Triton is available")
        print(f"  Version: {triton.__version__}")

        # Check if we're on CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            print("✗ CUDA is NOT available - Triton kernels will not run")
            return False

        return True

    except ImportError as e:
        print(f"✗ Triton is NOT available: {e}")
        print("  Install with: pip install triton")
        return False


def check_kernel_imports():
    """Check if all kernel modules can be imported."""
    print_section("KERNEL MODULE IMPORT CHECK")

    modules_to_check = [
        ("ava.kernels.moe", ["TRITON_AVAILABLE", "fused_softmax_topk", "fused_gating_topk"]),
        ("ava.kernels.activations", ["TRITON_AVAILABLE", "fused_swiglu", "fused_geglu"]),
        ("ava.kernels.fused_experts", ["TRITON_AVAILABLE", "fused_expert_forward", "get_async_pipeline"]),
        ("ava.nn.routing", ["MixtralRouter", "TRITON_AVAILABLE"]),
        ("ava.nn.experts", ["ExpertParallelGroup", "FUSED_EXPERT_AVAILABLE"]),
    ]

    all_ok = True
    for module_name, symbols in modules_to_check:
        try:
            module = __import__(module_name, fromlist=symbols)
            status = "✓"

            # Check TRITON_AVAILABLE flag
            if hasattr(module, "TRITON_AVAILABLE"):
                triton_status = getattr(module, "TRITON_AVAILABLE")
                status += f" [TRITON_AVAILABLE={triton_status}]"

            if hasattr(module, "FUSED_EXPERT_AVAILABLE"):
                fused_status = getattr(module, "FUSED_EXPERT_AVAILABLE")
                status += f" [FUSED_EXPERT_AVAILABLE={fused_status}]"

            print(f"{status} {module_name}")

        except Exception as e:
            print(f"✗ {module_name}: {e}")
            all_ok = False

    return all_ok


def test_softmax_topk_kernels(verbose: bool = False):
    """Test softmax_topk kernel activation at different batch sizes."""
    print_section("SOFTMAX_TOPK KERNEL ACTIVATION TEST")

    try:
        from ava.kernels.moe import fused_softmax_topk, TRITON_AVAILABLE
        from ava.kernels.fused_experts import enable_kernel_logging, disable_kernel_logging, get_kernel_log_summary
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return

    if not TRITON_AVAILABLE:
        print("✗ Triton not available, skipping kernel tests")
        return

    if verbose:
        enable_kernel_logging(log_every_n=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("✗ CUDA not available, skipping kernel tests")
        return

    # Test different batch sizes to see threshold effects
    test_cases = [
        (64, 32, 2),      # Small batch - should use PyTorch
        (256, 32, 2),     # At threshold - should use Triton
        (512, 32, 2),     # Medium batch - should use Triton
        (1024, 32, 2),    # Medium batch
        (4096, 32, 2),    # Large batch
        (8192, 64, 2),    # Large with more experts
        (16384, 32, 2),   # Very large
        (256, 128, 2),    # At expert threshold
        (256, 256, 2),    # Over expert threshold - should use PyTorch
    ]

    print(f"{'Tokens':>8} | {'Experts':>8} | {'Top-K':>6} | {'Path':>30} | {'Time (ms)':>10}")
    print("-" * 80)

    for num_tokens, num_experts, top_k in test_cases:
        logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = fused_softmax_topk(logits, top_k, use_triton=True)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = fused_softmax_topk(logits, top_k, use_triton=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10 * 1000

        # Determine which path was taken
        if num_tokens < 256:
            path = "PyTorch (small_batch)"
        elif num_experts > 128:
            path = "PyTorch (too_many_experts)"
        elif top_k > 8:
            path = "PyTorch (topk_too_large)"
        else:
            path = "Triton"

        print(f"{num_tokens:>8} | {num_experts:>8} | {top_k:>6} | {path:>30} | {elapsed:>10.3f}")

    if verbose:
        disable_kernel_logging()
        summary = get_kernel_log_summary()
        print(f"\nKernel Log Summary: {summary}")


def test_expert_dispatch_strategies(verbose: bool = False):
    """Test different expert dispatch strategies."""
    print_section("EXPERT DISPATCH STRATEGY COMPARISON")

    try:
        from ava.nn.experts import ExpertParallelGroup
        from ava.kernels.fused_experts import enable_kernel_logging, disable_kernel_logging, get_kernel_stats, reset_kernel_stats
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("✗ CUDA not available, skipping expert dispatch tests")
        return

    if verbose:
        enable_kernel_logging(log_every_n=1)

    # Create expert group
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 4096
    k = 2
    num_tokens = 4096

    experts = ExpertParallelGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation='swiglu',
        dtype=torch.float16,
    ).to(device)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
    expert_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
    expert_weights = torch.softmax(torch.randn(num_tokens, k, device=device), dim=-1).to(torch.float16)

    strategies = [
        ("Fused Triton", dict(use_fused_triton=True, use_async_pipeline=False, use_loop_experts=False)),
        ("Async Pipeline", dict(use_fused_triton=False, use_async_pipeline=True, use_loop_experts=False)),
        ("Loop Experts", dict(use_fused_triton=False, use_async_pipeline=False, use_loop_experts=True)),
        ("Grouped GEMM", dict(use_fused_triton=False, use_async_pipeline=False, use_loop_experts=False, use_sparse_dispatch=False)),
        ("Sparse Gather", dict(use_fused_triton=False, use_async_pipeline=False, use_loop_experts=False, use_sparse_dispatch=True)),
    ]

    print(f"Test Config: {num_tokens} tokens, {num_experts} experts, top-{k}, {hidden_size}D hidden")
    print()
    print(f"{'Strategy':>20} | {'Time (ms)':>10} | {'Speedup':>8} | {'Status'}")
    print("-" * 60)

    baseline_time = None
    for name, kwargs in strategies:
        reset_kernel_stats()

        try:
            # Warmup
            for _ in range(3):
                _ = experts(hidden_states, expert_indices, expert_weights, **kwargs)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                _ = experts(hidden_states, expert_indices, expert_weights, **kwargs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 10 * 1000

            if baseline_time is None:
                baseline_time = elapsed
                speedup = "1.00x"
            else:
                speedup = f"{baseline_time / elapsed:.2f}x"

            stats = get_kernel_stats()
            status = "✓"
            if stats.triton_calls > 0:
                status += f" (Triton: {stats.triton_calls})"
            if stats.pytorch_calls > 0:
                status += f" (PyTorch: {stats.pytorch_calls})"

            print(f"{name:>20} | {elapsed:>10.3f} | {speedup:>8} | {status}")

        except Exception as e:
            print(f"{name:>20} | {'N/A':>10} | {'N/A':>8} | ✗ Error: {str(e)[:30]}")

    if verbose:
        disable_kernel_logging()


def estimate_d2d_overhead():
    """Estimate D2D memory copy overhead."""
    print_section("D2D MEMORY COPY OVERHEAD ESTIMATION")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("✗ CUDA not available")
        return

    # Parameters from typical training
    num_tokens = 4096
    k = 2
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 4096

    print("Typical training configuration:")
    print(f"  Tokens per batch: {num_tokens}")
    print(f"  Experts per token: {k}")
    print(f"  Total experts: {num_experts}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print()

    # Calculate memory for different approaches
    element_size = 2  # FP16

    # Old approach: index_select creates [N*k, H, I*2] tensor
    old_gate_up_bytes = num_tokens * k * hidden_size * intermediate_size * 2 * element_size
    old_down_bytes = num_tokens * k * intermediate_size * hidden_size * element_size
    old_total = old_gate_up_bytes + old_down_bytes

    # New approach: loop loads [H, I*2] per expert (but for all experts)
    new_per_expert = hidden_size * intermediate_size * 2 * element_size
    new_total = new_per_expert * num_experts

    # Fused Triton: no intermediate copies, just input/output
    fused_input = num_tokens * hidden_size * element_size
    fused_output = num_tokens * k * hidden_size * element_size
    fused_total = fused_input + fused_output

    print("D2D Memory Estimates (per forward pass):")
    print(f"  Old (index_select + bmm):")
    print(f"    gate_up weights: {old_gate_up_bytes / 1e9:.2f} GB")
    print(f"    down weights:    {old_down_bytes / 1e9:.2f} GB")
    print(f"    TOTAL:           {old_total / 1e9:.2f} GB")
    print()
    print(f"  New (loop over experts):")
    print(f"    Per expert:      {new_per_expert / 1e6:.2f} MB")
    print(f"    Total (E={num_experts}):  {new_total / 1e6:.2f} MB")
    print(f"    Reduction:       {old_total / new_total:.1f}x")
    print()
    print(f"  Fused Triton:")
    print(f"    Input:           {fused_input / 1e6:.2f} MB")
    print(f"    Output:          {fused_output / 1e6:.2f} MB")
    print(f"    TOTAL:           {fused_total / 1e6:.2f} MB")
    print(f"    Reduction:       {old_total / fused_total:.1f}x")


def run_minimal_training_test(verbose: bool = False):
    """Run a minimal training step to test full integration."""
    print_section("MINIMAL TRAINING INTEGRATION TEST")

    try:
        from ava.models.moe_layer import SparseMoELayer
        from ava.kernels.fused_experts import enable_kernel_logging, disable_kernel_logging, get_kernel_log_summary
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("✗ CUDA not available")
        return

    if verbose:
        enable_kernel_logging(log_every_n=1)

    # Create MoE layer
    moe = SparseMoELayer(
        hidden_size=512,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_token=2,
        router_type='mixtral',
        use_triton_kernels=True,
        dtype=torch.float16,
    ).to(device)

    print(f"MoE Layer Config:")
    print(f"  Hidden size: 512")
    print(f"  Intermediate size: 2048")
    print(f"  Num experts: 8")
    print(f"  Experts per token: 2")
    print()

    # Test forward pass
    batch_size = 4
    seq_len = 256
    x = torch.randn(batch_size, seq_len, 512, device=device, dtype=torch.float16)

    print("Running forward pass...")
    try:
        output, aux_loss, metrics = moe(x, training=True)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Aux loss: {aux_loss.item():.4f}")
        print(f"  Metrics: {list(metrics.keys())}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return

    # Test backward pass
    print("\nRunning backward pass...")
    try:
        loss = output.sum() + aux_loss
        loss.backward()
        print(f"✓ Backward pass successful")
        print(f"  Router grad norm: {moe.router.gate.weight.grad.norm().item():.4f}")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")

    if verbose:
        disable_kernel_logging()
        summary = get_kernel_log_summary()
        print(f"\nKernel Path Summary: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Kernel diagnostics test script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable kernel logging")
    parser.add_argument("--profile", action="store_true", help="Run with minimal output for profiling")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KERNEL DIAGNOSTICS FOR AVA MoE")
    print("="*60)

    if args.profile:
        # Minimal output for Nsight profiling
        test_expert_dispatch_strategies(verbose=False)
        return

    # Run all diagnostics
    triton_ok = check_triton_availability()
    imports_ok = check_kernel_imports()

    if triton_ok and imports_ok:
        test_softmax_topk_kernels(verbose=args.verbose)
        test_expert_dispatch_strategies(verbose=args.verbose)
        estimate_d2d_overhead()
        run_minimal_training_test(verbose=args.verbose)
    else:
        print("\n⚠ Some checks failed. Fix issues above before running kernel tests.")

    print("\n" + "="*60)
    print("  DIAGNOSTICS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
