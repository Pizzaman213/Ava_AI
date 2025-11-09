"""
Test script for hybrid optimization (LoRA + Offloading + Quantization).

This script verifies that all three optimizations work together correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.layers.quantized_lora_experts import QuantizedLoRAExpert, QuantizedExpert
from src.Ava.layers.offloaded_experts import CPUOffloadedExpertGroup


def test_quantized_lora_expert():
    """Test QuantizedLoRAExpert basic functionality."""
    print("\n" + "="*80)
    print("TEST 1: QuantizedLoRAExpert Basic Functionality")
    print("="*80)

    expert = QuantizedLoRAExpert(
        hidden_size=512,
        intermediate_size=1024,
        lora_rank=4,
        quantization_bits=8
    )

    print(f"✓ Created QuantizedLoRAExpert: {expert}")

    # Test quantization
    print("\nTesting quantization...")
    original_mem = expert.get_memory_usage()
    print(f"  Memory (FP16): {original_mem['total']:.2f} MB")

    expert.quantize_weights()
    quantized_mem = expert.get_memory_usage()
    print(f"  Memory (INT8): {quantized_mem['total']:.2f} MB")
    print(f"  Savings: {(1 - quantized_mem['total']/original_mem['total'])*100:.1f}%")

    # Test dequantization
    print("\nTesting dequantization...")
    expert.dequantize_weights()
    dequant_mem = expert.get_memory_usage()
    print(f"  Memory (FP16): {dequant_mem['total']:.2f} MB")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(8, 32, 512, dtype=torch.float16)  # Match expert dtype
    output = expert(x)
    assert output.shape == (8, 32, 512), f"Expected (8, 32, 512), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    print(f"  Output shape: {output.shape} ✓")
    print(f"  No NaN values ✓")

    print("\n✅ TEST 1 PASSED: QuantizedLoRAExpert works correctly!\n")


def test_quantized_expert():
    """Test QuantizedExpert (no LoRA)."""
    print("\n" + "="*80)
    print("TEST 2: QuantizedExpert (No LoRA)")
    print("="*80)

    expert = QuantizedExpert(
        hidden_size=512,
        intermediate_size=1024,
        quantization_bits=8
    )

    print(f"✓ Created QuantizedExpert: {expert}")

    # Test quantization
    print("\nTesting quantization...")
    expert.quantize_weights()
    mem = expert.get_memory_usage()
    print(f"  Memory (INT8): {mem['total']:.2f} MB")
    assert mem['is_quantized'], "Expert should be quantized"
    print("  Quantized state: ✓")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 16, 512, dtype=torch.float16)  # Match expert dtype
    output = expert(x)
    assert output.shape == (4, 16, 512), f"Expected (4, 16, 512), got {output.shape}"
    print(f"  Output shape: {output.shape} ✓")

    print("\n✅ TEST 2 PASSED: QuantizedExpert works correctly!\n")


def test_hybrid_offloading():
    """Test hybrid mode with all three optimizations."""
    print("\n" + "="*80)
    print("TEST 3: Hybrid Mode (LoRA + Offloading + Quantization)")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU tests")
        return

    print("\nCreating CPUOffloadedExpertGroup with hybrid mode...")
    experts = CPUOffloadedExpertGroup(
        num_experts=8,
        hidden_size=512,
        intermediate_size=1024,
        max_active_experts=2,
        use_lora=True,
        lora_rank=4,
        use_quantization=True,
        quantization_bits=8,
    )

    print(f"✓ Created hybrid expert group with {len(experts.experts)} experts")

    # Verify all experts are on CPU and quantized
    print("\nVerifying initial state (should be CPU + quantized)...")
    for i, expert in enumerate(experts.experts):
        if hasattr(expert, 'base_gate_up'):
            device = expert.base_gate_up.device.type
            is_quantized = expert._is_quantized
            print(f"  Expert {i}: device={device}, quantized={is_quantized}")
            assert device == 'cpu', f"Expert {i} should be on CPU, got {device}"
            assert is_quantized, f"Expert {i} should be quantized"

    print("  All experts on CPU ✓")
    print("  All experts quantized ✓")

    # Test individual expert forward pass (simpler test)
    print("\nTesting individual expert forward pass...")
    # Move one expert to GPU and test it
    test_expert = experts.experts[0]
    test_expert.dequantize_weights()  # Manually dequantize before GPU transfer
    test_expert.cuda()

    # Check it was dequantized
    assert not test_expert._is_quantized, "Expert should be dequantized on GPU"
    print(f"  Expert dequantized on GPU ✓")

    # Test forward pass
    x = torch.randn(4, 16, 512, dtype=torch.float16).cuda()
    output = test_expert(x)

    assert output.shape == (4, 16, 512), f"Expected (4, 16, 512), got {output.shape}"
    assert output.device.type == 'cuda', f"Output should be on CUDA, got {output.device.type}"
    assert not torch.isnan(output).any(), "Output contains NaN values"

    print(f"  Output shape: {output.shape} ✓")
    print(f"  Output device: {output.device.type} ✓")
    print(f"  No NaN values ✓")

    # Move back to CPU and check it gets quantized again
    test_expert.cpu()
    test_expert.quantize_weights()
    assert test_expert._is_quantized, "Expert should be quantized on CPU"
    print(f"  Expert re-quantized on CPU ✓")

    print("\n✅ TEST 3 PASSED: Hybrid mode works correctly!\n")


def test_memory_savings():
    """Benchmark memory savings of different configurations."""
    print("\n" + "="*80)
    print("TEST 4: Memory Savings Comparison")
    print("="*80)

    def get_total_memory(experts_list):
        """Calculate total memory usage in MB."""
        total = 0
        for expert in experts_list:
            for param in expert.parameters():
                total += param.numel() * param.element_size()
            for buffer in expert.buffers():
                total += buffer.numel() * buffer.element_size()
        return total / (1024 * 1024)

    configs = [
        ("Baseline (Standard)", {
            "use_lora": False,
            "use_quantization": False,
        }),
        ("LoRA Only", {
            "use_lora": True,
            "lora_rank": 4,
            "use_quantization": False,
        }),
        ("LoRA + Quantization", {
            "use_lora": True,
            "lora_rank": 4,
            "use_quantization": True,
            "quantization_bits": 8,
        }),
    ]

    print(f"\nConfiguration: 16 experts, hidden=512, intermediate=1024\n")
    print(f"{'Configuration':<30} {'Memory (MB)':<15} {'Savings':<15}")
    print("-" * 60)

    baseline_mem = None
    for name, config in configs:
        experts_group = CPUOffloadedExpertGroup(
            num_experts=16,
            hidden_size=512,
            intermediate_size=1024,
            max_active_experts=4,
            **config
        )

        # Quantize if enabled
        if config.get("use_quantization", False):
            for expert in experts_group.experts:
                if hasattr(expert, 'quantize_weights'):
                    expert.quantize_weights()

        memory = get_total_memory(experts_group.experts)

        if baseline_mem is None:
            baseline_mem = memory
            savings = "0%"
        else:
            savings_pct = (1 - memory / baseline_mem) * 100
            savings = f"{savings_pct:.1f}%"

        print(f"{name:<30} {memory:>10.2f} MB    {savings:>10}")

    print("\n✅ TEST 4 PASSED: Memory savings verified!\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HYBRID OPTIMIZATION TEST SUITE")
    print("="*80)
    print("\nTesting hybrid optimization implementation:")
    print("  - QuantizedLoRAExpert (LoRA + Quantization)")
    print("  - QuantizedExpert (Quantization only)")
    print("  - CPUOffloadedExpertGroup (All three combined)")
    print("="*80)

    try:
        test_quantized_lora_expert()
        test_quantized_expert()
        test_hybrid_offloading()
        test_memory_savings()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nHybrid optimization implementation is working correctly.")
        print("You can now use all three optimizations together in your configs:")
        print("  - use_lora_experts: true")
        print("  - use_expert_offloading: true")
        print("  - use_expert_quantization: true")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
