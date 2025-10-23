#!/usr/bin/env python3
"""
Find Optimal Batch Size for Training

This script automatically finds the largest batch size that fits in GPU memory
without causing OOM errors.
"""

import os
import sys
import torch
import gc
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel
from src.Ava.models.colossalai_moe_model import create_colossal_moe_model


def test_batch_size(model, batch_size, seq_len, vocab_size, device="cuda"):
    """Test if a batch size fits in memory"""
    try:
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        # Create dummy batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        if loss is not None:
            loss.backward()

        # Check memory
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # Clear gradients
        model.zero_grad()

        print(f"✓ Batch size {batch_size}: Success (Memory: {memory_gb:.2f}GB)")
        return True, memory_gb

    except torch.cuda.OutOfMemoryError:
        print(f"✗ Batch size {batch_size}: OOM")
        torch.cuda.empty_cache()
        gc.collect()
        return False, None

    except Exception as e:
        print(f"✗ Batch size {batch_size}: Error - {e}")
        return False, None


def find_optimal_batch_size(model_config, seq_len=256, min_batch=1, max_batch=128):
    """Binary search to find optimal batch size"""

    print("\n" + "="*60)
    print("Finding Optimal Batch Size")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("No GPU available, using CPU")
        return 4

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f}GB")
    print(f"Sequence length: {seq_len}")

    # Create model
    print("\nCreating model...")
    model = create_colossal_moe_model(model_config)
    model = model.to(device)
    model.train()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Binary search
    print("\nTesting batch sizes...")
    left, right = min_batch, max_batch
    best_batch = min_batch
    best_memory = 0

    while left <= right:
        mid = (left + right) // 2

        # Test current batch size
        success, memory = test_batch_size(
            model, mid, seq_len, model_config.vocab_size, device
        )

        if success:
            best_batch = mid
            best_memory = memory if memory else 0
            left = mid + 1  # Try larger
        else:
            right = mid - 1  # Try smaller

    print("\n" + "="*60)
    print(f"Optimal batch size: {best_batch}")
    print(f"Memory usage: {best_memory:.2f}GB")
    print(f"Memory efficiency: {(best_memory / gpu_memory) * 100:.1f}%")

    # Recommend gradient accumulation
    target_effective_batch = 48  # Good for convergence
    if best_batch < target_effective_batch:
        grad_accum = target_effective_batch // best_batch
        print(f"\nRecommended settings:")
        print(f"  batch_size: {best_batch}")
        print(f"  gradient_accumulation_steps: {grad_accum}")
        print(f"  effective_batch_size: {best_batch * grad_accum}")
    else:
        print(f"\nRecommended settings:")
        print(f"  batch_size: {best_batch}")
        print(f"  gradient_accumulation_steps: 1")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_batch


def test_with_colossalai():
    """Test optimal batch size with Colossal-AI optimizations"""

    print("\n" + "="*60)
    print("Testing with Colossal-AI Optimizations")
    print("="*60)

    # Test different configurations
    configs = [
        ("No optimization", False, False, 1),
        ("Flash Attention", True, False, 1),
        ("ZeRO-1", True, True, 1),
        ("ZeRO-2", True, True, 2),
    ]

    results = []

    for name, use_flash, use_zero, zero_stage in configs:
        print(f"\n--- {name} ---")

        config = EnhancedMoEConfig(
            vocab_size=65536,
            hidden_size=512,
            num_layers=14,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=256,
            num_experts=8,
            num_experts_per_token=2,
            use_flash_attention=use_flash,
        )

        if use_zero:
            # This would need actual Colossal-AI initialization
            print(f"(ZeRO-{zero_stage} simulation)")

        optimal_batch = find_optimal_batch_size(config, seq_len=256)
        results.append((name, optimal_batch))

    # Summary
    print("\n" + "="*60)
    print("Summary of Optimal Batch Sizes")
    print("="*60)
    for name, batch_size in results:
        print(f"{name:<20}: {batch_size}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size")
    parser.add_argument("--max-batch", type=int, default=128, help="Maximum batch size")
    parser.add_argument("--test-colossalai", action="store_true", help="Test with Colossal-AI")

    args = parser.parse_args()

    if args.test_colossalai:
        test_with_colossalai()
    else:
        # Your model config
        config = EnhancedMoEConfig(
            vocab_size=65536,
            hidden_size=512,
            num_layers=14,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=256,
            num_experts=8,
            num_experts_per_token=2,
            use_flash_attention=True,
        )

        find_optimal_batch_size(
            config,
            seq_len=args.seq_len,
            min_batch=args.min_batch,
            max_batch=args.max_batch,
        )