#!/usr/bin/env python3
"""
Fix GPU Cache Issues

This script sets up proper PyTorch memory management to prevent
cache accumulation and OOM errors.
"""

import os
import gc
import torch

def setup_memory_management():
    """Configure PyTorch for aggressive memory management"""

    print("Setting up GPU memory management...")

    # 1. Set environment variables BEFORE importing PyTorch modules
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Keep async for speed

    # 2. Configure PyTorch memory allocator
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use max 90% of GPU

        # Empty cache aggressively
        torch.cuda.empty_cache()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Synchronize to ensure cleanup
        torch.cuda.synchronize()

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"Memory Fraction Set: 90%")
        print(f"Max Memory Available: {torch.cuda.get_device_properties(0).total_memory * 0.9 / 1024**3:.1f}GB")

    # 3. Configure garbage collection
    gc.set_threshold(400, 10, 10)  # More aggressive GC

    print("Memory management configured!")
    print("\nRecommended training command:")
    print("python code/scripts/5_training/train.py --config configs/gpu/small.yaml")

    return True


def clear_gpu_memory():
    """Aggressively clear GPU memory"""

    if torch.cuda.is_available():
        # Get current memory
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        cached_before = torch.cuda.memory_reserved() / 1024**3

        print(f"\nBefore cleanup:")
        print(f"  Allocated: {allocated_before:.2f}GB")
        print(f"  Cached: {cached_before:.2f}GB")

        # Clear everything
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Force cache release
        if hasattr(torch.cuda, 'memory._dump_snapshot'):
            torch.cuda.memory._dump_snapshot()

        # Get new memory
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        cached_after = torch.cuda.memory_reserved() / 1024**3

        print(f"\nAfter cleanup:")
        print(f"  Allocated: {allocated_after:.2f}GB")
        print(f"  Cached: {cached_after:.2f}GB")
        print(f"  Freed: {(cached_before - cached_after):.2f}GB")


def monitor_memory_usage():
    """Monitor current GPU memory usage"""

    if not torch.cuda.is_available():
        print("No GPU available")
        return

    # Get memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved

    print("\n" + "="*50)
    print("GPU Memory Status")
    print("="*50)
    print(f"Total GPU Memory: {total:.2f}GB")
    print(f"Allocated: {allocated:.2f}GB ({allocated/total*100:.1f}%)")
    print(f"Reserved (Cache): {reserved:.2f}GB ({reserved/total*100:.1f}%)")
    print(f"Free: {free:.2f}GB ({free/total*100:.1f}%)")
    print(f"Peak Allocated: {max_allocated:.2f}GB")

    # Warnings
    if reserved / total > 0.9:
        print("\n⚠️  WARNING: Cache usage is very high!")
        print("   Running cleanup...")
        clear_gpu_memory()
    elif reserved / total > 0.8:
        print("\n⚠️  Cache usage is high, consider cleanup")


def test_batch_size_with_cache_management(batch_size=32):
    """Test if batch size works with proper cache management"""

    print(f"\nTesting batch size {batch_size} with cache management...")

    # Setup memory management first
    setup_memory_management()

    # Import model after setup
    from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel

    try:
        # Create model
        config = EnhancedMoEConfig(
            vocab_size=65536,
            hidden_size=512,
            num_layers=14,
            num_attention_heads=8,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_token=2,
        )

        model = EnhancedMoEModel(config)
        model = model.cuda()
        model.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Test multiple batches
        for i in range(5):
            # Clear cache before each batch
            if i > 0:
                torch.cuda.empty_cache()

            # Create batch
            input_ids = torch.randint(0, config.vocab_size, (batch_size, 256)).cuda()
            labels = input_ids.clone()

            # Forward
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Monitor memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  Batch {i+1}: Allocated={allocated:.2f}GB, Cached={cached:.2f}GB")

            # Aggressive cleanup every batch
            del outputs, loss, input_ids, labels

        print(f"✅ Batch size {batch_size} works with cache management!")

        # Cleanup
        del model, optimizer
        clear_gpu_memory()

        return True

    except torch.cuda.OutOfMemoryError:
        print(f"❌ Batch size {batch_size} still causes OOM")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Setup memory management")
    parser.add_argument("--clear", action="store_true", help="Clear GPU memory")
    parser.add_argument("--monitor", action="store_true", help="Monitor memory usage")
    parser.add_argument("--test", type=int, help="Test batch size with cache management")

    args = parser.parse_args()

    if args.setup:
        setup_memory_management()
    elif args.clear:
        clear_gpu_memory()
    elif args.monitor:
        monitor_memory_usage()
    elif args.test:
        test_batch_size_with_cache_management(args.test)
    else:
        # Default: setup and monitor
        setup_memory_management()
        monitor_memory_usage()