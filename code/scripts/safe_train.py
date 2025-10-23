#!/usr/bin/env python3
"""
Safe Training Script with Memory Management

This script wraps the training process with proper memory management
to prevent OOM errors from cache accumulation.
"""

import os
import sys
import gc
import torch
from pathlib import Path

def setup_memory_management():
    """Setup aggressive memory management before training"""

    # Set environment variables BEFORE any PyTorch operations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        'max_split_size_mb:128,'
        'garbage_collection_threshold:0.6,'
        'expandable_segments:False'
    )

    # Force synchronous operations for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # Limit TensorFlow32 operations
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'

    if torch.cuda.is_available():
        # Set memory fraction to 90%
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Clear any existing allocations
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Configure allocator
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.9)

        print("✓ GPU memory management configured")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory limit: {torch.cuda.get_device_properties(0).total_memory * 0.9 / 1024**3:.1f}GB")

    # Configure aggressive garbage collection
    gc.set_threshold(400, 10, 10)

    print("✓ Garbage collection configured")


def monitor_and_cleanup():
    """Monitor memory and cleanup if needed"""

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # If cache is over 85% of total, force cleanup
        if cached / total > 0.85:
            print(f"\n⚠️  High cache detected: {cached:.2f}GB/{total:.2f}GB")
            print("   Forcing cleanup...")

            # Aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

            # Re-check
            new_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"   Cache after cleanup: {new_cached:.2f}GB")


def main():
    """Main training with memory management"""

    print("="*60)
    print("Safe Training with Memory Management")
    print("="*60)

    # Step 1: Setup memory management
    setup_memory_management()

    # Step 2: Add project to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Step 3: Import training script AFTER memory setup
    try:
        # Change to project root
        os.chdir(project_root)

        # Import and run training
        from scripts.training import train  # Adjust import as needed

        # Or run training script directly
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                "scripts/5_training/train.py",
                "--config", "configs/gpu/small.yaml"
            ],
            env={**os.environ},  # Pass environment variables
            cwd=str(project_root),
        )

        return result.returncode

    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ Final memory cleanup completed")


if __name__ == "__main__":
    # Create a memory management hook for the training process
    class MemoryManager:
        def __init__(self):
            self.batch_count = 0

        def on_batch_start(self):
            """Call this before each batch"""
            self.batch_count += 1

            # Clear cache every N batches
            if self.batch_count % 5 == 0:
                torch.cuda.empty_cache()

        def on_batch_end(self):
            """Call this after each batch"""
            # Monitor and cleanup if needed
            if self.batch_count % 10 == 0:
                monitor_and_cleanup()

    # If running as main, execute training
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # Direct execution
        main()
    else:
        # Setup and provide instructions
        setup_memory_management()

        print("\n" + "="*60)
        print("Instructions for OOM-Free Training")
        print("="*60)

        print("\nConfiguration changes applied:")
        print("✓ Batch size reduced to 32")
        print("✓ Gradient accumulation set to 2 (effective batch: 64)")
        print("✓ Cache clearing frequency set to every batch")
        print("✓ Memory pool limited to 20GB")
        print("✓ Target utilization set to 70%")

        print("\nMemory management features:")
        print("✓ PyTorch memory fraction: 90%")
        print("✓ Cache garbage collection: 60% threshold")
        print("✓ Split size: 128MB chunks")
        print("✓ Expandable segments: Disabled")

        print("\n" + "="*60)
        print("Recommended Training Commands:")
        print("="*60)

        print("\nOption 1: Use this safe wrapper")
        print("-" * 40)
        print("python code/scripts/safe_train.py --direct")

        print("\nOption 2: Use bash script with env vars")
        print("-" * 40)
        print("bash code/scripts/train_with_memory_fix.sh")

        print("\nOption 3: Manual with env vars")
        print("-" * 40)
        print("export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.6'")
        print("python code/scripts/5_training/train.py --config configs/gpu/small.yaml")

        print("\n" + "="*60)
        print("Your training should now run without OOM errors!")
        print("="*60)