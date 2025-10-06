#!/usr/bin/env python3
"""
Automatic Optimization Enabler for train.py

This script automatically enables all optimizations in your training
without modifying train.py directly. Simply import this module at the
top of train.py or run training with this wrapper.

Usage Option 1 - Add to train.py:
    # Add at the very top of train.py, before other imports
    import enable_optimizations
    enable_optimizations.auto_enable()

Usage Option 2 - Wrapper script:
    python enable_optimizations.py train.py --config configs/gpu/small.yaml

Usage Option 3 - Environment variable:
    export ENABLE_TRAINING_OPTIMIZATIONS=1
    python train.py --config configs/gpu/small.yaml
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def auto_enable():
    """
    Automatically enable all optimizations.

    Call this at the top of train.py before any other imports.
    """
    # Check if already enabled
    if os.environ.get('_OPTIMIZATIONS_ENABLED') == '1':
        return

    # Mark as enabled
    os.environ['_OPTIMIZATIONS_ENABLED'] = '1'

    print("\n" + "=" * 80)
    print("🚀 AUTOMATIC TRAINING OPTIMIZATIONS ENABLED")
    print("=" * 80)

    # 1. Apply hardware optimizations immediately
    _apply_hardware_optimizations()

    # 2. Monkey-patch PyTorch components
    _monkey_patch_torch_components()

    # 3. Set optimal environment variables
    _set_environment_variables()

    print("✅ All optimizations auto-enabled!")
    print("=" * 80 + "\n")


def _apply_hardware_optimizations():
    """Apply hardware optimizations."""
    try:
        from Ava.optimization.hardware_optimizations import auto_optimize_hardware
        hw_opt = auto_optimize_hardware()
        print("✓ Hardware optimizations applied")
    except Exception as e:
        print(f"⚠️  Hardware optimization failed: {e}")


def _monkey_patch_torch_components():
    """Monkey-patch PyTorch to use optimized versions."""
    import torch

    # Patch Adam -> FusedAdam
    original_adam = torch.optim.Adam
    original_adamw = torch.optim.AdamW

    def patched_adam(*args, **kwargs):
        """Patched Adam that uses FusedAdam if available."""
        try:
            from Ava.optimization.fused_optimizers import FusedAdam
            kwargs.setdefault('fused', torch.cuda.is_available())
            kwargs.setdefault('foreach', True)
            print("ℹ️  Using FusedAdam instead of standard Adam")
            return FusedAdam(*args, **kwargs)
        except Exception as e:
            print(f"⚠️  FusedAdam not available ({e}), using standard Adam")
            return original_adam(*args, **kwargs)

    def patched_adamw(*args, **kwargs):
        """Patched AdamW that uses FusedAdam."""
        try:
            from Ava.optimization.fused_optimizers import FusedAdam
            kwargs.setdefault('fused', torch.cuda.is_available())
            kwargs.setdefault('foreach', True)
            print("ℹ️  Using FusedAdam instead of standard AdamW")
            return FusedAdam(*args, **kwargs)
        except Exception as e:
            print(f"⚠️  FusedAdam not available ({e}), using standard AdamW")
            return original_adamw(*args, **kwargs)

    # Only patch if optimizations requested
    if os.environ.get('ENABLE_TRAINING_OPTIMIZATIONS', '1') == '1':
        torch.optim.Adam = patched_adam  # type: ignore
        torch.optim.AdamW = patched_adamw  # type: ignore
        print("✓ Optimizers patched (Adam -> FusedAdam)")


def _set_environment_variables():
    """Set optimal environment variables."""
    env_vars = {
        # CUDA optimizations
        'CUDA_LAUNCH_BLOCKING': '0',  # Async execution
        'CUDA_MODULE_LOADING': 'LAZY',  # Lazy loading

        # NCCL optimizations (if using distributed)
        'NCCL_IB_TIMEOUT': '22',
        'NCCL_SOCKET_NTHREADS': '4',
        'NCCL_NSOCKS_PERTHREAD': '4',

        # PyTorch optimizations
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',

        # Mark optimizations as enabled
        'TRAINING_OPTIMIZATIONS_ACTIVE': '1'
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    print("✓ Environment variables optimized")


class OptimizationContext:
    """
    Context manager for automatic optimization injection.

    Usage:
        with OptimizationContext(config):
            # Your training code here
            model = create_model()
            # model is automatically optimized
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize context."""
        self.config = config or {}
        self.original_components = {}

    def __enter__(self):
        """Enter context and enable optimizations."""
        auto_enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


def wrap_training_script(script_path: str, args: list):
    """
    Wrap and run training script with optimizations.

    Args:
        script_path: Path to train.py
        args: Command-line arguments
    """
    # Enable optimizations
    auto_enable()

    # Import and run the script
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_module", script_path)
    if spec and spec.loader:
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["train_module"] = train_module

        # Set command-line args
        sys.argv = [script_path] + args

        # Execute
        spec.loader.exec_module(train_module)
    else:
        print(f"Error: Could not load {script_path}")
        sys.exit(1)


def create_optimized_wrapper(train_function):
    """
    Decorator to automatically optimize a training function.

    Usage:
        @create_optimized_wrapper
        def train(model, dataloader, optimizer):
            # Your training code
            pass
    """
    def wrapper(*args, **kwargs):
        # Enable optimizations
        auto_enable()

        # Call original function
        return train_function(*args, **kwargs)

    return wrapper


# === AUTO-ENABLE ON IMPORT ===

# Check if we should auto-enable
if os.environ.get('ENABLE_TRAINING_OPTIMIZATIONS', '0') == '1':
    auto_enable()


# === COMMAND-LINE INTERFACE ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enable training optimizations and run train.py"
    )
    parser.add_argument(
        'script',
        type=str,
        nargs='?',
        default='train.py',
        help='Training script to run (default: train.py)'
    )
    parser.add_argument(
        '--no-auto',
        action='store_true',
        help='Do not auto-enable optimizations'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test that optimizations can be loaded'
    )

    # Parse known args, pass rest to training script
    args, remaining = parser.parse_known_args()

    if args.test:
        print("Testing optimization modules...")
        try:
            from Ava.optimization.hardware_optimizations import auto_optimize_hardware
            from Ava.optimization.fused_optimizers import FusedAdam
            from Ava.optimization.gradient_optimizations import MixedPrecisionManager
            from Ava.data.optimized_dataloader import create_production_dataloader
            from Ava.training.optimization_integration import OptimizedTrainingSetup

            print("✅ All optimization modules loaded successfully!")
            print("\nAvailable optimizations:")
            print("  - Hardware optimizations (TF32, cuDNN, etc.)")
            print("  - Fused optimizers (FusedAdam, Adam8bit, Lion, Sophia)")
            print("  - Mixed precision training (BF16/FP16)")
            print("  - Optimized dataloaders (prefetch, packing)")
            print("  - Model compilation (torch.compile)")
            print("  - Gradient optimizations (compression, adaptive clipping)")
            print("  - Advanced attention (Flash Attention, MQA, GQA)")
            print("  - Profiling and monitoring")

            sys.exit(0)
        except Exception as e:
            print(f"❌ Failed to load optimizations: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Enable optimizations if not disabled
    if not args.no_auto:
        auto_enable()

    # Run training script if provided
    if args.script and os.path.exists(args.script):
        print(f"\n▶️  Running {args.script} with optimizations enabled...\n")
        wrap_training_script(args.script, remaining)
    else:
        print("\n" + "=" * 80)
        print("OPTIMIZATION ENABLER - USAGE")
        print("=" * 80)
        print("\nOption 1 - Import in train.py:")
        print("    # Add at top of train.py")
        print("    import enable_optimizations")
        print("    enable_optimizations.auto_enable()")
        print("\nOption 2 - Wrapper script:")
        print("    python enable_optimizations.py train.py --config configs/gpu/small.yaml")
        print("\nOption 3 - Environment variable:")
        print("    export ENABLE_TRAINING_OPTIMIZATIONS=1")
        print("    python train.py --config configs/gpu/small.yaml")
        print("\nOption 4 - Test optimizations:")
        print("    python enable_optimizations.py --test")
        print("=" * 80 + "\n")
