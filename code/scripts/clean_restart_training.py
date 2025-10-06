#!/usr/bin/env python3
"""
Clean restart script for training with problematic models.
This script ensures a completely fresh start without any checkpoint contamination.
"""

import os
import sys
import shutil
from pathlib import Path
import torch  # type: ignore[import-not-found]
import random
import numpy as np

# Add project root to path
sys.path.append('/project/code')

def clean_restart():
    """Perform a completely clean restart of training."""

    print("🧹 Starting clean restart procedure...")

    # 1. Clear all PyTorch caches
    print("  Clearing PyTorch caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 2. Reset random seeds for reproducibility
    print("  Resetting random seeds...")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 3. Clear any problematic environment variables
    print("  Clearing training environment...")
    env_vars_to_clear = [
        'WANDB_RESUME',
        'WANDB_RUN_ID',
        'DEEPSPEED_CONFIG',
        'CUDA_LAUNCH_BLOCKING'
    ]

    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            print(f"    Cleared {var}")

    # 4. Set clean environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error debugging
    os.environ['WANDB_START_METHOD'] = 'thread'

    print("✅ Clean restart preparation complete!")
    print("\n🚀 Starting training with stable configuration...")

    # 5. Launch training with explicit fresh start flags
    training_command = [
        'python', '/project/code/scripts/training/train.py',
        '--config', '/project/code/configs/gpu/small_stable.yaml',
        '--epochs', '3',  # Start with just 3 epochs to test stability
        '--batch-size', '2',  # Extra small for stability testing
        '--max-samples', '1000',  # Limit data for quick testing
        '--no-resume',  # Ensure no checkpoint resumption
        '--fresh-start'  # Force fresh start
    ]

    # Execute the training
    import subprocess
    try:
        result = subprocess.run(training_command,
                              cwd='/project/code',
                              capture_output=False,
                              text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def diagnose_model_issues():
    """Diagnose potential model initialization issues."""
    print("\n🔍 Diagnosing model initialization...")

    try:
        from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig  # type: ignore[import-not-found]

        # Test model creation with different initialization schemes
        config = EnhancedMoEConfig(
            vocab_size=50257,
            hidden_size=384,
            num_layers=6,
            num_attention_heads=6,
            intermediate_size=1024,
            num_experts=4,
            num_experts_per_token=1,
            initializer_range=0.02  # Key: proper initialization
        )

        print("  Creating test model...")
        model = EnhancedMoEModel(config)

        # Check parameter initialization
        param_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                std = param.std().item()
                mean = param.mean().item()
                param_stats.append((name, mean, std))

        print("  Parameter initialization check:")
        suspicious_params = []
        for name, mean, std in param_stats[:10]:  # Check first 10 parameters
            print(f"    {name}: mean={mean:.6f}, std={std:.6f}")
            if std > 0.1 or std < 0.001:  # Suspicious values
                suspicious_params.append(name)

        if suspicious_params:
            print(f"  ⚠️  Suspicious parameter initialization: {suspicious_params}")
        else:
            print("  ✅ Parameter initialization looks normal")

        # Test forward pass
        print("  Testing forward pass...")
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, 50257, (1, 10))
            try:
                outputs = model(input_ids)
                loss = outputs.get('loss', 'No loss computed')
                print(f"  ✅ Forward pass successful, loss: {loss}")
            except Exception as e:
                print(f"  ❌ Forward pass failed: {e}")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Model diagnosis failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Clean Training Restart Tool")
    print("=" * 50)

    # First diagnose the model
    model_ok = diagnose_model_issues()

    if not model_ok:
        print("\n❌ Model issues detected. Please check model implementation.")
        sys.exit(1)

    # Then perform clean restart
    success = clean_restart()

    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Check logs for details.")
        sys.exit(1)