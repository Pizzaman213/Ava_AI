#!/usr/bin/env python3
"""
Emergency restart script - completely fresh training start
"""

import os
import sys
import torch
import subprocess

def emergency_restart():
    """Emergency restart with minimal parameters for stability testing."""

    print("🚨 EMERGENCY RESTART - STOPPING ALL TRAINING")
    print("=" * 50)

    # 1. Kill any existing training processes
    try:
        subprocess.run(['pkill', '-f', 'train.py'], capture_output=True)
        print("✅ Stopped existing training processes")
    except:
        pass

    # 2. Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Cleared GPU cache")

    # 3. Start completely fresh training with minimal config
    print("\n🆕 Starting fresh training with minimal stable config...")

    # Ultra-conservative parameters for stability
    command = [
        'python', '/project/code/scripts/training/train.py',
        '--config', '/project/code/configs/gpu/small_stable.yaml',
        '--epochs', '1',  # Just 1 epoch for testing
        '--batch-size', '1',  # Minimal batch size
        '--max-samples', '100',  # Very limited samples for quick test
        '--learning-rate', '1e-5',  # Conservative LR
        '--max-gradient-norm', '5.0',  # Higher gradient clip threshold
        '--gradient-accumulation-steps', '4'
    ]

    print(f"Command: {' '.join(command)}")
    print("\n" + "=" * 50)

    # Execute
    os.chdir('/project/code')
    os.execvp('python', command)

if __name__ == "__main__":
    emergency_restart()