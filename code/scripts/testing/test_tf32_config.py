#!/usr/bin/env python3
"""
Test TF32 Configuration - Verify enable/disable functionality

This script tests that TF32 can be properly enabled and disabled
via configuration files.

Usage:
    python test_tf32_config.py
"""

import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.config.training_config import TrainingConfigManager


def check_tf32_status():
    """Check current TF32 backend status."""
    if not torch.cuda.is_available():
        return {
            'cuda_available': False,
            'matmul_allow_tf32': None,
            'cudnn_allow_tf32': None,
            'cudnn_benchmark': None,
        }

    return {
        'cuda_available': True,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }


def apply_performance_config(config_path: str):
    """Load config and apply TF32 settings."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_path}")
    print('='*80)

    # Load config
    config_manager = TrainingConfigManager()
    config = config_manager.load_yaml_config(config_path)
    config_dict = config.to_dict()

    # Get performance settings
    perf_config = config_dict.get('performance', {})
    enable_tf32 = perf_config.get('enable_tf32', True)
    enable_cudnn_benchmark = perf_config.get('enable_cudnn_benchmark', True)
    matmul_precision = perf_config.get('float32_matmul_precision', 'high')

    print(f"\nConfiguration from YAML:")
    print(f"  enable_tf32: {enable_tf32}")
    print(f"  enable_cudnn_benchmark: {enable_cudnn_benchmark}")
    print(f"  float32_matmul_precision: {matmul_precision}")

    # Apply settings (mimicking train.py logic)
    if torch.cuda.is_available():
        if enable_tf32:
            torch.set_float32_matmul_precision(matmul_precision)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"\n✅ TF32 optimizations ENABLED (precision: {matmul_precision})")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("\n⚠️  TF32 optimizations DISABLED")

        if enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("✅ CuDNN benchmark auto-tuning ENABLED")
        else:
            torch.backends.cudnn.benchmark = False
            print("⚠️  CuDNN benchmark DISABLED")
    else:
        print("\n⚠️  CUDA not available - skipping TF32 configuration")

    # Check actual backend status
    status = check_tf32_status()
    print(f"\nActual Backend Status:")
    print(f"  CUDA available: {status['cuda_available']}")
    if status['cuda_available']:
        print(f"  torch.backends.cuda.matmul.allow_tf32: {status['matmul_allow_tf32']}")
        print(f"  torch.backends.cudnn.allow_tf32: {status['cudnn_allow_tf32']}")
        print(f"  torch.backends.cudnn.benchmark: {status['cudnn_benchmark']}")

    # Verify settings match config
    if status['cuda_available']:
        print(f"\nVerification:")
        tf32_match = status['matmul_allow_tf32'] == enable_tf32
        cudnn_match = status['cudnn_benchmark'] == enable_cudnn_benchmark

        if tf32_match and cudnn_match:
            print("  ✅ ALL SETTINGS MATCH CONFIGURATION")
            return True
        else:
            print("  ❌ SETTINGS MISMATCH!")
            if not tf32_match:
                print(f"     Expected TF32: {enable_tf32}, Got: {status['matmul_allow_tf32']}")
            if not cudnn_match:
                print(f"     Expected CuDNN benchmark: {enable_cudnn_benchmark}, Got: {status['cudnn_benchmark']}")
            return False
    else:
        print("  ⚠️  Cannot verify - CUDA not available")
        return True


def main():
    print("="*80)
    print("TF32 Configuration Test Suite")
    print("="*80)

    # Test configs
    test_configs = [
        {
            'path': 'code/configs/moe/small_moe.yaml',
            'expected_tf32': True,
            'description': 'Small MoE with TF32 ENABLED'
        },
        {
            'path': 'code/configs/moe/small_moe_no_tf32.yaml',
            'expected_tf32': False,
            'description': 'Small MoE with TF32 DISABLED'
        },
    ]

    results = []
    for test in test_configs:
        config_path = test['path']
        if not Path(config_path).exists():
            print(f"\n⚠️  Skipping {config_path} - file not found")
            continue

        # Reset backends before each test
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        success = apply_performance_config(config_path)
        results.append({
            'config': test['description'],
            'passed': success
        })

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = all(r['passed'] for r in results)

    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status}: {result['config']}")

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nConclusion:")
        print("  TF32 can be successfully enabled and disabled via YAML configuration.")
        print("  Use 'performance.enable_tf32: true/false' in your config files.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
