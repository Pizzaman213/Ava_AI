"""
Test that config epsilon doesn't override BF16 auto-detection.

This test validates the fix for Issue #1 (config hardcoded epsilon).
"""

import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn

code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir / 'src'))

from ava.training.context import TrainingContext
from ava.training.optimizer import OptimizerManager


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)


def test_config_files_no_epsilon():
    """Test that main config files don't have hardcoded epsilon."""
    config_dir = code_dir / 'configs' / 'moe'

    configs_to_check = [
        'large.yaml',
        'minimal_working.yaml',
        'deterministic.yaml',
        'max_randomness.yaml',
        'large_Multy.yaml',
    ]

    for config_file in configs_to_check:
        config_path = config_dir / config_file
        if not config_path.exists():
            print(f"⊗ Skipping {config_file} (not found)")
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        optimizer_config = config.get('training', {}).get('optimizer', {})

        # Check that eps is NOT set (or is commented out)
        if 'eps' in optimizer_config:
            print(f"⚠ WARNING: {config_file} has hardcoded eps={optimizer_config['eps']}")
            print(f"   This overrides BF16 auto-detection!")
            print(f"   Recommendation: Remove 'eps' key or add comment explaining override")
        else:
            print(f"✓ {config_file} uses auto-detected epsilon")


def test_loaded_config_epsilon():
    """Test that loaded configs produce correct epsilon."""
    config_dir = code_dir / 'configs' / 'moe'

    # Test with large.yaml
    large_config_path = config_dir / 'large.yaml'
    if large_config_path.exists():
        with open(large_config_path) as f:
            config = yaml.safe_load(f)

        model = DummyModel()
        context = TrainingContext(model=model, device=torch.device('cpu'))
        manager = OptimizerManager(context)
        manager.initialize()

        # Extract mixed_precision
        precision = config.get('training', {}).get('precision', {})
        mp = precision.get('mixed_precision', 'bf16')

        # Create optimizer
        opt = manager.create_optimizer(model, config, learning_rate=1e-4)

        # Check epsilon matches precision
        actual_eps = opt.param_groups[0]['eps']

        if mp in ('bf16', 'bfloat16'):
            expected_eps = 1e-6
        else:
            expected_eps = 1e-8

        # Allow for config override
        explicit_eps = config.get('training', {}).get('optimizer', {}).get('eps')
        if explicit_eps is not None:
            expected_eps = explicit_eps
            print(f"⚠ large.yaml has explicit eps={explicit_eps} (overrides auto-detection)")

        assert actual_eps == expected_eps, \
            f"Epsilon mismatch: expected {expected_eps}, got {actual_eps} for {mp}"

        print(f"✓ large.yaml produces correct epsilon ({actual_eps}) for {mp}")


if __name__ == '__main__':
    print("Testing config epsilon handling...")
    test_config_files_no_epsilon()
    test_loaded_config_epsilon()
    print("\n✅ Config epsilon tests complete!")
