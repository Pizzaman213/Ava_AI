#!/usr/bin/env python3
"""
Test script to verify constants loading from config file.
"""

import sys
sys.path.insert(0, '/project/code')

from src.Ava.config.training_config import DynamicConfig
from src.Ava.config.constants import DATA_CONSTANTS, update_constants_from_config
import yaml

def test_constants_loading():
    """Test that constants are loaded correctly from YAML config."""

    print("=" * 80)
    print("Testing Constants Loading from Config")
    print("=" * 80)

    # Load the config file
    config_path = '/project/code/configs/moe/tiny_moe_single_gpu.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert to DynamicConfig (same as training script does)
    config = DynamicConfig(config_dict)

    # Print original values
    print("\n1. Original DEFAULT values:")
    print(f"   buffer_size_default: {DATA_CONSTANTS.BUFFER_SIZE_DEFAULT}")
    print(f"   prefetch_max_workers: {DATA_CONSTANTS.PREFETCH_MAX_WORKERS}")
    print(f"   adaptive_samples_large_multiplier: {DATA_CONSTANTS.ADAPTIVE_SAMPLES_LARGE_MULTIPLIER}")
    print(f"   max_bucket_size: {DATA_CONSTANTS.MAX_BUCKET_SIZE}")

    # Update constants from config
    print("\n2. Loading constants from config...")
    update_constants_from_config(config)

    # Print updated values
    print("\n3. Updated values from YAML:")
    print(f"   buffer_size_default: {DATA_CONSTANTS.BUFFER_SIZE_DEFAULT}")
    print(f"   prefetch_max_workers: {DATA_CONSTANTS.PREFETCH_MAX_WORKERS}")
    print(f"   adaptive_samples_large_multiplier: {DATA_CONSTANTS.ADAPTIVE_SAMPLES_LARGE_MULTIPLIER}")
    print(f"   max_bucket_size: {DATA_CONSTANTS.MAX_BUCKET_SIZE}")

    # Verify values match config
    print("\n4. Verification:")
    expected_buffer = 50000
    expected_prefetch = 6
    expected_multiplier = 6
    expected_bucket = 300

    assert DATA_CONSTANTS.BUFFER_SIZE_DEFAULT == expected_buffer, \
        f"Expected buffer_size_default={expected_buffer}, got {DATA_CONSTANTS.BUFFER_SIZE_DEFAULT}"
    print(f"   ✓ buffer_size_default matches config: {expected_buffer}")

    assert DATA_CONSTANTS.PREFETCH_MAX_WORKERS == expected_prefetch, \
        f"Expected prefetch_max_workers={expected_prefetch}, got {DATA_CONSTANTS.PREFETCH_MAX_WORKERS}"
    print(f"   ✓ prefetch_max_workers matches config: {expected_prefetch}")

    assert DATA_CONSTANTS.ADAPTIVE_SAMPLES_LARGE_MULTIPLIER == expected_multiplier, \
        f"Expected adaptive_samples_large_multiplier={expected_multiplier}, got {DATA_CONSTANTS.ADAPTIVE_SAMPLES_LARGE_MULTIPLIER}"
    print(f"   ✓ adaptive_samples_large_multiplier matches config: {expected_multiplier}")

    assert DATA_CONSTANTS.MAX_BUCKET_SIZE == expected_bucket, \
        f"Expected max_bucket_size={expected_bucket}, got {DATA_CONSTANTS.MAX_BUCKET_SIZE}"
    print(f"   ✓ max_bucket_size matches config: {expected_bucket}")

    print("\n" + "=" * 80)
    print("✅ All tests passed! Constants are loaded correctly from config.")
    print("=" * 80)

if __name__ == '__main__':
    test_constants_loading()
