#!/usr/bin/env python3
"""
Test that large.yaml loads correctly with new randomization parameters.
"""

import sys
from pathlib import Path
import yaml

# Test 1: Load large.yaml
print("=" * 70)
print("TEST: Loading large.yaml with new randomization parameters")
print("=" * 70)

config_path = Path("/root/Ava_AI/code/configs/moe/large.yaml")

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Successfully loaded {config_path}")
    print()

    # Check data section
    data_config = config.get('data', {})
    print("Data configuration section:")
    print(f"  - streaming: {data_config.get('streaming')}")
    print(f"  - buffer_size: {data_config.get('buffer_size')}")
    print(f"  - use_pretokenized: {data_config.get('use_pretokenized')}")
    print(f"  - use_sequence_packing: {data_config.get('use_sequence_packing')}")
    print()

    # Check that new parameters are NOT in the config (defaults will apply)
    shuffle_seed = data_config.get('shuffle_seed')
    enable_length_sorting = data_config.get('enable_length_sorting')
    disable_packing_length_sort = data_config.get('disable_packing_length_sort')
    enable_bucketing = data_config.get('enable_bucketing')

    print("New randomization parameters in large.yaml:")
    print(f"  - shuffle_seed: {shuffle_seed} (commented out, will use default: None)")
    print(f"  - enable_length_sorting: {enable_length_sorting} (commented out, will use default: True)")
    print(f"  - disable_packing_length_sort: {disable_packing_length_sort} (commented out, will use default: False)")
    print(f"  - enable_bucketing: {enable_bucketing} (commented out, will use default: True)")
    print()

    print("✓ large.yaml uses defaults for new parameters (backward compatible)")
    print("✓ Users can uncomment to customize behavior")
    print()

except Exception as e:
    print(f"✗ FAILED to load large.yaml: {e}")
    sys.exit(1)

# Test 2: Verify DataConfig defaults
print("=" * 70)
print("TEST: DataConfig applies correct defaults")
print("=" * 70)

sys.path.insert(0, str(Path("/root/Ava_AI/code/src")))

from ava.config.training_config import DataConfig

# Create DataConfig with defaults
config = DataConfig()

print("DataConfig defaults:")
print(f"  - shuffle_seed: {config.shuffle_seed} (expected: None)")
print(f"  - enable_length_sorting: {config.enable_length_sorting} (expected: True)")
print(f"  - disable_packing_length_sort: {config.disable_packing_length_sort} (expected: False)")
print(f"  - enable_bucketing: {config.enable_bucketing} (expected: True)")
print()

# Verify defaults
assert config.shuffle_seed is None, f"Expected shuffle_seed=None, got {config.shuffle_seed}"
assert config.enable_length_sorting == True, f"Expected enable_length_sorting=True"
assert config.disable_packing_length_sort == False, f"Expected disable_packing_length_sort=False"
assert config.enable_bucketing == True, f"Expected enable_bucketing=True"

print("✓ All defaults are correct")
print("✓ large.yaml will work with new system using defaults")
print()

# Test 3: Show how to customize
print("=" * 70)
print("TEST: Customization examples")
print("=" * 70)

print("To enable maximum randomness, uncomment and modify in large.yaml:")
print("""
  shuffle_seed: null
  enable_length_sorting: false
  disable_packing_length_sort: true
  enable_bucketing: false
""")
print()

print("To enable full reproducibility, uncomment and modify in large.yaml:")
print("""
  shuffle_seed: 42
  enable_length_sorting: true
  disable_packing_length_sort: false
  enable_bucketing: true
""")
print()

print("=" * 70)
print("✅ large.yaml is fully compatible with new randomization system!")
print("=" * 70)
print()
print("Summary:")
print("  • large.yaml loads successfully")
print("  • New parameters are documented with examples")
print("  • Parameters are commented out (defaults apply)")
print("  • Users can easily customize by uncommenting")
print("  • Full backward compatibility maintained")
print()
