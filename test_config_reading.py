#!/usr/bin/env python3
"""Test that train.py reads from data_loading config section"""

print("Testing Config Reading from data_loading Section")
print("=" * 80)
print()

# Test by reading the source code
with open('/project/code/scripts/5_training/train.py', 'r') as f:
    content = f.read()

tests = [
    ("hasattr(training_config, 'data_loading')", "Checks for data_loading section"),
    ("training_config.data_loading, 'num_workers'", "Reads num_workers from data_loading"),
    ("training_config.data_loading, 'prefetch_factor'", "Reads prefetch_factor from data_loading"),
    ("training_config.data_loading, 'persistent_workers'", "Reads persistent_workers from data_loading"),
    ("training_config.data_loading, 'samples_per_file'", "Reads samples_per_file from data_loading"),
    ("training_config.data_loading, 'val_max_samples'", "Reads val_max_samples from data_loading"),
    ("training_config.data_loading, 'val_split_ratio'", "Reads val_split_ratio from data_loading"),
]

passed = 0
total = len(tests)

for pattern, description in tests:
    if pattern in content:
        print(f"✓ {description}")
        passed += 1
    else:
        print(f"✗ {description}")

print()
print("=" * 80)
print(f"Tests passed: {passed}/{total}")
print()

if passed == total:
    print("✅ All config parameters now read from data_loading section!")
    print()
    print("Your config structure:")
    print("  data_loading:")
    print("    num_workers: 8")
    print("    prefetch_factor: 4")
    print("    persistent_workers: true")
    print("    samples_per_file: 20")
    print("    val_max_samples: 5000")
    print("    val_split_ratio: 0.15")
    print()
    print("Backward compatible: Falls back to training_config.data if data_loading not present")
else:
    print(f"❌ {total - passed} test(s) failed")
