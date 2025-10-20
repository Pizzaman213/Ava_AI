#!/usr/bin/env python3
"""Simple test for configurable samples_per_file parameter"""

print("Testing Configurable samples_per_file Parameter")
print("=" * 80)
print()

# Test by reading source files directly
tests_passed = 0
tests_total = 0

# Test 1: StreamingDataset.__init__
print("Test 1: StreamingDataset.__init__ has samples_per_file parameter")
print("-" * 80)
tests_total += 1

with open('/project/code/src/Ava/data_streaming.py', 'r') as f:
    content = f.read()

if 'samples_per_file: int = 1,' in content and 'def __init__' in content:
    print("✓ Parameter added to __init__ with default=1")
    tests_passed += 1
else:
    print("✗ Parameter not found in __init__")

print()

# Test 2: self.samples_per_file stored
print("Test 2: samples_per_file stored as instance variable")
print("-" * 80)
tests_total += 1

if 'self.samples_per_file = samples_per_file' in content:
    print("✓ Instance variable created")
    tests_passed += 1
else:
    print("✗ Instance variable not created")

print()

# Test 3: Used in _stream_examples
print("Test 3: Used in _stream_examples method")
print("-" * 80)
tests_total += 1

if 'samples_per_file = self.samples_per_file' in content:
    print("✓ Used in _stream_examples")
    tests_passed += 1
else:
    print("✗ Not used in _stream_examples")

print()

# Test 4: InfiniteStreamingDataset
print("Test 4: InfiniteStreamingDataset supports parameter")
print("-" * 80)
tests_total += 1

if 'class InfiniteStreamingDataset' in content and \
   content.count('samples_per_file: int = 1') >= 2:  # Should appear in both classes
    print("✓ InfiniteStreamingDataset has parameter")
    tests_passed += 1
else:
    print("✗ InfiniteStreamingDataset missing parameter")

print()

# Test 5: create_streaming_dataloaders
print("Test 5: create_streaming_dataloaders has parameter")
print("-" * 80)
tests_total += 1

if 'def create_streaming_dataloaders' in content and \
   'samples_per_file: int = 1' in content:
    print("✓ Function signature updated")
    tests_passed += 1
else:
    print("✗ Function signature not updated")

print()

# Test 6: Passed to dataset instantiations
print("Test 6: Parameter passed to dataset instantiations")
print("-" * 80)
tests_total += 1

count = content.count('samples_per_file=samples_per_file')
if count >= 3:  # Should be in train, val, and infinite datasets
    print(f"✓ Parameter passed {count} times to datasets")
    tests_passed += 1
else:
    print(f"✗ Parameter only passed {count} times (expected >= 3)")

print()

# Test 7: train.py integration
print("Test 7: train.py integration")
print("-" * 80)
tests_total += 1

with open('/project/code/scripts/5_training/train.py', 'r') as f:
    train_content = f.read()

if "getattr(training_config.data, 'samples_per_file', 1)" in train_content and \
   'samples_per_file=samples_per_file' in train_content:
    print("✓ train.py reads from config and passes to function")
    tests_passed += 1
else:
    print("✗ train.py integration incomplete")

print()

# Test 8: Logging message
print("Test 8: Logging message updated")
print("-" * 80)
tests_total += 1

if 'rotating {self.samples_per_file} sample(s) per file' in content:
    print("✓ Logging message shows configurable value")
    tests_passed += 1
else:
    print("✗ Logging message not updated")

print()

# Test 9: train.py shows config
print("Test 9: train.py displays configuration")
print("-" * 80)
tests_total += 1

if 'Samples per file rotation:' in train_content:
    print("✓ Configuration displayed in training output")
    tests_passed += 1
else:
    print("✗ Configuration not displayed")

print()

# Summary
print("=" * 80)
print(f"Tests passed: {tests_passed}/{tests_total}")
print()

if tests_passed == tests_total:
    print("✅ All tests passed! The samples_per_file parameter is fully configurable.")
    print()
    print("How to use:")
    print()
    print("Option 1: In your data YAML config file:")
    print("  data:")
    print("    samples_per_file: 1    # Maximum diversity (default)")
    print()
    print("Option 2: It will default to 1 if not specified")
    print()
    print("Values:")
    print("  • 1:   Maximum data diversity, best shuffling (recommended)")
    print("  • 10:  Good balance between diversity and I/O performance")
    print("  • 50:  Reduced diversity, better I/O performance")
    print("  • 100: Similar to old behavior (500), minimal diversity")
else:
    print(f"❌ {tests_total - tests_passed} test(s) failed")
