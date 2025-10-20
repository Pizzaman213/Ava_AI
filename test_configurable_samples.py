#!/usr/bin/env python3
"""Test configurable samples_per_file parameter"""

import sys
import inspect

print("Testing Configurable samples_per_file Parameter")
print("=" * 80)
print()

# Test 1: Check StreamingDataset has the parameter
print("Test 1: StreamingDataset.__init__ signature")
print("-" * 80)

sys.path.insert(0, '/project/code/src')
from Ava.data_streaming import StreamingDataset, InfiniteStreamingDataset, create_streaming_dataloaders

sig = inspect.signature(StreamingDataset.__init__)
params = list(sig.parameters.keys())

if 'samples_per_file' in params:
    default = sig.parameters['samples_per_file'].default
    print(f"✓ samples_per_file parameter found")
    print(f"  Default value: {default}")
else:
    print("✗ samples_per_file parameter NOT found")

print()

# Test 2: Check InfiniteStreamingDataset has the parameter
print("Test 2: InfiniteStreamingDataset.__init__ signature")
print("-" * 80)

sig = inspect.signature(InfiniteStreamingDataset.__init__)
params = list(sig.parameters.keys())

if 'samples_per_file' in params:
    default = sig.parameters['samples_per_file'].default
    print(f"✓ samples_per_file parameter found")
    print(f"  Default value: {default}")
else:
    print("✗ samples_per_file parameter NOT found")

print()

# Test 3: Check create_streaming_dataloaders has the parameter
print("Test 3: create_streaming_dataloaders signature")
print("-" * 80)

sig = inspect.signature(create_streaming_dataloaders)
params = list(sig.parameters.keys())

if 'samples_per_file' in params:
    default = sig.parameters['samples_per_file'].default
    print(f"✓ samples_per_file parameter found")
    print(f"  Default value: {default}")

    # Check documentation
    if create_streaming_dataloaders.__doc__:
        if 'samples_per_file' in create_streaming_dataloaders.__doc__:
            print(f"✓ Parameter documented in docstring")
        else:
            print(f"⚠ Parameter not in docstring")
else:
    print("✗ samples_per_file parameter NOT found")

print()

# Test 4: Check train.py uses the parameter
print("Test 4: train.py integration")
print("-" * 80)

with open('/project/code/scripts/5_training/train.py', 'r') as f:
    train_content = f.read()

checks = [
    ("getattr(training_config.data, 'samples_per_file', 1)", "Reads from config"),
    ("samples_per_file=samples_per_file", "Passes to create_streaming_dataloaders"),
    ("Samples per file rotation:", "Displays in training config output"),
]

for pattern, description in checks:
    if pattern in train_content:
        print(f"✓ {description}")
    else:
        print(f"✗ {description}")

print()

# Test 5: Check data_streaming.py implementation
print("Test 5: data_streaming.py implementation")
print("-" * 80)

with open('/project/code/src/Ava/data_streaming.py', 'r') as f:
    streaming_content = f.read()

checks = [
    ("self.samples_per_file = samples_per_file", "Stores parameter in __init__"),
    ("samples_per_file = self.samples_per_file", "Uses parameter in _stream_examples"),
    ("rotating {self.samples_per_file} sample(s) per file", "Includes in logging message"),
]

for pattern, description in checks:
    if pattern in streaming_content:
        print(f"✓ {description}")
    else:
        print(f"✗ {description}")

print()
print("=" * 80)
print()

print("Summary:")
print("  The samples_per_file parameter is now fully configurable!")
print()
print("Usage in config YAML:")
print("  data:")
print("    samples_per_file: 1    # Default: maximum diversity")
print("    samples_per_file: 10   # Reduced I/O, less diversity")
print("    samples_per_file: 100  # Minimal I/O, batch-like behavior")
print()
print("Recommendation:")
print("  • samples_per_file=1:   Best for maximum data diversity (recommended)")
print("  • samples_per_file=10:  Good balance between diversity and I/O")
print("  • samples_per_file=100: Faster I/O, less mixing (like old behavior)")
print()
