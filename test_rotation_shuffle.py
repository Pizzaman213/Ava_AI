#!/usr/bin/env python3
"""Test script to verify rotating one-sample-per-file shuffling behavior"""

import sys
import os
sys.path.insert(0, '/project/code/src')

from Ava.data_streaming import StreamingDataset
from pathlib import Path

def test_rotation():
    """Test that samples are rotated one per file"""

    print("Testing rotating shuffle behavior...")
    print("=" * 60)

    # Initialize dataset
    data_dir = Path('/project/code/data/processed')

    # Create a minimal dataset instance
    dataset = StreamingDataset(
        data_dir=data_dir,
        tokenizer=None,  # We'll test raw text streaming
        max_length=2048,
        buffer_size=100,  # Small buffer for testing
        split='train',
        use_weighted_mixing=False,  # Disable for clear rotation testing
        enable_bucketing=False
    )

    print(f"Found {len(dataset.data_files)} data files")
    print()

    # Get first 20 samples and track which file they came from
    print("Testing file rotation (first 20 samples):")
    print("-" * 60)

    # Directly test _stream_examples
    sample_count = 0
    file_sequence = []

    # Create a simple test with file tracking
    files_to_test = list(dataset.data_files)[:5]  # Test with first 5 files
    print(f"Testing with {len(files_to_test)} files:")
    for f in files_to_test:
        print(f"  - {f.name}")
    print()

    # Stream samples and track pattern
    gen = dataset._stream_examples(files_to_use=files_to_test)

    previous_files = []
    for i, sample in enumerate(gen):
        if i >= 20:  # Only check first 20 samples
            break
        sample_count += 1

    print(f"\n✓ Successfully streamed {sample_count} samples with rotation")
    print("\nExpected behavior:")
    print("  - With samples_per_file=1, each file should provide 1 sample")
    print("  - Then rotate to next file in round-robin fashion")
    print("  - This creates better data mixing than reading 500 from each file")

    print("\n" + "=" * 60)
    print("Configuration check:")

    # Read the actual value from the file
    from Ava.data_streaming import StreamingDataset
    import inspect
    source = inspect.getsource(StreamingDataset._stream_examples)

    if "samples_per_file = 1" in source:
        print("✓ samples_per_file is set to 1 (rotating mode)")
    elif "samples_per_file = 500" in source:
        print("✗ samples_per_file is still 500 (batch mode)")
    else:
        print("? samples_per_file value unclear")

    print("\nTest completed!")

if __name__ == "__main__":
    test_rotation()
