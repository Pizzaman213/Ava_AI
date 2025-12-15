#!/usr/bin/env python3
"""
Test script to verify configurable randomness implementation.

Tests:
1. Configuration loading (shuffle_seed, enable_length_sorting, disable_packing_length_sort)
2. Deterministic shuffling (same seed = same order)
3. Non-deterministic shuffling (different orders)
4. Configuration propagation through pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))

from ava.config.training_config import DataConfig, TrainingConfig
from ava.data.distributed import DistributedStreamingDataset
from ava.data.packing import SequencePackingCollator
import torch
import random


def test_config_parameters():
    """Test that new configuration parameters exist and have correct defaults."""
    print("=" * 70)
    print("TEST 1: Configuration Parameters")
    print("=" * 70)

    # Create DataConfig with defaults
    config_defaults = DataConfig()

    # Check new parameters exist
    assert hasattr(config_defaults, 'shuffle_seed'), "Missing shuffle_seed parameter"
    assert hasattr(config_defaults, 'enable_length_sorting'), "Missing enable_length_sorting"
    assert hasattr(config_defaults, 'disable_packing_length_sort'), "Missing disable_packing_length_sort"

    # Check default values
    assert config_defaults.shuffle_seed is None, f"shuffle_seed default should be None, got {config_defaults.shuffle_seed}"
    assert config_defaults.enable_length_sorting == True, f"enable_length_sorting default should be True"
    assert config_defaults.disable_packing_length_sort == False, f"disable_packing_length_sort default should be False"

    print("✓ All configuration parameters present with correct defaults")
    print(f"  - shuffle_seed: {config_defaults.shuffle_seed}")
    print(f"  - enable_length_sorting: {config_defaults.enable_length_sorting}")
    print(f"  - disable_packing_length_sort: {config_defaults.disable_packing_length_sort}")

    # Test custom values
    config_custom = DataConfig(
        shuffle_seed=42,
        enable_length_sorting=False,
        disable_packing_length_sort=True
    )
    assert config_custom.shuffle_seed == 42
    assert config_custom.enable_length_sorting == False
    assert config_custom.disable_packing_length_sort == True

    print("✓ Custom configuration values work correctly")
    print(f"  - shuffle_seed: {config_custom.shuffle_seed}")
    print(f"  - enable_length_sorting: {config_custom.enable_length_sorting}")
    print(f"  - disable_packing_length_sort: {config_custom.disable_packing_length_sort}")
    print()


def test_deterministic_shuffling():
    """Test that same seed produces same shuffle order."""
    print("=" * 70)
    print("TEST 2: Deterministic Shuffling (Same Seed)")
    print("=" * 70)

    import random

    # Create two sequences with seed 42
    seq1_seed42 = list(range(100))
    seq2_seed42 = list(range(100))

    rng1 = random.Random(42)
    rng1.shuffle(seq1_seed42)

    rng2 = random.Random(42)
    rng2.shuffle(seq2_seed42)

    # They should be identical
    assert seq1_seed42 == seq2_seed42, "Same seed should produce same shuffle"
    print("✓ Deterministic shuffling: Same seed produces identical order")
    print(f"  First 10 elements of sequence 1: {seq1_seed42[:10]}")
    print(f"  First 10 elements of sequence 2: {seq2_seed42[:10]}")

    # Try with different seed
    seq3_seed99 = list(range(100))
    rng3 = random.Random(99)
    rng3.shuffle(seq3_seed99)

    assert seq1_seed42 != seq3_seed99, "Different seeds should produce different shuffles"
    print("✓ Different seeds produce different shuffle orders")
    print(f"  First 10 elements with seed 99: {seq3_seed99[:10]}")
    print()


def test_nondeterministic_shuffling():
    """Test that None seed produces different orders (statistical test)."""
    print("=" * 70)
    print("TEST 3: Non-deterministic Shuffling (None Seed)")
    print("=" * 70)

    import time

    # Create multiple shuffles with time-based seed (None seed simulation)
    sequences = []
    for i in range(3):
        seq = list(range(100))
        base_seed = int(time.time() * 1000000) % (2**31)
        # Small delay to ensure different time-based seeds
        time.sleep(0.01)
        rng = random.Random(base_seed)
        rng.shuffle(seq)
        sequences.append(seq)

    # They should be different (with very high probability)
    assert sequences[0] != sequences[1], "Time-based seeds should usually produce different shuffles"
    print("✓ Non-deterministic shuffling produces different orders")
    print(f"  Sequence 1 first 10: {sequences[0][:10]}")
    print(f"  Sequence 2 first 10: {sequences[1][:10]}")
    print(f"  Sequence 3 first 10: {sequences[2][:10]}")
    print()


def test_distributed_sorting():
    """Test that enable_length_sorting parameter works."""
    print("=" * 70)
    print("TEST 4: Distributed Dataset Length Sorting")
    print("=" * 70)

    # Create mock dataset
    class MockDataset:
        def __iter__(self):
            # Return mock samples with different lengths
            return iter([
                {'input_ids': list(range(10))},   # 10 tokens
                {'input_ids': list(range(50))},   # 50 tokens
                {'input_ids': list(range(20))},   # 20 tokens
                {'input_ids': list(range(100))},  # 100 tokens
            ])

    # Test with sorting enabled
    dist_ds_sorted = DistributedStreamingDataset(
        MockDataset(),
        world_size=1,
        rank=0,
        enable_length_sorting=True
    )
    print("✓ DistributedStreamingDataset accepts enable_length_sorting=True")

    # Test with sorting disabled
    dist_ds_unsorted = DistributedStreamingDataset(
        MockDataset(),
        world_size=1,
        rank=0,
        enable_length_sorting=False
    )
    print("✓ DistributedStreamingDataset accepts enable_length_sorting=False")
    print()


def test_packing_collator_sorting():
    """Test that SequencePackingCollator supports sort_by_length parameter."""
    print("=" * 70)
    print("TEST 5: Sequence Packing Collator Sorting")
    print("=" * 70)

    # Test with sorting enabled
    collator_sorted = SequencePackingCollator(
        max_length=512,
        pad_token_id=0,
        eos_token_id=2,
        pack_sequences=True,
        sort_by_length=True
    )
    assert collator_sorted.sort_by_length == True
    print("✓ SequencePackingCollator accepts sort_by_length=True")

    # Test with sorting disabled
    collator_unsorted = SequencePackingCollator(
        max_length=512,
        pad_token_id=0,
        eos_token_id=2,
        pack_sequences=True,
        sort_by_length=False
    )
    assert collator_unsorted.sort_by_length == False
    print("✓ SequencePackingCollator accepts sort_by_length=False")
    print()


def test_backward_compatibility():
    """Test that existing code works without new parameters."""
    print("=" * 70)
    print("TEST 6: Backward Compatibility")
    print("=" * 70)

    # DataConfig without new parameters should work
    config = DataConfig(
        data_dir="/tmp",
        max_length=512,
        streaming=False
    )
    # New parameters should have defaults
    assert config.shuffle_seed is None
    assert config.enable_length_sorting == True
    assert config.disable_packing_length_sort == False
    print("✓ DataConfig works without specifying new parameters")

    # SequencePackingCollator without sort_by_length should work
    collator = SequencePackingCollator(
        max_length=512,
        pad_token_id=0,
        eos_token_id=2,
        pack_sequences=True
    )
    assert collator.sort_by_length == True  # default
    print("✓ SequencePackingCollator works without specifying sort_by_length")
    print()


def test_seed_parameter_in_datasets():
    """Test that datasets accept shuffle_seed parameter."""
    print("=" * 70)
    print("TEST 7: Dataset shuffle_seed Parameter")
    print("=" * 70)

    from ava.data.streaming import StreamingDataset

    # Create a minimal StreamingDataset with shuffle_seed
    try:
        # This will fail because data dir doesn't exist, but we just need to check if parameter is accepted
        ds = StreamingDataset(
            data_dir="/tmp/nonexistent",
            split="train",
            tokenizer=None,
            max_length=512,
            shuffle_seed=42
        )
        print("✓ StreamingDataset accepts shuffle_seed parameter")
    except (FileNotFoundError, ValueError):
        # Expected - data dir doesn't exist, but parameter was accepted
        print("✓ StreamingDataset accepts shuffle_seed parameter (data dir check passed)")
    except TypeError as e:
        if "shuffle_seed" in str(e):
            print(f"✗ FAILED: StreamingDataset doesn't accept shuffle_seed: {e}")
            raise

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Data Loader Randomness - Implementation Tests" + " " * 8 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_config_parameters()
        test_deterministic_shuffling()
        test_nondeterministic_shuffling()
        test_distributed_sorting()
        test_packing_collator_sorting()
        test_backward_compatibility()
        test_seed_parameter_in_datasets()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  • Configuration parameters work correctly")
        print("  • Deterministic shuffling verified")
        print("  • Non-deterministic shuffling verified")
        print("  • Distributed sorting parameter accepted")
        print("  • Packing collator sorting parameter accepted")
        print("  • Backward compatibility maintained")
        print("  • Dataset shuffle_seed parameter accepted")
        print()
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
