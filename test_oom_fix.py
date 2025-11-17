#!/usr/bin/env python3
"""
Test script to verify the OOM fix in pretokenized_loader.py
"""

import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))

from Ava.data.pretokenized_loader import UltraFastPretokenizedDataset

def test_collate_fn_validation():
    """Test that collate_fn validates batch structure correctly"""

    # Create dataset
    dataset = UltraFastPretokenizedDataset(
        data_dir="/project/code/data/processed",
        split="train",
        max_length=256,
        max_samples=100,
        pad_token_id=0
    )

    print("✓ Dataset created successfully")

    # Test 1: Normal batch (should work)
    print("\n[TEST 1] Normal batch...")
    normal_batch = [
        {
            'input_ids': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'attention_mask': np.array([1, 1, 1, 1, 1], dtype=np.int64),
            'labels': np.array([1, 2, 3, 4, 5], dtype=np.int64)
        }
        for _ in range(32)
    ]

    try:
        result = dataset.collate_fn(normal_batch)
        print(f"✓ Normal batch collated successfully: {result['input_ids'].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 2: Wrong batch type (should fail with error message)
    print("\n[TEST 2] Wrong batch type (not a list)...")
    wrong_type_batch = "not a list"

    try:
        result = dataset.collate_fn(wrong_type_batch)
        print(f"✗ Should have failed but didn't!")
    except TypeError as e:
        print(f"✓ Correctly caught TypeError: {e}")
    except Exception as e:
        print(f"? Unexpected error: {type(e).__name__}: {e}")

    # Test 3: Absurdly large batch (should fail with error message)
    print("\n[TEST 3] Abnormally large batch (>10k items)...")
    huge_batch = [
        {
            'input_ids': np.array([1], dtype=np.int64),
            'attention_mask': np.array([1], dtype=np.int64),
            'labels': np.array([1], dtype=np.int64)
        }
        for _ in range(100000)  # 100k items
    ]

    try:
        result = dataset.collate_fn(huge_batch)
        print(f"✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")
    except Exception as e:
        print(f"? Unexpected error: {type(e).__name__}: {e}")

    # Test 4: Edge case - exactly at limit (should work)
    print("\n[TEST 4] Edge case - exactly at safety limit (10000 items)...")
    edge_batch = [
        {
            'input_ids': np.array([1], dtype=np.int64),
            'attention_mask': np.array([1], dtype=np.int64),
            'labels': np.array([1], dtype=np.int64)
        }
        for _ in range(10000)
    ]

    try:
        result = dataset.collate_fn(edge_batch)
        print(f"✓ Edge case batch collated successfully: {result['input_ids'].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    test_collate_fn_validation()