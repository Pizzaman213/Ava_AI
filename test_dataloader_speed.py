#!/usr/bin/env python
"""Quick test to measure dataloader speed after worker optimization fixes."""

import sys
sys.path.insert(0, '/project/code')

import time
import torch
from src.Ava.data.pretokenized_loader import create_ultra_fast_dataloaders

if __name__ == '__main__':
    print("Creating dataloaders with 8 workers...")
    start = time.time()

    train_loader, val_loader = create_ultra_fast_dataloaders(
        batch_size=96,
        max_length=256,
        data_dir='/project/code/data/processed',
        num_workers=8,
        buffer_size=50000,
        prefetch_factor=4,
        persistent_workers=True,
        samples_per_file=64,
        cache_size=50,
        pad_token_id=0,
    )

    print(f"Dataloaders created in {time.time() - start:.2f}s\n")

    print("Testing first 5 batches...")
    iter_start = time.time()

    for i, batch in enumerate(train_loader):
        batch_time = time.time() - iter_start
        print(f"  Batch {i+1}: {batch_time:.3f}s (batch_size={batch['input_ids'].shape[0]}, seq_len={batch['input_ids'].shape[1]})")

        if i >= 4:  # Test 5 batches
            break
        iter_start = time.time()

    print("\n✓ Dataloader speed test complete!")
    print(f"  Average time per batch (last): {batch_time:.3f}s")
    print(f"  Expected iteration speed: ~{1/batch_time:.1f} it/s")
