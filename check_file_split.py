#!/usr/bin/env python3
"""Check which files go to train vs val split"""

from pathlib import Path

data_dir = Path('/project/code/data/processed')
files = sorted(data_dir.glob("*_processed.jsonl"))

print("File split analysis:")
print("=" * 80)
print()

train_files = []
val_files = []

for file_path in files:
    file_hash = hash(file_path.name) % 100

    if file_hash < 85:
        split = "TRAIN"
        train_files.append(file_path)
    else:
        split = "VAL"
        val_files.append(file_path)

    size = file_path.stat().st_size
    size_mb = size / (1024 * 1024)

    print(f"{split:5s} (hash={file_hash:2d}) | {size_mb:8.1f} MB | {file_path.name}")

print()
print("=" * 80)
print(f"TRAIN files: {len(train_files)}/{len(files)} ({len(train_files)/len(files)*100:.1f}%)")
print(f"VAL files:   {len(val_files)}/{len(files)} ({len(val_files)/len(files)*100:.1f}%)")
print()

if len(train_files) < 5:
    print(f"⚠️  WARNING: Only {len(train_files)} files assigned to TRAIN split!")
    print("   This is why you're seeing 'Streaming from 3 unique files'")
    print()
    print("   The hash-based split is too aggressive for small datasets.")
    print("   With only 12 files, random hashing can create imbalanced splits.")
