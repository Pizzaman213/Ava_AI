#!/usr/bin/env python3
"""Diagnose why only 3 files are being used during training"""

from pathlib import Path

data_dir = Path('/project/code/data/processed')
all_files = sorted(data_dir.glob("*_processed.jsonl"))

print("File Distribution Analysis")
print("=" * 80)
print()

# Step 1: Train/Val split
train_files = []
val_files = []

for file_path in all_files:
    file_hash = hash(file_path.name) % 100
    if file_hash < 85:
        train_files.append(file_path)
    else:
        val_files.append(file_path)

print(f"Step 1: Train/Val Split")
print(f"  Total files: {len(all_files)}")
print(f"  Train files: {len(train_files)}")
print(f"  Val files: {len(val_files)}")
print()

# Step 2: Worker distribution (8 workers)
num_workers = 8
print(f"Step 2: Worker Distribution ({num_workers} workers)")
print()

for worker_id in range(num_workers):
    worker_files = [f for i, f in enumerate(train_files) if i % num_workers == worker_id]
    print(f"  Worker {worker_id}: {len(worker_files)} files")
    for f in worker_files:
        print(f"    - {f.name}")
print()

# Step 3: Check for file opening errors
print(f"Step 3: File Accessibility Check")
print()

for f in train_files:
    try:
        size = f.stat().st_size
        size_mb = size / (1024 * 1024)

        # Try to open and read first line
        with open(f, 'r') as fp:
            first_line = fp.readline()
            status = "✓ OK"
    except Exception as e:
        status = f"✗ ERROR: {e}"
        size_mb = 0

    print(f"  {status:40s} | {size_mb:8.1f} MB | {f.name}")

print()
print("=" * 80)
print()
print("Possible Reasons for 'Streaming from 3 unique files':")
print()
print("1. Worker distribution: Each worker only gets 1-2 files")
print("   → The message is printed by worker 0 only, which may have 2-3 files")
print()
print("2. Weighted mixing: If enabled, some files might be filtered out")
print("   → Check use_weighted_mixing=True in train.py")
print()
print("3. File opening errors: Some files might fail to open")
print("   → Check above for any ERROR entries")
print()
print("4. The message counts unique files AFTER worker distribution")
print("   → Worker 0 (the one printing) might only have 3 files")
