#!/usr/bin/env python3
"""
Convert Parquet Files to Arrow Format

Converts all .parquet files in /project/code/data/pretokenized to .arrow format
so the training script can load them.

Usage:
    python convert_parquet_to_arrow.py

    # Optional: specify directory
    python convert_parquet_to_arrow.py /path/to/data/dir
"""

import pandas as pd
from datasets import Dataset
from pathlib import Path
import sys

def convert_parquet_to_arrow(data_dir: str = '/project/code/data/pretokenized'):
    """Convert all parquet files to arrow format."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return False

    parquet_files = sorted(data_dir.glob('*.parquet'))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return False

    print(f"Converting {len(parquet_files)} parquet files to arrow format...\n")
    print("=" * 70)

    total_rows = 0
    converted = 0
    failed = 0

    for i, parquet_file in enumerate(parquet_files, 1):
        try:
            # Read parquet
            print(f"\n[{i}/{len(parquet_files)}] Processing: {parquet_file.name}")
            df = pd.read_parquet(parquet_file)
            n_rows = len(df)
            total_rows += n_rows

            # Verify expected columns
            expected_cols = {'input_ids', 'attention_mask'}
            actual_cols = set(df.columns)
            if not expected_cols.issubset(actual_cols):
                print(f"  ✗ Missing columns. Found: {actual_cols}, Expected: {expected_cols}")
                failed += 1
                continue

            # Convert to Arrow (HuggingFace format)
            dataset = Dataset.from_pandas(df)
            arrow_file = parquet_file.with_suffix('.arrow')
            dataset.save_to_disk(str(arrow_file))

            size_mb = arrow_file.stat().st_size / 1e6
            print(f"  ✓ Converted to: {arrow_file.name}")
            print(f"    Rows: {n_rows:,} | Size: {size_mb:.1f}MB")

            converted += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"\nSummary:")
    print(f"  Converted: {converted}/{len(parquet_files)}")
    print(f"  Failed: {failed}")
    print(f"  Total rows: {total_rows:,}")

    if converted > 0:
        print(f"\n✓ Success! Training script will now load {converted} arrow files automatically.")
        return True
    else:
        print(f"\n✗ No files were converted successfully.")
        return False

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/project/code/data/pretokenized'
    success = convert_parquet_to_arrow(data_dir)
    sys.exit(0 if success else 1)
