#!/usr/bin/env python3
"""
Data Integrity Checker for Ava Training Dataset.

This script scans all training data files to detect:
- Corrupted sequences (length > 10x configured max_length)
- Mismatched sequence lengths (input_ids vs attention_mask vs labels)
- Invalid token IDs (negative or too large)
- Empty sequences
- Arrow file format issues

Usage:
    python check_data_integrity.py --data-dir /path/to/data --max-length 256
    python check_data_integrity.py --parquet-files data/*.parquet
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print(" ERROR: pyarrow not installed. Install with: pip install pyarrow")
    sys.exit(1)


class DataIntegrityChecker:
    """Checks Arrow/Parquet files for data corruption."""

    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self.max_allowed_len = min(max_length * 10, 32768)

        # Metrics
        self.total_files = 0
        self.total_sequences = 0
        self.corrupted_sequences = 0
        self.empty_sequences = 0
        self.mismatched_sequences = 0
        self.error_files = 0

        # Detailed tracking
        self.errors: List[Dict] = []

    def check_file(self, file_path: str) -> bool:
        """Check a single Parquet/Arrow file for data integrity."""
        print(f"\n Checking {Path(file_path).name}...", end=" ")
        self.total_files += 1

        try:
            # Read the table
            table = pq.read_table(file_path)
            num_rows = len(table)

            # Check schema
            required_cols = {"input_ids"}
            optional_cols = {"attention_mask", "labels"}
            schema_cols = set(table.schema.names)

            if not required_cols.issubset(schema_cols):
                print(f"\n   Missing required columns: {required_cols - schema_cols}")
                self.error_files += 1
                return False

            print(f"({num_rows} sequences)", end="")

            # Check each row
            input_ids_col = table.column("input_ids")
            attention_mask_col = table.column("attention_mask") if "attention_mask" in schema_cols else None
            labels_col = table.column("labels") if "labels" in schema_cols else None

            rows_checked = 0
            errors_in_file = []

            for i in range(num_rows):
                rows_checked += 1
                self.total_sequences += 1

                try:
                    # Get input_ids
                    ids = input_ids_col[i].as_py()
                    if ids is None:
                        ids = []

                    # Convert to list if needed
                    if not isinstance(ids, list):
                        ids = list(ids)

                    seq_len = len(ids)

                    # Check for empty sequences
                    if seq_len == 0:
                        self.empty_sequences += 1
                        errors_in_file.append(
                            {"row": i, "type": "empty", "details": "Sequence has length 0"}
                        )
                        continue

                    # Check for catastrophic length
                    if seq_len > self.max_allowed_len:
                        self.corrupted_sequences += 1
                        errors_in_file.append({
                            "row": i,
                            "type": "catastrophic_length",
                            "details": f"Length {seq_len:,} exceeds max {self.max_allowed_len:,}",
                            "length": seq_len,
                        })
                        continue

                    # Check token validity
                    if seq_len > 0:
                        ids_array = np.array(ids, dtype=np.int64)
                        min_token = ids_array.min()
                        max_token = ids_array.max()

                        if min_token < 0:
                            errors_in_file.append({
                                "row": i,
                                "type": "invalid_token",
                                "details": f"Negative token ID: {min_token}",
                            })

                        if max_token > 100_000:  # Reasonable upper bound for token IDs
                            errors_in_file.append({
                                "row": i,
                                "type": "invalid_token",
                                "details": f"Token ID {max_token} seems too large",
                            })

                    # Check attention_mask length match
                    if attention_mask_col is not None:
                        mask = attention_mask_col[i].as_py()
                        if mask is not None:
                            mask = list(mask) if not isinstance(mask, list) else mask
                            if len(mask) != seq_len:
                                self.mismatched_sequences += 1
                                errors_in_file.append({
                                    "row": i,
                                    "type": "length_mismatch",
                                    "details": f"input_ids length {seq_len} ≠ attention_mask length {len(mask)}",
                                })

                    # Check labels length match
                    if labels_col is not None:
                        labels = labels_col[i].as_py()
                        if labels is not None:
                            labels = list(labels) if not isinstance(labels, list) else labels
                            if len(labels) != seq_len:
                                self.mismatched_sequences += 1
                                errors_in_file.append({
                                    "row": i,
                                    "type": "length_mismatch",
                                    "details": f"input_ids length {seq_len} ≠ labels length {len(labels)}",
                                })

                except Exception as e:
                    errors_in_file.append({
                        "row": i,
                        "type": "exception",
                        "details": f"Error processing row: {str(e)}",
                    })

            # Report file results
            if errors_in_file:
                print(f"\n    Found {len(errors_in_file)} errors in this file")
                for error in errors_in_file[:5]:  # Show first 5
                    print(f"    - Row {error['row']}: {error['type']}: {error['details']}")
                if len(errors_in_file) > 5:
                    print(f"    ... and {len(errors_in_file) - 5} more errors")
                self.errors.extend([{"file": file_path, **e} for e in errors_in_file])
                return False
            else:
                print(" ")
                return True

        except Exception as e:
            print(f"\n   Failed to read file: {e}")
            self.error_files += 1
            self.errors.append({"file": file_path, "type": "read_error", "details": str(e)})
            return False

    def check_directory(self, directory: str):
        """Recursively check all Parquet files in a directory."""
        print(f" Scanning directory: {directory}")
        directory = Path(directory)

        parquet_files = list(directory.glob("**/*.parquet")) + list(directory.glob("**/*.arrow"))

        if not parquet_files:
            print(f"  No Parquet/Arrow files found in {directory}")
            return

        print(f"Found {len(parquet_files)} files to check\n")

        for file_path in parquet_files:
            self.check_file(str(file_path))

    def print_report(self):
        """Print data integrity report."""
        print("\n" + "="*80)
        print(" DATA INTEGRITY REPORT")
        print("="*80)

        print(f"\n STATISTICS:")
        print(f"   Total files checked:        {self.total_files}")
        print(f"   Total sequences scanned:    {self.total_sequences:,}")
        print(f"   Files with errors:         {self.error_files}")

        print(f"\n  ISSUES FOUND:")
        print(f"   Corrupted sequences:       {self.corrupted_sequences:,}")
        print(f"   Empty sequences:           {self.empty_sequences:,}")
        print(f"   Mismatched lengths:        {self.mismatched_sequences:,}")

        if self.corrupted_sequences > 0:
            print(f"\n CRITICAL: Found {self.corrupted_sequences} sequences with catastrophic length!")
            print(f"   These sequences likely caused your 256 GiB OOM errors.")
            print(f"   Max allowed: {self.max_allowed_len:,} tokens")

        if self.errors:
            print(f"\n FIRST 10 DETAILED ERRORS:")
            for i, error in enumerate(self.errors[:10]):
                print(f"\n   {i+1}. {error.get('file', 'Unknown')}")
                print(f"      Row: {error.get('row', 'N/A')}")
                print(f"      Type: {error.get('type', 'Unknown')}")
                print(f"      Details: {error.get('details', 'N/A')}")

        print("\n" + "="*80)

        if self.corrupted_sequences > 0 or self.error_files > 0:
            print(" DATA INTEGRITY CHECK FAILED")
            print("\nRECOMMENDATIONS:")
            print("1. Re-run data preprocessing/tokenization pipeline")
            print("2. Check for bugs in collate_fn or data loading code")
            print("3. Verify Arrow/Parquet files are not corrupted")
            print("4. Ensure tokenizer output is valid (no massive sequences)")
            return False
        else:
            print(" ALL DATA INTEGRITY CHECKS PASSED")
            return True


def main():
    parser = argparse.ArgumentParser(description="Check Ava training data for corruption")
    parser.add_argument("--data-dir", type=str, help="Directory containing training data")
    parser.add_argument("--parquet-files", nargs="+", help="List of Parquet/Arrow files to check")
    parser.add_argument("--max-length", type=int, default=256, help="Configured max sequence length")

    args = parser.parse_args()

    if not args.data_dir and not args.parquet_files:
        print(" ERROR: Provide either --data-dir or --parquet-files")
        parser.print_help()
        sys.exit(1)

    checker = DataIntegrityChecker(max_length=args.max_length)

    if args.data_dir:
        checker.check_directory(args.data_dir)
    else:
        for file_path in args.parquet_files:
            checker.check_file(file_path)

    success = checker.print_report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
