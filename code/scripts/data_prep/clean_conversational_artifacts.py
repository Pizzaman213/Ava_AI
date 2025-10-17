#!/usr/bin/env python3
"""
Clean conversational AI artifacts from training data.

This script removes examples containing conversational patterns like:
- "Assistant:", "User:", "Human:"
- "You are an AI assistant"
- Other instruction-following artifacts

This prevents the model from learning to output these tokens.
"""

import json
import re
from pathlib import Path
import argparse
from typing import List
import sys


def is_contaminated(text: str, patterns: List[str]) -> bool:
    """Check if text contains any contamination patterns."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_file(input_file: Path, output_file: Path, contamination_patterns: List[str], dry_run: bool = False):
    """Remove contaminated examples from a JSONL file."""

    cleaned_count = 0
    total_count = 0
    contaminated_examples = []

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {input_file.name}")

    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    if not dry_run:
        f_out = open(output_file, 'w', encoding='utf-8')

    for line in lines:
        total_count += 1

        try:
            data = json.loads(line)
            text = data.get('text', '')

            # Check for contamination
            if is_contaminated(text, contamination_patterns):
                contaminated_examples.append(text[:200])  # Store first 200 chars for review
            else:
                if not dry_run:
                    f_out.write(line)
                cleaned_count += 1

        except json.JSONDecodeError:
            print(f"  ⚠️  Warning: Skipping invalid JSON line {total_count}")
            continue

    if not dry_run:
        f_out.close()

    removed = total_count - cleaned_count
    removal_pct = (removed / total_count * 100) if total_count > 0 else 0

    print(f"  Total examples: {total_count:,}")
    print(f"  Kept: {cleaned_count:,}")
    print(f"  Removed: {removed:,} ({removal_pct:.1f}%)")

    if contaminated_examples and len(contaminated_examples) <= 5:
        print(f"\n  Sample contaminated text:")
        for i, example in enumerate(contaminated_examples[:3], 1):
            print(f"    {i}. {example}...")

    return total_count, cleaned_count, removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean conversational AI artifacts from training data"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/project/code/data/processed'),
        help='Directory containing JSONL training data files'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='.cleaned',
        help='Suffix to add to cleaned files (default: .cleaned)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually modifying files'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Replace original files (backup created with .bak extension)'
    )

    args = parser.parse_args()

    # Contamination patterns to detect
    contamination_patterns = [
        r'\bAssistant:\s',
        r'\bUser:\s',
        r'\bHuman:\s',
        r'\bAI:\s',
        r'you are an AI assistant',
        r'I am an AI',
        r'as an AI language model',
        r'I am a helpful assistant',
        r'I\'m an AI',
        r'as a language model',
        r'I don\'t have personal',
        r'I cannot ',
        r'I can\'t ',
        # Instruction patterns
        r'^\s*Answer the following question',
        r'^\s*Write an article based on',
        r'^\s*Translate the following',
    ]

    print("=" * 70)
    print("CONVERSATIONAL ARTIFACT CLEANER")
    print("=" * 70)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified")

    print(f"\nSearching for JSONL files in: {args.data_dir}")

    # Find all JSONL files
    jsonl_files = list(args.data_dir.glob("*.jsonl"))

    # Filter out already cleaned files
    jsonl_files = [f for f in jsonl_files if args.output_suffix not in f.name and '.bak' not in f.name]

    if not jsonl_files:
        print(f"\n❌ No JSONL files found in {args.data_dir}")
        return 1

    print(f"Found {len(jsonl_files)} file(s) to process\n")

    total_examples = 0
    total_kept = 0
    total_removed = 0

    for input_file in sorted(jsonl_files):
        if args.in_place:
            output_file = input_file.with_suffix('.jsonl.new')
            backup_file = input_file.with_suffix('.jsonl.bak')
        else:
            # Insert suffix before .jsonl extension
            output_file = input_file.with_suffix(f'{args.output_suffix}.jsonl')

        examples, kept, removed = clean_file(
            input_file,
            output_file,
            contamination_patterns,
            dry_run=args.dry_run
        )

        total_examples += examples
        total_kept += kept
        total_removed += removed

        # If in-place mode and not dry run, replace original file
        if args.in_place and not args.dry_run:
            input_file.rename(backup_file)
            output_file.rename(input_file)
            print(f"  ✓ Replaced original file (backup: {backup_file.name})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples processed: {total_examples:,}")
    print(f"Total kept: {total_kept:,}")
    print(f"Total removed: {total_removed:,} ({total_removed/total_examples*100:.1f}%)")

    if not args.dry_run:
        print(f"\n✓ Cleaned files saved to: {args.data_dir}")
        if args.in_place:
            print("  (Original files backed up with .bak extension)")
        else:
            print(f"  (Look for files with '{args.output_suffix}' in filename)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
