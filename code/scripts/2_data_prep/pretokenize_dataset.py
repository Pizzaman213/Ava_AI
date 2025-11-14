#!/usr/bin/env python3
"""
Simple script to pretokenize a dataset

Usage examples:

1. Pretokenize a single JSONL file:
   python pretokenize_dataset.py --input /project/code/data/processed/tatsu-lab_alpaca_processed.jsonl --output /tmp/output.bin

2. Pretokenize all files from processed directory:
   python pretokenize_dataset.py --input_dir /project/code/data/processed --output_dir /project/code/data/pretokenized

3. Use a custom tokenizer:
   python pretokenize_dataset.py --input file.jsonl --output output.bin --tokenizer /path/to/tokenizer

4. Process with multiple workers:
   python pretokenize_dataset.py --input_dir /project/code/data/processed --output_dir /project/code/data/pretokenized --num_workers 8
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, '/project/code/src')

# Get the directory of this script
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from create_pretokenized_dataset import tokenize_file, PreTokenizedDatasetReader
from transformers import AutoTokenizer


def pretokenize_single_file(
    input_file: str,
    output_file: str,
    tokenizer_path: str = '/project/code/models/tokenizer/enhanced-50680',
    max_length: int = 2048,
    min_length: int = 10
):
    """Pretokenize a single file."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        return False

    print(f"📚 Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"📄 Processing {input_path.name}")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")

    try:
        num_sequences = tokenize_file(
            input_path,
            output_path,
            tokenizer,
            max_length,
            min_length
        )

        print(f"\n✅ Success!")
        print(f"   Created: {output_path}")
        print(f"   Sequences: {num_sequences:,}")

        # Verify the file can be read
        print(f"\n🔍 Verifying output...")
        reader = PreTokenizedDatasetReader(output_path)
        print(f"   ✓ Can read {len(reader):,} sequences")
        sample = reader[0]
        print(f"   ✓ Sample length: {len(sample['input_ids'])} tokens")
        reader.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def pretokenize_directory(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str = '/project/code/models/tokenizer/enhanced-50680',
    max_length: int = 2048,
    min_length: int = 10,
    num_workers: int = 4
):
    """Pretokenize all files in a directory."""

    from create_pretokenized_dataset import main as pretokenize_main
    import sys

    # Override sys.argv for the script
    old_argv = sys.argv
    sys.argv = [
        'create_pretokenized_dataset.py',
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--tokenizer_path', tokenizer_path,
        '--max_length', str(max_length),
        '--min_length', str(min_length),
        '--num_workers', str(num_workers)
    ]

    try:
        pretokenize_main()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(
        description='Pretokenize a dataset for fast loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Single input file (JSONL/Parquet/Arrow)')
    input_group.add_argument('--input_dir', type=str, help='Directory with input files')

    # Output options
    parser.add_argument('--output', type=str, help='Output .bin file (for single file mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for directory mode)')

    # Tokenizer options
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='/project/code/models/tokenizer/enhanced-50680',
        help='Path to tokenizer (default: /project/code/models/tokenizer/enhanced-50680)'
    )

    # Processing options
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--min_length', type=int, default=10, help='Minimum sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    print("=" * 70)
    print("Pretokenize Dataset for Fast Loading")
    print("=" * 70)

    # Process
    if args.input:
        success = pretokenize_single_file(
            args.input,
            args.output,
            args.tokenizer,
            args.max_length,
            args.min_length
        )
    else:
        success = pretokenize_directory(
            args.input_dir,
            args.output_dir,
            args.tokenizer,
            args.max_length,
            args.min_length,
            args.num_workers
        )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()