#!/usr/bin/env python3
"""
Simple script to pretokenize a dataset (auto-detects inputs and outputs)

Usage examples:

1. Auto-detect everything from a config file (recommended):
   python pretokenize_dataset.py --config /project/code/configs/moe/tiny_moe.yaml

2. Auto-detect from standard locations:
   python pretokenize_dataset.py

3. Preview what would be processed (dry-run):
   python pretokenize_dataset.py --dry-run

4. Pretokenize a single JSONL file:
   python pretokenize_dataset.py --input /project/code/data/processed/file.jsonl

5. Pretokenize all files from a directory:
   python pretokenize_dataset.py --input-dir /project/code/data/processed

6. Force re-tokenization of existing files:
   python pretokenize_dataset.py --force

7. Use a custom tokenizer:
   python pretokenize_dataset.py --tokenizer /path/to/tokenizer

8. Process with multiple workers:
   python pretokenize_dataset.py --num-workers 8
"""

import argparse
import sys
from pathlib import Path
import yaml

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
    num_workers: int = 4,
    force: bool = False,
    dry_run: bool = False
):
    """Pretokenize all files in a directory."""

    from create_pretokenized_dataset import main as pretokenize_main
    import sys

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all JSONL files in input directory
    jsonl_files = sorted(input_path.glob('*.jsonl'))

    if not jsonl_files:
        print(f"⚠️  No JSONL files found in {input_dir}")
        return False

    print(f"📚 Found {len(jsonl_files)} file(s) to process")

    if dry_run:
        print(f"\n🔍 DRY-RUN MODE - No files will be created")

    print(f"\n📂 Input directory:  {input_path}")
    print(f"📂 Output directory: {output_path}")
    print(f"🤖 Tokenizer: {tokenizer_path}")
    print(f"⚙️  Max length: {max_length}, Min length: {min_length}")
    print(f"👷 Workers: {num_workers}\n")

    if dry_run:
        print("Files that would be processed:")
        for i, file in enumerate(jsonl_files, 1):
            output_file = output_path / f"{file.stem}.parquet"
            exists = output_file.exists()
            status = "✓ exists" if exists else "✗ new"
            will_process = (not exists) or force
            process_marker = "→ PROCESS" if will_process else "⊘ SKIP"
            print(f"  {i}. {file.name} ({status}) {process_marker}")
        print(f"\nFiles to be created: {sum(1 for f in jsonl_files if (not (output_path / f'{f.stem}.arrow').exists()) or force)}")
        return True

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


def load_config(config_file: str):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None


def auto_detect_input_dir(config_file: str = None) -> str:
    """Auto-detect input directory from config or standard locations."""
    candidates = []

    # Priority 1: Config file's data.data_dir if it points to processed
    if config_file:
        config = load_config(config_file)
        if config and 'data' in config and 'data_dir' in config['data']:
            data_dir = config['data']['data_dir']
            # Check if it points to processed directory
            if 'processed' in data_dir:
                candidates.append((data_dir, "from config (processed data)"))
            # If it points to pretokenized, use processed instead
            elif 'pretokenized' in data_dir:
                processed_dir = data_dir.replace('pretokenized', 'processed')
                candidates.append((processed_dir, "inferred from config (processed)"))

    # Priority 2: Standard location
    standard_processed = '/project/code/data/processed'
    candidates.append((standard_processed, "standard location"))

    # Priority 3: Current directory
    candidates.append((str(Path.cwd()), "current directory"))

    # Check candidates in order
    for path_str, source in candidates:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            jsonl_files = list(path.glob('*.jsonl'))
            if jsonl_files:
                print(f"✓ Found {len(jsonl_files)} JSONL file(s) {source}: {path_str}")
                return path_str

    # No suitable input found
    return None


def auto_detect_tokenizer(config_file: str = None) -> str:
    """Auto-detect tokenizer from config or use default."""
    if config_file:
        config = load_config(config_file)
        if config and 'data' in config and 'tokenizer_name' in config['data']:
            tokenizer = config['data']['tokenizer_name']
            print(f"✓ Using tokenizer from config: {tokenizer}")
            return tokenizer

    default_tokenizer = '/project/code/models/tokenizer/enhanced-50680'
    print(f"✓ Using default tokenizer: {default_tokenizer}")
    return default_tokenizer


def auto_detect_max_length(config_file: str = None, default: int = 2048) -> int:
    """Auto-detect max_length from config or use default."""
    if config_file:
        config = load_config(config_file)
        if config and 'data' in config and 'max_length' in config['data']:
            max_len = config['data']['max_length']
            print(f"✓ Using max_length from config: {max_len}")
            return max_len

    print(f"✓ Using default max_length: {default}")
    return default


def auto_detect_output_dir(config_file: str = None, input_dir: str = None) -> str:
    """Auto-detect or create output directory."""
    # Standard output location
    output_dir = '/project/code/data/pretokenized'

    # Create if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Using output directory: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Pretokenize a dataset for fast loading (auto-detects inputs/outputs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config option
    parser.add_argument(
        '--config',
        type=str,
        help='Load settings from config file (e.g., configs/moe/tiny_moe.yaml)'
    )

    # Input options (optional with smart defaults)
    parser.add_argument(
        '--input',
        type=str,
        help='Single input file (JSONL/Parquet/Arrow) - auto-detected if not specified'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Directory with input files - auto-detected if not specified'
    )

    # Output options (optional with smart defaults)
    parser.add_argument(
        '--output',
        type=str,
        help='Output .bin file (auto-generated if not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (defaults to /project/code/data/pretokenized)'
    )

    # Tokenizer options
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer (auto-detected from config or uses default)'
    )

    # Processing options
    parser.add_argument(
        '--max-length',
        type=int,
        help='Maximum sequence length (auto-detected from config or defaults to 2048)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum sequence length (default: 10)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    # Behavior options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-tokenization of existing files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be processed without creating files'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Pretokenize Dataset for Fast Loading (Auto-Detect Mode)")
    print("=" * 70 + "\n")

    # Load config if provided
    config_file = args.config
    if config_file:
        print(f"📋 Loading config: {config_file}")
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            sys.exit(1)

    # Auto-detect or use provided values
    print("\n🔍 Auto-detecting settings...\n")

    # Tokenizer
    tokenizer = args.tokenizer or auto_detect_tokenizer(config_file)

    # Max length
    max_length = args.max_length or auto_detect_max_length(config_file)

    # Input directory (unless single file specified)
    if args.input:
        print(f"✓ Using specified input file: {args.input}")
        input_dir = None
        input_file = args.input
    else:
        input_dir = args.input_dir or auto_detect_input_dir(config_file)
        if not input_dir:
            print("\n❌ Could not auto-detect input directory!")
            print("\nPlease:")
            print("  1. Specify --input-dir, or")
            print("  2. Provide --config with data.data_dir set, or")
            print("  3. Place JSONL files in /project/code/data/processed/")
            print("\nTo download datasets first, run:")
            print("  python /project/code/scripts/1_data_download/unified_download.py --help")
            sys.exit(1)
        input_file = None

    # Output directory
    output_dir = args.output_dir or auto_detect_output_dir(config_file, input_dir)

    print("\n" + "=" * 70)

    # Process single file or directory
    if input_file:
        output_file = args.output
        if not output_file:
            # Auto-generate output filename
            input_path = Path(input_file)
            output_file = str(Path(output_dir) / f"{input_path.stem}.parquet")

        print(f"\n📄 Processing single file...")
        success = pretokenize_single_file(
            input_file,
            output_file,
            tokenizer,
            max_length,
            args.min_length
        )
    else:
        print(f"\n📚 Processing directory...")
        success = pretokenize_directory(
            input_dir,
            output_dir,
            tokenizer,
            max_length,
            args.min_length,
            args.num_workers,
            force=args.force,
            dry_run=args.dry_run
        )

    print("\n" + "=" * 70)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()