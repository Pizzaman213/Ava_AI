#!/usr/bin/env python3
"""
Unified data preparation script - handles downloading and pretokenizing datasets

This is the main entry point for data preparation. It:
1. Checks if raw data exists
2. Optionally downloads datasets if needed
3. Pretokenizes all JSONL files to Arrow format
4. Reports what was created

Usage examples:

1. Auto-prepare everything with a config (recommended):
   python prepare_data.py --config configs/moe/tiny_moe.yaml

2. Preview what would happen (dry-run):
   python prepare_data.py --config configs/moe/tiny_moe.yaml --dry-run

3. Force re-download and re-tokenize:
   python prepare_data.py --config configs/moe/tiny_moe.yaml --force

4. Just tokenize (skip download):
   python prepare_data.py --config configs/moe/tiny_moe.yaml --no-download

5. Just download (skip tokenization):
   python prepare_data.py --config configs/moe/tiny_moe.yaml --download-only

6. Specify custom locations:
   python prepare_data.py --input-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
import subprocess
from pathlib import Path
import yaml

# Default locations
DEFAULT_PROCESSED_DIR = '/project/code/data/processed'
DEFAULT_PRETOKENIZED_DIR = '/project/code/data/pretokenized'
DEFAULT_TOKENIZER = '/project/code/models/tokenizer/enhanced-50680'

# Script locations
DOWNLOAD_SCRIPT = '/project/code/scripts/1_data_download/unified_download.py'
PRETOKENIZE_SCRIPT = '/project/code/scripts/2_data_prep/pretokenize_dataset.py'


def load_config(config_file: str):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None


def get_config_value(config, key_path, default=None):
    """Get nested config value using dot notation (e.g., 'data.data_dir')."""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def check_raw_data_exists(input_dir: str) -> tuple[bool, list]:
    """Check if JSONL files exist in input directory."""
    path = Path(input_dir)
    if not path.exists():
        return False, []

    jsonl_files = sorted(path.glob('*.jsonl'))
    return len(jsonl_files) > 0, jsonl_files


def check_pretokenized_data_exists(output_dir: str) -> tuple[bool, list]:
    """Check if Arrow files exist in output directory."""
    path = Path(output_dir)
    if not path.exists():
        return False, []

    arrow_files = sorted(path.glob('*.arrow'))
    return len(arrow_files) > 0, arrow_files


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def download_datasets(config_file: str = None, force: bool = False, dry_run: bool = False) -> bool:
    """Run the download script to get raw data."""
    print_section("Downloading Datasets")

    cmd = [
        sys.executable,
        DOWNLOAD_SCRIPT,
        '--help'
    ]

    print(f"📥 Dataset download script: {DOWNLOAD_SCRIPT}")
    print(f"⚙️  Config file: {config_file or 'None (will use default datasets)'}")
    print(f"🔄 Force re-download: {force}")
    print(f"🔍 Dry-run mode: {dry_run}")

    print("\n⚠️  Download script not automatically called (requires manual review)")
    print(f"\nTo download datasets, run:")
    print(f"  python {DOWNLOAD_SCRIPT} --help\n")

    return True


def pretokenize_datasets(
    input_dir: str,
    output_dir: str,
    config_file: str = None,
    force: bool = False,
    dry_run: bool = False,
    num_workers: int = 4
) -> bool:
    """Run the pretokenization script."""
    print_section("Pretokenizing Datasets")

    cmd = [
        sys.executable,
        PRETOKENIZE_SCRIPT,
    ]

    if config_file:
        cmd.extend(['--config', config_file])

    cmd.extend(['--input-dir', input_dir])
    cmd.extend(['--output-dir', output_dir])
    cmd.extend(['--num-workers', str(num_workers)])

    if force:
        cmd.append('--force')

    if dry_run:
        cmd.append('--dry-run')

    print(f"🤖 Pretokenization script: {PRETOKENIZE_SCRIPT}")
    print(f"📂 Input directory:  {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"👷 Workers: {num_workers}")
    print(f"🔄 Force re-tokenize: {force}")
    print(f"🔍 Dry-run mode: {dry_run}")

    if dry_run:
        print("\n🔍 DRY-RUN: Would execute:")
        print(f"  {' '.join(cmd)}\n")
        return True

    print(f"\nExecuting: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pretokenization: {e}")
        return False


def verify_data(input_dir: str, output_dir: str) -> tuple[bool, dict]:
    """Verify that data exists and report statistics."""
    raw_exists, raw_files = check_raw_data_exists(input_dir)
    pretok_exists, pretok_files = check_pretokenized_data_exists(output_dir)

    stats = {
        'raw_count': len(raw_files) if raw_exists else 0,
        'raw_size_mb': sum(f.stat().st_size for f in raw_files) / (1024 * 1024) if raw_files else 0,
        'pretok_count': len(pretok_files) if pretok_exists else 0,
        'pretok_size_mb': sum(f.stat().st_size for f in pretok_files) / (1024 * 1024) if pretok_files else 0,
    }

    return (raw_exists or pretok_exists), stats


def print_final_report(config_file: str, input_dir: str, output_dir: str, stats: dict):
    """Print final status report."""
    print_section("Summary")

    if config_file:
        print(f"📋 Config: {config_file}")

    print(f"\n📂 Raw data directory ({input_dir}):")
    if stats['raw_count'] > 0:
        print(f"   ✓ {stats['raw_count']} JSONL file(s)")
        print(f"   ✓ Total size: {stats['raw_size_mb']:.2f} MB")
    else:
        print(f"   ✗ No files found (run 'python {DOWNLOAD_SCRIPT}' to download)")

    print(f"\n📂 Pretokenized directory ({output_dir}):")
    if stats['pretok_count'] > 0:
        print(f"   ✓ {stats['pretok_count']} Arrow file(s)")
        print(f"   ✓ Total size: {stats['pretok_size_mb']:.2f} MB")
    else:
        print(f"   ✗ No files found (pretokenize raw data first)")

    print(f"\n💡 Next steps:")
    if stats['raw_count'] == 0:
        print(f"   1. Download datasets:")
        print(f"      python {DOWNLOAD_SCRIPT} --help")
    elif stats['pretok_count'] == 0:
        print(f"   1. Pretokenize raw data:")
        print(f"      python {PRETOKENIZE_SCRIPT} --input-dir {input_dir}")

    if stats['pretok_count'] > 0:
        print(f"   2. Start training with your pretokenized data!")


def main():
    parser = argparse.ArgumentParser(
        description='Unified data preparation - download and pretokenize datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config option
    parser.add_argument(
        '--config',
        type=str,
        help='Load settings from config file (e.g., configs/moe/tiny_moe.yaml)'
    )

    # Directory options
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DEFAULT_PROCESSED_DIR,
        help=f'Raw data directory (default: {DEFAULT_PROCESSED_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_PRETOKENIZED_DIR,
        help=f'Pretokenized output directory (default: {DEFAULT_PRETOKENIZED_DIR})'
    )

    # Processing options
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers for pretokenization (default: 4)'
    )

    # Behavior options
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download datasets, skip pretokenization'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip download, only pretokenize existing data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download and re-tokenization'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would happen without making changes'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Unified Data Preparation Pipeline")
    print("=" * 70)

    # Load config if provided
    config_file = args.config
    if config_file:
        print(f"\n📋 Loading config: {config_file}")
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            sys.exit(1)
        config = load_config(config_file)
        if not config:
            sys.exit(1)
    else:
        config = None

    # Get directories from config if available
    input_dir = args.input_dir
    output_dir = args.output_dir

    if config:
        # Try to get from config
        config_input = get_config_value(config, 'data.data_dir')
        if config_input and 'processed' in config_input:
            input_dir = config_input
            print(f"✓ Using input directory from config: {input_dir}")
        elif config_input and 'pretokenized' in config_input:
            # Config points to pretokenized, but we need raw data
            input_dir = config_input.replace('pretokenized', 'processed')
            print(f"✓ Using inferred input directory: {input_dir}")

    # Create directories
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Input directory:  {input_dir}")
    print(f"📂 Output directory: {output_dir}")

    # Check what data exists
    print("\n🔍 Checking data status...")
    raw_exists, _ = check_raw_data_exists(input_dir)
    pretok_exists, _ = check_pretokenized_data_exists(output_dir)

    if raw_exists:
        print(f"   ✓ Raw data found in {input_dir}")
    else:
        print(f"   ✗ No raw data in {input_dir}")

    if pretok_exists:
        print(f"   ✓ Pretokenized data found in {output_dir}")
    else:
        print(f"   ✗ No pretokenized data in {output_dir}")

    # Determine what to do
    if args.download_only:
        download_datasets(config_file, args.force, args.dry_run)
    elif args.no_download:
        if not raw_exists:
            print(f"\n⚠️  No raw data found and --no-download specified!")
            print(f"Please run:")
            print(f"  python {DOWNLOAD_SCRIPT} --help")
            sys.exit(1)
        success = pretokenize_datasets(
            input_dir, output_dir, config_file, args.force, args.dry_run, args.num_workers
        )
        sys.exit(0 if success else 1)
    else:
        # Default: prepare everything
        if not raw_exists and not args.dry_run:
            print(f"\n⚠️  No raw data found!")
            print(f"\nYou can:")
            print(f"  1. Download datasets first:")
            print(f"     python {DOWNLOAD_SCRIPT} --help")
            print(f"  2. Or use --dry-run to see what would happen")
            response = input(f"\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(0)

        # Download (if needed and not dry-run)
        if not raw_exists or args.force:
            download_datasets(config_file, args.force, args.dry_run)

        # Pretokenize
        success = pretokenize_datasets(
            input_dir, output_dir, config_file, args.force, args.dry_run, args.num_workers
        )

        if not args.dry_run:
            # Final report
            _, stats = verify_data(input_dir, output_dir)
            print_final_report(config_file, input_dir, output_dir, stats)

        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
