#!/usr/bin/env python3
"""
Multi-Dataset Downloader: OpenOrca, C4, TinyStories

Features:
- Download multiple datasets: OpenOrca, C4, TinyStories
- Parallel downloads with configurable workers
- Resume incomplete downloads
- Skip already downloaded files
- Optional pre-tokenization support
"""

# Auto-install requirements if needed (must be before other imports)
import subprocess
import sys
from pathlib import Path

def auto_install_requirements():
    """Auto-install requirements if imports fail"""
    project_root = Path(__file__).resolve().parents[2]
    requirements_file = project_root / "requirements.txt"

    print("🔧 Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file), "--upgrade", "-q"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

# Try imports with auto-install
try:
    import os
    import argparse
except (ImportError, ModuleNotFoundError) as e:
    print(f"❌ Import error: {e}")
    print("🔧 Attempting to install requirements...")
    if auto_install_requirements():
        print("🔄 Retrying imports...")
        import os
        import argparse
    else:
        print("❌ Failed to install requirements. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


def download_dataset(dataset_name, output_dir, num_workers=4, max_partitions=None):
    """Download a Hugging Face dataset efficiently"""
    save_path = output_dir / dataset_name.split("/")[-1]

    print(f"\n{'='*70}")
    print(f" DOWNLOADING: {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Destination: {save_path}")
    print(f"Workers: {num_workers}")
    if max_partitions:
        print(f"Max partitions: {max_partitions}")
    print()

    # Check if already exists with actual data
    if save_path.exists():
        try:
            files = [f for f in save_path.glob("*") if f.name != ".git"]
            if files:
                print(f"✓ Dataset already exists with {len(files)} items")
                print(f"  Using: incremental sync\n")
        except OSError as e:
            print(f"⚠ Warning: Could not check existing dataset: {e}")

    try:
        from huggingface_hub import snapshot_download

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading from Hugging Face...")
        print(f"(This may take a while for the first download...)\n")

        # Use snapshot_download with allow_patterns for efficient selective download
        kwargs = {
            "repo_id": dataset_name,
            "repo_type": "dataset",
            "local_dir": str(save_path),
            "resume_download": True,
            "force_download": False,
            "max_workers": num_workers,
        }

        if max_partitions:
            # Download only specific partitions
            partition_nums = [f"{i:06d}" for i in range(max_partitions)]
            patterns = [f"data/partition_{num}*.parquet" for num in partition_nums]
            patterns.extend(["dataset_infos.json", "README.md", "*.json"])
            kwargs["allow_patterns"] = patterns
            print(f"Downloading first {max_partitions} partitions only...\n")

        snapshot_download(**kwargs)

        print(f"\n✓ Dataset synced successfully!")
        print(f"  Location: {save_path}")

        # Count files
        parquet_files = list(save_path.glob("**/*.parquet"))
        json_files = list(save_path.glob("**/*.json"))
        print(f"  Files: {len(parquet_files)} parquet + {len(json_files)} metadata files\n")

        return True, save_path

    except ImportError:
        print("✗ huggingface_hub not installed")
        print("\nInstall it with:")
        print("  pip install huggingface_hub")
        return False, None

    except Exception as e:
        error_msg = str(e)
        print(f"✗ Download failed: {error_msg}\n")

        if "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"To authenticate with HuggingFace:")
            print(f"  1. Get a token: https://huggingface.co/settings/tokens")
            print(f"  2. Login: huggingface-cli login")
            print(f"  3. Run this script again\n")
        else:
            print(f"Manual download:")
            print(f"  https://huggingface.co/datasets/{dataset_name}\n")

        return False, None


def download_multiple_datasets(output_dir, datasets, num_workers=4, max_partitions=None):
    """Download multiple datasets"""
    print(f"\n{'='*70}")
    print(" MULTI-DATASET DOWNLOADER")
    print(f"{'='*70}")
    print(f"Datasets to download: {', '.join(datasets)}")
    print(f"Output directory: {output_dir}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}
    for dataset in datasets:
        success, path = download_dataset(dataset, output_dir, num_workers, max_partitions)
        if success:
            downloaded_paths[dataset] = path
        else:
            print(f"⚠ Failed to download {dataset}\n")

    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download multiple datasets: OpenOrca, C4, TinyStories"
    )
    parser.add_argument("--output-dir", default="/root/Ava_AI/code/data",
                       help="Output directory (default: /root/Ava_AI/code/data)")
    parser.add_argument("--datasets", nargs="+",
                       default=["allenai/c4"],
                       help="Datasets to download (default: allenai/c4)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel download workers (default: 4)")
    parser.add_argument("--max-partitions", type=int, default=1,
                       help="Download only first N partitions per dataset (default: 1 = ~10GB)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Download a single specific dataset")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    try:
        # If --dataset is specified, use that; otherwise use --datasets
        datasets = [args.dataset] if args.dataset else args.datasets

        downloaded_paths = download_multiple_datasets(
            output_dir,
            datasets,
            num_workers=args.workers,
            max_partitions=args.max_partitions
        )

        if downloaded_paths:
            print(f"\n{'='*70}")
            print(" SUMMARY")
            print(f"{'='*70}")
            print(f"Successfully downloaded {len(downloaded_paths)} dataset(s):\n")
            for dataset, path in downloaded_paths.items():
                print(f"  ✓ {dataset}")
                print(f"    → {path}\n")
            sys.exit(0)
        else:
            print(f"\n{'='*70}")
            print(" ERROR: No datasets downloaded successfully")
            print(f"{'='*70}\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Download cancelled by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
