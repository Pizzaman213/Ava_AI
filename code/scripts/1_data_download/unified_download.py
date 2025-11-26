#!/usr/bin/env python3
"""
Efficient Pizzaman21/Ava_Ai Dataset Downloader

Features:
- Parallel downloads with configurable workers
- Resume incomplete downloads
- Skip already downloaded files
- Progress tracking
- Optional selective partition download
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


def download_hf_dataset(output_dir, num_workers=4, max_partitions=None):
    """Download Pizzaman21/Ava_Ai dataset efficiently"""
    dataset_name = "Pizzaman21/Ava_Ai"
    save_path = output_dir / "Ava_Ai"

    print(f"\n{'='*70}")
    print(" EFFICIENT PIZZAMAN21/AVA_AI DATASET DOWNLOADER")
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
                print(f" Dataset already exists with {len(files)} items")
                print(f"  Using: huggingface_hub (incremental sync)")
                print()
        except:
            pass

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
            patterns.append("dataset_infos.json")
            patterns.append("README.md")
            kwargs["allow_patterns"] = patterns
            print(f"Downloading first {max_partitions} partitions only...\n")

        snapshot_download(**kwargs)

        print(f"\n Dataset synced successfully!")
        print(f"Location: {save_path}")

        # Count files
        parquet_files = list(save_path.glob("data/*.parquet"))
        json_files = list(save_path.glob("data/*.meta.json"))
        print(f"Files: {len(parquet_files)} parquet + {len(json_files)} metadata files")

        return True

    except ImportError:
        print(" huggingface_hub not installed")
        print("\nInstall it with:")
        print("  pip install huggingface_hub")
        return False

    except Exception as e:
        error_msg = str(e)
        print(f" Download failed: {error_msg}")

        if "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"\nTo authenticate with HuggingFace:")
            print(f"  1. Get a token: https://huggingface.co/settings/tokens")
            print(f"  2. Login: huggingface-cli login")
            print(f"  3. Run this script again")
        else:
            print(f"\nManual download:")
            print(f"  https://huggingface.co/datasets/{dataset_name}")

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Efficiently download Pizzaman21/Ava_Ai dataset"
    )
    parser.add_argument("--output-dir", default="/project/code/data",
                       help="Output directory (default: /project/code/data)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel download workers (default: 4)")
    parser.add_argument("--max-partitions", type=int, default=None,
                       help="Download only first N partitions (optional)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    try:
        success = download_hf_dataset(output_dir, num_workers=args.workers,
                                     max_partitions=args.max_partitions)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
