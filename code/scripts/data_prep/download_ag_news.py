#!/usr/bin/env python3
"""
Download AG News dataset only to a separate folder.
AG News is a news classification dataset with 4 classes.
"""

import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_ag_news(output_dir: str = "/project/code/data/ag_news/raw"):
    """Download AG News dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading AG News Dataset")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print()

    # Check if already downloaded
    if (output_path / "train").exists() and any((output_path / "train").iterdir()):
        print("AG News dataset already exists. Skipping download.")
        print("To re-download, delete the folder and run again.")
        return

    try:
        # Load AG News dataset
        print("Loading AG News from HuggingFace...")
        dataset = load_dataset("ag_news", trust_remote_code=True)

        print(f"\nDataset info:")
        print(f"  Train samples: {len(dataset['train']):,}")
        print(f"  Test samples: {len(dataset['test']):,}")
        print(f"  Total: {len(dataset['train']) + len(dataset['test']):,}")
        print()

        # Save splits to separate folders
        for split_name in ['train', 'test']:
            split_path = output_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            split_data = dataset[split_name]
            print(f"Saving {split_name} split...")

            # Save in batches as JSONL files
            batch_size = 10000
            batch_num = 0
            batch = []

            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                batch.append(sample)

                if len(batch) >= batch_size or idx == len(split_data) - 1:
                    # Save batch
                    batch_file = split_path / f"batch_{batch_num:04d}.jsonl"
                    with open(batch_file, 'w') as f:
                        for item in batch:
                            f.write(json.dumps(item) + '\n')

                    batch = []
                    batch_num += 1

            # Create summary
            summary = {
                "split": split_name,
                "total_samples": len(split_data),
                "num_batches": batch_num,
                "batch_size": batch_size
            }

            with open(split_path / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"  Saved {len(split_data):,} samples in {batch_num} batches")

        # Save dataset info
        info = {
            "dataset_name": "ag_news",
            "description": "AG News classification dataset with 4 classes",
            "classes": {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Sci/Tech"
            },
            "total_samples": len(dataset['train']) + len(dataset['test']),
            "train_samples": len(dataset['train']),
            "test_samples": len(dataset['test']),
            "fields": ["text", "label"],
            "source": "https://huggingface.co/datasets/ag_news"
        }

        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        print()
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Location: {output_path}")
        print(f"Train samples: {len(dataset['train']):,}")
        print(f"Test samples: {len(dataset['test']):,}")
        print()

    except Exception as e:
        print(f"\nError downloading AG News: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    download_ag_news()
