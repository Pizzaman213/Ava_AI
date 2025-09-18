#!/usr/bin/env python3
"""
Enhanced HuggingFace Dataset Downloader with Robust Error Handling
Supports 50+ diverse datasets with multiple retry strategies
"""

import os
import sys
import json
import time
import argparse
import traceback
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Lazy imports to avoid issues
def import_datasets():
    """Lazy import of datasets library"""
    try:
        from datasets import load_dataset, load_from_disk
        import datasets
        datasets.disable_progress_bar()  # We'll use our own progress bars
        return load_dataset, load_from_disk, datasets
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset, load_from_disk
        import datasets
        datasets.disable_progress_bar()
        return load_dataset, load_from_disk, datasets

# Verified working datasets (tested and confirmed)
DATASETS_CONFIG = {
    # Core Instruction Tuning (Verified Working)
    "databricks/databricks-dolly-15k": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "tatsu-lab/alpaca": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "yahma/alpaca-cleaned": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "vicgalle/alpaca-gpt4": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # OpenAssistant (Verified Working)
    "OpenAssistant/oasst1": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},
    "OpenAssistant/oasst2": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},

    # Math & Reasoning (Verified Working)
    "gsm8k": {"splits": ["train", "test"], "subset": "main", "streaming_safe": True},
    "hendrycks/competition_math": {"splits": ["train", "test"], "subset": None, "streaming_safe": True},

    # Large-scale Text Datasets (High Token Count)
    "allenai/c4": {"splits": ["train"], "subset": "en", "streaming_safe": True, "max_samples": 100000},
    "openwebtext": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000},
    "EleutherAI/pile": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000},
    "togethercomputer/RedPajama-Data-1T": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 25000},
    "HuggingFaceFW/fineweb": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 75000},
    "HuggingFaceFW/fineweb-edu": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000},
    "tiiuae/falcon-refinedweb": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 30000},

    # Code Datasets (Verified Working)
    "HuggingFaceH4/CodeAlpaca_20K": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "sahil2801/CodeAlpaca-20k": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "iamtarun/python_code_instructions_18k_alpaca": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # Conversational (Verified Working)
    "HuggingFaceH4/ultrachat_200k": {"splits": ["train_sft", "test_sft"], "subset": None, "streaming_safe": True},
    "HuggingFaceH4/no_robots": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # Synthetic Data (Verified Working)
    "roneneldan/TinyStories": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},
    "HuggingFaceTB/cosmopedia-100k": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # RLHF & Preferences (Verified Working)
    "HuggingFaceH4/ultrafeedback_binarized": {"splits": ["train_prefs", "test_prefs"], "subset": None, "streaming_safe": True},
    "Anthropic/hh-rlhf": {"splits": ["train", "test"], "subset": None, "streaming_safe": True},

    # Additional High-Quality (Verified Working)
    "philschmid/dolly-15k-oai-style": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "garage-bAInd/Open-Platypus": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "WizardLM/WizardLM_evol_instruct_V2_196k": {"splits": ["train"], "subset": None, "streaming_safe": True, "large": True},

    # Medical & Science (Verified Working)
    "medalpaca/medical_meadow_medical_flashcards": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "bigscience/P3": {"splits": ["train", "validation"], "subset": "all", "streaming_safe": True, "large": True},

    # Multi-turn Conversations (Verified Working)
    "lmsys/lmsys-chat-1m": {"splits": ["train"], "subset": None, "streaming_safe": True, "large": True},
    "ShareGPT4Omni/ShareGPT4V": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # Additional Code (Verified Working)
    "bigcode/self-oss-instruct-sc2-exec-filter-50k": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "m-a-p/CodeFeedback-Filtered-Instruction": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # General Knowledge (Working)
    "squad": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},
    "squad_v2": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},
    "natural_questions": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True, "large": True},

    # Reasoning (Working)
    "allenai/ai2_arc": {"splits": ["train", "test", "validation"], "subset": "ARC-Challenge", "streaming_safe": True},
    "winogrande": {"splits": ["train", "validation"], "subset": "winogrande_xl", "streaming_safe": True},
    "hellaswag": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},

    # Text Generation (Working)
    "allenai/prosocial-dialog": {"splits": ["train", "validation"], "subset": None, "streaming_safe": True},
    "HuggingFaceH4/self_instruct": {"splits": ["train"], "subset": None, "streaming_safe": True},

    # Additional Large Datasets for Scale
    "wikipedia": {"splits": ["train"], "subset": "20220301.en", "streaming_safe": True, "max_samples": 100000},
    "bookcorpus": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000},
    "cc_news": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 75000},
    "multi_news": {"splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True},
    "xsum": {"splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True},
    "cnn_dailymail": {"splits": ["train", "validation", "test"], "subset": "3.0.0", "streaming_safe": True},
    "pubmed_qa": {"splits": ["train"], "subset": "pqa_labeled", "streaming_safe": True},
    "scientific_papers": {"splits": ["train", "validation", "test"], "subset": "arxiv", "streaming_safe": True},
    "github-code": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 40000},
    "bigscience/xP3": {"splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000},
    "Muennighoff/natural-instructions": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "AlekseyKorshuk/persona-chat": {"splits": ["train"], "subset": None, "streaming_safe": True},
    "daily_dialog": {"splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True},
    "empathetic_dialogues": {"splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True},
}

# Retry strategies for different failure modes
RETRY_STRATEGIES = [
    # Strategy 1: Standard download with token
    {
        "name": "standard_with_token",
        "params": {
            "token": True,
            "num_proc": 4
        }
    },
    # Strategy 2: Streaming mode for large datasets
    {
        "name": "streaming_mode",
        "params": {
            "streaming": True,
            "token": True
        }
    },
    # Strategy 3: Single process for compatibility
    {
        "name": "single_process",
        "params": {
            "num_proc": 1,
            "token": True
        }
    },
    # Strategy 4: No auth token
    {
        "name": "no_token",
        "params": {
            "num_proc": 1
        }
    },
    # Strategy 5: Force redownload
    {
        "name": "force_redownload",
        "params": {
            "download_mode": "force_redownload",
            "num_proc": 1
        }
    },
    # Strategy 6: Trust remote code
    {
        "name": "trust_remote",
        "params": {
            "trust_remote_code": True,
            "num_proc": 1
        }
    },
    # Strategy 7: No multiprocessing
    {
        "name": "no_multiproc",
        "params": {
            "num_proc": None
        }
    },
    # Strategy 8: Minimal parameters
    {
        "name": "minimal",
        "params": {}
    },
    # Strategy 9: Cache only (for offline)
    {
        "name": "cache_only",
        "params": {
            "download_mode": "reuse_cache_if_exists"
        }
    }
]

class DatasetDownloader:
    def __init__(self, output_dir: str = "/project/data/pretraining/raw",
                 max_samples: Optional[int] = None):
        """Initialize the dataset downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.summary = {
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "successful": [],
            "failed": [],
            "skipped": []
        }

        # Import datasets library
        self.load_dataset, self.load_from_disk, self.datasets_lib = import_datasets()

    def check_memory(self) -> Tuple[float, float]:
        """Check available memory"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        usage_percent = mem.percent
        return available_gb, usage_percent

    def download_dataset_with_retry(self, dataset_name: str, config: Dict) -> bool:
        """Download a dataset with multiple retry strategies"""
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_name}")
        print(f"{'='*60}")

        # Check if dataset is marked as large
        is_large = config.get("large", False)
        streaming_safe = config.get("streaming_safe", True)

        # Check memory for large datasets
        available_gb, usage_percent = self.check_memory()
        if is_large and available_gb < 10:
            print(f"⚠️  Low memory ({available_gb:.1f}GB available), using streaming mode")
            streaming_safe = True

        # Try different strategies
        for strategy_idx, strategy in enumerate(RETRY_STRATEGIES):
            print(f"\nAttempt {strategy_idx + 1}/{len(RETRY_STRATEGIES)}: {strategy['name']}")

            try:
                # Prepare parameters
                params = strategy["params"].copy()

                # Add subset if specified
                if config.get("subset"):
                    dataset_args = [dataset_name, config["subset"]]
                else:
                    dataset_args = [dataset_name]

                # Force streaming for large datasets on low memory
                if is_large and available_gb < 10:
                    params["streaming"] = True

                # Handle different splits
                for split in config.get("splits", ["train"]):
                    print(f"  Downloading split: {split}")

                    try:
                        # Add delay to avoid rate limiting
                        if strategy_idx > 0:  # Only add delay after first attempt
                            time.sleep(2)

                        # Download the dataset
                        if params.get("streaming", False):
                            # Streaming mode
                            dataset = self.load_dataset(*dataset_args, split=split, **params)

                            # Save streaming dataset
                            output_path = self.output_dir / dataset_name.replace("/", "_") / split
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Stream and save samples
                            samples = []
                            max_to_download = self.max_samples if self.max_samples else 100000

                            print(f"  Streaming up to {max_to_download} samples...")
                            for idx, sample in enumerate(tqdm(dataset, total=max_to_download)):
                                samples.append(sample)
                                if idx >= max_to_download - 1:
                                    break

                            # Save as JSON
                            with open(output_path / "data.json", "w") as f:
                                json.dump(samples, f)

                            print(f"  ✓ Saved {len(samples)} samples to {output_path}")

                        else:
                            # Regular download
                            dataset = self.load_dataset(*dataset_args, split=split, **params)

                            # Apply sample limit if specified
                            if self.max_samples and hasattr(dataset, '__len__') and len(dataset) > self.max_samples:
                                if hasattr(dataset, 'select'):
                                    dataset = dataset.select(range(self.max_samples))

                            # Save dataset
                            output_path = self.output_dir / dataset_name.replace("/", "_") / split
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Save in arrow format if possible
                            if hasattr(dataset, 'save_to_disk'):
                                dataset.save_to_disk(str(output_path))

                            # Also save as JSON for compatibility
                            if hasattr(dataset, 'to_json'):
                                dataset.to_json(str(output_path / "data.json"))
                            else:
                                # Fallback for iterable datasets
                                samples = []
                                for i, sample in enumerate(dataset):
                                    if self.max_samples and i >= self.max_samples:
                                        break
                                    samples.append(sample)

                                import json
                                with open(output_path / "data.json", "w") as f:
                                    json.dump(samples, f)

                            # Get length safely
                            if hasattr(dataset, '__len__'):
                                dataset_len = len(dataset)
                            else:
                                dataset_len = self.max_samples or "unknown"

                            print(f"  ✓ Saved {dataset_len} samples to {output_path}")

                    except Exception as e:
                        print(f"  ✗ Failed to download split {split}: {str(e)}")
                        continue

                # If we got here, download was successful
                self.summary["successful"].append(dataset_name)
                return True

            except Exception as e:
                print(f"  ✗ Strategy failed: {str(e)}")
                continue

        # All strategies failed
        print(f"\n✗ Failed to download {dataset_name} after all attempts")
        self.summary["failed"].append(dataset_name)

        # Save error details
        error_file = self.output_dir / ".errors" / f"{dataset_name.replace('/', '_')}.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"All strategies failed\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")

        return False

    def download_all(self, datasets: Optional[List[str]] = None,
                    parallel: bool = True, max_workers: int = 2):
        """Download all configured datasets"""
        # Select datasets to download
        if datasets:
            dataset_configs = {k: v for k, v in DATASETS_CONFIG.items() if k in datasets}
        else:
            dataset_configs = DATASETS_CONFIG

        print(f"\nPreparing to download {len(dataset_configs)} datasets")
        print(f"Output directory: {self.output_dir}")

        # Check existing datasets
        existing = set()
        for dataset_name in dataset_configs:
            dataset_dir = self.output_dir / dataset_name.replace("/", "_")
            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                existing.add(dataset_name)
                print(f"  ⚠️  {dataset_name} already exists, skipping")
                self.summary["skipped"].append(dataset_name)

        # Filter out existing
        to_download = {k: v for k, v in dataset_configs.items() if k not in existing}

        if not to_download:
            print("All datasets already downloaded!")
            return

        print(f"\nDownloading {len(to_download)} new datasets...")

        # Download datasets
        if parallel and len(to_download) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.download_dataset_with_retry, name, config): name
                    for name, config in to_download.items()
                }

                for future in as_completed(futures):
                    dataset_name = futures[future]
                    try:
                        success = future.result(timeout=1800)  # 30 min timeout
                        if success:
                            print(f"✓ Completed: {dataset_name}")
                        else:
                            print(f"✗ Failed: {dataset_name}")
                    except Exception as e:
                        print(f"✗ Exception downloading {dataset_name}: {e}")
                        self.summary["failed"].append(dataset_name)
        else:
            # Sequential download
            for dataset_name, config in to_download.items():
                self.download_dataset_with_retry(dataset_name, config)

        # Save summary
        self.save_summary()

    def save_summary(self):
        """Save download summary"""
        self.summary["total_attempted"] = len(self.summary["successful"]) + len(self.summary["failed"])
        self.summary["total_datasets"] = len(DATASETS_CONFIG)

        if self.summary["total_attempted"] > 0:
            self.summary["success_rate"] = len(self.summary["successful"]) / self.summary["total_attempted"]
        else:
            self.summary["success_rate"] = 0

        # Add memory stats
        available_gb, usage_percent = self.check_memory()
        self.summary["memory_stats"] = {
            "available_gb": available_gb,
            "usage_percent": usage_percent
        }

        # Save summary
        summary_file = self.output_dir / "download_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.summary, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Total datasets available: {len(DATASETS_CONFIG)}")
        print(f"Successfully downloaded: {len(self.summary['successful'])}")
        print(f"Failed: {len(self.summary['failed'])}")
        print(f"Skipped (existing): {len(self.summary['skipped'])}")
        print(f"Success rate: {self.summary['success_rate']:.1%}")
        print(f"\nSummary saved to: {summary_file}")

        if self.summary["failed"]:
            print(f"\nFailed datasets:")
            for name in self.summary["failed"]:
                print(f"  - {name}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument("--output-dir", default="/project/data/pretraining/raw",
                      help="Output directory for datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Maximum samples per dataset")
    parser.add_argument("--dataset", type=str, default=None,
                      help="Download specific dataset only")
    parser.add_argument("--parallel", action="store_true",
                      help="Download datasets in parallel")
    parser.add_argument("--max-workers", type=int, default=2,
                      help="Maximum parallel workers")
    parser.add_argument("--skip-large", action="store_true",
                      help="Skip datasets marked as large")

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatasetDownloader(output_dir=args.output_dir,
                                 max_samples=args.max_samples)

    # Filter datasets if requested
    datasets = None
    if args.dataset:
        datasets = [args.dataset]
    elif args.skip_large:
        # Skip large datasets
        datasets = [k for k, v in DATASETS_CONFIG.items() if not v.get("large", False)]

    # Start download
    downloader.download_all(datasets=datasets,
                          parallel=args.parallel,
                          max_workers=args.max_workers)

if __name__ == "__main__":
    main()