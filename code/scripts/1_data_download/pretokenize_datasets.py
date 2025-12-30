#!/usr/bin/env python3
"""
Advanced pre-tokenization script for all datasets.

Features:
- Automatically detects and tokenizes all datasets in data folder
- Supports multiple formats: parquet, json, jsonl, arrow, csv
- Batch processing for memory efficiency
- Progress tracking and error recovery
- Uses 16k tokenizer by default
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

def auto_install_requirements():
    """Auto-install requirements if needed"""
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

try:
    import torch
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Installing dependencies...")
    auto_install_requirements()
    import torch
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer

import argparse


class DatasetTokenizer:
    """Tokenizes datasets intelligently based on format and structure"""

    def __init__(self, tokenizer_path: str):
        """Initialize with tokenizer"""
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.stats = {}

    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from path"""
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}\n")
        return tokenizer

    def _detect_text_columns(self, dataset) -> List[str]:
        """Detect which columns contain text"""
        if not hasattr(dataset, 'column_names'):
            return ["text"]

        text_columns = []
        for col in dataset.column_names:
            if col in ["text", "content", "data", "input", "instruction", "output", "completion"]:
                text_columns.append(col)

        if not text_columns:
            # Pick first string column
            for col in dataset.column_names:
                try:
                    if isinstance(dataset[0][col], str):
                        text_columns.append(col)
                        break
                except:
                    pass

        return text_columns if text_columns else ["text"]

    def _tokenize_batch(self, examples: Dict, text_columns: List[str]) -> Dict:
        """Tokenize a batch of examples"""
        texts = []

        for col in text_columns:
            if col in examples:
                col_texts = examples[col]
                if isinstance(col_texts, str):
                    col_texts = [col_texts]
                texts.extend(col_texts)

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        encodings = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=True
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings.get("attention_mask", []),
        }

    def tokenize_dataset(self, dataset_path: Path, output_path: Path, dataset_name: str = "dataset") -> Tuple[bool, Optional[int]]:
        """
        Load, tokenize, and save dataset

        Returns: (success, num_examples)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f" TOKENIZING: {dataset_name}")
        print(f"{'='*70}")
        print(f"Input:  {dataset_path}")
        print(f"Output: {output_path}\n")

        try:
            # Load dataset
            print(f"Loading dataset...")
            dataset = self._load_dataset(dataset_path)

            if dataset is None:
                print(f"✗ Could not load dataset from {dataset_path}")
                return False, None

            # Handle split datasets
            if isinstance(dataset, dict) and "train" in dataset:
                dataset = dataset["train"]
                print(f"✓ Using 'train' split")

            print(f"✓ Dataset loaded")
            print(f"  Columns: {dataset.column_names}")
            print(f"  Examples: {len(dataset):,}\n")

            # Detect text columns
            text_columns = self._detect_text_columns(dataset)
            print(f"  Text columns: {', '.join(text_columns)}")
            print(f"  Batch size: 32\n")

            # Tokenize
            print(f"Tokenizing dataset...")
            tokenized = dataset.map(
                lambda x: self._tokenize_batch(x, text_columns),
                batched=True,
                batch_size=32,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )

            # Save
            print(f"\nSaving to disk...")
            tokenized.save_to_disk(str(output_path))

            num_examples = len(tokenized)
            print(f"✓ Tokenization complete!")
            print(f"  Examples: {num_examples:,}")
            print(f"  Location: {output_path}\n")

            return True, num_examples

        except Exception as e:
            print(f"✗ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return False, None

    def _load_dataset(self, dataset_path: Path) -> Optional[Dataset]:
        """Load dataset from any format"""
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            return None

        # Try different formats
        try:
            # Check for parquet files
            parquet_files = list(dataset_path.glob("**/*.parquet"))
            if parquet_files:
                print(f"  Format: Parquet ({len(parquet_files)} files)")
                return load_dataset("parquet", data_dir=str(dataset_path))
        except Exception as e:
            print(f"  Parquet load failed: {e}")

        try:
            # Check for JSON files
            json_files = list(dataset_path.glob("**/*.json"))
            if json_files:
                print(f"  Format: JSON ({len(json_files)} files)")
                return load_dataset("json", data_dir=str(dataset_path))
        except Exception as e:
            print(f"  JSON load failed: {e}")

        try:
            # Check for JSONL files
            jsonl_files = list(dataset_path.glob("**/*.jsonl"))
            if jsonl_files:
                print(f"  Format: JSONL ({len(jsonl_files)} files)")
                return load_dataset("json", data_files=[str(f) for f in jsonl_files])
        except Exception as e:
            print(f"  JSONL load failed: {e}")

        try:
            # Check for CSV files
            csv_files = list(dataset_path.glob("**/*.csv"))
            if csv_files:
                print(f"  Format: CSV ({len(csv_files)} files)")
                return load_dataset("csv", data_dir=str(dataset_path))
        except Exception as e:
            print(f"  CSV load failed: {e}")

        try:
            # Try Arrow format
            arrow_files = list(dataset_path.glob("**/*.arrow"))
            if arrow_files:
                print(f"  Format: Arrow ({len(arrow_files)} files)")
                return load_dataset("arrow", data_dir=str(dataset_path))
        except Exception as e:
            print(f"  Arrow load failed: {e}")

        return None


def find_datasets(data_dir: Path) -> List[Tuple[Path, str]]:
    """Find all dataset directories to tokenize"""
    datasets = []

    # Skip these directories
    skip_dirs = {"Ava_Ai", "models", ".cache", "__pycache__"}

    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        if item.name in skip_dirs:
            continue

        # Skip already tokenized datasets
        if item.name.endswith("-tokenized"):
            continue

        # Check if directory contains data files
        has_data = False
        for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv", "*.arrow"]:
            if list(item.glob(f"**/{ext}")):
                has_data = True
                break

        if has_data:
            # Check if already tokenized
            tokenized_path = item.parent / f"{item.name}-tokenized"
            if not tokenized_path.exists():
                datasets.append((item, item.name))

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize all datasets with the 16k tokenizer"
    )
    parser.add_argument("--data-dir", default="/root/Ava_AI/code/data",
                       help="Data directory containing datasets (default: /root/Ava_AI/code/data)")
    parser.add_argument("--dataset-path", type=str,
                       help="Tokenize single dataset (optional)")
    parser.add_argument("--output-path", type=str,
                       help="Output path for single dataset (required with --dataset-path)")
    parser.add_argument("--tokenizer-path",
                       default="/root/Ava_AI/code/data/Ava_Ai/tokenizer_16k",
                       help="Path to tokenizer (default: 16k tokenizer)")

    args = parser.parse_args()

    tokenizer = DatasetTokenizer(args.tokenizer_path)
    results = {}

    if args.dataset_path:
        # Single dataset mode
        if not args.output_path:
            print("Error: --output-path is required with --dataset-path")
            sys.exit(1)

        success, num_examples = tokenizer.tokenize_dataset(
            Path(args.dataset_path),
            Path(args.output_path),
            Path(args.dataset_path).name
        )
        sys.exit(0 if success else 1)

    else:
        # Auto-detect mode
        data_dir = Path(args.data_dir)
        datasets = find_datasets(data_dir)

        if not datasets:
            print(f"No datasets found in {data_dir}")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(" AUTO-TOKENIZATION MODE")
        print(f"{'='*70}")
        print(f"Found {len(datasets)} dataset(s) to tokenize:\n")
        for path, name in datasets:
            print(f"  • {name}")
        print()

        # Tokenize each dataset
        for dataset_path, dataset_name in datasets:
            output_path = dataset_path.parent / f"{dataset_name}-tokenized"
            success, num_examples = tokenizer.tokenize_dataset(
                dataset_path,
                output_path,
                dataset_name
            )
            results[dataset_name] = (success, num_examples)

        # Print summary
        print(f"\n{'='*70}")
        print(" SUMMARY")
        print(f"{'='*70}\n")

        success_count = sum(1 for s, _ in results.values() if s)
        total_examples = sum(n for _, n in results.values() if n is not None)

        for dataset_name, (success, num_examples) in results.items():
            status = "✓" if success else "✗"
            examples_str = f"{num_examples:,}" if num_examples else "N/A"
            print(f"  {status} {dataset_name}: {examples_str} examples")

        print(f"\nSuccessfully tokenized: {success_count}/{len(results)}")
        print(f"Total examples: {total_examples:,}\n")

        sys.exit(0 if success_count == len(results) else 1)


if __name__ == "__main__":
    main()
