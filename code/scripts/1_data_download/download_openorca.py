#!/usr/bin/env python3
"""
OpenOrca Dataset Downloader with Custom Tokenizer

Downloads and tokenizes the OpenOrca dataset using the custom Ava tokenizer.
Handles the dataset efficiently with streaming and batched tokenization.

Features:
- Downloads OpenOrca dataset from HuggingFace
- Uses custom Ava tokenizer for tokenization
- Saves tokenized data in Arrow/Parquet format
- Configurable batch size and max samples
- Progress tracking
- Resume support

Usage:
    # Download and tokenize full dataset
    python download_openorca.py

    # Download with limited samples (for testing)
    python download_openorca.py --max-samples 10000

    # Specify custom output directory
    python download_openorca.py --output-dir /path/to/output

    # Use different batch size for tokenization
    python download_openorca.py --batch-size 1000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def format_openorca_conversation(example: Dict) -> str:
    """
    Format OpenOrca example into a conversation format.

    OpenOrca format:
    - system_prompt: System instructions
    - question: User query
    - response: Assistant response

    Output format:
    System: {system_prompt}
    User: {question}
    Assistant: {response}
    """
    system = example.get("system_prompt", "").strip()
    question = example.get("question", "").strip()
    response = example.get("response", "").strip()

    # Create conversation format
    parts = []
    if system:
        parts.append(f"System: {system}")
    parts.append(f"User: {question}")
    parts.append(f"Assistant: {response}")

    return "\n".join(parts)


def tokenize_batch(
    texts: List[str],
    tokenizer,
    max_length: int = 2048,
) -> Dict[str, List]:
    """Tokenize a batch of texts."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,  # Return lists
        return_attention_mask=True,
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }


def download_and_tokenize_openorca(
    output_dir: Path,
    tokenizer_path: Path,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    max_length: int = 2048,
    split: str = "train",
    chunk_size: int = 50000,  # Write to disk every 50k samples
) -> None:
    """
    Download OpenOrca dataset and tokenize with custom tokenizer.

    Uses chunked writing to avoid memory issues with large datasets.

    Args:
        output_dir: Directory to save tokenized data
        tokenizer_path: Path to custom tokenizer
        max_samples: Maximum number of samples to process (None = all)
        batch_size: Batch size for tokenization
        max_length: Maximum sequence length
        split: Dataset split to download (train/validation)
        chunk_size: Write to disk every N samples (to avoid OOM)
    """

    print("\n" + "=" * 80)
    print("  OPENORCA DATASET DOWNLOADER WITH CUSTOM TOKENIZER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: Open-Orca/OpenOrca")
    print(f"  Split: {split}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Output: {output_dir}")
    print(f"  Max samples: {max_samples if max_samples else 'All (4.2M)'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Chunk size: {chunk_size:,} (write to disk)")
    print(f"  Max length: {max_length}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_dir = output_dir / "arrow"
    arrow_dir.mkdir(exist_ok=True)

    # Load custom tokenizer
    print("Loading custom tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"  ✓ Loaded tokenizer (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"  ✗ Failed to load tokenizer: {e}")
        sys.exit(1)

    # Download dataset
    print(f"\nDownloading OpenOrca dataset...")
    print("(This may take several minutes on first download...)")

    try:
        # Load dataset with streaming for memory efficiency
        dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=split,
            streaming=True,  # Stream to avoid loading all into memory
        )
        print("  ✓ Dataset loaded")
    except Exception as e:
        print(f"  ✗ Failed to download dataset: {e}")
        sys.exit(1)

    # Process dataset in batches with chunked writing
    print(f"\nTokenizing dataset with chunked writing...")
    print(f"(Writing to disk every {chunk_size:,} samples to avoid memory issues)\n")

    text_batch = []
    chunk_input_ids = []
    chunk_attention_masks = []
    processed = 0
    chunk_count = 0
    chunk_files = []

    # Create progress bar
    pbar = tqdm(
        total=max_samples if max_samples else 4_233_923,
        desc="Processing",
        unit="samples",
    )

    try:
        for example in dataset:
            # Format conversation
            text = format_openorca_conversation(example)
            text_batch.append(text)

            # Process batch when full
            if len(text_batch) >= batch_size:
                tokenized = tokenize_batch(text_batch, tokenizer, max_length)
                chunk_input_ids.extend(tokenized["input_ids"])
                chunk_attention_masks.extend(tokenized["attention_mask"])

                processed += len(text_batch)
                pbar.update(len(text_batch))
                text_batch = []

                # Write chunk to disk when reaching chunk_size
                if len(chunk_input_ids) >= chunk_size:
                    chunk_file = arrow_dir / f"chunk_{chunk_count:04d}.parquet"
                    _write_chunk(
                        chunk_file,
                        chunk_input_ids,
                        chunk_attention_masks,
                    )
                    chunk_files.append(chunk_file)
                    chunk_count += 1

                    # Clear memory
                    chunk_input_ids = []
                    chunk_attention_masks = []

                # Check if reached max samples
                if max_samples and processed >= max_samples:
                    break

        # Process remaining text batch
        if text_batch:
            tokenized = tokenize_batch(text_batch, tokenizer, max_length)
            chunk_input_ids.extend(tokenized["input_ids"])
            chunk_attention_masks.extend(tokenized["attention_mask"])
            processed += len(text_batch)
            pbar.update(len(text_batch))

        # Write final chunk
        if chunk_input_ids:
            chunk_file = arrow_dir / f"chunk_{chunk_count:04d}.parquet"
            _write_chunk(
                chunk_file,
                chunk_input_ids,
                chunk_attention_masks,
            )
            chunk_files.append(chunk_file)

        pbar.close()

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        pbar.close()

        # Save partial data if any
        if chunk_input_ids:
            print(f"Saving partial chunk ({len(chunk_input_ids)} samples)...")
            chunk_file = arrow_dir / f"chunk_{chunk_count:04d}.parquet"
            _write_chunk(
                chunk_file,
                chunk_input_ids,
                chunk_attention_masks,
            )
            chunk_files.append(chunk_file)

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in chunk_files) / (1024 * 1024)

    print(f"\n" + "=" * 80)
    print("  TOKENIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput:")
    print(f"  Directory: {arrow_dir}")
    print(f"  Chunks: {len(chunk_files)} files")
    print(f"  Total samples: {processed:,}")
    print(f"  Total size: {total_size_mb:.1f} MB")

    # Save metadata
    metadata = {
        "dataset": "Open-Orca/OpenOrca",
        "split": split,
        "tokenizer": str(tokenizer_path),
        "samples": processed,
        "chunks": len(chunk_files),
        "max_length": max_length,
        "vocab_size": tokenizer.vocab_size,
    }

    metadata_file = output_dir / "dataset_info.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {metadata_file}")

    print(f"\nTo use in training, set in your config:")
    print(f"  data:")
    print(f"    data_dir: {arrow_dir}")
    print()


def _write_chunk(
    output_file: Path,
    input_ids: List,
    attention_masks: List,
) -> None:
    """Write a chunk of tokenized data to disk."""
    table = pa.table({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    })
    pq.write_table(table, output_file, compression="snappy")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and tokenize OpenOrca with custom tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full dataset (4.2M samples)
  python download_openorca.py

  # Download limited samples for testing
  python download_openorca.py --max-samples 10000

  # Custom output directory
  python download_openorca.py --output-dir /custom/path

  # Larger batches for faster processing (if memory allows)
  python download_openorca.py --batch-size 2000
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/Ava_AI/code/data/fine-tuning/OpenOrca",
        help="Output directory for tokenized data (default: /root/Ava_AI/code/data/fine-tuning/OpenOrca)",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/root/Ava_AI/code/data/Ava_Ai/tokenizer",
        help="Path to custom tokenizer (default: /root/Ava_AI/code/data/Ava_Ai/tokenizer)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None = all 4.2M samples)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for tokenization (default: 1000)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset split to download (default: train)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Write to disk every N samples to avoid OOM (default: 50000)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Convert paths
    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer_path)

    # Validate tokenizer exists
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("\nExpected tokenizer files:")
        print("  - tokenizer.json")
        print("  - tokenizer_config.json")
        print("  - special_tokens_map.json")
        sys.exit(1)

    # Run download and tokenization
    download_and_tokenize_openorca(
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        split=args.split,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
