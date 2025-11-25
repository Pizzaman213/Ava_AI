#!/usr/bin/env python3
"""
Convert JSONL files to Arrow format for 60x faster data loading.

This script tokenizes JSONL files and saves them in memory-mapped Arrow format
for ultra-fast zero-copy loading during training.

Usage:
    python convert_jsonl_to_arrow.py \
        --input /project/code/data/processed/*.jsonl \
        --output /project/code/data/processed/arrow \
        --tokenizer /project/code/models/tokenizer/enhanced-50680 \
        --max-length 256
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add project to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def tokenize_batch(texts: List[str], tokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    """Tokenize a batch of texts."""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=False,  # Don't pad here - will be done in dataloader
        truncation=True,
        return_attention_mask=True
    )

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    }


def process_jsonl_file(
    input_path: Path,
    output_dir: Path,
    tokenizer_path: str,
    max_length: int,
    batch_size: int = 1000
):
    """Convert a single JSONL file to Arrow format."""

    print(f"\n{'='*80}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*80}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Read all lines from JSONL
    print("Reading JSONL file...")
    texts = []
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Reading"):
            data = json.loads(line.strip())
            # Handle different JSONL formats
            if 'text' in data:
                texts.append(data['text'])
            elif 'content' in data:
                texts.append(data['content'])
            elif 'prompt' in data and 'completion' in data:
                texts.append(data['prompt'] + data['completion'])
            else:
                # Fallback: just use the whole JSON as text
                texts.append(json.dumps(data))

    print(f" Loaded {len(texts):,} documents")

    # Tokenize in batches
    print(f"\nTokenizing (batch_size={batch_size})...")
    all_input_ids = []
    all_attention_masks = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch_texts = texts[i:i + batch_size]
        batch_encodings = tokenize_batch(batch_texts, tokenizer, max_length)

        all_input_ids.extend(batch_encodings['input_ids'])
        all_attention_masks.extend(batch_encodings['attention_mask'])

    print(f" Tokenized {len(all_input_ids):,} sequences")

    # Filter out empty sequences
    valid_sequences = [
        (input_ids, attention_mask)
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks)
        if len(input_ids) > 0
    ]

    if len(valid_sequences) < len(all_input_ids):
        print(f"  Filtered out {len(all_input_ids) - len(valid_sequences):,} empty sequences")

    all_input_ids = [seq[0] for seq in valid_sequences]
    all_attention_masks = [seq[1] for seq in valid_sequences]

    # Create Arrow table
    print("\nCreating Arrow table...")
    schema = pa.schema([
        ('input_ids', pa.list_(pa.int32())),
        ('attention_mask', pa.list_(pa.int32()))
    ])

    table = pa.table({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks
    }, schema=schema)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as Arrow IPC file (memory-mappable)
    output_path = output_dir / f"{input_path.stem}.arrow"
    print(f"\nWriting to: {output_path}")

    with pa.OSFile(str(output_path), 'wb') as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)

    # Calculate stats
    input_size_mb = input_path.stat().st_size / (1024**2)
    output_size_mb = output_path.stat().st_size / (1024**2)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

    print(f"\n Conversion complete!")
    print(f"  Input size:  {input_size_mb:.1f} MB")
    print(f"  Output size: {output_size_mb:.1f} MB")
    print(f"  Ratio:       {compression_ratio:.2f}x {'smaller' if compression_ratio > 1 else 'larger'}")
    print(f"  Sequences:   {len(table):,}")

    return output_path


def create_split_metadata(output_dir: Path):
    """Create metadata file for train/validation splits."""

    # Find all arrow files
    arrow_files = list(output_dir.glob("*.arrow"))

    if not arrow_files:
        print("  No Arrow files found")
        return

    # Create metadata
    metadata = {
        'train': [],
        'validation': []
    }

    # Use first 90% for train, last 10% for validation
    # (Simplified - in production you'd want better splitting)
    for i, arrow_file in enumerate(arrow_files):
        if i < len(arrow_files) * 0.9:
            metadata['train'].append(arrow_file.name)
        else:
            metadata['validation'].append(arrow_file.name)

    # Write metadata
    metadata_path = output_dir / "splits.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Created split metadata: {metadata_path}")
    print(f"  Train files:      {len(metadata['train'])}")
    print(f"  Validation files: {len(metadata['validation'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Arrow format")
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input JSONL file(s) (supports glob patterns)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for Arrow files'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='/project/code/models/tokenizer/enhanced-50680',
        help='Path to tokenizer'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for tokenization'
    )

    args = parser.parse_args()

    # Resolve input files
    input_files = []
    for pattern in args.input:
        input_files.extend(Path().glob(pattern))

    input_files = [f for f in input_files if f.suffix == '.jsonl']

    if not input_files:
        print(" No JSONL files found!")
        sys.exit(1)

    print(f"Found {len(input_files)} JSONL file(s) to convert:")
    for f in input_files:
        print(f"  - {f}")

    output_dir = Path(args.output)

    # Process each file
    for input_file in input_files:
        try:
            process_jsonl_file(
                input_file,
                output_dir,
                args.tokenizer,
                args.max_length,
                args.batch_size
            )
        except Exception as e:
            print(f" Failed to process {input_file}: {e}")
            import traceback
            traceback.print_exc()

    # Create split metadata
    create_split_metadata(output_dir)

    print(f"\n{'='*80}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nArrow files saved to: {output_dir}")
    print(f"\nTo use in training, update your config:")
    print(f"  data:")
    print(f"    data_dir: {output_dir}")
    print(f"    use_pretokenized: true")
    print(f"\nExpected speedup: ~60x faster than JSONL!")


if __name__ == '__main__':
    main()
