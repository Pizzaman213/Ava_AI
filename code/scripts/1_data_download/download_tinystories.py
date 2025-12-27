#!/usr/bin/env python3
"""
Download and prepare TinyStories dataset with proper BOS/EOS tokens.

This script:
1. Downloads TinyStories from HuggingFace
2. Tokenizes with the Ava tokenizer
3. Adds BOS (token 2) at start and EOS (token 1) at end
4. Saves as Arrow files for fast loading
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    # Configuration
    output_dir = Path("/root/Ava_AI/code/data/tinystories_clean")
    tokenizer_path = "/root/Ava_AI/code/data/Ava_Ai/tokenizer/tokenizer.json"
    max_length = 512
    samples_per_file = 50000

    # Special tokens
    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    BOS_TOKEN_ID = 2

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TinyStories Download & Preparation")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Download TinyStories
    print("\nDownloading TinyStories from HuggingFace...")
    print("  This may take a few minutes...")
    dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)

    print(f"\nDataset info:")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")

    def process_split(split_name, split_data, output_path):
        """Process a dataset split and save as Arrow files."""
        print(f"\nProcessing {split_name} split...")

        all_input_ids = []
        all_attention_masks = []
        file_count = 0

        for idx, example in enumerate(tqdm(split_data, desc=f"Tokenizing {split_name}")):
            text = example['text']

            # Tokenize
            encoded = tokenizer.encode(text)
            tokens = encoded.ids

            # Add BOS at start, EOS at end
            tokens = [BOS_TOKEN_ID] + tokens + [EOS_TOKEN_ID]

            # Truncate if too long (leave room for padding)
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [EOS_TOKEN_ID]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)

            # Pad to max_length
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens = tokens + [PAD_TOKEN_ID] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)

            # Save file when we have enough samples
            if len(all_input_ids) >= samples_per_file:
                save_arrow_file(
                    all_input_ids,
                    all_attention_masks,
                    output_path / f"data-{file_count:05d}.arrow"
                )
                file_count += 1
                all_input_ids = []
                all_attention_masks = []

        # Save remaining samples
        if all_input_ids:
            save_arrow_file(
                all_input_ids,
                all_attention_masks,
                output_path / f"data-{file_count:05d}.arrow"
            )
            file_count += 1

        print(f"  Saved {file_count} Arrow files to {output_path}")
        return file_count

    def save_arrow_file(input_ids, attention_masks, file_path):
        """Save data as Arrow IPC file."""
        # Convert to numpy arrays
        input_ids_np = np.array(input_ids, dtype=np.int32)
        attention_masks_np = np.array(attention_masks, dtype=np.int8)

        # Create Arrow arrays
        input_ids_arrow = pa.array([row.tolist() for row in input_ids_np], type=pa.list_(pa.int32()))
        attention_mask_arrow = pa.array([row.tolist() for row in attention_masks_np], type=pa.list_(pa.int8()))

        # Create table
        table = pa.table({
            'input_ids': input_ids_arrow,
            'attention_mask': attention_mask_arrow,
        })

        # Write as IPC file
        with pa.OSFile(str(file_path), 'wb') as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    # Process both splits
    train_files = process_split("train", dataset['train'], train_dir)
    val_files = process_split("validation", dataset['validation'], val_dir)

    # Verify a sample
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    sample_file = train_dir / "data-00000.arrow"
    with pa.memory_map(str(sample_file), 'r') as source:
        table = ipc.open_file(source).read_all()

    sample_tokens = table.column('input_ids')[0].as_py()
    print(f"\nSample token structure:")
    print(f"  First 5 tokens: {sample_tokens[:5]}")
    print(f"  Last 5 tokens: {sample_tokens[-5:]}")
    print(f"  Length: {len(sample_tokens)}")
    print(f"  Starts with BOS (2): {sample_tokens[0] == 2}")

    # Find where content ends (first PAD or EOS)
    content_end = len(sample_tokens)
    for i, t in enumerate(sample_tokens):
        if t == PAD_TOKEN_ID:
            content_end = i
            break
    print(f"  Content length (before padding): {content_end}")

    # Decode sample
    decoded = tokenizer.decode(sample_tokens[:content_end])
    print(f"\nDecoded sample (first 200 chars):")
    print(f"  {decoded[:200]}...")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Train files: {train_files}")
    print(f"Val files: {val_files}")
    print(f"\nTo use this data, update your config:")
    print(f"  data_dir: code/data/tinystories_clean/train")


if __name__ == "__main__":
    main()