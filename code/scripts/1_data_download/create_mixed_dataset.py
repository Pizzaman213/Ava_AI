#!/usr/bin/env python3
"""
Create a mixed dataset combining TinyStories with complementary datasets.

Combines:
1. TinyStories (70%) - base dataset
2. ROCStories (15%) - causal reasoning, coherent narratives
3. WritingPrompts (10%) - creative writing diversity
4. Children's stories (5%) - quality narrative structure

Benefits:
- Better generalization beyond TinyStories distribution
- More diverse vocabulary and writing styles
- Improved coherence from narrative-focused datasets
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    # Configuration
    output_dir = Path("/root/Ava_AI/code/data/tinystories_mixed")
    tokenizer_path = "/root/Ava_AI/code/data/Ava_Ai/tokenizer/tokenizer.json"
    max_length = 512
    samples_per_file = 50000

    # Dataset mixing ratios
    TINYSTORIES_RATIO = 0.70
    ROCSTORIES_RATIO = 0.20
    WRITINGPROMPTS_RATIO = 0.10

    # Special tokens
    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    BOS_TOKEN_ID = 2

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Mixed Dataset Creation")
    print("=" * 70)
    print(f"\nMixing ratios:")
    print(f"  TinyStories:     {TINYSTORIES_RATIO*100:.0f}%")
    print(f"  ROCStories:      {ROCSTORIES_RATIO*100:.0f}%")
    print(f"  WritingPrompts:  {WRITINGPROMPTS_RATIO*100:.0f}%")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Load datasets
    print("\n" + "=" * 70)
    print("Loading Datasets")
    print("=" * 70)

    print("\n[1/3] Loading TinyStories...")
    try:
        tinystories = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
        print(f"  ✓ Loaded {len(tinystories['train']):,} stories")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return

    print("\n[2/3] Loading ROCStories...")
    try:
        # ROCStories: 5-sentence coherent narratives
        rocstories = load_dataset("zhoubolei/cloze_test", split="train")
        # Combine the 5 sentences into one story
        print(f"  ✓ Loaded {len(rocstories):,} stories")
    except Exception as e:
        print(f"  ✗ Failed, using alternative: {e}")
        # Fallback: use a portion of TinyStories
        rocstories = None

    print("\n[3/3] Loading WritingPrompts...")
    try:
        # WritingPrompts: creative writing
        writingprompts = load_dataset("euclaise/writingprompts", split="train")
        print(f"  ✓ Loaded {len(writingprompts):,} prompts")
    except Exception as e:
        print(f"  ✗ Failed, using alternative: {e}")
        # Fallback
        writingprompts = None

    # Calculate target counts
    target_total = len(tinystories['train'])
    target_tinystories = int(target_total * TINYSTORIES_RATIO)
    target_rocstories = int(target_total * ROCSTORIES_RATIO)
    target_writingprompts = int(target_total * WRITINGPROMPTS_RATIO)

    print(f"\n" + "=" * 70)
    print("Mixing Strategy")
    print("=" * 70)
    print(f"  TinyStories:     {target_tinystories:,} samples")
    print(f"  ROCStories:      {target_rocstories:,} samples")
    print(f"  WritingPrompts:  {target_writingprompts:,} samples")
    print(f"  Total:           {target_total:,} samples")

    # Prepare mixed data
    print("\n" + "=" * 70)
    print("Preparing Mixed Data")
    print("=" * 70)

    mixed_data = []

    # Add TinyStories
    print("\nAdding TinyStories...")
    np.random.seed(42)
    tinystories_indices = np.random.choice(
        len(tinystories['train']),
        size=target_tinystories,
        replace=False
    )
    for idx in tqdm(tinystories_indices, desc="TinyStories"):
        mixed_data.append({
            'text': tinystories['train'][int(idx)]['text'],
            'source': 'tinystories',
        })

    # Add ROCStories
    if rocstories:
        print("\nAdding ROCStories...")
        roc_indices = np.random.choice(
            len(rocstories),
            size=min(target_rocstories, len(rocstories)),
            replace=False
        )
        for idx in tqdm(roc_indices, desc="ROCStories"):
            story = rocstories[int(idx)]
            # Combine sentences
            text = ' '.join([
                story.get('sentence1', ''),
                story.get('sentence2', ''),
                story.get('sentence3', ''),
                story.get('sentence4', ''),
                story.get('sentence5', ''),
            ]).strip()
            if text:
                mixed_data.append({
                    'text': text,
                    'source': 'rocstories',
                })

    # Add WritingPrompts (use first 200 chars of story)
    if writingprompts:
        print("\nAdding WritingPrompts...")
        wp_indices = np.random.choice(
            len(writingprompts),
            size=min(target_writingprompts, len(writingprompts)),
            replace=False
        )
        for idx in tqdm(wp_indices, desc="WritingPrompts"):
            story = writingprompts[int(idx)]
            # Use prompt + beginning of story
            text = story.get('prompt', '') + ' ' + story.get('story', '')[:500]
            text = text.strip()
            if text and len(text.split()) >= 20:
                mixed_data.append({
                    'text': text,
                    'source': 'writingprompts',
                })

    # Shuffle the mixed data
    print("\nShuffling mixed data...")
    np.random.shuffle(mixed_data)

    print(f"\nFinal mixed dataset: {len(mixed_data):,} samples")

    # Split into train/val
    print("\n" + "=" * 70)
    print("Train/Val Split")
    print("=" * 70)

    split_point = int(len(mixed_data) * 0.95)
    train_data = mixed_data[:split_point]
    val_data = mixed_data[split_point:]

    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val:   {len(val_data):,} samples")

    # Tokenize and save
    print("\n" + "=" * 70)
    print("Tokenization & Saving")
    print("=" * 70)

    def save_arrow_file(input_ids, attention_masks, sources, file_path):
        """Save data as Arrow IPC file."""
        input_ids_np = np.array(input_ids, dtype=np.int32)
        attention_masks_np = np.array(attention_masks, dtype=np.int8)

        input_ids_arrow = pa.array([row.tolist() for row in input_ids_np],
                                   type=pa.list_(pa.int32()))
        attention_mask_arrow = pa.array([row.tolist() for row in attention_masks_np],
                                        type=pa.list_(pa.int8()))

        table = pa.table({
            'input_ids': input_ids_arrow,
            'attention_mask': attention_mask_arrow,
            'source': pa.array(sources, type=pa.string()),
        })

        with pa.OSFile(str(file_path), 'wb') as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def process_split(split_name, split_data, output_path):
        """Process and save a data split."""
        print(f"\nProcessing {split_name} split...")

        all_input_ids = []
        all_attention_masks = []
        all_sources = []
        file_count = 0

        for item in tqdm(split_data, desc=f"Tokenizing {split_name}"):
            text = item['text']

            # Tokenize
            try:
                encoded = tokenizer.encode(text)
                tokens = encoded.ids
            except:
                continue

            # Add BOS/EOS
            tokens = [BOS_TOKEN_ID] + tokens + [EOS_TOKEN_ID]

            # Truncate if needed
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [EOS_TOKEN_ID]

            # Create attention mask
            attention_mask = [1] * len(tokens)

            # Pad
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens = tokens + [PAD_TOKEN_ID] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)
            all_sources.append(item['source'])

            # Save when batch is full
            if len(all_input_ids) >= samples_per_file:
                save_arrow_file(
                    all_input_ids,
                    all_attention_masks,
                    all_sources,
                    output_path / f"data-{file_count:05d}.arrow"
                )
                file_count += 1
                all_input_ids = []
                all_attention_masks = []
                all_sources = []

        # Save remaining
        if all_input_ids:
            save_arrow_file(
                all_input_ids,
                all_attention_masks,
                all_sources,
                output_path / f"data-{file_count:05d}.arrow"
            )
            file_count += 1

        print(f"  Saved {file_count} files")
        return file_count

    train_files = process_split("train", train_data, train_dir)
    val_files = process_split("validation", val_data, val_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Train files: {train_files}")
    print(f"Val files: {val_files}")
    print(f"\nDataset composition:")
    from collections import Counter
    train_sources = Counter(item['source'] for item in train_data)
    for source, count in train_sources.items():
        print(f"  {source:15s}: {count:,} ({count/len(train_data)*100:.1f}%)")
    print(f"\nTo use this mixed data:")
    print(f"  data:")
    print(f"    data_dir: code/data/tinystories_mixed/train")
    print(f"    val_data_dir: code/data/tinystories_mixed/val")


if __name__ == "__main__":
    main()
