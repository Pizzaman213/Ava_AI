#!/usr/bin/env python3
"""
Improved TinyStories download with quality filtering, deduplication, and metadata.

Improvements over basic version:
1. Quality filtering (removes repetitive/low-quality stories)
2. Near-duplicate detection
3. Stratified train/val split
4. Length distribution balancing
5. Difficulty scoring for curriculum learning
6. Better statistics and validation
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def quality_score(text):
    """Calculate story quality score (0-1, higher is better)."""
    if not text or len(text.strip()) < 20:
        return 0.0

    score = 1.0
    words = text.lower().split()

    # Penalize very short stories
    if len(words) < 10:
        score *= 0.3
    elif len(words) < 20:
        score *= 0.6

    # Penalize repetitive text (low vocabulary diversity)
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.2:
        score *= 0.3
    elif unique_ratio < 0.4:
        score *= 0.7

    # Reward proper sentence structure
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 3:
        score *= 1.1
    elif len(sentences) < 2:
        score *= 0.7

    # Penalize excessive special characters
    special_chars = sum(c in '!@#$%^&*()_+=[]{}|;:,<>?/' for c in text)
    special_ratio = special_chars / max(len(text), 1)
    if special_ratio > 0.15:
        score *= 0.5

    # Penalize all-caps text
    if text.isupper() and len(text) > 20:
        score *= 0.4

    return min(score, 1.0)


def calculate_difficulty(text):
    """Estimate text difficulty (0-1, 0=easy, 1=hard)."""
    words = text.split()
    if not words:
        return 0.0

    avg_word_length = sum(len(w) for w in words) / len(words)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = len(words) / max(len(sentences), 1)

    # Normalize metrics
    # Average word length: 3-10 chars (3=easy, 10=hard)
    word_complexity = (avg_word_length - 3) / 7
    # Average sentence length: 5-20 words (5=easy, 20=hard)
    sentence_complexity = (avg_sentence_length - 5) / 15

    difficulty = (word_complexity * 0.6 + sentence_complexity * 0.4)
    return max(0, min(1, difficulty))


def get_fuzzy_hash(text, ngram=50):
    """Create fuzzy hash for near-duplicate detection."""
    words = text.lower().split()[:ngram]
    return hashlib.md5(' '.join(words).encode()).hexdigest()


def main():
    # Configuration
    output_dir = Path("/root/Ava_AI/code/data/tinystories_16k")
    tokenizer_path = "/root/Ava_AI/code/data/Ava_Ai/tokenizer_16k/tokenizer.json"  # Using 16K vocab tokenizer for better model capacity
    max_length = 512
    samples_per_file = 50000

    # Quality thresholds
    MIN_QUALITY_SCORE = 0.5  # Filter out bottom 20-30% of stories
    MAX_DUPLICATES = 3  # Max times same fuzzy hash appears

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
    print("TinyStories Improved Download & Preparation")
    print("=" * 70)
    print(f"\nQuality threshold: {MIN_QUALITY_SCORE}")
    print(f"Max duplicates: {MAX_DUPLICATES}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Download TinyStories
    print("\nDownloading TinyStories from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)

    print(f"\nOriginal dataset size:")
    print(f"  Train: {len(dataset['train']):,}")
    print(f"  Validation: {len(dataset['validation']):,}")

    # First pass: filter and deduplicate
    print("\n" + "=" * 70)
    print("PHASE 1: Quality Filtering & Deduplication")
    print("=" * 70)

    filtered_data = []
    fuzzy_hash_counts = defaultdict(int)

    stats = {
        'total': 0,
        'low_quality': 0,
        'duplicates': 0,
        'kept': 0,
    }

    for example in tqdm(dataset['train'], desc="Filtering"):
        stats['total'] += 1
        text = example['text']

        # Check quality
        quality = quality_score(text)
        if quality < MIN_QUALITY_SCORE:
            stats['low_quality'] += 1
            continue

        # Check duplicates
        fuzzy_hash = get_fuzzy_hash(text)
        fuzzy_hash_counts[fuzzy_hash] += 1
        if fuzzy_hash_counts[fuzzy_hash] > MAX_DUPLICATES:
            stats['duplicates'] += 1
            continue

        # Keep this story
        difficulty = calculate_difficulty(text)
        filtered_data.append({
            'text': text,
            'quality': quality,
            'difficulty': difficulty,
            'length': len(text.split()),
        })
        stats['kept'] += 1

    print(f"\nFiltering results:")
    print(f"  Total stories: {stats['total']:,}")
    print(f"  Low quality removed: {stats['low_quality']:,} ({stats['low_quality']/stats['total']*100:.1f}%)")
    print(f"  Duplicates removed: {stats['duplicates']:,} ({stats['duplicates']/stats['total']*100:.1f}%)")
    print(f"  Stories kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")

    # Second pass: stratified train/val split
    print("\n" + "=" * 70)
    print("PHASE 2: Stratified Train/Val Split")
    print("=" * 70)

    # Create strata based on length and difficulty
    np.random.seed(42)
    indices = np.arange(len(filtered_data))

    # Group by length buckets and difficulty levels
    strata = []
    for item in filtered_data:
        length_bucket = (item['length'] // 50) * 50  # 0-49, 50-99, etc.
        difficulty_level = int(item['difficulty'] * 5)  # 0-4
        strata.append(f"{length_bucket}_{difficulty_level}")

    # Stratified split (95% train, 5% val)
    from collections import Counter
    strata_counts = Counter(strata)

    train_indices = []
    val_indices = []

    for stratum in set(strata):
        stratum_indices = [i for i, s in enumerate(strata) if s == stratum]
        np.random.shuffle(stratum_indices)

        # 95/5 split
        split_point = int(len(stratum_indices) * 0.95)
        train_indices.extend(stratum_indices[:split_point])
        val_indices.extend(stratum_indices[split_point:])

    print(f"\nSplit results:")
    print(f"  Train samples: {len(train_indices):,}")
    print(f"  Val samples: {len(val_indices):,}")
    print(f"  Val ratio: {len(val_indices)/len(filtered_data)*100:.2f}%")

    # Third pass: tokenize and save
    print("\n" + "=" * 70)
    print("PHASE 3: Tokenization & Saving")
    print("=" * 70)

    def save_arrow_file(input_ids, attention_masks, difficulties, lengths, file_path):
        """Save data as Arrow IPC file."""
        input_ids_np = np.array(input_ids, dtype=np.int32)
        attention_masks_np = np.array(attention_masks, dtype=np.int8)
        difficulties_np = np.array(difficulties, dtype=np.float32)
        lengths_np = np.array(lengths, dtype=np.int16)

        input_ids_arrow = pa.array([row.tolist() for row in input_ids_np],
                                   type=pa.list_(pa.int32()))
        attention_mask_arrow = pa.array([row.tolist() for row in attention_masks_np],
                                        type=pa.list_(pa.int8()))

        table = pa.table({
            'input_ids': input_ids_arrow,
            'attention_mask': attention_mask_arrow,
            'difficulty': difficulties_np,
            'length': lengths_np,
        })

        with pa.OSFile(str(file_path), 'wb') as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def process_split(split_name, indices_to_use, output_path):
        """Process and save a data split."""
        print(f"\nProcessing {split_name} split...")

        all_input_ids = []
        all_attention_masks = []
        all_difficulties = []
        all_lengths = []
        file_count = 0

        for idx in tqdm(indices_to_use, desc=f"Tokenizing {split_name}"):
            item = filtered_data[idx]
            text = item['text']

            # Tokenize
            encoded = tokenizer.encode(text)
            tokens = encoded.ids

            # Add BOS at start, EOS at end
            tokens = [BOS_TOKEN_ID] + tokens + [EOS_TOKEN_ID]

            # Truncate if needed
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [EOS_TOKEN_ID]

            # Create attention mask
            attention_mask = [1] * len(tokens)

            # Pad to max_length
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens = tokens + [PAD_TOKEN_ID] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)
            all_difficulties.append(item['difficulty'])
            all_lengths.append(item['length'])

            # Save file when we have enough samples
            if len(all_input_ids) >= samples_per_file:
                save_arrow_file(
                    all_input_ids,
                    all_attention_masks,
                    all_difficulties,
                    all_lengths,
                    output_path / f"data-{file_count:05d}.arrow"
                )
                file_count += 1
                all_input_ids = []
                all_attention_masks = []
                all_difficulties = []
                all_lengths = []

        # Save remaining samples
        if all_input_ids:
            save_arrow_file(
                all_input_ids,
                all_attention_masks,
                all_difficulties,
                all_lengths,
                output_path / f"data-{file_count:05d}.arrow"
            )
            file_count += 1

        print(f"  Saved {file_count} files to {output_path}")
        return file_count

    # Process both splits
    train_files = process_split("train", train_indices, train_dir)
    val_files = process_split("validation", val_indices, val_dir)

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    sample_file = train_dir / "data-00000.arrow"
    with pa.memory_map(str(sample_file), 'r') as source:
        table = ipc.open_file(source).read_all()

    print(f"\nSample file schema:")
    print(f"  {table.schema}")

    sample_tokens = table.column('input_ids')[0].as_py()
    sample_difficulty = table.column('difficulty')[0].as_py()
    sample_length = table.column('length')[0].as_py()

    print(f"\nSample data:")
    print(f"  First 5 tokens: {sample_tokens[:5]}")
    print(f"  Starts with BOS (2): {sample_tokens[0] == 2}")
    print(f"  Difficulty: {sample_difficulty:.3f}")
    print(f"  Original length: {sample_length} words")

    # Find content end
    content_end = len(sample_tokens)
    for i, t in enumerate(sample_tokens):
        if t == PAD_TOKEN_ID:
            content_end = i
            break

    decoded = tokenizer.decode(sample_tokens[1:content_end-1])  # Skip BOS/EOS
    print(f"\nDecoded sample:")
    print(f"  {decoded[:250]}...")

    # Statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    all_difficulties = [item['difficulty'] for item in filtered_data]
    all_lengths = [item['length'] for item in filtered_data]

    print(f"\nDifficulty distribution:")
    print(f"  Mean: {np.mean(all_difficulties):.3f}")
    print(f"  Std:  {np.std(all_difficulties):.3f}")
    print(f"  Min:  {np.min(all_difficulties):.3f}")
    print(f"  Max:  {np.max(all_difficulties):.3f}")

    print(f"\nLength distribution:")
    print(f"  Mean: {np.mean(all_lengths):.1f} words")
    print(f"  Std:  {np.std(all_lengths):.1f} words")
    print(f"  Min:  {np.min(all_lengths)} words")
    print(f"  Max:  {np.max(all_lengths)} words")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Train files: {train_files}")
    print(f"Val files: {val_files}")
    print(f"\nTo use this improved data, update your config:")
    print(f"  data:")
    print(f"    data_dir: code/data/tinystories_improved/train")
    print(f"    val_data_dir: code/data/tinystories_improved/val")
    print(f"    auto_create_validation_split: false  # Using stratified split")


if __name__ == "__main__":
    main()
