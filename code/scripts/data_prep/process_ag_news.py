#!/usr/bin/env python3
"""
Process AG News dataset into training-ready format with tokenization.
"""

import json
import random
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label names for AG News
AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def get_tokenizer(tokenizer_path: str = "/project/code/models/tokenizer/enhanced-500"):
    """Load the tokenizer."""
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    logger.info(f"Tokenizer loaded! Vocab size: {len(tokenizer)}")
    return tokenizer

def check_data_quality(text: str, token_ids: List[int]) -> tuple:
    """
    Check if text/tokens meet quality standards.

    Returns:
        (is_valid, reason) - True if passes quality checks, False with reason if fails
    """
    # Check text repetition
    words = str(text).split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > len(words) * 0.3:
            return False, "excessive_word_repetition"

    # Check token diversity
    if len(token_ids) >= 20:
        unique_tokens = len(set(token_ids))
        diversity = unique_tokens / len(token_ids)
        if diversity < 0.3:
            return False, "low_diversity"

    # Check for extremely short text
    if len(words) < 3:
        return False, "too_short"

    return True, "pass"

def prepare_training_sample(sample: Dict, tokenizer) -> Optional[Dict]:
    """Convert AG News sample to training format."""
    # AG News has 'text' and 'label' fields
    text = sample.get('text', '')
    label = sample.get('label', 0)

    if not text or len(str(text)) < 10:
        return None

    # Add label information to text for language modeling
    label_name = AG_NEWS_LABELS.get(label, "Unknown")
    formatted_text = f"Category: {label_name}\n{text}"

    try:
        # Tokenize
        encoding = tokenizer(
            formatted_text,
            truncation=True,
            max_length=512,  # AG News texts are usually short
            add_special_tokens=True
        )

        # Quality check
        if len(encoding['input_ids']) < 5:
            return None

        is_valid, reason = check_data_quality(formatted_text, encoding['input_ids'])
        if not is_valid:
            return None

        return {
            "text": formatted_text,
            "input_ids": encoding['input_ids'],
            "attention_mask": encoding['attention_mask'],
            "num_tokens": len(encoding['input_ids']),
            "label": label,
            "label_name": label_name,
            "dataset": "ag_news"
        }
    except Exception as e:
        logger.warning(f"Failed to tokenize sample: {e}")
        return None

def process_split(raw_dir: Path, split_name: str, output_dir: Path, tokenizer) -> int:
    """Process a single split (train or test)."""
    split_path = raw_dir / split_name
    output_file = output_dir / f"{split_name}.jsonl"

    if not split_path.exists():
        logger.warning(f"Split {split_name} not found at {split_path}")
        return 0

    logger.info(f"Processing {split_name} split...")

    total_processed = 0
    total_raw = 0

    # Find all JSONL batch files
    batch_files = sorted(split_path.glob("batch_*.jsonl"))

    if not batch_files:
        logger.warning(f"No batch files found in {split_path}")
        return 0

    with open(output_file, 'w') as outfile:
        for batch_file in tqdm(batch_files, desc=f"Processing {split_name}"):
            with open(batch_file, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        total_raw += 1

                        processed = prepare_training_sample(sample, tokenizer)
                        if processed:
                            outfile.write(json.dumps(processed) + '\n')
                            total_processed += 1
                    except json.JSONDecodeError:
                        continue

    logger.info(f"  {split_name}: {total_processed:,}/{total_raw:,} samples processed")
    return total_processed

def create_combined_dataset(output_dir: Path, train_samples: int, test_samples: int):
    """Create combined train/val datasets."""
    logger.info("\nCreating combined train/validation datasets...")

    # Load all samples from train and test
    all_samples = []

    for split_file in ['train.jsonl', 'test.jsonl']:
        file_path = output_dir / split_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue

    if not all_samples:
        logger.warning("No samples found for combined dataset")
        return 0, 0

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Split 85/15 for train/val
    val_size = max(500, int(len(all_samples) * 0.15))
    val_samples = all_samples[:val_size]
    train_samples_combined = all_samples[val_size:]

    # Write combined files
    train_file = output_dir / "combined_train.jsonl"
    with open(train_file, 'w') as f:
        for sample in tqdm(train_samples_combined, desc="Writing combined train"):
            f.write(json.dumps(sample) + '\n')

    val_file = output_dir / "combined_val.jsonl"
    with open(val_file, 'w') as f:
        for sample in tqdm(val_samples, desc="Writing combined validation"):
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Combined train: {len(train_samples_combined):,} samples")
    logger.info(f"Combined validation: {len(val_samples):,} samples")

    return len(train_samples_combined), len(val_samples)

def main():
    """Process AG News dataset."""
    raw_dir = Path("/project/code/data/ag_news/raw")
    output_dir = Path("/project/code/data/ag_news/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Processing AG News Dataset")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Process splits
    train_count = process_split(raw_dir, "train", output_dir, tokenizer)
    test_count = process_split(raw_dir, "test", output_dir, tokenizer)

    # Create combined datasets
    combined_train, combined_val = create_combined_dataset(output_dir, train_count, test_count)

    # Calculate token statistics
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0

    for file_path, counter_name in [
        (output_dir / "combined_train.jsonl", "train"),
        (output_dir / "combined_val.jsonl", "val")
    ]:
        if file_path.exists():
            tokens = 0
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        tokens += sample.get('num_tokens', 0)
                    except:
                        pass

            if counter_name == "train":
                train_tokens = tokens
            else:
                val_tokens = tokens

            total_tokens += tokens

    # Save statistics
    stats = {
        "dataset": "ag_news",
        "train_samples": combined_train,
        "train_tokens": train_tokens,
        "val_samples": combined_val,
        "val_tokens": val_tokens,
        "total_samples": combined_train + combined_val,
        "total_tokens": total_tokens,
        "tokenizer": "/project/code/models/tokenizer/enhanced-500",
        "vocab_size": len(tokenizer),
        "classes": AG_NEWS_LABELS
    }

    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Train samples: {combined_train:,} ({train_tokens:,} tokens)")
    logger.info(f"Validation samples: {combined_val:,} ({val_tokens:,} tokens)")
    logger.info(f"Total samples: {combined_train + combined_val:,}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Avg tokens per sample: {total_tokens / (combined_train + combined_val):.1f}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - {output_dir}/combined_train.jsonl")
    logger.info(f"  - {output_dir}/combined_val.jsonl")
    logger.info(f"  - {output_dir}/processing_stats.json")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
