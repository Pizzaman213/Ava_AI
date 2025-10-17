#!/usr/bin/env python3
"""
Process raw HuggingFace datasets into combined JSONL format for training.

This script:
1. Loads datasets from /project/code/data/raw/
2. Extracts text content
3. Tokenizes and formats for training
4. Splits into train/eval sets
5. Saves as combined_train.jsonl and combined_eval.jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def extract_text_from_sample(sample: Dict[str, Any], dataset_name: str) -> str:
    """Extract text content from a dataset sample based on dataset type."""

    # Common text fields to try
    text_fields = ['text', 'content', 'prompt', 'response', 'instruction',
                   'question', 'answer', 'chosen', 'rejected', 'messages']

    # Try to find text content
    for field in text_fields:
        if field in sample:
            if isinstance(sample[field], str):
                return sample[field].strip()
            elif isinstance(sample[field], list):
                # Handle conversations/messages
                if len(sample[field]) > 0:
                    if isinstance(sample[field][0], dict):
                        # Extract text from message dicts
                        texts = []
                        for msg in sample[field]:
                            if 'content' in msg:
                                texts.append(msg['content'])
                            elif 'text' in msg:
                                texts.append(msg['text'])
                        if texts:
                            return ' '.join(texts).strip()
                    else:
                        # Join list of strings
                        return ' '.join(str(x) for x in sample[field]).strip()

    # For specific dataset formats
    if 'instruction' in sample and 'output' in sample:
        # Instruction-following format
        instr = sample.get('instruction', '')
        inp = sample.get('input', '')
        out = sample.get('output', '')
        if inp:
            return f"{instr}\n{inp}\n{out}".strip()
        return f"{instr}\n{out}".strip()

    if 'prompt' in sample and 'completion' in sample:
        # Prompt-completion format
        return f"{sample['prompt']}\n{sample['completion']}".strip()

    # Last resort: concatenate all string fields
    text_parts = []
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 10:
            text_parts.append(value)

    if text_parts:
        return ' '.join(text_parts).strip()

    return ""


def process_dataset(dataset_path: str, tokenizer, max_samples: int = None) -> List[Dict[str, Any]]:
    """Process a single dataset and return tokenized samples."""

    dataset_name = Path(dataset_path).name
    print(f"\n📁 Processing: {dataset_name}")

    try:
        # Load dataset
        dataset = load_from_disk(dataset_path)

        # Handle dataset with splits
        if isinstance(dataset, dict):
            # Try train split first, then other splits
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    dataset = dataset[split_name]
                    print(f"   Using split: {split_name}")
                    break

        if not isinstance(dataset, Dataset):
            print(f"   ⚠️  Skipping - unsupported format")
            return []

        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        print(f"   📊 Samples: {len(dataset)}")

        # Process samples
        processed = []
        skipped = 0

        for idx, sample in enumerate(tqdm(dataset, desc=f"   Tokenizing {dataset_name}", leave=False)):
            text = extract_text_from_sample(sample, dataset_name)

            if not text or len(text) < 10:
                skipped += 1
                continue

            # Tokenize
            try:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_attention_mask=False,
                    add_special_tokens=True
                )

                # Only keep samples with reasonable length
                if len(encoding['input_ids']) >= 20:
                    processed.append({
                        'text': text,
                        'input_ids': encoding['input_ids'],
                        'source_dataset': dataset_name
                    })
            except Exception as e:
                skipped += 1
                continue

        print(f"   ✅ Processed: {len(processed)} samples ({skipped} skipped)")
        return processed

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return []


def main():
    print("="*80)
    print("🚀 Processing Raw Datasets for Training")
    print("="*80)

    # Setup paths
    raw_dir = Path("/project/code/data/raw")
    output_dir = Path("/project/code/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer_path = "/project/code/models/tokenizer/enhanced-65k"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Vocab size: {len(tokenizer)}")

    # Find all dataset directories
    dataset_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"\n📂 Found {len(dataset_dirs)} datasets")

    # Process datasets
    all_samples = []
    max_samples_per_dataset = 100000  # Limit to 100k per dataset for initial processing

    for dataset_path in dataset_dirs:
        samples = process_dataset(str(dataset_path), tokenizer, max_samples_per_dataset)
        all_samples.extend(samples)

        # Stop if we have enough data
        if len(all_samples) >= 1000000:  # 1M samples total
            print(f"\n⚠️  Reached 1M samples, stopping early")
            break

    if not all_samples:
        print("\n❌ No samples processed! Check dataset formats.")
        return

    print(f"\n📊 Total samples: {len(all_samples)}")

    # Shuffle
    print("\n🔀 Shuffling samples...")
    random.seed(42)
    random.shuffle(all_samples)

    # Split train/eval
    eval_size = int(len(all_samples) * 0.05)  # 5% for eval
    train_samples = all_samples[eval_size:]
    eval_samples = all_samples[:eval_size]

    print(f"   Train: {len(train_samples)} samples")
    print(f"   Eval:  {len(eval_samples)} samples")

    # DISABLED: No longer create combined files - use individual processed files instead
    # This prevents creating massive combined_train.jsonl and combined_eval.jsonl files
    print(f"\n⚠️  Combined file creation disabled")
    print(f"   Use individual processed files instead of combined_train.jsonl/combined_eval.jsonl")
    print(f"   The data loader will automatically split files for train/val")

    # Keep the code commented out for reference:
    # # Save train set
    # train_file = output_dir / "combined_train.jsonl"
    # print(f"\n💾 Saving train set to {train_file}...")
    # with open(train_file, 'w') as f:
    #     for sample in tqdm(train_samples, desc="   Writing train"):
    #         f.write(json.dumps(sample) + '\n')
    #
    # # Save eval set
    # eval_file = output_dir / "combined_eval.jsonl"
    # print(f"\n💾 Saving eval set to {eval_file}...")
    # with open(eval_file, 'w') as f:
    #     for sample in tqdm(eval_samples, desc="   Writing eval"):
    #         f.write(json.dumps(sample) + '\n')

    train_file = None
    eval_file = None

    # Print summary
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    print("="*80)
    print(f"\nNo combined files created (feature disabled)")
    print(f"Use individual *_processed.jsonl files from data/processed/ directory")
    print(f"\nStatistics:")
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Eval samples: {len(eval_samples):,}")

    # Sample statistics
    total_tokens = sum(len(s['input_ids']) for s in all_samples)
    avg_tokens = total_tokens / len(all_samples)
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/sample: {avg_tokens:.1f}")

    print("\n🎉 Data loader will use individual processed files for training!")


if __name__ == "__main__":
    main()
