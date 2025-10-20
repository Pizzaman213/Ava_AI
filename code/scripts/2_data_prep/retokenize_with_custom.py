#!/usr/bin/env python3
"""
Re-tokenize existing processed JSONL files with the custom 65k tokenizer.

This script:
1. Reads existing *_processed.jsonl files (tokenized with Qwen)
2. Extracts the 'text' field
3. Re-tokenizes with the custom 65k tokenizer
4. Saves new JSONL files with updated input_ids
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def retokenize_file(input_path: Path, output_path: Path, tokenizer):
    """Re-tokenize a single JSONL file."""

    print(f"\n📄 Processing: {input_path.name}")

    processed_samples = []
    skipped = 0
    total = 0

    # Count lines first for progress bar
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc=f"   Re-tokenizing", leave=False):
            total += 1
            try:
                sample = json.loads(line.strip())

                # Extract text
                text = sample.get('text', '')
                if not text or len(text) < 10:
                    skipped += 1
                    continue

                # Re-tokenize with custom tokenizer
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_attention_mask=False,
                    add_special_tokens=True
                )

                # Create new sample
                new_sample = {
                    'text': text,
                    'input_ids': encoding['input_ids'],
                    'source_dataset': sample.get('source_dataset', 'unknown')
                }

                # Only keep samples with reasonable length
                if len(encoding['input_ids']) >= 20:
                    processed_samples.append(new_sample)
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                continue

    # Write output
    with open(output_path, 'w') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"   ✅ Processed: {len(processed_samples)}/{total} samples ({skipped} skipped)")
    return len(processed_samples)


def main():
    print("="*80)
    print("🔄 Re-tokenizing Data with Custom 65k Tokenizer")
    print("="*80)

    # Setup paths
    input_dir = Path("/project/code/data/processed_qwen_backup")
    output_dir = Path("/project/code/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom tokenizer
    print("\n📝 Loading custom tokenizer...")
    tokenizer_path = "/project/code/models/tokenizer/enhanced-65k"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Vocab size: {len(tokenizer)}")

    # Find all JSONL files
    jsonl_files = sorted(input_dir.glob("*_processed.jsonl"))
    print(f"\n📂 Found {len(jsonl_files)} JSONL files to re-tokenize")

    # Process each file
    total_samples = 0
    for jsonl_file in jsonl_files:
        output_file = output_dir / jsonl_file.name
        samples = retokenize_file(jsonl_file, output_file, tokenizer)
        total_samples += samples

    print(f"\n✅ Complete! Total samples: {total_samples:,}")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
