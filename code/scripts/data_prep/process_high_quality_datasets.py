#!/usr/bin/env python3
"""
Process high-quality datasets into unified JSONL format for training.

Input: /project/code/data/raw/*
Output: /project/code/data/processed/*.jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

RAW_DATA_DIR = Path("/project/code/data/raw")
PROCESSED_DATA_DIR = Path("/project/code/data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = "/project/code/models/tokenizer/enhanced-65k"

def load_tokenizer():
    """Load tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        print(f"✅ Loaded tokenizer from {TOKENIZER_PATH}")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None


def extract_text_from_sample(sample: Dict[str, Any], dataset_name: str) -> Optional[str]:
    """Extract text content from a dataset sample based on dataset type."""

    # ========================================================================
    # WEB TEXT DATASETS
    # ========================================================================

    if "fineweb" in dataset_name.lower() or "c4" in dataset_name.lower():
        # FineWeb-Edu / C4: {'text': 'content...'}
        if 'text' in sample:
            return sample['text'].strip()

    if "dolma" in dataset_name.lower():
        # Dolma: {'text': 'content...'}
        if 'text' in sample:
            return sample['text'].strip()

    # ========================================================================
    # INSTRUCTION DATASETS
    # ========================================================================

    if "ultrachat" in dataset_name.lower():
        # Ultrachat: {'messages': [{'role': 'user', 'content': '...'}, ...]}
        if 'messages' in sample and isinstance(sample['messages'], list):
            texts = []
            for msg in sample['messages']:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                texts.append(f"{role.capitalize()}: {content}")
            return '\n'.join(texts).strip()

    if "lima" in dataset_name.lower():
        # LIMA: {'conversations': ['user prompt', 'assistant response']}
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            texts = []
            for i, turn in enumerate(sample['conversations']):
                role = 'User' if i % 2 == 0 else 'Assistant'
                texts.append(f"{role}: {turn}")
            return '\n'.join(texts).strip()

    if "orca" in dataset_name.lower():
        # Orca datasets: {'question': '...', 'answer': '...'}
        if 'question' in sample and 'answer' in sample:
            return f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        # OR: {'system': '...', 'user': '...', 'assistant': '...'}
        if 'system' in sample and 'user' in sample and 'assistant' in sample:
            return f"System: {sample['system']}\nUser: {sample['user']}\nAssistant: {sample['assistant']}"

    if "slimorca" in dataset_name.lower():
        # SlimOrca: {'conversations': [{'from': 'human', 'value': '...'}, ...]}
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            texts = []
            for msg in sample['conversations']:
                from_who = msg.get('from', 'unknown')
                value = msg.get('value', '')
                role = 'User' if from_who == 'human' else 'Assistant'
                texts.append(f"{role}: {value}")
            return '\n'.join(texts).strip()

    # ========================================================================
    # CODE DATASETS
    # ========================================================================

    if "starcoder" in dataset_name.lower() or "stack" in dataset_name.lower():
        # StarCoder / The Stack: {'content': 'code...'}
        if 'content' in sample:
            return sample['content'].strip()
        # OR: {'code': 'code...'}
        if 'code' in sample:
            return sample['code'].strip()

    if "codesearchnet" in dataset_name.lower() or "code_search_net" in dataset_name.lower():
        # CodeSearchNet: {'func_documentation_string': '...', 'func_code_string': '...'}
        if 'func_documentation_string' in sample and 'func_code_string' in sample:
            doc = sample['func_documentation_string']
            code = sample['func_code_string']
            return f"# {doc}\n{code}"

    # ========================================================================
    # KNOWLEDGE DATASETS
    # ========================================================================

    if "wikipedia" in dataset_name.lower():
        # Wikipedia: {'text': 'article content...'}
        if 'text' in sample:
            return sample['text'].strip()

    if "bookcorpus" in dataset_name.lower():
        # BookCorpus: {'text': 'book content...'}
        if 'text' in sample:
            return sample['text'].strip()

    if "pubmed" in dataset_name.lower():
        # PubMed: {'MedlineCitation': {'Article': {'Abstract': {'AbstractText': '...'}}}}
        if 'MedlineCitation' in sample:
            try:
                abstract = sample['MedlineCitation']['Article']['Abstract']['AbstractText']
                if isinstance(abstract, list):
                    abstract = ' '.join(abstract)
                return abstract.strip()
            except (KeyError, TypeError):
                pass
        # OR simpler format: {'text': '...'}
        if 'text' in sample:
            return sample['text'].strip()

    # ========================================================================
    # CONVERSATIONAL DATASETS
    # ========================================================================

    if "sharegpt" in dataset_name.lower():
        # ShareGPT: {'conversations': [{'from': 'human', 'value': '...'}, ...]}
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            texts = []
            for msg in sample['conversations']:
                from_who = msg.get('from', 'unknown')
                value = msg.get('value', '')
                role = 'User' if from_who in ['human', 'user'] else 'Assistant'
                texts.append(f"{role}: {value}")
            return '\n'.join(texts).strip()

    if "hh-rlhf" in dataset_name.lower() or "anthropic" in dataset_name.lower():
        # Anthropic HH-RLHF: {'chosen': 'better response', 'rejected': 'worse response'}
        # Use the chosen response
        if 'chosen' in sample:
            return sample['chosen'].strip()

    # ========================================================================
    # GENERIC FALLBACK
    # ========================================================================

    # Try common text fields
    text_fields = ['text', 'content', 'prompt', 'response', 'instruction',
                   'question', 'answer', 'chosen', 'messages', 'conversations']

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
                            elif 'value' in msg:
                                texts.append(msg['value'])
                            elif 'text' in msg:
                                texts.append(msg['text'])
                        if texts:
                            return ' '.join(texts).strip()
                    elif isinstance(sample[field][0], str):
                        return ' '.join(sample[field]).strip()

    return None


def process_dataset(dataset_path: Path, tokenizer, min_length: int = 10, max_length: int = 512):
    """Process a single dataset into JSONL format."""

    dataset_name = dataset_path.name
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")

    try:
        # Load dataset
        print(f"Loading from {dataset_path}...")
        ds = load_from_disk(str(dataset_path))
        print(f"Loaded {len(ds)} examples")

        # Process samples
        processed = []
        skipped = 0

        for sample in tqdm(ds, desc=f"Processing {dataset_name}"):
            # Extract text
            text = extract_text_from_sample(sample, dataset_name)

            if not text:
                skipped += 1
                continue

            # Basic filtering
            if len(text) < min_length:
                skipped += 1
                continue

            # Tokenize to check length
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)

            if len(tokens) < 5:  # Too short after tokenization
                skipped += 1
                continue

            # Add to processed
            processed.append({
                'text': text,
                'source': dataset_name
            })

        # Save to JSONL
        output_file = PROCESSED_DATA_DIR / f"{dataset_name}_processed.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(processed)} examples to {output_file}")
        print(f"⚠️  Skipped {skipped} examples (no text or too short)")

        return len(processed)

    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        return 0


def main():
    """Process all raw datasets."""

    print("\n" + "="*80)
    print("HIGH-QUALITY DATASET PROCESSING")
    print("="*80)
    print(f"Input: {RAW_DATA_DIR}")
    print(f"Output: {PROCESSED_DATA_DIR}")
    print("="*80 + "\n")

    # Load tokenizer
    tokenizer = load_tokenizer()
    if not tokenizer:
        print("❌ Cannot proceed without tokenizer")
        return

    # Find all raw datasets
    raw_datasets = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(raw_datasets)} raw datasets to process\n")

    if not raw_datasets:
        print("⚠️  No raw datasets found. Run download_high_quality_datasets.py first.")
        return

    # Process each dataset
    total_processed = 0
    for dataset_path in raw_datasets:
        num_processed = process_dataset(dataset_path, tokenizer)
        total_processed += num_processed

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)

    # Count all processed files
    all_processed_files = list(PROCESSED_DATA_DIR.glob("*.jsonl"))
    total_lines = 0
    total_size = 0

    for file in all_processed_files:
        with open(file, 'r') as f:
            lines = sum(1 for _ in f)
            total_lines += lines
        total_size += file.stat().st_size

    print(f"Total processed datasets: {len(all_processed_files)}")
    print(f"Total examples: {total_lines:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print("="*80 + "\n")

    # Estimate tokens
    avg_tokens_per_example = 250
    total_tokens = total_lines * avg_tokens_per_example
    print(f"Estimated tokens: ~{total_tokens/1e6:.1f}M tokens ({total_tokens/1e9:.2f}B)")

    # Check if we hit targets
    print(f"\nTarget comparison:")
    print(f"  Minimum (500M tokens): {'✅' if total_tokens >= 500e6 else '❌'}")
    print(f"  Optimal (1B tokens): {'✅' if total_tokens >= 1e9 else '❌'}")

    if total_tokens >= 500e6:
        # Calculate epochs
        steps = 30000
        batch_size = 8
        grad_accum = 16
        effective_batch = batch_size * grad_accum
        total_examples_needed = steps * effective_batch

        epochs = total_examples_needed / total_lines

        print(f"\nTraining statistics:")
        print(f"  Total examples: {total_lines:,}")
        print(f"  Examples per epoch: {total_lines:,}")
        print(f"  Training steps: 30,000")
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Total examples seen: {total_examples_needed:,}")
        print(f"  Epochs: {epochs:.1f}")

        if epochs < 5:
            print(f"\n✅ EXCELLENT! Only {epochs:.1f} epochs - minimal memorization risk")
        elif epochs < 10:
            print(f"\n✅ GOOD! {epochs:.1f} epochs - acceptable memorization risk")
        elif epochs < 15:
            print(f"\n⚠️  FAIR: {epochs:.1f} epochs - moderate memorization risk")
        else:
            print(f"\n⚠️  HIGH: {epochs:.1f} epochs - consider more data or fewer steps")

    print("\nNext steps:")
    print("1. Verify data quality by sampling a few examples")
    print("2. Update training config if needed")
    print("3. Restart training with expanded dataset")
    print("4. Monitor loss curve - should be slower than before\n")


if __name__ == "__main__":
    main()
