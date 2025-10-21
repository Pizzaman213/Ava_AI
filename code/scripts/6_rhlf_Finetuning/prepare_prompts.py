#!/usr/bin/env python3
"""
Utility script to prepare prompt datasets for RLHF training.

This script helps convert various formats into the required prompts.json format.
"""

import json
import argparse
from pathlib import Path
from typing import List


def load_text_file(file_path: str) -> List[str]:
    """Load prompts from text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_jsonl_file(file_path: str, prompt_key: str = 'prompt') -> List[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if prompt_key in data:
                prompts.append(data[prompt_key])
    return prompts


def load_json_file(file_path: str, prompt_key: str = 'prompt') -> List[str]:
    """Load prompts from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        # List of strings or dicts
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and prompt_key in item:
                prompts.append(item[prompt_key])
        return prompts
    elif isinstance(data, dict):
        # Single dict with prompts key
        if 'prompts' in data:
            return data['prompts']
        elif prompt_key in data:
            return [data[prompt_key]]

    return []


def save_prompts(prompts: List[str], output_path: str):
    """Save prompts in the required format."""
    output_data = {'prompts': prompts}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} prompts to {output_path}")


def create_sample_prompts(num_prompts: int = 100) -> List[str]:
    """Create sample prompts for testing."""
    templates = [
        "Explain {} in simple terms:",
        "What is {}?",
        "How does {} work?",
        "Write a tutorial about {}:",
        "Describe the benefits of {}:",
        "Compare {} with similar concepts:",
        "What are the applications of {}?",
        "Define {} and provide examples:",
    ]

    topics = [
        "machine learning", "neural networks", "deep learning", "reinforcement learning",
        "natural language processing", "computer vision", "robotics", "quantum computing",
        "blockchain", "cloud computing", "edge computing", "big data", "data science",
        "artificial intelligence", "supervised learning", "unsupervised learning",
        "transfer learning", "fine-tuning", "optimization", "gradient descent",
        "backpropagation", "attention mechanism", "transformers", "BERT", "GPT"
    ]

    prompts = []
    for i in range(num_prompts):
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        prompts.append(template.format(topic))

    return prompts


def split_prompts(prompts: List[str], train_ratio: float = 0.9):
    """Split prompts into train and eval sets."""
    split_idx = int(len(prompts) * train_ratio)
    train_prompts = prompts[:split_idx]
    eval_prompts = prompts[split_idx:]
    return train_prompts, eval_prompts


def main():
    parser = argparse.ArgumentParser(description='Prepare prompt datasets for RLHF training')

    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (txt, json, or jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/project/code/data/rlhf/prompts.json',
        help='Output path for prompts.json'
    )
    parser.add_argument(
        '--eval-output',
        type=str,
        default='/project/code/data/rlhf/eval_prompts.json',
        help='Output path for eval_prompts.json'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'json', 'jsonl', 'auto'],
        default='auto',
        help='Input file format (auto-detect by default)'
    )
    parser.add_argument(
        '--prompt-key',
        type=str,
        default='prompt',
        help='Key name for prompts in JSON/JSONL files'
    )
    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample prompts for testing'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of sample prompts to create'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of prompts to use for training (rest for eval)'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Do not split into train/eval sets'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create prompts
    prompts = []  # Initialize to avoid possibly unbound variable
    if args.create_samples:
        print(f"Creating {args.num_samples} sample prompts...")
        prompts = create_sample_prompts(args.num_samples)
    elif args.input:
        print(f"Loading prompts from {args.input}...")

        # Detect format
        if args.format == 'auto':
            if args.input.endswith('.txt'):
                file_format = 'txt'
            elif args.input.endswith('.jsonl'):
                file_format = 'jsonl'
            elif args.input.endswith('.json'):
                file_format = 'json'
            else:
                raise ValueError("Cannot auto-detect format. Please specify --format")
        else:
            file_format = args.format

        # Load prompts
        if file_format == 'txt':
            prompts = load_text_file(args.input)
        elif file_format == 'json':
            prompts = load_json_file(args.input, args.prompt_key)
        elif file_format == 'jsonl':
            prompts = load_jsonl_file(args.input, args.prompt_key)

        print(f"Loaded {len(prompts)} prompts")
    else:
        raise ValueError("Must specify either --input or --create-samples")

    # Split or save all
    if args.no_split:
        if prompts:  # Check if prompts is not empty
            save_prompts(prompts, args.output)
    else:
        train_prompts, eval_prompts = split_prompts(prompts, args.train_ratio)
        print(f"Split: {len(train_prompts)} train, {len(eval_prompts)} eval")
        save_prompts(train_prompts, args.output)
        save_prompts(eval_prompts, args.eval_output)

    print("\nDone! You can now run RLHF training:")
    print(f"  python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/small.yaml")


if __name__ == '__main__':
    main()
