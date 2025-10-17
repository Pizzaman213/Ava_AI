#!/usr/bin/env python3
"""
Download high-quality datasets to expand training data from 773K to 2-5M examples.

Target: 2-5M examples (500M-1B tokens)
Current: 773K examples (~200M tokens)
Needed: 1.2-4.2M additional examples

High-quality dataset sources:
- SlimPajama-627B (sample)
- RedPajama-v2 (sample)
- C4 (sample)
- FineWeb-Edu (larger sample)
- Dolma (sample)
- StarCoder (code)
- The Stack v2 (code)
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Output directory
RAW_DATA_DIR = Path("/project/code/data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset(name: str, config: str, split: str, streaming: bool = False, num_samples: int = None):
    """Download a dataset from HuggingFace."""
    print(f"\n{'='*80}")
    print(f"Downloading: {name}")
    print(f"Config: {config}")
    print(f"Split: {split}")
    print(f"Samples: {num_samples if num_samples else 'ALL'}")
    print(f"{'='*80}\n")

    try:
        if streaming or num_samples:
            # Stream for large datasets or when sampling
            ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)

            # Take limited samples
            if num_samples:
                samples = []
                for i, sample in enumerate(tqdm(ds, total=num_samples, desc=f"Sampling {name}")):
                    samples.append(sample)
                    if i + 1 >= num_samples:
                        break

                # Convert to dataset
                from datasets import Dataset
                ds = Dataset.from_list(samples)
        else:
            # Load full dataset
            ds = load_dataset(name, config, split=split, trust_remote_code=True)

        # Save to disk
        output_name = name.replace("/", "_")
        if config and config != "default":
            output_name += f"_{config}"
        output_path = RAW_DATA_DIR / output_name

        print(f"Saving to: {output_path}")
        ds.save_to_disk(str(output_path))
        print(f"✅ Saved {len(ds)} examples")

        return len(ds)

    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        return 0


def main():
    """Download high-quality datasets."""

    total_downloaded = 0

    print("\n" + "="*80)
    print("HIGH-QUALITY DATASET DOWNLOAD")
    print("="*80)
    print(f"Current data: ~773K examples")
    print(f"Target: 2-5M examples")
    print(f"Output: {RAW_DATA_DIR}")
    print("="*80 + "\n")

    # ========================================================================
    # TIER 1: HIGHEST QUALITY - General Web Text
    # ========================================================================

    print("\n🌟 TIER 1: HIGHEST QUALITY WEB TEXT")

    # FineWeb-Edu (100k additional samples) - Educational web content
    total_downloaded += download_dataset(
        name="HuggingFaceFW/fineweb-edu",
        config="default",
        split="train",
        streaming=True,
        num_samples=100000
    )

    # C4 (200k samples) - Colossal Clean Crawled Corpus
    total_downloaded += download_dataset(
        name="allenai/c4",
        config="en",
        split="train",
        streaming=True,
        num_samples=200000
    )

    # Dolma (150k samples) - AI2's diverse corpus
    total_downloaded += download_dataset(
        name="allenai/dolma",
        config="v1_6-sample",
        split="train",
        streaming=True,
        num_samples=150000
    )

    # ========================================================================
    # TIER 2: HIGH-QUALITY INSTRUCTION DATA
    # ========================================================================

    print("\n📚 TIER 2: HIGH-QUALITY INSTRUCTION DATA")

    # Ultrachat 200k - High-quality multi-turn conversations
    total_downloaded += download_dataset(
        name="HuggingFaceH4/ultrachat_200k",
        config="default",
        split="train_sft",
        streaming=False,
        num_samples=None
    )

    # LIMA - Less is More for Alignment (1k high-quality examples)
    total_downloaded += download_dataset(
        name="GAIR/lima",
        config="default",
        split="train",
        streaming=False,
        num_samples=None
    )

    # Orca Math (200k samples) - Math reasoning
    total_downloaded += download_dataset(
        name="microsoft/orca-math-word-problems-200k",
        config="default",
        split="train",
        streaming=False,
        num_samples=None
    )

    # SlimOrca (500k samples) - Distilled reasoning
    total_downloaded += download_dataset(
        name="Open-Orca/SlimOrca",
        config="default",
        split="train",
        streaming=True,
        num_samples=500000
    )

    # ========================================================================
    # TIER 3: CODE DATASETS
    # ========================================================================

    print("\n💻 TIER 3: HIGH-QUALITY CODE")

    # StarCoder Data (200k samples) - Clean code
    total_downloaded += download_dataset(
        name="bigcode/starcoderdata",
        config="python",
        split="train",
        streaming=True,
        num_samples=200000
    )

    # The Stack v2 - Python (150k samples)
    total_downloaded += download_dataset(
        name="bigcode/the-stack-v2",
        config="python",
        split="train",
        streaming=True,
        num_samples=150000
    )

    # CodeSearchNet (all) - Docstrings + code
    total_downloaded += download_dataset(
        name="code_search_net",
        config="python",
        split="train",
        streaming=False,
        num_samples=None
    )

    # ========================================================================
    # TIER 4: REASONING & KNOWLEDGE
    # ========================================================================

    print("\n🧠 TIER 4: REASONING & KNOWLEDGE")

    # Wiki (English Wikipedia) - 6M articles, sample 200k
    total_downloaded += download_dataset(
        name="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        streaming=True,
        num_samples=200000
    )

    # BookCorpus (74k books)
    total_downloaded += download_dataset(
        name="bookcorpus",
        config="default",
        split="train",
        streaming=True,
        num_samples=100000
    )

    # PubMed Abstracts (100k samples) - Scientific text
    total_downloaded += download_dataset(
        name="pubmed",
        config="default",
        split="train",
        streaming=True,
        num_samples=100000
    )

    # ========================================================================
    # TIER 5: CONVERSATIONAL & CHAT
    # ========================================================================

    print("\n💬 TIER 5: CONVERSATIONAL DATA")

    # ShareGPT (90k samples) - Real ChatGPT conversations
    total_downloaded += download_dataset(
        name="anon8231489123/ShareGPT_Vicuna_unfiltered",
        config="default",
        split="train",
        streaming=True,
        num_samples=90000
    )

    # Anthropic HH-RLHF (expanded, 200k samples)
    total_downloaded += download_dataset(
        name="Anthropic/hh-rlhf",
        config="default",
        split="train",
        streaming=False,
        num_samples=None
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total downloaded: {total_downloaded:,} examples")
    print(f"Existing data: ~773,000 examples")
    print(f"New total: ~{total_downloaded + 773000:,} examples")
    print("="*80 + "\n")

    # Estimate tokens
    avg_tokens_per_example = 250
    total_tokens = (total_downloaded + 773000) * avg_tokens_per_example
    print(f"Estimated tokens: ~{total_tokens/1e6:.1f}M tokens")
    print(f"Target: 500M-1B tokens")

    if total_tokens >= 500e6:
        print("\n✅ TARGET REACHED! You have enough data for high-quality training.")
    else:
        print(f"\n⚠️  Need {(500e6 - total_tokens)/1e6:.1f}M more tokens to reach 500M target")

    print("\nNext steps:")
    print("1. Run: python code/scripts/data_prep/process_high_quality_datasets.py")
    print("2. Update training config to use expanded dataset")
    print("3. Restart training with more data\n")


if __name__ == "__main__":
    main()
