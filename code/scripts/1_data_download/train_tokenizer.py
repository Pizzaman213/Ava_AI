#!/usr/bin/env python3
"""
Train a custom tokenizer for Ava from scratch.

This script trains a BPE (Byte-Pair Encoding) tokenizer optimized for your data.

Features:
- Configurable vocabulary size
- Proper special tokens (PAD, EOS, BOS, UNK)
- Trained on TinyStories or custom datasets
- Compatible with Hugging Face ecosystem
- Saves in both Tokenizers and Transformers formats

Usage:
    # Basic: Train on TinyStories
    python code/scripts/1_data_download/train_tokenizer.py

    # Custom vocab size
    python code/scripts/1_data_download/train_tokenizer.py --vocab-size 32000

    # Train on custom dataset
    python code/scripts/1_data_download/train_tokenizer.py --dataset roneneldan/TinyStories

    # Train on local text files
    python code/scripts/1_data_download/train_tokenizer.py --text-files "data/*.txt"
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import glob


def get_training_corpus(
    dataset_name: str = "roneneldan/TinyStories",
    text_column: str = "text",
    split: str = "train",
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    text_files: Optional[str] = None,
) -> Iterator[List[str]]:
    """
    Create an iterator over the training corpus.

    Args:
        dataset_name: HuggingFace dataset name
        text_column: Column containing text data
        split: Dataset split to use
        max_samples: Maximum samples to use (None = all)
        batch_size: Batch size for iteration
        text_files: Pattern for local text files (e.g., "data/*.txt")

    Yields:
        Batches of text strings
    """
    if text_files:
        # Load from local files
        print(f"Loading text from files: {text_files}")
        files = glob.glob(text_files)
        print(f"Found {len(files)} files")

        texts = []
        for file_path in tqdm(files, desc="Reading files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        if len(texts) >= batch_size:
                            yield texts
                            texts = []
                            if max_samples and len(texts) >= max_samples:
                                return

        if texts:
            yield texts
    else:
        # Load from HuggingFace
        print(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to TinyStories...")
            dataset = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)

        # Limit samples if requested
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Training on {len(dataset):,} samples")

        # Yield in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if text_column in batch:
                yield batch[text_column]
            elif 'text' in batch:
                yield batch['text']
            else:
                raise ValueError(f"Could not find text column. Available: {list(batch.keys())}")


def analyze_corpus_stats(corpus_iterator: Iterator[List[str]], num_samples: int = 10000):
    """Analyze corpus to determine good tokenizer parameters."""
    print("\nAnalyzing corpus statistics...")

    total_chars = 0
    total_words = 0
    total_samples = 0
    unique_chars = set()

    for batch in corpus_iterator:
        for text in batch[:num_samples - total_samples]:
            total_chars += len(text)
            total_words += len(text.split())
            unique_chars.update(text)
            total_samples += 1

            if total_samples >= num_samples:
                break

        if total_samples >= num_samples:
            break

    avg_chars_per_sample = total_chars / max(total_samples, 1)
    avg_words_per_sample = total_words / max(total_samples, 1)
    avg_chars_per_word = total_chars / max(total_words, 1)

    print(f"\nCorpus Statistics (from {total_samples:,} samples):")
    print(f"  Unique characters: {len(unique_chars)}")
    print(f"  Avg chars/sample:  {avg_chars_per_sample:.1f}")
    print(f"  Avg words/sample:  {avg_words_per_sample:.1f}")
    print(f"  Avg chars/word:    {avg_chars_per_word:.1f}")

    # Recommendations
    if avg_chars_per_word < 4:
        print("\n💡 Recommendation: Use smaller vocab (16K-24K) for simple text")
    elif avg_chars_per_word > 6:
        print("\n💡 Recommendation: Use larger vocab (32K-64K) for complex text")
    else:
        print("\n💡 Recommendation: Use medium vocab (24K-32K) for general text")

    return {
        'unique_chars': len(unique_chars),
        'avg_chars_per_sample': avg_chars_per_sample,
        'avg_words_per_sample': avg_words_per_sample,
        'avg_chars_per_word': avg_chars_per_word,
    }


def train_tokenizer(
    output_dir: str,
    vocab_size: int = 50000,
    dataset_name: str = "roneneldan/TinyStories",
    text_files: Optional[str] = None,
    min_frequency: int = 2,
    max_samples: Optional[int] = None,
    special_tokens: Optional[List[str]] = None,
    lowercase: bool = False,
    analyze_stats: bool = True,
):
    """
    Train a BPE tokenizer.

    Args:
        output_dir: Directory to save tokenizer
        vocab_size: Target vocabulary size
        dataset_name: HuggingFace dataset name
        text_files: Pattern for local text files
        min_frequency: Minimum frequency for subword to be added
        max_samples: Maximum samples to train on
        special_tokens: List of special tokens
        lowercase: Whether to lowercase text
        analyze_stats: Whether to analyze corpus before training
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Tokenizer Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  Dataset: {dataset_name if not text_files else text_files}")
    print(f"  Lowercase: {lowercase}")
    print(f"  Min frequency: {min_frequency}")

    # Special tokens (must match model config!)
    if special_tokens is None:
        special_tokens = [
            "[PAD]",  # ID 0 - Padding
            "[EOS]",  # ID 1 - End of sequence
            "[BOS]",  # ID 2 - Beginning of sequence
            "[UNK]",  # ID 3 - Unknown token
        ]

    print(f"\nSpecial tokens (in order):")
    for i, token in enumerate(special_tokens):
        print(f"  {i}: {token}")

    # Initialize tokenizer
    print("\n" + "=" * 70)
    print("Step 1: Initialize BPE Tokenizer")
    print("=" * 70)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalizer
    normalizer_sequence = [NFD(), StripAccents()]
    if lowercase:
        normalizer_sequence.insert(0, Lowercase())

    tokenizer.normalizer = normalizers.Sequence(normalizer_sequence)

    # Pre-tokenizer: split on whitespace and handle bytes
    # IMPORTANT: add_prefix_space=True allows proper space reconstruction during decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        Whitespace(),
        ByteLevel(add_prefix_space=True)  # Fixed: was False, now True
    ])

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    # Analyze corpus
    if analyze_stats:
        print("\n" + "=" * 70)
        print("Step 2: Corpus Analysis")
        print("=" * 70)
        corpus_iter = get_training_corpus(
            dataset_name=dataset_name,
            text_files=text_files,
            max_samples=10000,  # Sample for stats
        )
        stats = analyze_corpus_stats(corpus_iter)

    # Train tokenizer
    print("\n" + "=" * 70)
    print("Step 3: Train Tokenizer")
    print("=" * 70)
    print("\nThis may take several minutes...")

    corpus_iter = get_training_corpus(
        dataset_name=dataset_name,
        text_files=text_files,
        max_samples=max_samples,
    )

    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    print(f"\n✓ Training complete!")
    print(f"  Final vocab size: {tokenizer.get_vocab_size():,}")

    # Save tokenizer
    print("\n" + "=" * 70)
    print("Step 4: Save Tokenizer")
    print("=" * 70)

    # Save in Tokenizers format (for Ava)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  ✓ Saved Tokenizers format: {tokenizer_path}")

    # Save in Transformers format (for compatibility)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=512,
    )

    transformers_dir = output_dir / "transformers"
    transformers_dir.mkdir(exist_ok=True)
    fast_tokenizer.save_pretrained(str(transformers_dir))
    print(f"  ✓ Saved Transformers format: {transformers_dir}")

    # Verify special tokens
    print("\n" + "=" * 70)
    print("Step 5: Verification")
    print("=" * 70)

    print("\nSpecial token IDs:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token:10s} = {token_id}")

    # Test encoding/decoding
    test_texts = [
        "Once upon a time, there was a little girl named Lily.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
    ]

    print("\nTest encodings:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        print(f"\n  Original: {text}")
        print(f"  Tokens:   {encoded.tokens[:10]}{'...' if len(encoded.tokens) > 10 else ''}")
        print(f"  IDs:      {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
        print(f"  Length:   {len(encoded.ids)} tokens")
        print(f"  Decoded:  {decoded}")
        print(f"  Match:    {'✓' if text == decoded else '✗ MISMATCH!'}")

    # Vocabulary statistics
    print("\n" + "=" * 70)
    print("Vocabulary Statistics")
    print("=" * 70)

    vocab = tokenizer.get_vocab()
    print(f"  Total tokens: {len(vocab):,}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Regular tokens: {len(vocab) - len(special_tokens):,}")

    # Sample tokens
    print("\n  Sample vocabulary (non-special):")
    non_special = [k for k in list(vocab.keys())[:50] if k not in special_tokens]
    for token in non_special[:20]:
        print(f"    {vocab[token]:5d}: {repr(token)}")

    # Usage instructions
    print("\n" + "=" * 70)
    print("DONE! Usage Instructions")
    print("=" * 70)
    print(f"\n1. Update your config to use this tokenizer:")
    print(f"   tokenizer_name: {tokenizer_path}")
    print(f"\n2. Verify token IDs match your model config:")
    print(f"   model:")
    print(f"     pad_token_id: 0  # Must match [PAD]")
    print(f"     eos_token_id: 1  # Must match [EOS]")
    print(f"     bos_token_id: 2  # Must match [BOS]")
    print(f"     vocab_size: {tokenizer.get_vocab_size()}")
    print(f"\n3. Re-tokenize your data with new tokenizer:")
    print(f"   python code/scripts/1_data_download/download_tinystories_improved.py")
    print(f"\n4. Start training:")
    print(f"   python code/scripts/5_training/train_pipeline.py --config code/configs/moe/minimal_working.yaml")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a custom tokenizer")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/Ava_AI/code/data/Ava_Ai/tokenizer_v2",
        help="Output directory for tokenizer"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Target vocabulary size (default: 50000)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--text-files",
        type=str,
        default=None,
        help="Pattern for local text files (e.g., 'data/*.txt')"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for subword (default: 2)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to train on (default: all)"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip corpus statistics analysis"
    )

    args = parser.parse_args()

    train_tokenizer(
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        dataset_name=args.dataset,
        text_files=args.text_files,
        min_frequency=args.min_frequency,
        max_samples=args.max_samples,
        lowercase=args.lowercase,
        analyze_stats=not args.no_stats,
    )


if __name__ == "__main__":
    main()
