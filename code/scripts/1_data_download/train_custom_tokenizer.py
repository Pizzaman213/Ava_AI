#!/usr/bin/env python3
"""
Train a custom BPE tokenizer from datasets.
This tokenizer will be used for all pre-tokenization tasks.

Usage:
  python train_custom_tokenizer.py --vocab-size 16000 --add-eos-token true
  python train_custom_tokenizer.py --help
"""

import json
import argparse
from pathlib import Path
from typing import List
import sys

def train_custom_bpe_tokenizer(
    source_dir: Path,
    output_path: Path,
    vocab_size: int = 16000,
    add_eos_token: bool = True,
    eos_token: str = "[EOS]"
):
    """Train a BPE tokenizer from raw data files

    Args:
        source_dir: Directory containing training data
        output_path: Where to save the tokenizer
        vocab_size: Size of the vocabulary (default: 16000)
        add_eos_token: Whether to add EOS token (default: True)
        eos_token: EOS token string (default: "[EOS]")
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f" TRAINING CUSTOM BPE TOKENIZER ({vocab_size} vocab)")
    print("="*70)
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_path}")
    print(f"Vocab:   {vocab_size}\n")

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace, ByteLevel
        from tokenizers.processors import ByteLevel as ByteLevelProcessor

        # Collect text files
        print("Collecting text files...")
        text_files = []
        for path in source_dir.rglob("*"):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.txt']:
                text_files.append(str(path))

        if not text_files:
            print("❌ No text files found for training!")
            return False

        print(f"✓ Found {len(text_files)} text files\n")

        # Create tokenizer
        print("Creating BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Set up pre-tokenizer and post-processor
        tokenizer.pre_tokenizer = Whitespace()

        # Define special tokens
        special_tokens = [
            "[PAD]",    # Padding
            "[UNK]",    # Unknown
            "[CLS]",    # Classification
            "[SEP]",    # Separator
            "[MASK]"    # Mask
        ]

        if add_eos_token:
            special_tokens.append(eos_token)

        # Train the tokenizer
        print(f"Training BPE (vocab_size={vocab_size})...")
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            min_frequency=2
        )

        tokenizer.train(text_files, trainer)

        # Verify vocab size
        if len(tokenizer.get_vocab()) != vocab_size:
            actual_vocab = len(tokenizer.get_vocab())
            print(f"⚠️  Vocab size: expected {vocab_size}, got {actual_vocab}")

        print(f"✓ Tokenizer trained\n")

        # Save tokenizer.json
        print("Saving tokenizer...")
        tokenizer_json_path = output_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_json_path))
        print(f"✓ Saved: {tokenizer_json_path}")

        # Create tokenizer_config.json for HuggingFace
        config = {
            "add_bos_token": True,
            "add_eos_token": add_eos_token,
            "add_prefix_space": False,
            "bos_token": "[CLS]",
            "cls_token": "[CLS]",
            "clean_up_tokenization_spaces": True,
            "eos_token": eos_token if add_eos_token else "[SEP]",
            "errors": "replace",
            "mask_token": "[MASK]",
            "model_max_length": 2048,
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "spaces_between_special_tokens": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "[UNK]"
        }

        config_path = output_path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved: {config_path}")

        # Create special_tokens_map.json
        special_tokens_map = {
            "bos_token": "[CLS]",
            "cls_token": "[CLS]",
            "eos_token": eos_token if add_eos_token else "[SEP]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]"
        }

        special_tokens_path = output_path / "special_tokens_map.json"
        with open(special_tokens_path, 'w') as f:
            json.dump(special_tokens_map, f, indent=2)
        print(f"✓ Saved: {special_tokens_path}")

        # Test loading the tokenizer
        print("\nTesting tokenizer...")
        from transformers import AutoTokenizer
        test_tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        test_vocab_size = len(test_tokenizer)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {test_vocab_size}")
        print(f"  Special tokens: {test_tokenizer.special_tokens_map}")

        if test_vocab_size != vocab_size:
            print(f"⚠️  WARNING: Vocab size mismatch! Expected {vocab_size}, got {test_vocab_size}")

        print(f"\n✓ Custom BPE tokenizer created successfully!\n")
        return True

    except Exception as e:
        print(f"\n❌ Failed to train tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train custom BPE tokenizer from datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 16k vocab with EOS token
  python train_custom_tokenizer.py

  # Custom vocab size
  python train_custom_tokenizer.py --vocab-size 32000

  # Without EOS token
  python train_custom_tokenizer.py --no-eos-token

  # Custom EOS token string
  python train_custom_tokenizer.py --vocab-size 8000 --eos-token "[END]"

  # Custom output directory
  python train_custom_tokenizer.py --output-dir /path/to/tokenizer
        """
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Vocabulary size for the tokenizer (default: 16000)"
    )

    parser.add_argument(
        "--add-eos-token",
        "--eos-token",
        type=str,
        default="[EOS]",
        help="EOS token string to add (default: [EOS], use 'none' to skip)"
    )

    parser.add_argument(
        "--no-eos-token",
        action="store_true",
        help="Don't add EOS token"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for tokenizer (default: pretokenized_data/tokenizers/{vocab_size//1000}k)"
    )

    args = parser.parse_args()

    # Find data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[2]  # Go up to project root
    source_dir = project_root / "code" / "data"

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        vocab_label = f"{args.vocab_size // 1000}k" if args.vocab_size >= 1000 else f"{args.vocab_size}"
        output_dir = project_root / "pretokenized_data" / "tokenizers" / vocab_label

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print(f"   Looking in: {source_dir}")
        sys.exit(1)

    # Determine EOS token
    add_eos = not args.no_eos_token
    eos_token = args.add_eos_token if add_eos and args.add_eos_token.lower() != "none" else "[EOS]"

    success = train_custom_bpe_tokenizer(
        source_dir,
        output_dir,
        vocab_size=args.vocab_size,
        add_eos_token=add_eos,
        eos_token=eos_token
    )
    sys.exit(0 if success else 1)