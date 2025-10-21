#!/usr/bin/env python3
"""
Enhanced Tokenizer Training for Math/Code Optimization

Based on research findings:
- "Tokenization counts" paper: Proper tokenization improves math by 2-4x
- "Scaling Laws with Vocabulary" paper: Larger vocab for larger models
- SuperBPE: Multi-word tokenization for better code/math handling

This script trains a custom BPE tokenizer optimized for:
1. Mathematical expressions (numbers, operators, LaTeX)
2. Code syntax (indentation, keywords, common patterns)
3. General text (maintaining good compression)

Vocabulary sizes by model size (from research):
- 100M params: 32,000-50,000 tokens (GPT-2 default: 50,257)
- 1B params: 50,000-100,000 tokens
- 10B+ params: 100,000-256,000 tokens

For ultra-fast testing and learning, we'll use 500 tokens for minimal overhead.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from collections import Counter
from multiprocessing import Pool, cpu_count
import re

# Disable tokenizers parallelism warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        decoders,
        trainers,
        normalizers,
        processors,
    )
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Digits
    from tokenizers.normalizers import NFKC
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
except ImportError:
    print("Installing tokenizers library...")
    os.system(f"{sys.executable} -m pip install tokenizers transformers")
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        decoders,
        trainers,
        normalizers,
        processors,
    )
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Digits
    from tokenizers.normalizers import NFKC
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast


# Helper function for parallel processing (must be at module level for pickling)
def _process_json_file(json_file_path):
    """Process a single JSON file and return texts"""
    texts = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle list or single dict
        items = data if isinstance(data, list) else [data]

        for item in items:
            # Extract text from ANY field (flexible approach)
            def extract_all_text(obj, depth=0):
                """Recursively extract all text from nested structure"""
                if depth > 3:  # Limit recursion
                    return []

                result = []
                if isinstance(obj, str) and len(obj.strip()) > 20:
                    result.append(obj.strip())
                elif isinstance(obj, list):
                    for sub_item in obj[:10]:  # Limit list items
                        result.extend(extract_all_text(sub_item, depth+1))
                elif isinstance(obj, dict):
                    for value in list(obj.values())[:10]:  # Limit dict values
                        result.extend(extract_all_text(value, depth+1))
                return result

            texts.extend(extract_all_text(item))

    except (json.JSONDecodeError, Exception):
        pass

    return texts


class EnhancedTokenizerTrainer:
    """Train a custom tokenizer optimized for math, code, and general text"""

    def __init__(
        self,
        vocab_size: int = 500,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize tokenizer trainer.

        Args:
            vocab_size: Target vocabulary size (default: 500 for ultra-fast testing)
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens (default: GPT-2 style)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Enhanced special tokens for math/code
        if special_tokens is None:
            self.special_tokens = [
                "<|endoftext|>",  # End of document
                "<|pad|>",         # Padding
                "<|bos|>",         # Beginning of sequence
                "<|eos|>",         # End of sequence
                "<|unk|>",         # Unknown token
                # Math-specific
                "<|math|>",        # Math expression marker
                "<|code|>",        # Code block marker
                "<|latex|>",       # LaTeX expression
                # Special numbers
                "<|num|>",         # Generic number placeholder
                "<|float|>",       # Float number
                "<|int|>",         # Integer number
            ]
        else:
            self.special_tokens = special_tokens

        # Math-specific tokens to ensure they're in vocabulary
        self.math_tokens = [
            # Common operators
            "+=", "-=", "*=", "/=", "//=", "**=", "%=",
            "==", "!=", "<=", ">=", "->", "=>",
            # Math symbols
            "π", "∑", "∫", "∂", "√", "∞", "≈", "≠", "≤", "≥",
            "α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "σ", "ω",
            # LaTeX common
            "\\frac", "\\sum", "\\int", "\\sqrt", "\\partial",
            "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta",
            # Programming keywords
            "def", "class", "import", "from", "return", "if", "else",
            "for", "while", "break", "continue", "try", "except",
            "lambda", "yield", "async", "await",
        ]

        # Code patterns to preserve
        self.code_patterns = [
            "    ",  # 4-space indentation
            "\t",    # Tab
            "#!/",   # Shebang
            "...",   # Ellipsis
            "**kwargs", "**args", "*args",
        ]

    def create_tokenizer(self) -> Tokenizer:
        """Create the base tokenizer with enhanced pre-tokenization"""

        # Use BPE with byte-level encoding (handles all characters)
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

        # Enhanced pre-tokenization for better math/code handling
        # Split on:
        # 1. Whitespace (standard)
        # 2. Digits (preserve number integrity)
        # 3. Byte-level (handle all Unicode)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([  # type: ignore[assignment]
            Digits(individual_digits=False),  # Keep numbers together
            ByteLevel(add_prefix_space=True),  # Byte-level encoding
        ])

        # Normalize Unicode (convert variants to standard form)
        tokenizer.normalizer = normalizers.Sequence([  # type: ignore[assignment]
            NFKC(),  # Canonical decomposition + compatibility composition
        ])

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()  # type: ignore[assignment]

        # Post-processing (add special tokens)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)  # type: ignore[assignment]

        return tokenizer

    def get_training_corpus(
        self,
        data_dir: str = "/project/code/data/processed",
        max_samples_per_dataset: int = 50000,  # Default to 50k per dataset for speed
    ) -> Iterator[str]:
        """
        Get training corpus from processed datasets.

        Yields text samples from all processed datasets.
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all processed data files
        data_files = sorted(data_path.glob("*_processed.jsonl"))

        if not data_files:
            # Try raw data format (directories with train/*.json)
            print(f"   No processed files found, checking for raw data format...")
            dataset_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            if dataset_dirs:
                print(f"📚 Found {len(dataset_dirs)} raw datasets")
                # IMPORTANT: yield from, not return!
                yield from self._get_raw_corpus(data_path, max_samples_per_dataset)
                return
            else:
                raise FileNotFoundError(f"No processed or raw data files found in {data_dir}")

        print(f"📚 Found {len(data_files)} processed datasets")

        total_samples = 0
        for data_file in data_files:
            dataset_name = data_file.stem.replace("_processed", "")
            print(f"   Loading {dataset_name}...")

            samples = 0
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Only limit if max_samples_per_dataset is set
                        if max_samples_per_dataset and samples >= max_samples_per_dataset:
                            break

                        try:
                            data = json.loads(line.strip())

                            # Extract text from various formats
                            text = None
                            if isinstance(data, dict):
                                # Try common text fields
                                for field in ['text', 'content', 'instruction', 'input', 'question', 'prompt']:
                                    if field in data and data[field]:
                                        text = data[field]
                                        break

                                # Try conversation formats
                                if not text and 'messages' in data:
                                    messages = data['messages']
                                    if isinstance(messages, list):
                                        text = " ".join([
                                            msg.get('content', '') for msg in messages
                                            if isinstance(msg, dict)
                                        ])

                                # Concatenate response/output
                                if text and 'output' in data:
                                    text = text + " " + str(data['output'])

                            elif isinstance(data, str):
                                text = data

                            if text and len(text.strip()) > 10:
                                yield text.strip()
                                samples += 1
                                total_samples += 1

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"   ⚠️  Error reading {dataset_name}: {e}")
                continue

            print(f"   ✓ Loaded {samples:,} samples from {dataset_name}")

        print(f"\n📊 Total training samples: {total_samples:,}")

    def _get_raw_corpus(
        self,
        data_path: Path,
        max_samples_per_dataset: int = 50000,  # Default to 50k per dataset for speed
    ) -> Iterator[str]:
        """
        Get training corpus from raw dataset directories with parallel processing.

        Raw format: dataset_name/train/batch_XXXX.json files
        """
        dataset_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        total_samples = 0

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            train_dir = dataset_dir / "train"

            if not train_dir.exists():
                continue

            print(f"   Loading {dataset_name[:60]}...")

            json_files = sorted(train_dir.glob("*.json"))

            # Use multiprocessing for faster file reading
            num_workers = min(cpu_count(), 8)  # Cap at 8 workers

            samples = 0
            with Pool(num_workers) as pool:
                # Process files in parallel using module-level function
                for file_texts in pool.imap_unordered(_process_json_file, json_files, chunksize=10):
                    for text in file_texts:
                        # Only limit if max_samples_per_dataset is set
                        if max_samples_per_dataset and samples >= max_samples_per_dataset:
                            break
                        yield text
                        samples += 1
                        total_samples += 1

                    if max_samples_per_dataset and samples >= max_samples_per_dataset:
                        break

            if samples > 0:
                print(f"   ✓ Loaded {samples:,} samples from {dataset_name}")

        print(f"\n📊 Total training samples: {total_samples:,}")

    def train(
        self,
        data_dir: str = "/project/code/data/processed",
        output_dir: str = "/project/code/models/tokenizer",
        max_samples_per_dataset: int = 50000,  # Default to 50k per dataset for speed
    ) -> Tokenizer:
        """
        Train the tokenizer on your data.

        Args:
            data_dir: Directory containing processed data
            output_dir: Output directory for tokenizer
            max_samples_per_dataset: Max samples to use from each dataset

        Returns:
            Trained tokenizer
        """
        print("="*70)
        print("🚀 Training Enhanced Tokenizer for Math/Code")
        print("="*70)
        print(f"Target vocab size: {self.vocab_size:,}")
        print(f"Min frequency: {self.min_frequency}")
        print(f"Special tokens: {len(self.special_tokens)}")
        print(f"Math tokens: {len(self.math_tokens)}")
        print()

        # Create tokenizer
        tokenizer = self.create_tokenizer()

        # Create trainer with enhanced tokens
        all_special_tokens = (
            self.special_tokens +
            self.math_tokens +
            self.code_patterns
        )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=all_special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Get training corpus
        print("📚 Loading training corpus...")
        corpus = self.get_training_corpus(
            data_dir=data_dir,
            max_samples_per_dataset=max_samples_per_dataset,
        )

        # Train tokenizer - OPTIMIZED: Single pass, no batching
        print("\n🔥 Training tokenizer (single pass for maximum speed)...")

        # OPTIMIZATION: Train directly from iterator without collecting all in memory
        # BPE training is smart enough to handle large datasets efficiently
        print("   Streaming samples directly to trainer (no intermediate collection)...")

        # Use a generator to count and show progress while training
        def counting_iterator(corpus_iter):
            count = 0
            for text in corpus_iter:
                count += 1
                if count % 100000 == 0:
                    print(f"      Processing sample {count:,}...")
                yield text

        # Train in one pass - much faster than batching!
        tokenizer.train_from_iterator(counting_iterator(corpus), trainer=trainer)

        print("\n   ✓ Training complete!")

        print("\n✅ Training complete!")

        # Save tokenizer
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer_file = output_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_file))
        print(f"💾 Saved tokenizer to: {tokenizer_file}")

        # Create HuggingFace-compatible wrapper
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            pad_token="<|pad|>",
            mask_token=None,
            model_max_length=4096,
        )

        # Save wrapped tokenizer
        wrapped_tokenizer.save_pretrained(str(output_path))
        print(f"💾 Saved HuggingFace tokenizer to: {output_path}")

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "math_tokens": self.math_tokens,
            "code_patterns": self.code_patterns,
            "training_data": data_dir,
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata to: {output_path / 'metadata.json'}")

        # Analyze tokenizer
        self.analyze_tokenizer(tokenizer, output_path)

        return tokenizer

    def analyze_tokenizer(self, tokenizer: Tokenizer, output_dir: Path):
        """Analyze tokenizer performance on math/code samples"""

        print("\n" + "="*70)
        print("📊 Tokenizer Analysis")
        print("="*70)

        # Test samples
        test_samples = {
            "General Text": "The quick brown fox jumps over the lazy dog.",
            "Math Expression": "Calculate the derivative: f(x) = x^2 + 2x + 1, f'(x) = 2x + 2",
            "LaTeX Math": r"The quadratic formula is: $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$",
            "Python Code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
            "Numbers": "The value of π is approximately 3.14159, and e ≈ 2.71828",
            "Operators": "x += 1; y *= 2; z //= 3; result != expected",
        }

        analysis_results = {}

        print("\n🧪 Test Tokenization:\n")
        for name, text in test_samples.items():
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            ids = encoding.ids

            # Calculate compression ratio
            chars = len(text)
            token_count = len(tokens)
            ratio = chars / token_count if token_count > 0 else 0

            print(f"📝 {name}:")
            print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"   Tokens ({token_count}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"   Compression: {ratio:.2f} chars/token")
            print()

            analysis_results[name] = {
                "text": text,
                "token_count": token_count,
                "char_count": chars,
                "compression_ratio": ratio,
                "tokens": tokens[:20],  # First 20 tokens
            }

        # Save analysis
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)

        print(f"💾 Saved analysis to: {output_dir / 'analysis.json'}")

        # Vocab statistics
        vocab = tokenizer.get_vocab()
        print(f"\n📈 Vocabulary Statistics:")
        print(f"   Total tokens: {len(vocab):,}")
        print(f"   Special tokens: {len(self.special_tokens)}")
        print(f"   Math tokens: {len(self.math_tokens)}")

        # Check math token coverage
        math_coverage = sum(1 for token in self.math_tokens if token in vocab)
        print(f"   Math token coverage: {math_coverage}/{len(self.math_tokens)} ({math_coverage/len(self.math_tokens)*100:.1f}%)")

        print("\n✅ Analysis complete!")


def main():
    """Main function to train enhanced tokenizer"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Train enhanced tokenizer for math/code optimization"
    )
    parser.add_argument(
        "--use-config",
        action="store_true",
        default=True,
        help="Read vocab size from config file (default: True)"
    )
    parser.add_argument(
        "--no-config",
        action="store_false",
        dest="use_config",
        help="Don't read from config, use --vocab-size instead"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/project/code/configs/gpu/small.yaml",
        help="Config file to read vocab size from (default: /project/code/configs/gpu/small.yaml)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Target vocabulary size (if not specified, reads from config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/project/code/data/processed",
        help="Directory with training data (supports both raw and processed formats)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tokenizer (if not specified, auto-generated from vocab size)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Max samples per dataset (default: 50000 for faster training)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)"
    )

    args = parser.parse_args()

    # Read vocab size and data dir from config if --use-config is set (default)
    if args.use_config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"📖 Reading settings from config: {config_path}")
            with open(config_path) as f:
                config = yaml.safe_load(f)

                # Read vocab size
                if args.vocab_size is None:
                    args.vocab_size = config.get('model', {}).get('vocab_size', 500)
                print(f"   Vocab size from config: {args.vocab_size}")

                # Read data directory from config
                config_data_dir = config.get('data', {}).get('data_dir', None)
                if config_data_dir:
                    args.data_dir = config_data_dir
                    print(f"   Data directory from config: {args.data_dir}")
        else:
            print(f"⚠️  Config file not found: {config_path}, using defaults")
            if args.vocab_size is None:
                args.vocab_size = 500
    else:
        print(f"📝 Using manual settings:")
        if args.vocab_size is None:
            args.vocab_size = 500
        print(f"   Vocab size: {args.vocab_size}")
        print(f"   Data dir: {args.data_dir}")

    # Auto-generate output dir if not specified
    if args.output_dir is None:
        args.output_dir = f"/project/code/models/tokenizer/enhanced-{args.vocab_size}"
        print(f"📁 Auto-generated output dir: {args.output_dir}")

    # Create trainer
    trainer = EnhancedTokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Train tokenizer
    try:
        tokenizer = trainer.train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_samples_per_dataset=args.max_samples,
        )

        # Auto-update config file with tokenizer path
        if args.use_config:
            config_path = Path(args.config)
            if config_path.exists():
                print("\n" + "="*70)
                print("📝 Auto-updating config file...")
                print("="*70)

                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Update tokenizer_name in data section
                if 'data' not in config:
                    config['data'] = {}
                config['data']['tokenizer_name'] = args.output_dir

                # Save updated config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                print(f"✅ Updated {config_path}:")
                print(f"   tokenizer_name: {args.output_dir}")
                print(f"   (vocab_size already set: {args.vocab_size})")

        print("\n" + "="*70)
        print("🎉 SUCCESS! Enhanced tokenizer trained successfully!")
        print("="*70)
        print(f"\nTokenizer saved to: {args.output_dir}")
        print(f"Config automatically updated with tokenizer path!")
        print(f"\nReady to train:")
        print(f"  cd /project/code/scripts/5_training")
        print(f"  python train.py --config {args.config}")
        print("\n" + "="*70)

        return 0

    except Exception as e:
        print(f"\n❌ Error training tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
