#!/usr/bin/env python3
"""
Build complete pre-tokenized data repository.

Creates a clean `/pretokenized_data/` folder with:
- Tokenizer/ (configurable tokenizers)
- Data/ (all tokenized datasets)

Features:
- Train new tokenizers from raw data
- Re-tokenize all datasets
- Organize everything cleanly
- Summary and statistics
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import shutil
import json

def auto_install_requirements():
    """Auto-install requirements if needed"""
    project_root = Path(__file__).resolve().parents[2]
    requirements_file = project_root / "requirements.txt"

    print("🔧 Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file), "--upgrade", "-q"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

try:
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import torch
except ImportError:
    print("Installing dependencies...")
    auto_install_requirements()
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import torch

import argparse
import multiprocessing


# Get CPU count for parallel processing
NUM_CPUS = multiprocessing.cpu_count()



class TokenizerBuilder:
    """Creates and manages tokenizers"""

    @staticmethod
    def copy_tokenizer(source_path: Path, dest_path: Path, tokenizer_name: str = "16k"):
        """Copy existing tokenizer to pre-tokenized folder"""
        source = Path(source_path)
        dest = dest_path / tokenizer_name

        if not source.exists():
            print(f"✗ Tokenizer not found: {source}")
            return False

        print(f"Copying tokenizer: {tokenizer_name}")
        print(f"  From: {source}")
        print(f"  To:   {dest}")

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)
        print(f"✓ Tokenizer copied\n")
        return True

    @staticmethod
    def list_available_tokenizers(tokenizer_dir: Path):
        """List all available tokenizers"""
        if not tokenizer_dir.exists():
            return []

        tokenizers = [d.name for d in tokenizer_dir.iterdir() if d.is_dir()]
        return sorted(tokenizers)


class DataTokenizer:
    """Tokenizes datasets with multiprocessing support and Rust tokenizer"""

    def __init__(self, tokenizer_path: str, num_proc: int = None):
        """Initialize with tokenizer

        Args:
            tokenizer_path: Path to tokenizer
            num_proc: Number of processes for parallel tokenization (default: CPU count)
        """
        self.fast_tokenizer = None
        self.use_fast = False
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.num_proc = num_proc if num_proc else max(1, NUM_CPUS - 2)  # Leave 2 cores free

    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer - prefer fast Rust tokenizer for 5-10x speedup"""
        from tokenizers import Tokenizer as RustTokenizer

        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer_path = Path(tokenizer_path)
        tokenizer_json = tokenizer_path / "tokenizer.json"

        # Try to load fast Rust tokenizer first (5-10x faster)
        if tokenizer_json.exists():
            try:
                self.fast_tokenizer = RustTokenizer.from_file(str(tokenizer_json))
                self.use_fast = True
                print(f"✓ Loaded FAST Rust tokenizer (vocab: {self.fast_tokenizer.get_vocab_size()})")
            except Exception as e:
                print(f"⚠ Could not load Rust tokenizer: {e}")
                self.fast_tokenizer = None
                self.use_fast = False

        # Also load HuggingFace tokenizer for compatibility
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        except:
            # If that fails, look for tokenizer.json in transformers subdirectory
            transformers_path = tokenizer_path / "transformers"
            if transformers_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(transformers_path))
            else:
                raise ValueError(f"Could not load tokenizer from {tokenizer_path}")

        if not self.use_fast:
            print(f"✓ Tokenizer loaded (HuggingFace). Vocab size: {len(tokenizer)}\n")
        else:
            print(f"  (HuggingFace fallback also loaded)\n")

        return tokenizer

    def _is_image_column(self, col_name: str) -> bool:
        """Check if column name suggests it contains images"""
        image_keywords = ['image', 'img', 'picture', 'photo', 'visual', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp']
        return any(keyword in col_name.lower() for keyword in image_keywords)

    def _is_image_data(self, value) -> bool:
        """Check if value is image data (bytes, PIL Image, etc)"""
        try:
            # Check for PIL Image
            from PIL import Image
            if isinstance(value, Image.Image):
                return True
        except:
            pass

        # Check for bytes/bytearray that look like images
        if isinstance(value, (bytes, bytearray)):
            # Check magic numbers for common image formats
            magic_numbers = [
                b'\x89PNG',      # PNG
                b'\xff\xd8\xff', # JPEG
                b'GIF8',         # GIF
                b'BM',           # BMP
                b'RIFF',         # WEBP/WAV
            ]
            for magic in magic_numbers:
                if value.startswith(magic):
                    return True
            return False

        # Check for dict-like structures (HuggingFace Image feature)
        if isinstance(value, dict):
            if 'bytes' in value or 'path' in value:
                return True

        return False

    def _detect_text_columns(self, dataset) -> List[str]:
        """Detect which columns contain text, filtering out image columns"""
        if not hasattr(dataset, 'column_names'):
            return ["text"]

        text_columns = []
        image_columns_found = []

        for col in dataset.column_names:
            # Skip if column name suggests images
            if self._is_image_column(col):
                image_columns_found.append(col)
                continue

            # Check for known text column names
            if col in ["text", "content", "data", "input", "instruction", "output", "completion"]:
                text_columns.append(col)
                continue

            # Check actual data type
            try:
                sample_value = dataset[0][col]

                # Skip if it's image data
                if self._is_image_data(sample_value):
                    image_columns_found.append(col)
                    continue

                # Include if it's text
                if isinstance(sample_value, str):
                    text_columns.append(col)
            except:
                pass

        # Log filtered columns
        if image_columns_found:
            print(f"  ⊘ Filtered out image columns: {', '.join(image_columns_found)}")

        return text_columns if text_columns else ["text"]

    def _tokenize_batch(self, examples: Dict, text_columns: List[str]) -> Dict:
        """Tokenize a batch of examples - uses fast Rust tokenizer when available"""
        texts = []

        for col in text_columns:
            if col in examples:
                col_texts = examples[col]
                if isinstance(col_texts, str):
                    col_texts = [col_texts]
                texts.extend(col_texts)

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        # Use fast Rust tokenizer if available (5-10x faster)
        if self.use_fast and self.fast_tokenizer is not None:
            encodings = self.fast_tokenizer.encode_batch(texts)
            return {
                "input_ids": [e.ids for e in encodings],
                "attention_mask": [e.attention_mask for e in encodings],
            }

        # Fallback to HuggingFace tokenizer
        encodings = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=True
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings.get("attention_mask", []),
        }

    def _tokenize_and_save_streaming(self, dataset, output_path: Path, text_columns: List[str], shard_size: int = 100000):
        """True streaming tokenization - never loads full dataset into RAM"""
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import time

        start = time.time()
        output_path.mkdir(parents=True, exist_ok=True)

        # Load fast tokenizer for this worker
        from tokenizers import Tokenizer as RustTokenizer
        tokenizer_path = Path(self.tokenizer.name_or_path) / "tokenizer.json"
        fast_tok = RustTokenizer.from_file(str(tokenizer_path))

        print(f"  TRUE STREAMING MODE (constant ~500MB RAM)")
        print(f"  Shard size: {shard_size:,} sequences per file")

        # Schema for Arrow output
        schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('attention_mask', pa.list_(pa.int32()))
        ])

        # Streaming state
        shard_idx = 0
        total_sequences = 0
        buffer_ids = []
        buffer_masks = []
        batch_size = 1000  # Tokenize in batches for efficiency
        text_buffer = []

        def flush_shard():
            """Write current buffer to disk as Arrow shard"""
            nonlocal shard_idx, buffer_ids, buffer_masks, total_sequences
            if not buffer_ids:
                return

            shard_file = output_path / f"data-{shard_idx:05d}.arrow"
            table = pa.Table.from_pydict({
                'input_ids': buffer_ids,
                'attention_mask': buffer_masks
            }, schema=schema)

            with open(str(shard_file), 'wb') as f:
                writer = ipc.new_stream(f, schema)
                writer.write_table(table)
                writer.close()

            total_sequences += len(buffer_ids)
            shard_idx += 1
            buffer_ids = []
            buffer_masks = []

        def flush_text_buffer():
            """Tokenize text buffer and add to shard buffer"""
            nonlocal text_buffer, buffer_ids, buffer_masks
            if not text_buffer:
                return

            # Batch tokenize
            encodings = fast_tok.encode_batch(text_buffer)
            for enc in encodings:
                buffer_ids.append(enc.ids)
                buffer_masks.append(enc.attention_mask)

            text_buffer = []

            # Flush shard if full
            if len(buffer_ids) >= shard_size:
                flush_shard()

        # Stream through dataset - TRUE STREAMING with iterator
        row_count = 0
        try:
            # Try to get length for progress (may not be available for true streaming)
            total_rows = len(dataset) if hasattr(dataset, '__len__') else None
        except:
            total_rows = None

        # Iterate through dataset row by row
        for row in dataset:
            row_count += 1

            # Extract text from this row
            for col in text_columns:
                if col in row and row[col]:
                    text = str(row[col])
                    if text.strip():
                        text_buffer.append(text)

            # Batch tokenize when buffer is full
            if len(text_buffer) >= batch_size:
                flush_text_buffer()

            # Progress every 10k rows
            if row_count % 10000 == 0:
                if total_rows:
                    pct = (row_count / total_rows) * 100
                    print(f"  [{pct:5.1f}%] {row_count:,} rows, {total_sequences:,} sequences, {shard_idx} shards", end='\r')
                else:
                    print(f"  {row_count:,} rows, {total_sequences:,} sequences, {shard_idx} shards", end='\r')

        # Flush remaining
        flush_text_buffer()
        flush_shard()

        elapsed = time.time() - start
        rate = total_sequences / max(1, elapsed)
        print(f"\n  ✓ COMPLETE: {total_sequences:,} sequences in {elapsed:.1f}s ({rate:,.0f} seq/s)")
        print(f"  Written {shard_idx} shards to {output_path}")

        return shard_idx, total_sequences


# Module-level worker functions (required for multiprocessing)
def _tokenize_chunk_worker(args):
    """Worker: tokenize a chunk of texts"""
    texts, tok_path = args
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    encodings = tok.encode_batch(texts)
    return [(e.ids, e.attention_mask) for e in encodings]


def _write_shard_worker(args):
    """Worker: write a single Arrow shard"""
    import pyarrow as pa
    from pathlib import Path

    idx, ids_chunk, mask_chunk, out_path_str = args
    out_path = Path(out_path_str)
    out_path.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([('input_ids', pa.list_(pa.int32())),
                        ('attention_mask', pa.list_(pa.int32()))])
    table = pa.Table.from_pydict({'input_ids': ids_chunk, 'attention_mask': mask_chunk}, schema=schema)

    shard_file = out_path / f"data-{idx:05d}.arrow"
    with open(str(shard_file), 'wb') as f:
        writer = pa.ipc.new_stream(f, schema)
        writer.write_table(table)
        writer.close()


# Add methods back to DataTokenizer class
DataTokenizer._load_dataset = lambda self, dataset_path: _load_dataset_impl(self, dataset_path)
DataTokenizer._load_text_files = lambda self: None


def _load_dataset_impl(tokenizer_instance, dataset_path: Path) -> Optional[Dataset]:
    """Load dataset in STREAMING mode - never loads full dataset into RAM"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return None

    # Skip image files when loading
    skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg"}

    # Try streaming mode first for memory efficiency
    formats = [
        ("parquet", "Parquet", lambda files: load_dataset("parquet", data_files=files, streaming=True)),
        ("json.gz", "JSON.GZ", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("jsonl.gz", "JSONL.GZ", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("json", "JSON", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("jsonl", "JSONL", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("csv", "CSV", lambda files: load_dataset("csv", data_files=files, streaming=True)),
        ("arrow", "Arrow", lambda files: load_dataset("arrow", data_files=files, streaming=True)),
    ]

    for ext, name, loader in formats:
        try:
            # Find files for this format
            if ext == "json.gz":
                files = [str(f) for f in dataset_path.glob("**/*.json.gz") if f.suffix.lower() not in skip_exts]
            elif ext == "jsonl.gz":
                files = [str(f) for f in dataset_path.glob("**/*.jsonl.gz") if f.suffix.lower() not in skip_exts]
            elif ext == "json":
                files = [str(f) for f in dataset_path.glob("**/*.json") if f.suffix.lower() not in skip_exts and not f.name.endswith('.gz')]
            elif ext == "jsonl":
                files = [str(f) for f in dataset_path.glob("**/*.jsonl") if f.suffix.lower() not in skip_exts and not f.name.endswith('.gz')]
            elif ext == "parquet":
                files = [str(f) for f in dataset_path.glob("**/*.parquet") if f.suffix.lower() not in skip_exts]
            elif ext == "csv":
                files = [str(f) for f in dataset_path.glob("**/*.csv") if f.suffix.lower() not in skip_exts]
            elif ext == "arrow":
                files = [str(f) for f in dataset_path.glob("**/*.arrow") if f.suffix.lower() not in skip_exts]
            else:
                files = [str(f) for f in dataset_path.glob(f"**/*.{ext}")]

            if files:
                print(f"  Format: {name} ({len(files)} files) [STREAMING]")
                dataset = loader(files)
                # Handle DatasetDict vs IterableDataset
                if hasattr(dataset, 'keys') and 'train' in dataset:
                    return dataset['train']
                return dataset
        except Exception as e:
            # If streaming fails, try non-streaming as fallback
            try:
                if files:
                    print(f"  Streaming failed, trying memory-mapped mode...")
                    dataset = load_dataset(ext if ext != "jsonl" else "json", data_files=files)
                    if hasattr(dataset, 'keys') and 'train' in dataset:
                        return dataset['train']
                    return dataset
            except:
                pass
            continue

    return None


def _tokenize_dataset_impl(self, dataset_path: Path, output_path: Path, dataset_name: str = "dataset", skip_images: bool = True, preloaded_dataset=None) -> Tuple[bool, Optional[int]]:
    """Load, tokenize, and save dataset using true streaming"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" TOKENIZING: {dataset_name}")
    print(f"{'='*70}")
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    if skip_images:
        print(f"Image filtering: ENABLED\n")
    else:
        print(f"Image filtering: DISABLED\n")

    try:
        if preloaded_dataset is not None:
            print(f"Using prefetched dataset...")
            dataset = preloaded_dataset
        else:
            print(f"Loading dataset (streaming mode)...")
            dataset = self._load_dataset(dataset_path)

        if dataset is None:
            print(f"Could not load dataset from {dataset_path}")
            return False, None

        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
            print(f"Using 'train' split")

        # Handle both streaming (IterableDataset) and regular datasets
        is_streaming = hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')

        if is_streaming:
            print(f"Dataset: STREAMING MODE (memory efficient)")
            # For streaming, peek at first item to get columns
            first_item = next(iter(dataset.take(1)))
            column_names = list(first_item.keys())
            print(f"  Columns: {', '.join(column_names)}")
        else:
            print(f"Dataset ready")
            column_names = dataset.column_names
            print(f"  Total columns: {len(column_names)}")
            print(f"  Column names: {', '.join(column_names)}")
            try:
                print(f"  Examples: {len(dataset):,}\n")
            except:
                print(f"  Examples: (streaming)\n")

        # Detect text columns
        if skip_images:
            image_keywords = ['image', 'img', 'picture', 'photo', 'visual', 'png', 'jpg']
            text_columns = [c for c in column_names if not any(kw in c.lower() for kw in image_keywords)]
            # Prefer known text columns
            preferred = ['text', 'content', 'data', 'input', 'instruction', 'output', 'completion']
            text_columns = [c for c in text_columns if c in preferred] or text_columns
        else:
            text_columns = column_names

        if not text_columns:
            text_columns = column_names[:1]  # Fallback to first column

        print(f"  Text columns: {', '.join(text_columns)}")
        print(f"  Mode: TRUE STREAMING\n")

        print(f"Tokenizing and saving...")
        num_shards, num_examples = self._tokenize_and_save_streaming(
            dataset, output_path, text_columns, shard_size=100000
        )

        print(f"\nTokenization complete!")
        print(f"  Sequences: {num_examples:,}")
        print(f"  Shards: {num_shards}")
        print(f"  Location: {output_path}\n")

        return True, num_examples

    except Exception as e:
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False, None


# Attach methods to DataTokenizer class
DataTokenizer.tokenize_dataset = _tokenize_dataset_impl


def find_source_datasets(source_dir: Path) -> List[Tuple[Path, str]]:
    """Find ALL datasets in source directory and subdirectories"""
    datasets = []
    skip_dirs = {"Ava_Ai", "models", ".cache", "__pycache__", ".git", ".gitattributes"}
    skip_files = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg"}
    processed_paths = set()

    source_dir = Path(source_dir)

    # Find all directories with data files
    for path in source_dir.rglob("*"):
        if not path.is_dir():
            continue

        # Skip hidden directories and known skip dirs
        if path.name.startswith(".") or path.name in skip_dirs or path.name.endswith("-tokenized"):
            continue

        # Check if this directory has data files
        has_data = False
        for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv", "*.arrow", "*.txt", "*.gz"]:
            files = list(path.glob(ext))
            # Filter out non-data files (images, etc.)
            data_files = [f for f in files if f.suffix.lower() not in skip_files]
            if data_files:
                has_data = True
                break

        if has_data and path not in processed_paths:
            # Create a meaningful name from the directory structure
            rel_path = path.relative_to(source_dir)
            dataset_name = str(rel_path).replace("/", "_").replace("\\", "_")

            datasets.append((path, dataset_name))
            processed_paths.add(path)

    return sorted(datasets, key=lambda x: x[1])


def build_pretokenized_repo(source_dir: Path, output_dir: Path, tokenizer_path: Optional[Path] = None, vocab_size: int = 50680, skip_images: bool = True, num_proc: int = None):
    """Build complete pre-tokenized data repository

    Args:
        source_dir: Directory containing raw data
        output_dir: Output directory for pre-tokenized data
        tokenizer_path: Path to tokenizer (auto-created if not exists)
        vocab_size: Vocabulary size for tokenizer training
        skip_images: Whether to skip image columns during tokenization
        num_proc: Number of processes for parallel tokenization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data subdirectory
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Use provided tokenizer path or default
    if tokenizer_path is None:
        tokenizer_name = f"vocab_{vocab_size}" if vocab_size != 50680 else "vocab"
        tokenizer_path = output_dir / "tokenizers" / tokenizer_name

    print(f"\n{'='*70}")
    print(" BUILDING PRE-TOKENIZED DATA REPOSITORY")
    print(f"{'='*70}")
    print(f"Source:      {source_dir}")
    print(f"Output:      {output_dir}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Data:        {data_dir}\n")

    # Setup tokenizer
    print(f"{'='*70}")
    print(" STEP 1: Training custom tokenizer")
    print(f"{'='*70}\n")

    tokenizer_path = Path(tokenizer_path)

    if tokenizer_path.exists() and (tokenizer_path / "tokenizer.json").exists():
        print(f"✓ Using existing tokenizer at: {tokenizer_path}\n")
    else:
        # Train custom tokenizer from datasets
        print(f"Training custom tokenizer from data...")
        tokenizer_path.mkdir(parents=True, exist_ok=True)

        try:
            from transformers import AutoTokenizer
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import Whitespace
            from tokenizers.trainers import BpeTrainer

            # First, collect all text files from datasets
            print(f"Collecting text data from datasets...")
            text_files = []
            for path in source_dir.rglob("*"):
                if not path.is_dir():
                    # Collect json, jsonl, parquet files
                    if path.suffix in ['.json', '.jsonl']:
                        text_files.append(str(path))

            if not text_files:
                print(f"⚠ No text files found, using fallback tokenizer\n")
                base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                base_tokenizer.save_pretrained(str(tokenizer_path))
                print(f"✓ Fallback tokenizer created\n")
            else:
                print(f"Found {len(text_files)} text files for training")

                # Train BPE tokenizer with configurable vocab size
                tokenizer = Tokenizer(BPE())
                tokenizer.pre_tokenizer = Whitespace()
                trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"])

                print(f"Training BPE tokenizer (vocab size: {vocab_size:,})...")
                tokenizer.train(text_files, trainer)

                # Save the tokenizer
                print(f"Saving tokenizer to: {tokenizer_path}")
                tokenizer.save(str(tokenizer_path / "tokenizer.json"))

                # Create tokenizer config and special tokens map for HF compatibility
                config = {
                    "add_prefix_space": False,
                    "eos_token": "[SEP]",
                    "model_max_length": 512,
                    "pad_token": "[PAD]",
                    "tokenizer_class": "BertTokenizer",
                    "unk_token": "[UNK]"
                }

                import json as json_lib
                with open(tokenizer_path / "tokenizer_config.json", "w") as f:
                    json_lib.dump(config, f, indent=2)

                special_tokens = {
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "pad_token": "[PAD]",
                    "sep_token": "[SEP]",
                    "unk_token": "[UNK]"
                }

                with open(tokenizer_path / "special_tokens_map.json", "w") as f:
                    json_lib.dump(special_tokens, f, indent=2)

                print(f"✓ Custom tokenizer trained and saved successfully\n")
        except Exception as e:
            print(f"✗ Failed to create tokenizer: {e}\n")
            import traceback
            traceback.print_exc()
            return False

    # Step 2: Find datasets
    print(f"{'='*70}")
    print(" STEP 2: Discovering datasets")
    print(f"{'='*70}\n")

    datasets = find_source_datasets(source_dir)

    if not datasets:
        print(f"No datasets found in {source_dir}\n")
        return False

    print(f"Found {len(datasets)} dataset(s):\n")
    for path, name in datasets:
        print(f"  • {name}")
    print()

    # Step 3: Tokenize datasets (true streaming - no prefetch needed)
    print(f"{'='*70}")
    print(" STEP 3: Tokenizing datasets (true streaming)")
    print(f"{'='*70}")

    # Load tokenizer once
    tokenizer = DataTokenizer(str(tokenizer_path), num_proc=num_proc)
    results = {}

    # Process each dataset with streaming
    for dataset_path, dataset_name in datasets:
        output_dataset_path = data_dir / dataset_name
        success, num_examples = tokenizer.tokenize_dataset(
            dataset_path,
            output_dataset_path,
            dataset_name,
            skip_images=skip_images
        )
        results[dataset_name] = (success, num_examples)

    # Step 4: Summary
    print(f"{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}\n")

    success_count = sum(1 for s, _ in results.values() if s)
    total_examples = sum(n for _, n in results.values() if n is not None)

    print(f"Tokenizers:")
    print(f"  Location: {tokenizer_path}")
    tokenizers_dir = tokenizer_path.parent
    available = TokenizerBuilder.list_available_tokenizers(tokenizers_dir)
    for tok in available:
        print(f"  ✓ {tok}")
    print()

    print(f"Datasets:")
    for dataset_name, (success, num_examples) in results.items():
        status = "✓" if success else "✗"
        examples_str = f"{num_examples:,}" if num_examples else "N/A"
        print(f"  {status} {dataset_name}: {examples_str} examples")

    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Total examples: {total_examples:,}")
    print(f"Repository location: {output_dir}\n")

    # Save metadata with vocab size info
    metadata = {
        "created": str(Path.cwd()),
        "vocab_size": vocab_size,
        "skip_images": skip_images,
        "tokenizers": available,
        "datasets": {name: num for name, (success, num) in results.items() if success},
        "total_examples": total_examples,
        "ready_for_training": success_count == len(results),
        "timestamp": str(__import__('datetime').datetime.now().isoformat())
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save a training-ready indicator
    if success_count == len(results):
        with open(output_dir / ".training_ready", "w") as f:
            f.write(f"vocab_size={vocab_size}\n")
            f.write(f"total_examples={total_examples}\n")
            f.write(f"datasets={success_count}\n")

    return success_count == len(results)


def find_latest_moe_config(config_dir: Path) -> Optional[Path]:
    """Find the latest MOE config file"""
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return None

    moe_configs = sorted(config_dir.glob("*.yaml"), key=lambda x: x.stat().st_mtime, reverse=True)
    return moe_configs[0] if moe_configs else None


def load_vocab_size_from_config(config_path: Path) -> Optional[int]:
    """Load vocab_size from MOE config YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        vocab_size = config.get('model', {}).get('vocab_size')
        if vocab_size:
            print(f"✓ Loaded vocab_size from config: {vocab_size}")
            return vocab_size
    except Exception as e:
        print(f"⚠ Could not load vocab_size from config: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build pre-tokenized data repository with configurable tokenizer size"
    )
    parser.add_argument("--source-dir", default="/root/Ava_AI/code/data",
                       help="Source data directory (default: /root/Ava_AI/code/data)")
    parser.add_argument("--output-dir", default="/root/Ava_AI/pretokenized_data",
                       help="Output directory (default: /root/Ava_AI/pretokenized_data)")
    parser.add_argument("--tokenizer-path",
                       default=None,
                       help="Path to tokenizer (auto-discovered if not provided)")
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Tokenizer vocabulary size (reads from config if not provided)")
    parser.add_argument("--config", type=str, default=None,
                       help="MOE config file to extract vocab_size from (e.g., code/configs/moe/large.yaml)")
    parser.add_argument("--tokenizer-name", default="vocab",
                       help="Name for tokenizer directory (default: vocab)")
    parser.add_argument("--auto-discover-config", action="store_true",
                       help="Auto-discover latest config from code/configs/moe/")
    parser.add_argument("--skip-images", action="store_true", default=True,
                       help="Skip image columns during tokenization (default: True)")
    parser.add_argument("--include-images", action="store_true",
                       help="Include image columns (overrides --skip-images)")
    parser.add_argument("--num-proc", type=int, default=None,
                       help=f"Number of processes for parallel tokenization (default: {NUM_CPUS - 2})")

    args = parser.parse_args()

    # Determine vocab size with priority order
    vocab_size = args.vocab_size
    config_source = None

    # Priority 1: Explicit --vocab-size argument
    if vocab_size is None:
        # Priority 2: Load from --config file
        if args.config:
            config_path = Path(args.config)
            vocab_size = load_vocab_size_from_config(config_path)
            config_source = str(config_path)

        # Priority 3: Auto-discover latest config
        if vocab_size is None and args.auto_discover_config:
            config_path = find_latest_moe_config(Path("/root/Ava_AI/code/configs/moe"))
            if config_path:
                print(f"Auto-discovered config: {config_path}")
                vocab_size = load_vocab_size_from_config(config_path)
                config_source = str(config_path)

        # Priority 4: Default value
        if vocab_size is None:
            vocab_size = 50680  # Default from large.yaml
            print(f"⚠ Using default vocab_size: {vocab_size}")

    # Setup tokenizer name and path
    tokenizer_name = f"{args.tokenizer_name}_{vocab_size}" if vocab_size != 50680 else args.tokenizer_name

    if args.tokenizer_path is None:
        args.tokenizer_path = f"/root/Ava_AI/pretokenized_data/tokenizers/{tokenizer_name}"

    # Determine image filtering
    skip_images = True
    if args.include_images:
        skip_images = False
    elif hasattr(args, 'skip_images') and not args.skip_images:
        skip_images = False

    print(f"\n{'='*70}")
    print(" CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Source directory:    {args.source_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Tokenizer path:      {args.tokenizer_path}")
    print(f"Vocabulary size:     {vocab_size}")
    print(f"Image filtering:     {'ENABLED ⊘' if skip_images else 'DISABLED'}")
    print(f"Num processes:       {args.num_proc if args.num_proc else NUM_CPUS - 2}")
    if config_source:
        print(f"Config source:       {config_source}")
    print(f"{'='*70}\n")

    try:
        success = build_pretokenized_repo(
            Path(args.source_dir),
            Path(args.output_dir),
            tokenizer_path=Path(args.tokenizer_path),
            vocab_size=vocab_size,
            skip_images=skip_images,
            num_proc=args.num_proc
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠ Build cancelled by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
