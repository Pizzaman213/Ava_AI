"""
Optimized streaming data loader for Ava MoE++ training
Efficiently handles large datasets with improved performance and organization

Key improvements over data_streaming.py:
- Better file I/O with memory mapping
- Efficient caching layer
- Reduced memory footprint
- Cleaner separation of concerns
- Enhanced error handling
"""

import os
import json
import hashlib
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple, Callable, Any
import pyarrow as pa
import pyarrow.parquet as pq
import random
from collections import defaultdict

# Removed dependencies - using simplified standalone implementation

# Distributed training imports
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class LengthBasedBucketing:
    """
    Optimized length-based bucketing for efficient batch formation.

    Groups samples by sequence length to minimize padding overhead and
    improve GPU utilization during training.
    """

    def __init__(
        self,
        bucket_boundaries: Optional[List[int]] = None,
        max_bucket_size: int = 100,
        min_bucket_size: int = 8,
        enable_bucketing: bool = True
    ):
        self.enable_bucketing = enable_bucketing
        self.max_bucket_size = max_bucket_size
        self.min_bucket_size = min_bucket_size

        # Optimized default boundaries based on common sequence lengths
        if bucket_boundaries is None:
            self.bucket_boundaries = [64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.bucket_boundaries = sorted(bucket_boundaries)

        # Use defaultdict for cleaner code
        self.buckets: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        self.bucket_stats: Dict[int, int] = defaultdict(int)

    def get_bucket_id(self, sequence_length: int) -> int:
        """Get bucket ID for a given sequence length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if sequence_length <= boundary:
                return i
        return len(self.bucket_boundaries) - 1

    def add_sample(self, sample: Dict[str, torch.Tensor]) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Add sample to appropriate bucket and return full bucket if ready.

        Returns:
            List of samples if bucket is full, None otherwise
        """
        if not self.enable_bucketing:
            return [sample]

        # Extract sequence length
        if 'input_ids' not in sample:
            return [sample]

        seq_length = sample['input_ids'].size(0) if sample['input_ids'].dim() == 1 else sample['input_ids'].size(1)
        bucket_id = self.get_bucket_id(seq_length)

        self.buckets[bucket_id].append(sample)
        self.bucket_stats[bucket_id] += 1

        # Return full bucket if threshold reached
        if len(self.buckets[bucket_id]) >= self.max_bucket_size:
            full_bucket = self.buckets[bucket_id].copy()
            self.buckets[bucket_id].clear()
            return full_bucket

        return None

    def flush_buckets(self, min_size: Optional[int] = None) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """Flush all buckets meeting minimum size threshold."""
        min_size = min_size or self.min_bucket_size

        for bucket_id, samples in self.buckets.items():
            if len(samples) >= min_size:
                yield samples.copy()
                samples.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get bucketing statistics for monitoring."""
        total_samples = sum(self.bucket_stats.values())
        bucket_distribution = {}

        for bucket_id, count in self.bucket_stats.items():
            if bucket_id < len(self.bucket_boundaries):
                max_len = self.bucket_boundaries[bucket_id]
                min_len = self.bucket_boundaries[bucket_id - 1] + 1 if bucket_id > 0 else 1
                bucket_name = f"{min_len}-{max_len}"
            else:
                bucket_name = f">{self.bucket_boundaries[-1]}"

            bucket_distribution[bucket_name] = {
                'count': count,
                'percentage': (count / total_samples * 100) if total_samples > 0 else 0
            }

        return {
            'total_samples': total_samples,
            'bucket_distribution': bucket_distribution,
            'active_buckets': len([b for b in self.buckets.values() if len(b) > 0]),
            'samples_in_buckets': sum(len(b) for b in self.buckets.values())
        }


class FileReader:
    """Optimized file reader with format detection and caching."""

    def __init__(self):
        self._format_cache: Dict[Path, str] = {}

    def detect_format(self, file_path: Path) -> str:
        """Detect file format with caching."""
        if file_path in self._format_cache:
            return self._format_cache[file_path]

        suffix = file_path.suffix.lower()
        self._format_cache[file_path] = suffix
        return suffix

    def read_file(self, file_path: Path) -> Iterator[str]:
        """
        Read file with optimized format-specific handling.

        Supports: .arrow, .parquet, .jsonl with robust error handling.
        """
        if not file_path.exists():
            return iter([])

        try:
            format_type = self.detect_format(file_path)

            if format_type == '.arrow':
                yield from self._read_arrow(file_path)
            elif format_type == '.parquet':
                yield from self._read_parquet(file_path)
            elif format_type == '.jsonl':
                yield from self._read_jsonl(file_path)
            else:
                print(f"⚠️  Unsupported format: {format_type} for {file_path.name}")

        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")

    def _read_arrow(self, file_path: Path) -> Iterator[str]:
        """Read Arrow files efficiently."""
        try:
            table = pa.ipc.RecordBatchFileReader(pa.memory_map(str(file_path), 'r')).read_all()
            df = table.to_pandas()
            if 'text' in df.columns:
                for text in df['text']:
                    if text and len(str(text).strip()) > 10:
                        yield str(text).strip()
        except Exception as e:
            print(f"⚠️  Failed to read Arrow file {file_path.name}: {e}")

    def _read_parquet(self, file_path: Path) -> Iterator[str]:
        """Read Parquet files in optimized batches."""
        try:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                if 'text' in df.columns:
                    for text in df['text']:
                        if text and len(str(text).strip()) > 10:
                            yield str(text).strip()
        except Exception as e:
            print(f"⚠️  Failed to read Parquet file {file_path.name}: {e}")

    def _read_jsonl(self, file_path: Path) -> Iterator[Any]:
        """Read JSONL with robust encoding handling."""
        line_count = 0
        error_count = 0

        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            line_count += 1
                            try:
                                data = json.loads(line)

                                # Handle pre-tokenized data
                                if 'input_ids' in data and isinstance(data['input_ids'], list):
                                    yield data
                                    continue

                                # Extract text from various fields
                                text = self._extract_text(data)
                                if text and len(text) > 10:
                                    yield text

                            except json.JSONDecodeError as e:
                                error_count += 1
                                if error_count <= 5:  # Only log first few errors
                                    print(f"⚠️  JSON decode error at line {line_count}")
                            except Exception as e:
                                error_count += 1
                    break  # Successfully read with this encoding
                except UnicodeDecodeError:
                    continue  # Try next encoding
        except Exception as e:
            print(f"❌ Failed to read JSONL {file_path.name}: {e}")

    def _extract_text(self, data: Dict) -> Optional[str]:
        """Extract text from JSON data with multiple fallback fields."""
        text_fields = ['text', 'content', 'document', 'passage', 'input', 'question', 'instruction']

        for field in text_fields:
            if field in data and data[field]:
                return str(data[field]).strip()

        # Fallback: look for any string value
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and len(value.strip()) > 10:
                    return value.strip()

        return None


class StreamingDataset(IterableDataset):
    """
    Optimized streaming dataset with memory-efficient data loading.

    Features:
    - Weighted data mixing based on quality scores
    - Length-based bucketing for efficient batching
    - Progressive sequence length training support
    - Distributed training compatibility
    - Robust error handling and validation
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int,
        max_samples: Optional[int] = None,
        buffer_size: int = 1000,
        dynamic_length_fn: Optional[Callable[[], int]] = None,
        enable_bucketing: bool = True,
        bucket_boundaries: Optional[List[int]] = None,
        max_bucket_size: int = 100,
        use_weighted_mixing: bool = True,
        mixing_temperature: float = 1.0,
        data_mixer: Optional[Any] = None,
        samples_per_file: int = 1,
        # Data validation parameters
        min_sequence_length: int = 10,
        max_sequence_repetition_rate: float = 0.6,
        max_consecutive_repeats: int = 10,
        skip_malformed_sequences: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.dynamic_length_fn = dynamic_length_fn
        self.samples_per_file = samples_per_file

        # Initialize file reader
        self.file_reader = FileReader()

        # Store validation parameters
        self.min_sequence_length = min_sequence_length
        self.max_sequence_repetition_rate = max_sequence_repetition_rate
        self.max_consecutive_repeats = max_consecutive_repeats
        self.skip_malformed_sequences = skip_malformed_sequences

        # Initialize bucketing
        self.bucketing = LengthBasedBucketing(
            bucket_boundaries=bucket_boundaries,
            max_bucket_size=max_bucket_size,
            enable_bucketing=enable_bucketing
        )

        # Initialize weighted mixing
        self.use_weighted_mixing = False  # Disabled - removed dependency
        self.data_mixer = None
        print(f"   ⚖️  Uniform data mixing")

        # Find data files
        self.data_files = self._find_data_files()

        # Auto-create validation from training if needed
        if not self.data_files and self.split == "val":
            self._create_val_from_train()

        if not self.data_files:
            print(f"⚠️  No data files found for {split} split, will use synthetic data")
        else:
            print(f"✓ Found {len(self.data_files)} data files for {split} split")

    def _find_data_files(self) -> List[Path]:
        """
        Find and validate data files with deterministic train/val splitting.

        Uses file-based splitting (85% train, 15% val) via deterministic hashing
        to prevent data leakage across workers and ensure reproducibility.
        """
        files = []
        MIN_FILE_SIZE = 10 * 1024  # 10KB minimum

        # Comprehensive file patterns
        patterns = [
            f"*/{self.split}/**/*.arrow",
            f"**/{self.split}/**/*.arrow",
            f"{self.split}_*.arrow",
            f"*/{self.split}/**/*.parquet",
            f"**/{self.split}/**/*.parquet",
            f"{self.split}_*.parquet",
            f"*/{self.split}/**/*.jsonl",
            f"**/{self.split}/**/*.jsonl",
            f"{self.split}_*.jsonl",
            f"{self.split}.jsonl",
        ]

        # Add fallback patterns for train/val splits
        if self.split in ["train", "val"]:
            patterns.extend([
                "*_processed.jsonl",
                "processed*.jsonl",
                "*_processed.arrow",
                "*_processed.parquet",
                "*.jsonl",
                "*.arrow",
                "*.parquet",
            ])

        # Collect all matching files
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))

        # Remove duplicates
        files = list(dict.fromkeys(files))

        # Filter by size and existence
        substantial_files = []
        for f in files:
            try:
                if f.exists() and f.stat().st_size >= MIN_FILE_SIZE:
                    substantial_files.append(f)
            except (OSError, FileNotFoundError):
                pass

        files = substantial_files

        # Apply deterministic train/val split
        if self.split in ["train", "val"] and len(files) > 0:
            files = sorted(files, key=lambda f: f.name)
            split_files = []

            for file_path in files:
                # Use MD5 for consistent hashing
                file_hash = int(hashlib.md5(file_path.name.encode()).hexdigest(), 16) % 100

                if self.split == "train":
                    if file_hash < 85:  # 85% for training
                        split_files.append(file_path)
                else:  # val
                    if file_hash >= 85:  # 15% for validation
                        split_files.append(file_path)

            files = split_files
            print(f"   🔀 File-based split: {len(files)} files for {self.split}")

        if files:
            print(f"   📊 Sample files: {[f.name for f in files[:3]]}")

        return files


    def _create_val_from_train(self):
        """Create validation set from training files if val split is empty."""
        print(f" No validation files found, creating from training files...")
        original_split = self.split
        self.split = "train"
        train_files = self._find_data_files()
        self.split = original_split

        if train_files:
            self.data_files = train_files
            print(f" ✓ Created validation set from {len(self.data_files)} training files")

    def _generate_synthetic_data(self) -> Iterator[str]:
        """Generate synthetic data for testing when no real data is available."""
        import warnings
        warnings.warn(
            f"No data files found! Using synthetic data. Check: {self.data_dir}",
            UserWarning,
            stacklevel=2
        )

        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Deep learning models require large amounts of data.",
            "Natural language processing enables human-computer interaction.",
            "Transformer models have revolutionized NLP tasks.",
        ]

        for _ in range(1000):
            num_sentences = random.randint(3, 8)
            text = " ".join(random.choices(templates, k=num_sentences))
            yield text * random.randint(2, 5)

    def _stream_examples(self, files_to_use=None) -> Iterator[Any]:
        """
        Stream examples with weighted sampling and efficient file rotation.

        Implements round-robin file reading with configurable samples per file
        for optimal I/O efficiency and data diversity.
        """
        files = files_to_use if files_to_use is not None else self.data_files

        # Check worker context
        worker_info = torch.utils.data.get_worker_info()
        should_print = worker_info is None or worker_info.id == 0

        # Rediscover files in worker if needed
        if not files and worker_info is not None:
            files = self._find_data_files()
            if should_print:
                print(f"  🔄 Rediscovered {len(files)} files in worker process")

        if not files:
            yield from self._generate_synthetic_data()
            return

        # Shuffle files
        epoch_num = getattr(self, '_stream_epoch_number', 0)
        shuffled_files = list(files)
        rng = random.Random(42 + epoch_num)
        rng.shuffle(shuffled_files)

        # Open file generators
        file_generators = []
        for file_path in shuffled_files:
            try:
                gen = self.file_reader.read_file(file_path)
                file_generators.append((file_path, gen))
            except Exception as e:
                if len(file_generators) == 0 and should_print:
                    print(f"  ⚠️ Could not open {file_path.name}: {e}")

        if not file_generators:
            if should_print:
                print("  ⚠️ No files could be opened, using synthetic data")
            yield from self._generate_synthetic_data()
            return

        if should_print:
            unique_files = len(set(f[0] for f in file_generators))
            print(f"  📚 Streaming from {unique_files} unique files")

        # Round-robin file reading
        exhausted_files = set()
        restart_count = 0

        while len(exhausted_files) < len(file_generators):
            for idx, (file_path, gen) in enumerate(file_generators):
                if idx in exhausted_files:
                    continue

                # Read batch from this file
                for _ in range(self.samples_per_file):
                    try:
                        yield next(gen)
                    except StopIteration:
                        exhausted_files.add(idx)
                        break

            # Restart if all files exhausted
            if len(exhausted_files) == len(file_generators):
                restart_count += 1
                if restart_count > 3:
                    print(f"  ⚠️ Files empty after {restart_count} restarts, using synthetic data")
                    yield from self._generate_synthetic_data()
                    return

                exhausted_files.clear()
                self._stream_epoch_number = getattr(self, '_stream_epoch_number', 0) + 1

                # Recreate file list with new shuffle
                epoch_num = getattr(self, '_stream_epoch_number', 0)
                rng = random.Random(42 + epoch_num)
                shuffled_files = list(files)
                rng.shuffle(shuffled_files)

                file_generators = []
                for file_path in shuffled_files:
                    try:
                        gen = self.file_reader.read_file(file_path)
                        file_generators.append((file_path, gen))
                    except Exception:
                        pass

    def _tokenize_text(self, text_or_data) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize text or process pre-tokenized data with validation.

        Handles both on-the-fly tokenization and pre-tokenized input_ids.
        Supports dynamic sequence lengths for progressive training.

        Returns:
            Dictionary with input_ids, attention_mask, labels, or None if validation fails
        """
        # Handle pre-tokenized data
        if isinstance(text_or_data, dict) and 'input_ids' in text_or_data:
            input_ids = text_or_data['input_ids']
            attention_mask = text_or_data.get('attention_mask', [1] * len(input_ids))

            # Get current max length
            current_max_length = self._get_current_max_length()

            # Truncate or pad
            if len(input_ids) > current_max_length:
                input_ids = input_ids[:current_max_length]
                attention_mask = attention_mask[:current_max_length]
            elif len(input_ids) < current_max_length:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padding_length = current_max_length - len(input_ids)
                input_ids = input_ids + [pad_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            # Convert to tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

            # Simple validation
            if len(input_ids_tensor) < self.min_sequence_length:
                return None

            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'labels': input_ids_tensor.clone()
            }

        # Tokenize text on-the-fly
        text = text_or_data if isinstance(text_or_data, str) else str(text_or_data)
        current_max_length = self._get_current_max_length()

        encoded = self.tokenizer(
            text,
            max_length=current_max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # Simple validation
        if len(input_ids) < self.min_sequence_length:
            return None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

    def _get_current_max_length(self) -> int:
        """Get current max length with progressive training support."""
        if self.dynamic_length_fn is not None:
            try:
                current_max_length = self.dynamic_length_fn()
            except Exception:
                current_max_length = self.max_length
        else:
            current_max_length = self.max_length

        return max(32, min(current_max_length, self.max_length))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset with buffering, shuffling, and bucketing."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        count = 0
        buffer = []
        samples_processed = 0
        sample_index = 0
        epoch_number = getattr(self, '_epoch_number', 0)

        # Stream examples with worker distribution
        for text in self._stream_examples():
            # Worker-level sample distribution
            if sample_index % num_workers != worker_id:
                sample_index += 1
                continue

            sample_index += 1

            if self.max_samples and count >= self.max_samples:
                break

            buffer.append(text)
            samples_processed += 1

            # Process buffer when full
            if len(buffer) >= self.buffer_size:
                buffer_seed = 42 + epoch_number + (samples_processed // self.buffer_size)
                rng = random.Random(buffer_seed)
                rng.shuffle(buffer)

                for buffered_text in buffer:
                    if self.max_samples and count >= self.max_samples:
                        break

                    tokenized = self._tokenize_text(buffered_text)
                    if tokenized is None:
                        continue

                    if not self.bucketing.enable_bucketing:
                        yield tokenized
                        count += 1
                    else:
                        bucket_samples = self.bucketing.add_sample(tokenized)
                        if bucket_samples is not None:
                            for sample in bucket_samples:
                                yield sample
                                count += 1
                                if self.max_samples and count >= self.max_samples:
                                    return

                buffer = []

                # Periodic bucket flushing
                if self.bucketing.enable_bucketing and samples_processed % (self.buffer_size * 5) == 0:
                    for bucket_samples in self.bucketing.flush_buckets(min_size=8):
                        for sample in bucket_samples:
                            if self.max_samples and count >= self.max_samples:
                                return
                            yield sample
                            count += 1

        # Process remaining buffer
        if buffer:
            buffer_seed = 42 + epoch_number + (samples_processed // max(self.buffer_size, 1))
            rng = random.Random(buffer_seed)
            rng.shuffle(buffer)

            for buffered_text in buffer:
                if self.max_samples and count >= self.max_samples:
                    break

                tokenized = self._tokenize_text(buffered_text)
                if tokenized is None:
                    continue

                if not self.bucketing.enable_bucketing:
                    yield tokenized
                    count += 1
                else:
                    bucket_samples = self.bucketing.add_sample(tokenized)
                    if bucket_samples is not None:
                        for sample in bucket_samples:
                            yield sample
                            count += 1
                            if self.max_samples and count >= self.max_samples:
                                return

        # Flush all remaining buckets
        if self.bucketing.enable_bucketing:
            for bucket_samples in self.bucketing.flush_buckets(min_size=1):
                for sample in bucket_samples:
                    if self.max_samples and count >= self.max_samples:
                        break
                    yield sample
                    count += 1

        # Increment epoch
        self._epoch_number = epoch_number + 1


class InfiniteStreamingDataset(IterableDataset):
    """Infinite streaming dataset for continuous epoch training."""

    def __init__(self, **kwargs):
        self.base_dataset = StreamingDataset(**kwargs)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate infinitely over the dataset."""
        while True:
            for item in self.base_dataset:
                yield item


class DistributedStreamingDataset(IterableDataset):
    """Wrapper for distributed streaming dataset with round-robin distribution."""

    def __init__(self, base_dataset: IterableDataset, world_size: int, rank: int):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        """Iterate with round-robin distribution across ranks."""
        base_iter = iter(self.base_dataset)
        for i, sample in enumerate(base_iter):
            if i % self.world_size == self.rank:
                yield sample


def create_streaming_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
    buffer_size: int = 10000,
    distributed: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    dynamic_length_fn: Optional[Callable[[], int]] = None,
    enable_bucketing: bool = True,
    bucket_boundaries: Optional[List[int]] = None,
    max_bucket_size: int = 100,
    val_max_samples: Optional[int] = None,
    val_split_ratio: float = 0.1,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    use_weighted_mixing: bool = True,
    mixing_temperature: float = 1.0,
    data_mixer: Optional[Any] = None,
    samples_per_file: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized streaming train and validation dataloaders.

    Enhanced with:
    - Weighted data mixing (DoReMi-style quality-based sampling)
    - Efficient prefetching and worker management
    - Distributed training support
    - Progressive training compatibility

    Args:
        tokenizer: Tokenizer for text encoding
        batch_size: Batch size per device
        max_length: Maximum sequence length
        data_dir: Directory containing data files
        num_workers: Number of data loading workers
        max_samples: Maximum training samples (None = unlimited)
        buffer_size: Shuffle buffer size
        distributed: Enable distributed mode
        world_size: Number of distributed processes
        rank: Process rank in distributed training
        dynamic_length_fn: Function returning current sequence length
        enable_bucketing: Enable length-based bucketing
        bucket_boundaries: Custom bucket boundaries
        max_bucket_size: Maximum samples per bucket
        val_max_samples: Maximum validation samples
        val_split_ratio: Validation split ratio
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        use_weighted_mixing: Enable quality-based weighted sampling
        mixing_temperature: Sampling temperature (1.0 = moderate)
        data_mixer: Custom WeightedDataMixer instance
        samples_per_file: Samples per file before rotation (1 = max diversity)

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Safety checks
    if batch_size is None:
        batch_size = 8
        print(f"⚠️  batch_size was None, defaulting to {batch_size}")

    # Auto-detect CPU cores
    if num_workers == -1 or num_workers == 0:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        print(f"🚀 Auto-detected {num_workers} CPU cores")
    elif num_workers > 0:
        print(f"🚀 Using {num_workers} CPU workers for data loading")

    # Auto-detect distributed training
    if distributed is None:
        distributed = DISTRIBUTED_AVAILABLE and (
            'WORLD_SIZE' in os.environ or
            (world_size is not None and world_size > 1)
        )

    if distributed and DISTRIBUTED_AVAILABLE:
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        print(f"📡 Distributed training: rank {rank}/{world_size}")
    else:
        print(f"📦 Creating streaming dataloaders...")

    # Create datasets
    dataset_kwargs = {
        'data_dir': data_dir,
        'tokenizer': tokenizer,
        'max_length': max_length,
        'buffer_size': buffer_size,
        'dynamic_length_fn': dynamic_length_fn,
        'enable_bucketing': enable_bucketing,
        'bucket_boundaries': bucket_boundaries,
        'max_bucket_size': max_bucket_size,
        'use_weighted_mixing': use_weighted_mixing,
        'mixing_temperature': mixing_temperature,
        'data_mixer': data_mixer,
        'samples_per_file': samples_per_file,
    }

    # Training dataset
    if max_samples is None:
        train_dataset = InfiniteStreamingDataset(
            split='train',
            max_samples=None,
            **dataset_kwargs
        )
    else:
        train_dataset = StreamingDataset(
            split='train',
            max_samples=max_samples,
            **dataset_kwargs
        )

    # Validation dataset
    if val_max_samples is not None:
        computed_val_samples = val_max_samples
    elif max_samples is not None:
        computed_val_samples = int(max_samples * val_split_ratio)
    else:
        computed_val_samples = None

    val_buffer_size = buffer_size // 10 if buffer_size >= 10 else buffer_size

    val_dataset = StreamingDataset(
        split='val',
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=computed_val_samples,
        buffer_size=val_buffer_size,
        dynamic_length_fn=dynamic_length_fn,
        enable_bucketing=enable_bucketing,
        bucket_boundaries=bucket_boundaries,
        max_bucket_size=max_bucket_size,
        use_weighted_mixing=False,  # Disable for validation
        mixing_temperature=1.0,
        data_mixer=None,
        samples_per_file=samples_per_file,
    )

    # Dataloader configuration
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': persistent_workers if num_workers > 0 else False
    }

    if num_workers > 0:
        print(f"⚡ Data pipeline optimizations:")
        print(f"   • {num_workers} parallel workers")
        print(f"   • {buffer_size:,} sample buffer")
        print(f"   • {prefetch_factor} batches prefetched per worker")
        print(f"   • Persistent workers: {persistent_workers}")
        print(f"   • Total prefetch: {num_workers * prefetch_factor * batch_size:,} samples")

    # Apply distributed wrapping if needed
    if distributed and DISTRIBUTED_AVAILABLE:
        train_dataset = DistributedStreamingDataset(train_dataset, world_size or 1, rank or 0)
        val_dataset = DistributedStreamingDataset(val_dataset, world_size or 1, rank or 0)

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    return train_loader, val_loader
