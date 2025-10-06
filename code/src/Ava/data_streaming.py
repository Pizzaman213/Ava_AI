"""
Streaming data loader for Ava MoE++ training
Efficiently handles large datasets without loading everything into memory
"""

import os
import json
import torch  # type: ignore[import]
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler  # type: ignore[import]
from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple, Callable, Any
import pyarrow as pa
import pyarrow.parquet as pq
import random
from itertools import cycle, islice
from collections import defaultdict
from .data.arrow_reader import arrow_reader  # type: ignore[import]
from .data.encoding_detector import encoding_detector  # type: ignore[import]

# Distributed training imports
try:
    import torch.distributed as dist  # type: ignore[import]
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class LengthBasedBucketing:
    """
    Length-based bucketing system for efficient training with variable sequence lengths.

    Groups samples by sequence length to minimize padding and improve GPU utilization.
    """

    def __init__(
        self,
        bucket_boundaries: Optional[List[int]] = None,
        max_bucket_size: int = 100,
        min_bucket_size: int = 8,
        enable_bucketing: bool = True
    ):
        """
        Initialize length-based bucketing.

        Args:
            bucket_boundaries: Sequence length boundaries for buckets
            max_bucket_size: Maximum number of samples per bucket
            min_bucket_size: Minimum samples to form a bucket
            enable_bucketing: Whether to enable bucketing
        """
        self.enable_bucketing = enable_bucketing
        self.max_bucket_size = max_bucket_size
        self.min_bucket_size = min_bucket_size

        # Default bucket boundaries if not provided
        if bucket_boundaries is None:
            self.bucket_boundaries = [64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.bucket_boundaries = sorted(bucket_boundaries)

        # Storage for bucketed samples
        self.buckets = defaultdict(list)
        self.bucket_stats = defaultdict(int)

    def get_bucket_id(self, sequence_length: int) -> int:
        """Get bucket ID for a given sequence length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if sequence_length <= boundary:
                return i
        # If longer than all boundaries, use the last bucket
        return len(self.bucket_boundaries) - 1

    def add_sample(self, sample: Dict[str, torch.Tensor]) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Add a sample to appropriate bucket.

        Args:
            sample: Tokenized sample with input_ids, attention_mask, labels

        Returns:
            List of samples if bucket is full and ready to yield, None otherwise
        """
        if not self.enable_bucketing:
            return [sample]  # Return immediately if bucketing is disabled

        # Get sequence length from sample
        if 'input_ids' in sample:
            seq_length = sample['input_ids'].size(0) if sample['input_ids'].dim() == 1 else sample['input_ids'].size(1)
        else:
            # Fallback: return immediately if no input_ids
            return [sample]

        bucket_id = self.get_bucket_id(seq_length)
        self.buckets[bucket_id].append(sample)
        self.bucket_stats[bucket_id] += 1

        # Check if bucket is full
        if len(self.buckets[bucket_id]) >= self.max_bucket_size:
            # Return full bucket and clear it
            full_bucket = self.buckets[bucket_id].copy()
            self.buckets[bucket_id].clear()
            return full_bucket

        return None  # Bucket not full yet

    def flush_buckets(self, min_size: Optional[int] = None) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """
        Flush all buckets that have at least min_size samples.

        Args:
            min_size: Minimum bucket size to flush (defaults to min_bucket_size)

        Yields:
            Lists of samples from each bucket
        """
        if min_size is None:
            min_size = self.min_bucket_size

        for bucket_id, samples in self.buckets.items():
            if len(samples) >= min_size:
                yield samples.copy()
                samples.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get bucketing statistics."""
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


class StreamingDataset(IterableDataset):
    """Streaming dataset that loads data on-the-fly with dynamic sequence length support"""

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
        max_bucket_size: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.dynamic_length_fn = dynamic_length_fn  # Function to get current sequence length

        # Initialize length-based bucketing
        self.bucketing = LengthBasedBucketing(
            bucket_boundaries=bucket_boundaries,
            max_bucket_size=max_bucket_size,
            enable_bucketing=enable_bucketing
        )

        # Find all data files
        self.data_files = self._find_data_files()

        # Auto-create validation from training files if val split is empty
        if not self.data_files and self.split == "val":
            print(f" No validation files found, creating validation set from training files...")
            # Temporarily switch to train split to find files
            original_split = self.split
            self.split = "train"
            train_files = self._find_data_files()
            self.split = original_split

            if train_files:
                # Use all training files for validation (they'll be sampled differently via random seed)
                self.data_files = train_files
                print(f" ✓ Created validation set from {len(self.data_files)} training files")
            else:
                print(f" WARNING: No training files found either, will use synthetic data")

        if not self.data_files:
            print(f" WARNING: No data files found for {split} split, will use synthetic data")
            print(f" Data directory checked: {self.data_dir}")
            if self.data_dir.exists():
                all_files = list(self.data_dir.glob('*.jsonl')) + list(self.data_dir.glob('*.arrow')) + list(self.data_dir.glob('*.parquet'))
                print(f" Available files in directory: {len(all_files)}")
                if all_files:
                    print(f" Sample files: {[f.name for f in all_files[:3]]}")
        else:
            print(f" Found {len(self.data_files)} data files for {split} split")

    def _find_data_files(self) -> List[Path]:
        """Find all relevant data files with improved pattern matching"""
        files = []

        # More flexible patterns that match actual data structure
        patterns = [
            # Standard split-based patterns
            f"*/{self.split}/**/*.arrow",      # dataset_name/train/...
            f"**/{self.split}/**/*.arrow",     # nested structures
            f"{self.split}_*/**/*.arrow",      # original pattern
            f"*/{self.split}/**/*.parquet",
            f"**/{self.split}/**/*.parquet",
            f"*/{self.split}/**/*.jsonl",
            f"**/{self.split}/**/*.jsonl",
            f"{self.split}_*.arrow",           # direct files
            f"{self.split}_*.parquet",
            f"{self.split}_*.jsonl",
            f"{self.split}.jsonl",             # exact match
        ]

        # For both 'train' and 'val' splits, look for all processed files
        # Each split will use all available data (differentiated by random seed/sampling)
        if self.split in ["train", "val"]:
            patterns.extend([
                "*_processed.jsonl",              # processed files (MAIN PATTERN)
                "processed*.jsonl",               # processed prefix
                "*_processed.arrow",              # processed arrow files
                "*_processed.parquet",            # processed parquet files
                "*.jsonl",                        # fallback: any jsonl file
                "*.arrow",                        # any arrow files
                "*.parquet",                      # any parquet files
            ])

        for pattern in patterns:
            matched_files = sorted(self.data_dir.glob(pattern))
            files.extend(matched_files)

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        files = unique_files

        # Filter out empty files (0 bytes) and very small files (<10KB - likely just metadata)
        MIN_FILE_SIZE = 10 * 1024  # 10KB minimum
        substantial_files = []
        filtered_count = 0
        for f in files:
            if f.stat().st_size >= MIN_FILE_SIZE:
                substantial_files.append(f)
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"   Filtered out {filtered_count} empty/tiny files (<10KB)")

        files = substantial_files

        # If still no files, provide diagnostic info
        if not files:
            if not self.data_dir.exists():
                print(f"   ⚠️  Data directory does not exist: {self.data_dir}")
            else:
                print(f"   Searching in: {self.data_dir}")
                all_files = list(self.data_dir.glob("**/*.arrow")) + list(self.data_dir.glob("**/*.jsonl"))
                if all_files:
                    print(f"   Found {len(all_files)} data files, but none match split '{self.split}'")
                    print(f"   Sample files: {[f.relative_to(self.data_dir) for f in all_files[:3]]}")

        print(f"   ✓ Found {len(files)} data files for {self.split} split")
        if len(files) > 0:
            print(f"   📊 Sample files: {[f.name for f in files[:3]]}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more files")
        return files

    def _read_file(self, file_path: Path) -> Iterator[str]:
        """Read data from a single file"""
        try:
            if file_path.suffix == '.arrow':
                # Use robust Arrow file reader instead of skipping
                print(f"📖 Reading Arrow file: {file_path.name}")
                try:
                    for text in arrow_reader.read_arrow_file(file_path):
                        if text and len(text.strip()) > 10:
                            yield text.strip()
                except Exception as e:
                    print(f"⚠️  Failed to read Arrow file {file_path.name}: {e}")
                    # Don't return, continue to fallback methods

            elif file_path.suffix == '.parquet':
                # Read Parquet file in chunks
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=100):
                    df = batch.to_pandas()

                    if 'text' in df.columns:
                        for text in df['text']:
                            if text and len(str(text).strip()) > 10:
                                yield str(text)

            elif file_path.suffix == '.jsonl':
                # Read JSONL file with robust Unicode handling
                # Reduced logging - only log if verbose mode enabled
                line_count = 0
                error_count = 0

                try:
                    for line in encoding_detector.read_file_robust(file_path):
                        line_count += 1
                        try:
                            # Try to parse as JSON
                            data = json.loads(line)

                            # Extract text from various possible fields
                            text = None
                            text_fields = ['text', 'content', 'document', 'passage', 'input', 'question', 'instruction']

                            for field in text_fields:
                                if field in data and data[field]:
                                    text = str(data[field]).strip()
                                    break

                            # If no standard text field, try to extract from complex structures
                            if not text:
                                if isinstance(data, dict):
                                    # Look for any string value
                                    for key, value in data.items():
                                        if isinstance(value, str) and len(value.strip()) > 10:
                                            text = value.strip()
                                            break

                            if text and len(text) > 10:
                                yield text

                        except json.JSONDecodeError as e:
                            error_count += 1
                            if error_count % 1000 == 1:  # Log occasionally
                                print(f"  ⚠️  JSON decode error at line {line_count}: {str(e)[:100]}")
                            continue
                        except Exception as e:
                            error_count += 1
                            if error_count % 1000 == 1:
                                print(f"  ⚠️  Text extraction error at line {line_count}: {str(e)[:100]}")
                            continue

                    # Summary for this file - only log non-empty files
                    if line_count > 0:
                        success_rate = (line_count - error_count) / line_count * 100
                        if line_count > error_count:  # Only log if there were valid lines
                            pass  # Silent - too verbose
                    # Skip logging empty files to reduce noise

                except Exception as e:
                    print(f"  ❌ Failed to read {file_path.name}: {e}")
                    return

        except Exception as e:
            print(f" Error reading {file_path}: {e}")

    def _generate_synthetic_data(self) -> Iterator[str]:
        """Generate synthetic data for testing"""
        import warnings
        warnings.warn(
            "No data files found! Falling back to synthetic data generation. "
            "This is for testing only and will not produce meaningful training results. "
            f"Check that data files exist in: {self.data_dir}",
            UserWarning,
            stacklevel=2
        )
        print(f" WARNING: Using synthetic data - no real data files found in {self.data_dir}")

        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Deep learning models require large amounts of data.",
            "Natural language processing enables human-computer interaction.",
            "Transformer models have revolutionized NLP tasks.",
            "Artificial neural networks mimic biological neurons.",
            "Gradient descent optimizes model parameters.",
            "Backpropagation computes gradients efficiently.",
            "Attention mechanisms improve sequence modeling.",
            "Pre-training creates powerful language representations.",
        ]

        # Generate finite number of synthetic examples (1000) to avoid infinite blocking
        for _ in range(1000):
            # Randomly combine templates
            num_sentences = random.randint(3, 8)
            text = " ".join(random.choices(templates, k=num_sentences))
            yield text * random.randint(2, 5)

    def _stream_examples(self, files_to_use=None) -> Iterator[str]:
        """Stream examples from all files with interleaving"""
        # Use provided files or fall back to self.data_files
        files = files_to_use if files_to_use is not None else self.data_files

        if not files:
            # Use synthetic data if no files found
            yield from self._generate_synthetic_data()
        else:
            # Shuffle files initially for variety
            shuffled_files = list(files)
            random.shuffle(shuffled_files)

            # Open all files and create generators
            file_generators = []
            for file_path in shuffled_files:
                try:
                    gen = self._read_file(file_path)
                    file_generators.append((file_path, gen))
                    print(f"  ✓ Added {file_path.name} to streaming pool")
                except Exception as e:
                    print(f"  ⚠️ Could not open {file_path.name}: {e}")

            if not file_generators:
                print("  ⚠️ No files could be opened, using synthetic data")
                yield from self._generate_synthetic_data()
                return

            print(f"  📚 Interleaving data from {len(file_generators)} files")

            # Interleave samples from all files
            # With multi-worker loading, each worker gets fewer files, so read more per file
            samples_per_file = 100  # Read 100 samples from each file before switching (was 10)
            exhausted_files = set()

            while len(exhausted_files) < len(file_generators):
                # Go through each file in round-robin fashion
                for idx, (file_path, gen) in enumerate(file_generators):
                    if idx in exhausted_files:
                        continue

                    # Read a batch of samples from this file
                    samples_read = 0
                    for _ in range(samples_per_file):
                        try:
                            text = next(gen)
                            yield text
                            samples_read += 1
                        except StopIteration:
                            exhausted_files.add(idx)
                            # Removed verbose logging
                            break

                # If all files exhausted, restart with fresh generators
                if len(exhausted_files) == len(file_generators):
                    # Reduced logging - only log first restart
                    if not hasattr(self, '_restart_count'):
                        self._restart_count = 0
                        print(f"  🔄 Restarting data streaming (all files processed)")
                    self._restart_count += 1

                    # Prevent infinite restarts for empty files (max 3 restarts)
                    if self._restart_count > 3:
                        print(f"  ⚠️ Files appear to be empty after {self._restart_count} restarts, using synthetic data")
                        yield from self._generate_synthetic_data()
                        return

                    exhausted_files.clear()

                    # Reshuffle and recreate generators
                    random.shuffle(shuffled_files)
                    file_generators = []
                    for file_path in shuffled_files:
                        try:
                            gen = self._read_file(file_path)
                            file_generators.append((file_path, gen))
                        except Exception as e:
                            print(f"  ⚠️ Could not reopen {file_path.name}: {e}")

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text with dynamic sequence length support

        The text should already contain Human/Assistant conversation format.
        Example: "\\n\\nHuman: Hello\\n\\nAssistant: Hi there!"

        This function preserves the full conversation structure in the tokenized output.
        """
        # Get current sequence length (progressive training fix)
        if self.dynamic_length_fn is not None:
            try:
                current_max_length = self.dynamic_length_fn()
            except Exception:
                # Fallback to default if dynamic function fails
                current_max_length = self.max_length
        else:
            current_max_length = self.max_length

        # Ensure current_max_length is within reasonable bounds
        current_max_length = max(32, min(current_max_length, self.max_length))

        # Tokenize the full conversation (Human/Assistant format is preserved)
        encoded = self.tokenizer(
            text,
            max_length=current_max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()  # For causal LM, labels = input_ids
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset with length-based bucketing"""
        # CRITICAL FIX: Handle multi-worker data loading correctly
        worker_info = torch.utils.data.get_worker_info()
        original_files = None  # Track original files for restoration
        if worker_info is not None:
            # Split data files across workers to avoid duplication/deadlock
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Save original files and use worker subset
            original_files = self.data_files
            worker_files = [f for i, f in enumerate(self.data_files) if i % num_workers == worker_id]
        else:
            worker_files = None  # Use all files

        count = 0
        buffer = []
        samples_processed = 0  # Track total samples processed from files

        for text in self._stream_examples(files_to_use=worker_files):
            if self.max_samples and count >= self.max_samples:
                break

            buffer.append(text)
            samples_processed += 1

            # When buffer is full, process with bucketing
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)

                # Tokenize and add to buckets
                for buffered_text in buffer:
                    if self.max_samples and count >= self.max_samples:
                        break

                    tokenized = self._tokenize_text(buffered_text)

                    # If bucketing is disabled, yield immediately
                    if not self.bucketing.enable_bucketing:
                        yield tokenized
                        count += 1
                    else:
                        # Add to bucket and check if bucket is ready
                        bucket_samples = self.bucketing.add_sample(tokenized)

                        if bucket_samples is not None:
                            # Bucket is full, yield all samples in bucket
                            for sample in bucket_samples:
                                yield sample
                                count += 1
                                if self.max_samples and count >= self.max_samples:
                                    return

                buffer = []

                # Periodically flush partial buckets to ensure continuous data flow
                if self.bucketing.enable_bucketing and samples_processed % (self.buffer_size * 5) == 0:
                    for bucket_samples in self.bucketing.flush_buckets(min_size=8):  # Flush if at least 8 samples
                        for sample in bucket_samples:
                            if self.max_samples and count >= self.max_samples:
                                return
                            yield sample
                            count += 1

        # Process remaining buffer
        if buffer:
            random.shuffle(buffer)
            for buffered_text in buffer:
                if self.max_samples and count >= self.max_samples:
                    break

                tokenized = self._tokenize_text(buffered_text)

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

        # Flush ALL remaining buckets at the end (use min_size=1 to not lose ANY samples)
        if self.bucketing.enable_bucketing:
            for bucket_samples in self.bucketing.flush_buckets(min_size=1):
                for sample in bucket_samples:
                    if self.max_samples and count >= self.max_samples:
                        break
                    yield sample
                    count += 1

        # Log statistics if we processed data
        if samples_processed > 0:
            if self.bucketing.enable_bucketing:
                stats = self.bucketing.get_statistics()
                if count < samples_processed * 0.5:  # If we yielded less than 50% of what we read
                    print(f"⚠️  Warning: Read {samples_processed} samples but only yielded {count}")
                    print(f"    Bucketing stats: {stats['samples_in_buckets']} samples still in buckets")

        # Restore original data_files if we're in a worker
        if original_files is not None:
            self.data_files = original_files


def create_streaming_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    buffer_size: int = 1000,
    distributed: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    dynamic_length_fn: Optional[Callable[[], int]] = None,
    enable_bucketing: bool = True,
    bucket_boundaries: Optional[List[int]] = None,
    max_bucket_size: int = 100
) -> Tuple[DataLoader, DataLoader]:
    """Create streaming train and validation dataloaders with distributed support"""

    # Safety check for batch_size
    if batch_size is None:
        batch_size = 8
        print(f" batch_size was None, defaulting to {batch_size}")

    # Auto-detect CPU core count if num_workers is set to use all cores
    if num_workers == -1 or num_workers == 0:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        print(f" 🚀 Auto-detected {num_workers} CPU cores, using all for data loading")
    elif num_workers > 0:
        print(f" 🚀 Using {num_workers} CPU workers for parallel data loading")

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

        print(f" Creating distributed streaming dataloaders (rank {rank}/{world_size})")
    else:
        print(f"Creating streaming dataloaders...")

    # Create streaming datasets
    train_dataset = StreamingDataset(
        data_dir=data_dir,
        split='train',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        buffer_size=buffer_size,
        dynamic_length_fn=dynamic_length_fn,
        enable_bucketing=enable_bucketing,
        bucket_boundaries=bucket_boundaries,
        max_bucket_size=max_bucket_size
    )

    val_dataset = StreamingDataset(
        data_dir=data_dir,
        split='val',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples // 10 if max_samples else 1000,  # Limit validation
        buffer_size=buffer_size // 10,
        dynamic_length_fn=dynamic_length_fn,
        enable_bucketing=enable_bucketing,
        bucket_boundaries=bucket_boundaries,
        max_bucket_size=max_bucket_size
    )

    # Create dataloaders with distributed support
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True,  # Important for distributed training
        'prefetch_factor': 4 if num_workers > 0 else None,  # Prefetch 4 batches per worker
        'persistent_workers': True if num_workers > 0 else False  # Keep workers alive between epochs
    }

    # For IterableDataset, we don't use DistributedSampler,
    # but we need to handle distributed iteration in the dataset itself
    if distributed and DISTRIBUTED_AVAILABLE:
        # Modify datasets for distributed iteration
        train_dataset = DistributedStreamingDataset(train_dataset, world_size or 1, rank or 0)
        val_dataset = DistributedStreamingDataset(val_dataset, world_size or 1, rank or 0)

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    return train_loader, val_loader


class DistributedStreamingDataset(IterableDataset):
    """Wrapper for distributed streaming dataset"""

    def __init__(self, base_dataset: StreamingDataset, world_size: int, rank: int):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        # Create iterator from base dataset
        base_iter = iter(self.base_dataset)

        # Skip samples to ensure each rank gets different data
        # This is a simple round-robin distribution
        for i, sample in enumerate(base_iter):
            if i % self.world_size == self.rank:
                yield sample


class InfiniteStreamingDataset(IterableDataset):
    """Infinite streaming dataset for continuous training"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int,
        buffer_size: int = 1000
    ):
        self.base_dataset = StreamingDataset(
            data_dir=data_dir,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=None,
            buffer_size=buffer_size
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate infinitely over the dataset"""
        while True:
            # Create a new iterator for each pass
            for item in self.base_dataset:
                yield item