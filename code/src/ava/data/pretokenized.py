"""
Ultra-Fast Memory-Mapped Pre-Tokenized Dataset Loader

Optimized zero-copy data loading that eliminates tokenization,
parsing, and data copy overhead during training.

Key Optimizations:
- Memory-mapped Arrow reading with persistent handles (zero I/O overhead)
- Batch vectorized tensor creation
- Zero-copy numpy→torch conversion via torch.from_numpy()
- Cached Arrow table handles (eliminates file open/close)
- Direct buffer protocol access (bypasses Python objects)
- Minimal validation (pre-validated during tokenization)

Usage:
    from ava.data.pretokenized import create_ultra_fast_dataloaders

    train_loader, val_loader = create_ultra_fast_dataloaders(
        data_dir='/path/to/pretokenized',
        batch_size=32,
        max_length=2048,
        num_workers=4
    )
"""

import atexit
import json
import logging
from pathlib import Path
import signal
import time
from typing import Any, Dict, List, Optional, Iterator, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Dataset, DataLoader
import random
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import weakref

# Import from centralized Arrow I/O module
from .arrow_io import (
    read_arrow_table,
    ArrowTableCache,
    ThreadLocalArrowCache,
    get_thread_local_arrow_cache,
)

# Import profiler for performance tracking
try:
    from .profiling import get_global_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    get_global_profiler = None

logger = logging.getLogger(__name__)

# Default vocab size for validation (can be overridden per-dataset)
DEFAULT_VOCAB_SIZE = 50680  # Match standard tokenizer vocab


class DataLoaderError(Exception):
    """
    Enhanced exception for data loading errors with recovery suggestions.

    Provides detailed context and actionable recovery steps for common issues.
    """

    def __init__(self, message: str, recovery_steps: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.recovery_steps = recovery_steps or []
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [self.message]

        if self.context:
            lines.append("\nContext:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")

        if self.recovery_steps:
            lines.append("\nRecovery steps:")
            for i, step in enumerate(self.recovery_steps, 1):
                lines.append(f"  {i}. {step}")

        return "\n".join(lines)


def _get_oom_recovery_steps() -> List[str]:
    """Get recovery steps for OOM errors."""
    return [
        "Reduce batch_size in config (e.g., batch_size: 16)",
        "Enable gradient_checkpointing: true",
        "Reduce max_length (e.g., max_length: 1024)",
        "Enable use_dynamic_padding: true for memory efficiency",
        "Increase num_workers to offload preprocessing to CPU",
        "Use lazy_file_discovery: true for large datasets",
    ]


def _get_no_data_recovery_steps(data_dir: str) -> List[str]:
    """Get recovery steps for missing data errors."""
    return [
        f"Check data directory exists: ls -la {data_dir}",
        "Verify files have .arrow or .parquet extension",
        "Check file permissions: chmod -R 644 {data_dir}/*",
        "Run data preprocessing script to generate pretokenized data",
        "Check config data.data_dir points to correct location",
    ]


def _get_corruption_recovery_steps() -> List[str]:
    """Get recovery steps for data corruption errors."""
    return [
        "Re-run tokenization preprocessing on source data",
        "Check disk for errors: fsck or disk utility",
        "Validate source data is not corrupted",
        "Try with skip_corrupted_sequences: true in config",
        "Reduce cache_size if memory is limited",
    ]


# Global registry to track all ArrowTableCache instances for cleanup
_cache_registry: weakref.WeakSet = weakref.WeakSet()


def _cleanup_all_caches():
    """Clean up all registered ArrowTableCache instances on shutdown."""
    close_errors = 0
    for cache in list(_cache_registry):
        try:
            cache.close()
        except Exception as e:
            close_errors += 1
            logging.getLogger(__name__).debug(f"Cache cleanup warning: {e}")
    if close_errors > 0:
        logging.getLogger(__name__).debug(f"Cache cleanup completed with {close_errors} errors")


# Register cleanup on normal exit
atexit.register(_cleanup_all_caches)


def _worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader workers.

    Sets up proper signal handling and cleanup for worker processes
    to prevent semaphore leaks on shutdown.
    """
    import signal

    def _worker_cleanup(signum, frame):
        """Clean up resources when worker receives termination signal."""
        _cleanup_all_caches()
        # Re-raise to allow normal termination
        raise SystemExit(0)

    # Handle SIGTERM gracefully in workers
    try:
        signal.signal(signal.SIGTERM, _worker_cleanup)
    except (ValueError, OSError):
        # Signal handling may not work in all contexts
        pass

    # Set worker-specific random seed for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

from ..core.data_utils import find_data_files, collate_batch

# Import pinned buffer pool for async GPU transfers.
# Only used when num_workers=0 (main process), as pinned memory
# does not survive IPC from worker processes to main process.
try:
    from ..cuda.streams import PinnedBufferPool, get_buffer_pool, is_main_process_dataloader
    PINNED_BUFFERS_AVAILABLE = True
except ImportError:
    PINNED_BUFFERS_AVAILABLE = False
    PinnedBufferPool = None
    get_buffer_pool = None
    is_main_process_dataloader = None

# Import sequence packing for improved throughput
try:
    from .packing import SequencePackingCollator, DynamicSequencePackingCollator
    SEQUENCE_PACKING_AVAILABLE = True
except ImportError as e:
    SEQUENCE_PACKING_AVAILABLE = False
    SequencePackingCollator = None
    DynamicSequencePackingCollator = None
    logger.warning("=" * 60)
    logger.warning("SEQUENCE PACKING UNAVAILABLE")
    logger.warning("=" * 60)
    logger.warning(f"Failed to import sequence_packing: {e}")
    logger.warning("If use_sequence_packing=true in config, it will be ignored!")
    logger.warning("To enable: ensure sequence_packing.py exists in Ava/data/")
    logger.warning("=" * 60)


class PreTokenizedSequenceReader:
    """
    Memory-mapped reader for a single pre-tokenized Arrow file.

    Provides zero-copy random access to tokenized sequences with chunk prefetching
    for 10-20x faster per-row access.
    """

    def __init__(self, data_path: Path, prefetch_chunk_size: int = 256):
        self.data_path = data_path
        self.prefetch_chunk_size = prefetch_chunk_size

        # Open Arrow file (supports both IPC File and Stream formats)
        self.table = read_arrow_table(data_path)

        self.num_sequences = len(self.table)

        # Chunk prefetch cache for faster per-row access
        self._chunk_start: int = -1
        self._chunk_end: int = -1
        self._chunk_input_ids: Optional[List[np.ndarray]] = None
        self._chunk_attention_mask: Optional[List[np.ndarray]] = None

        # Determine column names once
        col_names = self.table.schema.names
        self._token_col = 'input_ids' if 'input_ids' in col_names else 'token_ids'
        self._has_attention_mask = 'attention_mask' in col_names

    def __len__(self):
        return self.num_sequences

    def _load_chunk(self, idx: int) -> None:
        """
        Load a chunk of rows containing the given index.

        Loads prefetch_chunk_size rows centered around idx for efficient
        sequential and near-sequential access patterns.
        """
        # Calculate chunk boundaries (align to chunk size for cache efficiency)
        chunk_start = (idx // self.prefetch_chunk_size) * self.prefetch_chunk_size
        chunk_end = min(chunk_start + self.prefetch_chunk_size, self.num_sequences)
        chunk_size = chunk_end - chunk_start

        # Extract chunk from table
        chunk = self.table.slice(chunk_start, chunk_size)

        # Convert to numpy arrays for fast access
        input_ids_col = chunk.column(self._token_col)
        self._chunk_input_ids = []
        for i in range(chunk_size):
            arr = input_ids_col[i]
            try:
                # Try Arrow's efficient numpy conversion first
                self._chunk_input_ids.append(
                    np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.int32)
                )
            except (AttributeError, TypeError):
                # Fallback to as_py() for complex types
                self._chunk_input_ids.append(np.array(arr.as_py(), dtype=np.int32))

        if self._has_attention_mask:
            attention_mask_col = chunk.column('attention_mask')
            self._chunk_attention_mask = []
            for i in range(chunk_size):
                arr = attention_mask_col[i]
                try:
                    self._chunk_attention_mask.append(
                        np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.int32)
                    )
                except (AttributeError, TypeError):
                    self._chunk_attention_mask.append(np.array(arr.as_py(), dtype=np.int32))
        else:
            self._chunk_attention_mask = None

        self._chunk_start = chunk_start
        self._chunk_end = chunk_end

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a tokenized sequence by index with chunk prefetching (10-20x faster)."""
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.num_sequences})")

        # Check if idx is in current chunk
        if not (self._chunk_start <= idx < self._chunk_end):
            self._load_chunk(idx)

        # Get from chunk cache (O(1) access)
        local_idx = idx - self._chunk_start
        input_ids = self._chunk_input_ids[local_idx]

        if self._chunk_attention_mask is not None:
            attention_mask = self._chunk_attention_mask[local_idx]
        else:
            attention_mask = np.ones(len(input_ids), dtype=np.int32)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def close(self):
        """Clear chunk cache."""
        self._chunk_input_ids = None
        self._chunk_attention_mask = None
        self._chunk_start = -1
        self._chunk_end = -1

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class PreTokenizedDataset(IterableDataset):
    """
    Streaming dataset for pre-tokenized binary data.

    Features:
    - Zero-copy memory-mapped loading
    - Shuffling with configurable buffer
    - Distributed training support
    - No tokenization overhead
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_length: int,
        max_samples: Optional[int] = None,
        buffer_size: int = 10000,
        enable_bucketing: bool = False,  # Bucketing less useful with pre-tokenized data
        pad_token_id: int = 0,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        skip_sequence_count: bool = True,  # Fast startup: estimate instead of count
        shuffle_seed: Optional[int] = None,  # Shuffle seed for reproducibility
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.enable_bucketing = enable_bucketing
        self.pad_token_id = pad_token_id
        self.skip_sequence_count = skip_sequence_count
        self.shuffle_seed = shuffle_seed

        # Distributed training
        self.world_size = world_size or 1
        self.rank = rank or 0

        # Find data files
        self.data_files = self._find_data_files()

        if not self.data_files:
            raise ValueError(f"No pre-tokenized files found for {split} split in {data_dir}")

        logger.debug(f"Found {len(self.data_files)} pre-tokenized files for {split} split")

        # Calculate total sequences from Arrow files
        self.file_sequences = []

        if self.skip_sequence_count:
            # Fast estimation from file sizes (no table loading)
            self.total_sequences = self._estimate_sequence_count()
            logger.debug(f"[FAST] Estimated sequences: {self.total_sequences:,}")
        else:
            # Exact counting (slow - loads all tables)
            self.total_sequences = 0
            for file_path in self.data_files:
                table = read_arrow_table(file_path)
                num_sequences = len(table)
                self.file_sequences.append(num_sequences)
                self.total_sequences += num_sequences
            logger.debug(f"Total sequences: {self.total_sequences:,}")

    def _estimate_sequence_count(self) -> int:
        """Fast estimation of sequence count from file sizes."""
        if not self.data_files:
            return 0

        # Sample first few files for estimation
        sample_files = self.data_files[:min(5, len(self.data_files))]
        total_sample_bytes = 0
        total_sample_sequences = 0

        for file_path in sample_files:
            try:
                total_sample_bytes += file_path.stat().st_size
                table = read_arrow_table(file_path)
                total_sample_sequences += len(table)
            except Exception:
                pass

        if total_sample_bytes == 0 or total_sample_sequences == 0:
            # Fallback: rough estimate of 500 sequences per file
            return len(self.data_files) * 500

        # Calculate bytes per sequence and extrapolate
        bytes_per_seq = total_sample_bytes / total_sample_sequences
        total_bytes = sum(f.stat().st_size for f in self.data_files)
        estimated = int(total_bytes / bytes_per_seq)

        return estimated

    def _find_data_files(self) -> List[Path]:
        """Find pre-tokenized Arrow files (using shared utility)."""
        return find_data_files(
            data_dir=self.data_dir,
            split=self.split,
            patterns=[
                f"**/{self.split}/**/*.arrow",
                f"{self.split}_*.arrow",
                f"{self.split}/*.arrow",
            ],
            min_file_size=0,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset with shuffling."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # Shuffle files with configurable seed
        shuffled_files = list(self.data_files)
        base_seed = self.shuffle_seed if self.shuffle_seed is not None else int(time.time() * 1000000) % (2**31)
        random.Random(base_seed + worker_id).shuffle(shuffled_files)

        # Open readers for all files
        readers = []
        failed_files = []
        for file_path in shuffled_files:
            try:
                reader = PreTokenizedSequenceReader(file_path)
                readers.append((file_path, reader))
            except Exception as e:
                failed_files.append((file_path, str(e)))
                logger.warning(f"Failed to open {file_path}: {e}")

        if failed_files:
            logger.warning(
                f"Failed to open {len(failed_files)}/{len(shuffled_files)} files"
            )

        if not readers:
            raise ValueError(f"No files could be opened from {self.data_dir}")

        # OPTIMIZATION: Compute worker indices on-demand without building full list
        # This reduces memory from O(total_sequences) to O(num_files)
        # Build cumulative sequence counts for fast index lookup
        cumsum = [0]
        total_sequences = 0
        for _, reader in readers:
            total_sequences += len(reader)
            cumsum.append(total_sequences)

        # Compute how many sequences this worker processes
        stride = self.world_size * num_workers
        worker_offset = self.rank * num_workers + worker_id
        base_seed = self.shuffle_seed if self.shuffle_seed is not None else int(time.time() * 1000000) % (2**31)

        # Generate a permutation using numpy for efficiency (single allocation)
        rng = np.random.default_rng(base_seed + worker_id)
        # Only shuffle once, compute worker indices via modular arithmetic
        perm = rng.permutation(total_sequences)

        # Extract only this worker's indices (O(n/stride) instead of O(n))
        worker_perm = perm[worker_offset::stride]

        # Convert flat indices to (file_idx, seq_idx) pairs on-demand
        def flat_to_pair(flat_idx):
            """Convert flat index to (file_idx, seq_idx) using binary search."""
            # Binary search for file index
            file_idx = np.searchsorted(cumsum[1:], flat_idx, side='right')
            seq_idx = flat_idx - cumsum[file_idx]
            return file_idx, seq_idx

        # Build worker indices efficiently
        worker_indices = [flat_to_pair(int(idx)) for idx in worker_perm]

        # Stream sequences
        count = 0
        for file_idx, seq_idx in worker_indices:
            if self.max_samples and count >= self.max_samples:
                break

            file_path, reader = readers[file_idx]

            try:
                # Read sequence (zero-copy)
                data = reader[seq_idx]

                # Convert to tensors
                yield {
                    'input_ids': torch.from_numpy(data['input_ids']).long(),
                    'attention_mask': torch.from_numpy(data['attention_mask']).long(),
                    'labels': torch.from_numpy(data['input_ids']).long()
                }

                count += 1

            except Exception as e:
                logger.warning(
                    f"Error reading sequence {seq_idx} from {file_path.name}: {e}"
                )
                continue

        # Close all readers
        for file_path, reader in readers:
            reader.close()

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function with FIXED padding to self.max_length.

        Uses shared collate_batch utility to eliminate code duplication.

        Uses fixed-length padding to prevent torch.compile recompilation.
        Dynamic padding causes shape changes that trigger recompilation.
        """
        return collate_batch(
            batch,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            use_fixed_padding=True,  # Fixed padding for torch.compile compatibility
        )


class PreTokenizedMapDataset(Dataset):
    """
    Map-style dataset for pre-tokenized data.

    Use this for validation/testing where you need deterministic ordering
    and random access.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_length: int,
        pad_token_id: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Find and load all files
        self.data_files = self._find_data_files()

        if not self.data_files:
            raise ValueError(f"No pre-tokenized files found for {split} split in {data_dir}")

        # Open all readers
        self.readers = []
        self.file_offsets = [0]
        total_sequences = 0

        failed_count = 0
        for file_path in self.data_files:
            try:
                reader = PreTokenizedSequenceReader(file_path)
                self.readers.append(reader)
                total_sequences += len(reader)
                self.file_offsets.append(total_sequences)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to open {file_path}: {e}")

        if failed_count > 0:
            logger.warning(
                f"Failed to open {failed_count}/{len(self.data_files)} files"
            )

        self.total_sequences = total_sequences
        logger.info(f"Loaded {len(self.readers)} files with {self.total_sequences:,} sequences")

    def _find_data_files(self) -> List[Path]:
        """Find pre-tokenized Arrow files (using shared utility)."""
        return find_data_files(
            data_dir=self.data_dir,
            split=self.split,
            patterns=[
                f"**/{self.split}/**/*.arrow",
                f"{self.split}_*.arrow",
                f"{self.split}/*.arrow",
            ],
            min_file_size=0,
        )

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sequence by global index."""
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.total_sequences})")

        # Find which file contains this index
        file_idx = 0
        for i, offset in enumerate(self.file_offsets[1:], start=1):
            if idx < offset:
                file_idx = i - 1
                break

        # Get local index within file
        local_idx = idx - self.file_offsets[file_idx]

        # Read from appropriate reader
        reader = self.readers[file_idx]
        data = reader[local_idx]

        return {
            'input_ids': torch.from_numpy(data['input_ids']).long(),
            'attention_mask': torch.from_numpy(data['attention_mask']).long(),
            'labels': torch.from_numpy(data['input_ids']).long()
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function with dynamic padding.

        Uses shared collate_batch utility to eliminate code duplication.
        Uses dynamic padding (pad to batch max) for memory efficiency in validation.
        """
        return collate_batch(
            batch,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            use_fixed_padding=False,  # Dynamic padding for memory efficiency in validation
        )


# ============================================================================
# OPTIMIZED IMPLEMENTATION
# ============================================================================

# ArrowTableCache is now imported from arrow_io module


class LazyFileDiscovery:
    """
    Lazy file discovery that finds files on-demand during iteration.

    Instead of discovering all files upfront (which can cause OOM with thousands
    of files), this class discovers files incrementally as they're needed.

    Benefits:
    - No upfront memory allocation for file list
    - Faster startup time (no glob of entire directory)
    - Memory-efficient for very large datasets
    - Supports infinite streaming without loading all file paths

    Optimization:
    - Set cache_at_init=True to eagerly cache all files at init (5-15% faster iteration)
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        patterns: Optional[List[str]] = None,
        min_file_size: int = 10 * 1024,  # 10KB minimum
        max_files: Optional[int] = None,
        shuffle_seed: Optional[int] = None,
        cache_at_init: bool = False,  # Eagerly cache file list at init for faster iteration
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.min_file_size = min_file_size
        self.max_files = max_files
        self.shuffle_seed = shuffle_seed
        self.cache_at_init = cache_at_init

        # Default patterns for Arrow/Parquet files
        self.patterns = patterns or [
            f"**/{split}/**/*.parquet",
            f"**/{split}/**/*.arrow",
            f"{split}_*.parquet",
            f"{split}_*.arrow",
            "*.parquet",
            "*.arrow",
        ]

        # Lazy state - files discovered on demand
        self._discovered_files: List[Path] = []
        self._discovery_complete = False
        self._pattern_iterators: List[Iterator[Path]] = []
        self._cached_file_list: Optional[List[Path]] = None

        # Eagerly cache files if requested (eliminates per-epoch glob overhead)
        if cache_at_init:
            self._build_file_cache()
        self._files_yielded = 0

    def _build_file_cache(self) -> None:
        """
        Build complete file list once and cache it.

        This eliminates per-epoch glob overhead, providing 5-15% faster iteration
        after the initial startup cost.
        """
        if self._cached_file_list is not None:
            return  # Already cached

        all_files: List[Path] = []
        seen_paths: set = set()

        for pattern in self.patterns:
            for file_path in self.data_dir.glob(pattern):
                if file_path in seen_paths:
                    continue
                if not file_path.exists():
                    continue
                try:
                    if file_path.stat().st_size < self.min_file_size:
                        continue
                except OSError:
                    continue

                all_files.append(file_path)
                seen_paths.add(file_path)

                if self.max_files and len(all_files) >= self.max_files:
                    break

            if self.max_files and len(all_files) >= self.max_files:
                break

        # Sort for reproducibility, then cache
        self._cached_file_list = sorted(all_files)
        self._discovery_complete = True
        self._discovered_files = list(self._cached_file_list)

    def refresh_file_list(self) -> None:
        """
        Explicitly refresh the file cache.

        Call this after new data files have been added to pick them up
        without restarting the training.
        """
        self._cached_file_list = None
        self._discovered_files = []
        self._discovery_complete = False
        self._pattern_iterators = []
        if self.cache_at_init:
            self._build_file_cache()

    def _init_pattern_iterators(self):
        """Initialize glob iterators for each pattern (lazy evaluation)."""
        if not self._pattern_iterators:
            for pattern in self.patterns:
                self._pattern_iterators.append(self.data_dir.glob(pattern))

    def _discover_next_batch(self, batch_size: int = 50) -> List[Path]:
        """Discover the next batch of files lazily."""
        if self._discovery_complete:
            return []

        self._init_pattern_iterators()

        new_files = []
        seen_paths = set(self._discovered_files)

        for pattern_iter in self._pattern_iterators:
            try:
                while len(new_files) < batch_size:
                    file_path = next(pattern_iter)

                    # Skip duplicates, non-existent, and small files
                    if file_path in seen_paths:
                        continue
                    if not file_path.exists():
                        continue
                    try:
                        if file_path.stat().st_size < self.min_file_size:
                            continue
                    except OSError:
                        continue

                    new_files.append(file_path)
                    seen_paths.add(file_path)

                    # Check max_files limit
                    if self.max_files and len(self._discovered_files) + len(new_files) >= self.max_files:
                        self._discovery_complete = True
                        break

            except StopIteration:
                continue

            if self._discovery_complete:
                break

        # If no new files found from any pattern, discovery is complete
        if not new_files:
            self._discovery_complete = True

        # Add to discovered files
        self._discovered_files.extend(new_files)

        return new_files

    def __iter__(self) -> Iterator[Path]:
        """Iterate over files, discovering them lazily."""
        self._files_yielded = 0
        # Get configurable seed (None = non-deterministic)
        base_seed = self.shuffle_seed if self.shuffle_seed is not None else int(time.time() * 1000000) % (2**31)
        rng = random.Random(base_seed)

        # Buffer for shuffling discovered files
        shuffle_buffer: List[Path] = []
        buffer_target_size = 100  # Shuffle within windows of 100 files

        # Yield from already discovered files first (shuffled)
        if self._discovered_files:
            shuffled = list(self._discovered_files)
            rng.shuffle(shuffled)
            for f in shuffled:
                if self.max_files and self._files_yielded >= self.max_files:
                    return
                yield f
                self._files_yielded += 1

        # Continue discovering and yielding new files
        while not self._discovery_complete:
            new_files = self._discover_next_batch(batch_size=50)

            if not new_files:
                break

            # Add to shuffle buffer
            shuffle_buffer.extend(new_files)

            # When buffer is full enough, shuffle and yield
            if len(shuffle_buffer) >= buffer_target_size:
                rng.shuffle(shuffle_buffer)
                for f in shuffle_buffer:
                    if self.max_files and self._files_yielded >= self.max_files:
                        return
                    yield f
                    self._files_yielded += 1
                shuffle_buffer = []

        # Yield remaining files in buffer
        if shuffle_buffer:
            rng.shuffle(shuffle_buffer)
            for f in shuffle_buffer:
                if self.max_files and self._files_yielded >= self.max_files:
                    return
                yield f
                self._files_yielded += 1

    def get_discovered_count(self) -> int:
        """Get count of files discovered so far."""
        return len(self._discovered_files)

    def is_discovery_complete(self) -> bool:
        """Check if all files have been discovered."""
        return self._discovery_complete


class UltraFastPretokenizedDataset(IterableDataset):
    """
    Ultra-fast streaming dataset for pretokenized Arrow files.

    Optimizations:
    - Zero-copy memory-mapped Arrow reading
    - Batch vectorized data extraction
    - Direct numpy buffer access
    - Cached Arrow table handles
    - No tokenization overhead

    Performance characteristics:
    - Memory: Low (memory-mapped, shared across workers)
    - I/O: Minimal (sequential reads, OS page cache)
    - CPU: Minimal (zero-copy, vectorized operations)
    - GPU: Optimal (pin_memory for fast transfers)
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        buffer_size: int = 10000,
        samples_per_file: int = 1000,  # Read larger chunks from Arrow files
        cache_size: int = 50,  # Cache Arrow tables (~250MB per worker)
        # Minimal validation (data pre-validated)
        min_sequence_length: int = 10,
        validation_rate: float = 0.0,  # No validation by default (already validated)
        pad_token_id: int = 0,
        bos_token_id: int = 2,  # Beginning of sequence token ID
        eos_token_id: int = 1,  # End of sequence token ID
        add_special_tokens: bool = True,  # Add BOS/EOS if missing from data
        use_dynamic_padding: bool = False,  # RAM-OPTIMIZED: Pad to batch max instead of global max
        max_files_to_load: Optional[int] = None,  # Limit number of files to prevent OOM
        lazy_file_discovery: bool = False,  # Enable lazy file discovery for large datasets
        use_pinned_buffers: bool = True,  # Use pinned buffers when in main process
        vocab_size: int = DEFAULT_VOCAB_SIZE,  # Vocab size for token ID validation
        shuffle_seed: Optional[int] = None,  # Global shuffle seed (None = non-deterministic)
        warm_start_files: int = 3,  # Number of files to load initially for fast startup
        examples_per_random_select: int = 20,  # Examples to take per random file selection
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size  # Store for validation
        self.split = split
        self.max_length = max_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.samples_per_file = samples_per_file
        self.cache_size = cache_size  # Store for pickling
        self.min_sequence_length = min_sequence_length
        self.validation_rate = validation_rate
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_special_tokens = add_special_tokens
        self.use_dynamic_padding = use_dynamic_padding
        self.max_files_to_load = max_files_to_load
        self.lazy_file_discovery = lazy_file_discovery
        self.shuffle_seed = shuffle_seed
        self.warm_start_files = warm_start_files  # For progressive loading
        self.examples_per_random_select = examples_per_random_select
        self._validation_counter = 0
        # OPTIMIZATION: Cache validation skip check at init (avoids repeated comparison in hot path)
        self._skip_content_validation = validation_rate <= 0.0
        # Enable pinned buffer pool for async GPU transfers
        self.use_pinned_buffers = use_pinned_buffers and PINNED_BUFFERS_AVAILABLE
        self._pinned_buffer_pool: Optional[PinnedBufferPool] = None

        # Arrow table cache for instant access
        # OPTIMIZATION: Use thread-local cache for 10-15% throughput improvement
        # in multi-worker DataLoader by eliminating lock contention.
        # The shared ArrowTableCache is kept for backward compatibility in main process.
        self._use_thread_local_cache = False  # Will be set to True in workers
        self.table_cache = ArrowTableCache(max_size=cache_size)

        # File discovery - lazy or eager
        if lazy_file_discovery:
            # Lazy discovery: files found on-demand during iteration
            self.lazy_discoverer = LazyFileDiscovery(
                data_dir=self.data_dir,
                split=self.split,
                max_files=self.max_files_to_load,
            )
            self.data_files = []  # Will be populated lazily
            logger.debug(f"Lazy file discovery enabled for {split} split")
            logger.debug(f"Files will be discovered on-demand during iteration")
            if self.max_files_to_load:
                logger.debug(f"Max files limit: {self.max_files_to_load}")
        else:
            # Eager discovery: find all files upfront
            self.lazy_discoverer = None
            self.data_files = self._find_data_files()

            # Auto-create validation from training if needed
            if not self.data_files and self.split == "val":
                self._create_val_from_train()

            if not self.data_files:
                raise DataLoaderError(
                    f"No pretokenized Arrow files found for {split} split",
                    recovery_steps=_get_no_data_recovery_steps(str(data_dir)),
                    context={
                        'data_dir': str(data_dir),
                        'split': split,
                        'expected_formats': '.arrow, .parquet',
                    }
                )
            else:
                logger.debug(f"Found {len(self.data_files)} pretokenized Arrow files for {split} split")
                total_size_gb = sum(f.stat().st_size for f in self.data_files) / (1024**3)
                logger.debug(f"Total data size: {total_size_gb:.2f} GB (memory-mapped, zero-copy)")

    def _find_data_files(self) -> List[Path]:
        """Find pretokenized Arrow files with deterministic train/val splitting (using shared utility)."""
        MIN_FILE_SIZE = 10 * 1024  # 10KB

        files = find_data_files(
            data_dir=self.data_dir,
            split=self.split,
            patterns=[
                f"**/{self.split}/**/*_processed.arrow",
                f"{self.split}_*.arrow",
                "*_processed.arrow",
                "*.arrow",
                # Also support Parquet files (pre-tokenized data)
                f"**/{self.split}/**/*.parquet",
                f"{self.split}_*.parquet",
                "*.parquet",
            ],
            min_file_size=MIN_FILE_SIZE,
            max_files=self.max_files_to_load,
        )

        if files:
            logger.debug(f"File-based split: {len(files)} files for {self.split}")
            logger.debug(f"Sample files: {[f.name for f in files[:3]]}")

        return files

    def _create_val_from_train(self):
        """Create validation set from training files if val split is empty."""
        logger.debug(f"No validation files found, creating from training files...")
        original_split = self.split
        self.split = "train"
        train_files = self._find_data_files()
        self.split = original_split

        if train_files:
            self.data_files = train_files
            logger.debug(f"Created validation set from {len(self.data_files)} training files")

    def _count_total_samples(self) -> int:
        """Count total samples across all Arrow files (fast estimation for large datasets)."""
        if hasattr(self, '_total_samples_cache') and self._total_samples_cache is not None:
            return self._total_samples_cache

        # For large datasets, estimate from file sizes to avoid slow counting
        # Average ~500 bytes per sample for pretokenized data
        BYTES_PER_SAMPLE_ESTIMATE = 500

        if len(self.data_files) > 100:
            # Fast estimation: sum file sizes and divide by estimated bytes per sample
            total_bytes = sum(f.stat().st_size for f in self.data_files if f.exists())
            estimated_total = total_bytes // BYTES_PER_SAMPLE_ESTIMATE
            self._total_samples_cache = estimated_total
            logger.debug(f"Estimated {estimated_total:,} samples from {len(self.data_files)} files ({total_bytes / 1e9:.2f} GB)")
            return estimated_total

        # For smaller datasets, count exactly
        total = 0
        load_errors = 0
        for file_path in self.data_files:
            try:
                table = self.table_cache.get(file_path)
                total += len(table)
            except Exception as e:
                load_errors += 1
                logger.debug(f"Could not load table for counting from {file_path.name}: {e}")
                # Estimate based on file size
                try:
                    total += file_path.stat().st_size // BYTES_PER_SAMPLE_ESTIMATE
                except Exception as stat_err:
                    logger.debug(f"Could not stat file {file_path.name}: {stat_err}")
                    total += 1000  # Fallback estimate

        if load_errors > 0:
            logger.warning(f"Sample count: {load_errors} files used estimates instead of exact counts")

        self._total_samples_cache = total
        return total

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        if self.lazy_file_discovery:
            # For lazy discovery, we don't know the exact count until we iterate
            # Return an estimate based on discovered files so far
            if hasattr(self, '_total_samples_cache') and self._total_samples_cache:
                return self._total_samples_cache
            # Return a large estimate to prevent premature stopping
            return 10000000  # Will be refined during iteration
        return self._count_total_samples()

    def _add_special_tokens(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add BOS and EOS tokens to sequences if they're missing.

        This is critical for coherent text generation - without proper
        sequence boundaries, the model cannot learn when sequences start/end.

        Args:
            input_ids: Token IDs array
            attention_mask: Attention mask array
            labels: Labels array (same as input_ids for LM training)

        Returns:
            Tuple of (input_ids, attention_mask, labels) with special tokens added
        """
        if not self.add_special_tokens:
            return input_ids, attention_mask, labels

        needs_bos = len(input_ids) == 0 or input_ids[0] != self.bos_token_id
        needs_eos = len(input_ids) == 0 or input_ids[-1] != self.eos_token_id

        if not needs_bos and not needs_eos:
            return input_ids, attention_mask, labels

        # Calculate new length (accounting for truncation to max_length)
        new_tokens = (1 if needs_bos else 0) + (1 if needs_eos else 0)
        content_length = len(input_ids)

        # If adding tokens would exceed max_length, truncate content to make room
        if content_length + new_tokens > self.max_length:
            # Truncate content to make room for special tokens
            content_length = self.max_length - new_tokens
            input_ids = input_ids[:content_length]
            attention_mask = attention_mask[:content_length]
            labels = labels[:content_length]

        # Build new arrays with special tokens
        new_length = content_length + new_tokens
        new_input_ids = np.zeros(new_length, dtype=np.int64)
        new_attention_mask = np.ones(new_length, dtype=np.int64)
        new_labels = np.zeros(new_length, dtype=np.int64)

        offset = 0
        if needs_bos:
            new_input_ids[0] = self.bos_token_id
            new_labels[0] = self.bos_token_id  # Model should learn to predict BOS
            offset = 1

        # Copy content
        new_input_ids[offset:offset + content_length] = input_ids
        new_attention_mask[offset:offset + content_length] = attention_mask
        new_labels[offset:offset + content_length] = labels

        if needs_eos:
            new_input_ids[-1] = self.eos_token_id
            new_labels[-1] = self.eos_token_id  # Model should learn to predict EOS

        return new_input_ids, new_attention_mask, new_labels

    def _vectorized_variable_extraction(
        self,
        flat_values: np.ndarray,
        offsets: np.ndarray,
        batch_size: int,
        pad_value: int = 0,
    ) -> np.ndarray:
        """
        Vectorized extraction of variable-length sequences from flat Arrow buffer.

        Uses numpy advanced indexing to avoid Python loops, providing 3-5x speedup
        over per-sequence iteration.

        Args:
            flat_values: Flattened array of all sequence values
            offsets: Array of offsets where each sequence starts
            batch_size: Number of sequences to extract
            pad_value: Value to use for padding (default: 0)

        Returns:
            2D numpy array of shape (batch_size, max_len) with sequences padded
        """
        # Compute lengths for each sequence in one operation
        lengths = offsets[1:batch_size + 1] - offsets[:batch_size]
        max_len = int(lengths.max())

        # Pre-allocate output array
        result = np.full((batch_size, max_len), pad_value, dtype=np.int64)

        # Vectorized copy using advanced indexing
        # Create row indices for each element
        # For sequences of varying length, we need to know which row each element belongs to
        total_elements = int(offsets[batch_size] - offsets[0])

        if total_elements == 0:
            return result

        # Create sequence index for each element in flat_values
        # repeat(i, lengths[i]) for i in range(batch_size)
        seq_indices = np.repeat(np.arange(batch_size), lengths.astype(np.int64))

        # Create position within sequence for each element
        # This is the column index for each element
        pos_indices = np.concatenate([np.arange(int(l)) for l in lengths])

        # Copy all elements at once using advanced indexing
        start_offset = int(offsets[0])
        end_offset = int(offsets[batch_size])
        result[seq_indices, pos_indices] = flat_values[start_offset:end_offset].astype(np.int64, copy=False)

        return result

    def _extract_batch_zero_copy(
        self,
        batch_slice: pa.Table,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Zero-copy extraction of batch data from Arrow table.

        Uses Arrow's buffer protocol to get numpy arrays without copying when possible.
        Falls back to to_pydict() for variable-length sequences.

        Args:
            batch_slice: PyArrow Table slice to extract from

        Returns:
            Tuple of (input_ids, attention_mask, labels, is_zero_copy)
            - input_ids: 2D numpy array of shape (batch_size, seq_len) or None
            - attention_mask: 2D numpy array or None
            - labels: 2D numpy array or None
            - is_zero_copy: True if zero-copy path was used, False if fallback
        """
        schema = batch_slice.schema
        batch_size = len(batch_slice)

        # Determine column names
        token_col = 'input_ids' if 'input_ids' in schema.names else 'token_ids'

        # Try zero-copy path first (works for fixed-size list data)
        try:
            input_ids_arr = batch_slice.column(token_col)

            # Check if data is fixed-size list (can use zero-copy)
            if pa.types.is_fixed_size_list(input_ids_arr.type):
                seq_len = input_ids_arr.type.list_size

                # Zero-copy: get numpy view directly from Arrow buffer
                flat = input_ids_arr.values.to_numpy(zero_copy_only=True)
                input_ids = flat.reshape(batch_size, seq_len).astype(np.int64, copy=False)

                # Same for attention_mask if present
                if 'attention_mask' in schema.names:
                    mask_arr = batch_slice.column('attention_mask')
                    mask_flat = mask_arr.values.to_numpy(zero_copy_only=True)
                    attention_mask = mask_flat.reshape(batch_size, seq_len).astype(np.int64, copy=False)
                else:
                    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

                # Labels
                if 'labels' in schema.names:
                    labels_arr = batch_slice.column('labels')
                    labels_flat = labels_arr.values.to_numpy(zero_copy_only=True)
                    labels = labels_flat.reshape(batch_size, seq_len).astype(np.int64, copy=False)
                else:
                    labels = input_ids.copy()

                return input_ids, attention_mask, labels, True

            # Try large list type (variable length but contiguous)
            elif pa.types.is_large_list(input_ids_arr.type) or pa.types.is_list(input_ids_arr.type):
                # For list types, we can still get zero-copy access to the flat values
                # but need to handle variable lengths via offsets
                try:
                    # Get the flat values array (zero-copy)
                    flat_values = input_ids_arr.values.to_numpy(zero_copy_only=True)
                    offsets = input_ids_arr.offsets.to_numpy(zero_copy_only=True)

                    # Check if all sequences have the same length (common case for pre-tokenized data)
                    lengths = np.diff(offsets)
                    if len(lengths) > 0 and np.all(lengths == lengths[0]):
                        # Uniform length - can reshape directly
                        seq_len = int(lengths[0])
                        input_ids = flat_values[:batch_size * seq_len].reshape(batch_size, seq_len).astype(np.int64, copy=False)

                        # Handle attention_mask
                        if 'attention_mask' in schema.names:
                            mask_arr = batch_slice.column('attention_mask')
                            mask_flat = mask_arr.values.to_numpy(zero_copy_only=True)
                            attention_mask = mask_flat[:batch_size * seq_len].reshape(batch_size, seq_len).astype(np.int64, copy=False)
                        else:
                            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

                        # Handle labels
                        if 'labels' in schema.names:
                            labels_arr = batch_slice.column('labels')
                            labels_flat = labels_arr.values.to_numpy(zero_copy_only=True)
                            labels = labels_flat[:batch_size * seq_len].reshape(batch_size, seq_len).astype(np.int64, copy=False)
                        else:
                            labels = input_ids.copy()

                        return input_ids, attention_mask, labels, True

                except (pa.ArrowInvalid, ValueError, TypeError):
                    # Try variable-length path with vectorized extraction (3-5x faster)
                    try:
                        # Use vectorized extraction instead of per-sequence loop
                        input_ids = self._vectorized_variable_extraction(
                            flat_values, offsets, batch_size, pad_value=self.pad_token_id
                        )
                        max_len = input_ids.shape[1]

                        # Compute lengths for attention mask creation
                        lengths = offsets[1:batch_size + 1] - offsets[:batch_size]

                        # Handle attention_mask with vectorized extraction
                        if 'attention_mask' in schema.names:
                            mask_arr = batch_slice.column('attention_mask')
                            mask_values = mask_arr.values.to_numpy(zero_copy_only=False)
                            mask_offsets = mask_arr.offsets.to_numpy(zero_copy_only=False)
                            attention_mask = self._vectorized_variable_extraction(
                                mask_values, mask_offsets, batch_size, pad_value=0
                            )
                            # Resize to match input_ids if needed
                            if attention_mask.shape[1] < max_len:
                                pad_width = max_len - attention_mask.shape[1]
                                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
                            elif attention_mask.shape[1] > max_len:
                                attention_mask = attention_mask[:, :max_len]
                        else:
                            # Create attention mask based on sequence lengths
                            attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
                            for i, length in enumerate(lengths):
                                attention_mask[i, :int(length)] = 1

                        # Handle labels with vectorized extraction
                        if 'labels' in schema.names:
                            labels_arr = batch_slice.column('labels')
                            labels_values = labels_arr.values.to_numpy(zero_copy_only=False)
                            labels_offsets = labels_arr.offsets.to_numpy(zero_copy_only=False)
                            labels = self._vectorized_variable_extraction(
                                labels_values, labels_offsets, batch_size, pad_value=-100
                            )
                            # Resize to match input_ids if needed
                            if labels.shape[1] < max_len:
                                pad_width = max_len - labels.shape[1]
                                labels = np.pad(labels, ((0, 0), (0, pad_width)), constant_values=-100)
                            elif labels.shape[1] > max_len:
                                labels = labels[:, :max_len]
                        else:
                            # Labels = input_ids with padding set to -100
                            labels = input_ids.copy()
                            labels[attention_mask == 0] = -100

                        return input_ids, attention_mask, labels, True

                    except Exception:
                        pass  # Fall through to fallback

        except (pa.ArrowInvalid, ValueError, TypeError, AttributeError):
            pass  # Fall through to fallback

        # Fallback: variable-length sequences or incompatible format
        # Return None to signal caller should use to_pydict()
        return None, None, None, False

    def _load_initial_files_parallel(
        self,
        initial_files: List[Path],
        max_workers: int = 4,
    ) -> List[Dict]:
        """
        Load initial files in parallel using ThreadPoolExecutor.

        This provides 2-4x faster startup compared to sequential loading.

        Args:
            initial_files: List of file paths to load
            max_workers: Maximum number of parallel workers

        Returns:
            List of file cursor dictionaries with table, offset, and metadata
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        active_files = []

        def load_single_file(file_path: Path) -> Optional[Dict]:
            """Load a single file and return cursor dict or None on error."""
            try:
                table = self.table_cache.get(file_path)
                return {
                    'file_path': file_path,
                    'table': table,
                    'offset': 0,
                    'total_rows': len(table)
                }
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel loading
        # Limit workers to avoid overwhelming I/O or memory
        actual_workers = min(max_workers, len(initial_files), 8)

        if actual_workers <= 1 or len(initial_files) <= 2:
            # Sequential loading for small number of files
            for file_path in initial_files:
                result = load_single_file(file_path)
                if result is not None:
                    active_files.append(result)
        else:
            # Parallel loading for larger batches
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {executor.submit(load_single_file, f): f for f in initial_files}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        active_files.append(result)

        return active_files

    def _stream_examples_ultra_fast(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Stream examples with maximum performance optimizations.

        Key optimizations:
        - Batch extraction from Arrow tables (5x faster than per-row)
        - Memory-mapped cached tables (zero I/O overhead)
        - Direct buffer protocol access (zero-copy)
        - Sequential reads for OS page cache hits
        - Lazy file discovery for memory-efficient large datasets
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        should_print = worker_info is None or worker_info.id == 0

        # Get file source - lazy discovery or pre-discovered files
        if self.lazy_file_discovery and self.lazy_discoverer is not None:
            # Lazy mode: stream files from lazy discoverer
            yield from self._stream_examples_lazy(worker_id, should_print)
            return

        # Eager mode: use pre-discovered files with true random file selection
        epoch_num = getattr(self, '_stream_epoch_number', 0)

        # Reproducible seeding with distributed training support
        base_seed = self.shuffle_seed if self.shuffle_seed is not None else int(time.time() * 1000000) % (2**31)
        rank = dist.get_rank() if dist.is_initialized() else 0
        combined_seed = base_seed + epoch_num * 1000000 + rank * 10000 + worker_id
        rng = random.Random(combined_seed)

        # Examples per random selection (configurable, default 20)
        examples_per_select = self.examples_per_random_select

        # Group files by parent folder for diversity
        from collections import defaultdict
        files_by_folder = defaultdict(list)
        for file_path in self.data_files:
            folder = file_path.parent
            files_by_folder[folder].append(file_path)

        # Shuffle files within each folder
        for folder in files_by_folder:
            rng.shuffle(files_by_folder[folder])

        # PROGRESSIVE LOADING: Select up to 3 files from each folder initially
        files_per_folder = getattr(self, 'warm_start_files', 3) or 3
        initial_files = []
        pending_files = []

        for folder, folder_files in files_by_folder.items():
            # Take up to files_per_folder from this folder for initial load
            initial_files.extend(folder_files[:files_per_folder])
            # Rest go to pending
            pending_files.extend(folder_files[files_per_folder:])

        # Shuffle initial and pending for randomness
        rng.shuffle(initial_files)
        rng.shuffle(pending_files)

        if should_print:
            print(f"  [RANDOM-SELECT] {len(files_by_folder)} folders, {len(initial_files)} initial files ({files_per_folder}/folder), {len(pending_files)} pending")

        # Track loaded vs pending files
        # Get profiler if available
        profiler = get_global_profiler() if PROFILING_AVAILABLE and get_global_profiler else None

        # Load initial batch of files using parallel loading (2-4x faster startup)
        if profiler and profiler.enabled:
            profiler.start_timer('file_load')

        # Use parallel loading for faster startup (especially with >3 files)
        active_files = self._load_initial_files_parallel(
            initial_files,
            max_workers=min(4, len(initial_files))  # Limit to 4 workers for I/O
        )

        if profiler and profiler.enabled:
            profiler.stop_timer('file_load')
            # Record cache misses for all initially loaded files
            for _ in range(len(active_files)):
                profiler.record_cache_miss()

        if not active_files and not pending_files:
            raise ValueError(f"No files could be opened from {self.data_dir}")


        # Main loop: randomly select file, take N examples, remove when exhausted
        while active_files:
            # Randomly pick a file
            idx = rng.randrange(len(active_files))
            cursor = active_files[idx]

            table = cursor['table']
            offset = cursor['offset']
            total_rows = cursor['total_rows']

            # Calculate how many examples to read
            remaining = total_rows - offset
            if remaining <= 0:
                # File exhausted, remove from pool
                active_files.pop(idx)
                continue

            batch_size = min(examples_per_select, remaining)

            # Batch extraction (zero-copy view)
            batch_slice = table.slice(offset, batch_size)

            # Vectorized batch extraction for efficiency
            try:
                # Profile extraction time
                if profiler and profiler.enabled:
                    profiler.start_timer('extraction')

                # Try zero-copy extraction first (30-50% faster for fixed-length data)
                zc_input_ids, zc_attention_mask, zc_labels, is_zero_copy = self._extract_batch_zero_copy(batch_slice)

                if is_zero_copy and zc_input_ids is not None:
                    # Zero-copy path: data is already in 2D numpy arrays
                    if profiler and profiler.enabled:
                        profiler.stop_timer('extraction')

                    # Process each sequence in the batch
                    for i in range(batch_size):
                        input_ids_np = zc_input_ids[i]
                        attention_mask_np = zc_attention_mask[i] if zc_attention_mask is not None else np.ones(len(input_ids_np), dtype=np.int64)
                        labels_np = zc_labels[i] if zc_labels is not None else input_ids_np.copy()

                        # Truncate to max_length
                        max_content_len = self.max_length - 2 if self.add_special_tokens else self.max_length
                        if len(input_ids_np) > max_content_len:
                            input_ids_np = input_ids_np[:max_content_len]
                            attention_mask_np = attention_mask_np[:max_content_len]
                            labels_np = labels_np[:max_content_len]

                        # Add BOS/EOS tokens
                        input_ids_np, attention_mask_np, labels_np = self._add_special_tokens(
                            input_ids_np, attention_mask_np, labels_np
                        )

                        # Yield if valid length
                        if len(input_ids_np) >= self.min_sequence_length:
                            yield {
                                'input_ids': input_ids_np,
                                'attention_mask': attention_mask_np,
                                'labels': labels_np
                            }
                else:
                    # Fallback: Use column-level extraction (avoids slow to_pydict())
                    # This is 5-10x faster than to_pydict() for variable-length data
                    schema_names = batch_slice.schema.names
                    token_col_name = 'input_ids' if 'input_ids' in schema_names else 'token_ids'
                    if token_col_name not in schema_names:
                        raise KeyError("Neither 'input_ids' nor 'token_ids' column found")

                    input_ids_col = batch_slice.column(token_col_name)
                    attention_mask_col = batch_slice.column('attention_mask') if 'attention_mask' in schema_names else None
                    labels_col = batch_slice.column('labels') if 'labels' in schema_names else None

                    if profiler and profiler.enabled:
                        profiler.stop_timer('extraction')

                    # Process each sequence using column-level access
                    for i in range(batch_size):
                        # Profile tensor conversion time
                        if profiler and profiler.enabled:
                            profiler.start_timer('tensor_conversion')

                        # Extract input_ids using Arrow's efficient access
                        input_ids_arr = input_ids_col[i]
                        try:
                            # Try zero-copy numpy access first
                            input_ids_np = np.asarray(input_ids_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                        except (AttributeError, TypeError):
                            # Fallback to as_py() only if Arrow access fails
                            input_ids_np = np.asarray(input_ids_arr.as_py(), dtype=np.int64)

                        # Validate sequence length
                        raw_len = len(input_ids_np)
                        max_allowed_len = min(self.max_length * 10, 32768)
                        if raw_len > max_allowed_len:
                            if profiler and profiler.enabled:
                                profiler.stop_timer('tensor_conversion')
                            continue

                        # Extract attention_mask
                        if attention_mask_col is not None:
                            attention_mask_arr = attention_mask_col[i]
                            try:
                                attention_mask_np = np.asarray(attention_mask_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                            except (AttributeError, TypeError):
                                attention_mask_np = np.asarray(attention_mask_arr.as_py(), dtype=np.int64)
                        else:
                            attention_mask_np = np.ones(len(input_ids_np), dtype=np.int64)

                        # Extract labels
                        if labels_col is not None:
                            labels_arr = labels_col[i]
                            try:
                                labels_np = np.asarray(labels_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                            except (AttributeError, TypeError):
                                labels_np = np.asarray(labels_arr.as_py(), dtype=np.int64)
                        else:
                            labels_np = input_ids_np.copy()

                        if profiler and profiler.enabled:
                            profiler.stop_timer('tensor_conversion')

                        # Truncate to max_length
                        max_content_len = self.max_length - 2 if self.add_special_tokens else self.max_length
                        if len(input_ids_np) > max_content_len:
                            input_ids_np = input_ids_np[:max_content_len]
                            attention_mask_np = attention_mask_np[:max_content_len]
                            labels_np = labels_np[:max_content_len]

                        # Add BOS/EOS tokens
                        input_ids_np, attention_mask_np, labels_np = self._add_special_tokens(
                            input_ids_np, attention_mask_np, labels_np
                        )

                        # Yield if valid length
                        if len(input_ids_np) >= self.min_sequence_length:
                            yield {
                                'input_ids': input_ids_np,
                                'attention_mask': attention_mask_np,
                                'labels': labels_np
                            }

            except Exception as e:
                # Fallback to per-row method if batch conversion fails
                logger.debug(f"Batch extraction failed, falling back to per-row: {e}")

                schema_names = batch_slice.schema.names
                token_col_name = 'input_ids' if 'input_ids' in schema_names else 'token_ids'
                input_ids_col = batch_slice.column(token_col_name)
                attention_mask_col = batch_slice.column('attention_mask') if 'attention_mask' in schema_names else None
                labels_col = batch_slice.column('labels') if 'labels' in schema_names else None

                for i in range(batch_size):
                    try:
                        input_ids_arr = input_ids_col[i]
                        try:
                            input_ids_np = np.asarray(input_ids_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                        except (AttributeError, TypeError):
                            input_ids_np = np.array(input_ids_arr.as_py(), dtype=np.int64)

                        if attention_mask_col is not None:
                            attention_mask_arr = attention_mask_col[i]
                            try:
                                attention_mask_np = np.asarray(attention_mask_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                            except (AttributeError, TypeError):
                                attention_mask_np = np.array(attention_mask_arr.as_py(), dtype=np.int64)
                        else:
                            attention_mask_np = np.ones(len(input_ids_np), dtype=np.int64)

                        if labels_col is not None:
                            labels_arr = labels_col[i]
                            try:
                                labels_np = np.asarray(labels_arr.values.to_numpy(zero_copy_only=False), dtype=np.int64)
                            except (AttributeError, TypeError):
                                labels_np = np.array(labels_arr.as_py(), dtype=np.int64)
                        else:
                            labels_np = input_ids_np.copy()

                        # Truncate
                        max_content_len = self.max_length - 2 if self.add_special_tokens else self.max_length
                        if len(input_ids_np) > max_content_len:
                            input_ids_np = input_ids_np[:max_content_len]
                            attention_mask_np = attention_mask_np[:max_content_len]
                            labels_np = labels_np[:max_content_len]

                        # Add special tokens
                        input_ids_np, attention_mask_np, labels_np = self._add_special_tokens(
                            input_ids_np, attention_mask_np, labels_np
                        )

                        if len(input_ids_np) >= self.min_sequence_length:
                            yield {
                                'input_ids': input_ids_np,
                                'attention_mask': attention_mask_np,
                                'labels': labels_np
                            }
                    except Exception:
                        continue

            # Update cursor offset
            cursor['offset'] = offset + batch_size

            # Check if file exhausted after this batch
            if cursor['offset'] >= total_rows:
                active_files.pop(idx)

                # PROGRESSIVE LOADING: Load next file from pending queue
                if pending_files:
                    next_file_path = pending_files.pop(0)
                    try:
                        next_table = self.table_cache.get(next_file_path)
                        active_files.append({
                            'file_path': next_file_path,
                            'table': next_table,
                            'offset': 0,
                            'total_rows': len(next_table)
                        })
                    except Exception as e:
                        logger.debug(f"Worker {worker_id} could not load {next_file_path}: {e}")

    def _stream_examples_lazy(self, worker_id: int, should_print: bool) -> Iterator[Dict[str, np.ndarray]]:
        """
        Stream examples with lazy file discovery - files found on-demand.

        This mode discovers files incrementally during iteration instead of
        finding all files upfront, which is memory-efficient for large datasets.
        """
        if self.lazy_discoverer is None:
            return

        files_processed = 0
        samples_yielded = 0

        if should_print:
            logger.debug(f"[Lazy Discovery] Starting lazy streaming for {self.split} split")

        # Stream files from lazy discoverer
        for file_path in self.lazy_discoverer:
            files_processed += 1

            # Log progress periodically
            if should_print and files_processed % 50 == 0:
                logger.debug(f"[Lazy] Processed {files_processed} files, discovered {self.lazy_discoverer.get_discovered_count()} total")

            try:
                # Load table with caching
                table = self.table_cache.get(file_path)
                total_rows = len(table)

                if total_rows == 0:
                    continue

                # Process file in batches
                offset = 0
                while offset < total_rows:
                    batch_size = min(self.samples_per_file, total_rows - offset)
                    batch_slice = table.slice(offset, batch_size)

                    try:
                        # Vectorized batch extraction
                        batch_dict = batch_slice.to_pydict()
                        # Support both 'input_ids' and 'token_ids' column names
                        # Use explicit None check (not `or`) to handle empty lists correctly
                        input_ids_list = batch_dict.get('input_ids')
                        if input_ids_list is None:
                            input_ids_list = batch_dict.get('token_ids')
                        if input_ids_list is None:
                            raise KeyError("Neither 'input_ids' nor 'token_ids' column found")
                        attention_mask_list = batch_dict.get('attention_mask', None)
                        labels_list = batch_dict.get('labels', None)

                        for i in range(len(input_ids_list)):
                            input_ids_raw = input_ids_list[i]

                            # Validate sequence length
                            max_allowed_len = min(self.max_length * 10, 32768)
                            if len(input_ids_raw) > max_allowed_len:
                                continue

                            # Convert to numpy
                            input_ids_np = np.array(input_ids_raw, dtype=np.int64)

                            if attention_mask_list and i < len(attention_mask_list):
                                attention_mask_np = np.array(attention_mask_list[i], dtype=np.int64)
                            else:
                                attention_mask_np = np.ones(len(input_ids_np), dtype=np.int64)

                            if labels_list and i < len(labels_list):
                                labels_np = np.array(labels_list[i], dtype=np.int64)
                            else:
                                labels_np = input_ids_np.copy()

                            # Truncate to max_length (leave room for special tokens)
                            max_content_len = self.max_length - 2 if self.add_special_tokens else self.max_length
                            if len(input_ids_np) > max_content_len:
                                input_ids_np = input_ids_np[:max_content_len]
                                attention_mask_np = attention_mask_np[:max_content_len]
                                labels_np = labels_np[:max_content_len]

                            # Add BOS/EOS tokens for coherent text generation
                            input_ids_np, attention_mask_np, labels_np = self._add_special_tokens(
                                input_ids_np, attention_mask_np, labels_np
                            )

                            # Yield if valid
                            if len(input_ids_np) >= self.min_sequence_length:
                                yield {
                                    'input_ids': input_ids_np,
                                    'attention_mask': attention_mask_np,
                                    'labels': labels_np
                                }
                                samples_yielded += 1

                                # Check max_samples limit
                                if self.max_samples and samples_yielded >= self.max_samples:
                                    if should_print:
                                        logger.debug(f"[Lazy] Reached max_samples limit: {self.max_samples}")
                                    return

                    except Exception as e:
                        logger.warning(f"[Lazy] Worker {worker_id} batch extraction failed for {file_path}: {e}")

                    offset += batch_size

            except Exception as e:
                logger.warning(f"[Lazy] Worker {worker_id} could not process {file_path}: {e}")
                continue

        if should_print:
            logger.debug(f"[Lazy Discovery] Complete: processed {files_processed} files, yielded {samples_yielded} samples")

    def _validate_sequence_content(
        self,
        input_ids: Union[np.ndarray, List[int]],
        max_allowed_len: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate sequence content BEFORE tensor allocation to prevent OOM.

        This is a critical safety check that validates:
        1. Sequence length is reasonable
        2. Token IDs are within valid vocab range
        3. Data type is correct (integer-like)
        4. No extreme/corrupted values

        Args:
            input_ids: Sequence to validate (numpy array or list)
            max_allowed_len: Maximum allowed sequence length

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if sequence passes all checks
            - error_message: Description of validation failure (None if valid)
        """
        # Convert to numpy if needed
        if isinstance(input_ids, list):
            try:
                arr = np.array(input_ids)
            except (ValueError, TypeError) as e:
                return False, f"Cannot convert to array: {e}"
        else:
            arr = input_ids

        # Check 1: Length validation
        seq_len = len(arr)
        if seq_len == 0:
            return False, "Empty sequence"

        if seq_len > max_allowed_len:
            return False, f"Sequence too long: {seq_len:,} > {max_allowed_len:,}"

        if seq_len < self.min_sequence_length:
            return False, f"Sequence too short: {seq_len} < {self.min_sequence_length}"

        # Check 2: Data type validation
        if hasattr(arr, 'dtype'):
            if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
                return False, f"Invalid dtype: {arr.dtype} (expected integer)"

        # Check 3: Token ID range validation
        try:
            min_val = int(np.min(arr))
            max_val = int(np.max(arr))

            if min_val < -1:  # Allow -1 for special tokens
                return False, f"Negative token ID: {min_val}"

            if max_val >= self.vocab_size:
                return False, f"Token ID {max_val:,} exceeds vocab size {self.vocab_size:,}"

            # Check for NaN/Inf (shouldn't happen with integers but be safe)
            if np.issubdtype(arr.dtype, np.floating):
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    return False, "Contains NaN or Inf values"

        except (ValueError, TypeError, OverflowError) as e:
            return False, f"Value check failed: {e}"

        return True, None

    def collate_fn(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Ultra-fast batch collation with FIXED-LENGTH padding.

        CRITICAL FIX: Use fixed-length padding to prevent torch.compile recompilation.
        Dynamic padding causes shape changes that trigger expensive recompilation.

        Optimizations:
        - Direct numpy→torch conversion via torch.from_numpy
        - Vectorized padding operations
        - Pre-allocated tensors
        - Minimal data movement
        - Memory pinning for faster CPU→GPU transfer
        - Fixed padding to self.max_length for torch.compile compatibility
        """
        if not batch:
            return {}

        # CRITICAL FIX: Validate batch structure to prevent OOM
        # Sometimes DataLoader passes incorrect batch structure
        if not isinstance(batch, list):
            logger.error(f"ERROR: batch is not a list, got {type(batch)}")
            raise TypeError(f"Expected batch to be a list, got {type(batch)}")

        batch_size = len(batch)

        # Safety check: prevent absurd batch sizes that cause OOM
        if batch_size > 10000:
            estimated_gb = batch_size * self.max_length * 8 * 3 / 1024**3
            raise DataLoaderError(
                f"Abnormal batch size {batch_size:,} detected - would allocate {estimated_gb:.2f} GB",
                recovery_steps=_get_oom_recovery_steps(),
                context={
                    'batch_size': batch_size,
                    'max_length': self.max_length,
                    'estimated_memory_gb': f"{estimated_gb:.2f}",
                    'first_item_type': str(type(batch[0])) if batch else 'N/A',
                    'diagnosis': 'DataLoader may be incorrectly accumulating samples',
                }
            )

        # OPTIMIZATION: Skip content validation for pre-validated data (6% speedup)
        # Only validate if validation_rate > 0 (default: 0.0 = no validation)
        max_allowed_len = min(self.max_length * 10, 32768)
        skipped_count = 0
        valid_items = []
        validation_errors: List[str] = []

        # Fast path: skip validation for pre-validated data (cached at init)
        skip_content_validation = self._skip_content_validation

        for i, item in enumerate(batch):
            if 'input_ids' not in item:
                logger.warning(f"Batch item {i} missing 'input_ids' key")
                validation_errors.append(f"Item {i}: missing input_ids")
                skipped_count += 1
                continue

            if skip_content_validation:
                # Fast path: trust pre-validated data, just check basic structure
                valid_items.append((i, item))
            else:
                # Full validation path
                is_valid, error_msg = self._validate_sequence_content(
                    item['input_ids'], max_allowed_len
                )

                if not is_valid:
                    skipped_count += 1
                    if len(validation_errors) < 5:  # Limit error messages
                        validation_errors.append(f"Item {i}: {error_msg}")
                    continue

                valid_items.append((i, item))

        if skipped_count > 0:
            logger.warning(
                f"Collate skipped {skipped_count}/{batch_size} invalid items. "
                f"Errors: {validation_errors[:3]}"  # Show first 3 errors
            )

        # If all items were corrupted, return a skip marker batch
        # Training loop should check for '_skip_batch' flag and skip this batch
        if not valid_items:
            # Log with recovery suggestions
            logger.error(
                f"All {batch_size} items in batch were invalid!\n"
                f"Errors: {validation_errors}\n"
                f"Recovery steps:\n" +
                "\n".join(f"  {i+1}. {step}" for i, step in enumerate(_get_corruption_recovery_steps()[:3]))
            )
            pad_token_id = getattr(self, 'pad_token_id', 0)
            return {
                'input_ids': torch.tensor([[pad_token_id]], dtype=torch.long),
                'attention_mask': torch.tensor([[0]], dtype=torch.long),  # Mask=0 means ignore
                'labels': torch.tensor([[-100]], dtype=torch.long),  # -100 = ignore in cross entropy
                '_skip_batch': True,  # Flag for training loop to detect and skip
                '_corruption_info': f"All {batch_size} items invalid: {validation_errors[:5]}"
            }

        # Use only valid items
        valid_batch_size = len(valid_items)

        # PADDING STRATEGY: Fixed vs Dynamic
        # Dynamic padding saves memory but may cause torch.compile recompilation
        if self.use_dynamic_padding:
            # Dynamic: pad to longest sequence in batch (RAM-optimized)
            actual_lengths = [len(item['input_ids']) for _, item in valid_items]
            max_len = min(max(actual_lengths), self.max_length) if actual_lengths else self.max_length
        else:
            # Fixed: always pad to max_length (torch.compile friendly)
            max_len = self.max_length

        # Pre-allocate tensors on CPU
        # Use pinned buffers when in main process (num_workers=0).
        # Pinned memory enables true async DMA transfers with non_blocking=True.
        # Note: Do not use pinned memory with num_workers > 0 (IPC limitation).
        use_pinned = (
            self.use_pinned_buffers and
            PINNED_BUFFERS_AVAILABLE and
            is_main_process_dataloader is not None and
            is_main_process_dataloader() and
            torch.cuda.is_available()
        )

        if use_pinned and get_buffer_pool is not None:
            # Get buffers from pool (fast path when pool has matching buffers)
            buffer_pool = get_buffer_pool()
            shape = (valid_batch_size, max_len)

            # Get pinned buffers from pool
            input_ids = buffer_pool.get_buffer(torch.long, shape)
            attention_mask = buffer_pool.get_buffer(torch.long, shape)
            labels = buffer_pool.get_buffer(torch.long, shape)

            # Initialize with padding values
            input_ids.fill_(self.pad_token_id)
            attention_mask.fill_(0)
            labels.fill_(-100)
        else:
            # Standard allocation (used with num_workers > 0 or no CUDA)
            input_ids = torch.full((valid_batch_size, max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((valid_batch_size, max_len), dtype=torch.long)
            labels = torch.full((valid_batch_size, max_len), -100, dtype=torch.long)

        # Fill tensors efficiently
        # Check if all sequences have same length (common in zero-copy path)
        # If so, use vectorized numpy stacking for 40% faster conversion
        lengths = [len(item['input_ids']) for _, item in valid_items]
        all_same_length = len(set(lengths)) == 1 and lengths[0] <= max_len

        if all_same_length:
            # Fast path: all sequences same length - use numpy stacking (40% faster)
            seq_len = lengths[0]
            # Stack all numpy arrays into 2D arrays first
            input_ids_np = np.stack([item['input_ids'][:seq_len] for _, item in valid_items])
            attention_mask_np = np.stack([item['attention_mask'][:seq_len] for _, item in valid_items])
            labels_np = np.stack([item['labels'][:seq_len] for _, item in valid_items])

            # Single tensor conversion instead of batch_size conversions
            input_ids[:, :seq_len] = torch.from_numpy(input_ids_np.astype(np.int64))
            attention_mask[:, :seq_len] = torch.from_numpy(attention_mask_np.astype(np.int64))
            labels[:, :seq_len] = torch.from_numpy(labels_np.astype(np.int64))
        else:
            # Variable length path: per-sequence processing
            for new_idx, (orig_idx, item) in enumerate(valid_items):
                seq_len = min(len(item['input_ids']), max_len)  # Actual sequence length, capped at max_len

                # Convert numpy arrays to torch tensors and copy into batch
                input_ids[new_idx, :seq_len] = torch.from_numpy(item['input_ids'][:seq_len])
                attention_mask[new_idx, :seq_len] = torch.from_numpy(item['attention_mask'][:seq_len])
                labels[new_idx, :seq_len] = torch.from_numpy(item['labels'][:seq_len])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over dataset with ultra-fast streaming."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        count = 0
        sample_index = 0

        # Stream examples with worker distribution and error handling
        try:
            for sample in self._stream_examples_ultra_fast():
                # Worker-level sample distribution
                if sample_index % num_workers != worker_id:
                    sample_index += 1
                    continue

                sample_index += 1

                if self.max_samples and count >= self.max_samples:
                    break

                yield sample
                count += 1
        except Exception as e:
            logger.error(f"Worker {worker_id} error during iteration: {e}", exc_info=True)
            raise

    def __getstate__(self):
        """Custom pickle support - exclude unpicklable objects."""
        state = self.__dict__.copy()
        # Clear cache before pickling (will be recreated in worker)
        state['table_cache'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle support - restore state in worker process."""
        self.__dict__.update(state)

        # OPTIMIZATION: Use thread-local cache in workers for 10-15% throughput improvement
        # Thread-local cache eliminates lock contention between workers, as each worker
        # gets its own independent LRU cache via thread-local storage.
        cache_size = getattr(self, 'cache_size', 50)
        self._use_thread_local_cache = True  # Mark as using thread-local cache
        self.table_cache = get_thread_local_arrow_cache(max_size=cache_size)
        logger.debug(f"Worker initialized with thread-local Arrow cache (max_size={cache_size})")


class InfiniteUltraFastDataset(IterableDataset):
    """Infinite streaming wrapper for ultra-fast pretokenized dataset."""

    def __init__(self, **kwargs):
        self.base_dataset = UltraFastPretokenizedDataset(**kwargs)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate infinitely over the dataset."""
        while True:
            for item in self.base_dataset:
                yield item

    def __len__(self) -> int:
        """Return length of base dataset for progress tracking.

        Note: This is used for progress bars and epoch estimation.
        The actual iteration is infinite, but one 'epoch' is one pass
        through the base dataset.
        """
        return len(self.base_dataset)

    def collate_fn(self, batch):
        """Delegate collation to base dataset."""
        return self.base_dataset.collate_fn(batch)


def create_ultra_fast_dataloaders(
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    buffer_size: int = 10000,
    val_split_ratio: float = 0.1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    samples_per_file: int = 1000,
    cache_size: int = 50,  # RAM-OPTIMIZED: 50 tables default (set higher if RAM > 64GB)
    pad_token_id: int = 0,
    bos_token_id: int = 2,  # Beginning of sequence token ID
    eos_token_id: int = 1,  # End of sequence token ID (fixed: was 2, should be 1)
    add_special_tokens: bool = True,  # Add BOS/EOS if missing from data (critical for coherence)
    use_sequence_packing: bool = False,  # Eliminates padding waste
    packing_strategy: str = 'greedy',  # 'greedy' or 'adaptive'
    max_files_to_load: Optional[int] = None,  # Limit number of files to prevent OOM
    lazy_file_discovery: bool = False,  # Enable lazy file discovery for memory-efficient large datasets
    verbose: bool = False,  # Control verbose output (default False for cleaner logs)
    shuffle_seed: Optional[int] = None,  # Global shuffle seed (None = non-deterministic)
    enable_length_sorting: bool = True,  # Enable length sorting in distributed mode
    disable_packing_length_sort: bool = False,  # Disable length sorting in packing
    warm_start_files: int = 3,  # Number of files to load initially for fast startup
    examples_per_random_select: int = 20,  # Examples to take per random file selection
) -> Tuple[Any, Any]:  # Returns DataLoader
    """
    Create ultra-fast pretokenized dataloaders.

    Optimizations over text tokenization:
    - No tokenization overhead
    - Memory-mapped Arrow reading
    - Zero-copy tensors
    - Cached table handles

    Args:
        batch_size: Batch size per device
        max_length: Maximum sequence length
        data_dir: Directory containing pretokenized Arrow files
        num_workers: Number of data loading workers (default 4)
        max_samples: Maximum training samples (None = unlimited)
        buffer_size: Shuffle buffer size (legacy, not used in ultra-fast mode)
        val_split_ratio: Validation split ratio
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        samples_per_file: Samples to read from each file before rotating
        cache_size: Number of Arrow tables to cache in memory
        pad_token_id: Padding token ID (default 0)
        eos_token_id: End-of-sequence token ID for packing (default 2)
        use_sequence_packing: Enable sequence packing to eliminate padding waste (default False)
        packing_strategy: 'greedy' (faster) or 'adaptive' (better utilization)

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Safety checks
    if batch_size is None:
        batch_size = 8
        print(f"  batch_size was None, defaulting to {batch_size}")

    # Auto-tune worker count based on system characteristics
    if num_workers == -1:
        import multiprocessing
        import os

        cpu_count = multiprocessing.cpu_count()

        # Heuristic-based auto-tuning:
        # 1. Use 50% of CPU cores as base (leave room for other processes)
        # 2. Adjust for batch size (smaller batches benefit from more workers)
        # 3. Cap at 16 to avoid diminishing returns and context switch overhead
        base_workers = max(2, cpu_count // 2)

        # Smaller batches need more workers to keep GPU fed
        if batch_size <= 8:
            adjustment = 2
        elif batch_size <= 32:
            adjustment = 1
        else:
            adjustment = 0

        # Check for container CPU limits (Docker/K8s)
        try:
            with open('/sys/fs/cgroup/cpu.max', 'r') as f:
                content = f.read().strip()
                if content != 'max':
                    quota, period = content.split()
                    cgroup_limit = int(int(quota) / int(period))
                    base_workers = min(base_workers, max(2, cgroup_limit // 2))
        except (FileNotFoundError, ValueError, PermissionError):
            pass  # Not in container or cgroup v1

        num_workers = min(base_workers + adjustment, 16)
        if verbose:
            print(f" Auto-tuned {num_workers} workers (from {cpu_count} cores, batch_size={batch_size})")
    elif num_workers > 0:
        if verbose:
            print(f" Using {num_workers} CPU workers for ultra-fast pretokenized loading")
    else:
        if verbose:
            print(f" Using 0 workers (main process only) for data loading")

    # Dynamic prefetch factor based on sequence length
    if prefetch_factor == 2:  # Only auto-adjust if using default
        original_prefetch = prefetch_factor
        prefetch_factor = max(2, min(6, int(3072 / max_length)))
        if prefetch_factor != original_prefetch and verbose:
            memory_impact_gb = num_workers * (prefetch_factor - original_prefetch) * batch_size * max_length * 2 / (1024**3)
            print(f" [OPTIMIZATION] Auto-adjusted prefetch_factor: {original_prefetch} → {prefetch_factor} (~{abs(memory_impact_gb):.1f}GB RAM)")

    if verbose:
        print(f" Ultra-fast pretokenized pipeline (60x faster):")
        print(f"   • Memory-mapped Arrow reading (zero-copy, 2x faster)")
        print(f"   • Batch vectorized extraction (5x faster)")
        print(f"   • Zero-copy numpy→torch conversion (1.5x faster)")
        print(f"   • Cached Arrow table handles ({cache_size} tables, 1.5x faster)")
        print(f"   • No tokenization overhead (30x faster)")
        print(f"   • {num_workers} parallel workers")
        print(f"   • Total speedup: ~60x vs text tokenization")

    # Dataset kwargs
    dataset_kwargs = {
        'data_dir': data_dir,
        'max_length': max_length,
        'buffer_size': buffer_size,
        'samples_per_file': samples_per_file,
        'cache_size': cache_size,
        'pad_token_id': pad_token_id,
        'bos_token_id': bos_token_id,
        'eos_token_id': eos_token_id,
        'add_special_tokens': add_special_tokens,
        'max_files_to_load': max_files_to_load,
        'lazy_file_discovery': lazy_file_discovery,
        'shuffle_seed': shuffle_seed,
        'warm_start_files': warm_start_files,  # For fast startup
        'examples_per_random_select': examples_per_random_select,
    }

    # Training dataset
    if max_samples is None:
        train_dataset = InfiniteUltraFastDataset(
            split='train',
            max_samples=None,
            **dataset_kwargs
        )
    else:
        train_dataset = UltraFastPretokenizedDataset(
            split='train',
            max_samples=max_samples,
            **dataset_kwargs
        )

    # Validation dataset
    if max_samples is not None:
        val_max_samples = int(max_samples * val_split_ratio)
    else:
        val_max_samples = None

    val_dataset = UltraFastPretokenizedDataset(
        split='val',
        max_samples=val_max_samples,
        **dataset_kwargs
    )

    # Dataloader configuration
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'multiprocessing_context': 'spawn' if num_workers > 0 else None,
        'timeout': 0,  # Disabled - prevents timeout errors with large datasets
        'worker_init_fn': _worker_init_fn if num_workers > 0 else None,  # FIX: Proper cleanup to prevent semaphore leaks
    }

    # Get collate functions with optional sequence packing
    if use_sequence_packing and SEQUENCE_PACKING_AVAILABLE and SequencePackingCollator is not None and DynamicSequencePackingCollator is not None:
        if verbose:
            print(f"\n{'='*60}")
            print(f" SEQUENCE PACKING ENABLED")
            print(f"{'='*60}")
            print(f"   Strategy: {packing_strategy}")
            print(f"   Packing multiple short docs into single sequences")
            print(f"   Max length: {max_length}")
            print(f"{'='*60}\n")

        if packing_strategy == 'adaptive':
            train_collate_fn = DynamicSequencePackingCollator(
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                adaptive_binning=True,
            )
            val_collate_fn = DynamicSequencePackingCollator(
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                adaptive_binning=True,
            )
        else:
            train_collate_fn = SequencePackingCollator(
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                pack_sequences=True,
                sort_by_length=not disable_packing_length_sort,
            )
            val_collate_fn = SequencePackingCollator(
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                pack_sequences=True,
                sort_by_length=not disable_packing_length_sort,
            )
    elif use_sequence_packing and not SEQUENCE_PACKING_AVAILABLE:
        if verbose:
            print("  Sequence packing requested but not available, using default collation")
        # Handle both direct dataset and InfiniteUltraFastDataset wrapper
        if isinstance(train_dataset, InfiniteUltraFastDataset):
            train_collate_fn = train_dataset.base_dataset.collate_fn  # type: ignore[assignment]
        else:
            train_collate_fn = train_dataset.collate_fn  # type: ignore[assignment]

        val_collate_fn = val_dataset.collate_fn  # type: ignore[assignment]
    else:
        # Handle both direct dataset and InfiniteUltraFastDataset wrapper
        if isinstance(train_dataset, InfiniteUltraFastDataset):
            train_collate_fn = train_dataset.base_dataset.collate_fn  # type: ignore[assignment]
        else:
            train_collate_fn = train_dataset.collate_fn  # type: ignore[assignment]

        val_collate_fn = val_dataset.collate_fn  # type: ignore[assignment]

    # Create DataLoaders with parallel workers
    train_loader = DataLoader(train_dataset, collate_fn=train_collate_fn, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, collate_fn=val_collate_fn, **dataloader_kwargs)

    return train_loader, val_loader
