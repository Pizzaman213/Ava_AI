"""
Indexed Arrow Dataset for true random-access data loading.

This module provides a map-style dataset that pre-builds a global index
over all Arrow/Parquet files, enabling true random shuffling at epoch start.

Key advantages over IterableDataset:
- True random shuffling (not buffer-based)
- Workers read only their assigned samples (no wasted I/O)
- Sample-level train/val split (not file-based)
- Pre-computed lengths for efficient binned sampling
- ~5% padding waste with binning (vs 50-80% with fixed padding)

Usage:
    from ava.data.indexed import create_indexed_dataloaders

    train_loader, val_loader = create_indexed_dataloaders(
        data_dir='/path/to/data',
        batch_size=32,
        max_length=2048,
    )
"""

import logging
import os
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, Subset

# Use centralized Arrow I/O utilities
from .arrow_io import ArrowTableCache, read_arrow_or_parquet
from .base_dataset import discover_data_files
from .validation import validate_arrow_data_format

logger = logging.getLogger(__name__)

# Backward compatibility alias
ArrowTableLRUCache = ArrowTableCache


# =============================================================================
# Indexed Arrow Dataset
# =============================================================================


class IndexedArrowDataset(Dataset):
    """
    Map-style dataset providing true random access to Arrow/Parquet files.

    Pre-builds a global index at initialization using NumPy arrays for memory efficiency.
    This enables:
    - True random shuffling via PyTorch Sampler
    - Efficient worker distribution (each worker reads only its indices)
    - Pre-computed lengths for efficient binned sampling

    Memory Optimization: Uses NumPy int32 arrays instead of Python list of tuples,
    reducing memory by 50-70% for large datasets (10M samples: ~200MB -> ~80MB).

    Args:
        data_files: List of Arrow/Parquet file paths
        max_length: Maximum sequence length (for truncation)
        cache_size: Number of Arrow tables to cache in LRU cache
        pad_token_id: Padding token ID (for length computation)
        compute_lengths: Whether to pre-compute sequence lengths (enables binning)
        index_workers: Number of parallel workers for indexing (default: CPU count)
        numpy_cache_size: Max files to keep in numpy cache (LRU eviction). Default 10.
        numpy_cache_max_mb: Maximum memory for numpy cache in MB. Default 512MB.
    """

    def __init__(
        self,
        data_files: List[Path],
        max_length: int = 2048,
        cache_size: Optional[int] = None,  # None = adaptive based on RAM (Phase 2 optimization)
        pad_token_id: int = 0,
        compute_lengths: bool = True,
        index_workers: Optional[int] = None,
        numpy_cache_size: int = 10,  # MEMORY FIX: Reduced from 20 to limit per-worker memory
        num_dataloader_workers: int = 4,  # For adaptive cache sizing
        numpy_cache_max_mb: int = 512,  # MEMORY FIX: Bounded cache by memory, not just file count
    ):
        self.data_files = list(data_files)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.compute_lengths = compute_lengths
        self._num_dataloader_workers = num_dataloader_workers

        # Will be initialized per-worker
        self._table_cache: Optional[ArrowTableLRUCache] = None
        self._cache_size = cache_size

        # Numpy column cache with LRU eviction: file_path -> {column_name: numpy_array}
        # PERF: Pre-converts Arrow columns to numpy once per file, avoiding
        # slow .as_py() calls on every __getitem__ (100-1000x speedup)
        # MEMORY FIX: Limited by both file count AND total memory to prevent unbounded growth
        self._numpy_cache: OrderedDict[str, Tuple[Dict[str, np.ndarray], int]] = OrderedDict()
        self._numpy_cache_maxsize = numpy_cache_size
        self._numpy_cache_max_bytes = numpy_cache_max_mb * 1024 * 1024
        self._numpy_cache_bytes = 0  # Track current cache memory usage

        # Determine number of indexing workers - MEMORY FIX: reduced default
        # High parallelism during indexing can cause memory spikes
        if index_workers is None:
            # Use fewer workers to reduce memory pressure during startup
            index_workers = min(os.cpu_count() or 4, 8)  # Reduced from 16 to 8

        # Build global index with parallel workers
        # OPTIMIZATION: Uses NumPy int32 arrays instead of List[Tuple[int, int]]
        # Memory reduction: ~200MB -> ~80MB for 10M samples
        self._file_indices, self._row_indices, self._lengths = self._build_index_parallel(index_workers)

        # VALIDATION: Check data format at init to catch issues early
        self._validate_data_format()

    def _build_index_parallel(
        self, num_workers: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Build global index using parallel workers for faster startup.

        OPTIMIZATION: Returns NumPy int32 arrays instead of List[Tuple[int, int]].
        Memory reduction: ~200MB -> ~80MB for 10M samples (50-70% savings).
        Faster indexing: NumPy array access is O(1) vs list lookup overhead.
        """
        logger.info(f"Building global index for {len(self.data_files)} files using {num_workers} workers...")
        start_time = time.time()

        # Results will be collected here: {file_idx: (num_rows, lengths)}
        results: Dict[int, Tuple[int, List[int]]] = {}
        failed_files = 0

        def index_single_file(file_idx: int, file_path: Path) -> Tuple[int, int, List[int]]:
            """Index a single file and return (file_idx, num_rows, lengths).

            PERFORMANCE FIX: Use vectorized operations instead of row-by-row iteration.
            Old code used .as_py() on every row which is 100-1000x slower than vectorized ops.

            MEMORY FIX: Returns num_rows count instead of list of tuples.
            The actual index arrays are built after collecting all results.
            """
            lengths = []

            try:
                table = self._load_table_for_indexing(file_path)
                num_rows = len(table)
                schema_names = table.schema.names

                if self.compute_lengths:
                    # PERFORMANCE FIX: Vectorized length computation
                    # Instead of row-by-row .as_py() calls, use vectorized numpy operations
                    try:
                        if 'attention_mask' in schema_names:
                            # Convert entire column to numpy array at once
                            mask_col = table.column('attention_mask').to_numpy(zero_copy_only=False)
                            # Sum along sequence dimension (axis 1 if 2D, else count non-zero)
                            if mask_col.ndim == 2:
                                lengths = mask_col.sum(axis=1).tolist()
                            elif mask_col.ndim == 1:
                                # FIX: For 1D array of variable-length sequences (object array),
                                # iterate and sum each element. Use np.sum for proper array summation.
                                lengths = []
                                for m in mask_col:
                                    if hasattr(m, '__iter__') and not isinstance(m, (int, float)):
                                        # It's a sequence - sum its values
                                        lengths.append(int(np.sum(m)))
                                    else:
                                        # Scalar or unknown - use max_length
                                        lengths.append(self.max_length)
                            else:
                                lengths = [self.max_length] * num_rows
                        elif 'input_ids' in schema_names:
                            # Convert column to numpy and get lengths
                            ids_col = table.column('input_ids').to_numpy(zero_copy_only=False)
                            if ids_col.ndim == 2:
                                # Fixed-size 2D array - use shape
                                lengths = [ids_col.shape[1]] * len(ids_col)
                            else:
                                # Variable-length lists
                                lengths = [len(ids) if hasattr(ids, '__len__') else self.max_length for ids in ids_col]
                        elif 'token_ids' in schema_names:
                            # Convert column to numpy and get lengths
                            ids_col = table.column('token_ids').to_numpy(zero_copy_only=False)
                            if ids_col.ndim == 2:
                                lengths = [ids_col.shape[1]] * len(ids_col)
                            else:
                                lengths = [len(ids) if hasattr(ids, '__len__') else self.max_length for ids in ids_col]
                        else:
                            # No length column found - use max_length
                            lengths = [self.max_length] * num_rows

                        # Cap at max_length
                        lengths = [min(length, self.max_length) for length in lengths]

                    except Exception as e:
                        # FAIL FAST: Don't silently use conservative estimates
                        raise ValueError(
                            f"Vectorized length computation failed for {file_path.name}: {e}\n"
                            f"This indicates variable-length data that will cause slow loading.\n"
                            f"\n"
                            f"Recovery steps:\n"
                            f"  1. Re-tokenize data with fixed sequence length\n"
                            f"  2. Set data.use_indexed_loader: false to use streaming loader\n"
                            f"  3. Disable length computation (but this reduces batching efficiency)"
                        )

                return (file_idx, num_rows, lengths)

            except Exception as e:
                logger.warning(f"Failed to index {file_path.name}: {e}")
                return (file_idx, 0, [])

        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all indexing jobs
            futures = {
                executor.submit(index_single_file, idx, path): idx
                for idx, path in enumerate(self.data_files)
            }

            # Collect results with progress updates
            completed = 0
            total_samples = 0
            # TIMEOUT FIX: Add timeout to prevent indefinite hanging
            # Each file should complete within reasonable time (60s per file)
            timeout_per_file = 60.0
            for future in as_completed(futures, timeout=timeout_per_file * len(self.data_files)):
                try:
                    file_idx, num_rows, lengths = future.result(timeout=timeout_per_file)
                    if num_rows > 0:
                        results[file_idx] = (num_rows, lengths)
                        total_samples += num_rows
                        # Log each file completion for debugging hanging issues
                        logger.debug(f"  Indexed file {self.data_files[file_idx].name}: {num_rows} samples")
                    else:
                        failed_files += 1
                        logger.warning(f"  File {self.data_files[file_idx].name} returned no samples")
                except Exception as e:
                    failed_files += 1
                    logger.error(f"  Indexing timeout/error for file {futures[future]}: {e}")

                completed += 1
                # Log progress more frequently for better visibility (every 10 files or less)
                log_freq = max(1, min(10, len(self.data_files) // 10))
                if completed % log_freq == 0 or completed == len(self.data_files):
                    elapsed = time.time() - start_time
                    rate = total_samples / elapsed if elapsed > 0 else 0
                    logger.info(f"  Indexed {completed}/{len(self.data_files)} files, "
                               f"{total_samples:,} samples ({rate:,.0f} samples/sec)")

        # OPTIMIZATION: Build NumPy arrays instead of list of tuples
        # Memory reduction: ~200MB -> ~80MB for 10M samples (50-70% savings)

        # Calculate total samples to pre-allocate arrays
        total_samples = sum(num_rows for num_rows, _ in results.values())

        # Pre-allocate NumPy arrays (int32 is sufficient for most datasets)
        # Using int32 instead of int64 saves 50% memory
        file_indices = np.empty(total_samples, dtype=np.int32)
        row_indices = np.empty(total_samples, dtype=np.int32)
        final_lengths = np.empty(total_samples, dtype=np.int32) if self.compute_lengths else None

        # Fill arrays in file order
        offset = 0
        for file_idx in range(len(self.data_files)):
            if file_idx in results:
                num_rows, lengths = results[file_idx]
                end_offset = offset + num_rows

                # Fill file indices (constant for all rows in this file)
                file_indices[offset:end_offset] = file_idx

                # Fill row indices (0 to num_rows-1)
                row_indices[offset:end_offset] = np.arange(num_rows, dtype=np.int32)

                # Fill lengths if computed
                if final_lengths is not None and lengths:
                    final_lengths[offset:end_offset] = np.array(lengths, dtype=np.int32)

                offset = end_offset

        elapsed = time.time() - start_time
        logger.info(f"Indexed {total_samples:,} samples in {elapsed:.1f}s "
                   f"({total_samples/elapsed:,.0f} samples/sec)")

        if failed_files > 0:
            logger.warning(f"  {failed_files} files failed to index")

        if final_lengths is not None:
            avg_len = float(np.mean(final_lengths)) if len(final_lengths) > 0 else 0
            logger.info(f"Average sequence length: {avg_len:.0f}")

        return file_indices, row_indices, final_lengths

    def _validate_data_format(self) -> None:
        """
        Validate data format at init to ensure zero-copy path works.

        Uses shared validation utility to check the first file.
        """
        if not self.data_files:
            return

        try:
            first_file = self.data_files[0]
            table = self._load_table_for_indexing(first_file)
            validate_arrow_data_format(
                table=table,
                file_name=first_file.name,
                input_columns=['input_ids', 'token_ids', 'text'],
            )
        except Exception as e:
            # Don't fail on validation errors - just log and continue
            logger.debug(f"Data format validation skipped: {e}")

    def _load_table_for_indexing(self, file_path: Path) -> pa.Table:
        """Load table for indexing (temporary, not cached)."""
        return read_arrow_or_parquet(file_path)

    def _get_row_length(self, table: pa.Table, row_idx: int) -> int:
        """Get sequence length for a row (non-padding tokens)."""
        schema_names = table.schema.names

        try:
            if 'attention_mask' in schema_names:
                mask = table.column('attention_mask')[row_idx].as_py()
                return sum(mask) if mask else 0
            elif 'input_ids' in schema_names:
                ids = table.column('input_ids')[row_idx].as_py()
                return len(ids) if ids else 0
            elif 'token_ids' in schema_names:
                ids = table.column('token_ids')[row_idx].as_py()
                return len(ids) if ids else 0
        except Exception as e:
            logger.debug(f"Row length computation failed for row {row_idx}: {e}")
        return self.max_length  # Conservative fallback

    def _ensure_cache(self):
        """Ensure table cache is initialized (per-worker)."""
        if self._table_cache is None:
            self._table_cache = ArrowTableCache(max_size=self._cache_size)

    def _get_numpy_columns(self, file_path: str, table: pa.Table) -> Dict[str, np.ndarray]:
        """
        Get numpy arrays for table columns, caching with memory-bounded LRU eviction.

        PERF: Converting Arrow columns to numpy once per file avoids the expensive
        .as_py() call on every __getitem__. Direct numpy indexing is 100-1000x faster.

        MEMORY: LRU eviction with memory tracking prevents unbounded cache growth.
        Cache is bounded by both file count AND total memory usage.
        With 100+ files and 8 workers, an unbounded cache could consume 160GB+ RAM.
        """
        if file_path in self._numpy_cache:
            # Move to end (most recently used) for LRU
            self._numpy_cache.move_to_end(file_path)
            cache_entry, _ = self._numpy_cache[file_path]
            return cache_entry

        # Cache miss - load and cache
        schema_names = table.schema.names
        token_col = 'input_ids' if 'input_ids' in schema_names else 'token_ids'

        # Convert columns to numpy arrays (one-time cost per file)
        cache: Dict[str, np.ndarray] = {}
        entry_bytes = 0

        # Input IDs - always required
        col = table.column(token_col)
        cache['input_ids'] = self._column_to_numpy(col)
        entry_bytes += cache['input_ids'].nbytes

        # Attention mask - optional
        if 'attention_mask' in schema_names:
            col = table.column('attention_mask')
            cache['attention_mask'] = self._column_to_numpy(col)
            entry_bytes += cache['attention_mask'].nbytes

        # Labels - optional
        if 'labels' in schema_names:
            col = table.column('labels')
            cache['labels'] = self._column_to_numpy(col)
            entry_bytes += cache['labels'].nbytes

        # MEMORY FIX: Evict until we have room for new entry (both count and bytes)
        # This prevents memory leaks from large files staying in cache
        while (len(self._numpy_cache) >= self._numpy_cache_maxsize or
               self._numpy_cache_bytes + entry_bytes > self._numpy_cache_max_bytes):
            if not self._numpy_cache:
                break  # Cache empty, nothing to evict
            # popitem(last=False) removes oldest (first) entry
            _, (_, evicted_bytes) = self._numpy_cache.popitem(last=False)
            self._numpy_cache_bytes -= evicted_bytes

        self._numpy_cache[file_path] = (cache, entry_bytes)
        self._numpy_cache_bytes += entry_bytes
        return cache

    def _column_to_numpy(self, column: pa.ChunkedArray) -> np.ndarray:
        """
        Convert Arrow column to numpy array, preferring zero-copy when possible.

        FAIL FAST: Raises error if data format requires slow conversion.
        """
        # Try zero-copy first (faster, no memory allocation)
        try:
            return column.to_numpy(zero_copy_only=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
            pass

        # Fallback: convert with copy (still fast for fixed-size data)
        try:
            return column.to_numpy(zero_copy_only=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
            pass

        # FAIL FAST: Don't use slow to_pylist() fallback
        # Variable-length sequences cause 10-100x slowdown
        raise ValueError(
            f"Data format incompatible with fast loading!\n"
            f"Column type: {column.type}\n"
            f"Expected: fixed_size_list or primitive array\n"
            f"\n"
            f"Recovery steps:\n"
            f"  1. Re-tokenize data with fixed sequence length (pad during tokenization)\n"
            f"  2. Use pyarrow fixed_size_list type instead of list type\n"
            f"  3. Or switch to pretokenized loader: data.use_indexed_loader: false"
        )

    def __len__(self) -> int:
        return len(self._file_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample by global index."""
        self._ensure_cache()

        # OPTIMIZATION: NumPy array indexing is O(1) and faster than list access
        file_idx = int(self._file_indices[idx])
        row_idx = int(self._row_indices[idx])
        file_path = self.data_files[file_idx]
        table = self._table_cache.get(file_path)

        # PERF: Use cached numpy columns instead of .as_py() per row
        # This is 100-1000x faster for repeated access
        cols = self._get_numpy_columns(str(file_path), table)

        # Extract row data via direct numpy indexing
        # OPTIMIZATION: Use asarray first, only copy if needed for ownership
        # This saves 3-5% on data loading by avoiding unnecessary copies
        row_data = cols['input_ids'][row_idx]
        input_ids = np.asarray(row_data, dtype=np.int64)
        # Copy only if: 1) not writeable, or 2) doesn't own data (shares with cache)
        if not input_ids.flags['OWNDATA'] or not input_ids.flags['WRITEABLE']:
            input_ids = input_ids.copy()

        if 'attention_mask' in cols:
            row_data = cols['attention_mask'][row_idx]
            attention_mask = np.asarray(row_data, dtype=np.int64)
            if not attention_mask.flags['OWNDATA'] or not attention_mask.flags['WRITEABLE']:
                attention_mask = attention_mask.copy()
        else:
            attention_mask = np.ones(len(input_ids), dtype=np.int64)

        if 'labels' in cols:
            row_data = cols['labels'][row_idx]
            labels = np.asarray(row_data, dtype=np.int64)
            if not labels.flags['OWNDATA'] or not labels.flags['WRITEABLE']:
                labels = labels.copy()
        else:
            labels = input_ids.copy()

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        # Return with length for collator
        return {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'labels': torch.from_numpy(labels),
            'length': len(input_ids),  # For dynamic padding
        }

    def get_lengths(self) -> Optional[List[int]]:
        """Get pre-computed lengths for binned sampling.

        Note: Returns a list for backward compatibility with LengthBinnedSampler.
        The internal storage is NumPy int32 for memory efficiency.
        """
        if self._lengths is None:
            return None
        # Convert to list for compatibility (one-time conversion, not per-access)
        return self._lengths.tolist()

    def clear_cache(self):
        """Clear numpy cache to free memory without closing table cache.

        Call this periodically during long training runs to prevent
        memory accumulation from persistent DataLoader workers.
        """
        if self._numpy_cache:
            self._numpy_cache.clear()
            self._numpy_cache_bytes = 0
            logger.debug(f"Cleared numpy cache for IndexedArrowDataset")

    def cleanup(self):
        """Clean up resources."""
        if self._table_cache is not None:
            self._table_cache.close()
            self._table_cache = None
        # Clear numpy cache to free memory
        self._numpy_cache.clear()
        self._numpy_cache_bytes = 0

    def __getstate__(self):
        """Custom pickle support - exclude caches."""
        state = self.__dict__.copy()
        state['_table_cache'] = None
        state['_numpy_cache'] = OrderedDict()  # Don't pickle numpy cache, use fresh OrderedDict
        state['_numpy_cache_bytes'] = 0  # Reset cache byte counter
        return state

    def __setstate__(self, state):
        """Custom unpickle support."""
        self.__dict__.update(state)
        # Ensure numpy cache is OrderedDict after unpickle
        if not isinstance(self._numpy_cache, OrderedDict):
            self._numpy_cache = OrderedDict()
        # Ensure cache byte counter exists
        if not hasattr(self, '_numpy_cache_bytes'):
            self._numpy_cache_bytes = 0


# =============================================================================
# Length-Binned Sampler
# =============================================================================


class LengthBinnedSampler(Sampler[int]):
    """
    Sampler that groups samples by length into bins for efficient batching.

    Key features:
    - Reduces padding waste by grouping similar-length sequences
    - Shuffles within bins AND shuffles bin order for diversity
    - Supports set_epoch() for reproducible shuffling each epoch
    - Compatible with DistributedSampler pattern

    Args:
        lengths: Pre-computed sequence lengths for all samples
        batch_size: Number of samples per batch
        num_bins: Number of length bins
        drop_last: Whether to drop incomplete batches
        seed: Random seed for shuffling (None = non-deterministic)
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_bins: int = 8,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Compute bin boundaries using percentiles
        if lengths:
            length_array = np.array(lengths)
            percentiles = np.linspace(0, 100, num_bins + 1)
            self.bin_boundaries = list(np.percentile(length_array, percentiles))
        else:
            self.bin_boundaries = [0] * (num_bins + 1)

        # Assign each sample to a bin
        self.bin_indices: List[List[int]] = [[] for _ in range(num_bins)]
        for idx, length in enumerate(lengths):
            bin_idx = self._get_bin(length)
            self.bin_indices[bin_idx].append(idx)

        # Log bin distribution
        non_empty_bins = sum(1 for b in self.bin_indices if b)
        logger.info(f"LengthBinnedSampler: {non_empty_bins}/{num_bins} bins active, batch_size={batch_size}")
        for i, indices in enumerate(self.bin_indices):
            if indices:
                min_len = min(self.lengths[j] for j in indices)
                max_len = max(self.lengths[j] for j in indices)
                logger.debug(f"  Bin {i}: {len(indices)} samples, lengths [{min_len}, {max_len}]")

    def _get_bin(self, length: int) -> int:
        """Get bin index for a given length."""
        for i, boundary in enumerate(self.bin_boundaries[1:], start=0):
            if length <= boundary:
                return min(i, self.num_bins - 1)
        return self.num_bins - 1

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """
        Yield indices with length-aware shuffling.

        Strategy:
        1. Shuffle samples within each bin (using numpy for large bins)
        2. Shuffle the order of bins
        3. Yield batch-sized chunks from each bin in shuffled order

        OPTIMIZATION: Uses numpy for shuffling large arrays (2-5x faster for >10K samples).
        """
        import numpy as np

        # Compute seed for this epoch
        if self.seed is not None:
            epoch_seed = self.seed + self.epoch
        else:
            epoch_seed = int(time.time() * 1000000) % (2**31)

        # Use numpy RNG for efficient large array shuffling
        np_rng = np.random.default_rng(epoch_seed)

        # Shuffle within each bin (numpy for large bins, Python for small)
        shuffled_bins = []
        for bin_indices in self.bin_indices:
            if bin_indices:
                # Numpy shuffle is significantly faster for large arrays (>1000 elements)
                if len(bin_indices) > 1000:
                    arr = np.array(bin_indices, dtype=np.int64)
                    np_rng.shuffle(arr)
                    shuffled_bins.append(arr)
                else:
                    # Small bins: Python shuffle is fine
                    shuffled = list(bin_indices)
                    # Use numpy integers for Python random seed
                    py_rng = random.Random(int(np_rng.integers(2**31)))
                    py_rng.shuffle(shuffled)
                    shuffled_bins.append(shuffled)

        # Shuffle bin order
        bin_order = list(range(len(shuffled_bins)))
        np_rng.shuffle(bin_order)
        shuffled_bins = [shuffled_bins[i] for i in bin_order]

        # Yield indices, interleaving from different bins
        all_indices = []
        for bin_data in shuffled_bins:
            # Convert numpy array to list if needed for slicing
            bin_indices = bin_data.tolist() if isinstance(bin_data, np.ndarray) else bin_data

            # Process in batches to maintain some locality
            for i in range(0, len(bin_indices), self.batch_size):
                batch = bin_indices[i:i + self.batch_size]

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                # Shuffle within batch for more diversity (small, so Python shuffle is fine)
                random.shuffle(batch)
                all_indices.extend(batch)

        yield from all_indices

    def __len__(self) -> int:
        if self.drop_last:
            total = 0
            for b in self.bin_indices:
                total += (len(b) // self.batch_size) * self.batch_size
            return total
        return len(self.lengths)


# =============================================================================
# Dynamic Padding Collator (now in collators.py)
# =============================================================================

# Import from unified collators module for backward compatibility
from .collators import DynamicPaddingCollator


# =============================================================================
# Index-Mapped Dataset Wrapper
# =============================================================================


class _IndexMappedDataset(Dataset):
    """Wrapper that maps indices to a subset of the base dataset."""

    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


# =============================================================================
# Factory Function
# =============================================================================


def create_indexed_dataloaders(
    data_dir: str,
    batch_size: int,
    max_length: int,
    val_split_ratio: float = 0.1,
    num_workers: int = 4,
    cache_size: Optional[int] = None,  # None = adaptive based on RAM (Phase 2 optimization)
    pad_token_id: int = 0,
    num_bins: int = 8,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    seed: Optional[int] = None,
    max_files: Optional[int] = None,
    index_workers: Optional[int] = None,
    timeout: float = 300.0,  # Worker timeout in seconds (0 disables, masks genuine hangs)
    numpy_cache_size: int = 10,  # Per-worker numpy cache size (MEMORY: reduce if hitting swap)
    val_num_workers: Optional[int] = None,  # Validation workers (None = min(2, num_workers) to reduce contention)
    val_timeout: float = 0.0,  # Validation timeout (0 = disabled to prevent spurious timeouts)
    val_persistent_workers: Optional[bool] = None,  # Validation persistent workers (None = False)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with indexed random access.

    This is the single entry point for the new indexed data loading system.

    Args:
        data_dir: Directory containing Arrow/Parquet files
        batch_size: Samples per batch
        max_length: Maximum sequence length
        val_split_ratio: Fraction of samples for validation
        num_workers: DataLoader workers
        cache_size: Arrow table cache size per worker
        pad_token_id: Padding token ID
        num_bins: Number of length bins for sampling
        prefetch_factor: DataLoader prefetch factor
        persistent_workers: Keep workers alive between epochs
        seed: Random seed (None = non-deterministic)
        max_files: Maximum files to load (None = all)
        index_workers: Parallel workers for indexing (None = CPU count, capped at 16)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Use centralized file discovery
    valid_files = discover_data_files(
        data_dir,
        extensions=['.arrow', '.parquet'],
        max_files=max_files,
    )

    # Filter by minimum size
    MIN_FILE_SIZE = 10 * 1024  # 10KB
    valid_files = [f for f in valid_files if f.stat().st_size >= MIN_FILE_SIZE]

    if not valid_files:
        raise ValueError(f"No valid Arrow/Parquet files found in {data_dir}")

    logger.info(f"Found {len(valid_files)} data files")

    # Create single dataset with all files (parallel indexing)
    dataset = IndexedArrowDataset(
        data_files=valid_files,
        max_length=max_length,
        cache_size=cache_size,
        pad_token_id=pad_token_id,
        compute_lengths=True,
        index_workers=index_workers,
        numpy_cache_size=numpy_cache_size,
        num_dataloader_workers=num_workers,  # For adaptive cache sizing
    )

    # Sample-level train/val split
    total_samples = len(dataset)
    indices = list(range(total_samples))

    # Deterministic shuffle for split (always same split)
    split_seed = 42  # Fixed for reproducibility
    random.Random(split_seed).shuffle(indices)

    val_size = int(total_samples * val_split_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    logger.info(f"Split: {len(train_indices):,} train, {len(val_indices):,} val")

    # Get lengths for binned sampling
    all_lengths = dataset.get_lengths()
    if all_lengths:
        train_lengths = [all_lengths[i] for i in train_indices]
    else:
        train_lengths = [max_length] * len(train_indices)

    # Create train sampler with length binning
    train_sampler = LengthBinnedSampler(
        lengths=train_lengths,
        batch_size=batch_size,
        num_bins=num_bins,
        drop_last=True,
        seed=seed,
    )

    # Create collator
    collator = DynamicPaddingCollator(
        pad_token_id=pad_token_id,
        max_length=max_length,
    )

    # Training DataLoader kwargs
    # Note: timeout only applies when num_workers > 0
    # A reasonable timeout (default 300s) helps detect genuine worker hangs
    train_loader_kwargs: Dict[str, Any] = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': collator,
    }

    if num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor
        train_loader_kwargs['persistent_workers'] = persistent_workers
        train_loader_kwargs['timeout'] = timeout  # Only apply timeout with workers

    # Validation DataLoader kwargs (reduced resources to prevent contention)
    # DEADLOCK FIX: Default to 0 workers to prevent tokenizer fork deadlock
    # Validation is typically limited to max_batches (e.g., 20) so worker overhead isn't worth it
    effective_val_workers = val_num_workers if val_num_workers is not None else 0
    effective_val_persistent = val_persistent_workers if val_persistent_workers is not None else False

    val_loader_kwargs: Dict[str, Any] = {
        'num_workers': effective_val_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': collator,
    }

    if effective_val_workers > 0:
        val_loader_kwargs['prefetch_factor'] = prefetch_factor
        val_loader_kwargs['persistent_workers'] = effective_val_persistent
        val_loader_kwargs['timeout'] = val_timeout  # 0 = disabled by default

    # Create train dataset subset
    train_dataset = _IndexMappedDataset(dataset, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **train_loader_kwargs,
    )

    # Create val dataset subset (simple random sampling, no binning)
    val_dataset = _IndexMappedDataset(dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic validation
        **val_loader_kwargs,
    )

    return train_loader, val_loader


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    'IndexedArrowDataset',
    'LengthBinnedSampler',
    'DynamicPaddingCollator',
    'ArrowTableLRUCache',
    'create_indexed_dataloaders',
]
