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

logger = logging.getLogger(__name__)

# Backward compatibility alias
ArrowTableLRUCache = ArrowTableCache


# =============================================================================
# Indexed Arrow Dataset
# =============================================================================


class IndexedArrowDataset(Dataset):
    """
    Map-style dataset providing true random access to Arrow/Parquet files.

    Pre-builds a global index at initialization: [(file_idx, row_idx), ...]
    This enables:
    - True random shuffling via PyTorch Sampler
    - Efficient worker distribution (each worker reads only its indices)
    - Pre-computed lengths for efficient binned sampling

    Args:
        data_files: List of Arrow/Parquet file paths
        max_length: Maximum sequence length (for truncation)
        cache_size: Number of Arrow tables to cache in LRU cache
        pad_token_id: Padding token ID (for length computation)
        compute_lengths: Whether to pre-compute sequence lengths (enables binning)
        index_workers: Number of parallel workers for indexing (default: CPU count)
    """

    def __init__(
        self,
        data_files: List[Path],
        max_length: int = 2048,
        cache_size: int = 50,
        pad_token_id: int = 0,
        compute_lengths: bool = True,
        index_workers: Optional[int] = None,
    ):
        self.data_files = list(data_files)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.compute_lengths = compute_lengths

        # Will be initialized per-worker
        self._table_cache: Optional[ArrowTableLRUCache] = None
        self._cache_size = cache_size

        # Determine number of indexing workers
        if index_workers is None:
            index_workers = min(os.cpu_count() or 4, 16)  # Cap at 16

        # Build global index with parallel workers
        self._index, self._lengths = self._build_index_parallel(index_workers)

    def _build_index_parallel(
        self, num_workers: int
    ) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
        """Build global index using parallel workers for faster startup."""
        logger.info(f"Building global index for {len(self.data_files)} files using {num_workers} workers...")
        start_time = time.time()

        # Results will be collected here: {file_idx: (indices, lengths)}
        results: Dict[int, Tuple[List[Tuple[int, int]], List[int]]] = {}
        failed_files = 0

        def index_single_file(file_idx: int, file_path: Path) -> Tuple[int, List[Tuple[int, int]], List[int]]:
            """Index a single file and return (file_idx, indices, lengths).

            PERFORMANCE FIX: Use vectorized operations instead of row-by-row iteration.
            Old code used .as_py() on every row which is 100-1000x slower than vectorized ops.
            """
            indices = []
            lengths = []

            try:
                table = self._load_table_for_indexing(file_path)
                num_rows = len(table)
                schema_names = table.schema.names

                # Create indices for all rows at once (much faster than append in loop)
                indices = [(file_idx, row_idx) for row_idx in range(num_rows)]

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
                            else:
                                lengths = [sum(m) if hasattr(m, '__iter__') else self.max_length for m in mask_col]
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
                        # Vectorized approach failed - fall back to conservative estimate
                        logger.debug(f"Vectorized length computation failed for {file_path.name}: {e}")
                        lengths = [self.max_length] * num_rows

                return (file_idx, indices, lengths)

            except Exception as e:
                logger.warning(f"Failed to index {file_path.name}: {e}")
                return (file_idx, [], [])

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
                    file_idx, indices, lengths = future.result(timeout=timeout_per_file)
                    if indices:
                        results[file_idx] = (indices, lengths)
                        total_samples += len(indices)
                        # Log each file completion for debugging hanging issues
                        logger.debug(f"  Indexed file {self.data_files[file_idx].name}: {len(indices)} samples")
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

        # Merge results in file order
        final_index: List[Tuple[int, int]] = []
        final_lengths: List[int] = [] if self.compute_lengths else None

        for file_idx in range(len(self.data_files)):
            if file_idx in results:
                indices, lengths = results[file_idx]
                final_index.extend(indices)
                if final_lengths is not None:
                    final_lengths.extend(lengths)

        elapsed = time.time() - start_time
        logger.info(f"Indexed {len(final_index):,} samples in {elapsed:.1f}s "
                   f"({len(final_index)/elapsed:,.0f} samples/sec)")

        if failed_files > 0:
            logger.warning(f"  {failed_files} files failed to index")

        if final_lengths:
            avg_len = sum(final_lengths) / len(final_lengths) if final_lengths else 0
            logger.info(f"Average sequence length: {avg_len:.0f}")

        return final_index, final_lengths

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

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample by global index."""
        self._ensure_cache()

        file_idx, row_idx = self._index[idx]
        file_path = self.data_files[file_idx]
        table = self._table_cache.get(file_path)

        # Extract row data
        schema_names = table.schema.names
        token_col = 'input_ids' if 'input_ids' in schema_names else 'token_ids'

        input_ids = np.array(table.column(token_col)[row_idx].as_py(), dtype=np.int64)

        if 'attention_mask' in schema_names:
            attention_mask = np.array(table.column('attention_mask')[row_idx].as_py(), dtype=np.int64)
        else:
            attention_mask = np.ones(len(input_ids), dtype=np.int64)

        if 'labels' in schema_names:
            labels = np.array(table.column('labels')[row_idx].as_py(), dtype=np.int64)
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
        """Get pre-computed lengths for binned sampling."""
        return self._lengths

    def cleanup(self):
        """Clean up resources."""
        if self._table_cache is not None:
            self._table_cache.close()
            self._table_cache = None

    def __getstate__(self):
        """Custom pickle support - exclude cache."""
        state = self.__dict__.copy()
        state['_table_cache'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle support."""
        self.__dict__.update(state)


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
        1. Shuffle samples within each bin
        2. Shuffle the order of bins
        3. Yield batch-sized chunks from each bin in shuffled order
        """
        # Compute seed for this epoch
        if self.seed is not None:
            epoch_seed = self.seed + self.epoch
        else:
            epoch_seed = int(time.time() * 1000000) % (2**31)

        rng = random.Random(epoch_seed)

        # Shuffle within each bin
        shuffled_bins = []
        for bin_indices in self.bin_indices:
            if bin_indices:
                shuffled = list(bin_indices)
                rng.shuffle(shuffled)
                shuffled_bins.append(shuffled)

        # Shuffle bin order
        rng.shuffle(shuffled_bins)

        # Yield indices, interleaving from different bins
        all_indices = []
        for bin_indices in shuffled_bins:
            # Process in batches to maintain some locality
            for i in range(0, len(bin_indices), self.batch_size):
                batch = bin_indices[i:i + self.batch_size]

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                # Shuffle within batch for more diversity
                rng.shuffle(batch)
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
    cache_size: int = 50,
    pad_token_id: int = 0,
    num_bins: int = 8,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    seed: Optional[int] = None,
    max_files: Optional[int] = None,
    index_workers: Optional[int] = None,
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

    # DataLoader kwargs
    loader_kwargs: Dict[str, Any] = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': collator,
    }

    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    # Create train dataset subset
    train_dataset = _IndexMappedDataset(dataset, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **loader_kwargs,
    )

    # Create val dataset subset (simple random sampling, no binning)
    val_dataset = _IndexMappedDataset(dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic validation
        **loader_kwargs,
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
