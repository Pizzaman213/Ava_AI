"""
Data loading and processing utilities for Ava training.

This package provides:
- Streaming datasets for memory-efficient data loading
- Bucketing utilities for efficient batch formation
- Distributed data loading support
- Factory functions for creating dataloaders
- Centralized Arrow I/O utilities

Usage:
    from ava.data import create_streaming_dataloaders, StreamingDataset

    # Create dataloaders with all optimizations
    train_loader, val_loader = create_streaming_dataloaders(
        tokenizer=tokenizer,
        batch_size=32,
        max_length=2048,
        data_dir='/path/to/data',
    )

    # Or use StreamingDataset directly
    dataset = StreamingDataset(
        data_dir='/path/to/data',
        split='train',
        tokenizer=tokenizer,
        max_length=2048,
    )
"""

# Base dataset utilities
from .base_dataset import (
    discover_data_files,
    get_file_format,
    DatasetFileMixin,
    ShuffleBufferMixin,
    SUPPORTED_EXTENSIONS,
)

# Arrow I/O utilities (centralized)
from .arrow_io import (
    read_arrow_table,
    read_arrow_or_parquet,
    ArrowTableCache,
    ThreadLocalArrowCache,
    # Backward compatibility aliases
    ArrowTableLRUCache,
    ThreadSafeFileCache,
    ThreadLocalFileCache,
)
from .bucketing import (
    AsyncFilePrefetcher,
    DynamicTokenBatcher,
    LengthBasedBucketing,
)
from .distributed import DistributedStreamingDataset, AdvancedDistributedSampler
from .factory import create_streaming_dataloaders, create_dataloaders
# Collators (unified)
from .collators import (
    BaseCollator,
    DynamicPaddingCollator,
    FixedPaddingCollator,
    create_collator,
)
from .indexed import (
    IndexedArrowDataset,
    LengthBinnedSampler,
    create_indexed_dataloaders,
)
from .streaming import (
    FileReader,
    InfiniteStreamingDataset,
    StreamingDataset,
    _worker_init_fn,
    get_worker_context,
    retry_on_error,
)

__all__ = [
    # Base dataset utilities
    'discover_data_files',
    'get_file_format',
    'DatasetFileMixin',
    'ShuffleBufferMixin',
    'SUPPORTED_EXTENSIONS',
    # Arrow I/O (centralized)
    'read_arrow_table',
    'read_arrow_or_parquet',
    'ArrowTableCache',
    'ThreadLocalArrowCache',
    'ArrowTableLRUCache',  # Backward compatibility
    'ThreadSafeFileCache',  # Backward compatibility
    'ThreadLocalFileCache',  # Backward compatibility
    # Factory
    'create_dataloaders',  # Unified factory
    'create_streaming_dataloaders',
    'create_indexed_dataloaders',
    # Streaming
    'StreamingDataset',
    'InfiniteStreamingDataset',
    'FileReader',
    'get_worker_context',
    'retry_on_error',
    '_worker_init_fn',
    # Collators (unified)
    'BaseCollator',
    'DynamicPaddingCollator',
    'FixedPaddingCollator',
    'create_collator',
    # Indexed (Map-style with true random shuffling)
    'IndexedArrowDataset',
    'LengthBinnedSampler',
    # Bucketing
    'DynamicTokenBatcher',
    'LengthBasedBucketing',
    'AsyncFilePrefetcher',
    # Distributed
    'DistributedStreamingDataset',
    'AdvancedDistributedSampler',
]
