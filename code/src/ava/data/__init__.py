"""
Data loading and processing utilities for Ava training.

This package provides:
- Pretokenized Arrow/Parquet dataloaders (ultra-fast)
- Indexed Arrow dataloaders (map-style with true random shuffling)
- Distributed data loading support
- Centralized Arrow I/O utilities

Usage:
    from ava.data import IndexedArrowDataset, create_indexed_dataloaders

    # Create indexed dataloaders for true random shuffling
    train_loader, val_loader = create_indexed_dataloaders(
        data_dir='/path/to/data',
        batch_size=32,
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
from .distributed import DistributedStreamingDataset, AdvancedDistributedSampler
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
    # Indexed dataloaders
    'create_indexed_dataloaders',
    # Collators (unified)
    'BaseCollator',
    'DynamicPaddingCollator',
    'FixedPaddingCollator',
    'create_collator',
    # Indexed (Map-style with true random shuffling)
    'IndexedArrowDataset',
    'LengthBinnedSampler',
    # Distributed
    'DistributedStreamingDataset',
    'AdvancedDistributedSampler',
]
