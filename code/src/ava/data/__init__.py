"""
Data loading and processing utilities for Ava training.

This package provides:
- Streaming datasets for memory-efficient data loading
- Bucketing utilities for efficient batch formation
- Distributed data loading support
- Factory functions for creating dataloaders

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

from .bucketing import (
    AsyncFilePrefetcher,
    DynamicTokenBatcher,
    LengthBasedBucketing,
)
from .distributed import DistributedStreamingDataset
from .factory import create_streaming_dataloaders
from .streaming import (
    FileReader,
    InfiniteStreamingDataset,
    StreamingDataset,
    _worker_init_fn,
    get_worker_context,
    retry_on_error,
)

__all__ = [
    # Factory
    'create_streaming_dataloaders',
    # Streaming
    'StreamingDataset',
    'InfiniteStreamingDataset',
    'FileReader',
    'get_worker_context',
    'retry_on_error',
    '_worker_init_fn',
    # Bucketing
    'DynamicTokenBatcher',
    'LengthBasedBucketing',
    'AsyncFilePrefetcher',
    # Distributed
    'DistributedStreamingDataset',
]
