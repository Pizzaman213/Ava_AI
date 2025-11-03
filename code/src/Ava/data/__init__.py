"""
Data Loading and Processing Module

Consolidated data functionality including:
- Streaming dataset loaders
- Multi-column dataset handling
- Distributed sampling
- Bucketing and batching strategies
"""

# Note: The following files have been moved to _archived/data/:
# - data_profiler.py (moved from parent directory)
# - deduplication.py
# - optimized_dataloader.py
# These are data preparation/analysis tools not used in the core training loop

from .dataloader import (
    StreamingDataset,
    InfiniteStreamingDataset,
    DistributedStreamingDataset,
    LengthBasedBucketing,
    FileReader,
    create_streaming_dataloaders,
)

from .multi_column_data import (
    MultiColumnDataset,
    StreamingMultiColumnDataset,
    AdvancedDistributedSampler,
)

__all__ = [
    "StreamingDataset",
    "InfiniteStreamingDataset",
    "DistributedStreamingDataset",
    "LengthBasedBucketing",
    "FileReader",
    "create_streaming_dataloaders",
    "MultiColumnDataset",
    "StreamingMultiColumnDataset",
    "AdvancedDistributedSampler",
]
