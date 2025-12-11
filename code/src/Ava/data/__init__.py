"""
Data Loading and Processing Module

Provides data loaders for:
- Streaming dataset loaders
- Pre-tokenized Arrow loaders (60x faster)
- Multi-column dataset handling
- Turn-aware conversation loaders
- Distributed sampling
- Bucketing and batching strategies

Recommended usage:
    from Ava.data.dataloader import create_streaming_dataloaders
    from Ava.data.pretokenized_loader import create_ultra_fast_dataloaders
"""

# Streaming loaders
from .dataloader import (
    StreamingDataset,
    InfiniteStreamingDataset,
    DistributedStreamingDataset,
    LengthBasedBucketing,
    FileReader,
    create_streaming_dataloaders,
)

# Multi-column loaders
from .multi_column_data import (
    MultiColumnDataset,
    StreamingMultiColumnDataset,
    AdvancedDistributedSampler,
)

__all__ = [
    # Streaming
    "StreamingDataset",
    "InfiniteStreamingDataset",
    "DistributedStreamingDataset",
    "LengthBasedBucketing",
    "FileReader",
    "create_streaming_dataloaders",
    # Multi-column
    "MultiColumnDataset",
    "StreamingMultiColumnDataset",
    "AdvancedDistributedSampler",
]
