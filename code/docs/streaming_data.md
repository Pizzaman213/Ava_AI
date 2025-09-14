# Data Streaming Guide

## Overview

Data streaming is a memory-efficient approach to handling large datasets that prevents loading entire datasets into RAM. This is crucial when working with datasets that are tens or hundreds of gigabytes in size.

## Why Use Data Streaming?

### Problem: Memory Overflow
Without streaming, the `TensorDataset` class loads ALL data shards into memory:
- 6 shards × 12GB each = 72GB RAM required
- This causes system crashes and makes training impossible on most machines

### Solution: Stream Data On-Demand
With streaming enabled:
- Only loads one batch or shard at a time
- Reduces memory usage from 70GB to ~2-4GB
- Enables training on datasets of any size

## How to Enable Streaming

### Method 1: Configuration File (Recommended)
```yaml
data:
  dataset_type: tensor
  train_path: ./data/tensor/train
  val_path: ./data/tensor/validation
  dataset_args:
    streaming: true  # Enable streaming
```

### Method 2: Command Line Override
```bash
python scripts/train.py --config configs/mps/small.yaml --streaming
```

### Method 3: Direct API Usage
```python
from src.data.datasets import get_dataset

# Streaming tensor dataset
dataset = get_dataset(
    data_path="./data/tensor/train",
    dataset_type="tensor",
    streaming=True  # Enable streaming
)
```

## Streaming Implementation Details

### StreamingTensorDataset
- Loads metadata once to know total examples and shards
- Iterates through shards sequentially
- Loads only one shard at a time into memory
- Automatically handles multi-worker data loading

### Memory-Efficient TensorDataset
- When `streaming=True`, loads shards on-demand
- Caches only the current shard being accessed
- Automatically loads next shard when needed

## Performance Considerations

### Batch Size vs Memory
- Larger batch sizes = more memory per iteration
- Recommended: Start with batch_size=8 for safety
- Increase gradually based on available memory

### Disk I/O
- Streaming requires frequent disk reads
- Use SSD storage for best performance
- Consider prefetch_factor in dataloader settings

### Training Speed
- Minimal impact on training speed with SSD
- ~5-10% slower than in-memory on HDD
- Negligible difference with NVMe SSDs

## Configuration Examples

### Ultra Low Memory (2-4GB)
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
data:
  dataset_args:
    streaming: true
    prefetch_factor: 2
```

### Standard Memory (8-16GB)
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  gradient_checkpointing: false
data:
  dataset_args:
    streaming: true
    prefetch_factor: 4
```

## Monitoring Memory Usage

### During Training
The trainer now shows:
- Current batch / total batches
- Percentage complete
- Estimated time remaining
- Memory usage warnings if exceeding limits

### Memory Test Script
```bash
python scripts/test_memory.py
```

This will show:
- Model memory requirements
- Dataset shard sizes
- Recommended settings

## Troubleshooting

### Still Running Out of Memory?
1. Reduce batch_size further
2. Enable gradient_checkpointing
3. Use smaller sequence lengths
4. Enable CPU offloading (if supported)

### Slow Training?
1. Use SSD/NVMe storage
2. Reduce num_workers if thrashing
3. Increase prefetch_factor
4. Check disk I/O bottlenecks

### Can't Find Shards?
1. Verify data path exists
2. Check metadata.json is present
3. Ensure shard files match metadata

## Best Practices

1. **Always use streaming for large datasets** (>10GB)
2. **Start with conservative batch sizes** then increase
3. **Monitor first epoch closely** for memory issues
4. **Use gradient accumulation** to simulate larger batches
5. **Enable checkpointing** for very long training runs