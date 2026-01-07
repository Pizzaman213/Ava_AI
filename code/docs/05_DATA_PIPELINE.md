# Data Pipeline Guide

Complete guide to data loading, preprocessing, and efficient data handling in Ava.

## Overview

Ava supports multiple data loading strategies:

| Strategy | Memory | Throughput | Use Case |
|----------|--------|------------|----------|
| Streaming | Low | High | Large datasets |
| Pre-tokenized | Medium | Very High | Production |
| Indexed | High | Highest | Random access needed |

## Data Formats

### Supported Formats

- **Arrow/Parquet**: Recommended for large datasets
- **JSONL**: Text data with metadata
- **HuggingFace Datasets**: Direct HF integration

### Data Structure

Pre-tokenized Arrow files should contain:

```python
{
    'input_ids': List[int],      # Token IDs
    'attention_mask': List[int],  # 1 for tokens, 0 for padding
    'labels': List[int],          # Target token IDs (optional)
}
```

## Data Download

### Using Unified Download

```bash
# Download all supported datasets
python code/scripts/1_data_download/unified_download.py

# Limit partitions for testing
python code/scripts/1_data_download/unified_download.py --max-partitions 10
```

### Pre-tokenization

For maximum training speed, pre-tokenize your data:

```bash
python code/scripts/1_data_download/build_pretokenized_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --tokenizer code/data/Ava_Ai/tokenizer \
    --max-length 512
```

## Streaming Dataset

Located in `code/src/ava/data/streaming.py`:

```python
class StreamingDataset:
    """Memory-efficient streaming from Arrow files."""

    def __init__(
        self,
        data_dir: str,
        buffer_size: int = 50000,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
```

### Configuration

```yaml
data:
  streaming: true
  buffer_size: 50000      # Shuffle buffer size
  num_workers: 8          # Parallel workers
  prefetch_factor: 4      # Batches to prefetch
  persistent_workers: true
```

### How It Works

1. Scans `data_dir` for Arrow files
2. Loads files lazily using memory-mapped I/O
3. Fills shuffle buffer from multiple files
4. Yields shuffled batches

## Pre-tokenized Dataset

Located in `code/src/ava/data/pretokenized.py`:

```python
class PreTokenizedDataset:
    """High-performance loading from pre-tokenized Arrow files."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        use_mmap: bool = True,
    ):
        # Memory-map for zero-copy access
        self.table = pa.ipc.open_file(pa.memory_map(data_path, 'r')).read_all()
```

### Benefits

- **Zero-copy loading**: No tokenization overhead
- **Memory mapping**: OS handles caching
- **Consistent sequences**: Reproducible training

## Distributed Dataset

For multi-GPU training:

```python
# code/src/ava/data/distributed.py
class DistributedStreamingDataset:
    """Shards data across distributed workers."""

    def __init__(
        self,
        data_dir: str,
        rank: int,
        world_size: int,
        ...
    ):
        # Assign file shards to each rank
        self.files = self._shard_files(all_files, rank, world_size)
```

## Data Factory

The recommended way to create dataloaders:

```python
from ava.data.factory import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1,
)
```

## Sequence Packing

Combine short sequences to eliminate padding waste:

```yaml
data:
  use_sequence_packing: true
  packing_strategy: 'greedy'   # or 'adaptive'
```

### How Packing Works

```
Before packing:
[A, A, A, PAD, PAD, PAD]  # 50% waste
[B, B, PAD, PAD, PAD, PAD]  # 67% waste

After packing:
[A, A, A, SEP, B, B]        # 0% waste
```

Implementation in `code/src/ava/data/packing.py`:

```python
class SequencePacker:
    def pack_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Greedy bin-packing of sequences."""
        packed = []
        current_bin = []
        current_length = 0

        for seq in sorted(sequences, key=len, reverse=True):
            if current_length + len(seq) + 1 <= self.max_length:
                current_bin.extend(seq + [self.sep_token])
                current_length += len(seq) + 1
            else:
                packed.append(current_bin)
                current_bin = seq + [self.sep_token]
                current_length = len(seq) + 1

        return packed
```

## Length-Based Bucketing

Group similar-length sequences for efficient batching:

```yaml
data:
  enable_bucketing: true
```

Located in `code/src/ava/data/bucketing.py`:

```python
class LengthBasedBucketing:
    """Groups sequences by length to minimize padding."""

    def __init__(self, num_buckets: int = 8):
        self.buckets = [[] for _ in range(num_buckets)]

    def add(self, sequence, length):
        bucket_idx = self._get_bucket(length)
        self.buckets[bucket_idx].append(sequence)
```

## Conversation Data

Turn-aware loading for dialogue:

```yaml
data:
  use_conversation_format: true
  turn_separator: "<|turn|>"
```

Located in `code/src/ava/data/conversation.py`:

```python
class ConversationDataset:
    """Preserves dialogue structure with turn markers."""

    def format_conversation(self, turns: List[Dict]) -> str:
        formatted = []
        for turn in turns:
            role = turn['role']
            content = turn['content']
            formatted.append(f"<|{role}|>{content}")
        return self.turn_separator.join(formatted)
```

## Multi-Column Data

Load datasets with multiple columns:

```yaml
multi_column_data:
  use_multi_column: true
  hf_dataset: "openai/gsm8k"
  column_names: "question,answer"
  column_types: "text,text"
  combine_strategy: 'template'
  column_template: "Question: {question}\nAnswer: {answer}"
```

## Data Randomization

### Deterministic Training

```yaml
data:
  shuffle_seed: 42
  randomization:
    deterministic: true
```

### Maximum Randomness

```yaml
data:
  shuffle_seed: null     # Random seed each run
  buffer_size: 100000    # Large shuffle buffer
  randomization:
    deterministic: false
```

## Performance Optimization

### Parallel Loading

```yaml
data:
  num_workers: 8              # CPU cores for loading
  prefetch_factor: 4          # Batches ahead to load
  persistent_workers: true    # Keep workers between epochs
  dataloader_pin_memory: true # Faster GPU transfer
```

### Worker Configuration

| GPUs | Recommended Workers | Prefetch |
|------|-------------------|----------|
| 1 | 4-8 | 2-4 |
| 4 | 4 per GPU | 2 |
| 8 | 2-4 per GPU | 2 |

### Memory-Mapped Loading

For large datasets:

```yaml
data:
  use_mmap: true              # Memory-map files
  dataloader_drop_last: true  # Avoid partial batches
```

## Validation Split

### Automatic Split

```yaml
data:
  auto_create_validation_split: true
  validation_split_ratio: 0.1
  val_max_samples: 1000       # Limit validation size
```

### Separate Files

```yaml
data:
  train_split: 'train'
  eval_split: 'validation'
```

## Tokenizer Integration

### Custom Tokenizer

```yaml
data:
  tokenizer_name: "code/data/Ava_Ai/tokenizer"
  padding_side: 'right'
  truncation: true
  max_length: 512
```

### Special Tokens

Tokens are configured in model config:

```yaml
model:
  pad_token_id: 0
  eos_token_id: 1
  bos_token_id: 2
```

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DataLoaderManager                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 create_dataloaders()                     │ │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │ │
│  │  │ StreamingDS  │  │ Pretokenized  │  │  ConversationDS │  │ │
│  │  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘  │ │
│  │         │                  │                 │          │ │
│  │         └──────────────────┼─────────────────┘          │ │
│  │                            ▼                            │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │   LengthBasedBucketing  │                │ │
│  │              └────────────┬────────────┘                │ │
│  │                           ▼                             │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │    SequencePacker       │                │ │
│  │              └────────────┬────────────┘                │ │
│  │                           ▼                             │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │      DataLoader         │                │ │
│  │              └─────────────────────────┘                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Data Not Loading

1. Check data directory exists and contains Arrow files
2. Verify file permissions
3. Check tokenizer path

### Out of Memory During Loading

1. Reduce `buffer_size`
2. Reduce `num_workers`
3. Enable streaming mode

### Slow Data Loading

1. Increase `num_workers`
2. Use pre-tokenized data
3. Enable `persistent_workers`
4. Use SSD storage

## Next Steps

- [Configuration Reference](./04_CONFIGURATION.md) - Data config options
- [Training Guide](./03_TRAINING_GUIDE.md) - Training workflow
- [Performance Tuning](./10_PERFORMANCE.md) - Optimization tips
