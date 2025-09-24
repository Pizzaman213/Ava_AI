"""
Streaming data loader for Ava MoE++ training
Efficiently handles large datasets without loading everything into memory
"""

import os
import json
import torch
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
import random
from itertools import cycle, islice

# Distributed training imports
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class StreamingDataset(IterableDataset):
    """Streaming dataset that loads data on-the-fly"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int,
        max_samples: Optional[int] = None,
        buffer_size: int = 1000
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size

        # Find all data files
        self.data_files = self._find_data_files()

        if not self.data_files:
            print(f"⚠️ No data files found for {split} split, will use synthetic data")

    def _find_data_files(self) -> List[Path]:
        """Find all relevant data files"""
        files = []

        # Look for Arrow files
        pattern = f"{self.split}_*"
        arrow_files = sorted(self.data_dir.glob(f"{pattern}/**/*.arrow"))
        files.extend(arrow_files)

        # Look for Parquet files
        parquet_files = sorted(self.data_dir.glob(f"{pattern}/**/*.parquet"))
        files.extend(parquet_files)

        # Also check direct files
        direct_arrow = sorted(self.data_dir.glob(f"{self.split}_*.arrow"))
        direct_parquet = sorted(self.data_dir.glob(f"{self.split}_*.parquet"))
        direct_jsonl = sorted(self.data_dir.glob(f"{self.split}_*.jsonl"))
        files.extend(direct_arrow)
        files.extend(direct_parquet)
        files.extend(direct_jsonl)

        # If no split-specific files found, look for any JSONL files
        if not files:
            all_jsonl = sorted(self.data_dir.glob("*.jsonl"))
            files.extend(all_jsonl)

        print(f"Found {len(files)} data files for {self.split} split")
        return files

    def _read_file(self, file_path: Path) -> Iterator[str]:
        """Read data from a single file"""
        try:
            if file_path.suffix == '.arrow':
                # Read Arrow file
                with pa.memory_map(str(file_path), 'r') as source:
                    batch_reader = pa.ipc.open_file(source)
                    for i in range(batch_reader.num_record_batches):
                        batch = batch_reader.get_batch(i)
                        df = batch.to_pandas()

                        if 'text' in df.columns:
                            for text in df['text']:
                                if text and len(str(text).strip()) > 10:
                                    yield str(text)

            elif file_path.suffix == '.parquet':
                # Read Parquet file in chunks
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=100):
                    df = batch.to_pandas()

                    if 'text' in df.columns:
                        for text in df['text']:
                            if text and len(str(text).strip()) > 10:
                                yield str(text)

            elif file_path.suffix == '.jsonl':
                # Read JSONL file line by line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = data.get('text', '')
                            if text and len(str(text).strip()) > 10:
                                yield str(text)
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    def _generate_synthetic_data(self) -> Iterator[str]:
        """Generate synthetic data for testing"""
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Deep learning models require large amounts of data.",
            "Natural language processing enables human-computer interaction.",
            "Transformer models have revolutionized NLP tasks.",
            "Artificial neural networks mimic biological neurons.",
            "Gradient descent optimizes model parameters.",
            "Backpropagation computes gradients efficiently.",
            "Attention mechanisms improve sequence modeling.",
            "Pre-training creates powerful language representations.",
        ]

        while True:
            # Randomly combine templates
            num_sentences = random.randint(3, 8)
            text = " ".join(random.choices(templates, k=num_sentences))
            yield text * random.randint(2, 5)

    def _stream_examples(self) -> Iterator[str]:
        """Stream examples from all files"""
        if not self.data_files:
            # Use synthetic data if no files found
            yield from self._generate_synthetic_data()
        else:
            # Shuffle files for each epoch
            shuffled_files = list(self.data_files)
            random.shuffle(shuffled_files)

            for file_path in shuffled_files:
                yield from self._read_file(file_path)

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text"""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()  # For causal LM, labels = input_ids
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset"""
        count = 0
        buffer = []

        for text in self._stream_examples():
            if self.max_samples and count >= self.max_samples:
                break

            buffer.append(text)

            # When buffer is full, shuffle and yield
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for buffered_text in buffer:
                    yield self._tokenize_text(buffered_text)
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return
                buffer = []

        # Process remaining buffer
        if buffer:
            random.shuffle(buffer)
            for buffered_text in buffer:
                if self.max_samples and count >= self.max_samples:
                    break
                yield self._tokenize_text(buffered_text)
                count += 1


def create_streaming_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    buffer_size: int = 1000,
    distributed: bool = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create streaming train and validation dataloaders with distributed support"""

    # Auto-detect distributed training
    if distributed is None:
        distributed = DISTRIBUTED_AVAILABLE and (
            'WORLD_SIZE' in os.environ or
            (world_size is not None and world_size > 1)
        )

    if distributed and DISTRIBUTED_AVAILABLE:
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        print(f"🌐 Creating distributed streaming dataloaders (rank {rank}/{world_size})")
    else:
        print(f"Creating streaming dataloaders...")

    # Create streaming datasets
    train_dataset = StreamingDataset(
        data_dir=data_dir,
        split='train',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        buffer_size=buffer_size
    )

    val_dataset = StreamingDataset(
        data_dir=data_dir,
        split='val',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples // 10 if max_samples else 1000,  # Limit validation
        buffer_size=buffer_size // 10
    )

    # Create dataloaders with distributed support
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True  # Important for distributed training
    }

    # For IterableDataset, we don't use DistributedSampler,
    # but we need to handle distributed iteration in the dataset itself
    if distributed and DISTRIBUTED_AVAILABLE:
        # Modify datasets for distributed iteration
        train_dataset = DistributedStreamingDataset(train_dataset, world_size, rank)
        val_dataset = DistributedStreamingDataset(val_dataset, world_size, rank)

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    return train_loader, val_loader


class DistributedStreamingDataset(IterableDataset):
    """Wrapper for distributed streaming dataset"""

    def __init__(self, base_dataset: StreamingDataset, world_size: int, rank: int):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        # Create iterator from base dataset
        base_iter = iter(self.base_dataset)

        # Skip samples to ensure each rank gets different data
        # This is a simple round-robin distribution
        for i, sample in enumerate(base_iter):
            if i % self.world_size == self.rank:
                yield sample


class InfiniteStreamingDataset(IterableDataset):
    """Infinite streaming dataset for continuous training"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int,
        buffer_size: int = 1000
    ):
        self.base_dataset = StreamingDataset(
            data_dir=data_dir,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=None,
            buffer_size=buffer_size
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate infinitely over the dataset"""
        while True:
            # Create a new iterator for each pass
            for item in self.base_dataset:
                yield item