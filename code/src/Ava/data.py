"""
Data loading utilities for Ava MoE++ training
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq


class SimpleDataset(Dataset):
    """Simple dataset for loading preprocessed data"""

    def __init__(self, data_dir: str, split: str, tokenizer, max_length: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        data_path = Path(data_dir)

        # Try to load from Arrow files
        pattern = f"{split}_*"
        data_files = sorted(data_path.glob(f"{pattern}/*.arrow"))[:10]  # Limit files for memory

        print(f"📚 Found {len(list(data_path.glob(f'{pattern}/*.arrow')))} data files in {data_path}")

        if data_files:
            for i, file_path in enumerate(data_files):
                if max_samples and len(self.examples) >= max_samples:
                    break

                try:
                    table = pa.ipc.open_file(str(file_path)).read_all()
                    df = table.to_pandas()

                    if 'text' in df.columns:
                        texts = df['text'].tolist()
                        self.examples.extend(texts)
                except:
                    pass

        # If no data loaded, create dummy data
        if not self.examples:
            print(f"⚠️ No data found, creating dummy dataset")
            self.examples = [
                "The quick brown fox jumps over the lazy dog." * 5,
                "Machine learning is transforming artificial intelligence." * 5,
            ] * 50

        # Limit samples if specified
        if max_samples:
            self.examples = self.examples[:max_samples]

        print(f"✅ Loaded {len(self.examples)} samples for {split} split")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize
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


def create_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 0,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    # Create datasets
    train_dataset = SimpleDataset(
        data_dir=data_dir,
        split='train',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples
    )

    val_dataset = SimpleDataset(
        data_dir=data_dir,
        split='val',
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples // 10 if max_samples else None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader