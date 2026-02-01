"""
Mock dataloaders and datasets for testing.

These mocks provide realistic data shapes without requiring actual
data files, enabling fast and isolated testing of training components.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import Mock, MagicMock

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class MockDataset(Dataset):
    """
    A mock dataset that generates random token sequences.

    Suitable for testing training loops without actual data files.
    """

    def __init__(
        self,
        size: int = 1000,
        seq_len: int = 64,
        vocab_size: int = 1000,
        include_labels: bool = True,
    ):
        """
        Initialize mock dataset.

        Args:
            size: Number of samples in dataset
            seq_len: Sequence length for each sample
            vocab_size: Vocabulary size for token IDs
            include_labels: Whether to include labels in output
        """
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.include_labels = include_labels

        # Pre-generate data for deterministic tests
        torch.manual_seed(42)
        self._input_ids = torch.randint(0, vocab_size, (size, seq_len))
        self._attention_mask = torch.ones(size, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {
            'input_ids': self._input_ids[idx],
            'attention_mask': self._attention_mask[idx],
        }
        if self.include_labels:
            # Shift input_ids for next-token prediction
            result['labels'] = self._input_ids[idx].clone()
        return result


class MockIterableDataset(IterableDataset):
    """
    A mock iterable dataset for streaming data tests.

    Generates random batches on-the-fly without storing all data.
    """

    def __init__(
        self,
        num_batches: int = 100,
        batch_size: int = 32,
        seq_len: int = 64,
        vocab_size: int = 1000,
    ):
        """
        Initialize mock iterable dataset.

        Args:
            num_batches: Number of batches to generate
            batch_size: Samples per batch
            seq_len: Sequence length
            vocab_size: Vocabulary size
        """
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.num_batches * self.batch_size):
            yield {
                'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
                'labels': torch.randint(0, self.vocab_size, (self.seq_len,)),
            }

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


def create_sample_batch(
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 1000,
    device: torch.device = torch.device('cpu'),
    include_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Create a sample batch for testing.

    Args:
        batch_size: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Target device
        include_labels: Whether to include labels

    Returns:
        Dictionary with input_ids, attention_mask, and optionally labels
    """
    torch.manual_seed(42)
    batch = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
    }
    if include_labels:
        batch['labels'] = batch['input_ids'].clone()
    return batch


def create_mock_dataloader(
    num_batches: int = 10,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 1000,
    device: torch.device = torch.device('cpu'),
) -> DataLoader:
    """
    Create a mock DataLoader for testing.

    Args:
        num_batches: Number of batches to generate
        batch_size: Samples per batch
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Target device (note: DataLoader returns CPU tensors by default)

    Returns:
        DataLoader that yields mock batches
    """
    dataset = MockDataset(
        size=num_batches * batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class MockBatchIterator:
    """
    Mock batch iterator for testing prefetching and async operations.

    Simulates a real dataloader with configurable behavior.
    """

    def __init__(
        self,
        num_batches: int = 10,
        batch_size: int = 4,
        seq_len: int = 64,
        vocab_size: int = 1000,
        device: torch.device = torch.device('cpu'),
        delay_ms: float = 0.0,  # Simulate I/O delay
    ):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.delay_ms = delay_ms
        self._batch_idx = 0

    def __iter__(self) -> 'MockBatchIterator':
        self._batch_idx = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._batch_idx >= self.num_batches:
            raise StopIteration

        if self.delay_ms > 0:
            import time
            time.sleep(self.delay_ms / 1000.0)

        self._batch_idx += 1
        return create_sample_batch(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            device=self.device,
        )

    def __len__(self) -> int:
        return self.num_batches


class MockArrowTable:
    """Mock PyArrow table for testing Arrow I/O."""

    def __init__(
        self,
        num_rows: int = 1000,
        seq_len: int = 64,
        vocab_size: int = 1000,
    ):
        import numpy as np

        self.num_rows = num_rows
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate mock data
        self._input_ids = np.random.randint(0, vocab_size, (num_rows, seq_len))
        self._attention_mask = np.ones((num_rows, seq_len), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_rows

    def column(self, name: str):
        """Get a column by name."""
        mock_col = Mock()
        if name == 'input_ids':
            mock_col.to_numpy = Mock(return_value=self._input_ids)
        elif name == 'attention_mask':
            mock_col.to_numpy = Mock(return_value=self._attention_mask)
        else:
            mock_col.to_numpy = Mock(return_value=self._input_ids)  # Default
        return mock_col

    @property
    def column_names(self) -> List[str]:
        return ['input_ids', 'attention_mask']

    def slice(self, offset: int, length: int) -> 'MockArrowTable':
        """Return a slice of the table."""
        sliced = MockArrowTable(length, self.seq_len, self.vocab_size)
        sliced._input_ids = self._input_ids[offset:offset + length]
        sliced._attention_mask = self._attention_mask[offset:offset + length]
        return sliced

    def to_pandas(self):
        """Convert to pandas DataFrame (mock)."""
        import pandas as pd
        return pd.DataFrame({
            'input_ids': list(self._input_ids),
            'attention_mask': list(self._attention_mask),
        })


def create_mock_arrow_file(
    filepath: str,
    num_rows: int = 1000,
    seq_len: int = 64,
) -> MockArrowTable:
    """
    Create a mock Arrow file representation.

    Note: This doesn't actually create a file, just returns a mock table.

    Args:
        filepath: Path (unused, for API compatibility)
        num_rows: Number of rows
        seq_len: Sequence length

    Returns:
        MockArrowTable that can be used like a real Arrow table
    """
    return MockArrowTable(num_rows, seq_len)


class MockTokenizer:
    """Mock tokenizer for testing text processing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        self.bos_token = '[BOS]'

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Any:
        """Encode text to token IDs."""
        # Simple mock: convert each char to a token ID
        tokens = [ord(c) % self.vocab_size for c in text]

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        if max_length and padding == 'max_length' and len(tokens) < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [
                self.pad_token_id, self.eos_token_id, self.bos_token_id
            ]]

        # Simple mock: convert token IDs back to chars
        return ''.join(chr(t % 128) for t in token_ids)

    def __call__(
        self,
        text: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Tokenize text."""
        tokens = self.encode(text, **kwargs)
        if isinstance(tokens, torch.Tensor):
            attention_mask = (tokens != self.pad_token_id).long()
            return {'input_ids': tokens, 'attention_mask': attention_mask}
        attention_mask = [1 if t != self.pad_token_id else 0 for t in tokens]
        return {'input_ids': tokens, 'attention_mask': attention_mask}
