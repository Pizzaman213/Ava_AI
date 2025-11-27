"""
Tests for the DynamicBatchIterator class.

Tests the dynamic batching functionality that adjusts batch sizes based on
simulated GPU memory pressure.
"""

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, Iterator
from unittest.mock import MagicMock, patch


class MockDataset(IterableDataset):
    """Mock dataset that yields numbered samples."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 128):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(self.num_samples):
            yield {
                'input_ids': torch.full((self.seq_length,), i, dtype=torch.long),
                'attention_mask': torch.ones(self.seq_length, dtype=torch.long),
                'labels': torch.full((self.seq_length,), i, dtype=torch.long),
            }


def mock_collate_fn(batch):
    """Simple collate function for testing."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


class TestDynamicBatchIterator:
    """Test suite for DynamicBatchIterator."""

    def test_import(self):
        """Test that the module can be imported."""
        from src.Ava.data.dynamic_batch_iterator import (
            DynamicBatchIterator,
            PassThroughBatchIterator,
            create_dynamic_batch_iterator,
        )
        assert DynamicBatchIterator is not None
        assert PassThroughBatchIterator is not None
        assert create_dynamic_batch_iterator is not None

    def test_pass_through_iterator(self):
        """Test PassThroughBatchIterator doesn't modify batches."""
        from src.Ava.data.dynamic_batch_iterator import PassThroughBatchIterator

        dataset = MockDataset(num_samples=100)
        base_loader = DataLoader(dataset, batch_size=10, collate_fn=mock_collate_fn)

        iterator = PassThroughBatchIterator(base_loader)

        batch_count = 0
        for batch in iterator:
            batch_count += 1
            assert batch['input_ids'].shape[0] == 10  # Fixed batch size

        assert batch_count == 10  # 100 samples / 10 batch size
        assert iterator.get_statistics()['total_batches'] == 10

    def test_dynamic_batch_iterator_basic(self):
        """Test DynamicBatchIterator with a mock scheduler."""
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock scheduler that always returns min_batch_size
        mock_scheduler = MagicMock()
        mock_scheduler.get_current_batch_size.return_value = 64  # 1x multiplier
        mock_scheduler.step.return_value = None

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        batches = list(iterator)

        # With 1x multiplier, should yield 4 batches of 64
        assert len(batches) == 4
        for batch in batches:
            assert batch['input_ids'].shape[0] == 64

    def test_dynamic_batch_iterator_concatenation(self):
        """Test that DynamicBatchIterator correctly concatenates mini-batches."""
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock scheduler that always returns 2x min_batch_size
        mock_scheduler = MagicMock()
        mock_scheduler.get_current_batch_size.return_value = 128  # 2x multiplier
        mock_scheduler.step.return_value = None

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        batches = list(iterator)

        # With 2x multiplier, should yield 2 batches of 128
        assert len(batches) == 2
        for batch in batches:
            assert batch['input_ids'].shape[0] == 128

    def test_dynamic_batch_iterator_max_multiplier(self):
        """Test that DynamicBatchIterator respects max_batch_size."""
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock scheduler that returns 4x min_batch_size
        mock_scheduler = MagicMock()
        mock_scheduler.get_current_batch_size.return_value = 256  # 4x multiplier
        mock_scheduler.step.return_value = None

        dataset = MockDataset(num_samples=512)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,  # Max 4x
        )

        batches = list(iterator)

        # Should yield 2 batches of 256
        assert len(batches) == 2
        for batch in batches:
            assert batch['input_ids'].shape[0] == 256

    def test_dynamic_batch_iterator_statistics(self):
        """Test that DynamicBatchIterator tracks statistics correctly."""
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator

        mock_scheduler = MagicMock()
        mock_scheduler.get_current_batch_size.return_value = 128
        mock_scheduler.step.return_value = None
        mock_scheduler.get_statistics.return_value = {'test': 'stats'}

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        # Consume all batches
        _ = list(iterator)

        stats = iterator.get_statistics()

        assert stats['total_batches'] == 2
        assert stats['avg_batch_size'] == 128.0
        assert stats['min_batch_size_used'] == 128
        assert stats['max_batch_size_used'] == 128
        assert 'scheduler_stats' in stats

    def test_dynamic_batch_iterator_varying_batch_sizes(self):
        """Test that DynamicBatchIterator handles varying batch sizes."""
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create scheduler that returns different sizes based on step
        step_sizes = [64, 128, 192, 64]  # Will cycle through these
        call_count = [0]

        def get_batch_size(step):
            return step_sizes[min(call_count[0], len(step_sizes) - 1)]

        mock_scheduler = MagicMock()
        mock_scheduler.get_current_batch_size.side_effect = get_batch_size
        def increment_step(step):
            call_count[0] += 1
            return None
        mock_scheduler.step.side_effect = increment_step
        mock_scheduler.get_statistics.return_value = {}

        # 640 samples with varying batch sizes
        dataset = MockDataset(num_samples=640)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        batches = list(iterator)

        # Batch sizes should vary
        batch_sizes = [b['input_ids'].shape[0] for b in batches]

        # Should have batches of 64, 128, 192, and then 64s for the rest
        assert 64 in batch_sizes
        assert 128 in batch_sizes
        assert 192 in batch_sizes


class TestCreateDynamicBatchIterator:
    """Test the factory function."""

    def test_create_with_config(self):
        """Test creating iterator from config dict."""
        from src.Ava.data.dynamic_batch_iterator import create_dynamic_batch_iterator

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        config = {
            'dynamic_batching': {
                'enabled': True,
                'min_batch_size': 64,
                'max_batch_size': 256,
                'low_memory_threshold': 0.5,
                'target_memory_threshold': 0.7,
                'high_memory_threshold': 0.85,
                'critical_memory_threshold': 0.95,
                'adjustment_frequency': 10,
                'warmup_steps': 100,
            },
            'training': {
                'batch_size': 64,
            }
        }

        iterator = create_dynamic_batch_iterator(base_loader, config)

        # Should be able to iterate
        batches = list(iterator)
        assert len(batches) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
