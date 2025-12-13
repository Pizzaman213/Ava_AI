"""
Tests for the DynamicBatchIterator and DynamicBatchScheduler.

Tests the dynamic batching functionality that adjusts batch sizes based on
GPU memory pressure.
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


class TestDynamicBatchScheduler:
    """Test suite for DynamicBatchScheduler."""

    def test_import(self):
        """Test that the module can be imported."""
        from ava.optimizations.dynamic_batching import (
            DynamicBatchScheduler,
            DynamicBatchConfig,
            BatchSizeCalculator,
            MemoryMonitor,
            create_dynamic_batch_scheduler,
        )
        assert DynamicBatchScheduler is not None
        assert DynamicBatchConfig is not None
        assert BatchSizeCalculator is not None
        assert MemoryMonitor is not None
        assert create_dynamic_batch_scheduler is not None

    def test_config_creation(self):
        """Test DynamicBatchConfig creation with defaults."""
        from ava.optimizations.dynamic_batching import DynamicBatchConfig

        config = DynamicBatchConfig()
        assert config.min_batch_size == 16
        assert config.max_batch_size == 256
        assert config.step_size == 16
        assert config.target_memory == 0.75
        assert config.high_memory == 0.85
        assert config.critical_memory == 0.92
        assert config.low_memory == 0.60

    def test_config_validation(self):
        """Test config validation."""
        from ava.optimizations.dynamic_batching import DynamicBatchConfig

        # Invalid min_batch_size
        with pytest.raises(ValueError):
            DynamicBatchConfig(min_batch_size=0)

        # Invalid max_batch_size < min_batch_size
        with pytest.raises(ValueError):
            DynamicBatchConfig(min_batch_size=64, max_batch_size=32)

        # Invalid step_size
        with pytest.raises(ValueError):
            DynamicBatchConfig(step_size=0)

    def test_batch_size_calculator_critical(self):
        """Test calculator handles critical memory."""
        from ava.optimizations.dynamic_batching import BatchSizeCalculator

        calc = BatchSizeCalculator(
            min_batch_size=16,
            max_batch_size=128,
            step_size=16,
        )

        # Critical memory should drop to minimum
        new_size, reason = calc.calculate(64, 0.95)
        assert new_size == 16
        assert 'CRITICAL' in reason

    def test_batch_size_calculator_high(self):
        """Test calculator decreases on high memory."""
        from ava.optimizations.dynamic_batching import BatchSizeCalculator

        calc = BatchSizeCalculator(
            min_batch_size=16,
            max_batch_size=128,
            step_size=16,
        )

        # High memory should decrease by step_size
        new_size, reason = calc.calculate(64, 0.88)
        assert new_size == 48  # 64 - 16
        assert 'HIGH' in reason

    def test_batch_size_calculator_low(self):
        """Test calculator increases on low memory."""
        from ava.optimizations.dynamic_batching import BatchSizeCalculator

        calc = BatchSizeCalculator(
            min_batch_size=16,
            max_batch_size=128,
            step_size=16,
        )

        # Low memory should increase by step_size
        new_size, reason = calc.calculate(64, 0.50)
        assert new_size == 80  # 64 + 16
        assert 'LOW' in reason

    def test_batch_size_calculator_optimal(self):
        """Test calculator maintains size in optimal range."""
        from ava.optimizations.dynamic_batching import BatchSizeCalculator

        calc = BatchSizeCalculator(
            min_batch_size=16,
            max_batch_size=128,
            step_size=16,
        )

        # Optimal memory should not change
        new_size, reason = calc.calculate(64, 0.72)
        assert new_size == 64
        assert 'OPTIMAL' in reason

    def test_scheduler_step(self):
        """Test scheduler step method."""
        from ava.optimizations.dynamic_batching import (
            DynamicBatchScheduler,
            DynamicBatchConfig,
        )

        config = DynamicBatchConfig(
            min_batch_size=16,
            max_batch_size=128,
            warmup_steps=0,  # Disable warmup for test
        )
        scheduler = DynamicBatchScheduler(config)

        # Initial batch size should be min
        assert scheduler.get_batch_size() == 16

    def test_scheduler_record_oom(self):
        """Test scheduler OOM handling."""
        from ava.optimizations.dynamic_batching import (
            DynamicBatchScheduler,
            DynamicBatchConfig,
        )

        config = DynamicBatchConfig(
            min_batch_size=16,
            max_batch_size=128,
            step_size=16,
        )
        scheduler = DynamicBatchScheduler(config)
        scheduler.current_batch_size = 64

        # Record OOM
        new_size = scheduler.record_oom()

        # Should drop by 2 * step_size
        assert new_size == 32  # 64 - 2*16
        assert scheduler.oom_count == 1

    def test_scheduler_statistics(self):
        """Test scheduler statistics tracking."""
        from ava.optimizations.dynamic_batching import (
            DynamicBatchScheduler,
            DynamicBatchConfig,
        )

        config = DynamicBatchConfig(min_batch_size=16, max_batch_size=128)
        scheduler = DynamicBatchScheduler(config)

        stats = scheduler.get_statistics()
        assert 'current_batch_size' in stats
        assert 'adjustments_made' in stats
        assert 'oom_count' in stats
        assert 'memory' in stats


class TestDynamicBatchIterator:
    """Test suite for DynamicBatchIterator."""

    def test_import(self):
        """Test that the module can be imported."""
        from ava.data.dynamic_batch_iterator import (
            DynamicBatchIterator,
            PassThroughIterator,
            create_dynamic_batch_iterator,
        )
        assert DynamicBatchIterator is not None
        assert PassThroughIterator is not None
        assert create_dynamic_batch_iterator is not None

    def test_pass_through_iterator(self):
        """Test PassThroughIterator doesn't modify batches."""
        from ava.data.dynamic_batch_iterator import PassThroughIterator

        dataset = MockDataset(num_samples=100)
        base_loader = DataLoader(dataset, batch_size=10, collate_fn=mock_collate_fn)

        iterator = PassThroughIterator(base_loader)

        batch_count = 0
        for batch in iterator:
            batch_count += 1
            assert batch['input_ids'].shape[0] == 10  # Fixed batch size

        assert batch_count == 10  # 100 samples / 10 batch size

    def test_dynamic_batch_iterator_basic(self):
        """Test DynamicBatchIterator with a mock scheduler."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.get_batch_size.return_value = 64
        mock_scheduler.step.return_value = None
        mock_scheduler.get_memory_stats.return_value = {'smoothed': 0.5}
        mock_scheduler.get_statistics.return_value = {}

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        batches = list(iterator)

        # Should yield 4 batches of 64
        assert len(batches) == 4
        for batch in batches:
            assert batch['input_ids'].shape[0] == 64

    def test_dynamic_batch_iterator_concatenation(self):
        """Test that DynamicBatchIterator correctly concatenates mini-batches."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock scheduler that targets 2x mini-batch size
        mock_scheduler = MagicMock()
        mock_scheduler.get_batch_size.return_value = 128
        mock_scheduler.step.return_value = None
        mock_scheduler.get_memory_stats.return_value = {'smoothed': 0.5}
        mock_scheduler.get_statistics.return_value = {}

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        batches = list(iterator)

        # Should yield 2 batches of 128
        assert len(batches) == 2
        for batch in batches:
            assert batch['input_ids'].shape[0] == 128

    def test_dynamic_batch_iterator_statistics(self):
        """Test that DynamicBatchIterator tracks statistics correctly."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        mock_scheduler = MagicMock()
        mock_scheduler.get_batch_size.return_value = 128
        mock_scheduler.step.return_value = None
        mock_scheduler.get_memory_stats.return_value = {'smoothed': 0.5}
        mock_scheduler.get_statistics.return_value = {'test': 'stats'}

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        # Consume all batches
        _ = list(iterator)

        stats = iterator.get_statistics()

        assert stats['batches_yielded'] == 2
        assert stats['avg_batch_size'] == 128.0
        assert stats['min_batch_used'] == 128
        assert stats['max_batch_used'] == 128


class TestCreateDynamicBatchIterator:
    """Test the factory functions."""

    def test_create_scheduler_with_config(self):
        """Test creating scheduler from config dict."""
        from ava.optimizations.dynamic_batching import create_dynamic_batch_scheduler

        config = {
            'dynamic_batching': {
                'enabled': True,
                'min_batch_size': 32,
                'max_batch_size': 128,
                'step_size': 16,
                'target_memory': 0.70,
                'high_memory': 0.80,
                'critical_memory': 0.90,
                'low_memory': 0.55,
            }
        }

        scheduler = create_dynamic_batch_scheduler(config)
        assert scheduler.config.min_batch_size == 32
        assert scheduler.config.max_batch_size == 128
        assert scheduler.config.target_memory == 0.70

    def test_create_iterator_disabled(self):
        """Test creating iterator returns PassThrough when disabled."""
        from ava.data.dynamic_batch_iterator import (
            create_dynamic_batch_iterator,
            PassThroughIterator,
        )

        dataset = MockDataset(num_samples=100)
        base_loader = DataLoader(dataset, batch_size=10, collate_fn=mock_collate_fn)

        config = {
            'dynamic_batching': {
                'enabled': False,
            }
        }

        iterator = create_dynamic_batch_iterator(base_loader, config)
        assert isinstance(iterator, PassThroughIterator)

    def test_create_iterator_enabled(self):
        """Test creating iterator returns DynamicBatchIterator when enabled."""
        from ava.data.dynamic_batch_iterator import (
            create_dynamic_batch_iterator,
            DynamicBatchIterator,
        )

        dataset = MockDataset(num_samples=256)
        base_loader = DataLoader(dataset, batch_size=64, collate_fn=mock_collate_fn)

        config = {
            'dynamic_batching': {
                'enabled': True,
                'min_batch_size': 64,
                'max_batch_size': 256,
            }
        }

        iterator = create_dynamic_batch_iterator(base_loader, config)
        assert isinstance(iterator, DynamicBatchIterator)


class TestBatchController:
    """Tests for the BatchSizeController."""

    def test_import(self):
        """Test that BatchSizeController can be imported."""
        from ava.optimizations.batch_controller import (
            BatchSizeController,
            create_batch_size_controller,
        )
        assert BatchSizeController is not None
        assert create_batch_size_controller is not None

    def test_controller_creation(self):
        """Test creating BatchSizeController."""
        from ava.optimizations.batch_controller import BatchSizeController

        controller = BatchSizeController(
            min_batch_size=16,
            max_batch_size=128,
            target_memory=0.75,
        )

        assert controller.get_batch_size() == 16  # Starts at min
        assert controller._mode == 'calibrating'

    def test_controller_record_success(self):
        """Test recording successful batches."""
        from ava.optimizations.batch_controller import BatchSizeController

        controller = BatchSizeController(min_batch_size=16, max_batch_size=128)

        controller.record_success(batch_size=64, memory_util=0.70)

        assert 64 in controller._known_safe
        assert controller._safe_ceiling == 64

    def test_controller_record_failure(self):
        """Test recording OOM failures."""
        from ava.optimizations.batch_controller import BatchSizeController

        controller = BatchSizeController(min_batch_size=16, max_batch_size=128)
        controller._current_batch_size = 64

        new_size = controller.record_failure(batch_size=64)

        assert new_size < 64
        assert 64 in controller._known_unsafe
        assert controller._mode == 'recovering'

    def test_controller_factory(self):
        """Test factory function."""
        from ava.optimizations.batch_controller import create_batch_size_controller

        config = {
            'enabled': True,
            'min_batch_size': 32,
            'max_batch_size': 256,
            'target_memory': 0.80,
        }

        controller = create_batch_size_controller(config)
        assert controller is not None
        assert controller._min_batch_size == 32
        assert controller._max_batch_size == 256

    def test_controller_factory_disabled(self):
        """Test factory returns None when disabled."""
        from ava.optimizations.batch_controller import create_batch_size_controller

        config = {'enabled': False}

        controller = create_batch_size_controller(config)
        assert controller is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
