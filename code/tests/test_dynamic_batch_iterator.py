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
        mock_scheduler.current_batch_size = 64  # 1x multiplier (attribute, not method)
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
        mock_scheduler.current_batch_size = 128  # 2x multiplier (attribute, not method)
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
        mock_scheduler.current_batch_size = 256  # 4x multiplier (attribute, not method)
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
        mock_scheduler.current_batch_size = 128  # attribute, not method
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

        mock_scheduler = MagicMock()
        mock_scheduler.current_batch_size = 64  # Start with 64
        def step_and_update(step, batch=None):
            call_count[0] += 1
            mock_scheduler.current_batch_size = step_sizes[min(call_count[0], len(step_sizes) - 1)]
            return None
        mock_scheduler.step.side_effect = step_and_update
        mock_scheduler.get_statistics.return_value = {}
        mock_scheduler.get_memory_stats.return_value = {'utilization': 0.5, 'raw_utilization': 0.5}

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


class TestKalmanMemoryPredictor:
    """Tests for the Kalman filter memory predictor."""

    def test_kalman_predictor_import(self):
        """Test that KalmanMemoryPredictor can be imported."""
        from src.Ava.training.optimizations.dynamic_batching import KalmanMemoryPredictor
        assert KalmanMemoryPredictor is not None

    def test_kalman_online_learning(self):
        """Test that predictor learns without separate calibration."""
        from src.Ava.training.optimizations.dynamic_batching import KalmanMemoryPredictor

        predictor = KalmanMemoryPredictor(min_observations=5)

        # Simulate observations with linear memory model: memory = 2.0 + 0.00001 * tokens
        for i in range(10):
            batch_size = 32 * (i % 4 + 1)  # 32, 64, 96, 128
            seq_len = 128 * (i % 3 + 1)    # 128, 256, 384
            tokens = batch_size * seq_len
            memory = 2.0 + 0.00001 * tokens  # Simulated memory usage
            predictor.observe(batch_size, seq_len, memory)

        # Should be calibrated after min_observations
        assert predictor.calibrated
        assert predictor.observation_count == 10

        # Should be able to predict
        pred, uncertainty = predictor.predict_memory(64, 256)
        assert pred > 0
        assert uncertainty < float('inf')

    def test_kalman_uncertainty_decreases(self):
        """Test that uncertainty decreases with more observations."""
        from src.Ava.training.optimizations.dynamic_batching import KalmanMemoryPredictor

        predictor = KalmanMemoryPredictor(min_observations=3)
        predictor.set_total_memory(24 * 1e9)  # 24GB

        uncertainties = []
        # Use consistent observations to reduce uncertainty
        for i in range(30):
            predictor.observe(32, 128, 2.5 + 0.01 * (i % 3))  # Slight variation
            if predictor.calibrated:
                _, uncertainty = predictor.predict_memory(32, 128)
                uncertainties.append(uncertainty)

        # Uncertainty should decrease over time (later values < earlier values)
        assert len(uncertainties) > 10
        assert uncertainties[-1] < uncertainties[5]

    def test_kalman_is_safe(self):
        """Test safety check with uncertainty bounds."""
        from src.Ava.training.optimizations.dynamic_batching import KalmanMemoryPredictor

        predictor = KalmanMemoryPredictor(min_observations=5)
        predictor.set_total_memory(24 * 1e9)  # 24GB

        # Calibrate with small batches
        for i in range(10):
            predictor.observe(32, 128, 3.0)  # ~3GB per batch

        # Small batch should be safe
        assert predictor.is_safe(32, 128, threshold=0.85)

        # Very large batch should not be safe (would use ~90% of 24GB)
        assert not predictor.is_safe(1024, 1024, threshold=0.85)


class TestAdaptiveAdjustmentStrategy:
    """Tests for the adaptive adjustment strategy."""

    def test_adaptive_adjustment_import(self):
        """Test that AdaptiveAdjustmentStrategy can be imported."""
        from src.Ava.training.optimizations.dynamic_batching import (
            AdaptiveAdjustmentStrategy,
            ThroughputTracker,
            DynamicBatchConfig,
        )
        assert AdaptiveAdjustmentStrategy is not None
        assert ThroughputTracker is not None

    def test_momentum_increases_with_consecutive_adjustments(self):
        """Test that consecutive adjustments build momentum."""
        from src.Ava.training.optimizations.dynamic_batching import (
            AdaptiveAdjustmentStrategy,
            DynamicBatchConfig,
        )

        config = DynamicBatchConfig(
            low_memory_threshold=0.6,
            high_memory_threshold=0.85,
            critical_memory_threshold=0.92,
        )
        strategy = AdaptiveAdjustmentStrategy(config, max_momentum=3.0)

        # Initial state
        assert strategy.consecutive_increases == 0
        assert strategy.momentum_factor == 1.0

        # Simulate multiple low memory situations (should increase batch)
        for _ in range(3):
            new_size, reason, confidence = strategy.calculate_adjustment(
                current_batch_size=64,
                min_batch_size=32,
                max_batch_size=256,
                mem_stats={'utilization': 0.4, 'raw_utilization': 0.4},
            )

        # Should have built up momentum
        assert strategy.consecutive_increases > 0
        assert strategy.momentum_factor > 1.0

    def test_threshold_learning_from_oom(self):
        """Test that thresholds adapt after OOM events."""
        from src.Ava.training.optimizations.dynamic_batching import (
            AdaptiveAdjustmentStrategy,
            DynamicBatchConfig,
        )

        config = DynamicBatchConfig(critical_memory_threshold=0.92)
        strategy = AdaptiveAdjustmentStrategy(config, learn_thresholds=True)

        initial_critical = strategy.learned_thresholds['critical']

        # Record OOM at lower memory level
        strategy.record_oom(0.88)

        # Critical threshold should decrease
        assert strategy.learned_thresholds['critical'] < initial_critical

    def test_throughput_tracker(self):
        """Test throughput tracking for adjustment learning."""
        from src.Ava.training.optimizations.dynamic_batching import ThroughputTracker

        tracker = ThroughputTracker(window_size=10, stabilization_steps=3)

        # Record some throughput measurements before adjustment
        for step in range(10):
            tracker.record(step, batch_size=64, tokens_per_sec=1000.0)

        # Record adjustment at step 10
        adjustment_step = 10

        # Record measurements after adjustment (improved throughput)
        for step in range(11, 25):
            tracker.record(step, batch_size=128, tokens_per_sec=1500.0)

        # Check if adjustment was beneficial
        result = tracker.was_adjustment_beneficial(
            adjustment_step=adjustment_step,
            old_batch_size=64,
            new_batch_size=128,
        )
        assert result is True  # 50% improvement is beneficial


class TestUnifiedBatchingStrategy:
    """Tests for the unified token-budget + sequence-aware batching."""

    def test_unified_strategy_import(self):
        """Test that UnifiedBatchingStrategy can be imported."""
        from src.Ava.data.dynamic_batch_iterator import (
            UnifiedBatchingStrategy,
            UnifiedBatchingConfig,
        )
        assert UnifiedBatchingStrategy is not None
        assert UnifiedBatchingConfig is not None

    def test_short_sequences_get_higher_budget(self):
        """Test that shorter sequences get higher token budget."""
        from src.Ava.data.dynamic_batch_iterator import (
            UnifiedBatchingStrategy,
            UnifiedBatchingConfig,
        )

        config = UnifiedBatchingConfig(
            base_token_budget=8192,
            reference_seq_len=512,
            sequence_adjustment_strength=0.5,
        )
        strategy = UnifiedBatchingStrategy(config)

        short_budget = strategy.get_target_token_budget(avg_seq_len=128)
        long_budget = strategy.get_target_token_budget(avg_seq_len=1024)

        # Shorter sequences should have higher token budget
        assert short_budget > long_budget

    def test_effective_load_calculation(self):
        """Test effective load accounts for quadratic attention."""
        from src.Ava.data.dynamic_batch_iterator import (
            UnifiedBatchingStrategy,
            UnifiedBatchingConfig,
        )

        config = UnifiedBatchingConfig(attention_memory_factor=0.001)
        strategy = UnifiedBatchingStrategy(config)

        # Same token count, different seq lengths
        load_short = strategy.get_effective_load(tokens=8192, avg_seq_len=128)
        load_long = strategy.get_effective_load(tokens=8192, avg_seq_len=512)

        # Longer sequences should have higher effective load (more attention memory)
        assert load_long > load_short

    def test_should_yield_batch_decision(self):
        """Test yield decision based on effective load."""
        from src.Ava.data.dynamic_batch_iterator import (
            UnifiedBatchingStrategy,
            UnifiedBatchingConfig,
        )

        config = UnifiedBatchingConfig(
            base_token_budget=4096,
            reference_seq_len=256,
            sequence_adjustment_strength=0.5,
        )
        strategy = UnifiedBatchingStrategy(config)

        # Small accumulation - should not yield
        should_yield_small = strategy.should_yield_batch(
            accumulated_tokens=1000,
            accumulated_samples=10,
            avg_seq_len=256,  # Use reference seq len
            memory_utilization=0.5,
        )
        assert not should_yield_small

        # Large accumulation (well above budget) - should yield
        should_yield_large = strategy.should_yield_batch(
            accumulated_tokens=10000,  # Much higher than 4096 budget
            accumulated_samples=100,
            avg_seq_len=256,  # Use reference seq len
            memory_utilization=0.5,
        )
        assert should_yield_large


class TestIntegration:
    """Integration tests for all new features working together."""

    def test_full_dynamic_batching_pipeline(self):
        """Test complete dynamic batching with all new features."""
        from src.Ava.data.dynamic_batch_iterator import (
            DynamicBatchIterator,
            UnifiedBatchingConfig,
        )

        # Create mock scheduler with Kalman predictor
        mock_scheduler = MagicMock()
        mock_scheduler.current_batch_size = 64
        mock_scheduler.step.return_value = None
        mock_scheduler.get_memory_stats.return_value = {'utilization': 0.6, 'raw_utilization': 0.6}
        mock_scheduler.get_statistics.return_value = {}

        # Create unified config
        unified_config = UnifiedBatchingConfig(
            enabled=True,
            base_token_budget=4096,
            reference_seq_len=128,
            sequence_adjustment_strength=0.5,
        )

        dataset = MockDataset(num_samples=1000, seq_length=64)
        base_loader = DataLoader(dataset, batch_size=32, collate_fn=mock_collate_fn)

        iterator = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=mock_scheduler,
            min_batch_size=32,
            max_batch_size=256,
            unified_batching_config=unified_config,
        )

        # Verify unified strategy is active
        assert iterator.unified_strategy is not None

        # Iterate and collect batches
        batches = list(iterator)
        assert len(batches) > 0

        # Check statistics include unified strategy
        stats = iterator.get_statistics()
        assert stats['mode'] == 'unified'
        assert 'unified_strategy_stats' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
