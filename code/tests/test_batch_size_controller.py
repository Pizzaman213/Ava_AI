"""
Tests for BatchSizeController

Tests the core batch size management functionality:
- Safe/unsafe batch size tracking
- OOM recovery with known-safe fallback
- Startup calibration
- Sequence length adjustment
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Import the module under test
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from Ava.training.optimizations.batch_size_controller import (
    BatchSizeController,
    MemoryMonitor,
    BatchSizeState,
    create_batch_size_controller,
)


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(cache_interval_sec=1.0)
        assert monitor._cache_interval == 1.0

    def test_get_utilization_without_cuda(self):
        """Test utilization returns 0 when CUDA not available."""
        monitor = MemoryMonitor()
        monitor._total_memory = 0  # Simulate no CUDA
        util = monitor.get_utilization()
        assert util == 0.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_reserved', return_value=8_000_000_000)  # 8GB
    @patch('torch.cuda.memory_allocated', return_value=6_000_000_000)  # 6GB
    @patch('torch.cuda.current_device', return_value=0)
    def test_get_utilization_with_cuda(self, mock_device, mock_alloc, mock_reserved, mock_avail):
        """Test utilization calculation with CUDA."""
        monitor = MemoryMonitor()
        monitor._total_memory = 16_000_000_000  # 16GB
        monitor._device = 0

        util = monitor.get_utilization(force_refresh=True)
        # 8GB reserved / 16GB total = 0.5
        assert 0.4 <= util <= 0.6


class TestBatchSizeController:
    """Tests for BatchSizeController class."""

    def test_initialization(self):
        """Test controller initialization with default values."""
        controller = BatchSizeController(
            min_batch_size=16,
            max_batch_size=256,
            target_memory=0.75,
        )

        assert controller._min_batch_size == 16
        assert controller._max_batch_size == 256
        assert controller._target_memory == 0.75
        assert controller._current_batch_size == 16
        assert controller._mode == 'calibrating'

    def test_record_success_updates_safe_ceiling(self):
        """Test that recording success updates safe ceiling."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)

        controller.record_success(batch_size=64, memory_util=0.5)
        assert 64 in controller._known_safe
        assert controller._safe_ceiling == 64

        controller.record_success(batch_size=128, memory_util=0.7)
        assert 128 in controller._known_safe
        assert controller._safe_ceiling == 128

    def test_record_failure_returns_safe_size(self):
        """Test that recording failure returns known-safe size."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)

        # Record some safe sizes first
        controller.record_success(batch_size=32, memory_util=0.4)
        controller.record_success(batch_size=64, memory_util=0.6)

        # Now record failure at 128
        safe_size = controller.record_failure(batch_size=128)

        # Should return 64 (highest known-safe below 128)
        assert safe_size == 64
        assert 128 in controller._known_unsafe
        assert controller._mode == 'recovering'

    def test_record_failure_with_no_known_safe(self):
        """Test failure handling when no sizes are known safe."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)

        # Fail at 64 with no prior successes
        safe_size = controller.record_failure(batch_size=64)

        # Should fallback to min_batch_size or reduced size
        assert safe_size <= 64
        assert safe_size >= 16

    def test_known_unsafe_never_returned(self):
        """Test that unsafe sizes are never returned."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)

        controller.record_success(batch_size=32, memory_util=0.4)
        controller.record_success(batch_size=64, memory_util=0.6)
        controller.record_failure(batch_size=128)

        # After failure, get_batch_size should not return 128
        for _ in range(10):
            bs = controller.get_batch_size()
            assert bs != 128

    def test_get_safe_batch_for_seq_len_scaling(self):
        """Test sequence length adjustment."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)
        controller._current_batch_size = 128
        controller._safe_ceiling = 128

        # Same sequence length -> same batch size
        bs_same = controller.get_safe_batch_for_seq_len(seq_len=512, base_seq_len=512)
        assert bs_same == 128

        # Shorter sequence -> can use same or larger batch
        bs_shorter = controller.get_safe_batch_for_seq_len(seq_len=256, base_seq_len=512)
        assert bs_shorter == 128  # Won't exceed current

        # Longer sequence -> should reduce
        bs_longer = controller.get_safe_batch_for_seq_len(seq_len=1024, base_seq_len=512)
        assert bs_longer < 128
        # 512/1024 = 0.5, squared = 0.25, 128 * 0.25 = 32
        assert bs_longer == 32

    def test_get_state(self):
        """Test state snapshot."""
        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)
        controller.record_success(batch_size=64, memory_util=0.6)

        state = controller.get_state()
        assert isinstance(state, BatchSizeState)
        assert state.current_batch_size == 16
        assert state.min_batch_size == 16
        assert state.max_batch_size == 256
        assert state.safe_ceiling == 64
        assert state.known_safe_count == 1

    def test_is_stable_property(self):
        """Test stability detection."""
        controller = BatchSizeController(
            min_batch_size=16,
            max_batch_size=256,
            stability_threshold=3,
        )

        assert not controller.is_stable  # Initially calibrating

        # Record enough successes to stabilize
        controller._mode = 'recovering'
        for i in range(5):
            controller.record_success(batch_size=64, memory_util=0.6)

        assert controller.is_stable

    def test_recovery_to_stable(self):
        """Test transition from recovering to stable."""
        controller = BatchSizeController(
            min_batch_size=16,
            max_batch_size=256,
            stability_threshold=3,
        )

        # Force into recovering mode
        controller.record_failure(batch_size=128)
        assert controller._mode == 'recovering'

        # Record enough successes
        for i in range(5):
            controller.record_success(batch_size=32, memory_util=0.4)

        assert controller._mode == 'stable'


class TestCreateBatchSizeController:
    """Tests for factory function."""

    def test_create_disabled(self):
        """Test factory returns None when disabled."""
        config = {'enabled': False}
        controller = create_batch_size_controller(config)
        assert controller is None

    def test_create_enabled(self):
        """Test factory creates controller when enabled."""
        config = {
            'enabled': True,
            'min_batch_size': 32,
            'max_batch_size': 128,
            'target_memory_utilization': 0.8,
        }
        controller = create_batch_size_controller(config)
        assert controller is not None
        assert controller._min_batch_size == 32
        assert controller._max_batch_size == 128
        assert controller._target_memory == 0.8

    def test_create_with_nested_config(self):
        """Test factory handles nested config."""
        config = {
            'dynamic_batching': {
                'enabled': True,
                'min_batch_size': 16,
            }
        }
        controller = create_batch_size_controller(config)
        assert controller is not None


class MockModel(nn.Module):
    """Mock model for calibration testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass matching transformer interface."""
        x = self.embed(input_ids)
        return self.linear(x)


class TestCalibration:
    """Tests for startup calibration."""

    def test_calibration_mock(self):
        """Test calibration with mocked model."""
        controller = BatchSizeController(
            min_batch_size=16,
            max_batch_size=256,
            target_memory=0.75,
        )

        # Create a mock model that matches transformer interface
        model = MockModel()

        # Mock the memory monitor to return controlled values
        memory_sequence = iter([0.3, 0.5, 0.7, 0.8])  # Increasing memory

        def mock_get_util(force_refresh=False):
            try:
                return next(memory_sequence)
            except StopIteration:
                return 0.8

        controller._memory_monitor.get_utilization = mock_get_util

        def sample_batch_fn(bs):
            return {
                'input_ids': torch.randint(0, 100, (bs, 32)),
                'attention_mask': torch.ones(bs, 32),
            }

        # Run calibration with short timeout
        optimal = controller.startup_calibration(
            model=model,
            sample_batch_fn=sample_batch_fn,
            target_memory=0.75,
            max_time_seconds=1.0,  # Very short for test
        )

        # Should have found some size
        assert optimal >= 16
        assert controller._mode == 'stable'


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow: init -> success -> failure -> recovery."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.75,
            stability_threshold=3,
        )

        # Initial state
        assert controller.get_batch_size() == 8
        assert controller._mode == 'calibrating'

        # Record successes at various sizes
        controller.record_success(16, 0.3)
        controller.record_success(32, 0.5)
        controller.record_success(64, 0.7)

        # Update batch size to largest known safe
        controller._current_batch_size = 64

        # Now simulate OOM at 128
        safe_size = controller.record_failure(128)
        assert safe_size == 64  # Should return to 64
        assert controller._mode == 'recovering'

        # Record successes at 64 to stabilize
        for _ in range(5):
            controller.record_success(64, 0.7)

        assert controller._mode == 'stable'
        assert controller.is_stable

        # Get statistics
        stats = controller.get_statistics()
        assert 'current_batch_size' in stats
        assert 'safe_ceiling' in stats
        assert stats['mode'] == 'stable'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
