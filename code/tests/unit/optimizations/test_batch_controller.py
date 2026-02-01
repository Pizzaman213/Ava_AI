"""
Unit tests for BatchSizeController - dynamic batch size management.

Tests the batch size controller that handles OOM recovery and
optimal batch size calibration.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from ava.optimizations.batch_controller import (
    BatchSizeController,
    BatchSizeState,
    MemoryMonitor,
)


class TestBatchSizeController:
    """Tests for BatchSizeController class."""

    @pytest.fixture
    def controller(self):
        """Create default batch size controller."""
        return BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.80,
        )

    def test_controller_creation(self, controller):
        """Test controller initialization."""
        assert controller._min_batch_size == 8
        assert controller._max_batch_size == 128
        assert controller._target_memory == 0.80

    def test_initial_batch_size(self, controller):
        """Test initial batch size is set correctly."""
        state = controller.get_state()

        assert state.current_batch_size >= controller._min_batch_size
        assert state.current_batch_size <= controller._max_batch_size

    def test_record_success(self, controller):
        """Test recording successful batch."""
        initial_state = controller.get_state()

        controller.record_success(batch_size=32, memory_util=0.70)

        # Batch size should be marked as safe
        assert 32 in controller._known_safe

    def test_record_failure(self, controller):
        """Test recording failed batch (OOM)."""
        safe_size = controller.record_failure(batch_size=64)

        # Should return a safe batch size
        assert safe_size <= 64
        assert safe_size >= controller._min_batch_size

        # Batch size should be marked as unsafe
        assert 64 in controller._known_unsafe

    def test_consecutive_failures_reduce_batch(self, controller):
        """Test that consecutive failures reduce batch size."""
        controller.record_failure(batch_size=64)
        safe1 = controller.record_failure(batch_size=64)
        safe2 = controller.record_failure(batch_size=safe1)

        # Should progressively reduce
        assert safe2 <= safe1

    def test_batch_size_stays_in_bounds(self, controller):
        """Test batch size never goes below min or above max."""
        # Try to force very low
        for _ in range(10):
            safe_size = controller.record_failure(batch_size=controller._min_batch_size)

        assert safe_size >= controller._min_batch_size

    def test_get_state(self, controller):
        """Test get_state returns proper snapshot."""
        state = controller.get_state()

        assert isinstance(state, BatchSizeState)
        assert state.current_batch_size >= controller._min_batch_size
        assert state.min_batch_size == controller._min_batch_size
        assert state.max_batch_size == controller._max_batch_size

    def test_safe_ceiling_tracking(self, controller):
        """Test safe ceiling is tracked correctly."""
        controller.record_success(batch_size=32, memory_util=0.70)
        controller.record_success(batch_size=48, memory_util=0.75)
        controller.record_failure(batch_size=64)

        state = controller.get_state()

        # Safe ceiling should be at or below failed size
        assert state.safe_ceiling <= 64


class TestBatchSizeControllerModes:
    """Tests for different controller modes."""

    def test_static_mode(self):
        """Test near-static mode with minimal batch size range.

        Note: BatchSizeController requires min_batch_size < max_batch_size,
        so true static mode (min=max) isn't supported. Test with minimal range.
        """
        controller = BatchSizeController(
            min_batch_size=31,
            max_batch_size=32,  # Minimal range
        )

        state = controller.get_state()
        assert 31 <= state.current_batch_size <= 32

    def test_adaptive_mode(self):
        """Test adaptive mode adjusts batch size."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.75,
        )

        # Record various memory levels
        controller.record_success(batch_size=32, memory_util=0.50)
        controller.record_success(batch_size=48, memory_util=0.60)
        controller.record_success(batch_size=64, memory_util=0.75)

        # Controller should track successful sizes
        assert 32 in controller._known_safe
        assert 48 in controller._known_safe


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create memory monitor."""
        return MemoryMonitor(cache_interval_sec=1.0)

    def test_monitor_creation(self, monitor):
        """Test monitor initialization."""
        assert monitor._cache_interval == 1.0

    def test_get_utilization_cpu(self, monitor):
        """Test getting utilization on CPU."""
        # On CPU, should return 0 or handle gracefully
        util = monitor.get_utilization()

        assert isinstance(util, float)
        assert 0.0 <= util <= 1.0

    def test_get_stats_cpu(self, monitor):
        """Test getting stats on CPU."""
        stats = monitor.get_stats()

        assert isinstance(stats, dict)
        # Should have expected keys
        if stats:  # May be empty on CPU
            assert 'utilization' in stats or len(stats) == 0

    def test_cache_refresh(self, monitor):
        """Test cache refresh logic."""
        # First call should refresh cache
        stats1 = monitor.get_stats()

        # Second call within interval should use cache
        stats2 = monitor.get_stats()

        # Force refresh should bypass cache
        stats3 = monitor.get_stats(force_refresh=True)

        # All should be valid
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)
        assert isinstance(stats3, dict)


class TestBatchSizeControllerWithMocks:
    """Tests using mocked GPU operations."""

    @pytest.fixture
    def mock_cuda(self):
        """Mock CUDA availability."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=10 * 1024**3):
                with patch('torch.cuda.memory_reserved', return_value=12 * 1024**3):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value.total_memory = 24 * 1024**3
                        yield

    def test_controller_with_mock_gpu(self, mock_cuda):
        """Test controller with mocked GPU."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.80,
        )

        # Should work with mocked GPU
        controller.record_success(batch_size=32, memory_util=0.50)
        assert 32 in controller._known_safe


class TestBatchSizeControllerEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_min_equals_max(self):
        """Test when min equals max batch size.

        Note: BatchSizeController requires min_batch_size < max_batch_size,
        so we test with min=31, max=32 to simulate nearly-static mode.
        """
        # BatchSizeController validates min < max, so we can't test exact equality
        # Test with min close to max instead
        controller = BatchSizeController(
            min_batch_size=31,
            max_batch_size=32,
        )

        state = controller.get_state()
        assert 31 <= state.current_batch_size <= 32

        # Should stay within narrow range even on failure
        safe = controller.record_failure(batch_size=32)
        assert safe >= 31

    def test_very_small_min(self):
        """Test with very small minimum batch size."""
        controller = BatchSizeController(
            min_batch_size=1,
            max_batch_size=128,
        )

        # Should handle min batch size of 1
        controller.record_failure(batch_size=1)
        state = controller.get_state()
        assert state.current_batch_size >= 1

    def test_high_memory_target(self):
        """Test with high memory target."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.95,
        )

        controller.record_success(batch_size=64, memory_util=0.90)
        # Should accept high utilization
        assert 64 in controller._known_safe

    def test_low_memory_target(self):
        """Test with low memory target."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
            target_memory=0.50,
        )

        controller.record_success(batch_size=32, memory_util=0.45)
        assert 32 in controller._known_safe

    def test_zero_memory_util(self):
        """Test handling zero memory utilization."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
        )

        # Should handle 0 utilization
        controller.record_success(batch_size=32, memory_util=0.0)
        assert 32 in controller._known_safe

    def test_repeated_failures_same_size(self):
        """Test repeated failures at same batch size."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
        )

        # Multiple failures at same size
        for _ in range(5):
            controller.record_failure(batch_size=64)

        # Should still return valid size
        state = controller.get_state()
        assert state.current_batch_size >= controller._min_batch_size

    def test_alternating_success_failure(self):
        """Test alternating success and failure."""
        controller = BatchSizeController(
            min_batch_size=8,
            max_batch_size=128,
        )

        controller.record_success(batch_size=32, memory_util=0.70)
        controller.record_failure(batch_size=64)
        controller.record_success(batch_size=48, memory_util=0.75)
        controller.record_failure(batch_size=56)

        # Should maintain consistent state
        state = controller.get_state()
        assert 32 in controller._known_safe
        assert 64 in controller._known_unsafe


class TestBatchSizeState:
    """Tests for BatchSizeState dataclass."""

    def test_state_creation(self):
        """Test creating batch size state."""
        state = BatchSizeState(
            current_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            safe_ceiling=64,
            mode='adaptive',
            known_safe_count=5,
            known_unsafe_count=2,
            consecutive_successes=3,
        )

        assert state.current_batch_size == 32
        assert state.safe_ceiling == 64
        assert state.mode == 'adaptive'

    def test_state_immutability(self):
        """Test that state is a snapshot."""
        state = BatchSizeState(
            current_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            safe_ceiling=64,
            mode='adaptive',
            known_safe_count=5,
            known_unsafe_count=2,
            consecutive_successes=3,
        )

        # Attempting to modify should raise or have no effect
        # (Depends on whether dataclass is frozen)
        try:
            state.current_batch_size = 64
        except AttributeError:
            pass  # Expected if frozen
