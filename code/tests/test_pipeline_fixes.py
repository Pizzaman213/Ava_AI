"""
Unit tests for pipeline bug fixes.

Tests all fixes implemented across the training pipeline:
- Phase 1: Configuration validation
- Phase 2: Distributed training
- Phase 3: CUDA memory management
- Phase 4: Data streaming
- Phase 5: Architecture improvements

Run with: pytest code/tests/test_pipeline_fixes.py -v
"""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional, List
from unittest.mock import MagicMock, patch
import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Phase 1: Configuration Validation Tests
# ============================================================================

class TestBatchSizeValidation:
    """Tests for min/max batch size validation."""

    def test_batch_size_validation_min_greater_than_max(self):
        """Test that min_batch_size >= max_batch_size raises ValueError."""
        from ava.optimizations.batch_controller import BatchSizeController

        with pytest.raises(ValueError, match="must be < max_batch_size"):
            BatchSizeController(min_batch_size=256, max_batch_size=128)

    def test_batch_size_validation_equal_values(self):
        """Test that equal min/max batch sizes raise ValueError."""
        from ava.optimizations.batch_controller import BatchSizeController

        with pytest.raises(ValueError, match="must be < max_batch_size"):
            BatchSizeController(min_batch_size=64, max_batch_size=64)

    def test_batch_size_validation_valid(self):
        """Test that valid min < max passes."""
        from ava.optimizations.batch_controller import BatchSizeController

        controller = BatchSizeController(min_batch_size=16, max_batch_size=256)
        assert controller._min_batch_size == 16
        assert controller._max_batch_size == 256


class TestConfigValidator:
    """Tests for enhanced ConfigValidator."""

    def test_calibration_config_validation(self):
        """Test batch_size_calibration config validation."""
        from ava.config.validator import ConfigValidator

        # Invalid: min >= max
        config = {
            'training': {
                'batch_size_calibration': {
                    'enabled': True,
                    'min_batch_size': 256,
                    'max_batch_size': 128,
                }
            }
        }
        validator = ConfigValidator(config, strict=False)
        is_valid, errors = validator.validate(raise_on_error=False)
        assert not is_valid
        assert any("min_batch_size" in e for e in errors)

    def test_target_memory_range_validation(self):
        """Test target_memory must be in (0.0, 1.0]."""
        from ava.config.validator import ConfigValidator

        # Invalid: target_memory > 1.0
        config = {
            'training': {
                'batch_size_calibration': {
                    'enabled': True,
                    'min_batch_size': 1,
                    'max_batch_size': 256,
                    'target_memory': 1.5,  # Invalid
                }
            }
        }
        validator = ConfigValidator(config, strict=False)
        is_valid, errors = validator.validate(raise_on_error=False)
        assert not is_valid
        assert any("target_memory" in e for e in errors)

    def test_validate_against_dataclass(self):
        """Test dataclass-based validation."""
        from ava.config.validator import ConfigValidator

        @dataclass
        class TestConfig:
            hidden_size: int
            num_layers: int = 12
            dropout: Optional[float] = None

        # Valid config
        config = {'model': {'hidden_size': 768, 'num_layers': 16}}
        validator = ConfigValidator(config)
        is_valid, errors = validator.validate_against_dataclass(TestConfig, 'model')
        assert is_valid
        assert len(errors) == 0

        # Invalid: missing required field
        config = {'model': {'num_layers': 16}}
        validator = ConfigValidator(config)
        is_valid, errors = validator.validate_against_dataclass(TestConfig, 'model')
        assert not is_valid
        assert any("hidden_size" in e for e in errors)

        # Invalid: wrong type
        config = {'model': {'hidden_size': "768"}}  # String instead of int
        validator = ConfigValidator(config)
        is_valid, errors = validator.validate_against_dataclass(TestConfig, 'model')
        assert not is_valid
        assert any("expected int" in e for e in errors)


# ============================================================================
# Phase 3: CUDA Memory Management Tests
# ============================================================================

class TestKahanAccumulator:
    """Tests for numerically stable loss accumulation."""

    def test_kahan_accumulator_precision(self):
        """Test that Kahan summation maintains precision."""
        from ava.training.loop import KahanAccumulator

        accumulator = KahanAccumulator()

        # Add many small values - normal addition would accumulate error
        small_value = torch.tensor(0.0001)
        n_additions = 100000

        for _ in range(n_additions):
            accumulator.add(small_value)

        result = accumulator.get_sum()
        expected = small_value * n_additions

        # Kahan should be very close to expected
        assert torch.isclose(result, expected, rtol=1e-5)

    def test_kahan_accumulator_reset(self):
        """Test accumulator reset functionality."""
        from ava.training.loop import KahanAccumulator

        accumulator = KahanAccumulator()
        accumulator.add(torch.tensor(1.0))
        accumulator.add(torch.tensor(2.0))

        assert accumulator.get_count() == 2
        assert accumulator.get_sum() is not None

        accumulator.reset()

        assert accumulator.get_count() == 0
        assert accumulator.get_sum() is None

    def test_kahan_accumulator_mean(self):
        """Test mean calculation."""
        from ava.training.loop import KahanAccumulator

        accumulator = KahanAccumulator()
        accumulator.add(torch.tensor(1.0))
        accumulator.add(torch.tensor(2.0))
        accumulator.add(torch.tensor(3.0))

        mean = accumulator.get_mean()
        assert torch.isclose(mean, torch.tensor(2.0))


# ============================================================================
# Phase 4: Data Streaming Tests
# Note: streaming.py was deleted in a previous codebase reorganization.
# These tests document the intended hash algorithm for future implementation.
# ============================================================================

class TestContentHashAlgorithm:
    """
    Tests for robust content hash deduplication algorithm.

    Note: These tests verify the hash algorithm logic independent of
    the streaming module (which was removed from the codebase).
    """

    def _compute_robust_hash(self, sample):
        """Reference implementation of robust hash for testing."""
        if not isinstance(sample, dict) or 'input_ids' not in sample:
            return id(sample)

        ids = sample['input_ids']
        length = len(ids)

        if length <= 60:
            return hash((length, tuple(ids)))
        else:
            start = tuple(ids[:20])
            middle_start = length // 2 - 10
            middle = tuple(ids[middle_start:middle_start + 20])
            end = tuple(ids[-20:])
            return hash((length, start, middle, end))

    def test_content_hash_short_sequence(self):
        """Test hash for short sequences (< 60 tokens)."""
        sample1 = {'input_ids': list(range(50))}
        sample2 = {'input_ids': list(range(50))}
        sample3 = {'input_ids': list(range(49)) + [100]}

        # Same content should have same hash
        assert self._compute_robust_hash(sample1) == self._compute_robust_hash(sample2)
        # Different content should have different hash
        assert self._compute_robust_hash(sample1) != self._compute_robust_hash(sample3)

    def test_content_hash_long_sequence(self):
        """Test hash for long sequences (>= 60 tokens)."""
        sample1 = {'input_ids': list(range(1000))}
        sample2 = {'input_ids': list(range(1000))}

        ids3 = list(range(1000))
        ids3[500] = 9999  # Change middle
        sample3 = {'input_ids': ids3}

        # Same content should have same hash
        assert self._compute_robust_hash(sample1) == self._compute_robust_hash(sample2)
        # Different middle should give different hash
        assert self._compute_robust_hash(sample1) != self._compute_robust_hash(sample3)

    def test_content_hash_collision_resistance(self):
        """Test that similar sequences don't collide."""
        common_prefix = list(range(20))
        seq1 = common_prefix + list(range(20, 100))
        seq2 = common_prefix + list(range(100, 180))

        sample1 = {'input_ids': seq1}
        sample2 = {'input_ids': seq2}

        # Robust hash should differentiate them
        assert self._compute_robust_hash(sample1) != self._compute_robust_hash(sample2)


# ============================================================================
# Phase 5: Error Types Tests
# ============================================================================

class TestErrorTypes:
    """Tests for centralized error types."""

    def test_ava_training_error_base(self):
        """Test base error class."""
        from ava.core.errors import AvaTrainingError

        error = AvaTrainingError("Test error")
        assert "Test error" in str(error)

        error_with_context = AvaTrainingError("Test error", {"step": 100})
        assert "step=100" in str(error_with_context)

    def test_distributed_sync_error(self):
        """Test distributed sync error."""
        from ava.core.errors import DistributedSyncError

        error = DistributedSyncError(
            "NCCL timeout",
            rank=0,
            world_size=4,
            operation="all_reduce"
        )
        msg = str(error)
        assert "NCCL timeout" in msg
        assert "rank=0" in msg
        assert "world_size=4" in msg
        assert "operation=all_reduce" in msg

    def test_memory_error(self):
        """Test memory error."""
        from ava.core.errors import MemoryError

        error = MemoryError(
            "CUDA OOM",
            batch_size=64,
            allocated_gb=23.5,
            total_gb=24.0
        )
        msg = str(error)
        assert "CUDA OOM" in msg
        assert "batch_size=64" in msg
        assert "23.50" in msg

    def test_configuration_error(self):
        """Test configuration error."""
        from ava.core.errors import ConfigurationError

        error = ConfigurationError(
            "Invalid learning rate",
            field="training.learning_rate",
            value=-0.001,
            expected="float > 0"
        )
        msg = str(error)
        assert "Invalid learning rate" in msg
        assert "training.learning_rate" in msg
        assert "-0.001" in msg

    def test_handle_distributed_error_nccl(self):
        """Test distributed error handler for NCCL errors."""
        from ava.core.errors import handle_distributed_error, DistributedSyncError

        nccl_error = RuntimeError("NCCL error: connection refused")

        with pytest.raises(DistributedSyncError, match="NCCL"):
            handle_distributed_error(nccl_error, "loss_sync")

    def test_handle_distributed_error_timeout(self):
        """Test distributed error handler for timeout errors."""
        from ava.core.errors import handle_distributed_error, DistributedSyncError

        timeout_error = RuntimeError("Operation timeout after 300s")

        with pytest.raises(DistributedSyncError, match="Timeout"):
            handle_distributed_error(timeout_error, "barrier")


# ============================================================================
# Phase 2: Distributed Training Tests (Mock-based)
# ============================================================================

class TestDistributedTimeouts:
    """Tests for configurable distributed timeouts."""

    def test_default_timeouts(self):
        """Test default timeout values."""
        from ava.training.distributed import DistributedTimeouts
        from datetime import timedelta

        timeouts = DistributedTimeouts()

        assert timeouts.BARRIER_DEFAULT == timedelta(minutes=10)
        assert timeouts.CALIBRATION == timedelta(minutes=30)
        assert timeouts.CHECKPOINT == timedelta(minutes=15)
        assert timeouts.ALL_REDUCE == timedelta(seconds=300)
        assert timeouts.CLEANUP == timedelta(minutes=5)

    def test_custom_timeouts(self):
        """Test custom timeout values."""
        from ava.training.distributed import DistributedTimeouts
        from datetime import timedelta

        timeouts = DistributedTimeouts(
            barrier_minutes=5,
            calibration_minutes=15,
            checkpoint_minutes=10,
            all_reduce_seconds=120,
            cleanup_minutes=2,
        )

        assert timeouts.BARRIER_DEFAULT == timedelta(minutes=5)
        assert timeouts.CALIBRATION == timedelta(minutes=15)
        assert timeouts.CHECKPOINT == timedelta(minutes=10)
        assert timeouts.ALL_REDUCE == timedelta(seconds=120)
        assert timeouts.CLEANUP == timedelta(minutes=2)

    def test_from_config(self):
        """Test loading timeouts from config."""
        from ava.training.distributed import DistributedTimeouts
        from datetime import timedelta

        config = {
            'distributed': {
                'timeouts': {
                    'barrier_minutes': 20,
                    'calibration_minutes': 60,
                }
            }
        }

        timeouts = DistributedTimeouts.from_config(config)

        assert timeouts.BARRIER_DEFAULT == timedelta(minutes=20)
        assert timeouts.CALIBRATION == timedelta(minutes=60)
        # Others should use defaults
        assert timeouts.CHECKPOINT == timedelta(minutes=15)


# ============================================================================
# Integration Test Markers
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAIntegration:
    """CUDA-dependent integration tests."""

    def test_prefetcher_memory_threshold(self):
        """Test prefetcher respects memory threshold."""
        # This is a placeholder - actual test would need a DataLoader
        # and GPU memory monitoring
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
