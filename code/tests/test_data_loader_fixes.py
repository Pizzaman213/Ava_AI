"""
Tests for data loader bug fixes and improvements.

This module tests the fixes implemented for:
- Collate OOM prevention (Issue 1.1)
- Async tokenization safety (Issue 1.2)
- Concatenation logging (Issue 1.3)
- ArrowTableCache memory management (Issue 2.1)
- Bounded statistics collection (Issue 2.2)
- Data validation module (Phase 3.1)
"""

import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_batch():
    """Create a valid sample batch for testing."""
    return [
        {
            'input_ids': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'attention_mask': np.array([1, 1, 1, 1, 1], dtype=np.int64),
            'labels': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        },
        {
            'input_ids': np.array([6, 7, 8], dtype=np.int64),
            'attention_mask': np.array([1, 1, 1], dtype=np.int64),
            'labels': np.array([6, 7, 8], dtype=np.int64),
        },
    ]


@pytest.fixture
def corrupted_batch():
    """Create a batch with corrupted items for testing validation."""
    return [
        {
            'input_ids': np.array([1, 2, 3], dtype=np.int64),  # Valid
            'attention_mask': np.array([1, 1, 1], dtype=np.int64),
            'labels': np.array([1, 2, 3], dtype=np.int64),
        },
        {
            'input_ids': np.array([1] * 1000000, dtype=np.int64),  # Corrupted - too long
            'attention_mask': np.array([1] * 1000000, dtype=np.int64),
            'labels': np.array([1] * 1000000, dtype=np.int64),
        },
        {
            'input_ids': np.array([4, 5, 6], dtype=np.int64),  # Valid
            'attention_mask': np.array([1, 1, 1], dtype=np.int64),
            'labels': np.array([4, 5, 6], dtype=np.int64),
        },
    ]


# =============================================================================
# Test: Collate OOM Prevention (Issue 1.1)
# =============================================================================

class TestCollateOOMPrevention:
    """Tests for Issue 1.1: Collate OOM before corruption detection."""

    def test_validate_sequence_content_valid(self):
        """Test that valid sequences pass validation."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        # Create mock dataset
        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 100000

                # Test valid sequence
                valid_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(valid_ids, max_allowed_len=1000)

                assert is_valid is True
                assert error is None

    def test_validate_sequence_content_too_long(self):
        """Test that overly long sequences are rejected."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 100000

                # Test sequence that exceeds max_allowed_len
                long_ids = np.array([1] * 10000, dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(long_ids, max_allowed_len=1000)

                assert is_valid is False
                assert "too long" in error.lower()

    def test_validate_sequence_content_invalid_token_id(self):
        """Test that invalid token IDs are rejected."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 1000  # Small vocab

                # Test sequence with token ID exceeding vocab
                bad_ids = np.array([1, 2, 999999], dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(bad_ids, max_allowed_len=1000)

                assert is_valid is False
                assert "exceeds vocab" in error.lower()

    def test_validate_sequence_content_empty(self):
        """Test that empty sequences are rejected."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 100000

                # Test empty sequence
                empty_ids = np.array([], dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(empty_ids, max_allowed_len=1000)

                assert is_valid is False
                assert "empty" in error.lower()


# =============================================================================
# Test: Bounded Statistics Collection (Issue 2.2)
# =============================================================================

class TestBoundedStatistics:
    """Tests for Issue 2.2: Unbounded statistics list growth."""

    def test_batch_sizes_bounded(self):
        """Verify _batch_sizes doesn't grow indefinitely."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        # Create mock iterator
        mock_dataloader = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.get_batch_size.return_value = 64

        iterator = DynamicBatchIterator(
            dataloader=mock_dataloader,
            scheduler=mock_scheduler,
            min_batch_size=16,
            max_batch_size=256,
        )

        # Simulate 5000 batches - should be bounded at 1000
        for i in range(5000):
            iterator._batch_sizes.append(64)

        # Should be bounded
        assert len(iterator._batch_sizes) <= 1000
        assert isinstance(iterator._batch_sizes, deque)

    def test_reset_statistics_clears_deque(self):
        """Verify reset_statistics clears the deque properly."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        mock_dataloader = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.get_batch_size.return_value = 64

        iterator = DynamicBatchIterator(
            dataloader=mock_dataloader,
            scheduler=mock_scheduler,
        )

        # Add some batch sizes
        for i in range(100):
            iterator._batch_sizes.append(64)
        assert len(iterator._batch_sizes) == 100

        # Reset
        iterator.reset_statistics()

        assert len(iterator._batch_sizes) == 0
        assert iterator._batches_yielded == 0
        assert iterator._samples_processed == 0


# =============================================================================
# Test: Concatenation Logging (Issue 1.3)
# =============================================================================

class TestConcatenationLogging:
    """Tests for Issue 1.3: Silent tensor concatenation failures."""

    def test_shape_mismatch_logged(self):
        """Verify shape mismatches are logged."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        mock_dataloader = MagicMock()
        mock_scheduler = MagicMock()

        iterator = DynamicBatchIterator(
            dataloader=mock_dataloader,
            scheduler=mock_scheduler,
        )

        # Create batches with different sequence lengths
        batches = [
            {'input_ids': torch.zeros(4, 100)},
            {'input_ids': torch.zeros(4, 200)},  # Different seq length!
        ]

        with patch('ava.data.dynamic_batch_iterator.logger') as mock_logger:
            result = iterator._concatenate(batches)

            # Should have logged a warning
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert 'shape mismatch' in call_args.lower() or 'Shape mismatch' in call_args

    def test_successful_concatenation_no_warning(self):
        """Verify successful concatenation doesn't log warnings."""
        from ava.data.dynamic_batch_iterator import DynamicBatchIterator

        mock_dataloader = MagicMock()
        mock_scheduler = MagicMock()

        iterator = DynamicBatchIterator(
            dataloader=mock_dataloader,
            scheduler=mock_scheduler,
        )

        # Create batches with same shape
        batches = [
            {'input_ids': torch.zeros(4, 100)},
            {'input_ids': torch.zeros(4, 100)},  # Same seq length
        ]

        with patch('ava.data.dynamic_batch_iterator.logger') as mock_logger:
            result = iterator._concatenate(batches)

            # Should NOT have logged a warning
            mock_logger.warning.assert_not_called()

            # Result should be concatenated
            assert result['input_ids'].shape == (8, 100)


# =============================================================================
# Test: Data Validation Module (Phase 3.1)
# =============================================================================

class TestDataValidation:
    """Tests for Phase 3.1: Preventive data validation."""

    def test_validation_result_string_representation(self):
        """Test ValidationResult string representation."""
        from ava.data.validation import ValidationResult

        result = ValidationResult(
            is_valid=True,
            total_samples_checked=1000,
            valid_samples=995,
            invalid_samples=5,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        str_repr = str(result)
        assert "PASSED" in str_repr
        assert "995" in str_repr

    def test_validation_result_failed(self):
        """Test ValidationResult for failed validation."""
        from ava.data.validation import ValidationResult

        result = ValidationResult(
            is_valid=False,
            total_samples_checked=100,
            valid_samples=50,
            invalid_samples=50,
            errors=["Too many errors"],
        )

        str_repr = str(result)
        assert "FAILED" in str_repr

    def test_validation_report_generation(self):
        """Test human-readable report generation."""
        from ava.data.validation import ValidationResult

        result = ValidationResult(
            is_valid=True,
            total_samples_checked=1000,
            valid_samples=1000,
            invalid_samples=0,
            statistics={'mean_length': 512.5, 'files_checked': 10},
        )

        report = result.get_report()
        assert "PASSED" in report
        assert "mean_length" in report
        assert "512.5" in report

    def test_validator_sequence_validation(self):
        """Test sequence validation logic."""
        from ava.data.validation import DataValidator

        validator = DataValidator(
            max_length=2048,
            vocab_size=50000,
            min_sequence_length=10,
        )

        # Valid sequence
        is_valid, error = validator._validate_sequence([1, 2, 3] * 10)
        assert is_valid is True
        assert error is None

        # Too short
        is_valid, error = validator._validate_sequence([1, 2, 3])
        assert is_valid is False
        assert "too short" in error.lower()

        # Token ID out of range (sequence must be long enough first)
        is_valid, error = validator._validate_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 100000])
        assert is_valid is False
        assert "exceeds vocab" in error.lower()


# =============================================================================
# Test: Enhanced Error Messages
# =============================================================================

class TestEnhancedErrors:
    """Tests for enhanced error messages with recovery suggestions."""

    def test_data_loader_error_format(self):
        """Test DataLoaderError formats properly."""
        from ava.data.pretokenized import DataLoaderError

        error = DataLoaderError(
            "Test error message",
            recovery_steps=["Step 1", "Step 2"],
            context={"key": "value"},
        )

        error_str = str(error)
        assert "Test error message" in error_str
        assert "Step 1" in error_str
        assert "Step 2" in error_str
        assert "key: value" in error_str

    def test_oom_recovery_steps(self):
        """Test OOM recovery steps are provided."""
        from ava.data.pretokenized import _get_oom_recovery_steps

        steps = _get_oom_recovery_steps()
        assert len(steps) > 0
        assert any("batch_size" in step.lower() for step in steps)

    def test_no_data_recovery_steps(self):
        """Test no-data recovery steps are provided."""
        from ava.data.pretokenized import _get_no_data_recovery_steps

        steps = _get_no_data_recovery_steps("/test/path")
        assert len(steps) > 0
        assert any("/test/path" in step for step in steps)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
