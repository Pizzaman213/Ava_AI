"""
Data Validation Module - Preventive validation before training.

This module validates data BEFORE training starts to catch issues early,
preventing wasted compute time on corrupted or misconfigured data.

Key Features:
- File accessibility checks
- Schema validation (required columns exist)
- Sequence length validation
- Token ID range validation
- Data corruption detection
- Statistics collection

Usage:
    from ava.data.validation import DataValidator, ValidationResult

    validator = DataValidator(max_length=2048, vocab_size=50000)
    result = validator.validate_dataset(Path("/data/train"), split="train")

    if not result.is_valid:
        raise RuntimeError(f"Data validation failed: {result.errors}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import pyarrow for Arrow file validation
try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    ipc = None
    pq = None


@dataclass
class ValidationResult:
    """Results from data validation."""
    is_valid: bool
    total_samples_checked: int
    valid_samples: int
    invalid_samples: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"ValidationResult({status}): "
            f"{self.valid_samples}/{self.total_samples_checked} valid, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )

    def get_report(self) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Samples checked: {self.total_samples_checked:,}",
            f"Valid samples: {self.valid_samples:,}",
            f"Invalid samples: {self.invalid_samples:,}",
            "",
        ]

        if self.statistics:
            lines.append("Statistics:")
            for key, value in self.statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                lines.append(f"  - {error}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
            lines.append("")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {warning}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DataStatistics:
    """Comprehensive statistics about the dataset."""
    total_samples: int = 0
    total_tokens: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    std_length: float = 0.0
    percentiles: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[int, int] = field(default_factory=dict)
    files_checked: int = 0
    files_failed: int = 0


class DataValidator:
    """
    Validates data files before training.

    Performs comprehensive checks:
    - File accessibility (can open, read)
    - Schema validation (required columns exist)
    - Sequence length validation (within bounds)
    - Token ID range validation (within vocab)
    - Data corruption detection (invalid values)

    Args:
        max_length: Maximum allowed sequence length
        vocab_size: Vocabulary size for token ID validation
        min_sequence_length: Minimum sequence length
        sample_rate: Fraction of data to validate (0.0-1.0)
        max_samples: Maximum samples to check (for speed)
        strict: If True, validation fails on any error; if False, allows small error rate
    """

    def __init__(
        self,
        max_length: int,
        vocab_size: int = 100000,
        min_sequence_length: int = 10,
        sample_rate: float = 0.01,  # Check 1% of data
        max_samples: int = 10000,   # Cap at 10k samples for speed
        strict: bool = True,        # Block training on errors
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.min_sequence_length = min_sequence_length
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.strict = strict

        if not PYARROW_AVAILABLE:
            logger.warning("PyArrow not available - some validation features disabled")

    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single data file.

        Args:
            file_path: Path to the data file

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        valid_count = 0
        invalid_count = 0
        lengths: List[int] = []

        # Check file exists and is readable
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"File does not exist: {file_path}"]
            )

        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"File is empty: {file_path}"]
                )
        except OSError as e:
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Cannot access file: {e}"]
            )

        # Validate based on file type
        suffix = file_path.suffix.lower()
        try:
            if suffix == '.arrow':
                return self._validate_arrow_file(file_path)
            elif suffix == '.parquet':
                return self._validate_parquet_file(file_path)
            else:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"Unsupported file format: {suffix}"]
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Failed to validate file: {e}"]
            )

    def _validate_arrow_file(self, file_path: Path) -> ValidationResult:
        """Validate Arrow file contents."""
        if not PYARROW_AVAILABLE:
            return ValidationResult(
                is_valid=True,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                warnings=["PyArrow not available - skipping Arrow validation"]
            )

        errors: List[str] = []
        warnings: List[str] = []
        valid_count = 0
        invalid_count = 0
        lengths: List[int] = []

        try:
            # Try IPC File format first, then fall back to IPC Stream format
            try:
                with pa.memory_map(str(file_path), 'r') as source:
                    reader = ipc.open_file(source)
                    table = reader.read_all()
            except pa.ArrowInvalid:
                # IPC Stream format (HuggingFace datasets)
                with open(str(file_path), 'rb') as f:
                    table = ipc.open_stream(f).read_all()

            # Check required columns
            schema_names = table.schema.names
            if 'input_ids' not in schema_names:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"Missing 'input_ids' column in {file_path.name}"]
                )

            # Sample rows for validation
            total_rows = len(table)
            if total_rows == 0:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"No rows in {file_path.name}"]
                )

            # Calculate sample size
            sample_size = min(
                int(total_rows * self.sample_rate),
                self.max_samples
            )
            sample_size = max(sample_size, min(100, total_rows))  # At least 100 samples

            # Random sample indices
            sample_indices = np.random.choice(
                total_rows,
                size=sample_size,
                replace=False
            )

            # Validate sampled rows
            for idx in sample_indices:
                row = table.slice(int(idx), 1)
                input_ids = row['input_ids'][0].as_py()

                is_valid, error = self._validate_sequence(input_ids)
                if is_valid:
                    valid_count += 1
                    lengths.append(len(input_ids))
                else:
                    invalid_count += 1
                    if len(errors) < 10:
                        errors.append(f"Row {idx}: {error}")

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Failed to read Arrow file: {e}"]
            )

        # Calculate statistics
        statistics = {}
        if lengths:
            statistics = {
                'min_length': int(min(lengths)),
                'max_length': int(max(lengths)),
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
                'p50_length': int(np.percentile(lengths, 50)),
                'p90_length': int(np.percentile(lengths, 90)),
                'p99_length': int(np.percentile(lengths, 99)),
            }

        # Determine validity
        total_checked = valid_count + invalid_count
        error_rate = invalid_count / total_checked if total_checked > 0 else 0

        if self.strict:
            is_valid = invalid_count == 0
        else:
            is_valid = error_rate < 0.01  # Allow up to 1% errors in non-strict mode

        return ValidationResult(
            is_valid=is_valid,
            total_samples_checked=total_checked,
            valid_samples=valid_count,
            invalid_samples=invalid_count,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

    def _validate_parquet_file(self, file_path: Path) -> ValidationResult:
        """Validate Parquet file contents."""
        if not PYARROW_AVAILABLE:
            return ValidationResult(
                is_valid=True,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                warnings=["PyArrow not available - skipping Parquet validation"]
            )

        errors: List[str] = []
        warnings: List[str] = []
        valid_count = 0
        invalid_count = 0
        lengths: List[int] = []

        try:
            parquet_file = pq.ParquetFile(file_path)
            schema_names = parquet_file.schema_arrow.names

            # Check required columns
            if 'input_ids' not in schema_names:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"Missing 'input_ids' column in {file_path.name}"]
                )

            # Read and validate samples
            table = parquet_file.read()
            total_rows = len(table)

            if total_rows == 0:
                return ValidationResult(
                    is_valid=False,
                    total_samples_checked=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=[f"No rows in {file_path.name}"]
                )

            # Calculate sample size
            sample_size = min(
                int(total_rows * self.sample_rate),
                self.max_samples
            )
            sample_size = max(sample_size, min(100, total_rows))

            # Random sample indices
            sample_indices = np.random.choice(
                total_rows,
                size=sample_size,
                replace=False
            )

            # Validate sampled rows
            for idx in sample_indices:
                row = table.slice(int(idx), 1)
                input_ids = row['input_ids'][0].as_py()

                is_valid, error = self._validate_sequence(input_ids)
                if is_valid:
                    valid_count += 1
                    lengths.append(len(input_ids))
                else:
                    invalid_count += 1
                    if len(errors) < 10:
                        errors.append(f"Row {idx}: {error}")

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Failed to read Parquet file: {e}"]
            )

        # Calculate statistics
        statistics = {}
        if lengths:
            statistics = {
                'min_length': int(min(lengths)),
                'max_length': int(max(lengths)),
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
            }

        # Determine validity
        total_checked = valid_count + invalid_count
        error_rate = invalid_count / total_checked if total_checked > 0 else 0

        if self.strict:
            is_valid = invalid_count == 0
        else:
            is_valid = error_rate < 0.01

        return ValidationResult(
            is_valid=is_valid,
            total_samples_checked=total_checked,
            valid_samples=valid_count,
            invalid_samples=invalid_count,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

    def _validate_sequence(self, input_ids: List[int]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single sequence.

        Args:
            input_ids: Token IDs to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if input_ids is None:
            return False, "input_ids is None"

        if not isinstance(input_ids, (list, np.ndarray)):
            return False, f"Invalid type: {type(input_ids)}"

        # Check length
        seq_len = len(input_ids)
        if seq_len < self.min_sequence_length:
            return False, f"Sequence too short: {seq_len} < {self.min_sequence_length}"

        # Check for catastrophically long sequences
        max_allowed = self.max_length * 10  # 10x max_length is suspicious
        if seq_len > max_allowed:
            return False, f"Sequence too long: {seq_len} > {max_allowed}"

        # Check token IDs
        try:
            arr = np.array(input_ids)
            min_val = int(np.min(arr))
            max_val = int(np.max(arr))

            if min_val < -1:  # Allow -1 for special tokens
                return False, f"Negative token ID: {min_val}"

            if max_val >= self.vocab_size:
                return False, f"Token ID {max_val} exceeds vocab size {self.vocab_size}"

        except (ValueError, TypeError) as e:
            return False, f"Token validation failed: {e}"

        return True, None

    def validate_dataset(
        self,
        data_dir: Path,
        split: str = 'train',
        patterns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate all files in a dataset directory.

        Args:
            data_dir: Directory containing data files
            split: Data split to validate ('train', 'val', etc.)
            patterns: Glob patterns to find files (optional)

        Returns:
            ValidationResult with aggregated validation outcome
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Data directory does not exist: {data_dir}"]
            )

        # Find data files
        if patterns is None:
            patterns = [
                f"**/{split}/**/*.arrow",
                f"**/{split}/**/*.parquet",
                f"{split}_*.arrow",
                f"{split}_*.parquet",
                "*.arrow",
                "*.parquet",
            ]

        files: List[Path] = []
        for pattern in patterns:
            files.extend(data_dir.glob(pattern))

        # Remove duplicates
        files = list(set(files))

        if not files:
            return ValidationResult(
                is_valid=False,
                total_samples_checked=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"No data files found in {data_dir} for split '{split}'"]
            )

        logger.info(f"Validating {len(files)} files in {data_dir}")

        # Aggregate results
        all_errors: List[str] = []
        all_warnings: List[str] = []
        total_valid = 0
        total_invalid = 0
        total_checked = 0
        all_lengths: List[float] = []
        files_failed = 0

        for file_path in files:
            result = self.validate_file(file_path)

            if not result.is_valid:
                files_failed += 1

            all_errors.extend([f"{file_path.name}: {e}" for e in result.errors])
            all_warnings.extend([f"{file_path.name}: {w}" for w in result.warnings])
            total_valid += result.valid_samples
            total_invalid += result.invalid_samples
            total_checked += result.total_samples_checked

            if 'mean_length' in result.statistics:
                all_lengths.append(result.statistics['mean_length'])

        # Calculate overall statistics
        statistics = {
            'files_checked': len(files),
            'files_failed': files_failed,
            'overall_mean_length': float(np.mean(all_lengths)) if all_lengths else 0,
        }

        # Determine overall validity
        if self.strict:
            is_valid = total_invalid == 0 and files_failed == 0
        else:
            error_rate = total_invalid / total_checked if total_checked > 0 else 0
            is_valid = error_rate < 0.01 and files_failed < len(files) * 0.1

        return ValidationResult(
            is_valid=is_valid,
            total_samples_checked=total_checked,
            valid_samples=total_valid,
            invalid_samples=total_invalid,
            errors=all_errors[:50],  # Cap at 50 errors
            warnings=all_warnings[:20],
            statistics=statistics
        )


def validate_before_training(
    data_dir: Path,
    max_length: int,
    vocab_size: int = 100000,
    split: str = 'train',
    strict: bool = True,
) -> ValidationResult:
    """
    Convenience function to validate data before training.

    Args:
        data_dir: Directory containing data files
        max_length: Maximum sequence length
        vocab_size: Vocabulary size
        split: Data split to validate
        strict: Whether to use strict validation

    Returns:
        ValidationResult

    Raises:
        RuntimeError: If validation fails in strict mode
    """
    validator = DataValidator(
        max_length=max_length,
        vocab_size=vocab_size,
        strict=strict,
    )

    result = validator.validate_dataset(Path(data_dir), split)

    if not result.is_valid:
        logger.error(result.get_report())
        if strict:
            raise RuntimeError(
                f"Data validation failed: {len(result.errors)} errors. "
                f"Use skip_validation=True to bypass (not recommended)."
            )

    return result


__all__ = [
    'DataValidator',
    'ValidationResult',
    'DataStatistics',
    'validate_before_training',
]
