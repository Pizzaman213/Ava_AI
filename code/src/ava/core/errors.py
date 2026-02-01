"""
Centralized error types for the Ava training pipeline.

This module defines a hierarchy of exception classes that provide clear,
consistent error handling across the codebase. Each error type includes
context about where and why the error occurred.

Usage:
    from ava.core.errors import DistributedSyncError, ConfigurationError

    # In distributed code
    if not dist.is_initialized():
        raise DistributedSyncError("Process group not initialized")

    # In config validation
    if learning_rate <= 0:
        raise ConfigurationError("learning_rate must be positive", field="training.learning_rate")
"""

from typing import Any, Optional


class AvaTrainingError(Exception):
    """
    Base exception class for all Ava training errors.

    All custom exceptions in the Ava pipeline should inherit from this class
    to enable consistent error handling and logging.

    Attributes:
        message: Human-readable error description
        context: Optional additional context about the error
    """

    def __init__(self, message: str, context: Optional[dict] = None):
        """
        Initialize the error.

        Args:
            message: Human-readable error description
            context: Optional dict with additional context (e.g., {'step': 100, 'rank': 0})
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


class DistributedSyncError(AvaTrainingError):
    """
    Error during distributed training synchronization.

    Raised when:
    - NCCL operations fail or timeout
    - Process group is not properly initialized
    - Ranks become desynchronized
    - All-reduce or barrier operations fail

    Example:
        >>> raise DistributedSyncError(
        ...     "NCCL all_reduce failed",
        ...     rank=0,
        ...     world_size=4,
        ...     operation="loss_sync"
        ... )
    """

    def __init__(
        self,
        message: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize distributed sync error.

        Args:
            message: Error description
            rank: The rank where the error occurred
            world_size: Total number of processes
            operation: The distributed operation that failed (e.g., "all_reduce", "barrier")
            **kwargs: Additional context
        """
        context = kwargs
        if rank is not None:
            context["rank"] = rank
        if world_size is not None:
            context["world_size"] = world_size
        if operation is not None:
            context["operation"] = operation
        super().__init__(message, context)


class MemoryError(AvaTrainingError):
    """
    Memory-related training error (OOM, allocation failure, etc.).

    Raised when:
    - CUDA out of memory occurs
    - Memory allocation fails
    - Memory fragmentation prevents allocation
    - Prefetcher memory pressure is too high

    Example:
        >>> raise MemoryError(
        ...     "CUDA OOM during forward pass",
        ...     batch_size=64,
        ...     allocated_gb=23.5,
        ...     total_gb=24.0
        ... )
    """

    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        allocated_gb: Optional[float] = None,
        total_gb: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize memory error.

        Args:
            message: Error description
            batch_size: Batch size when error occurred
            allocated_gb: GPU memory allocated in GB
            total_gb: Total GPU memory in GB
            **kwargs: Additional context
        """
        context = kwargs
        if batch_size is not None:
            context["batch_size"] = batch_size
        if allocated_gb is not None:
            context["allocated_gb"] = f"{allocated_gb:.2f}"
        if total_gb is not None:
            context["total_gb"] = f"{total_gb:.2f}"
        super().__init__(message, context)


class ConfigurationError(AvaTrainingError):
    """
    Configuration validation or loading error.

    Raised when:
    - Required config fields are missing
    - Config values have incorrect types
    - Config values are out of valid range
    - Config file cannot be loaded or parsed

    Example:
        >>> raise ConfigurationError(
        ...     "learning_rate must be positive",
        ...     field="training.learning_rate",
        ...     value=-0.001,
        ...     expected="float > 0"
        ... )
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize configuration error.

        Args:
            message: Error description
            field: Config field path (e.g., "training.learning_rate")
            value: The invalid value that was provided
            expected: Description of expected value
            **kwargs: Additional context
        """
        context = kwargs
        if field is not None:
            context["field"] = field
        if value is not None:
            context["value"] = repr(value)
        if expected is not None:
            context["expected"] = expected
        super().__init__(message, context)


class CheckpointError(AvaTrainingError):
    """
    Checkpoint save/load error.

    Raised when:
    - Checkpoint file is corrupted
    - Checkpoint is incompatible with current model
    - Checkpoint save fails
    - Required checkpoint not found

    Example:
        >>> raise CheckpointError(
        ...     "Checkpoint file corrupted",
        ...     path="/path/to/checkpoint.pt",
        ...     expected_keys=["model", "optimizer"],
        ...     found_keys=["model"]
        ... )
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        step: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize checkpoint error.

        Args:
            message: Error description
            path: Path to the checkpoint file
            step: Training step associated with the checkpoint
            **kwargs: Additional context
        """
        context = kwargs
        if path is not None:
            context["path"] = path
        if step is not None:
            context["step"] = step
        super().__init__(message, context)


class DataLoadingError(AvaTrainingError):
    """
    Data loading or preprocessing error.

    Raised when:
    - Data file is corrupted or malformed
    - Tokenization fails
    - Data validation fails
    - DataLoader workers crash

    Example:
        >>> raise DataLoadingError(
        ...     "Failed to tokenize sample",
        ...     file="train_001.jsonl",
        ...     sample_idx=42,
        ...     reason="sequence too long"
        ... )
    """

    def __init__(
        self,
        message: str,
        file: Optional[str] = None,
        sample_idx: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize data loading error.

        Args:
            message: Error description
            file: Path to the data file
            sample_idx: Index of the problematic sample
            **kwargs: Additional context
        """
        context = kwargs
        if file is not None:
            context["file"] = file
        if sample_idx is not None:
            context["sample_idx"] = sample_idx
        super().__init__(message, context)


class AggregateError(AvaTrainingError):
    """
    Aggregate error for batch operations with multiple failures.

    Raised when multiple errors occur during batch processing and need
    to be reported together.

    Example:
        >>> errors = [(ValueError("bad value"), "file1.txt"), (IOError("read fail"), "file2.txt")]
        >>> raise AggregateError(
        ...     "Multiple files failed to load",
        ...     errors=errors,
        ...     operation="data_loading"
        ... )
    """

    def __init__(
        self,
        message: str,
        errors: Optional[list] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize aggregate error.

        Args:
            message: Error description
            errors: List of (exception, context) tuples
            operation: The batch operation that failed
            **kwargs: Additional context
        """
        context = kwargs
        if operation is not None:
            context["operation"] = operation
        if errors:
            context["error_count"] = len(errors)
        self.errors = errors or []
        super().__init__(message, context)

    def _format_message(self) -> str:
        """Format the error message with aggregated errors."""
        base = super()._format_message()
        if not self.errors:
            return base

        lines = [base]
        for i, (err, ctx) in enumerate(self.errors[:5]):
            lines.append(f"  {i+1}. {ctx}: {err}")
        if len(self.errors) > 5:
            lines.append(f"  ... and {len(self.errors) - 5} more errors")
        return "\n".join(lines)


def handle_distributed_error(error: Exception, context: str) -> None:
    """
    Centralized handler for distributed errors.

    Converts common distributed errors to appropriate Ava error types
    and re-raises them with additional context.

    Args:
        error: The original exception
        context: Description of where the error occurred

    Raises:
        DistributedSyncError: For NCCL or distributed errors
        AvaTrainingError: For other errors with added context
    """
    error_str = str(error).upper()

    if "NCCL" in error_str:
        raise DistributedSyncError(
            f"NCCL error in {context}: {error}",
            operation=context
        ) from error

    if "TIMEOUT" in error_str:
        raise DistributedSyncError(
            f"Timeout in {context}: {error}",
            operation=context
        ) from error

    if "CUDA" in error_str and ("OOM" in error_str or "OUT OF MEMORY" in error_str):
        raise MemoryError(f"CUDA OOM in {context}: {error}") from error

    # Re-raise with context
    raise AvaTrainingError(f"Error in {context}: {error}") from error
