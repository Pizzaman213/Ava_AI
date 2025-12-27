"""
Error tracking utilities for batch operations.

Provides tools to track errors during multi-step operations without
immediately failing, while still ensuring errors are surfaced.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from contextlib import contextmanager
import logging
import traceback


logger = logging.getLogger(__name__)


@dataclass
class ErrorTracker:
    """
    Track errors during batch operations without immediate failure.

    Use this when processing multiple items where you want to continue
    processing remaining items even if some fail, but still want to
    raise an error if too many failures occur.

    Example:
        tracker = ErrorTracker("DataLoading", max_errors=10, logger=logger)
        for file in files:
            try:
                process(file)
            except Exception as e:
                tracker.record(e, context=f"file={file}")
        tracker.raise_if_any()  # Raises if any errors occurred
    """
    name: str
    max_errors: int = 10
    errors: List[tuple] = field(default_factory=list)  # (exception, context, traceback)
    logger: Optional[logging.Logger] = None
    raise_on_threshold: bool = True

    def record(self, error: Exception, context: str = "") -> None:
        """
        Record an error. Raises immediately if threshold exceeded.

        Args:
            error: The exception that occurred
            context: Additional context about where/why the error occurred
        """
        tb = traceback.format_exc()
        self.errors.append((error, context, tb))

        if self.logger:
            self.logger.warning(f"{self.name}: {context}: {error}")

        if self.raise_on_threshold and len(self.errors) >= self.max_errors:
            self._raise_aggregate_error()

    def _raise_aggregate_error(self) -> None:
        """Raise an aggregated error from all recorded errors."""
        first_err, first_ctx, first_tb = self.errors[0]
        last_err, last_ctx, _ = self.errors[-1]

        error_msg = (
            f"{self.name}: Too many errors ({len(self.errors)}/{self.max_errors}).\n"
            f"First error ({first_ctx}): {first_err}\n"
            f"Last error ({last_ctx}): {last_err}\n"
            f"First traceback:\n{first_tb}"
        )
        raise RuntimeError(error_msg)

    def raise_if_any(self, message: Optional[str] = None) -> None:
        """
        Raise an aggregated error if any errors were recorded.

        Args:
            message: Optional custom message prefix
        """
        if not self.errors:
            return

        first_err, first_ctx, first_tb = self.errors[0]

        prefix = message or f"{self.name}: {len(self.errors)} error(s) occurred"
        error_msg = (
            f"{prefix}.\n"
            f"First error ({first_ctx}): {first_err}\n"
            f"First traceback:\n{first_tb}"
        )
        raise RuntimeError(error_msg)

    def raise_if_all_failed(self, total_items: int, message: Optional[str] = None) -> None:
        """
        Raise an error only if ALL items failed.

        Args:
            total_items: Total number of items that were processed
            message: Optional custom message prefix
        """
        if len(self.errors) >= total_items and total_items > 0:
            prefix = message or f"{self.name}: All {total_items} items failed"
            self.raise_if_any(prefix)

    @property
    def error_count(self) -> int:
        """Number of errors recorded."""
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        """Whether any errors have been recorded."""
        return len(self.errors) > 0

    def get_error_summary(self) -> str:
        """Get a summary of all recorded errors."""
        if not self.errors:
            return f"{self.name}: No errors"

        lines = [f"{self.name}: {len(self.errors)} error(s):"]
        for i, (err, ctx, _) in enumerate(self.errors[:5]):  # Show first 5
            lines.append(f"  {i+1}. {ctx}: {err}")
        if len(self.errors) > 5:
            lines.append(f"  ... and {len(self.errors) - 5} more")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()


class DataLoadingError(Exception):
    """Raised when data loading fails critically."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass


class CheckpointError(Exception):
    """Raised when checkpoint save/load fails."""
    pass


class DistributedTrainingError(Exception):
    """Raised when distributed training setup fails."""
    pass


@contextmanager
def error_context(
    operation: str,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    on_error: Optional[Callable[[Exception], Any]] = None
):
    """
    Context manager for structured error handling with logging.

    Example:
        with error_context("Loading checkpoint", logger=logger):
            checkpoint = torch.load(path)

    Args:
        operation: Description of the operation for error messages
        logger: Logger to use for error messages
        reraise: Whether to re-raise the exception after logging
        on_error: Optional callback to execute on error
    """
    try:
        yield
    except Exception as e:
        error_msg = f"{operation} failed: {e}"
        if logger:
            logger.error(error_msg)
            logger.debug(f"Traceback:\n{traceback.format_exc()}")

        if on_error:
            try:
                on_error(e)
            except Exception as callback_err:
                if logger:
                    logger.warning(f"Error callback failed: {callback_err}")

        if reraise:
            raise


def log_and_raise(
    error: Exception,
    message: str,
    logger: Optional[logging.Logger] = None,
    error_class: type = RuntimeError
) -> None:
    """
    Log an error and raise a new exception with context.

    Args:
        error: The original exception
        message: Context message to include
        logger: Logger to use
        error_class: Exception class to raise
    """
    full_message = f"{message}: {error}"
    if logger:
        logger.error(full_message)
    raise error_class(full_message) from error
