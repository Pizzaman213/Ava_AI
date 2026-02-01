"""
Error tracking utilities for batch operations.

Provides tools to track errors during multi-step operations without
immediately failing, while still ensuring errors are surfaced.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Callable, Any, Tuple, Type, Union
import logging
import sys
import traceback


logger = logging.getLogger(__name__)

# Import error classes early for use in ErrorTracker
# This is at the top to avoid circular imports since errors.py has no dependencies
from ava.core.errors import (
    AggregateError,
    AvaTrainingError,
    ConfigurationError,
    CheckpointError,
    DataLoadingError,
    DistributedSyncError as DistributedTrainingError,
)


# Type alias for error records: (exception, context, exc_info or traceback_str)
# exc_info is stored initially, converted to string only when needed
ErrorRecord = Tuple[Exception, str, Union[tuple, str, None]]


@dataclass
class ErrorTracker:
    """
    Track errors during batch operations without immediate failure.

    Use this when processing multiple items where you want to continue
    processing remaining items even if some fail, but still want to
    raise an error if too many failures occur.

    Optimizations:
    - Uses deque with maxlen to bound memory usage
    - Defers traceback formatting until actually needed

    Example:
        tracker = ErrorTracker("DataLoading", max_errors=10, logger=logger)
        for file in files:
            try:
                process(file)
            except Exception as e:
                tracker.record(e, context=f"file={file}")
        tracker.raise_if_any()  # Raises AggregateError if any errors occurred
    """
    name: str
    max_errors: int = 10
    max_stored: int = 100  # Maximum errors to store (memory bound)
    logger: Optional[logging.Logger] = None
    raise_on_threshold: bool = True
    error_class: Type[AvaTrainingError] = AggregateError  # Exception class to raise
    # Use deque with maxlen for bounded memory; initialized in __post_init__
    _errors: Deque[ErrorRecord] = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        """Initialize deque with proper maxlen."""
        if not isinstance(self._errors, deque) or self._errors.maxlen != self.max_stored:
            self._errors = deque(maxlen=self.max_stored)

    @property
    def errors(self) -> List[Tuple[Exception, str, str]]:
        """Return errors with formatted tracebacks (for backwards compatibility)."""
        result = []
        for err, ctx, tb_info in self._errors:
            if isinstance(tb_info, str):
                tb_str = tb_info
            elif tb_info is not None:
                # Format traceback lazily
                tb_str = ''.join(traceback.format_exception(*tb_info))
            else:
                tb_str = ''
            result.append((err, ctx, tb_str))
        return result

    def record(self, error: Exception, context: str = "") -> None:
        """
        Record an error. Raises immediately if threshold exceeded.

        Args:
            error: The exception that occurred
            context: Additional context about where/why the error occurred
        """
        # Store exc_info tuple instead of formatted string (deferred formatting)
        exc_info = sys.exc_info()
        # Only store if we have actual exception info, otherwise store None
        if exc_info[0] is None:
            exc_info = None
        self._errors.append((error, context, exc_info))

        if self.logger:
            self.logger.warning(f"{self.name}: {context}: {error}")

        if self.raise_on_threshold and len(self._errors) >= self.max_errors:
            self._raise_aggregate_error()

    def _format_traceback(self, tb_info: Union[tuple, str, None]) -> str:
        """Format traceback info lazily."""
        if isinstance(tb_info, str):
            return tb_info
        elif tb_info is not None:
            return ''.join(traceback.format_exception(*tb_info))
        return ''

    def _raise_aggregate_error(self) -> None:
        """Raise an aggregated error from all recorded errors."""
        first_err, first_ctx, first_tb_info = self._errors[0]
        last_err, last_ctx, _ = self._errors[-1]

        # Format traceback only when raising
        first_tb = self._format_traceback(first_tb_info)

        # Build list of (error, context) tuples for AggregateError
        error_list = [(err, ctx) for err, ctx, _ in self._errors]

        error_msg = (
            f"{self.name}: Too many errors ({len(self._errors)}/{self.max_errors}).\n"
            f"First error ({first_ctx}): {first_err}\n"
            f"Last error ({last_ctx}): {last_err}\n"
            f"First traceback:\n{first_tb}"
        )
        raise AggregateError(error_msg, errors=error_list, operation=self.name)

    def raise_if_any(self, message: Optional[str] = None) -> None:
        """
        Raise an aggregated error if any errors were recorded.

        Args:
            message: Optional custom message prefix
        """
        if not self._errors:
            return

        first_err, first_ctx, first_tb_info = self._errors[0]

        # Format traceback only when raising
        first_tb = self._format_traceback(first_tb_info)

        # Build list of (error, context) tuples for AggregateError
        error_list = [(err, ctx) for err, ctx, _ in self._errors]

        prefix = message or f"{self.name}: {len(self._errors)} error(s) occurred"
        error_msg = (
            f"{prefix}.\n"
            f"First error ({first_ctx}): {first_err}\n"
            f"First traceback:\n{first_tb}"
        )
        raise AggregateError(error_msg, errors=error_list, operation=self.name)

    def raise_if_all_failed(self, total_items: int, message: Optional[str] = None) -> None:
        """
        Raise an error only if ALL items failed.

        Args:
            total_items: Total number of items that were processed
            message: Optional custom message prefix
        """
        if len(self._errors) >= total_items and total_items > 0:
            prefix = message or f"{self.name}: All {total_items} items failed"
            self.raise_if_any(prefix)

    @property
    def error_count(self) -> int:
        """Number of errors recorded."""
        return len(self._errors)

    @property
    def has_errors(self) -> bool:
        """Whether any errors have been recorded."""
        return len(self._errors) > 0

    def get_error_summary(self) -> str:
        """Get a summary of all recorded errors."""
        if not self._errors:
            return f"{self.name}: No errors"

        lines = [f"{self.name}: {len(self._errors)} error(s):"]
        # Iterate over first 5 errors without formatting tracebacks
        for i, (err, ctx, _) in enumerate(list(self._errors)[:5]):
            lines.append(f"  {i+1}. {ctx}: {err}")
        if len(self._errors) > 5:
            lines.append(f"  ... and {len(self._errors) - 5} more")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all recorded errors."""
        self._errors.clear()


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
