"""
Backward compatibility module for utilities.

Re-exports from core package and provides stub functions for backward compatibility.
"""

import signal
import sys
import atexit
from typing import Callable, Optional

# Re-export from core
from .core import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    setup_colored_logging,
    get_logger,
)


def register_cleanup_handlers(
    cleanup_fn: Optional[Callable[[], None]] = None,
    handle_signals: bool = True,
) -> None:
    """
    Register cleanup handlers for graceful shutdown.

    Args:
        cleanup_fn: Optional cleanup function to call on shutdown
        handle_signals: Whether to handle SIGINT/SIGTERM signals
    """
    def _cleanup_handler(signum=None, frame=None):
        if cleanup_fn:
            try:
                cleanup_fn()
            except Exception:
                pass
        if signum is not None:
            sys.exit(0)

    # Register atexit handler
    if cleanup_fn:
        atexit.register(cleanup_fn)

    # Register signal handlers
    if handle_signals:
        try:
            signal.signal(signal.SIGINT, _cleanup_handler)
            signal.signal(signal.SIGTERM, _cleanup_handler)
        except (ValueError, OSError):
            # Signal handling may not be available in some environments
            pass


__all__ = [
    # From core
    'get_project_root',
    'get_data_dir',
    'get_outputs_dir',
    'setup_colored_logging',
    'get_logger',
    # Cleanup utilities
    'register_cleanup_handlers',
]
