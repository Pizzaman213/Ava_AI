"""
Utility Module

General utility functions for the Ava training framework.
"""

from .paths import get_project_root, resolve_path

# Colored logging utilities
from .colored_logging import (
    ColoredFormatter,
    CleanFormatter,
    Colors,
    setup_colored_logging,
    configure_root_logger,
    get_logger,
    supports_color,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
)

# Provide stub functions for removed modules
def register_cleanup_handlers(*args, **kwargs):
    """Stub for removed cleanup handler registration."""
    pass

def cleanup_gpu_memory(*args, **kwargs):
    """Stub for removed GPU memory cleanup."""
    pass

__all__ = [
    "get_project_root",
    "resolve_path",
    "get_logger",
    "register_cleanup_handlers",
    "cleanup_gpu_memory",
    # Colored logging
    "ColoredFormatter",
    "CleanFormatter",
    "Colors",
    "setup_colored_logging",
    "configure_root_logger",
    "supports_color",
    "print_header",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
]
