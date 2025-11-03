"""
Utility Module

General utility functions for the Ava training framework.

Components:
- checkpoint: Checkpoint saving and loading
- gpu_memory: GPU memory management

Note: Logging utilities have been moved to Ava.logging module.
      This module re-exports them for backward compatibility.
"""

# Re-export logging utilities from new location for backward compatibility
from ..logging import (
    setup_logging,
    get_logger,
    AsyncLogger,
)

# Import from async_logging module to get the configs
try:
    from ..logging.async_logging import (
        AsyncLoggingConfig,
        AsyncLoggingContext,
        create_fast_logging_config,
        create_comprehensive_logging_config,
        create_minimal_logging_config
    )
except ImportError:
    # Fallback for compatibility
    AsyncLoggingConfig = None
    AsyncLoggingContext = None
    create_fast_logging_config = None
    create_comprehensive_logging_config = None
    create_minimal_logging_config = None

from .checkpoint import save_checkpoint, load_checkpoint
from .gpu_memory import (
    GPUMemoryManager, get_memory_manager,
    cleanup_gpu_memory, register_cleanup_handlers,
    get_memory_stats, monitor_memory
)

__all__ = [
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",

    # GPU Memory Management
    "GPUMemoryManager",
    "get_memory_manager",
    "cleanup_gpu_memory",
    "register_cleanup_handlers",
    "get_memory_stats",
    "monitor_memory",

    # Logging (re-exported from Ava.logging for backward compatibility)
    "setup_logging",
    "get_logger",
    "AsyncLogger",
    "AsyncLoggingConfig",
    "AsyncLoggingContext",
    "create_fast_logging_config",
    "create_comprehensive_logging_config",
    "create_minimal_logging_config"
]