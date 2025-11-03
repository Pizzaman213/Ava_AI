"""
Logging Module

Centralized logging utilities for the Ava training framework.

Components:
- async_logging: Asynchronous logging with queue-based handling
- logging: Basic logging setup and configuration

This module provides:
- Non-blocking asynchronous logging for high-performance training
- Structured logging with proper formatting
- Log rotation and management
- Integration with distributed training environments
"""

from .async_logging import (
    AsyncLogger,
    AsyncLoggingHandler,
    setup_async_logging,
)

from .logging import (
    setup_logging,
    get_logger,
)

__all__ = [
    # Async logging
    "AsyncLogger",
    "AsyncLoggingHandler",
    "setup_async_logging",

    # Basic logging
    "setup_logging",
    "get_logger",
]
