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
    AsyncLoggingConfig,
    AsyncLoggingContext,
    create_fast_logging_config,
    create_comprehensive_logging_config,
    create_minimal_logging_config,
)

from .logging import (
    setup_logging,
    get_logger,
)

__all__ = [
    # Async logging
    "AsyncLogger",
    "AsyncLoggingConfig",
    "AsyncLoggingContext",
    "create_fast_logging_config",
    "create_comprehensive_logging_config",
    "create_minimal_logging_config",

    # Basic logging
    "setup_logging",
    "get_logger",
]
