"""
Async Metrics Logging Module.

Provides non-blocking metrics logging to eliminate GPU starvation
from blocking network I/O during training.

Features:
- Background thread for non-blocking log() calls
- Batches metrics before sending (reduces network calls)
- Queue-based buffering with configurable depth
- flush() method for epoch boundaries
- Graceful shutdown with timeout

Usage:
    from ava.logging.metrics import AsyncMetricsLogger

    logger = AsyncMetricsLogger(batch_size=10)
    logger.log({'loss': 0.5}, step=100)  # Non-blocking
    logger.flush()  # Wait for pending logs at epoch end
    logger.shutdown()  # Cleanup at training end
"""

from ava.logging.metrics.async_logger import (
    AsyncMetricsLogger,
    MetricEntry,
    get_async_logger,
    shutdown_async_logger,
)

__all__ = [
    'AsyncMetricsLogger',
    'MetricEntry',
    'get_async_logger',
    'shutdown_async_logger',
]
