"""
Async Metrics Logger for non-blocking WandB/TensorBoard logging.

Eliminates GPU starvation from blocking network I/O during metric logging.
WandB log calls can block the training loop if done synchronously.

Key features:
- Background thread for non-blocking log() calls
- Batches metrics before sending (reduces network calls)
- Queue-based buffering with configurable depth
- flush() method for epoch boundaries
- Graceful shutdown with timeout

Usage:
    from ava.cuda.metrics import AsyncMetricsLogger

    logger = AsyncMetricsLogger(batch_size=10)
    logger.log({'loss': 0.5}, step=100)  # Non-blocking
    logger.flush()  # Wait for pending logs at epoch end
    logger.shutdown()  # Cleanup at training end
"""

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Single metric entry for the queue."""
    metrics: Dict[str, Any]
    step: int
    timestamp: float


class AsyncMetricsLogger:
    """
    Async wrapper for WandB/TensorBoard logging to eliminate blocking I/O.

    The training loop calls log() which returns immediately after queuing.
    A background thread batches and sends metrics without blocking training.

    Args:
        backend: 'wandb' or 'tensorboard' or custom callable
        batch_size: Number of log calls to batch before sending (default 10)
        flush_interval_seconds: Max time to wait before sending partial batch (default 5.0)
        max_queue_size: Maximum pending log calls before blocking (default 1000)

    Example:
        >>> logger = AsyncMetricsLogger(backend='wandb', batch_size=10)
        >>> for step in range(1000):
        ...     loss = train_step()
        ...     logger.log({'loss': loss}, step=step)  # Returns immediately
        >>> logger.flush()  # Wait for all pending at epoch end
        >>> logger.shutdown()  # Cleanup
    """

    def __init__(
        self,
        backend: str = 'wandb',
        batch_size: int = 10,
        flush_interval_seconds: float = 5.0,
        max_queue_size: int = 1000,
        custom_log_fn: Optional[Callable[[Dict[str, Any], int], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.max_queue_size = max_queue_size
        self.custom_log_fn = custom_log_fn
        self._error_callback = error_callback

        # Queue for pending log calls
        self._queue: Queue[Optional[MetricEntry]] = Queue(maxsize=max_queue_size)

        # Control flags
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._flushed_event = threading.Event()

        # Stats
        self._total_logged = 0
        self._total_batches = 0
        self._dropped_count = 0
        self._last_logged_step = -1  # Track last logged step to ensure monotonic ordering
        self._last_error: Optional[Exception] = None  # Track last error for diagnostics

        # Backend setup
        self._wandb = None
        self._tb_writer = None
        self._setup_backend()

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncMetricsLogger",
            daemon=True
        )
        self._worker_thread.start()

        logger.debug(f"AsyncMetricsLogger started: backend={backend}, batch_size={batch_size}")

    def _setup_backend(self) -> None:
        """Initialize the logging backend."""
        if self.custom_log_fn is not None:
            # Custom logging function provided
            return

        if self.backend == 'wandb':
            try:
                from ava.core.wandb_logger import WANDB_AVAILABLE
                if not WANDB_AVAILABLE:
                    logger.warning("WandB not installed, async logging disabled")
                    return

                import wandb
                if wandb.run is None:
                    logger.warning("WandB run not initialized, async logging disabled")
                    return

                self._wandb = wandb
            except ImportError:
                logger.warning("WandB logger module not available, async logging disabled")

        elif self.backend == 'tensorboard':
            try:
                from torch.utils.tensorboard import SummaryWriter
                # Note: SummaryWriter path should be set externally
                logger.info("TensorBoard backend selected (writer must be set externally)")
            except ImportError:
                logger.warning("TensorBoard not installed, async logging disabled")

    def set_tensorboard_writer(self, writer: Any) -> None:
        """Set TensorBoard SummaryWriter for tensorboard backend."""
        self._tb_writer = writer

    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        timeout: float = 0.001,
    ) -> bool:
        """
        Queue metrics for async logging (non-blocking).

        Args:
            metrics: Dictionary of metrics to log
            step: Training step number
            timeout: Max time to wait if queue is full (default 1ms)

        Returns:
            True if queued successfully, False if dropped
        """
        if self._stop_event.is_set():
            return False

        entry = MetricEntry(
            metrics=metrics.copy(),  # Copy to prevent mutation
            step=step,
            timestamp=time.time()
        )

        try:
            self._queue.put(entry, timeout=timeout)
            return True
        except Full:
            self._dropped_count += 1
            if self._dropped_count % 100 == 1:
                logger.warning(
                    f"AsyncMetricsLogger queue full, dropped {self._dropped_count} entries. "
                    f"Consider increasing max_queue_size or batch_size."
                )
            return False

    def update_last_step(self, step: int) -> None:
        """
        Update the last logged step (call after synchronous wandb.log calls).

        This ensures the async logger skips entries older than sync-logged steps.
        """
        if step > self._last_logged_step:
            self._last_logged_step = step

    def _worker_loop(self) -> None:
        """Background worker that batches and sends metrics."""
        batch: List[MetricEntry] = []
        last_send_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Try to get item with short timeout
                entry = self._queue.get(timeout=0.1)

                if entry is None:
                    # Shutdown signal
                    break

                batch.append(entry)

                # Check if should send batch
                should_send = (
                    len(batch) >= self.batch_size or
                    (time.time() - last_send_time) >= self.flush_interval_seconds
                )

                if should_send and batch:
                    self._send_batch(batch)
                    batch = []
                    last_send_time = time.time()

            except Empty:
                # No items, check if flush requested or time-based send needed
                if self._flush_event.is_set():
                    if batch:
                        self._send_batch(batch)
                        batch = []
                    self._flushed_event.set()
                    self._flush_event.clear()
                elif batch and (time.time() - last_send_time) >= self.flush_interval_seconds:
                    self._send_batch(batch)
                    batch = []
                    last_send_time = time.time()

        # Send any remaining on shutdown
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: List[MetricEntry]) -> None:
        """Send a batch of metrics to the backend."""
        if not batch:
            return

        # Sort batch by step to ensure monotonic ordering
        batch = sorted(batch, key=lambda e: e.step)

        try:
            logged_count = 0

            if self.custom_log_fn is not None:
                # Custom logging function
                for entry in batch:
                    if entry.step > self._last_logged_step:
                        self.custom_log_fn(entry.metrics, entry.step)
                        self._last_logged_step = entry.step
                        logged_count += 1

            elif self._wandb is not None:
                # WandB: log each entry, skip stale steps to avoid monotonic warning
                for entry in batch:
                    if entry.step > self._last_logged_step:
                        self._wandb.log(entry.metrics, step=entry.step)
                        self._last_logged_step = entry.step
                        logged_count += 1

            elif self._tb_writer is not None:
                # TensorBoard: log each metric
                for entry in batch:
                    if entry.step > self._last_logged_step:
                        for key, value in entry.metrics.items():
                            if isinstance(value, (int, float)):
                                self._tb_writer.add_scalar(key, value, entry.step)
                        self._last_logged_step = entry.step
                        logged_count += 1

            self._total_logged += logged_count
            self._total_batches += 1

        except Exception as e:
            logger.error(f"AsyncMetricsLogger send failed: {e}")
            self._last_error = e
            if self._error_callback is not None:
                try:
                    self._error_callback(e)
                except Exception:
                    pass  # Don't let callback errors propagate

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending metrics to be sent.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if flushed successfully, False if timeout
        """
        if self._stop_event.is_set():
            return True

        self._flushed_event.clear()
        self._flush_event.set()

        return self._flushed_event.wait(timeout=timeout)

    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the async logger gracefully.

        Args:
            timeout: Maximum time to wait for pending logs
        """
        logger.debug("AsyncMetricsLogger shutting down...")

        # Signal stop
        self._stop_event.set()

        # Send shutdown sentinel
        try:
            self._queue.put(None, timeout=1.0)
        except Full:
            pass

        # Wait for worker thread
        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning("AsyncMetricsLogger worker thread did not terminate cleanly")

        logger.info(
            f"AsyncMetricsLogger shutdown: logged {self._total_logged} entries "
            f"in {self._total_batches} batches, dropped {self._dropped_count}"
        )

    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        return {
            'total_logged': self._total_logged,
            'total_batches': self._total_batches,
            'dropped_count': self._dropped_count,
            'queue_size': self._queue.qsize(),
        }

    def get_last_error(self) -> Optional[Exception]:
        """Get the last error that occurred during async logging."""
        return self._last_error

    def clear_last_error(self) -> None:
        """Clear the last error."""
        self._last_error = None

    @property
    def is_active(self) -> bool:
        """Check if async logging is active."""
        return not self._stop_event.is_set() and self._worker_thread.is_alive()


# Global instance for convenience (optional)
_global_async_logger: Optional[AsyncMetricsLogger] = None


def get_async_logger(
    backend: str = 'wandb',
    batch_size: int = 10,
    **kwargs
) -> AsyncMetricsLogger:
    """Get or create global async logger instance."""
    global _global_async_logger
    if _global_async_logger is None or not _global_async_logger.is_active:
        _global_async_logger = AsyncMetricsLogger(
            backend=backend,
            batch_size=batch_size,
            **kwargs
        )
    return _global_async_logger


def shutdown_async_logger() -> None:
    """Shutdown global async logger if active."""
    global _global_async_logger
    if _global_async_logger is not None:
        _global_async_logger.shutdown()
        _global_async_logger = None
