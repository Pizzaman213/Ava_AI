"""
Rank-Aware Error Handling System

Provides graceful error handling across distributed training ranks with failure detection,
recovery mechanisms, and coordinated error responses.
"""

import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import time
import threading
import traceback
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import signal
import sys

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorType(Enum):
    """Types of distributed training errors."""
    COMMUNICATION = "communication"
    MEMORY = "memory"
    COMPUTE = "compute"
    DATA = "data"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    TIMEOUT = "timeout"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Information about a distributed training error."""
    rank: int
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    traceback_str: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    retry_count: int = 0


@dataclass
class RankStatus:
    """Status information for a rank."""
    rank: int
    is_healthy: bool
    last_heartbeat: float
    error_count: int
    last_error: Optional[ErrorInfo] = None
    consecutive_failures: int = 0


class RankAwareErrorHandler:
    """
    Distributed training error handler that coordinates error responses across ranks.

    Features:
    - Rank failure detection and recovery
    - Error propagation and coordination
    - Graceful degradation strategies
    - Automatic retry mechanisms
    - Health monitoring and reporting
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        max_retries: int = 3,
        heartbeat_interval: float = 10.0,
        failure_timeout: float = 60.0,
        enable_recovery: bool = True
    ):
        """
        Initialize rank-aware error handler.

        Args:
            world_size: Total number of ranks
            rank: Current rank
            max_retries: Maximum retry attempts per error
            heartbeat_interval: Heartbeat interval in seconds
            failure_timeout: Timeout for considering rank failed
            enable_recovery: Enable automatic recovery mechanisms
        """
        self.world_size = world_size
        self.rank = rank
        self.max_retries = max_retries
        self.heartbeat_interval = heartbeat_interval
        self.failure_timeout = failure_timeout
        self.enable_recovery = enable_recovery

        # Rank status tracking
        self.rank_statuses: Dict[int, RankStatus] = {}
        self.failed_ranks: set = set()
        self.healthy_ranks: set = set(range(world_size))

        # Error handling state
        self.error_history: List[ErrorInfo] = []
        self.retry_counts: Dict[str, int] = {}
        self.recovery_callbacks: List[Callable] = []

        # Synchronization
        self._error_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Monitoring
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.last_heartbeat = time.time()

        # Initialize rank statuses
        self._initialize_rank_statuses()

        # Start monitoring if distributed
        if self.world_size > 1:
            self._start_monitoring()

    def _initialize_rank_statuses(self):
        """Initialize status tracking for all ranks."""
        current_time = time.time()
        for rank in range(self.world_size):
            self.rank_statuses[rank] = RankStatus(
                rank=rank,
                is_healthy=True,
                last_heartbeat=current_time,
                error_count=0
            )

    def _start_monitoring(self):
        """Start background monitoring threads."""
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"ErrorHandler-Heartbeat-{self.rank}"
        )
        self.heartbeat_thread.start()
        logger.info(f"Rank {self.rank}: Error handler monitoring started")

    def _heartbeat_loop(self):
        """Background heartbeat monitoring loop."""
        while not self._shutdown_event.wait(self.heartbeat_interval):
            try:
                self._send_heartbeat()
                self._check_rank_health()
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    def _send_heartbeat(self):
        """Send heartbeat to all ranks."""
        if not dist.is_initialized():
            return

        try:
            # Create heartbeat tensor with rank and timestamp
            heartbeat_data = torch.tensor(
                [self.rank, time.time()],
                dtype=torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Broadcast heartbeat to all ranks
            for target_rank in range(self.world_size):
                if target_rank != self.rank:
                    try:
                        dist.send(heartbeat_data, dst=target_rank)
                    except Exception as e:
                        logger.debug(f"Failed to send heartbeat to rank {target_rank}: {e}")

            # Update our own heartbeat
            self.last_heartbeat = time.time()
            with self._status_lock:
                self.rank_statuses[self.rank].last_heartbeat = self.last_heartbeat

        except Exception as e:
            logger.error(f"Heartbeat send failed: {e}")

    def _check_rank_health(self):
        """Check health of all ranks based on heartbeats."""
        current_time = time.time()

        with self._status_lock:
            for rank in range(self.world_size):
                if rank == self.rank:
                    continue

                status = self.rank_statuses[rank]
                time_since_heartbeat = current_time - status.last_heartbeat

                # Check if rank is considered failed
                if time_since_heartbeat > self.failure_timeout:
                    if status.is_healthy:
                        logger.warning(f"Rank {rank} appears to have failed (no heartbeat for {time_since_heartbeat:.1f}s)")
                        status.is_healthy = False
                        self.failed_ranks.add(rank)
                        self.healthy_ranks.discard(rank)
                        self._handle_rank_failure(rank)

    def _handle_rank_failure(self, failed_rank: int):
        """Handle the failure of a specific rank."""
        logger.error(f"Handling failure of rank {failed_rank}")

        # Create error info for the failure
        error_info = ErrorInfo(
            rank=failed_rank,
            timestamp=time.time(),
            error_type=ErrorType.COMMUNICATION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Rank {failed_rank} failed to respond (timeout)",
            recoverable=self.enable_recovery
        )

        self._record_error(error_info)

        # Trigger recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(failed_rank, error_info)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

        # Check if we can continue with remaining ranks
        if len(self.healthy_ranks) < (self.world_size // 2):
            logger.critical(f"Too many ranks failed ({len(self.failed_ranks)}/{self.world_size})")
            self._trigger_emergency_shutdown()

    def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown of distributed training."""
        logger.critical("Triggering emergency shutdown due to too many rank failures")

        # Set shutdown event
        self._shutdown_event.set()

        # Notify all recovery callbacks about shutdown
        shutdown_error = ErrorInfo(
            rank=self.rank,
            timestamp=time.time(),
            error_type=ErrorType.UNKNOWN,
            severity=ErrorSeverity.FATAL,
            message="Emergency shutdown due to excessive rank failures",
            recoverable=False
        )

        for callback in self.recovery_callbacks:
            try:
                callback(-1, shutdown_error)  # -1 indicates global failure
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")

    def handle_error(
        self,
        error: Exception,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> bool:
        """
        Handle an error with distributed coordination.

        Args:
            error: The exception that occurred
            error_type: Type of error
            severity: Severity level
            context: Additional context information
            recoverable: Whether the error is recoverable

        Returns:
            True if error was handled and training can continue, False otherwise
        """
        with self._error_lock:
            # Create error info
            error_info = ErrorInfo(
                rank=self.rank,
                timestamp=time.time(),
                error_type=error_type,
                severity=severity,
                message=str(error),
                traceback_str=traceback.format_exc(),
                context=context or {},
                recoverable=recoverable
            )

            # Record the error
            self._record_error(error_info)

            # Update rank status
            with self._status_lock:
                status = self.rank_statuses[self.rank]
                status.error_count += 1
                status.last_error = error_info

                if not recoverable or severity == ErrorSeverity.FATAL:
                    status.consecutive_failures += 1
                else:
                    status.consecutive_failures = 0

            # Log the error
            log_level = {
                ErrorSeverity.INFO: logging.INFO,
                ErrorSeverity.WARNING: logging.WARNING,
                ErrorSeverity.ERROR: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.FATAL: logging.CRITICAL
            }[severity]

            logger.log(log_level, f"Rank {self.rank} error: {error_info.message}")

            # Broadcast error to other ranks if critical
            if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                self._broadcast_error(error_info)

            # Determine if we should continue
            should_continue = self._should_continue_after_error(error_info)

            if not should_continue:
                logger.error(f"Rank {self.rank} cannot continue after error")
                self._mark_rank_as_failed(self.rank)

            return should_continue

    def _record_error(self, error_info: ErrorInfo):
        """Record error in history."""
        self.error_history.append(error_info)

        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        # Update retry count
        error_key = f"{error_info.error_type.value}:{error_info.message[:100]}"
        self.retry_counts[error_key] = self.retry_counts.get(error_key, 0) + 1

    def _broadcast_error(self, error_info: ErrorInfo):
        """Broadcast error information to other ranks."""
        if not dist.is_initialized():
            return

        try:
            # Serialize error info
            error_data = json.dumps(asdict(error_info)).encode('utf-8')
            error_tensor = torch.frombuffer(error_data, dtype=torch.uint8)

            # Broadcast error size first, then data
            size_tensor = torch.tensor([len(error_data)], dtype=torch.int64)

            for target_rank in range(self.world_size):
                if target_rank != self.rank:
                    try:
                        dist.send(size_tensor, dst=target_rank)
                        dist.send(error_tensor, dst=target_rank)
                    except Exception as e:
                        logger.debug(f"Failed to broadcast error to rank {target_rank}: {e}")

        except Exception as e:
            logger.error(f"Error broadcasting failed: {e}")

    def _should_continue_after_error(self, error_info: ErrorInfo) -> bool:
        """Determine if training should continue after an error."""
        # Fatal errors always stop training
        if error_info.severity == ErrorSeverity.FATAL:
            return False

        # Non-recoverable errors stop training
        if not error_info.recoverable:
            return False

        # Check retry limits
        error_key = f"{error_info.error_type.value}:{error_info.message[:100]}"
        retry_count = self.retry_counts.get(error_key, 0)

        if retry_count > self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded for error: {error_key}")
            return False

        # Check consecutive failures
        with self._status_lock:
            status = self.rank_statuses[self.rank]
            if status.consecutive_failures > 3:
                logger.error(f"Too many consecutive failures ({status.consecutive_failures})")
                return False

        # Check global health
        if len(self.healthy_ranks) < (self.world_size // 2):
            logger.error("Too many ranks are unhealthy")
            return False

        return True

    def _mark_rank_as_failed(self, rank: int):
        """Mark a rank as failed."""
        with self._status_lock:
            self.rank_statuses[rank].is_healthy = False
            self.failed_ranks.add(rank)
            self.healthy_ranks.discard(rank)

    @contextmanager
    def error_context(
        self,
        operation_name: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        recoverable: bool = True
    ):
        """
        Context manager for handling errors in distributed operations.

        Args:
            operation_name: Name of the operation
            error_type: Type of error expected
            recoverable: Whether errors in this context are recoverable
        """
        try:
            yield
        except Exception as e:
            # Determine severity based on error type
            severity = ErrorSeverity.ERROR
            if isinstance(e, (RuntimeError, MemoryError)):
                severity = ErrorSeverity.CRITICAL
            elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                severity = ErrorSeverity.FATAL

            # Handle the error
            context = {"operation": operation_name}
            handled = self.handle_error(e, error_type, severity, context, recoverable)

            if not handled:
                raise

    def register_recovery_callback(self, callback: Callable[[int, ErrorInfo], None]):
        """Register a callback to be called when rank failures occur."""
        self.recovery_callbacks.append(callback)

    def get_rank_status(self, rank: int) -> Optional[RankStatus]:
        """Get status information for a specific rank."""
        with self._status_lock:
            return self.rank_statuses.get(rank)

    def get_healthy_ranks(self) -> List[int]:
        """Get list of currently healthy ranks."""
        with self._status_lock:
            return list(self.healthy_ranks)

    def get_failed_ranks(self) -> List[int]:
        """Get list of currently failed ranks."""
        with self._status_lock:
            return list(self.failed_ranks)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors and rank health."""
        with self._error_lock, self._status_lock:
            # Count errors by type and severity
            error_counts = {}
            for error in self.error_history[-100:]:  # Last 100 errors
                key = f"{error.error_type.value}_{error.severity.value}"
                error_counts[key] = error_counts.get(key, 0) + 1

            return {
                'rank': self.rank,
                'world_size': self.world_size,
                'healthy_ranks': list(self.healthy_ranks),
                'failed_ranks': list(self.failed_ranks),
                'total_errors': len(self.error_history),
                'recent_errors': len([e for e in self.error_history if time.time() - e.timestamp < 300]),  # Last 5 minutes
                'error_counts': error_counts,
                'retry_counts': dict(self.retry_counts),
                'monitoring_active': not self._shutdown_event.is_set()
            }

    def should_abort_training(self) -> bool:
        """Check if training should be aborted due to too many failures."""
        with self._status_lock:
            failed_count = len(self.failed_ranks)
            healthy_count = len(self.healthy_ranks)

            # Abort if more than half the ranks have failed
            if failed_count >= (self.world_size // 2):
                return True

            # Abort if this rank has too many consecutive failures
            status = self.rank_statuses[self.rank]
            if status.consecutive_failures > 5:
                return True

            return False

    def cleanup(self):
        """Clean up error handler resources."""
        logger.info(f"Rank {self.rank}: Cleaning up error handler")

        # Stop monitoring
        self._shutdown_event.set()

        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)

        # Clear callbacks
        self.recovery_callbacks.clear()

        logger.info(f"Rank {self.rank}: Error handler cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Global error handler instance
_global_error_handler: Optional[RankAwareErrorHandler] = None


def get_error_handler(
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    **kwargs
) -> RankAwareErrorHandler:
    """Get or create global rank-aware error handler."""
    global _global_error_handler

    if _global_error_handler is None:
        # Auto-detect world_size and rank if not provided
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        _global_error_handler = RankAwareErrorHandler(world_size, rank, **kwargs)

    return _global_error_handler


def handle_distributed_error(
    error: Exception,
    error_type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None,
    recoverable: bool = True
) -> bool:
    """Convenience function to handle distributed errors."""
    handler = get_error_handler()
    return handler.handle_error(error, error_type, severity, context, recoverable)


@contextmanager
def distributed_error_context(
    operation_name: str,
    error_type: ErrorType = ErrorType.UNKNOWN,
    recoverable: bool = True
):
    """Convenience context manager for distributed error handling."""
    handler = get_error_handler()
    with handler.error_context(operation_name, error_type, recoverable):
        yield