"""
Hierarchical Logging System (Phase 7.1)

Enhanced logging with Debug/Info/Warning/Error levels, structured context,
and intelligent filtering for training observability.
"""

import logging
import time
import threading
import json
import traceback
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import contextvars


class LogLevel(Enum):
    """Enhanced log levels with numeric priorities."""
    DEBUG = (10, "DEBUG", "🔍")
    INFO = (20, "INFO", "ℹ️")
    WARNING = (30, "WARNING", "⚠️")
    ERROR = (40, "ERROR", "❌")
    CRITICAL = (50, "CRITICAL", "🚨")

    def __init__(self, level: int, display_name: str, emoji: str):
        self.level = level
        self.display_name = display_name
        self.emoji = emoji

    def __lt__(self, other):
        return self.level < other.level

    def __le__(self, other):
        return self.level <= other.level


@dataclass
class LogContext:
    """Contextual information for structured logging."""
    component: str = "unknown"
    operation: str = "unknown"
    step: Optional[int] = None
    epoch: Optional[int] = None
    batch_idx: Optional[int] = None
    sequence_length: Optional[int] = None
    batch_size: Optional[int] = None
    gpu_id: Optional[int] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry with enhanced information."""
    timestamp: float
    level: LogLevel
    message: str
    context: LogContext
    logger_name: str
    thread_id: str
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ContextManager:
    """Manages hierarchical logging contexts using contextvars."""

    def __init__(self):
        self.context_var: contextvars.ContextVar[LogContext] = contextvars.ContextVar('log_context')

    def get_context(self) -> LogContext:
        """Get current logging context or create default."""
        try:
            return self.context_var.get()
        except LookupError:
            return LogContext()

    def set_context(self, context: LogContext) -> contextvars.Token:
        """Set logging context and return token for cleanup."""
        return self.context_var.set(context)

    def update_context(self, **kwargs) -> contextvars.Token:
        """Update current context with new values."""
        current = self.get_context()
        # Create new context with updated values
        new_context = LogContext(
            component=kwargs.get('component', current.component),
            operation=kwargs.get('operation', current.operation),
            step=kwargs.get('step', current.step),
            epoch=kwargs.get('epoch', current.epoch),
            batch_idx=kwargs.get('batch_idx', current.batch_idx),
            sequence_length=kwargs.get('sequence_length', current.sequence_length),
            batch_size=kwargs.get('batch_size', current.batch_size),
            gpu_id=kwargs.get('gpu_id', current.gpu_id),
            thread_id=kwargs.get('thread_id', current.thread_id),
            session_id=kwargs.get('session_id', current.session_id),
            metadata={**current.metadata, **kwargs.get('metadata', {})}
        )
        return self.set_context(new_context)


class LogFilter:
    """Intelligent log filtering based on patterns and context."""

    def __init__(self):
        self.component_levels: Dict[str, LogLevel] = {}
        self.operation_levels: Dict[str, LogLevel] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'last_logged': 0,
            'interval': 60,  # seconds
            'max_count': 10
        })
        self.suppressed_patterns: List[str] = []
        self.boosted_patterns: List[str] = []

    def should_log(self, entry: LogEntry) -> bool:
        """Determine if log entry should be processed."""
        # Check component-specific level
        component_level = self.component_levels.get(entry.context.component)
        if component_level and entry.level < component_level:
            return False

        # Check operation-specific level
        operation_level = self.operation_levels.get(entry.context.operation)
        if operation_level and entry.level < operation_level:
            return False

        # Check rate limiting
        rate_key = f"{entry.context.component}:{entry.context.operation}:{entry.level.display_name}"
        if self._is_rate_limited(rate_key, entry.timestamp):
            return False

        # Check suppressed patterns
        if any(pattern in entry.message.lower() for pattern in self.suppressed_patterns):
            return False

        return True

    def _is_rate_limited(self, key: str, timestamp: float) -> bool:
        """Check if log entry is rate limited."""
        limit_info = self.rate_limits[key]

        # Reset counter if interval has passed
        if timestamp - limit_info['last_logged'] > limit_info['interval']:
            limit_info['count'] = 0
            limit_info['last_logged'] = timestamp

        limit_info['count'] += 1

        # Allow if under limit
        if limit_info['count'] <= limit_info['max_count']:
            return False

        return True

    def set_component_level(self, component: str, level: LogLevel):
        """Set minimum log level for a component."""
        self.component_levels[component] = level

    def set_operation_level(self, operation: str, level: LogLevel):
        """Set minimum log level for an operation."""
        self.operation_levels[operation] = level

    def suppress_pattern(self, pattern: str):
        """Suppress logs matching pattern."""
        self.suppressed_patterns.append(pattern.lower())

    def boost_pattern(self, pattern: str):
        """Boost priority of logs matching pattern."""
        self.boosted_patterns.append(pattern.lower())


class HierarchicalLogger:
    """
    Enhanced hierarchical logging system with intelligent filtering and context management.

    Features:
    - Structured logging with rich context
    - Intelligent filtering and rate limiting
    - Performance metrics integration
    - Async processing for non-blocking operation
    - Multiple output handlers (console, file, metrics)
    """

    def __init__(
        self,
        name: str = "training",
        min_level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        log_dir: str = "logs/observability",
        enable_async: bool = True,
        max_queue_size: int = 10000
    ):
        self.name = name
        self.min_level = min_level
        self.context_manager = ContextManager()
        self.filter = LogFilter()

        # Storage
        self.log_history = deque(maxlen=10000)
        self.log_lock = threading.Lock()

        # Async processing
        self.enable_async = enable_async
        if enable_async:
            import queue
            self.log_queue = queue.Queue(maxsize=max_queue_size)
            self.stop_event = threading.Event()
            self.processor_thread = threading.Thread(target=self._process_logs_async, daemon=True)
            self.processor_thread.start()

        # Output handlers
        self.handlers: List[Callable[[LogEntry], None]] = []

        if enable_console:
            self.handlers.append(self._console_handler)

        if enable_file:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.handlers.append(self._file_handler)

        # Statistics
        self.stats = defaultdict(int)
        self.start_time = time.time()

        # Performance tracking
        self.performance_samples = deque(maxlen=1000)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with optional exception."""
        self._log(LogLevel.ERROR, message, exception=exception, **kwargs)

    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Log critical message with optional exception."""
        self._log(LogLevel.CRITICAL, message, exception=exception, **kwargs)

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics with timing."""
        perf_metrics = {'duration': duration, **metrics}
        self.performance_samples.append({
            'operation': operation,
            'timestamp': time.time(),
            'metrics': perf_metrics
        })

        self._log(
            LogLevel.DEBUG,
            f"Performance: {operation} took {duration:.3f}s",
            performance_metrics=perf_metrics,
            operation=operation
        )

    def _log(self, level: LogLevel, message: str, exception: Exception = None, **kwargs):
        """Internal logging method."""
        if level < self.min_level:
            return

        # Get or create context
        context = self.context_manager.get_context()

        # Update context with kwargs
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.metadata[key] = value

        # Create log entry
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            context=context,
            logger_name=self.name,
            thread_id=threading.current_thread().name,
            exception=exception,
            stack_trace=traceback.format_exc() if exception else None,
            performance_metrics=kwargs.get('performance_metrics', {})
        )

        # Apply filters
        if not self.filter.should_log(entry):
            self.stats['filtered'] += 1
            return

        # Process entry
        if self.enable_async:
            try:
                self.log_queue.put_nowait(entry)
                self.stats['queued'] += 1
            except:
                self.stats['queue_full'] += 1
                # Fallback to synchronous processing
                self._process_entry(entry)
        else:
            self._process_entry(entry)

    def _process_logs_async(self):
        """Background thread for async log processing."""
        import queue

        while not self.stop_event.is_set():
            try:
                entry = self.log_queue.get(timeout=1.0)
                if entry is None:  # Sentinel for shutdown
                    break
                self._process_entry(entry)
            except queue.Empty:
                continue
            except Exception as e:
                self.stats['processing_errors'] += 1

    def _process_entry(self, entry: LogEntry):
        """Process a log entry through all handlers."""
        # Store in history
        with self.log_lock:
            self.log_history.append(entry)

        # Update statistics
        self.stats[f'level_{entry.level.display_name.lower()}'] += 1
        self.stats['total'] += 1

        # Process through handlers
        for handler in self.handlers:
            try:
                handler(entry)
            except Exception as e:
                self.stats['handler_errors'] += 1

    def _console_handler(self, entry: LogEntry):
        """Console output handler with rich formatting."""
        # Format timestamp
        timestamp = time.strftime('%H:%M:%S', time.localtime(entry.timestamp))

        # Format context
        context_parts = []
        if entry.context.component != "unknown":
            context_parts.append(f"[{entry.context.component}]")
        if entry.context.operation != "unknown":
            context_parts.append(f"({entry.context.operation})")
        if entry.context.step is not None:
            context_parts.append(f"step={entry.context.step}")

        context_str = " ".join(context_parts)

        # Format message
        level_str = f"{entry.level.emoji} {entry.level.display_name}"

        if context_str:
            message = f"{timestamp} | {level_str} | {context_str} | {entry.message}"
        else:
            message = f"{timestamp} | {level_str} | {entry.message}"

        # Add performance metrics if present
        if entry.performance_metrics:
            perf_str = " | ".join(f"{k}={v:.3f}" for k, v in entry.performance_metrics.items() if isinstance(v, (int, float)))
            if perf_str:
                message += f" | [{perf_str}]"

        # Print to console
        print(message)

        # Print exception details if present
        if entry.exception and entry.stack_trace:
            print(f"Exception: {entry.exception}")
            print(f"Stack trace:\n{entry.stack_trace}")

    def _file_handler(self, entry: LogEntry):
        """File output handler with JSON structured logging."""
        log_data = {
            'timestamp': entry.timestamp,
            'level': entry.level.display_name,
            'message': entry.message,
            'logger': entry.logger_name,
            'thread': entry.thread_id,
            'context': {
                'component': entry.context.component,
                'operation': entry.context.operation,
                'step': entry.context.step,
                'epoch': entry.context.epoch,
                'batch_idx': entry.context.batch_idx,
                'sequence_length': entry.context.sequence_length,
                'batch_size': entry.context.batch_size,
                'gpu_id': entry.context.gpu_id,
                'thread_id': entry.context.thread_id,
                'session_id': entry.context.session_id,
                'metadata': entry.context.metadata
            },
            'performance_metrics': entry.performance_metrics
        }

        if entry.exception:
            log_data['exception'] = str(entry.exception)
            log_data['stack_trace'] = entry.stack_trace

        # Write to daily log file
        date_str = time.strftime('%Y%m%d', time.localtime(entry.timestamp))
        log_file = self.log_dir / f"{self.name}_{date_str}.jsonl"

        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except Exception:
            self.stats['file_errors'] += 1

    def set_context(self, **kwargs) -> contextvars.Token:
        """Set logging context for current execution."""
        context = LogContext(**kwargs)
        return self.context_manager.set_context(context)

    def update_context(self, **kwargs) -> contextvars.Token:
        """Update current logging context."""
        return self.context_manager.update_context(**kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = time.time() - self.start_time
        queue_size = self.log_queue.qsize() if self.enable_async else 0

        stats = {
            'uptime_seconds': uptime,
            'queue_size': queue_size,
            'history_size': len(self.log_history),
            'performance_samples': len(self.performance_samples),
            **dict(self.stats)
        }

        # Calculate rates
        if uptime > 0:
            stats['logs_per_second'] = self.stats.get('total', 0) / uptime

        return stats

    def get_recent_logs(self, count: int = 100, level: LogLevel = None) -> List[LogEntry]:
        """Get recent log entries, optionally filtered by level."""
        with self.log_lock:
            logs = list(self.log_history)

        if level:
            logs = [log for log in logs if log.level >= level]

        return logs[-count:]

    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        samples = list(self.performance_samples)

        if operation:
            samples = [s for s in samples if s['operation'] == operation]

        if not samples:
            return {}

        # Calculate statistics
        durations = [s['metrics']['duration'] for s in samples]

        return {
            'operation': operation or 'all',
            'sample_count': len(samples),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations),
            'recent_samples': samples[-10:]
        }

    def shutdown(self):
        """Shutdown logger and cleanup resources."""
        if self.enable_async:
            self.stop_event.set()
            try:
                self.log_queue.put_nowait(None)  # Sentinel
            except:
                pass
            self.processor_thread.join(timeout=5.0)

        self.info("Logger shutdown completed")


# Context decorators and utilities
class log_context:
    """Decorator to set logging context for function execution."""

    def __init__(self, **context_kwargs):
        self.context_kwargs = context_kwargs

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            token = logger.update_context(**self.context_kwargs)
            try:
                return func(*args, **kwargs)
            finally:
                pass  # Context automatically restored
        return wrapper


class log_performance:
    """Decorator to automatically log function performance."""

    def __init__(self, operation_name: str = None, log_args: bool = False):
        self.operation_name = operation_name
        self.log_args = log_args

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            operation = self.operation_name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                metrics = {'success': True}
                if self.log_args and args:
                    metrics['arg_count'] = len(args)
                if self.log_args and kwargs:
                    metrics['kwarg_count'] = len(kwargs)

                logger.log_performance(operation, duration, **metrics)
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.log_performance(operation, duration, success=False, error=str(e))
                logger.error(f"Performance tracked function failed: {operation}", exception=e)
                raise

        return wrapper


# Global logger instance
_global_logger: Optional[HierarchicalLogger] = None


def get_logger() -> HierarchicalLogger:
    """Get the global hierarchical logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = HierarchicalLogger()
    return _global_logger


def setup_hierarchical_logging(
    name: str = "training",
    min_level: LogLevel = LogLevel.INFO,
    log_dir: str = "logs/observability",
    **kwargs
) -> HierarchicalLogger:
    """Setup and configure the global hierarchical logger."""
    global _global_logger
    _global_logger = HierarchicalLogger(
        name=name,
        min_level=min_level,
        log_dir=log_dir,
        **kwargs
    )
    return _global_logger