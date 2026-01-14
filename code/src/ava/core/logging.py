"""
Ava Pipeline Logging Utility.

Provides colorful, async-capable logging for training pipelines with:
- ANSI color support with auto-detection
- Level-based coloring with icons
- Async/parallel file logging with worker threads (non-blocking I/O)
- Pipeline-aware directory structure (integrates with RunManager)
- Metric highlighting (loss, LR, memory, etc.)
- Beautiful section headers and status indicators
- Distributed training support (rank-aware logging)

Usage:
    from ava.core.logging import setup_pipeline_logging, PipelineLogger

    # For pipeline training (integrates with RunManager):
    logger = setup_pipeline_logging(
        run_dir=run_manager.run_dir,
        rank=0,
        world_size=4,
        async_file_logging=True
    )
    logger.info("Training started")

    # Legacy usage (still works):
    from ava.core.logging import setup_colored_logging, print_header
    logger = setup_colored_logging('training', level=logging.INFO)

    print_header("Training Configuration")
    print_success("Model loaded successfully")
"""

import atexit
import logging
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Pre-compiled regex patterns for metric highlighting (performance optimization)
_RE_LOSS = re.compile(r'(Loss[:\s=]+)(\d+\.\d+)', re.IGNORECASE)
_RE_LR = re.compile(r'(LR[:\s=]+)(\d+\.\d+e[+-]?\d+)', re.IGNORECASE)
_RE_PARAMS = re.compile(r'(\d+[\d,]*\.?\d*[MBK]?\s*params?)', re.IGNORECASE)
_RE_DEVICE = re.compile(r'(cuda:\d+|GPU\s*\d+)', re.IGNORECASE)
_RE_MEMORY = re.compile(r'(\d+\.?\d*\s*[GM]B)', re.IGNORECASE)
_RE_PERCENT = re.compile(r'(\d+\.?\d*%)')
_RE_STEP_EPOCH = re.compile(r'(step|epoch)\s*(\d+)', re.IGNORECASE)
_RE_THROUGHPUT = re.compile(r'(\d+[\d,]*\.?\d*)\s*(samples?|tokens?)/s(ec)?', re.IGNORECASE)


class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Foreground colors (bright versions)
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    # Standard foreground colors
    DARK_RED = '\033[31m'
    DARK_GREEN = '\033[32m'
    DARK_YELLOW = '\033[33m'
    DARK_BLUE = '\033[34m'
    DARK_MAGENTA = '\033[35m'
    DARK_CYAN = '\033[36m'

    # Orange (256-color mode)
    ORANGE = '\033[38;5;208m'
    LIGHT_BLUE = '\033[38;5;117m'
    PINK = '\033[38;5;213m'
    LIME = '\033[38;5;118m'
    GOLD = '\033[38;5;220m'
    PURPLE = '\033[38;5;141m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'


class Icons:
    """Text-based icons for log messages (no emojis)."""
    # Status icons
    SUCCESS = '[OK]'
    ERROR = '[ERR]'
    WARNING = '[WARN]'
    INFO = '[INFO]'
    DEBUG = '[DBG]'

    # Progress icons
    ROCKET = '[>>]'
    GEAR = '[*]'
    CLOCK = '[T]'
    CHECK = '[+]'
    CROSS = '[X]'
    ARROW_RIGHT = '->'
    ARROW_DOWN = 'v'
    BULLET = '*'
    STAR = '*'
    SPARKLE = '*'

    # Training icons
    BRAIN = '[MODEL]'
    GPU = '[GPU]'
    MEMORY = '[MEM]'
    CHART = '[STATS]'
    FOLDER = '[DIR]'
    FILE = '[FILE]'
    DATA = '[DATA]'
    LIGHTNING = '[FAST]'
    FIRE = '[HOT]'
    TARGET = '[AIM]'
    HOURGLASS = '[WAIT]'
    SAVE = '[SAVE]'
    LOAD = '[LOAD]'
    NETWORK = '[NET]'
    EXPERIMENT = '[EXP]'


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check if stdout is a TTY
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False

    # Check for common environment variables that indicate color support
    import os
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True

    # Check TERM environment variable
    term = os.environ.get('TERM', '')
    if term in ('dumb', ''):
        return False

    return True


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log output based on level.

    Features:
    - Level-based coloring with icons (DEBUG=gray, INFO=cyan, WARNING=yellow, ERROR=red)
    - Clean timestamp format (HH:MM:SS only)
    - Auto-detection of terminal color support
    - Highlighting for important metrics (loss, accuracy, etc.)
    - Smart detection of message types for appropriate styling
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    LEVEL_ICONS = {
        logging.DEBUG: Icons.DEBUG,
        logging.INFO: Icons.BULLET,
        logging.WARNING: Icons.WARNING,
        logging.ERROR: Icons.ERROR,
        logging.CRITICAL: Icons.CROSS,
    }

    LEVEL_NAMES = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARN',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRIT',
    }

    def __init__(self, use_colors: Optional[bool] = None, show_level: bool = False, show_icons: bool = True):
        """
        Initialize the colored formatter.

        Args:
            use_colors: Force colors on/off. None for auto-detection.
            show_level: Whether to show log level in output.
            show_icons: Whether to show icons (default True).
        """
        super().__init__()
        self.use_colors = use_colors if use_colors is not None else supports_color()
        self.show_level = show_level
        self.show_icons = show_icons

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get timestamp (HH:MM:SS only for cleaner output)
        from datetime import datetime
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Get the message
        message = record.getMessage()

        if self.use_colors:
            # Get color for this level
            level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            level_icon = self.LEVEL_ICONS.get(record.levelno, Icons.BULLET) if self.show_icons else ''

            # Format timestamp with subtle styling
            colored_time = f"{Colors.DIM}{Colors.DARK_CYAN}[{timestamp}]{Colors.RESET}"

            # Format level if requested
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_color}{level_name}{Colors.RESET}"

            # Detect and style different message types
            message = self._style_message(message, record.levelno, level_icon)

            return f"{colored_time}{level_str} {message}"
        else:
            # Plain format without colors
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_name}"
            return f"[{timestamp}]{level_str} {message}"

    def _style_message(self, message: str, levelno: int, icon: str) -> str:
        """Apply appropriate styling based on message content and level."""
        stripped = message.strip()

        # Section headers (=== or ---)
        if stripped.startswith('=' * 10) or stripped.startswith('-' * 10):
            return f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}"

        # Major section titles (all caps)
        if stripped.isupper() and len(stripped) > 5 and ' ' in stripped:
            return f"{Colors.BOLD}{Colors.CYAN}{Icons.SPARKLE} {message}{Colors.RESET}"

        # Warning messages
        if levelno == logging.WARNING:
            return f"{Colors.YELLOW}{Icons.WARNING} {message}{Colors.RESET}"

        # Error messages
        if levelno >= logging.ERROR:
            return f"{Colors.RED}{Icons.ERROR} {message}{Colors.RESET}"

        # Success patterns
        if any(word in stripped.lower() for word in ['success', 'complete', 'finished', 'done', 'saved', 'loaded']):
            return f"{Colors.GREEN}{Icons.SUCCESS} {self._highlight_metrics(message)}{Colors.RESET}"

        # Training step patterns (Step XXX, Epoch XXX)
        if stripped.startswith(('Step ', 'Epoch ', '[Step', '[Epoch')):
            return f"{Colors.LIGHT_BLUE}{icon} {self._highlight_metrics(message)}{Colors.RESET}"

        # Configuration/settings patterns
        if any(word in stripped.lower() for word in ['config', 'setting', 'parameter', 'option']):
            return f"{Colors.PURPLE}{Icons.GEAR} {self._highlight_metrics(message)}{Colors.RESET}"

        # Data loading patterns
        if any(word in stripped.lower() for word in ['loading', 'dataset', 'dataloader', 'batch', 'samples']):
            return f"{Colors.ORANGE}{Icons.DATA} {self._highlight_metrics(message)}{Colors.RESET}"

        # GPU/Memory patterns
        if any(word in stripped.lower() for word in ['gpu', 'cuda', 'memory', 'vram']):
            return f"{Colors.MAGENTA}{Icons.GPU} {self._highlight_metrics(message)}{Colors.RESET}"

        # Model patterns
        if any(word in stripped.lower() for word in ['model', 'param', 'layer', 'weight']):
            return f"{Colors.PINK}{Icons.BRAIN} {self._highlight_metrics(message)}{Colors.RESET}"

        # Default: apply metric highlighting with icon
        return f"{icon} {self._highlight_metrics(message)}"

    def _highlight_metrics(self, message: str) -> str:
        """Highlight important metrics in the message using pre-compiled patterns."""
        # Use pre-compiled module-level regex patterns for performance
        message = _RE_LOSS.sub(f'\\1{Colors.GOLD}{Colors.BOLD}\\2{Colors.RESET}', message)
        message = _RE_LR.sub(f'\\1{Colors.PURPLE}{Colors.BOLD}\\2{Colors.RESET}', message)
        message = _RE_PARAMS.sub(f'{Colors.LIME}{Colors.BOLD}\\1{Colors.RESET}', message)
        message = _RE_DEVICE.sub(f'{Colors.MAGENTA}{Colors.BOLD}\\1{Colors.RESET}', message)
        message = _RE_MEMORY.sub(f'{Colors.ORANGE}\\1{Colors.RESET}', message)
        message = _RE_PERCENT.sub(f'{Colors.CYAN}{Colors.BOLD}\\1{Colors.RESET}', message)
        message = _RE_STEP_EPOCH.sub(f'\\1 {Colors.LIGHT_BLUE}{Colors.BOLD}\\2{Colors.RESET}', message)
        message = _RE_THROUGHPUT.sub(f'{Colors.GREEN}{Colors.BOLD}\\1{Colors.RESET} \\2/s\\3', message)
        return message


class CleanFormatter(logging.Formatter):
    """
    Simple clean formatter without colors but with improved formatting.
    Used for file logging or when colors are disabled.
    """

    def __init__(self, include_date: bool = True):
        """
        Initialize the clean formatter.

        Args:
            include_date: Whether to include date in timestamp.
        """
        if include_date:
            fmt = '%(asctime)s | %(levelname)s | %(message)s'
            datefmt = '%Y-%m-%d %H:%M:%S'
        else:
            fmt = '[%(asctime)s] %(message)s'
            datefmt = '%H:%M:%S'
        super().__init__(fmt=fmt, datefmt=datefmt)


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Logging handler that uses tqdm.write() when tqdm progress bars are active.

    This prevents log messages from breaking tqdm progress bar display.
    Falls back to standard StreamHandler behavior when tqdm is not available.

    Example:
        handler = TqdmLoggingHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using tqdm.write() if available."""
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except ImportError:
                # tqdm not available, use standard stream write
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


# =============================================================================
# Async/Parallel Logging Handlers
# =============================================================================

class AsyncLoggingHandler(logging.Handler):
    """
    Async file logging handler that writes to files in a background thread.

    Uses a queue-based approach to decouple log emission from file I/O,
    preventing file writes from blocking the training loop.

    Features:
    - Non-blocking log emission (puts to queue, returns immediately)
    - Dedicated worker thread for file writes
    - Automatic batching of writes for efficiency
    - Graceful shutdown with pending log flush
    - Thread-safe operation

    Example:
        handler = AsyncLoggingHandler(
            log_file='logs/training.log',
            flush_interval=1.0,  # Flush every 1 second
            max_queue_size=10000
        )
        handler.setFormatter(CleanFormatter())
        logger.addHandler(handler)
    """

    def __init__(
        self,
        log_file: Union[str, Path],
        flush_interval: float = 1.0,
        max_queue_size: int = 10000,
        level: int = logging.DEBUG
    ):
        """
        Initialize the async logging handler.

        Args:
            log_file: Path to log file.
            flush_interval: Seconds between flushes (default 1.0).
            max_queue_size: Max pending log messages (default 10000).
            level: Logging level for this handler.
        """
        super().__init__(level=level)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.flush_interval = flush_interval
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._file_handle: Optional[Any] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Start worker thread
        self._start_worker()

        # Register shutdown handler
        atexit.register(self.close)

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._file_handle = open(self.log_file, 'a', encoding='utf-8')
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"AsyncLogger-{self.log_file.name}",
            daemon=True
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker that processes the log queue."""
        batch: List[str] = []
        last_flush = time.time()

        while not self._shutdown.is_set():
            try:
                # Wait for messages with timeout
                try:
                    record = self._queue.get(timeout=0.1)
                    if record is not None:
                        batch.append(record)
                except queue.Empty:
                    pass

                # Drain remaining messages (non-blocking)
                while True:
                    try:
                        record = self._queue.get_nowait()
                        if record is not None:
                            batch.append(record)
                    except queue.Empty:
                        break

                # Flush if we have messages and enough time has passed
                now = time.time()
                if batch and (now - last_flush >= self.flush_interval or len(batch) >= 100):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                # Log error to stderr but don't crash worker
                print(f"AsyncLoggingHandler worker error: {e}", file=sys.stderr)

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[str]) -> None:
        """Write a batch of log messages to file."""
        if not batch or self._file_handle is None:
            return

        try:
            with self._lock:
                for msg in batch:
                    self._file_handle.write(msg + '\n')
                self._file_handle.flush()
        except Exception as e:
            print(f"AsyncLoggingHandler flush error: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record (non-blocking, adds to queue)."""
        if self._shutdown.is_set():
            return

        try:
            msg = self.format(record)
            # Non-blocking put - drop if queue is full
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                # Queue full, drop the message (don't block training)
                pass
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush pending messages (blocks until queue is empty)."""
        # Wait for queue to drain
        timeout = 5.0
        start = time.time()
        while not self._queue.empty() and (time.time() - start) < timeout:
            pass

    def close(self) -> None:
        """Shutdown the handler gracefully."""
        if self._shutdown.is_set():
            return

        self._shutdown.set()

        # Wait for worker to finish
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        # Close file handle
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None

        super().close()


class ParallelLoggerPool:
    """
    Pool of worker threads for parallel logging operations.

    Useful for heavy logging tasks like:
    - Writing large metric batches
    - Saving generation samples
    - Creating visualizations

    Example:
        pool = ParallelLoggerPool(num_workers=4)
        pool.submit(save_metrics, large_metrics_dict)
        pool.submit(save_samples, generated_texts)
        # Later...
        pool.shutdown()
    """

    _instance: Optional['ParallelLoggerPool'] = None
    _lock = threading.Lock()

    def __init__(self, num_workers: int = 4):
        """
        Initialize the worker pool.

        Args:
            num_workers: Number of worker threads (default 4).
        """
        self.num_workers = num_workers
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="LoggerPool"
        )
        self._pending_futures: List[Any] = []

    @classmethod
    def get_instance(cls, num_workers: int = 4) -> 'ParallelLoggerPool':
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(num_workers=num_workers)
        return cls._instance

    def submit(self, fn: Callable, *args, **kwargs) -> None:
        """
        Submit a logging task to the pool.

        Args:
            fn: Function to execute.
            *args, **kwargs: Arguments to pass to fn.
        """
        future = self._executor.submit(fn, *args, **kwargs)
        self._pending_futures.append(future)

        # Clean up completed futures periodically
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending tasks to complete."""
        from concurrent.futures import wait
        if self._pending_futures:
            wait(self._pending_futures, timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pool."""
        self._executor.shutdown(wait=wait)

    @classmethod
    def shutdown_global(cls) -> None:
        """Shutdown the global singleton instance."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None


def setup_colored_logging(
    logger_name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_colors: Optional[bool] = None,
    show_level: bool = False,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file logging.

    Args:
        logger_name: Name for the logger.
        level: Logging level (default: INFO).
        log_file: Optional path to log file.
        use_colors: Force colors on/off. None for auto-detection.
        show_level: Whether to show log level in output.
        propagate: Whether to propagate to parent loggers (default: False to avoid duplicates).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = propagate  # Prevent duplicate logs

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors, show_level=show_level))
    logger.addHandler(console_handler)

    # File handler (plain format, no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(CleanFormatter(include_date=True))
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(level: int = logging.WARNING):
    """
    Configure the root logger to reduce noise from other modules.

    Args:
        level: Minimum level for root logger (default: WARNING).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers on root
    root_logger.handlers.clear()

    # Add a simple handler for root that won't duplicate our custom logs
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    root_logger.addHandler(handler)


def setup_ava_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_colors: Optional[bool] = None,
    use_tqdm_handler: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> logging.Logger:
    """
    Configure unified logging for the Ava framework.

    This is the recommended way to set up logging in Ava entry points.
    It configures the 'ava' logger hierarchy with colored output,
    tqdm compatibility, and optional file logging.

    Args:
        level: Logging level for ava loggers (default: INFO).
        log_file: Optional path for file logging.
        use_colors: Force colors on/off. None for auto-detection.
        use_tqdm_handler: Use tqdm-compatible handler (default: True).
        rank: Process rank for distributed training (default: 0).
        world_size: Total number of processes (default: 1).

    Returns:
        The configured 'ava' logger.

    Example:
        # At the start of your training script:
        from ava.core.logging import setup_ava_logging

        setup_ava_logging(level=logging.INFO)

        # For distributed training:
        setup_ava_logging(level=logging.INFO, rank=local_rank)

        # With file logging:
        setup_ava_logging(log_file='training.log')
    """
    # 1. Configure root logger to suppress dependency noise
    configure_root_logger(level=logging.WARNING)

    # 2. Get or create the 'ava' logger
    ava_logger = logging.getLogger('ava')

    # Only rank 0 logs at requested level; others log warnings only
    ava_logger.setLevel(level if rank == 0 else logging.WARNING)

    # Clear existing handlers to avoid duplicates
    ava_logger.handlers.clear()

    # Don't propagate to root (we handle everything here)
    ava_logger.propagate = False

    # 3. Console handler (only for rank 0 in distributed training)
    if rank == 0:
        if use_tqdm_handler:
            console_handler = TqdmLoggingHandler(sys.stdout)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            ColoredFormatter(use_colors=use_colors, show_level=False, show_icons=True)
        )
        ava_logger.addHandler(console_handler)

    # 4. File handler (plain format, no colors) - all ranks can write
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_handler.setFormatter(CleanFormatter(include_date=True))
        ava_logger.addHandler(file_handler)

    return ava_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger by name. Convenience function for module-level logging.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# =============================================================================
# Pipeline-Integrated Logging
# =============================================================================

class PipelineLogger:
    """
    Pipeline-aware logger that integrates with RunManager directory structure.

    Uses the same file structure as RunManager:
    - logs/training.log - Main training output
    - logs/evaluation.log - Validation/eval output
    - logs/errors.log - Errors only
    - logs/debug.log - Verbose debug output

    Features:
    - Async file logging for non-blocking I/O
    - Colored console output with tqdm compatibility
    - Rank-aware logging for distributed training
    - Pipeline phase tracking
    - Structured metric logging

    Example:
        # Create from RunManager
        logger = PipelineLogger.from_run_manager(run_manager, rank=0)

        # Log training events
        logger.log_phase_start("Model Building")
        logger.log_calibration(batch_size=64, memory_pct=0.72)
        logger.log_training_step(step=100, loss=2.5, lr=1e-4)
        logger.log_phase_end("Model Building")
    """

    # Log file names (same as RunManager)
    LOG_FILES = {
        'training': 'training.log',
        'evaluation': 'evaluation.log',
        'errors': 'errors.log',
        'debug': 'debug.log',
    }

    def __init__(
        self,
        run_dir: Union[str, Path],
        rank: int = 0,
        world_size: int = 1,
        async_file_logging: bool = True,
        num_workers: int = 4,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_tqdm_handler: bool = True,
    ):
        """
        Initialize the pipeline logger.

        Args:
            run_dir: Run directory (from RunManager.run_dir).
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            async_file_logging: Use async handlers for file logging.
            num_workers: Number of parallel worker threads.
            console_level: Console logging level.
            file_level: File logging level.
            use_tqdm_handler: Use tqdm-compatible console handler.
        """
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.async_file_logging = async_file_logging

        # Create loggers dictionary
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: List[logging.Handler] = []

        # Setup each logger type
        for log_type, log_file in self.LOG_FILES.items():
            self._setup_logger(
                log_type=log_type,
                log_file=self.logs_dir / log_file,
                console=(log_type in ['training', 'errors']),
                console_level=console_level,
                file_level=file_level,
                use_tqdm=use_tqdm_handler,
            )

        # Initialize worker pool for heavy logging tasks
        if num_workers > 0:
            self._worker_pool = ParallelLoggerPool.get_instance(num_workers)
        else:
            self._worker_pool = None

        # Phase tracking
        self._current_phase: Optional[str] = None
        self._phase_start_time: Optional[float] = None

    def _setup_logger(
        self,
        log_type: str,
        log_file: Path,
        console: bool = False,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_tqdm: bool = True,
    ) -> None:
        """Setup an individual logger."""
        logger_name = f"ava.pipeline.{log_type}.{self.run_dir.name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(file_level)
        logger.handlers.clear()
        logger.propagate = False

        # File handler (async or sync)
        if self.async_file_logging:
            file_handler = AsyncLoggingHandler(
                log_file=log_file,
                flush_interval=1.0,
                level=file_level,
            )
        else:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)

        file_handler.setFormatter(CleanFormatter(include_date=True))
        logger.addHandler(file_handler)
        self._handlers.append(file_handler)

        # Console handler (only for main process and specific log types)
        if console and self.is_main_process:
            if use_tqdm:
                console_handler = TqdmLoggingHandler(sys.stdout)
            else:
                console_handler = logging.StreamHandler(sys.stdout)

            console_handler.setLevel(console_level)
            console_handler.setFormatter(ColoredFormatter(show_level=False, show_icons=True))
            logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        self._loggers[log_type] = logger

    @classmethod
    def from_run_manager(
        cls,
        run_manager: Any,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ) -> 'PipelineLogger':
        """
        Create a PipelineLogger from a RunManager instance.

        Args:
            run_manager: RunManager instance with run_dir attribute.
            rank: Process rank.
            world_size: Total processes.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Configured PipelineLogger.
        """
        return cls(
            run_dir=run_manager.run_dir,
            rank=rank,
            world_size=world_size,
            **kwargs
        )

    # =========================================================================
    # Core Logging Methods
    # =========================================================================

    def info(self, message: str) -> None:
        """Log an info message to training log."""
        self._loggers['training'].info(message)

    def warning(self, message: str) -> None:
        """Log a warning message to training and errors log."""
        self._loggers['training'].warning(message)
        self._loggers['errors'].warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message to training and errors log."""
        self._loggers['training'].error(message, exc_info=exc_info)
        self._loggers['errors'].error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """Log a debug message to debug log."""
        self._loggers['debug'].debug(message)

    def eval(self, message: str) -> None:
        """Log to evaluation log."""
        self._loggers['evaluation'].info(message)

    # =========================================================================
    # Pipeline Phase Logging
    # =========================================================================

    def log_phase_start(self, phase_name: str, phase_num: Optional[int] = None) -> None:
        """Log the start of a pipeline phase."""
        self._current_phase = phase_name
        self._phase_start_time = time.time()

        if phase_num is not None:
            if self.is_main_process:
                print_phase(phase_num, phase_name)
            self._loggers['training'].info(f"Phase {phase_num}: {phase_name} started")
        else:
            if self.is_main_process:
                print_subheader(f"{Icons.GEAR} {phase_name}")
            self._loggers['training'].info(f"Phase: {phase_name} started")

    def log_phase_end(self, phase_name: Optional[str] = None, status: str = "completed") -> None:
        """Log the end of a pipeline phase."""
        phase = phase_name or self._current_phase
        duration = ""
        if self._phase_start_time:
            elapsed = time.time() - self._phase_start_time
            duration = f" ({elapsed:.2f}s)"

        if self.is_main_process:
            print_success(f"{phase} {status}{duration}")
        self._loggers['training'].info(f"Phase: {phase} {status}{duration}")

        self._current_phase = None
        self._phase_start_time = None

    # =========================================================================
    # Calibration Logging
    # =========================================================================

    def log_calibration_start(self, config: Dict[str, Any]) -> None:
        """Log the start of batch size calibration."""
        if self.is_main_process:
            print_subheader(f"{Icons.CHART} Batch Size Calibration")
            print_config("min_batch_size", str(config.get('min_batch_size', 'auto')))
            print_config("max_batch_size", str(config.get('max_batch_size', 'auto')))
            print_config("target_memory", f"{config.get('target_memory', 0.7):.0%}")

        self._loggers['training'].info(f"Starting calibration: {config}")

    def log_calibration_step(
        self,
        batch_size: int,
        memory_pct: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log a calibration test step."""
        status = "OK" if success else "OOM"
        msg = f"Calibration: bs={batch_size}, mem={memory_pct:.1%}, status={status}"
        if error:
            msg += f", error={error}"

        if self.is_main_process:
            if success:
                print_calibration(f"Testing bs={batch_size}", batch_size=batch_size, memory_pct=memory_pct)
            else:
                print_warning(f"OOM at bs={batch_size}")

        self._loggers['training'].info(msg)

    def log_calibration_result(self, optimal_batch_size: int, memory_pct: float) -> None:
        """Log the final calibration result."""
        if self.is_main_process:
            print_success(f"Optimal batch size: {optimal_batch_size} (memory: {memory_pct:.1%})")

        self._loggers['training'].info(
            f"Calibration complete: optimal_batch_size={optimal_batch_size}, memory={memory_pct:.1%}"
        )

    # =========================================================================
    # Training Step Logging
    # =========================================================================

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        epoch: Optional[int] = None,
        total_steps: Optional[int] = None,
        tokens_per_sec: Optional[float] = None,
        memory_used: Optional[str] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a training step with formatted output."""
        # Build metrics dict
        metrics = {'loss': loss, 'lr': lr}
        if tokens_per_sec:
            metrics['tokens_per_sec'] = tokens_per_sec
        if additional_metrics:
            metrics.update(additional_metrics)

        # Console output (only main process)
        if self.is_main_process:
            print_training_step(
                step=step,
                total=total_steps or step,
                loss=loss,
                lr=lr,
                mem_used=memory_used,
                tokens_per_sec=tokens_per_sec,
            )

        # File logging (async)
        epoch_str = f" (epoch {epoch})" if epoch is not None else ""
        metrics_str = ", ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        self._loggers['training'].info(f"Step {step}{epoch_str}: {metrics_str}")

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        duration: Optional[str] = None,
    ) -> None:
        """Log an epoch completion summary."""
        if self.is_main_process:
            print_epoch_summary(epoch, total_epochs, train_loss, val_loss, duration)

        self._loggers['training'].info(
            f"Epoch {epoch + 1}/{total_epochs} complete: train_loss={train_loss:.4f}"
            + (f", val_loss={val_loss:.4f}" if val_loss else "")
            + (f", duration={duration}" if duration else "")
        )

    # =========================================================================
    # Model/Data Logging
    # =========================================================================

    def log_model_info(
        self,
        name: str,
        params: int,
        device: str,
        dtype: Optional[str] = None,
    ) -> None:
        """Log model information."""
        if self.is_main_process:
            print_model_info(name, params, device, dtype)

        self._loggers['training'].info(
            f"Model: {name}, params={params}, device={device}, dtype={dtype}"
        )

    def log_data_info(
        self,
        dataset_name: str,
        num_samples: int,
        batch_size: int,
        num_workers: Optional[int] = None,
    ) -> None:
        """Log data loading information."""
        if self.is_main_process:
            print_data_info(dataset_name, num_samples, batch_size, num_workers)

        self._loggers['training'].info(
            f"Data: {dataset_name}, samples={num_samples}, batch_size={batch_size}, workers={num_workers}"
        )

    def log_gpu_info(
        self,
        gpu_id: int,
        name: str,
        memory_total: str,
        memory_free: str,
    ) -> None:
        """Log GPU information."""
        if self.is_main_process:
            print_gpu_info(gpu_id, name, memory_total, memory_free)

        self._loggers['training'].info(
            f"GPU {gpu_id}: {name}, total={memory_total}, free={memory_free}"
        )

    # =========================================================================
    # Checkpoint Logging
    # =========================================================================

    def log_checkpoint_save(
        self,
        path: str,
        step: int,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log checkpoint save event."""
        msg = "Saved best checkpoint" if is_best else f"Saved checkpoint at step {step}"

        if self.is_main_process:
            print_checkpoint(msg, path, is_best)

        self._loggers['training'].info(f"{msg}: {path}")
        if additional_info:
            self._loggers['debug'].debug(f"Checkpoint details: {additional_info}")

    def log_checkpoint_load(self, path: str, epoch: int, step: int) -> None:
        """Log checkpoint load event."""
        msg = f"Loaded checkpoint: epoch={epoch}, step={step}"

        if self.is_main_process:
            print_checkpoint(f"Resumed from checkpoint", path)

        self._loggers['training'].info(f"{msg} from {path}")

    # =========================================================================
    # Async Task Submission
    # =========================================================================

    def submit_async(self, fn: Callable, *args, **kwargs) -> None:
        """Submit a heavy logging task to the worker pool."""
        if self._worker_pool:
            self._worker_pool.submit(fn, *args, **kwargs)
        else:
            # No pool, run synchronously
            fn(*args, **kwargs)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close all handlers and shutdown workers."""
        # Flush first
        self.flush()

        # Close handlers
        for handler in self._handlers:
            try:
                handler.close()
            except Exception:
                pass

        self._handlers.clear()


def setup_pipeline_logging(
    run_dir: Union[str, Path],
    rank: int = 0,
    world_size: int = 1,
    async_file_logging: bool = True,
    num_workers: int = 4,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    use_tqdm_handler: bool = True,
) -> PipelineLogger:
    """
    Setup pipeline logging with async file handlers and parallel workers.

    This is the recommended way to set up logging for the Ava training pipeline.
    Uses the same file structure as RunManager for consistency.

    Args:
        run_dir: Run directory (from RunManager.run_dir).
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        async_file_logging: Use async handlers for file logging (non-blocking).
        num_workers: Number of parallel worker threads for heavy tasks.
        console_level: Console logging level.
        file_level: File logging level.
        use_tqdm_handler: Use tqdm-compatible console handler.

    Returns:
        Configured PipelineLogger.

    Example:
        # In train_pipeline.py:
        from ava.core.logging import setup_pipeline_logging

        logger = setup_pipeline_logging(
            run_dir=run_manager.run_dir,
            rank=rank,
            world_size=world_size,
            async_file_logging=True
        )

        logger.log_phase_start("Model Building", phase_num=1)
        logger.log_model_info("EnhancedMoE", params=200_000_000, device="cuda:0")
        logger.log_phase_end()
    """
    # Configure root logger to suppress noise
    configure_root_logger(level=logging.WARNING)

    return PipelineLogger(
        run_dir=run_dir,
        rank=rank,
        world_size=world_size,
        async_file_logging=async_file_logging,
        num_workers=num_workers,
        console_level=console_level,
        file_level=file_level,
        use_tqdm_handler=use_tqdm_handler,
    )


# Convenience functions for colored output (can be used without logger)
def print_header(text: str, char: str = '═', width: int = 70, icon: str = None):
    """Print a colored section header with optional icon."""
    if supports_color():
        line = char * width
        icon_str = f"{icon} " if icon else ""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{icon_str}{text.center(width - len(icon_str) if icon else width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
    else:
        line = char * width
        print(f"\n{line}")
        print(text.center(width))
        print(line)


def print_subheader(text: str, char: str = '─', width: int = 60):
    """Print a colored sub-section header."""
    if supports_color():
        line = char * width
        print(f"\n{Colors.DARK_CYAN}{line}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
        print(f"{Colors.DARK_CYAN}{line}{Colors.RESET}")
    else:
        line = char * width
        print(f"\n{line}")
        print(text)
        print(line)


def print_success(text: str):
    """Print a success message in green with checkmark."""
    if supports_color():
        print(f"{Colors.GREEN}{Icons.SUCCESS} {text}{Colors.RESET}")
    else:
        print(f"[OK] {text}")


def print_warning(text: str):
    """Print a warning message in yellow with icon."""
    if supports_color():
        print(f"{Colors.YELLOW}{Icons.WARNING} {text}{Colors.RESET}")
    else:
        print(f"[WARNING] {text}")


def print_error(text: str):
    """Print an error message in red with icon."""
    if supports_color():
        print(f"{Colors.RED}{Icons.ERROR} {text}{Colors.RESET}")
    else:
        print(f"[ERROR] {text}")


def print_info(text: str):
    """Print an info message in cyan with icon."""
    if supports_color():
        print(f"{Colors.CYAN}{Icons.BULLET} {text}{Colors.RESET}")
    else:
        print(f"[INFO] {text}")


def print_step(text: str):
    """Print a step/progress message."""
    if supports_color():
        print(f"{Colors.LIGHT_BLUE}{Icons.ARROW_RIGHT} {text}{Colors.RESET}")
    else:
        print(f"-> {text}")


def print_config(key: str, value: str, indent: int = 2):
    """Print a configuration key-value pair with nice formatting."""
    spaces = ' ' * indent
    if supports_color():
        print(f"{spaces}{Colors.GRAY}{Icons.BULLET}{Colors.RESET} {Colors.WHITE}{key}:{Colors.RESET} {Colors.LIME}{value}{Colors.RESET}")
    else:
        print(f"{spaces}• {key}: {value}")


def print_phase(phase_num: int, name: str):
    """Print a training phase header (Phase 1, Phase 2, etc.)."""
    if supports_color():
        print(f"\n{Colors.BOLD}{Colors.PURPLE}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.PURPLE}  Phase {phase_num}: {name}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.PURPLE}{'═' * 70}{Colors.RESET}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Phase {phase_num}: {name}")
        print(f"{'=' * 70}")


def print_training_step(step: int, total: int, loss: float, lr: float,
                        mem_used: str = None, tokens_per_sec: float = None):
    """Print a formatted training step update with color-coded metrics."""
    if supports_color():
        step_str = f"{Colors.LIGHT_BLUE}Step {step}/{total}{Colors.RESET}"
        loss_str = f"{Colors.GOLD}Loss: {loss:.4f}{Colors.RESET}"
        lr_str = f"{Colors.PURPLE}LR: {lr:.2e}{Colors.RESET}"

        parts = [step_str, loss_str, lr_str]

        if mem_used:
            mem_str = f"{Colors.ORANGE}Mem: {mem_used}{Colors.RESET}"
            parts.append(mem_str)

        if tokens_per_sec:
            speed_str = f"{Colors.GREEN}{tokens_per_sec:.0f} tok/s{Colors.RESET}"
            parts.append(speed_str)

        print(f"  {' | '.join(parts)}")
    else:
        parts = [f"Step {step}/{total}", f"Loss: {loss:.4f}", f"LR: {lr:.2e}"]
        if mem_used:
            parts.append(f"Mem: {mem_used}")
        if tokens_per_sec:
            parts.append(f"{tokens_per_sec:.0f} tok/s")
        print(f"  {' | '.join(parts)}")


def print_epoch_summary(epoch: int, total_epochs: int, train_loss: float,
                        val_loss: float = None, duration: str = None):
    """Print an epoch summary with color-coded results."""
    if supports_color():
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  Epoch {epoch + 1}/{total_epochs} Complete{Colors.RESET}")
        print(f"  {Colors.WHITE}Train Loss:{Colors.RESET} {Colors.GOLD}{train_loss:.4f}{Colors.RESET}")
        if val_loss is not None:
            # Color code validation loss (green if lower than train, yellow otherwise)
            val_color = Colors.GREEN if val_loss < train_loss else Colors.YELLOW
            print(f"  {Colors.WHITE}Val Loss:{Colors.RESET}   {val_color}{val_loss:.4f}{Colors.RESET}")
        if duration:
            print(f"  {Colors.WHITE}Duration:{Colors.RESET}   {Colors.GRAY}{duration}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 50}{Colors.RESET}")
    else:
        print(f"\n{'-' * 50}")
        print(f"  Epoch {epoch + 1}/{total_epochs} Complete")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss:   {val_loss:.4f}")
        if duration:
            print(f"  Duration:   {duration}")
        print(f"{'-' * 50}")


def print_checkpoint(message: str, path: str = None, is_best: bool = False):
    """Print a checkpoint save/load message."""
    if supports_color():
        icon = Icons.SAVE if 'sav' in message.lower() else Icons.LOAD
        color = Colors.GREEN if is_best else Colors.CYAN
        print(f"{color}{icon} {message}{Colors.RESET}")
        if path:
            print(f"  {Colors.GRAY}Path: {path}{Colors.RESET}")
    else:
        icon = "[SAVE]" if 'sav' in message.lower() else "[LOAD]"
        print(f"{icon} {message}")
        if path:
            print(f"  Path: {path}")


def print_model_info(name: str, params: int, device: str, dtype: str = None):
    """Print model information with formatted parameter count."""
    if supports_color():
        param_str = f"{params / 1e6:.1f}M" if params >= 1e6 else f"{params / 1e3:.1f}K"
        print(f"{Colors.MAGENTA}{Icons.BRAIN} {name}{Colors.RESET}")
        print(f"  {Colors.WHITE}Parameters:{Colors.RESET} {Colors.LIME}{param_str}{Colors.RESET}")
        print(f"  {Colors.WHITE}Device:{Colors.RESET}     {Colors.MAGENTA}{device}{Colors.RESET}")
        if dtype:
            print(f"  {Colors.WHITE}Dtype:{Colors.RESET}      {Colors.CYAN}{dtype}{Colors.RESET}")
    else:
        param_str = f"{params / 1e6:.1f}M" if params >= 1e6 else f"{params / 1e3:.1f}K"
        print(f"[MODEL] {name}")
        print(f"  Parameters: {param_str}")
        print(f"  Device:     {device}")
        if dtype:
            print(f"  Dtype:      {dtype}")


def print_data_info(dataset_name: str, num_samples: int, batch_size: int,
                   num_workers: int = None):
    """Print data loading information."""
    if supports_color():
        print(f"{Colors.ORANGE}{Icons.DATA} Dataset: {dataset_name}{Colors.RESET}")
        print(f"  {Colors.WHITE}Samples:{Colors.RESET}    {Colors.LIME}{num_samples:,}{Colors.RESET}")
        print(f"  {Colors.WHITE}Batch Size:{Colors.RESET} {Colors.CYAN}{batch_size}{Colors.RESET}")
        if num_workers is not None:
            print(f"  {Colors.WHITE}Workers:{Colors.RESET}    {Colors.GRAY}{num_workers}{Colors.RESET}")
    else:
        print(f"[DATA] Dataset: {dataset_name}")
        print(f"  Samples:    {num_samples:,}")
        print(f"  Batch Size: {batch_size}")
        if num_workers is not None:
            print(f"  Workers:    {num_workers}")


def print_gpu_info(gpu_id: int, name: str, memory_total: str, memory_free: str):
    """Print GPU information."""
    if supports_color():
        print(f"{Colors.MAGENTA}{Icons.GPU} GPU {gpu_id}: {name}{Colors.RESET}")
        print(f"  {Colors.WHITE}Total:{Colors.RESET} {Colors.ORANGE}{memory_total}{Colors.RESET}")
        print(f"  {Colors.WHITE}Free:{Colors.RESET}  {Colors.GREEN}{memory_free}{Colors.RESET}")
    else:
        print(f"[GPU] GPU {gpu_id}: {name}")
        print(f"  Total: {memory_total}")
        print(f"  Free:  {memory_free}")


def print_calibration(message: str, batch_size: int = None, memory_pct: float = None):
    """Print batch size calibration information."""
    if supports_color():
        print(f"{Colors.PURPLE}{Icons.GEAR} {message}{Colors.RESET}")
        if batch_size is not None:
            print(f"  {Colors.WHITE}Batch Size:{Colors.RESET} {Colors.LIME}{batch_size}{Colors.RESET}")
        if memory_pct is not None:
            # Color code memory usage (green < 70%, yellow < 85%, red >= 85%)
            if memory_pct < 0.70:
                mem_color = Colors.GREEN
            elif memory_pct < 0.85:
                mem_color = Colors.YELLOW
            else:
                mem_color = Colors.RED
            print(f"  {Colors.WHITE}Memory:{Colors.RESET}     {mem_color}{memory_pct:.1%}{Colors.RESET}")
    else:
        print(f"[CALIBRATE] {message}")
        if batch_size is not None:
            print(f"  Batch Size: {batch_size}")
        if memory_pct is not None:
            print(f"  Memory:     {memory_pct:.1%}")


def print_metric(name: str, value: str, unit: str = '', good: bool = None):
    """Print a metric with optional status coloring."""
    if supports_color():
        if good is True:
            value_color = Colors.GREEN
        elif good is False:
            value_color = Colors.RED
        else:
            value_color = Colors.GOLD
        unit_str = f" {Colors.GRAY}{unit}{Colors.RESET}" if unit else ""
        print(f"  {Colors.WHITE}{name}:{Colors.RESET} {value_color}{Colors.BOLD}{value}{Colors.RESET}{unit_str}")
    else:
        unit_str = f" {unit}" if unit else ""
        print(f"  {name}: {value}{unit_str}")


def print_progress(current: int, total: int, prefix: str = '', width: int = 40):
    """Print a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)

    if supports_color():
        color = Colors.GREEN if percent >= 1.0 else Colors.CYAN
        print(f"\r{prefix}{color}{bar}{Colors.RESET} {Colors.BOLD}{percent*100:.1f}%{Colors.RESET} ({current}/{total})", end='', flush=True)
        if percent >= 1.0:
            print()  # New line when complete
    else:
        print(f"\r{prefix}{bar} {percent*100:.1f}% ({current}/{total})", end='', flush=True)
        if percent >= 1.0:
            print()


def print_table_row(columns: list, widths: list = None, header: bool = False):
    """Print a formatted table row."""
    if widths is None:
        widths = [20] * len(columns)

    formatted = []
    for col, width in zip(columns, widths):
        formatted.append(str(col).ljust(width)[:width])

    row = ' │ '.join(formatted)
    if supports_color():
        if header:
            print(f"{Colors.BOLD}{Colors.CYAN}{row}{Colors.RESET}")
            separator = '─┼─'.join('─' * w for w in widths)
            print(f"{Colors.DARK_CYAN}{separator}{Colors.RESET}")
        else:
            print(f"{Colors.WHITE}{row}{Colors.RESET}")
    else:
        print(row)
        if header:
            print('-+-'.join('-' * w for w in widths))


def print_box(text: str, style: str = 'info', width: int = 60):
    """Print text in a styled box."""
    lines = text.split('\n')
    max_len = min(max(len(line) for line in lines), width - 4)

    if supports_color():
        if style == 'success':
            border_color = Colors.GREEN
        elif style == 'warning':
            border_color = Colors.YELLOW
        elif style == 'error':
            border_color = Colors.RED
        else:
            border_color = Colors.CYAN

        top = f"╭{'─' * (max_len + 2)}╮"
        bottom = f"╰{'─' * (max_len + 2)}╯"

        print(f"{border_color}{top}{Colors.RESET}")
        for line in lines:
            padded = line.ljust(max_len)[:max_len]
            print(f"{border_color}│{Colors.RESET} {padded} {border_color}│{Colors.RESET}")
        print(f"{border_color}{bottom}{Colors.RESET}")
    else:
        top = f"+{'-' * (max_len + 2)}+"
        bottom = top
        print(top)
        for line in lines:
            print(f"| {line.ljust(max_len)[:max_len]} |")
        print(bottom)


__all__ = [
    # Core Classes
    'Colors',
    'Icons',
    'ColoredFormatter',
    'CleanFormatter',
    'TqdmLoggingHandler',
    # Async/Parallel Logging Classes
    'AsyncLoggingHandler',
    'ParallelLoggerPool',
    # Pipeline-Integrated Logging
    'PipelineLogger',
    'setup_pipeline_logging',
    # Legacy Setup Functions (still work)
    'setup_ava_logging',
    'setup_colored_logging',
    'configure_root_logger',
    'get_logger',
    'supports_color',
    # Print Helpers
    'print_header',
    'print_subheader',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_step',
    'print_config',
    'print_metric',
    'print_progress',
    'print_table_row',
    'print_box',
    # Training-Specific Print Helpers
    'print_phase',
    'print_training_step',
    'print_epoch_summary',
    'print_checkpoint',
    'print_model_info',
    'print_data_info',
    'print_gpu_info',
    'print_calibration',
]
