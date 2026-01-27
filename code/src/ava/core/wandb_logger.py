"""
Centralized logging module for the Ava training framework.

Consolidates all logging functionality including WandB, TensorBoard, and metrics tracking.

Features:
- WandB logging with batching and async support
- TensorBoard logging
- Metric accumulation and storage
- Generation table logging
- Unified MetricsManager for training pipelines

Usage:
    from ava.core.wandb_logger import WandBLogger, MetricsManager, WANDB_AVAILABLE

    # Standalone WandB logging
    logger = WandBLogger()
    logger.init(project="my-project", config={"lr": 0.001})
    logger.log({"loss": 0.5}, step=100)
    logger.finish()

    # Full metrics management (requires TrainingContext)
    metrics_mgr = MetricsManager(context)
    metrics_mgr.initialize()
    metrics_mgr.setup(log_dir, wandb_config={'project': 'my-project'})
    metrics_mgr.log_training_step(step=100, loss=0.5, lr=1e-4, batch_size=32)
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import deque

# Maximum history size for metrics to prevent memory leaks
MAX_METRICS_HISTORY = 200

# Import TrainingComponent for MetricsManager inheritance
if TYPE_CHECKING:
    from ava.training.context import TrainingContext, TrainingComponent as TrainingComponentType

# Check WandB availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore

# Async metrics logger
try:
    from ava.cuda.metrics import AsyncMetricsLogger
    ASYNC_METRICS_AVAILABLE = True
except ImportError:
    ASYNC_METRICS_AVAILABLE = False
    AsyncMetricsLogger = None

if TYPE_CHECKING:
    from ava.training.context import ManagerInterface, TrainingContext
    from ava.training.validation import ModelQualityScore
    from ava.training.diagnostics import LayerGradientStats, ExpertRoutingStats, MemoryBreakdown, TimingProfile

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Centralized WandB logging interface.

    Provides a unified API for all WandB operations including initialization,
    metric logging, table logging, and cleanup.

    Example:
        >>> wb_logger = WandBLogger()
        >>> wb_logger.init(project="ava-training", name="run-001")
        >>> wb_logger.log({"train/loss": 0.5}, step=100)
        >>> wb_logger.log_table("generations", columns=["text"], data=[["Hello"]])
        >>> wb_logger.finish()
    """

    def __init__(self):
        """Initialize the WandB logger."""
        self._initialized = False
        self._run_name: Optional[str] = None
        # Metric accumulator for batched logging
        self._pending_metrics: Dict[str, Any] = {}
        self._pending_step: Optional[int] = None

    @property
    def is_available(self) -> bool:
        """Check if WandB is available."""
        return WANDB_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """Check if WandB run is initialized."""
        return self._initialized and WANDB_AVAILABLE and wandb is not None and wandb.run is not None

    @property
    def run_name(self) -> Optional[str]:
        """Get the current run name."""
        if self.is_initialized and wandb is not None and wandb.run is not None:
            return wandb.run.name
        return self._run_name

    def init(
        self,
        project: str = "transformer-training",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        resume: str = "allow",
        dir: Optional[str] = None,
        ignore_globs: Optional[List[str]] = None,
    ) -> bool:
        """
        Initialize a WandB run.

        Args:
            project: WandB project name
            entity: WandB entity (team or user)
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            tags: List of tags for the run
            resume: Resume behavior ("allow", "must", "never", "auto")
            dir: Directory for WandB local files
            ignore_globs: File patterns to ignore

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not WANDB_AVAILABLE or wandb is None:
            logger.warning("WandB not installed. Install with: pip install wandb")
            return False

        if self._initialized:
            logger.warning("WandB already initialized, skipping re-initialization")
            return True

        try:
            # Create run name if not provided
            if name is None:
                name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._run_name = name

            # Create directory if specified
            if dir is not None:
                Path(dir).mkdir(parents=True, exist_ok=True)

            # Create settings with proper list types
            settings = wandb.Settings(
                ignore_globs=ignore_globs or ["*.patch"],
            )

            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config or {},
                tags=list(tags) if tags else [],
                resume=resume,
                dir=dir,
                settings=settings,
            )

            self._initialized = True
            logger.info(f"WandB initialized: {self.run_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self._initialized = False
            return False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step (optional)
            commit: Whether to commit immediately (default True)
        """
        if not self.is_initialized or wandb is None:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    def accumulate(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Accumulate metrics for batched logging.

        All metrics accumulated for the same step will be logged together
        when flush() is called.

        Args:
            metrics: Dictionary of metrics to accumulate
            step: Training step number
        """
        if not self.is_initialized:
            return

        # Flush previous step's metrics if step changed
        if self._pending_step is not None and self._pending_step != step:
            self.flush()

        self._pending_step = step
        self._pending_metrics.update(metrics)

    def flush(self) -> None:
        """
        Send all accumulated metrics to WandB in a single log() call.

        This ensures all metrics from the same training step appear at
        the same x-axis position in the WandB dashboard.
        """
        if not self.is_initialized or wandb is None:
            return

        if self._pending_metrics and self._pending_step is not None:
            try:
                wandb.log(self._pending_metrics, step=self._pending_step)
            except Exception as e:
                logger.warning(f"Failed to flush WandB metrics: {e}")

            self._pending_metrics = {}
            self._pending_step = None

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a table to WandB.

        Args:
            key: Table key in WandB
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Training step (optional)
        """
        if not self.is_initialized or wandb is None:
            logger.debug(
                f"Cannot log table '{key}': WandB not initialized "
                f"(is_initialized={self._initialized})"
            )
            return

        try:
            table = wandb.Table(columns=columns)
            for row in data:
                table.add_data(*row)

            # Debug: show latest row preview
            if data:
                latest_row = data[-1]
                text_preview = str(latest_row[2])[:50] if len(latest_row) > 2 else "N/A"
                logger.debug(f"WandB table '{key}': latest row step={latest_row[0]}, text='{text_preview}...'")

            wandb.log({key: table}, step=step)
            step_info = f"at step {step}" if step is not None else "(accumulating)"
            logger.debug(f"WandB table '{key}' logged with {len(data)} rows {step_info}")

            # Also log as summary table (persists across steps)
            if key == "generations":
                try:
                    if wandb.run is not None:
                        wandb.run.summary["generations_table"] = table
                        logger.debug("WandB generations table saved to summary")
                except Exception as summary_err:
                    logger.warning(f"Failed to save summary table: {summary_err}")

        except Exception as e:
            logger.error(f"Failed to log table '{key}' to WandB: {e}", exc_info=True)

    def create_table(self, columns: List[str]) -> Optional[Any]:
        """
        Create a WandB Table object.

        Args:
            columns: List of column names

        Returns:
            wandb.Table object or None if not available
        """
        if not WANDB_AVAILABLE or wandb is None:
            return None
        return wandb.Table(columns=columns)

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the run configuration.

        Args:
            config: Configuration dictionary to add/update
        """
        if not self.is_initialized or wandb is None:
            return

        try:
            wandb.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to update WandB config: {e}")

    def set_summary(self, key: str, value: Any) -> None:
        """
        Set a summary metric.

        Args:
            key: Metric key
            value: Metric value
        """
        if not self.is_initialized or wandb is None:
            return

        try:
            wandb.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to set WandB summary: {e}")

    def finish(self) -> None:
        """Finish the WandB run and upload remaining data."""
        if not self._initialized:
            return

        # Flush any pending metrics
        self.flush()

        if WANDB_AVAILABLE and wandb is not None:
            try:
                wandb.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.warning(f"Error finishing WandB run: {e}")

        self._initialized = False
        self._run_name = None

    def watch(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 1000,
    ) -> None:
        """
        Watch a model to log gradients and/or parameters.

        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency in steps
        """
        if not self.is_initialized or wandb is None:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")


# Global singleton instance for convenience
_global_logger: Optional[WandBLogger] = None


def get_wandb_logger() -> WandBLogger:
    """Get or create the global WandB logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WandBLogger()
    return _global_logger


def init_wandb(
    project: str = "transformer-training",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> bool:
    """
    Initialize WandB using the global logger.

    Convenience function for simple initialization.

    Args:
        project: WandB project name
        entity: WandB entity
        name: Run name
        config: Configuration dictionary
        **kwargs: Additional arguments passed to WandBLogger.init()

    Returns:
        True if initialization succeeded
    """
    return get_wandb_logger().init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        **kwargs
    )


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics using the global logger.

    Convenience function for simple logging.

    Args:
        metrics: Dictionary of metrics
        step: Training step
    """
    get_wandb_logger().log(metrics, step=step)


def finish_wandb() -> None:
    """Finish the global WandB run."""
    get_wandb_logger().finish()


# =============================================================================
# MetricsManager - Full training metrics management
# =============================================================================

class MetricsManager:
    """
    Manages all training metrics and logging.

    Provides unified interface for logging to TensorBoard and WandB.
    Integrates with RunManager for organized output structure.

    Note: This class can work standalone or with TrainingContext.
    For standalone use, pass context=None.

    Example:
        >>> metrics_mgr = MetricsManager()
        >>> metrics_mgr.initialize()
        >>> metrics_mgr.setup(log_dir, wandb_config={'project': 'my-project'})
        >>> metrics_mgr.log_training_step(step=100, loss=0.5, lr=1e-4, batch_size=32)
    """

    def __init__(self, context: Optional['TrainingContext'] = None):
        """
        Initialize the metrics manager.

        Args:
            context: Training context with shared state (optional)
        """
        self.context = context
        self._initialized = False
        self._writer: Optional[Any] = None  # SummaryWriter
        self._use_wandb = False
        self._metrics: Dict[str, deque] = {}  # Bounded deques to prevent memory leak
        self._log_dir: Optional[Path] = None
        self._generation_rows: deque = deque(maxlen=MAX_METRICS_HISTORY)  # Bounded to prevent memory leak
        # Async metrics logger for non-blocking WandB logging
        self._async_logger: Optional[Any] = None
        self._use_async_logging = False
        # Centralized WandB logger
        self._wandb_logger: WandBLogger = get_wandb_logger()
        # Running average loss tracking (EMA)
        self._ema_loss: Optional[float] = None
        self._ema_alpha: float = 0.1
        self._total_epoch_loss: float = 0.0
        self._epoch_step_count: int = 0
        # WandB metric accumulator
        self._pending_wandb_metrics: Dict[str, Any] = {}
        self._pending_step: Optional[int] = None
        # Async flush infrastructure
        self._flush_queue: Queue = Queue()
        self._flush_thread: Optional[threading.Thread] = None
        self._flush_thread_running = False
        self._last_flushed_step: int = -1  # Track last logged step for monotonicity
        # Logger for this instance
        self._logger = logging.getLogger(f"{__name__}.MetricsManager")
        # Logging config for fine-grained control (set via setup)
        self._logging_config: Dict[str, Any] = {}

    @property
    def logger(self):
        """Get the logger for this instance."""
        return self._logger

    def _should_log(self, log_type: str) -> bool:
        """Check if a specific log type is enabled.

        Args:
            log_type: Type of logging (e.g., 'training', 'gradients', 'moe', etc.)

        Returns:
            True if logging should proceed, False if disabled
        """
        # Master disable check
        if self._logging_config.get('disabled', False):
            return False

        # Check specific metric category
        category_key = f'log_{log_type}_metrics'
        return self._logging_config.get(category_key, True)

    def _should_log_wandb(self, log_type: str) -> bool:
        """Check if WandB logging is enabled for a specific type.

        Args:
            log_type: Type of logging (e.g., 'training', 'gradients', 'moe', etc.)

        Returns:
            True if WandB logging should proceed
        """
        if not self._use_wandb:
            return False
        if self._logging_config.get('disabled', False):
            return False

        # Check WandB-specific control
        wandb_key = f'wandb_log_{log_type}'
        return self._logging_config.get(wandb_key, True)

    def _should_log_tensorboard(self, log_type: str) -> bool:
        """Check if TensorBoard logging is enabled for a specific type.

        Args:
            log_type: Type of logging (e.g., 'training', 'gradients', 'moe', etc.)

        Returns:
            True if TensorBoard logging should proceed
        """
        if self._writer is None:
            return False
        if self._logging_config.get('disabled', False):
            return False

        # Check TensorBoard-specific control
        tb_key = f'tensorboard_log_{log_type}'
        return self._logging_config.get(tb_key, True)

    def assert_initialized(self) -> None:
        """Assert that the manager is initialized."""
        if not self._initialized:
            raise RuntimeError("MetricsManager not initialized. Call initialize() first.")

    def initialize(self) -> None:
        """Initialize the metrics manager."""
        self._initialized = True
        self.logger.debug("MetricsManager initialized")

    def cleanup(self) -> None:
        """Close writers and finish WandB run."""
        # Shutdown background flush thread first
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self.logger.debug("Shutting down WandB flush thread...")
            self._flush_thread_running = False
            # Send shutdown signal and wait for thread to finish
            self._flush_queue.put(None)
            self._flush_thread.join(timeout=10.0)
            if self._flush_thread.is_alive():
                self.logger.warning("WandB flush thread did not stop gracefully")
            self._flush_thread = None

        # Flush any remaining pending metrics synchronously
        self.flush_wandb_metrics(async_flush=False)

        # Shutdown async logger
        if self._async_logger is not None:
            try:
                self._async_logger.flush(timeout=30.0)
                self._async_logger.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down async logger: {e}")
            self._async_logger = None

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._use_wandb:
            try:
                self._wandb_logger.finish()
            except Exception as e:
                self.logger.warning(f"Error finishing WandB: {e}")

    def setup(
        self,
        log_dir: Path,
        wandb_config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        wandb_dir: Optional[Path] = None,
        async_logging: bool = True,
        async_batch_size: int = 10,
        logging_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Setup logging infrastructure.

        Args:
            log_dir: Directory for TensorBoard logs
            wandb_config: Configuration for WandB (optional)
            use_wandb: Whether to use WandB logging
            wandb_dir: Directory for WandB local files (optional)
            async_logging: Use async WandB logging (default True)
            async_batch_size: Number of log calls to batch before sending (default 10)
            logging_config: Fine-grained logging controls from LoggingConfig

        Raises:
            RuntimeError: If log directory cannot be created
        """
        self.assert_initialized()

        # Store logging config for checking in log methods
        self._logging_config = logging_config or {}

        # Check master disable flag
        if self._logging_config.get('disabled', False):
            self.logger.info("Logging is DISABLED via config - skipping all logging setup")
            self._use_wandb = False
            return

        self._log_dir = Path(log_dir)

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create log directory {log_dir}: {e}")

        # Setup TensorBoard writer (check logging_config.tensorboard_enabled)
        tensorboard_enabled = self._logging_config.get('tensorboard_enabled', True)
        if tensorboard_enabled and TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self._writer = SummaryWriter(str(self._log_dir))
            self.logger.info(f"TensorBoard logging to: {self._log_dir}")
        elif not tensorboard_enabled:
            self.logger.info("TensorBoard disabled via logging config")
        else:
            self.logger.warning("TensorBoard not available")

        # Setup WandB if requested and available (check logging_config.wandb_enabled)
        wandb_config_enabled = self._logging_config.get('wandb_enabled', True)
        self._use_wandb = use_wandb and wandb_config_enabled and WANDB_AVAILABLE and wandb_config is not None
        self.logger.info(f"WandB setup: use_wandb={use_wandb}, WANDB_AVAILABLE={WANDB_AVAILABLE}, "
                        f"wandb_config={'provided' if wandb_config else 'None'}, _use_wandb={self._use_wandb}")

        if wandb_config and not WANDB_AVAILABLE:
            self.logger.error(
                "WandB was requested but is NOT installed! "
                "Install with: pip install wandb"
            )

        if self._use_wandb:
            try:
                wandb_config = wandb_config or {}
                wb_dir = wandb_dir or wandb_config.get('dir', None)

                # Prepare WandB config
                wb_init_config = wandb_config.get('config', {}).copy()

                # Try to extract router info from context if available
                if self.context is not None and hasattr(self.context, 'config') and self.context.config:
                    model_config = self.context.config.get('model', {})
                    if 'router_type' in model_config:
                        wb_init_config['router_type'] = model_config['router_type']
                        wb_init_config['num_experts'] = model_config.get('num_experts', None)
                        wb_init_config['num_experts_per_token'] = model_config.get('num_experts_per_token', None)

                # Use centralized WandB logger for initialization
                success = self._wandb_logger.init(
                    project=wandb_config.get('project', 'transformer-training'),
                    entity=wandb_config.get('entity', None),
                    name=wandb_config.get('name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                    config=wb_init_config,
                    tags=list(wandb_config.get('tags', [])),
                    resume=wandb_config.get('resume', 'allow'),
                    dir=str(wb_dir) if wb_dir else None,
                    ignore_globs=["*.patch"],
                )

                if success:
                    self.logger.info(f"WandB initialized: {self._wandb_logger.run_name}")
                    if 'router_type' in wb_init_config:
                        self.logger.info(f"Router configuration logged to WandB: {wb_init_config.get('router_type')} "
                                       f"(experts={wb_init_config.get('num_experts')}, k={wb_init_config.get('num_experts_per_token')})")
                else:
                    self._use_wandb = False

                # Setup async WandB logging if requested
                self._use_async_logging = async_logging and ASYNC_METRICS_AVAILABLE
                if self._use_async_logging and AsyncMetricsLogger is not None:
                    self._async_logger = AsyncMetricsLogger(
                        backend='wandb',
                        batch_size=async_batch_size,
                        flush_interval_seconds=5.0,
                    )
                    self.logger.info(
                        f"Async WandB logging enabled (batch_size={async_batch_size}). "
                        f"Training loop will not block on network I/O."
                    )
                elif async_logging and not ASYNC_METRICS_AVAILABLE:
                    self.logger.warning(
                        "Async logging requested but AsyncMetricsLogger not available. "
                        "Falling back to synchronous logging."
                    )

            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                self.logger.error(
                    f"Failed to initialize WandB!\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{tb_str}"
                )
                self._use_wandb = False
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning(
                "WandB requested but not available. "
                "Install with: pip install wandb"
            )

    def accumulate_wandb_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Accumulate metrics for batched wandb logging.

        OPTIMIZATION: Routes to AsyncMetricsLogger when available for non-blocking
        network I/O. Falls back to batched sync logging otherwise.

        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Training step number
        """
        if not self._use_wandb:
            return

        # FAST PATH: Use AsyncMetricsLogger for non-blocking I/O
        if self._use_async_logging and self._async_logger is not None:
            # AsyncMetricsLogger handles batching internally
            self._async_logger.log(metrics, step=step)
            return

        # FALLBACK: Local batching with async flush thread
        if self._pending_step is not None and self._pending_step != step:
            self.flush_wandb_metrics()

        self._pending_step = step
        self._pending_wandb_metrics.update(metrics)

    def _start_flush_thread(self) -> None:
        """Start the background flush thread if not running."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return

        self._flush_thread_running = True
        self._flush_thread = threading.Thread(
            target=self._background_flush_loop,
            daemon=True,
            name="WandBFlushThread"
        )
        self._flush_thread.start()
        self.logger.debug("Started background WandB flush thread")

    def _background_flush_loop(self) -> None:
        """Background thread that flushes metrics to WandB without blocking training."""
        while self._flush_thread_running:
            try:
                # Wait for metrics with timeout to allow graceful shutdown
                item = self._flush_queue.get(timeout=1.0)
                if item is None:  # Shutdown signal
                    break
                metrics, step = item
                # Skip stale steps to maintain WandB monotonicity
                # This prevents "step X less than current step Y" warnings
                if step <= self._last_flushed_step:
                    continue
                try:
                    self._wandb_logger.log(metrics, step=step)
                    self._last_flushed_step = step
                except Exception as e:
                    self.logger.warning(f"Async WandB flush failed: {e}")
            except Empty:
                continue  # Timeout, check if we should stop
            except Exception as e:
                self.logger.warning(f"Error in flush thread: {e}")

    def flush_wandb_metrics(self, async_flush: bool = True) -> None:
        """Send all accumulated metrics to wandb.

        OPTIMIZATION: Uses AsyncMetricsLogger.flush() when available for
        non-blocking network I/O. Falls back to custom flush thread otherwise.

        Args:
            async_flush: If True (default), queue for background flush.
                        If False, flush synchronously (blocking).
        """
        if not self._use_wandb:
            return

        # FAST PATH: AsyncMetricsLogger handles flushing
        if self._use_async_logging and self._async_logger is not None:
            # AsyncMetricsLogger's flush() waits for pending logs
            # Use a reasonable timeout for epoch boundaries
            if not async_flush:
                self._async_logger.flush(timeout=30.0)
            # Async logger batches internally, no local queue to flush
            return

        # FALLBACK: Local batching with async flush thread
        if self._pending_wandb_metrics and self._pending_step is not None:
            metrics_copy = self._pending_wandb_metrics.copy()
            step = self._pending_step
            self._pending_wandb_metrics = {}
            self._pending_step = None

            if async_flush:
                # Start background thread if not running
                self._start_flush_thread()
                # Queue for async flush (non-blocking)
                self._flush_queue.put((metrics_copy, step))
            else:
                # Synchronous flush (blocking) - also check monotonicity
                if step > self._last_flushed_step:
                    try:
                        self._wandb_logger.log(metrics_copy, step=step)
                        self._last_flushed_step = step
                    except Exception as e:
                        self.logger.warning(f"Failed to flush wandb metrics: {e}")

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        batch_size: int,
        **extra_metrics
    ) -> None:
        """
        Log training step metrics with running average loss.

        Args:
            step: Current training step
            loss: Loss value
            lr: Current learning rate
            batch_size: Current batch size
            **extra_metrics: Additional metrics to log
        """
        # Check if training metrics logging is enabled
        if not self._should_log('training'):
            return

        # Update running average loss (EMA)
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss

        # Update epoch average tracking
        self._total_epoch_loss += loss
        self._epoch_step_count += 1
        avg_loss = self._total_epoch_loss / self._epoch_step_count

        # TensorBoard logging (check tensorboard_log_training)
        if self._should_log_tensorboard('training'):
            self._writer.add_scalar('train/loss', loss, step)
            self._writer.add_scalar('train/avg_loss', avg_loss, step)
            self._writer.add_scalar('train/smoothed_loss', self._ema_loss, step)
            self._writer.add_scalar('train/learning_rate', lr, step)
            self._writer.add_scalar('train/batch_size', batch_size, step)
            for key, value in extra_metrics.items():
                self._writer.add_scalar(f'train/{key}', value, step)

        # Store in memory for summary
        self._store_metric('loss', step, loss)
        self._store_metric('avg_loss', step, avg_loss)
        self._store_metric('smoothed_loss', step, self._ema_loss)
        self._store_metric('learning_rate', step, lr)
        self._store_metric('batch_size', step, batch_size)
        for key, value in extra_metrics.items():
            self._store_metric(key, step, value)

        # WandB logging (check wandb_log_training)
        if not self._should_log_wandb('training'):
            return

        log_dict = {
            'train/loss': loss,
            'train/avg_loss': avg_loss,
            'train/smoothed_loss': self._ema_loss,
            'train/learning_rate': lr,
            'train/batch_size': batch_size,
        }
        for key, value in extra_metrics.items():
            log_dict[f'train/{key}'] = value
        self.accumulate_wandb_metrics(log_dict, step)

    def log_gradients(self, step: int, grad_stats: Dict[str, float]) -> None:
        """Log gradient statistics."""
        if not self._should_log('gradient'):
            return

        avg_grad = grad_stats['total_norm'] / max(grad_stats['num_grads'], 1)
        grad_norm = grad_stats.get('grad_norm', grad_stats.get('total_norm', 0.0))

        if self._should_log_tensorboard('gradients'):
            self._writer.add_scalar('gradients/grad_norm', grad_norm, step)
            self._writer.add_scalar('gradients/avg_gradient', avg_grad, step)
            self._writer.add_scalar('gradients/max_gradient', grad_stats['max_grad'], step)
            self._writer.add_scalar('gradients/min_gradient', grad_stats['min_grad'], step)
            self._writer.add_scalar('gradients/num_zero_grads', grad_stats['num_zero_grads'], step)
            self._writer.add_scalar('gradients/num_params_with_grads', grad_stats['num_params'], step)

        if self._should_log_wandb('gradients'):
            log_dict = {
                'gradients/grad_norm': grad_norm,
                'gradients/avg_gradient': avg_grad,
                'gradients/max_gradient': grad_stats['max_grad'],
                'gradients/min_gradient': grad_stats['min_grad'],
                'gradients/num_zero_grads': grad_stats['num_zero_grads'],
                'gradients/num_params_with_grads': grad_stats['num_params'],
            }
            self.accumulate_wandb_metrics(log_dict, step)

    def log_moe_metrics(self, step: int, moe_metrics: Dict[str, float]) -> None:
        """Log MoE-specific metrics."""
        if not self._should_log('moe'):
            return

        if self._should_log_tensorboard('moe'):
            for key, value in moe_metrics.items():
                self._writer.add_scalar(f'moe/{key}', value, step)

        if self._should_log_wandb('moe'):
            log_dict = {f'moe/{key}': value for key, value in moe_metrics.items()}
            self.accumulate_wandb_metrics(log_dict, step)

    def log_validation(self, step: int, epoch: int, val_loss: float, **extra_metrics) -> None:
        """Log validation metrics."""
        if not self._should_log('validation'):
            return

        if self._should_log_tensorboard('validation'):
            self._writer.add_scalar('validation/loss', val_loss, step)
            self._writer.add_scalar('validation/epoch', epoch, step)
            for key, value in extra_metrics.items():
                self._writer.add_scalar(f'validation/{key}', value, step)

        if self._should_log_wandb('validation'):
            log_dict = {
                'validation/loss': val_loss,
                'validation/epoch': epoch,
            }
            for key, value in extra_metrics.items():
                log_dict[f'validation/{key}'] = value
            self.accumulate_wandb_metrics(log_dict, step)

        self.logger.info(f"Validation @ step {step}, epoch {epoch}: loss={val_loss:.4f}")

    def log_coherence(self, step: int, coherence_metrics: Dict[str, float]) -> None:
        """Log coherence metrics."""
        if not self._should_log('coherence'):
            return

        if self._should_log_tensorboard('coherence'):
            for key, value in coherence_metrics.items():
                self._writer.add_scalar(f'coherence/{key}', value, step)

        if self._should_log_wandb('coherence'):
            log_dict = {f'coherence/{key}': value for key, value in coherence_metrics.items()}
            self.accumulate_wandb_metrics(log_dict, step)

    def log_quality_score(self, step: int, quality_score: 'ModelQualityScore') -> None:
        """Log model quality score metrics."""
        if not self._should_log('quality'):
            return

        metrics = {
            'model_selection/quality_score': quality_score.quality_score,
            'model_selection/val_loss_normalized': quality_score.val_loss_normalized,
            'model_selection/coherence_normalized': quality_score.coherence_normalized,
            'model_selection/perplexity_normalized': quality_score.perplexity_normalized,
        }

        if quality_score.val_loss is not None:
            metrics['model_selection/raw_val_loss'] = quality_score.val_loss
        if quality_score.coherence_score is not None:
            metrics['model_selection/raw_coherence'] = quality_score.coherence_score
        if quality_score.perplexity is not None:
            metrics['model_selection/raw_perplexity'] = quality_score.perplexity

        if self._should_log_tensorboard('quality'):
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        if self._should_log_wandb('quality'):
            self.accumulate_wandb_metrics(metrics, step)

    def log_per_layer_gradients(self, step: int, layer_stats: List['LayerGradientStats']) -> None:
        """Log per-layer gradient statistics (expensive - disabled by default)."""
        if not self._logging_config.get('wandb_log_per_layer_grads', False):
            return

        for stat in layer_stats:
            layer_key = stat.layer_name.replace('.', '/')
            prefix = f'gradients/layer/{layer_key}'

            metrics = {
                f'{prefix}/norm': stat.grad_norm,
                f'{prefix}/mean': stat.grad_mean,
                f'{prefix}/std': stat.grad_std,
                f'{prefix}/max': stat.grad_max,
                f'{prefix}/min': stat.grad_min,
            }

            if self._writer is not None:
                for key, value in metrics.items():
                    self._writer.add_scalar(key, value, step)

            if self._use_wandb:
                self.accumulate_wandb_metrics(metrics, step)

    def log_routing_diagnostics(self, step: int, routing_stats: 'ExpertRoutingStats') -> None:
        """Log expert routing diagnostics (expensive - disabled by default)."""
        if not self._logging_config.get('wandb_log_routing_diagnostics', False):
            return

        metrics = {
            'routing/balance_score': routing_stats.balance_score,
            'routing/routing_entropy': routing_stats.routing_entropy,
            'routing/dropped_tokens': routing_stats.dropped_tokens,
        }

        for expert_id, load in routing_stats.per_expert_load.items():
            metrics[f'routing/expert_{expert_id}_load'] = load
        for expert_id, usage in routing_stats.capacity_usage.items():
            metrics[f'routing/expert_{expert_id}_capacity'] = usage

        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        if self._use_wandb:
            self.accumulate_wandb_metrics(metrics, step)

    def log_memory_breakdown(self, step: int, breakdown: 'MemoryBreakdown') -> None:
        """Log memory breakdown metrics."""
        if not self._should_log('memory'):
            return

        metrics = {
            'memory/total_allocated_gb': breakdown.total_allocated_gb,
            'memory/total_reserved_gb': breakdown.total_reserved_gb,
            'memory/parameters_gb': breakdown.parameters_gb,
            'memory/gradients_gb': breakdown.gradients_gb,
            'memory/optimizer_states_gb': breakdown.optimizer_states_gb,
            'memory/activations_gb': breakdown.activations_gb,
            'memory/other_gb': breakdown.other_gb,
        }

        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        if self._should_log_wandb('memory'):
            self.accumulate_wandb_metrics(metrics, step)

    def log_timing_profile(self, step: int, profile: 'TimingProfile') -> None:
        """Log timing profile metrics."""
        if not self._should_log('timing'):
            return

        metrics = {
            'timing/data_loading_ms': profile.data_loading_ms,
            'timing/forward_ms': profile.forward_ms,
            'timing/backward_ms': profile.backward_ms,
            'timing/optimizer_step_ms': profile.optimizer_step_ms,
            'timing/total_step_ms': profile.total_step_ms,
        }

        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        if self._should_log_wandb('timing'):
            self.accumulate_wandb_metrics(metrics, step)

    def log_generation(self, log_step: int, generation_data: Dict[str, Any]) -> None:
        """Log generation to WandB table (accumulates all generations)."""
        from tqdm import tqdm

        # Check if generation logging is disabled
        if not self._logging_config.get('log_generation_samples', True):
            return

        if not self._use_wandb or not self._logging_config.get('wandb_log_generations', True):
            tqdm.write(f"  [Gen] WARNING: WandB generation logging disabled")
            return

        try:
            gen_step = generation_data.get('step', log_step)
            gen_text = generation_data.get('generated_text', '')
            row = (
                gen_step,
                generation_data.get('prompt', ''),
                gen_text,
                generation_data.get('temperature', 1.0),
                generation_data.get('top_p', 1.0),
                generation_data.get('top_k', 50),
            )
            self._generation_rows.append(row)

            # Show preview of the new generation being logged
            text_preview = gen_text[:60].replace('\n', ' ') if gen_text else "(empty)"
            tqdm.write(f"  [Gen] Adding row for step {gen_step}: '{text_preview}...'")
            tqdm.write(f"  [Gen] Updating WandB table (total accumulated: {len(self._generation_rows)} rows)")

            # Log WITHOUT step parameter - this updates a single table instead of creating
            # separate table files for each step
            self._wandb_logger.log_table(
                key="generations",
                columns=["step", "prompt", "generated_text", "temperature", "top_p", "top_k"],
                data=list(self._generation_rows),
                step=None,  # No step = single table that accumulates all generations
            )

            if self._async_logger is not None:
                self._async_logger.update_last_step(gen_step)

            tqdm.write(f"  [Gen] SUCCESS: Generation at step {gen_step} logged to WandB")

        except Exception as e:
            tqdm.write(f"  [Gen] ERROR logging to WandB: {e}")
            self.logger.error(f"Failed to log generation to WandB: {e}", exc_info=True)

    def log_generation_table(self, generations: List[Dict[str, Any]]) -> None:
        """Log final generation table to WandB."""
        if not self._use_wandb or not generations:
            return

        try:
            data = [
                [
                    gen.get('step', 0),
                    gen.get('prompt', ''),
                    gen.get('generated_text', ''),
                    gen.get('temperature', 1.0),
                    gen.get('top_p', 1.0),
                    gen.get('top_k', 0),
                ]
                for gen in generations
            ]

            final_step = generations[-1].get('step', 0) if generations else 0

            self._wandb_logger.log_table(
                key="final_generations",
                columns=["step", "prompt", "generated_text", "temperature", "top_p", "top_k"],
                data=data,
                step=final_step,
            )

        except Exception as e:
            self.logger.warning(f"Failed to log generation table: {e}")

    def _store_metric(self, key: str, step: int, value: float) -> None:
        """Store metric in memory for summary (bounded to prevent memory leak)."""
        if key not in self._metrics:
            self._metrics[key] = deque(maxlen=MAX_METRICS_HISTORY)
        self._metrics[key].append((step, value))

    def save_summary(self, output_path: Path) -> None:
        """Save metrics summary to JSON file."""
        summary = {key: list(values) for key, values in self._metrics.items()}  # Convert deques to lists

        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved metrics summary to {output_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save metrics summary: {e}")

    def on_epoch_end(self, epoch: int) -> None:
        """Flush metrics at epoch end and reset epoch counters."""
        if self._writer is not None:
            self._writer.flush()

        if self._async_logger is not None:
            self._async_logger.flush(timeout=30.0)

        # Flush any pending WandB metrics before epoch reset
        self.flush_wandb_metrics()

        self._total_epoch_loss = 0.0
        self._epoch_step_count = 0

    def get_status(self) -> Dict[str, Any]:
        """Return current metrics status."""
        return {
            'log_dir': str(self._log_dir) if self._log_dir else None,
            'use_wandb': self._use_wandb,
            'num_metrics': len(self._metrics),
            'total_logged_values': sum(len(v) for v in self._metrics.values()),
        }

    def on_error(self, error: Exception) -> None:
        """Handle metrics errors."""
        self.logger.error(f"Metrics error: {error}", exc_info=True)
        if self._log_dir and self._metrics:
            try:
                self.save_summary(self._log_dir / 'metrics_on_error.json')
            except Exception as e:
                self.logger.debug(f"Failed to save error metrics: {e}")
