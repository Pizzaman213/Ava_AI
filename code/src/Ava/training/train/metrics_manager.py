"""
Metrics manager for the Ava pipeline.

Handles all training metrics tracking with TensorBoard and WandB integration.
Integrates with RunManager for organized output structure.

GPU SYNC FIX: Supports async WandB logging to eliminate network I/O blocking.
WandB log calls can take 5-50ms each, which blocks the training loop.
With async_logging=True, logs are queued and sent in a background thread.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import ManagerInterface, TrainingContext

logger = logging.getLogger(__name__)

# Check WandB availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# GPU SYNC FIX: Import async metrics logger for non-blocking WandB logging
try:
    from Ava.utils.async_metrics import AsyncMetricsLogger
    ASYNC_METRICS_AVAILABLE = True
except ImportError:
    ASYNC_METRICS_AVAILABLE = False
    AsyncMetricsLogger = None


class MetricsManager(ManagerInterface):
    """
    Manages all training metrics and logging.

    Provides unified interface for logging to TensorBoard and WandB.
    Integrates with RunManager for organized output structure.

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> metrics_mgr = MetricsManager(context)
        >>> metrics_mgr.initialize()
        >>> metrics_mgr.setup(log_dir, wandb_config={'project': 'my-project'})
        >>> metrics_mgr.log_training_step(step=100, loss=0.5, lr=1e-4, batch_size=32)
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the metrics manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self._writer: Optional[SummaryWriter] = None
        self._use_wandb = False
        self._metrics: Dict[str, List[tuple]] = {}
        self._log_dir: Optional[Path] = None
        self._generation_table = None
        # GPU SYNC FIX: Async metrics logger for non-blocking WandB logging
        self._async_logger: Optional['AsyncMetricsLogger'] = None
        self._use_async_logging = False

    def initialize(self) -> None:
        """Initialize the metrics manager."""
        self._initialized = True
        self.logger.debug("MetricsManager initialized")

    def cleanup(self) -> None:
        """Close writers and finish WandB run."""
        # GPU SYNC FIX: Shutdown async logger first to flush pending logs
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

        if self._use_wandb and WANDB_AVAILABLE:
            try:
                wandb.finish()
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
    ) -> None:
        """
        Setup logging infrastructure.

        Args:
            log_dir: Directory for TensorBoard logs
            wandb_config: Configuration for WandB (optional)
            use_wandb: Whether to use WandB logging
            wandb_dir: Directory for WandB local files (optional)
            async_logging: GPU SYNC FIX: Use async WandB logging (default True)
            async_batch_size: Number of log calls to batch before sending (default 10)

        Raises:
            RuntimeError: If log directory cannot be created
        """
        self.assert_initialized()

        self._log_dir = Path(log_dir)

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create log directory {log_dir}: {e}")

        # Setup TensorBoard writer
        self._writer = SummaryWriter(str(self._log_dir))
        self.logger.info(f"TensorBoard logging to: {self._log_dir}")

        # Setup WandB if requested and available
        self._use_wandb = use_wandb and WANDB_AVAILABLE and wandb_config is not None

        if self._use_wandb:
            try:
                wandb_config = wandb_config or {}
                # Use provided wandb_dir or fall back to config or default
                wb_dir = wandb_dir or wandb_config.get('dir', None)
                if wb_dir:
                    wb_dir = Path(wb_dir)
                    wb_dir.mkdir(parents=True, exist_ok=True)
                wandb.init(
                    project=wandb_config.get('project', 'transformer-training'),
                    entity=wandb_config.get('entity', None),
                    name=wandb_config.get('name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                    config=wandb_config.get('config', {}),
                    tags=wandb_config.get('tags', []),
                    resume=wandb_config.get('resume', 'allow'),
                    dir=str(wb_dir) if wb_dir else None,
                )
                self.logger.info(f"WandB initialized: {wandb.run.name}")

                # GPU SYNC FIX: Setup async WandB logging if requested
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
                self.logger.warning(f"Failed to initialize WandB: {e}")
                self._use_wandb = False
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning(
                "WandB requested but not available. "
                "Install with: pip install wandb"
            )

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        batch_size: int,
        **extra_metrics
    ) -> None:
        """
        Log training step metrics.

        Args:
            step: Current training step
            loss: Loss value
            lr: Current learning rate
            batch_size: Current batch size
            **extra_metrics: Additional metrics to log
        """
        if self._writer is None:
            return

        # Core metrics
        self._writer.add_scalar('train/loss', loss, step)
        self._writer.add_scalar('train/learning_rate', lr, step)
        self._writer.add_scalar('train/batch_size', batch_size, step)

        # Store in memory for summary
        self._store_metric('loss', step, loss)
        self._store_metric('learning_rate', step, lr)
        self._store_metric('batch_size', step, batch_size)

        # Extra metrics
        for key, value in extra_metrics.items():
            self._writer.add_scalar(f'train/{key}', value, step)
            self._store_metric(key, step, value)

        # WandB logging - GPU SYNC FIX: Use async logger if available
        if self._use_wandb:
            log_dict = {
                'train/loss': loss,
                'train/learning_rate': lr,
                'train/batch_size': batch_size,
            }
            for key, value in extra_metrics.items():
                log_dict[f'train/{key}'] = value

            if self._use_async_logging and self._async_logger is not None:
                # Non-blocking: returns immediately after queuing
                self._async_logger.log(log_dict, step=step)
            else:
                # Synchronous: may block for 5-50ms on network I/O
                wandb.log(log_dict, step=step)

    def log_gradients(self, step: int, grad_stats: Dict[str, float]) -> None:
        """
        Log gradient statistics.

        Args:
            step: Current training step
            grad_stats: Dictionary with gradient statistics
        """
        if self._writer is None:
            return

        avg_grad = grad_stats['total_norm'] / max(grad_stats['num_grads'], 1)
        grad_norm = grad_stats.get('grad_norm', grad_stats.get('total_norm', 0.0))

        self._writer.add_scalar('gradients/grad_norm', grad_norm, step)
        self._writer.add_scalar('gradients/avg_gradient', avg_grad, step)
        self._writer.add_scalar('gradients/max_gradient', grad_stats['max_grad'], step)
        self._writer.add_scalar('gradients/min_gradient', grad_stats['min_grad'], step)
        self._writer.add_scalar('gradients/num_zero_grads', grad_stats['num_zero_grads'], step)
        self._writer.add_scalar('gradients/num_params_with_grads', grad_stats['num_params'], step)

        if self._use_wandb:
            log_dict = {
                'gradients/grad_norm': grad_norm,
                'gradients/avg_gradient': avg_grad,
                'gradients/max_gradient': grad_stats['max_grad'],
                'gradients/min_gradient': grad_stats['min_grad'],
                'gradients/num_zero_grads': grad_stats['num_zero_grads'],
                'gradients/num_params_with_grads': grad_stats['num_params'],
            }
            if self._use_async_logging and self._async_logger is not None:
                self._async_logger.log(log_dict, step=step)
            else:
                wandb.log(log_dict, step=step)

    def log_moe_metrics(self, step: int, moe_metrics: Dict[str, float]) -> None:
        """
        Log MoE-specific metrics.

        Args:
            step: Current training step
            moe_metrics: Dictionary with MoE metrics (router load, expert utilization, etc.)
        """
        if self._writer is None:
            return

        for key, value in moe_metrics.items():
            self._writer.add_scalar(f'moe/{key}', value, step)

        if self._use_wandb:
            log_dict = {f'moe/{key}': value for key, value in moe_metrics.items()}
            if self._use_async_logging and self._async_logger is not None:
                self._async_logger.log(log_dict, step=step)
            else:
                wandb.log(log_dict, step=step)

    def log_validation(self, step: int, epoch: int, val_loss: float, **extra_metrics) -> None:
        """
        Log validation metrics.

        Args:
            step: Current training step
            epoch: Current epoch
            val_loss: Validation loss
            **extra_metrics: Additional validation metrics
        """
        if self._writer is None:
            return

        self._writer.add_scalar('validation/loss', val_loss, step)
        self._writer.add_scalar('validation/epoch', epoch, step)

        for key, value in extra_metrics.items():
            self._writer.add_scalar(f'validation/{key}', value, step)

        if self._use_wandb:
            log_dict = {
                'validation/loss': val_loss,
                'validation/epoch': epoch,
            }
            for key, value in extra_metrics.items():
                log_dict[f'validation/{key}'] = value
            if self._use_async_logging and self._async_logger is not None:
                self._async_logger.log(log_dict, step=step)
            else:
                wandb.log(log_dict, step=step)

        self.logger.info(f"Validation @ step {step}, epoch {epoch}: loss={val_loss:.4f}")

    def log_coherence(self, step: int, coherence_metrics: Dict[str, float]) -> None:
        """
        Log coherence metrics.

        Args:
            step: Current training step
            coherence_metrics: Dictionary with coherence metrics
        """
        if self._writer is None:
            return

        for key, value in coherence_metrics.items():
            self._writer.add_scalar(f'coherence/{key}', value, step)

        if self._use_wandb:
            log_dict = {f'coherence/{key}': value for key, value in coherence_metrics.items()}
            if self._use_async_logging and self._async_logger is not None:
                self._async_logger.log(log_dict, step=step)
            else:
                wandb.log(log_dict, step=step)

    def log_generation(self, step: int, generation_data: Dict[str, Any]) -> None:
        """
        Log generation to WandB table.

        Args:
            step: Current training step
            generation_data: Dictionary with generation details
        """
        if not self._use_wandb:
            return

        try:
            # Create table if needed
            if self._generation_table is None:
                self._generation_table = wandb.Table(columns=[
                    "step", "prompt", "generated_text", "temperature", "top_p", "top_k"
                ])

            # Add row
            self._generation_table.add_data(
                step,
                generation_data.get('prompt', ''),
                generation_data.get('generated_text', ''),
                generation_data.get('temperature', 1.0),
                generation_data.get('top_p', 1.0),
                generation_data.get('top_k', 0),
            )

            # Log table
            wandb.log({"generations": self._generation_table}, step=step)

        except Exception as e:
            self.logger.warning(f"Failed to log generation to WandB: {e}")

    def log_generation_table(self, generations: List[Dict[str, Any]]) -> None:
        """
        Log final generation table to WandB.

        Args:
            generations: List of generation dictionaries
        """
        if not self._use_wandb or not generations:
            return

        try:
            table = wandb.Table(columns=[
                "step", "prompt", "generated_text", "temperature", "top_p", "top_k"
            ])

            for gen in generations:
                table.add_data(
                    gen.get('step', 0),
                    gen.get('prompt', ''),
                    gen.get('generated_text', ''),
                    gen.get('temperature', 1.0),
                    gen.get('top_p', 1.0),
                    gen.get('top_k', 0),
                )

            wandb.log({"final_generations": table})

        except Exception as e:
            self.logger.warning(f"Failed to log generation table: {e}")

    def _store_metric(self, key: str, step: int, value: float) -> None:
        """Store metric in memory for summary."""
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append((step, value))

    def save_summary(self, output_path: Path) -> None:
        """
        Save metrics summary to JSON file.

        Args:
            output_path: Path to save the summary
        """
        summary = {key: values for key, values in self._metrics.items()}

        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved metrics summary to {output_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save metrics summary: {e}")

    def on_epoch_end(self, epoch: int) -> None:
        """Flush metrics at epoch end."""
        if self._writer is not None:
            self._writer.flush()

        # GPU SYNC FIX: Flush async logger at epoch end to ensure metrics are sent
        if self._async_logger is not None:
            self._async_logger.flush(timeout=30.0)

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
        # Try to save what we have
        if self._log_dir and self._metrics:
            try:
                self.save_summary(self._log_dir / 'metrics_on_error.json')
            except Exception as e:
                self.logger.debug(f"Failed to save error metrics: {e}")
