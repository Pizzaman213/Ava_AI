"""
Metrics manager for the Ava pipeline.

Handles all training metrics tracking with TensorBoard and WandB integration.
Integrates with RunManager for organized output structure.

Supports async WandB logging to eliminate network I/O blocking.
With async_logging=True, logs are queued and sent in a background thread.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter

from .context import ManagerInterface, TrainingContext

if TYPE_CHECKING:
    from .quality_score import ModelQualityScore
    from .diagnostics import LayerGradientStats, ExpertRoutingStats, MemoryBreakdown, TimingProfile

logger = logging.getLogger(__name__)

# Check WandB availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import async metrics logger for non-blocking WandB logging
try:
    from ava.cuda.metrics import AsyncMetricsLogger
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
        self._generation_rows: List[tuple] = []  # Accumulate rows, create fresh table each log
        # Async metrics logger for non-blocking WandB logging
        self._async_logger: Optional['AsyncMetricsLogger'] = None
        self._use_async_logging = False
        # Running average loss tracking (EMA)
        self._ema_loss: Optional[float] = None
        self._ema_alpha: float = 0.1  # EMA smoothing factor (lower = smoother)
        self._total_epoch_loss: float = 0.0
        self._epoch_step_count: int = 0
        # WandB metric accumulator - batches all metrics for single log() call per step
        self._pending_wandb_metrics: Dict[str, Any] = {}
        self._pending_step: Optional[int] = None

    def initialize(self) -> None:
        """Initialize the metrics manager."""
        self._initialized = True
        self.logger.debug("MetricsManager initialized")

    def cleanup(self) -> None:
        """Close writers and finish WandB run."""
        # Shutdown async logger first to flush pending logs
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
            async_logging: Use async WandB logging (default True)
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
        self.logger.info(f"WandB setup: use_wandb={use_wandb}, WANDB_AVAILABLE={WANDB_AVAILABLE}, "
                        f"wandb_config={'provided' if wandb_config else 'None'}, _use_wandb={self._use_wandb}")

        # Debug info for generation logging
        if wandb_config and not WANDB_AVAILABLE:
            self.logger.error(
                "⚠️  GENERATION LOGGING ISSUE: WandB was requested but is NOT installed! "
                "Generation samples will NOT be logged to the table. "
                "Install with: pip install wandb"
            )

        if self._use_wandb:
            try:
                wandb_config = wandb_config or {}
                # Use provided wandb_dir or fall back to config or default
                wb_dir = wandb_dir or wandb_config.get('dir', None)
                if wb_dir:
                    wb_dir = Path(wb_dir)
                    wb_dir.mkdir(parents=True, exist_ok=True)

                # Prepare WandB config - include model router information if available
                wb_init_config = wandb_config.get('config', {}).copy()

                # Try to extract router info from context if available
                if hasattr(self.context, 'config') and self.context.config:
                    model_config = self.context.config.get('model', {})
                    if 'router_type' in model_config:
                        wb_init_config['router_type'] = model_config['router_type']
                        wb_init_config['num_experts'] = model_config.get('num_experts', None)
                        wb_init_config['num_experts_per_token'] = model_config.get('num_experts_per_token', None)

                # Create Settings with lists instead of tuples to avoid Pydantic warnings
                wb_settings = wandb.Settings(
                    ignore_globs=["*.patch"],  # Default is tuple, use list
                )
                wandb.init(
                    project=wandb_config.get('project', 'transformer-training'),
                    entity=wandb_config.get('entity', None),
                    name=wandb_config.get('name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                    config=wb_init_config,
                    tags=list(wandb_config.get('tags', [])),
                    resume=wandb_config.get('resume', 'allow'),
                    dir=str(wb_dir) if wb_dir else None,
                    settings=wb_settings,
                )
                self.logger.info(f"WandB initialized: {wandb.run.name}")
                if 'router_type' in wb_init_config:
                    self.logger.info(f"Router configuration logged to WandB: {wb_init_config.get('router_type')} "
                                   f"(experts={wb_init_config.get('num_experts')}, k={wb_init_config.get('num_experts_per_token')})")

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
                    f"⚠️  GENERATION LOGGING ISSUE: Failed to initialize WandB!\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{tb_str}\n"
                    f"Generation samples will NOT be logged to the table. "
                    f"Common fixes:\n"
                    f"  1. Check WandB is installed: pip install wandb\n"
                    f"  2. Authenticate with: wandb login\n"
                    f"  3. Check network connectivity\n"
                    f"  4. Check wandb_config in training config"
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

        All metrics accumulated for the same step will be logged together
        in a single wandb.log() call when flush_wandb_metrics() is called.

        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Training step number
        """
        if not self._use_wandb:
            return

        # If step changed, flush previous step's metrics first
        if self._pending_step is not None and self._pending_step != step:
            self.flush_wandb_metrics()

        self._pending_step = step
        self._pending_wandb_metrics.update(metrics)

    def flush_wandb_metrics(self) -> None:
        """
        Send all accumulated metrics to wandb in a single log() call.

        This ensures all metrics from the same training step appear at
        the same x-axis position in the wandb dashboard.
        """
        if not self._use_wandb:
            return

        if self._pending_wandb_metrics and self._pending_step is not None:
            try:
                wandb.log(self._pending_wandb_metrics, step=self._pending_step)
            except Exception as e:
                self.logger.warning(f"Failed to flush wandb metrics: {e}")

            self._pending_wandb_metrics = {}
            self._pending_step = None

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

        Logs to both TensorBoard and WandB (if enabled):
        - train/loss: Raw loss for this step
        - train/avg_loss: Running average loss for this epoch
        - train/smoothed_loss: Exponential moving average (smoother trend)

        Args:
            step: Current training step
            loss: Loss value
            lr: Current learning rate
            batch_size: Current batch size
            **extra_metrics: Additional metrics to log
        """
        # Update running average loss (EMA)
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss

        # Update epoch average tracking
        self._total_epoch_loss += loss
        self._epoch_step_count += 1
        avg_loss = self._total_epoch_loss / self._epoch_step_count

        # TensorBoard logging (if writer is available)
        if self._writer is not None:
            self._writer.add_scalar('train/loss', loss, step)
            self._writer.add_scalar('train/avg_loss', avg_loss, step)
            self._writer.add_scalar('train/smoothed_loss', self._ema_loss, step)
            self._writer.add_scalar('train/learning_rate', lr, step)
            self._writer.add_scalar('train/batch_size', batch_size, step)

            # Extra metrics to TensorBoard
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

        # WandB logging: accumulate for batched logging
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
        """
        Log gradient statistics.

        Args:
            step: Current training step
            grad_stats: Dictionary with gradient statistics
        """
        avg_grad = grad_stats['total_norm'] / max(grad_stats['num_grads'], 1)
        grad_norm = grad_stats.get('grad_norm', grad_stats.get('total_norm', 0.0))

        # TensorBoard logging (if writer is available)
        if self._writer is not None:
            self._writer.add_scalar('gradients/grad_norm', grad_norm, step)
            self._writer.add_scalar('gradients/avg_gradient', avg_grad, step)
            self._writer.add_scalar('gradients/max_gradient', grad_stats['max_grad'], step)
            self._writer.add_scalar('gradients/min_gradient', grad_stats['min_grad'], step)
            self._writer.add_scalar('gradients/num_zero_grads', grad_stats['num_zero_grads'], step)
            self._writer.add_scalar('gradients/num_params_with_grads', grad_stats['num_params'], step)

        # WandB logging: accumulate for batched logging
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
        """
        Log MoE-specific metrics.

        Args:
            step: Current training step
            moe_metrics: Dictionary with MoE metrics (router load, expert utilization, etc.)
        """
        # TensorBoard logging (if writer is available)
        if self._writer is not None:
            for key, value in moe_metrics.items():
                self._writer.add_scalar(f'moe/{key}', value, step)

        # WandB logging: accumulate for batched logging
        log_dict = {f'moe/{key}': value for key, value in moe_metrics.items()}
        self.accumulate_wandb_metrics(log_dict, step)

    def log_validation(self, step: int, epoch: int, val_loss: float, **extra_metrics) -> None:
        """
        Log validation metrics.

        Args:
            step: Current training step
            epoch: Current epoch
            val_loss: Validation loss
            **extra_metrics: Additional validation metrics
        """
        # TensorBoard logging (if writer is available)
        if self._writer is not None:
            self._writer.add_scalar('validation/loss', val_loss, step)
            self._writer.add_scalar('validation/epoch', epoch, step)

            for key, value in extra_metrics.items():
                self._writer.add_scalar(f'validation/{key}', value, step)

        # WandB logging: accumulate for batched logging
        log_dict = {
            'validation/loss': val_loss,
            'validation/epoch': epoch,
        }
        for key, value in extra_metrics.items():
            log_dict[f'validation/{key}'] = value
        self.accumulate_wandb_metrics(log_dict, step)

        self.logger.info(f"Validation @ step {step}, epoch {epoch}: loss={val_loss:.4f}")

    def log_coherence(self, step: int, coherence_metrics: Dict[str, float]) -> None:
        """
        Log coherence metrics.

        Args:
            step: Current training step
            coherence_metrics: Dictionary with coherence metrics
        """
        # TensorBoard logging (if writer is available)
        if self._writer is not None:
            for key, value in coherence_metrics.items():
                self._writer.add_scalar(f'coherence/{key}', value, step)

        # WandB logging: accumulate for batched logging
        log_dict = {f'coherence/{key}': value for key, value in coherence_metrics.items()}
        self.accumulate_wandb_metrics(log_dict, step)

    def log_quality_score(self, step: int, quality_score: 'ModelQualityScore') -> None:
        """
        Log model quality score metrics.

        Args:
            step: Current training step
            quality_score: ModelQualityScore with composite and component scores
        """
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

        # TensorBoard logging
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        # WandB logging
        self.accumulate_wandb_metrics(metrics, step)

    def log_per_layer_gradients(
        self,
        step: int,
        layer_stats: List['LayerGradientStats']
    ) -> None:
        """
        Log per-layer gradient statistics.

        Args:
            step: Current training step
            layer_stats: List of LayerGradientStats from DiagnosticsManager
        """
        for stat in layer_stats:
            # Sanitize layer name for metric key (replace dots with slashes)
            layer_key = stat.layer_name.replace('.', '/')
            prefix = f'gradients/layer/{layer_key}'

            metrics = {
                f'{prefix}/norm': stat.grad_norm,
                f'{prefix}/mean': stat.grad_mean,
                f'{prefix}/std': stat.grad_std,
                f'{prefix}/max': stat.grad_max,
                f'{prefix}/min': stat.grad_min,
            }

            # TensorBoard logging
            if self._writer is not None:
                for key, value in metrics.items():
                    self._writer.add_scalar(key, value, step)

            # WandB logging
            self.accumulate_wandb_metrics(metrics, step)

    def log_routing_diagnostics(
        self,
        step: int,
        routing_stats: 'ExpertRoutingStats'
    ) -> None:
        """
        Log expert routing diagnostics.

        Args:
            step: Current training step
            routing_stats: ExpertRoutingStats from DiagnosticsManager
        """
        metrics = {
            'routing/balance_score': routing_stats.balance_score,
            'routing/routing_entropy': routing_stats.routing_entropy,
            'routing/dropped_tokens': routing_stats.dropped_tokens,
        }

        # Per-expert load
        for expert_id, load in routing_stats.per_expert_load.items():
            metrics[f'routing/expert_{expert_id}_load'] = load

        # Per-expert capacity usage
        for expert_id, usage in routing_stats.capacity_usage.items():
            metrics[f'routing/expert_{expert_id}_capacity'] = usage

        # TensorBoard logging
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        # WandB logging
        self.accumulate_wandb_metrics(metrics, step)

    def log_memory_breakdown(
        self,
        step: int,
        breakdown: 'MemoryBreakdown'
    ) -> None:
        """
        Log memory breakdown metrics.

        Args:
            step: Current training step
            breakdown: MemoryBreakdown from DiagnosticsManager
        """
        metrics = {
            'memory/total_allocated_gb': breakdown.total_allocated_gb,
            'memory/total_reserved_gb': breakdown.total_reserved_gb,
            'memory/parameters_gb': breakdown.parameters_gb,
            'memory/gradients_gb': breakdown.gradients_gb,
            'memory/optimizer_states_gb': breakdown.optimizer_states_gb,
            'memory/activations_gb': breakdown.activations_gb,
            'memory/other_gb': breakdown.other_gb,
        }

        # TensorBoard logging
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        # WandB logging
        self.accumulate_wandb_metrics(metrics, step)

    def log_timing_profile(
        self,
        step: int,
        profile: 'TimingProfile'
    ) -> None:
        """
        Log timing profile metrics.

        Args:
            step: Current training step
            profile: TimingProfile from DiagnosticsManager
        """
        metrics = {
            'timing/data_loading_ms': profile.data_loading_ms,
            'timing/forward_ms': profile.forward_ms,
            'timing/backward_ms': profile.backward_ms,
            'timing/optimizer_step_ms': profile.optimizer_step_ms,
            'timing/total_step_ms': profile.total_step_ms,
        }

        # TensorBoard logging
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)

        # WandB logging
        self.accumulate_wandb_metrics(metrics, step)

    def log_generation(self, log_step: int, generation_data: Dict[str, Any]) -> None:
        """
        Log generation to WandB table (accumulates all generations).

        Args:
            log_step: Current training step (for wandb.log to avoid monotonic warning)
            generation_data: Dictionary with generation details (includes 'step' for table)
        """
        # Debug: Check if WandB is enabled
        try:
            from tqdm import tqdm
            if not self._use_wandb:
                tqdm.write(f"  [Gen DEBUG] log_generation called but _use_wandb={self._use_wandb} - skipping WandB log")
            else:
                tqdm.write(f"  [Gen DEBUG] log_generation called with _use_wandb=True, WANDB_AVAILABLE={WANDB_AVAILABLE}")
        except ImportError:
            pass

        if not self._use_wandb:
            return

        try:
            # Accumulate row data
            row = (
                generation_data.get('step', log_step),  # Actual step when generation started
                generation_data.get('prompt', ''),
                generation_data.get('generated_text', ''),
                generation_data.get('temperature', 1.0),
                generation_data.get('top_p', 1.0),
                generation_data.get('top_k', 50),
            )
            self._generation_rows.append(row)

            # Create fresh table with all accumulated rows (WandB tables are immutable after log)
            table = wandb.Table(columns=[
                "step", "prompt", "generated_text", "temperature", "top_p", "top_k"
            ])
            for r in self._generation_rows:
                table.add_data(*r)

            # Log table at CURRENT step (to avoid monotonic step warning)
            wandb.log({"generations": table}, step=log_step)

            # Debug: confirm table was logged
            try:
                from tqdm import tqdm
                tqdm.write(f"  [Gen] Logged {len(self._generation_rows)} generation(s) to WandB table at step {log_step}")
            except ImportError:
                pass

            # Notify async logger of this sync log to prevent stale entries
            if self._async_logger is not None:
                self._async_logger.update_last_step(log_step)

        except Exception as e:
            self.logger.warning(f"Failed to log generation to WandB: {e}")
            # Also print to console for visibility
            try:
                from tqdm import tqdm
                tqdm.write(f"  [Gen ERROR] WandB table log failed: {e}")
            except ImportError:
                print(f"  [Gen ERROR] WandB table log failed: {e}")

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

            # Include step for proper timeline tracking
            final_step = generations[-1].get('step', 0) if generations else 0
            wandb.log({"final_generations": table}, step=final_step)

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
        """Flush metrics at epoch end and reset epoch counters."""
        if self._writer is not None:
            self._writer.flush()

        # Flush async logger at epoch end to ensure metrics are sent
        if self._async_logger is not None:
            self._async_logger.flush(timeout=30.0)

        # Reset epoch average counters for next epoch
        self._total_epoch_loss = 0.0
        self._epoch_step_count = 0
        # Note: EMA loss is NOT reset - it carries over across epochs for smooth tracking

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
