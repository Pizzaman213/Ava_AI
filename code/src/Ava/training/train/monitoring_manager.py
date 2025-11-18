"""
MonitoringManager - Tracks metrics, logs, and experiment tracking.

Responsibilities:
- Metrics collection and aggregation
- W&B integration
- Logging
- Performance metrics (throughput, memory, etc.)
- Training statistics
"""

import logging
import time
from typing import Any, Dict, Optional

import torch

from .base import TrainingContext, ManagerInterface

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MonitoringManager(ManagerInterface):
    """
    Manages all monitoring, logging, and experiment tracking.

    Handles metrics collection, W&B integration, and performance monitoring.
    """

    def __init__(self, context: TrainingContext):
        """Initialize monitoring manager."""
        super().__init__(context)
        self.metrics: Dict[str, float] = {}
        self.step_metrics: Dict[str, float] = {}
        self.use_wandb = False
        self.wandb_project: Optional[str] = None
        self.wandb_entity: Optional[str] = None
        self.log_frequency = 100

        # Performance tracking
        self.step_times: list = []
        self.max_step_history = 100
        self.epoch_start_time = 0.0
        self.step_start_time = 0.0

    def initialize(self) -> None:
        """Initialize monitoring manager."""
        # Load config
        if self.config:
            self.log_frequency = getattr(
                self.config.training, "log_frequency", 100
            )

            # Setup W&B if configured
            wandb_config = getattr(self.config, "wandb", None)
            if wandb_config and WANDB_AVAILABLE:
                self.use_wandb = getattr(wandb_config, "enabled", False)
                self.wandb_project = getattr(
                    wandb_config, "project", "ava-training"
                )
                self.wandb_entity = getattr(wandb_config, "entity", None)

                if self.use_wandb:
                    self._init_wandb()

        logger.info(
            f"Monitoring initialized: "
            f"log_frequency={self.log_frequency}, "
            f"wandb_enabled={self.use_wandb}"
        )

        super().initialize()

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        if self.use_wandb:
            try:
                wandb.finish()
                logger.info("W&B monitoring finished")
            except Exception as e:
                logger.warning(f"Error finishing W&B: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Step number (uses context.step if not provided)
        """
        if step is None:
            step = self.context.step

        self.metrics.update(metrics)
        self.step_metrics = metrics.copy()

        # Log to W&B
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Error logging to W&B: {e}")

        # Log to console (sampling)
        if step % self.log_frequency == 0:
            msg = f"Step {step}: "
            for name, value in metrics.items():
                if isinstance(value, float):
                    msg += f"{name}={value:.4f} "
                else:
                    msg += f"{name}={value} "
            logger.info(msg)

    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        """
        Log training step information.

        Args:
            step: Current step
            epoch: Current epoch
            loss: Loss value
            learning_rate: Learning rate
            grad_norm: Gradient norm
        """
        self.context.step = step
        self.context.epoch = epoch

        # Calculate step time
        if self.step_start_time > 0:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            if len(self.step_times) > self.max_step_history:
                self.step_times.pop(0)

        # Build metrics
        metrics = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        if self.step_times:
            metrics["step_time_ms"] = (sum(self.step_times) / len(self.step_times)) * 1000

        self.log_metrics(metrics, step=step)
        self.step_start_time = time.time()

    def on_epoch_start(self, epoch: int) -> None:
        """Called at epoch start."""
        self.context.epoch = epoch
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch: int) -> None:
        """Called at epoch end."""
        if self.epoch_start_time > 0:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

    def on_step_start(self, step: int) -> None:
        """Called at step start."""
        self.step_start_time = time.time()

    def on_step_end(self, step: int, loss: float) -> None:
        """Called at step end."""
        self.context.step = step
        self.context.current_loss = loss

    def get_average_step_time(self) -> float:
        """Get average step time in seconds."""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def get_throughput(self, batch_size: int) -> float:
        """
        Get training throughput in samples/second.

        Args:
            batch_size: Batch size

        Returns:
            Samples per second
        """
        avg_step_time = self.get_average_step_time()
        if avg_step_time == 0:
            return 0.0
        return batch_size / avg_step_time

    def log_memory_stats(self) -> None:
        """Log GPU memory statistics."""
        if not torch.cuda.is_available():
            return

        try:
            metrics = {
                "memory/allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory/reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "memory/max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }

            self.log_metrics(metrics)

        except Exception as e:
            logger.warning(f"Error logging memory stats: {e}")

    def log_model_stats(self) -> None:
        """Log model statistics."""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            metrics = {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            }

            self.log_metrics(metrics)

        except Exception as e:
            logger.warning(f"Error logging model stats: {e}")

    def _init_wandb(self) -> None:
        """Initialize W&B logging."""
        try:
            run_name = None
            if self.context.run_manager:
                run_name = self.context.run_manager.run_id

            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_name,
                config=vars(self.config) if self.config else {},
            )

            logger.info(
                f"W&B initialized: project={self.wandb_project}, "
                f"run={run_name}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def get_status(self) -> Dict[str, Any]:
        """Return monitoring manager status."""
        return {
            "metrics": self.step_metrics,
            "use_wandb": self.use_wandb,
            "log_frequency": self.log_frequency,
            "avg_step_time_ms": self.get_average_step_time() * 1000,
            "step_count": self.context.step,
            "epoch": self.context.epoch,
        }
