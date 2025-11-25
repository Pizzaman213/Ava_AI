"""
SimplifiedEnhancedTrainer - Composes all training managers.

This is the main training orchestrator that brings together all the
modular components (distributed, checkpoint, loss, monitoring) into
a cohesive training loop.

The trainer now focuses on orchestration and data flow rather than
implementing details of each responsibility.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...config.training_config import EnhancedTrainingConfig
from .base import TrainingContext
from .distributed_manager import DistributedTrainingManager
from .checkpoint_manager import CheckpointManager
from .loss_manager import LossComputationManager
from .monitoring_manager import MonitoringManager

logger = logging.getLogger(__name__)


class SimplifiedEnhancedTrainer:
    """
    Simplified trainer that composes modular training components.

    Responsibilities:
    - Orchestrate training loop
    - Coordinate between managers
    - Handle training phases (warmup, training, evaluation)
    - Manage overall training state

    Each specialized responsibility is delegated to appropriate manager.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[EnhancedTrainingConfig] = None,
        run_manager: Optional[Any] = None,
    ):
        """
        Initialize simplified trainer.

        Args:
            model: PyTorch model to train
            tokenizer: Tokenizer (optional)
            device: Device to train on
            config: Training configuration
            run_manager: Run manager for tracking
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.run_manager = run_manager

        # Create shared context for all managers
        self.context = TrainingContext(
            model=model,
            device=self.device,
            config=config,
            run_manager=run_manager,
        )

        # Initialize all managers
        self.distributed_manager = DistributedTrainingManager(self.context)
        self.checkpoint_manager = CheckpointManager(self.context)
        self.loss_manager = LossComputationManager(self.context)
        self.monitoring_manager = MonitoringManager(self.context)

        self._managers = [
            self.distributed_manager,
            self.checkpoint_manager,
            self.loss_manager,
            self.monitoring_manager,
        ]

        self._initialized = False
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def initialize(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Initialize trainer with optimizer.

        Args:
            optimizer: Optimizer to use for training
        """
        logger.info("Initializing trainer and all components...")

        self.optimizer = optimizer
        self.context.optimizer = optimizer

        # Initialize all managers in order
        for manager in self._managers:
            try:
                manager.initialize()
                logger.info(f" {manager.__class__.__name__} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize {manager.__class__.__name__}: {e}")
                raise

        self._initialized = True
        logger.info(" Trainer fully initialized")

    def cleanup(self) -> None:
        """Cleanup all resources."""
        if not self._initialized:
            return

        logger.info("Cleaning up trainer resources...")

        # Cleanup in reverse order
        for manager in reversed(self._managers):
            try:
                manager.cleanup()
                logger.info(f" {manager.__class__.__name__} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up {manager.__class__.__name__}: {e}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Input batch from dataloader
            loss_fn_outputs: Optional pre-computed outputs (for loss functions)

        Returns:
            Dictionary of metrics from this step
        """
        self._assert_initialized()

        step = self.context.step
        self.monitoring_manager.on_step_start(step)

        try:
            # Forward pass
            if loss_fn_outputs is None:
                # Extract inputs from batch - try common patterns
                if "input_ids" in batch:
                    # HuggingFace-style batch: just pass the input tensor
                    model_input = batch["input_ids"]
                    outputs = self.model(model_input)
                elif "inputs" in batch:
                    # Alternative style
                    model_input = batch["inputs"]
                    outputs = self.model(model_input)
                else:
                    # Fall back to passing all non-label keys
                    model_inputs = {k: v for k, v in batch.items()
                                   if k not in ['labels', 'targets']}
                    outputs = self.model(**model_inputs)
            else:
                outputs = loss_fn_outputs

            # Get targets from batch
            targets = batch.get("labels", batch.get("targets"))
            if targets is None:
                raise ValueError("Batch must contain 'labels' or 'targets'")

            # Compute loss
            loss, loss_breakdown = self.loss_manager.compute_loss(
                outputs, targets
            )

            # Backward pass
            self.loss_manager.backward(loss)

            # Gradient clipping
            grad_norm = self.loss_manager.clip_gradients(
                self.model.parameters()
            )

            # Optimizer step
            self.loss_manager.optimizer_step()

            # Update training state
            self.context.step += 1
            self.context.micro_step += 1

            # Log metrics
            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
            }
            metrics.update(loss_breakdown)

            self.monitoring_manager.log_training_step(
                step=step,
                epoch=self.context.epoch,
                loss=loss.item(),
                grad_norm=grad_norm,
            )

            self.monitoring_manager.on_step_end(step, loss.item())

            return metrics

        except Exception as e:
            logger.error(f"Error in training step {step}: {e}")
            for manager in self._managers:
                manager.on_error(e)
            raise

    def train_epoch(
        self,
        train_loader,
        eval_loader: Optional[Any] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
            steps_per_epoch: Limit steps per epoch (optional)

        Returns:
            Dictionary of epoch metrics
        """
        self._assert_initialized()

        epoch = self.context.epoch
        self.monitoring_manager.on_epoch_start(epoch)

        epoch_metrics = {
            "epoch": epoch,
            "total_loss": 0.0,
            "steps": 0,
        }

        try:
            for step_idx, batch in enumerate(train_loader):
                # Limit steps if configured
                if steps_per_epoch and step_idx >= steps_per_epoch:
                    break

                # Move batch to device
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # Execute training step
                step_metrics = self.train_step(batch)

                # Accumulate metrics
                epoch_metrics["total_loss"] += step_metrics.get("loss", 0.0)
                epoch_metrics["steps"] += 1

                # Checkpoint periodically
                checkpoint_freq = getattr(
                    self.config.training if self.config else None,
                    "checkpoint_frequency",
                    1000,
                )
                if checkpoint_freq > 0 and self.context.step % checkpoint_freq == 0:
                    self.save_checkpoint()

            # Compute epoch averages
            if epoch_metrics["steps"] > 0:
                epoch_metrics["avg_loss"] = (
                    epoch_metrics["total_loss"] / epoch_metrics["steps"]
                )

            # Evaluation
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                epoch_metrics.update(eval_metrics)

            self.monitoring_manager.on_epoch_end(epoch)
            self.context.epoch += 1

            return epoch_metrics

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}")
            raise

    def evaluate(self, eval_loader) -> Dict[str, float]:
        """
        Evaluate model on evaluation set.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self._assert_initialized()

        self.model.eval()
        eval_metrics = {
            "eval_loss": 0.0,
            "eval_steps": 0,
        }

        try:
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {
                        k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

                    # Forward pass - handle input extraction
                    if "input_ids" in batch:
                        model_input = batch["input_ids"]
                        outputs = self.model(model_input)
                    elif "inputs" in batch:
                        model_input = batch["inputs"]
                        outputs = self.model(model_input)
                    else:
                        model_inputs = {k: v for k, v in batch.items()
                                       if k not in ['labels', 'targets']}
                        outputs = self.model(**model_inputs)

                    # Compute loss
                    targets = batch.get("labels", batch.get("targets"))
                    loss, _ = self.loss_manager.compute_loss(outputs, targets)

                    eval_metrics["eval_loss"] += loss.item()
                    eval_metrics["eval_steps"] += 1

            if eval_metrics["eval_steps"] > 0:
                eval_metrics["eval_loss"] /= eval_metrics["eval_steps"]

            logger.info(f"Evaluation: loss={eval_metrics['eval_loss']:.4f}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
        finally:
            self.model.train()

        return eval_metrics

    def save_checkpoint(
        self,
        save_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            save_best: Whether to save as best checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        self._assert_initialized()

        try:
            if save_best:
                loss = self.context.current_loss
                self.checkpoint_manager.save_best_checkpoint(
                    loss=loss,
                    epoch=self.context.epoch,
                    step=self.context.step,
                    optimizer=self.optimizer,
                    metrics=metrics,
                )
            else:
                self.checkpoint_manager.save_checkpoint(
                    epoch=self.context.epoch,
                    step=self.context.step,
                    optimizer=self.optimizer,
                    metrics=metrics,
                )

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self._assert_initialized()

        try:
            from pathlib import Path
            ckpt_metadata = self.checkpoint_manager.load_checkpoint(
                Path(checkpoint_path)
            )

            self.context.epoch = ckpt_metadata.get("epoch", 0)
            self.context.step = ckpt_metadata.get("step", 0)
            self.context.current_loss = ckpt_metadata.get("loss", float("inf"))

            logger.info(
                f"Loaded checkpoint: epoch={self.context.epoch}, "
                f"step={self.context.step}"
            )

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get overall trainer status from all managers."""
        return {
            "training_context": {
                "epoch": self.context.epoch,
                "step": self.context.step,
                "current_loss": self.context.current_loss,
            },
            "distributed": self.distributed_manager.get_status(),
            "checkpoint": self.checkpoint_manager.get_status(),
            "loss": self.loss_manager.get_status(),
            "monitoring": self.monitoring_manager.get_status(),
        }

    def _assert_initialized(self) -> None:
        """Ensure trainer is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "Trainer not initialized. Call initialize(optimizer) first."
            )
