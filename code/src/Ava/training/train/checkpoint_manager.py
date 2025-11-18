"""
CheckpointManager - Handles model state persistence and recovery.

Responsibilities:
- Checkpoint saving (async and sync)
- Checkpoint loading and validation
- Best model tracking
- State dict management
- Checkpoint cleanup and rotation
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import TrainingContext, ManagerInterface

logger = logging.getLogger(__name__)


class CheckpointManager(ManagerInterface):
    """
    Manages model checkpointing and recovery.

    Handles saving/loading model state, tracking best model,
    and asynchronous checkpoint operations.
    """

    def __init__(self, context: TrainingContext):
        """Initialize checkpoint manager."""
        super().__init__(context)
        self.checkpoint_dir: Optional[Path] = None
        self.best_loss = float("inf")
        self.checkpoint_count = 0

        # Async checkpoint saving
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._checkpoint_lock = threading.Lock()
        self._pending_checkpoint: Optional[Dict[str, Any]] = None
        self._stop_async = False

    def initialize(self) -> None:
        """Initialize checkpoint manager."""
        # Create checkpoint directory from run manager if available
        if self.context.run_manager:
            self.checkpoint_dir = Path(self.context.run_manager.checkpoint_dir)
        else:
            self.checkpoint_dir = Path("./checkpoints")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        super().initialize()

    def cleanup(self) -> None:
        """Cleanup checkpoint resources."""
        # Wait for any pending async checkpoint
        self._stop_async = True
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            logger.info("Waiting for async checkpoint to complete...")
            self._checkpoint_thread.join(timeout=60)

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
        async_save: bool = False,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            step: Current step
            optimizer: Optimizer state to save
            metrics: Training metrics to save
            async_save: Save asynchronously

        Returns:
            Path to saved checkpoint
        """
        self.assert_initialized()

        checkpoint_data = self._create_checkpoint_data(
            epoch, step, optimizer, metrics
        )

        if async_save:
            return self._save_checkpoint_async(checkpoint_data)
        else:
            return self._save_checkpoint_sync(checkpoint_data)

    def save_best_checkpoint(
        self,
        loss: float,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint if loss is better than previous best.

        Args:
            loss: Current loss
            epoch: Current epoch
            step: Current step
            optimizer: Optimizer state
            metrics: Training metrics

        Returns:
            Path to saved checkpoint if new best, None otherwise
        """
        if loss < self.best_loss:
            self.best_loss = loss
            logger.info(
                f"New best loss: {loss:.4f} (previous: {self.best_loss:.4f})"
            )

            checkpoint_data = self._create_checkpoint_data(
                epoch, step, optimizer, metrics
            )
            checkpoint_data["is_best"] = True

            return self._save_checkpoint_sync(checkpoint_data)

        return None

    def load_checkpoint(
        self, checkpoint_path: Path, load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state

        Returns:
            Checkpoint metadata (epoch, step, metrics, etc.)
        """
        self.assert_initialized()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.context.device
            )

            # Load model state
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model state")

            # Load optimizer state
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                if self.context.optimizer:
                    self.context.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )
                    logger.info("Loaded optimizer state")

            # Return metadata
            metadata = {
                "epoch": checkpoint.get("epoch", 0),
                "step": checkpoint.get("step", 0),
                "loss": checkpoint.get("loss", float("inf")),
                "metrics": checkpoint.get("metrics", {}),
            }

            logger.info(
                f"Loaded checkpoint from epoch {metadata['epoch']}, "
                f"step {metadata['step']}, loss {metadata['loss']:.4f}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the best saved checkpoint.

        Returns:
            Checkpoint metadata if found, None otherwise
        """
        self.assert_initialized()

        best_ckpt = self.checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            return self.load_checkpoint(best_ckpt)

        return None

    def _create_checkpoint_data(
        self,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create checkpoint data structure."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer else None
            ),
            "epoch": epoch,
            "step": step,
            "loss": self.context.current_loss,
            "best_loss": self.best_loss,
            "metrics": metrics or {},
            "config": vars(self.config) if self.config else {},
        }

    def _save_checkpoint_sync(self, checkpoint_data: Dict[str, Any]) -> Path:
        """Synchronously save checkpoint."""
        with self._checkpoint_lock:
            self.checkpoint_count += 1
            epoch = checkpoint_data["epoch"]
            step = checkpoint_data["step"]

            # Save regular checkpoint
            ckpt_name = f"checkpoint_ep{epoch}_step{step}.pt"
            ckpt_path = self.checkpoint_dir / ckpt_name
            torch.save(checkpoint_data, ckpt_path)

            # Save as best if marked
            if checkpoint_data.get("is_best"):
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_data, best_path)
                logger.info(f"Saved best checkpoint: {best_path}")

            logger.info(f"Saved checkpoint: {ckpt_path}")
            return ckpt_path

    def _save_checkpoint_async(self, checkpoint_data: Dict[str, Any]) -> Path:
        """Asynchronously save checkpoint."""
        with self._checkpoint_lock:
            self._pending_checkpoint = checkpoint_data

        # Start async thread if not running
        if not self._checkpoint_thread or not self._checkpoint_thread.is_alive():
            self._checkpoint_thread = threading.Thread(
                target=self._async_checkpoint_worker, daemon=True
            )
            self._checkpoint_thread.start()

        # Return expected path
        epoch = checkpoint_data["epoch"]
        step = checkpoint_data["step"]
        return self.checkpoint_dir / f"checkpoint_ep{epoch}_step{step}.pt"

    def _async_checkpoint_worker(self) -> None:
        """Worker thread for async checkpoint saving."""
        while not self._stop_async:
            with self._checkpoint_lock:
                if self._pending_checkpoint:
                    checkpoint_data = self._pending_checkpoint
                    self._pending_checkpoint = None
                else:
                    checkpoint_data = None

            if checkpoint_data:
                try:
                    self._save_checkpoint_sync(checkpoint_data)
                except Exception as e:
                    logger.error(f"Error in async checkpoint save: {e}")

            import time
            time.sleep(0.1)

    def on_step_end(self, step: int, loss: float) -> None:
        """Update current loss."""
        self.context.current_loss = loss

    def get_status(self) -> Dict[str, Any]:
        """Return checkpoint manager status."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_loss": self.best_loss,
            "checkpoint_count": self.checkpoint_count,
            "pending_checkpoint": self._pending_checkpoint is not None,
        }
