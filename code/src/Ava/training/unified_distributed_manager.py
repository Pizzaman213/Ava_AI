"""
Unified Distributed Training Manager

This module provides a unified interface for distributed training
using PyTorch's native DDP (DistributedDataParallel).
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..config.training_config import EnhancedTrainingConfig
from .distributed_manager import DistributedManager

logger = logging.getLogger(__name__)


class UnifiedDistributedManager:
    """
    Unified manager that provides a consistent interface for
    native distributed training using PyTorch DDP.
    """

    def __init__(
        self,
        training_config: EnhancedTrainingConfig,
    ):
        """
        Initialize unified distributed manager

        Args:
            training_config: Training configuration
        """
        self.training_config = training_config

        # Initialize native distributed manager
        self.native_manager: Optional[DistributedManager] = None

        # Track initialization state
        self._initialized = False
        self._model = None
        self._optimizer = None

        # Setup native backend
        self._setup_backend()

    def _setup_backend(self):
        """Setup the native PyTorch distributed backend"""
        # Initialize native distributed manager
        self.native_manager = DistributedManager()
        self.native_manager.initialize()

    def initialize_model(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> Tuple[nn.Module, Optimizer, Optional[nn.Module], Optional[DataLoader]]:
        """
        Initialize model with distributed training support

        Args:
            model: Model to distribute
            optimizer: Optimizer
            criterion: Optional loss function
            dataloader: Optional dataloader
            lr_scheduler: Optional learning rate scheduler

        Returns:
            Tuple of distributed (model, optimizer, criterion, dataloader)
        """
        if self._initialized:
            logger.warning("Manager already initialized, returning existing model")
            return self._model, self._optimizer, criterion, dataloader

        # Use native DDP
        if torch.cuda.is_available():
            model = model.cuda()
            if dist.is_initialized():
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    output_device=torch.cuda.current_device(),
                    find_unused_parameters=False,
                )
                logger.info("Model wrapped with native PyTorch DDP")

        self._model = model
        self._optimizer = optimizer
        self._initialized = True

        # Log initialization details
        self._log_initialization_info()

        return model, optimizer, criterion, dataloader

    def _log_initialization_info(self):
        """Log detailed initialization information"""
        info = ["Unified Distributed Manager initialized with native PyTorch DDP"]

        if dist.is_initialized():
            info.append(f"World size: {dist.get_world_size()}")
            info.append(f"Rank: {dist.get_rank()}")

        logger.info("\n".join(info))

    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Perform backward pass

        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
        """
        loss.backward(retain_graph=retain_graph)

    def optimizer_step(self, lr_scheduler: Optional[Any] = None):
        """
        Perform optimizer step

        Args:
            lr_scheduler: Optional learning rate scheduler
        """
        if self._optimizer:
            self._optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        All-reduce operation across all processes

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ("sum", "mean", "max", "min")

        Returns:
            Reduced tensor
        """
        if self.native_manager:
            return self.native_manager.all_reduce(tensor, op)
        elif dist.is_initialized():
            # Direct PyTorch distributed call
            if op == "sum":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif op == "mean":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= dist.get_world_size()
            elif op == "max":
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif op == "min":
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from source rank

        Args:
            tensor: Tensor to broadcast
            src: Source rank

        Returns:
            Broadcasted tensor
        """
        if self.native_manager:
            return self.native_manager.broadcast(tensor, src)
        elif dist.is_initialized():
            dist.broadcast(tensor, src)
        return tensor

    def barrier(self):
        """Synchronization barrier"""
        if self.native_manager:
            self.native_manager.barrier()
        elif dist.is_initialized():
            dist.barrier()

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        step: int,
        best_loss: Optional[float] = None,
        **kwargs,
    ):
        """
        Save checkpoint with distributed support

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            step: Current step
            best_loss: Best validation loss
            **kwargs: Additional items to save
        """
        if not self.is_main_process:
            return

        # Standard checkpoint saving
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self._model.state_dict() if self._model else {},
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else {},
            "best_loss": best_loss,
            **kwargs,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with distributed support

        Args:
            checkpoint_path: Path to load checkpoint from
            strict: Whether to strictly enforce state dict matching

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if self._model:
            self._model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if self._optimizer and "optimizer_state_dict" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics

        Returns:
            Dictionary with memory stats
        """
        stats = {}

        # Get general GPU stats
        if torch.cuda.is_available():
            stats["gpu"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }

        return stats

    def check_health(self) -> Dict[str, Any]:
        """
        Check health of distributed training

        Returns:
            Health status dictionary
        """
        health = {
            "backend": "native",
            "initialized": self._initialized,
            "distributed": dist.is_initialized(),
        }

        # Memory health
        mem_stats = self.get_memory_stats()
        if "gpu" in mem_stats and mem_stats["gpu"]["reserved_gb"] > 0:
            gpu_usage = mem_stats["gpu"]["allocated_gb"] / mem_stats["gpu"]["reserved_gb"]
            health["gpu_memory_usage"] = f"{gpu_usage * 100:.1f}%"

        return health

    def handle_oom_error(self) -> bool:
        """
        Handle OOM error with coordinated recovery

        Returns:
            True if recovery successful, False otherwise
        """
        logger.warning("Handling OOM error in unified manager")

        # Use native manager's OOM handling if available
        if self.native_manager:
            return self.native_manager.coordinate_oom_recovery()

        # Basic cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def get_effective_batch_size(self) -> int:
        """
        Get effective batch size accounting for parallelism

        Returns:
            Effective batch size across all devices
        """
        # Calculate manually
        batch_size = getattr(self.training_config.training, 'batch_size', 1)
        grad_accum = getattr(self.training_config.training, 'gradient_accumulation_steps', 1)
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        return batch_size * grad_accum * world_size

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    @property
    def rank(self) -> int:
        """Get current process rank"""
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        """Get world size"""
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    def cleanup(self):
        """Cleanup all distributed resources"""
        if self.native_manager:
            self.native_manager.cleanup()

        self._initialized = False
        logger.info("Unified distributed manager cleaned up")


def create_unified_manager(
    training_config: EnhancedTrainingConfig,
    prefer_colossalai: bool = False,  # Kept for backward compatibility but ignored
) -> UnifiedDistributedManager:
    """
    Factory function to create unified distributed manager

    Args:
        training_config: Training configuration
        prefer_colossalai: Deprecated, ignored

    Returns:
        UnifiedDistributedManager instance
    """
    logger.info("Using native PyTorch DDP backend")
    return UnifiedDistributedManager(training_config)
