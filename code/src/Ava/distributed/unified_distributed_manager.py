"""
Unified Distributed Training Manager

This module provides a unified interface for distributed training
using PyTorch's native DDP (DistributedDataParallel).
"""

import logging
from contextlib import contextmanager
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
            assert self._model is not None and self._optimizer is not None, "Model and optimizer should be set if initialized"
            return self._model, self._optimizer, criterion, dataloader

        # Use native DDP with optimized settings
        if torch.cuda.is_available():
            # PHASE 3 OPTIMIZATION: Removed unnecessary barrier before model init (saves ~10-50ms)
            # Barrier only needed after DDP wrapping to ensure all ranks are ready
            model = model.cuda()

            if dist.is_initialized():
                # Calculate model size for optimal bucket size
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

                # Get configuration for bucket size (now configurable)
                bucket_cap_mb = getattr(self.training_config.distributed, 'ddp_bucket_cap_mb', None)  # type: ignore[attr-defined]
                if bucket_cap_mb is None:
                    # Adaptive bucket size with sequence length consideration
                    seq_len = getattr(self.training_config.model, 'max_position_embeddings', 2048)
                    batch_size = getattr(self.training_config.training, 'batch_size', 1)

                    # Consider both model size and data throughput
                    data_factor = (seq_len * batch_size) / (2048 * 32)  # Normalize to standard config

                    if model_size_mb > 1000:
                        bucket_cap_mb = min(200, int(100 * data_factor))
                    elif model_size_mb > 500:
                        bucket_cap_mb = min(100, int(50 * data_factor))
                    else:
                        bucket_cap_mb = min(50, int(25 * data_factor))

                # Validate bucket size
                if bucket_cap_mb > model_size_mb:
                    logger.warning(f"Bucket size {bucket_cap_mb}MB exceeds model size {model_size_mb:.1f}MB, adjusting...")
                    bucket_cap_mb = int(model_size_mb * 0.1)  # Use 10% of model size

                # PHASE 3 OPTIMIZATION: Removed barrier before DDP wrapping (not required, saves ~10-50ms)

                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    output_device=torch.cuda.current_device(),
                    find_unused_parameters=False,
                    bucket_cap_mb=bucket_cap_mb,  # Optimized bucket size
                    gradient_as_bucket_view=True,  # OPTIMIZED: Avoid gradient copy overhead
                    broadcast_buffers=True,  # Sync batch norm, etc.
                    static_graph=True,  # OPTIMIZED: Skip graph analysis after first iteration
                )

                # Single barrier after DDP initialization to ensure all ranks are ready
                dist.barrier()
                logger.info(f"Rank {dist.get_rank()}: ✓ Model wrapped with DDP (bucket_cap_mb={bucket_cap_mb}, model_size={model_size_mb:.1f}MB)")

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

    @contextmanager
    def no_sync_context(self):
        """
        Context manager to skip gradient synchronization during gradient accumulation.

        This significantly reduces communication overhead by only syncing gradients
        on the final accumulation step.

        Usage:
            with distributed_manager.no_sync_context():
                loss.backward()  # No gradient sync

        Yields:
            None
        """
        if self._model and isinstance(self._model, nn.parallel.DistributedDataParallel):
            with self._model.no_sync():
                yield
        else:
            # No DDP, just pass through
            yield

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
        # Convert string op to ReduceOp enum
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }

        if self.native_manager:
            reduce_op = op_map.get(op, dist.ReduceOp.SUM)
            result = self.native_manager.all_reduce(tensor, reduce_op)  # type: ignore[arg-type]
            # Handle mean operation
            if op == "mean" and isinstance(result, torch.Tensor):
                result = result / dist.get_world_size()
            return result if isinstance(result, torch.Tensor) else tensor
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
            result = self.native_manager.broadcast(tensor, src)
            return result if isinstance(result, torch.Tensor) else tensor
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
        # Synchronize all ranks before checkpoint save
        if dist.is_initialized():
            dist.barrier()

        if not self.is_main_process:
            # Non-main processes wait for main to finish saving
            if dist.is_initialized():
                dist.barrier()
            return

        # Standard checkpoint saving (only on main process)
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

        # Synchronize after save to ensure all ranks wait for completion
        if dist.is_initialized():
            dist.barrier()

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
        logger.warning(f"Rank {self.rank}: Handling OOM error in unified manager")

        # Coordinate OOM detection across all ranks
        if dist.is_initialized():
            # Create tensor to track OOM status (1 = OOM, 0 = OK)
            oom_flag = torch.tensor([1], dtype=torch.int32)
            if torch.cuda.is_available():
                oom_flag = oom_flag.cuda()

            # All-reduce to check if ANY rank has OOM
            dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)

            # If any rank has OOM, all ranks should handle it
            if oom_flag.item() > 0:
                logger.info(f"Rank {self.rank}: Collective OOM detected, all ranks clearing cache")
                # Synchronize before cleanup
                dist.barrier()

                # All ranks clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Synchronize after cleanup
                dist.barrier()
                logger.info(f"Rank {self.rank}: OOM recovery complete")
                return True

        # Use native manager's OOM handling if available
        elif self.native_manager:
            return self.native_manager.coordinate_oom_recovery()

        # Fallback: Basic cleanup for non-distributed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def check_collective_oom(self) -> bool:
        """
        Check if any rank is experiencing OOM pressure

        Returns:
            True if any rank is near OOM, False otherwise
        """
        if not dist.is_initialized():
            return False

        # Check local memory pressure
        local_oom = False
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            # Consider OOM if > 95% of reserved memory is allocated
            if reserved > 0:
                usage_ratio = allocated / reserved
                local_oom = usage_ratio > 0.95

        # Share OOM status across all ranks
        oom_tensor = torch.tensor([int(local_oom)], dtype=torch.int32)
        if torch.cuda.is_available():
            oom_tensor = oom_tensor.cuda()

        dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
        return oom_tensor.item() > 0

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
