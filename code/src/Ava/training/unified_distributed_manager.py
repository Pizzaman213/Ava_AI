"""
Unified Distributed Training Manager

This module provides a unified interface that seamlessly integrates
Colossal-AI with the existing DistributedManager, allowing for
flexible switching between different parallelization strategies.
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..config.training_config import EnhancedTrainingConfig
from .colossalai_integration import (
    ColossalAIConfig,
    ColossalAIIntegration,
    ParallelismStrategy,
    create_colossalai_integration,
)
from .distributed_manager import DistributedManager

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Available distributed training backends"""
    NATIVE = "native"  # Existing DistributedManager
    COLOSSALAI = "colossalai"  # Colossal-AI
    HYBRID = "hybrid"  # Use both for different aspects


class UnifiedDistributedManager:
    """
    Unified manager that provides a consistent interface for both
    native distributed training and Colossal-AI parallelization.
    """

    def __init__(
        self,
        training_config: EnhancedTrainingConfig,
        backend: DistributedBackend = DistributedBackend.HYBRID,
        colossalai_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize unified distributed manager

        Args:
            training_config: Training configuration
            backend: Which backend to use
            colossalai_config: Optional Colossal-AI configuration overrides
        """
        self.training_config = training_config
        self.backend = backend
        self.colossalai_config = colossalai_config or {}

        # Initialize components
        self.native_manager: Optional[DistributedManager] = None
        self.colossalai_integration: Optional[ColossalAIIntegration] = None

        # Track initialization state
        self._initialized = False
        self._model = None
        self._optimizer = None

        # Setup based on backend
        self._setup_backend()

    def _setup_backend(self):
        """Setup the selected backend"""
        if self.backend in [DistributedBackend.NATIVE, DistributedBackend.HYBRID]:
            # Initialize native distributed manager
            self.native_manager = DistributedManager(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout_minutes=30,
                enable_heartbeat=True,
                heartbeat_interval=10,
            )
            self.native_manager.initialize()

        if self.backend in [DistributedBackend.COLOSSALAI, DistributedBackend.HYBRID]:
            # Parse Colossal-AI config from training config if available
            if hasattr(self.training_config, "colossalai"):
                # Merge YAML config with overrides
                config_dict = self.training_config.colossalai.__dict__.copy()
                config_dict.update(self.colossalai_config)
                self.colossalai_config = config_dict

            # Create Colossal-AI integration
            self.colossalai_integration = create_colossalai_integration(
                self.training_config,
                self.native_manager,
                **self.colossalai_config,
            )

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

        # Choose initialization path based on backend
        if self.backend == DistributedBackend.COLOSSALAI:
            # Use Colossal-AI exclusively
            model, optimizer, criterion, dataloader = self.colossalai_integration.initialize(
                model, optimizer, criterion, dataloader, lr_scheduler
            )

        elif self.backend == DistributedBackend.NATIVE:
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

        else:  # HYBRID mode
            # Use Colossal-AI for parallelization but keep native manager for coordination
            if self.colossalai_integration:
                model, optimizer, criterion, dataloader = self.colossalai_integration.initialize(
                    model, optimizer, criterion, dataloader, lr_scheduler
                )
            else:
                # Fallback to native if Colossal-AI fails
                if torch.cuda.is_available():
                    model = model.cuda()
                    if dist.is_initialized():
                        model = nn.parallel.DistributedDataParallel(
                            model,
                            device_ids=[torch.cuda.current_device()],
                            output_device=torch.cuda.current_device(),
                            find_unused_parameters=False,
                        )

        self._model = model
        self._optimizer = optimizer
        self._initialized = True

        # Log initialization details
        self._log_initialization_info()

        return model, optimizer, criterion, dataloader

    def _log_initialization_info(self):
        """Log detailed initialization information"""
        info = [f"Unified Distributed Manager initialized with backend: {self.backend.value}"]

        if dist.is_initialized():
            info.append(f"World size: {dist.get_world_size()}")
            info.append(f"Rank: {dist.get_rank()}")

        if self.colossalai_integration and self.colossalai_integration._initialized:
            config = self.colossalai_integration.config
            info.append(f"Colossal-AI strategy: {config.parallel_strategy.value}")
            info.append(f"Tensor parallel size: {config.tensor_parallel_size}")
            info.append(f"Pipeline parallel size: {config.pipeline_parallel_size}")
            info.append(f"ZeRO stage: {config.zero_stage if config.use_zero else 'disabled'}")
            info.append(f"Mixed precision: {config.mixed_precision}")

            # Memory usage
            mem_stats = self.colossalai_integration.get_memory_usage()
            if mem_stats:
                info.append(f"GPU Memory: {mem_stats.get('gpu_allocated_gb', 0):.2f}GB allocated")

        logger.info("\n".join(info))

    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Perform backward pass

        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
        """
        if self.colossalai_integration and self.backend != DistributedBackend.NATIVE:
            self.colossalai_integration.backward(loss, self._optimizer, retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def optimizer_step(self, lr_scheduler: Optional[Any] = None):
        """
        Perform optimizer step

        Args:
            lr_scheduler: Optional learning rate scheduler
        """
        if self.colossalai_integration and self.backend != DistributedBackend.NATIVE:
            self.colossalai_integration.step(self._optimizer, lr_scheduler)
        else:
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
        elif self.colossalai_integration:
            self.colossalai_integration.barrier()
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

        if self.colossalai_integration and self.backend != DistributedBackend.NATIVE:
            self.colossalai_integration.save_checkpoint(
                checkpoint_path,
                self._model,
                self._optimizer,
                epoch,
                step,
                best_loss=best_loss,
                **kwargs,
            )
        else:
            # Standard checkpoint saving
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
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
        if self.colossalai_integration and self.backend != DistributedBackend.NATIVE:
            return self.colossalai_integration.load_checkpoint(
                checkpoint_path,
                self._model,
                self._optimizer,
                strict,
            )
        else:
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
            Dictionary with memory stats from all sources
        """
        stats = {}

        # Get native manager stats
        if self.native_manager:
            native_stats = self.native_manager.get_memory_stats()
            stats["native"] = native_stats

        # Get Colossal-AI stats
        if self.colossalai_integration:
            colossal_stats = self.colossalai_integration.get_memory_usage()
            stats["colossalai"] = colossal_stats

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
            "backend": self.backend.value,
            "initialized": self._initialized,
            "distributed": dist.is_initialized(),
        }

        if self.native_manager:
            health["native_health"] = self.native_manager.check_health()

        if self.colossalai_integration:
            health["colossalai_initialized"] = self.colossalai_integration._initialized

        # Memory health
        mem_stats = self.get_memory_stats()
        if "gpu" in mem_stats:
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

        # Try Colossal-AI memory optimization first
        if self.colossalai_integration:
            self.colossalai_integration.optimize_memory()

        # Use native manager's OOM handling
        if self.native_manager:
            return self.native_manager.coordinate_oom_recovery()

        # Basic cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def enable_activation_checkpointing(self, num_layers: Optional[int] = None):
        """
        Enable activation checkpointing for memory efficiency

        Args:
            num_layers: Number of layers to checkpoint (None = auto)
        """
        if self.colossalai_integration and self._model:
            self.colossalai_integration.enable_activation_checkpointing(
                self._model, num_layers
            )
        else:
            logger.warning("Activation checkpointing requires Colossal-AI integration")

    def get_effective_batch_size(self) -> int:
        """
        Get effective batch size accounting for parallelism

        Returns:
            Effective batch size across all devices
        """
        if self.colossalai_integration:
            return self.colossalai_integration.get_effective_batch_size()

        # Calculate manually
        batch_size = self.training_config.training.batch_size
        grad_accum = self.training_config.training.gradient_accumulation_steps
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        return batch_size * grad_accum * world_size

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        if self.colossalai_integration:
            return self.colossalai_integration.is_main_process
        elif self.native_manager:
            return self.native_manager.is_main_process
        elif dist.is_initialized():
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
        if self.colossalai_integration:
            self.colossalai_integration.cleanup()

        if self.native_manager:
            self.native_manager.cleanup()

        self._initialized = False
        logger.info("Unified distributed manager cleaned up")


def create_unified_manager(
    training_config: EnhancedTrainingConfig,
    prefer_colossalai: bool = True,
) -> UnifiedDistributedManager:
    """
    Factory function to create unified distributed manager

    Args:
        training_config: Training configuration
        prefer_colossalai: Whether to prefer Colossal-AI when available

    Returns:
        UnifiedDistributedManager instance
    """
    # Check if Colossal-AI config exists and is enabled
    use_colossalai = False
    if hasattr(training_config, "colossalai"):
        use_colossalai = getattr(training_config.colossalai, "enabled", False)

    # Determine backend
    if use_colossalai and prefer_colossalai:
        backend = DistributedBackend.HYBRID
        logger.info("Using HYBRID backend with Colossal-AI integration")
    else:
        backend = DistributedBackend.NATIVE
        logger.info("Using NATIVE backend with PyTorch DDP")

    return UnifiedDistributedManager(training_config, backend)