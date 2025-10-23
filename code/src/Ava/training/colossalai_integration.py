"""
Colossal-AI Integration Module

This module provides seamless integration between Colossal-AI's advanced
parallelization strategies and the existing Ava training infrastructure.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Colossal-AI imports
try:
    import colossalai
    from colossalai.amp import AMP_TYPE
    from colossalai.booster import Booster
    from colossalai.booster.plugin import (
        GeminiPlugin,
        HybridParallelPlugin,
        LowLevelZeroPlugin,
        TorchDDPPlugin,
    )
    from colossalai.cluster import DistCoordinator
    from colossalai.lazy import LazyInitContext
    from colossalai.nn.optimizer import HybridAdam
    from colossalai.zero import ColoInitContext

    COLOSSALAI_AVAILABLE = True
except ImportError:
    COLOSSALAI_AVAILABLE = False
    colossalai = None

from ..config.training_config import EnhancedTrainingConfig
from .distributed_manager import DistributedManager

logger = logging.getLogger(__name__)


class ParallelismStrategy(Enum):
    """Supported parallelism strategies"""
    DATA_PARALLEL = "data"
    TENSOR_PARALLEL = "tensor"
    PIPELINE_PARALLEL = "pipeline"
    HYBRID_PARALLEL = "hybrid"
    SEQUENCE_PARALLEL = "sequence"
    AUTO_PARALLEL = "auto"
    ZERO1 = "zero1"
    ZERO2 = "zero2"
    ZERO3 = "zero3"


@dataclass
class ColossalAIConfig:
    """Configuration for Colossal-AI integration"""

    # Parallelism configuration
    parallel_strategy: ParallelismStrategy = ParallelismStrategy.HYBRID_PARALLEL
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = -1  # -1 means auto-calculate
    sequence_parallel_size: int = 1

    # Memory optimization
    use_zero: bool = True
    zero_stage: int = 2  # 1, 2, or 3
    use_cpu_offload: bool = False
    use_nvme_offload: bool = False
    use_gemini: bool = False  # Gemini memory manager
    use_patrickstar: bool = False  # PatrickStar heterogeneous memory

    # Mixed precision
    mixed_precision: str = "bf16"  # "fp16", "bf16", "fp8", or "none"
    initial_scale: float = 2**16
    min_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 1000

    # Communication optimization
    use_gradient_compression: bool = True
    gradient_compression_ratio: float = 0.1
    use_hierarchical_allreduce: bool = False
    use_async_communication: bool = True

    # Checkpointing
    use_activation_checkpointing: bool = True
    checkpoint_num_layers: Optional[int] = None  # None means auto

    # Performance tuning
    enable_jit_fused: bool = True
    enable_fused_normalization: bool = True
    enable_flash_attention: bool = True
    enable_xformers: bool = False

    # Auto-parallelism (experimental)
    use_auto_parallel: bool = False
    search_space: str = "default"  # "default", "large", "custom"

    # Compatibility flags
    use_legacy_optimizer: bool = False
    compatibility_mode: bool = False  # Fallback to standard PyTorch if needed


class ColossalAIIntegration:
    """
    Main integration class that bridges Colossal-AI with existing training infrastructure
    """

    def __init__(
        self,
        config: ColossalAIConfig,
        training_config: EnhancedTrainingConfig,
        distributed_manager: Optional[DistributedManager] = None,
    ):
        """
        Initialize Colossal-AI integration

        Args:
            config: Colossal-AI specific configuration
            training_config: Existing training configuration
            distributed_manager: Optional existing distributed manager for compatibility
        """
        self.config = config
        self.training_config = training_config
        self.distributed_manager = distributed_manager
        self.booster: Optional[Booster] = None
        self.plugin = None
        self.coordinator: Optional[DistCoordinator] = None

        if not COLOSSALAI_AVAILABLE:
            logger.warning("Colossal-AI not available. Running in compatibility mode.")
            self.config.compatibility_mode = True

        self._initialized = False

    def initialize(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> Tuple[nn.Module, Optimizer, Optional[nn.Module], Optional[DataLoader]]:
        """
        Initialize Colossal-AI with model and optimizer

        Returns:
            Tuple of (model, optimizer, criterion, dataloader) wrapped by Colossal-AI
        """
        if self.config.compatibility_mode:
            logger.info("Running in compatibility mode without Colossal-AI")
            return self._compatibility_mode_init(model, optimizer, criterion, dataloader)

        try:
            # Select and configure plugin based on strategy
            self.plugin = self._create_plugin()

            # Create booster
            self.booster = Booster(plugin=self.plugin)

            # Boost model, optimizer, criterion, and dataloader
            model, optimizer, criterion, dataloader, lr_scheduler = self.booster.boost(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                dataloader=dataloader,
                lr_scheduler=lr_scheduler,
            )

            self._initialized = True
            logger.info(f"Successfully initialized Colossal-AI with {self.config.parallel_strategy.value} strategy")

            # Log memory usage after initialization
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory after Colossal-AI init: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            return model, optimizer, criterion, dataloader

        except Exception as e:
            logger.error(f"Failed to initialize Colossal-AI: {e}")
            logger.info("Falling back to compatibility mode")
            self.config.compatibility_mode = True
            return self._compatibility_mode_init(model, optimizer, criterion, dataloader)

    def _create_plugin(self):
        """Create appropriate Colossal-AI plugin based on configuration"""

        # Determine mixed precision type
        if self.config.mixed_precision == "fp16":
            precision = "fp16"
            amp_type = AMP_TYPE.TORCH
        elif self.config.mixed_precision == "bf16":
            precision = "bf16"
            amp_type = AMP_TYPE.TORCH
        else:
            precision = None
            amp_type = None

        # Create plugin based on strategy
        if self.config.parallel_strategy == ParallelismStrategy.ZERO1:
            plugin = LowLevelZeroPlugin(
                stage=1,
                precision=precision,
                initial_scale=self.config.initial_scale,
                min_scale=self.config.min_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                cpu_offload=self.config.use_cpu_offload,
                overlap_communication=self.config.use_async_communication,
                reduce_bucket_size=1024 * 1024,  # 1MB buckets
            )

        elif self.config.parallel_strategy == ParallelismStrategy.ZERO2:
            plugin = LowLevelZeroPlugin(
                stage=2,
                precision=precision,
                initial_scale=self.config.initial_scale,
                partition_grad=(self.config.zero_stage >= 2),
                cpu_offload=self.config.use_cpu_offload,
                overlap_communication=self.config.use_async_communication,
            )

        elif self.config.parallel_strategy == ParallelismStrategy.ZERO3:
            plugin = LowLevelZeroPlugin(
                stage=3,
                precision=precision,
                partition_grad=True,
                partition_param=True,
                cpu_offload=self.config.use_cpu_offload,
                overlap_communication=self.config.use_async_communication,
            )

        elif self.config.parallel_strategy == ParallelismStrategy.HYBRID_PARALLEL:
            # Calculate parallelism sizes
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.config.data_parallel_size == -1:
                # Auto-calculate data parallel size
                dp_size = world_size // (
                    self.config.tensor_parallel_size *
                    self.config.pipeline_parallel_size
                )
            else:
                dp_size = self.config.data_parallel_size

            plugin = HybridParallelPlugin(
                tp_size=self.config.tensor_parallel_size,
                pp_size=self.config.pipeline_parallel_size,
                dp_size=dp_size,
                sequence_parallel_size=self.config.sequence_parallel_size,
                zero_stage=self.config.zero_stage,
                precision=precision,
                enable_jit_fused=self.config.enable_jit_fused,
                enable_fused_normalization=self.config.enable_fused_normalization,
                cpu_offload=self.config.use_cpu_offload,
                overlap_communication=self.config.use_async_communication,
                initial_scale=self.config.initial_scale,
            )

        elif self.config.use_gemini:
            # Gemini memory manager for extremely large models
            plugin = GeminiPlugin(
                precision=precision,
                placement_policy="auto",  # Can be "cpu", "cuda", or "auto"
                pin_memory=True,
                force_outputs_fp32=False,
                strict_ddp_mode=False,
                search_range_m=128,
                hidden_dim=self.training_config.model.hidden_size,
                gpu_margin_mem_ratio=0.0,  # Reserve no margin initially
            )

        else:
            # Default to standard DDP
            plugin = TorchDDPPlugin(
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=False,
                check_reduction=False,
                gradient_as_bucket_view=True,
                static_graph=False,
            )

        return plugin

    def _compatibility_mode_init(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module],
        dataloader: Optional[DataLoader],
    ) -> Tuple[nn.Module, Optimizer, Optional[nn.Module], Optional[DataLoader]]:
        """
        Initialize in compatibility mode without Colossal-AI
        Falls back to standard PyTorch DDP if available
        """
        if dist.is_initialized() and torch.cuda.is_available():
            # Wrap with standard DDP
            model = model.cuda()
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=False,
            )
            logger.info("Model wrapped with standard PyTorch DDP")
        elif torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to CUDA (single GPU mode)")

        return model, optimizer, criterion, dataloader

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: Optional[Optimizer] = None,
        retain_graph: bool = False,
    ):
        """
        Perform backward pass with Colossal-AI optimization

        Args:
            loss: Loss tensor to backpropagate
            optimizer: Optional optimizer for gradient accumulation
            retain_graph: Whether to retain computation graph
        """
        if self.booster and self._initialized:
            self.booster.backward(loss, optimizer, retain_graph=retain_graph)
        else:
            # Fallback to standard backward
            loss.backward(retain_graph=retain_graph)

    def step(self, optimizer: Optimizer, lr_scheduler: Optional[Any] = None):
        """
        Perform optimizer step with Colossal-AI optimization

        Args:
            optimizer: Optimizer to step
            lr_scheduler: Optional learning rate scheduler
        """
        if self.booster and self._initialized:
            # Colossal-AI handles optimization internally
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
        else:
            # Standard optimizer step
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

    def save_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        **kwargs,
    ):
        """
        Save checkpoint with Colossal-AI support

        Args:
            checkpoint_path: Path to save checkpoint
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            **kwargs: Additional items to save
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **kwargs,
        }

        if self.booster and self._initialized:
            self.booster.save_model(model, checkpoint_path, shard=True)
            # Save optimizer separately if using ZeRO
            if self.config.zero_stage > 0:
                opt_path = checkpoint_path.replace(".pt", "_optimizer.pt")
                self.booster.save_optimizer(optimizer, opt_path)
        else:
            torch.save(checkpoint, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with Colossal-AI support

        Args:
            checkpoint_path: Path to load checkpoint from
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state dict matching

        Returns:
            Dictionary containing checkpoint data
        """
        if self.booster and self._initialized:
            self.booster.load_model(model, checkpoint_path, strict=strict)

            # Load optimizer if using ZeRO
            if optimizer and self.config.zero_stage > 0:
                opt_path = checkpoint_path.replace(".pt", "_optimizer.pt")
                if os.path.exists(opt_path):
                    self.booster.load_optimizer(optimizer, opt_path)

            # Load remaining checkpoint data
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics

        Returns:
            Dictionary with memory statistics in GB
        """
        stats = {}

        if torch.cuda.is_available():
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["gpu_free_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                - stats["gpu_reserved_gb"]
            )

        # Get Colossal-AI specific memory stats if available
        if self.booster and hasattr(self.booster.plugin, "get_memory_usage"):
            colossal_stats = self.booster.plugin.get_memory_usage()
            stats.update(colossal_stats)

        return stats

    def optimize_memory(self):
        """
        Trigger memory optimization strategies
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.booster and hasattr(self.booster.plugin, "optimize_memory"):
            self.booster.plugin.optimize_memory()
            logger.info("Triggered Colossal-AI memory optimization")

    def enable_activation_checkpointing(self, model: nn.Module, num_layers: Optional[int] = None):
        """
        Enable activation checkpointing for memory efficiency

        Args:
            model: Model to apply checkpointing to
            num_layers: Optional number of layers to checkpoint (None = auto)
        """
        if not self.config.use_activation_checkpointing:
            return

        try:
            from torch.utils.checkpoint import checkpoint

            # Apply checkpointing to transformer layers
            if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
                layers = model.transformer.layers
                if num_layers is None:
                    num_layers = len(layers) // 2  # Checkpoint half by default

                for i, layer in enumerate(layers[:num_layers]):
                    layer.forward = torch.utils.checkpoint.checkpoint(layer.forward)

                logger.info(f"Enabled activation checkpointing for {num_layers} layers")

        except Exception as e:
            logger.warning(f"Failed to enable activation checkpointing: {e}")

    def get_effective_batch_size(self) -> int:
        """
        Calculate effective batch size accounting for parallelism

        Returns:
            Effective batch size across all devices
        """
        base_batch_size = self.training_config.training.batch_size
        grad_accum = self.training_config.training.gradient_accumulation_steps

        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Account for data parallelism
            if self.config.parallel_strategy == ParallelismStrategy.HYBRID_PARALLEL:
                dp_size = world_size // (
                    self.config.tensor_parallel_size *
                    self.config.pipeline_parallel_size
                )
            else:
                dp_size = world_size

            effective_batch = base_batch_size * grad_accum * dp_size
        else:
            effective_batch = base_batch_size * grad_accum

        return effective_batch

    @property
    def is_main_process(self) -> bool:
        """Check if current process is the main process"""
        if self.coordinator:
            return self.coordinator.is_master()
        elif dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def barrier(self):
        """Synchronization barrier across all processes"""
        if dist.is_initialized():
            dist.barrier()

    def cleanup(self):
        """Cleanup Colossal-AI resources"""
        if self.booster:
            # Cleanup booster resources
            del self.booster
            self.booster = None

        self._initialized = False
        logger.info("Cleaned up Colossal-AI resources")


def create_colossalai_integration(
    training_config: EnhancedTrainingConfig,
    distributed_manager: Optional[DistributedManager] = None,
    **kwargs,
) -> ColossalAIIntegration:
    """
    Factory function to create Colossal-AI integration

    Args:
        training_config: Training configuration
        distributed_manager: Optional existing distributed manager
        **kwargs: Override configuration options

    Returns:
        ColossalAIIntegration instance
    """
    # Create default configuration
    config = ColossalAIConfig()

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Auto-detect optimal settings based on hardware
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Adjust settings based on GPU memory
        if gpu_memory < 16:  # Small GPU (< 16GB)
            config.use_cpu_offload = True
            config.zero_stage = 2
            config.use_activation_checkpointing = True
        elif gpu_memory < 32:  # Medium GPU (16-32GB)
            config.zero_stage = 2
            config.use_activation_checkpointing = True
        else:  # Large GPU (> 32GB)
            config.zero_stage = 1
            config.use_activation_checkpointing = False

        # Enable Flash Attention for Ampere+ GPUs
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 8:  # Ampere or newer
            config.enable_flash_attention = True

    return ColossalAIIntegration(config, training_config, distributed_manager)