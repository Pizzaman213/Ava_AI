"""
Distributed Optimizers Module

This module provides Colossal-AI's distributed optimizers including
GaLore, DistributedLamb, and other memory-efficient optimizers.
"""

import logging
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

try:
    import colossalai
    from colossalai.nn.optimizer import (
        DistributedGaloreAdamW,
        DistributedLamb,
        HybridAdam,
    )
    from colossalai.nn.optimizer.galore import GaLoreAdamW, GaLoreAdafactor

    DISTRIBUTED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    DISTRIBUTED_OPTIMIZERS_AVAILABLE = False
    DistributedGaloreAdamW = None
    DistributedLamb = None
    HybridAdam = None
    GaLoreAdamW = None
    GaLoreAdafactor = None

logger = logging.getLogger(__name__)


class OptimizerConfig:
    """Configuration for distributed optimizers"""

    def __init__(
        self,
        # Optimizer selection
        optimizer_type: str = "hybrid_adam",  # hybrid_adam, distributed_lamb, galore_adamw, galore_adafactor

        # Common optimizer parameters
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,

        # GaLore specific parameters
        use_galore: bool = False,
        galore_rank: int = 256,
        galore_update_proj_gap: int = 200,
        galore_scale: float = 0.25,
        galore_proj_type: str = "std",  # std, reverse_std, right, left, full

        # Distributed specific
        overlap_communication: bool = True,
        initial_scale: float = 2**16,
        min_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**24,

        # LAMB specific
        bias_correction: bool = True,
        max_coeff: float = 10.0,

        # Memory optimization
        use_nvme_offload: bool = False,
        nvme_offload_fraction: float = 1.0,
        nvme_offload_dir: str = "./nvme_offload",
    ):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # GaLore
        self.use_galore = use_galore
        self.galore_rank = galore_rank
        self.galore_update_proj_gap = galore_update_proj_gap
        self.galore_scale = galore_scale
        self.galore_proj_type = galore_proj_type

        # Distributed
        self.overlap_communication = overlap_communication
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis
        self.max_scale = max_scale

        # LAMB
        self.bias_correction = bias_correction
        self.max_coeff = max_coeff

        # Memory
        self.use_nvme_offload = use_nvme_offload
        self.nvme_offload_fraction = nvme_offload_fraction
        self.nvme_offload_dir = nvme_offload_dir


def create_distributed_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    param_groups: Optional[Iterable] = None,
) -> Optimizer:
    """
    Create a distributed optimizer based on configuration

    Args:
        model: Model to optimize
        config: Optimizer configuration
        param_groups: Optional custom parameter groups

    Returns:
        Configured distributed optimizer
    """
    if not DISTRIBUTED_OPTIMIZERS_AVAILABLE:
        logger.warning(
            "Colossal-AI distributed optimizers not available. "
            "Falling back to standard AdamW optimizer."
        )
        return torch.optim.AdamW(
            model.parameters() if param_groups is None else param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    # Get parameters
    if param_groups is None:
        param_groups = model.parameters()

    # Create optimizer based on type
    optimizer_type = config.optimizer_type.lower()

    try:
        if optimizer_type == "hybrid_adam":
            optimizer = HybridAdam(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                adamw_mode=True,
            )
            logger.info("Created HybridAdam optimizer")

        elif optimizer_type == "distributed_lamb":
            optimizer = DistributedLamb(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                bias_correction=config.bias_correction,
                max_coeff=config.max_coeff,
            )
            logger.info("Created DistributedLamb optimizer")

        elif optimizer_type == "galore_adamw":
            if not config.use_galore:
                logger.warning("GaLore optimizer requested but use_galore=False, enabling it")
                config.use_galore = True

            optimizer = GaLoreAdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                rank=config.galore_rank,
                update_proj_gap=config.galore_update_proj_gap,
                scale=config.galore_scale,
                proj_type=config.galore_proj_type,
            )
            logger.info(f"Created GaLoreAdamW optimizer with rank={config.galore_rank}")

        elif optimizer_type == "galore_adafactor":
            if not config.use_galore:
                logger.warning("GaLore optimizer requested but use_galore=False, enabling it")
                config.use_galore = True

            optimizer = GaLoreAdafactor(
                param_groups,
                lr=config.lr,
                eps=config.eps,
                weight_decay=config.weight_decay,
                rank=config.galore_rank,
                update_proj_gap=config.galore_update_proj_gap,
                scale=config.galore_scale,
                proj_type=config.galore_proj_type,
            )
            logger.info(f"Created GaLoreAdafactor optimizer with rank={config.galore_rank}")

        elif optimizer_type == "distributed_galore_adamw":
            optimizer = DistributedGaloreAdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                rank=config.galore_rank,
                update_proj_gap=config.galore_update_proj_gap,
                scale=config.galore_scale,
                proj_type=config.galore_proj_type,
            )
            logger.info(f"Created DistributedGaloreAdamW optimizer with rank={config.galore_rank}")

        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, falling back to HybridAdam")
            optimizer = HybridAdam(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                adamw_mode=True,
            )

        return optimizer

    except Exception as e:
        logger.error(f"Failed to create distributed optimizer: {e}")
        logger.info("Falling back to standard AdamW")
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )


def get_optimizer_recommendations(
    model_size: int,
    gpu_memory_gb: float,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    Get optimizer recommendations based on model size and available resources

    Args:
        model_size: Number of model parameters
        gpu_memory_gb: Available GPU memory in GB
        world_size: Number of GPUs

    Returns:
        Dictionary with optimizer recommendations
    """
    recommendations = {
        "optimizer_type": "hybrid_adam",
        "use_galore": False,
        "use_nvme_offload": False,
        "galore_rank": 256,
    }

    # Calculate approximate memory requirements
    model_memory = model_size * 4 / (1024**3)  # 4 bytes per parameter (float32)
    optimizer_memory = model_size * 8 / (1024**3)  # 8 bytes for Adam states

    total_memory_per_gpu = (model_memory + optimizer_memory) / world_size

    if total_memory_per_gpu > gpu_memory_gb * 0.8:
        # Not enough memory, recommend memory-efficient options

        if model_size > 1e9:  # > 1B parameters
            recommendations["optimizer_type"] = "galore_adamw"
            recommendations["use_galore"] = True
            recommendations["galore_rank"] = 128 if model_size > 7e9 else 256
            logger.info(
                f"Recommending GaLore optimizer for {model_size/1e9:.2f}B parameter model"
            )

        if total_memory_per_gpu > gpu_memory_gb * 1.2:
            recommendations["use_nvme_offload"] = True
            logger.info("Recommending NVMe offload due to memory constraints")

    elif model_size > 3e9:  # > 3B parameters
        # Large model with sufficient memory
        recommendations["optimizer_type"] = "distributed_lamb"
        logger.info("Recommending DistributedLamb for large model training")

    else:
        # Standard model size
        recommendations["optimizer_type"] = "hybrid_adam"
        logger.info("Recommending HybridAdam for standard model training")

    return recommendations


class OptimizerFactory:
    """Factory class for creating optimizers with common patterns"""

    @staticmethod
    def create_for_large_model(
        model: nn.Module,
        lr: float = 1e-4,
        use_galore: bool = True,
        galore_rank: int = 256,
    ) -> Optimizer:
        """Create optimizer optimized for large models"""
        config = OptimizerConfig(
            optimizer_type="galore_adamw" if use_galore else "distributed_lamb",
            lr=lr,
            use_galore=use_galore,
            galore_rank=galore_rank,
            weight_decay=0.01,
        )
        return create_distributed_optimizer(model, config)

    @staticmethod
    def create_for_memory_constrained(
        model: nn.Module,
        lr: float = 1e-4,
        galore_rank: int = 128,
    ) -> Optimizer:
        """Create memory-efficient optimizer"""
        config = OptimizerConfig(
            optimizer_type="galore_adafactor",
            lr=lr,
            use_galore=True,
            galore_rank=galore_rank,
            use_nvme_offload=True,
            weight_decay=0.01,
        )
        return create_distributed_optimizer(model, config)

    @staticmethod
    def create_for_speed(
        model: nn.Module,
        lr: float = 1e-4,
    ) -> Optimizer:
        """Create optimizer optimized for training speed"""
        config = OptimizerConfig(
            optimizer_type="hybrid_adam",
            lr=lr,
            overlap_communication=True,
            weight_decay=0.01,
        )
        return create_distributed_optimizer(model, config)

    @staticmethod
    def create_auto(
        model: nn.Module,
        lr: float = 1e-4,
        gpu_memory_gb: Optional[float] = None,
        world_size: int = 1,
    ) -> Optimizer:
        """
        Automatically select and create optimal optimizer

        Args:
            model: Model to optimize
            lr: Learning rate
            gpu_memory_gb: Available GPU memory (auto-detected if None)
            world_size: Number of GPUs

        Returns:
            Optimized optimizer instance
        """
        # Count parameters
        model_size = sum(p.numel() for p in model.parameters())

        # Auto-detect GPU memory if not provided
        if gpu_memory_gb is None and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Get recommendations
        recommendations = get_optimizer_recommendations(model_size, gpu_memory_gb, world_size)

        # Create config from recommendations
        config = OptimizerConfig(
            optimizer_type=recommendations["optimizer_type"],
            lr=lr,
            use_galore=recommendations["use_galore"],
            galore_rank=recommendations["galore_rank"],
            use_nvme_offload=recommendations["use_nvme_offload"],
            weight_decay=0.01,
        )

        logger.info(f"Auto-selected optimizer: {config.optimizer_type}")
        return create_distributed_optimizer(model, config)
