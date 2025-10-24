"""
Shardformer Integration Module

This module provides integration with Colossal-AI's Shardformer feature,
which automatically prepares HuggingFace models for tensor and pipeline parallelism.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

try:
    import colossalai
    from colossalai.shardformer import ShardConfig, ShardFormer
    from colossalai.shardformer.policies import PolicyLocation, get_autopolicy

    SHARDFORMER_AVAILABLE = True
except ImportError:
    SHARDFORMER_AVAILABLE = False
    ShardConfig = None
    ShardFormer = None

logger = logging.getLogger(__name__)


class ShardformerConfig:
    """Configuration for Shardformer integration"""

    def __init__(
        self,
        # Parallelism settings
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_stage_manager: Optional[Any] = None,

        # Memory optimization
        enable_tensor_parallelism: bool = True,
        enable_all_optimization: bool = False,
        enable_flash_attention: bool = True,
        enable_jit_fused: bool = True,
        enable_sequence_parallelism: bool = False,
        enable_sequence_overlap: bool = False,

        # Precision settings
        fp16: bool = False,
        bf16: bool = True,

        # Communication optimization
        use_lazy_init: bool = True,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_stage_manager = pipeline_stage_manager
        self.enable_tensor_parallelism = enable_tensor_parallelism
        self.enable_all_optimization = enable_all_optimization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.enable_sequence_overlap = enable_sequence_overlap
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_lazy_init = use_lazy_init
        self.extra_kwargs = extra_kwargs or {}


class ShardformerIntegration:
    """
    Integration class for Colossal-AI's Shardformer

    Shardformer automatically converts HuggingFace models to support
    tensor parallelism, pipeline parallelism, and other optimizations.
    """

    def __init__(self, config: ShardformerConfig):
        """
        Initialize Shardformer integration

        Args:
            config: Shardformer configuration
        """
        self.config = config
        self.shard_former: Optional[ShardFormer] = None
        self._initialized = False

        if not SHARDFORMER_AVAILABLE:
            logger.warning(
                "Shardformer not available. Please upgrade to Colossal-AI >= 0.3.0. "
                "Falling back to standard model initialization."
            )
            return

        self._setup_shardformer()

    def _setup_shardformer(self):
        """Initialize Shardformer with configuration"""
        if not SHARDFORMER_AVAILABLE:
            return

        try:
            # Create shard config
            shard_config = ShardConfig(
                tensor_parallel_process_group=None,  # Auto-initialize
                pipeline_stage_manager=self.config.pipeline_stage_manager,
                enable_tensor_parallelism=self.config.enable_tensor_parallelism,
                enable_all_optimization=self.config.enable_all_optimization,
                enable_flash_attention=self.config.enable_flash_attention,
                enable_jit_fused=self.config.enable_jit_fused,
                enable_sequence_parallelism=self.config.enable_sequence_parallelism,
                enable_sequence_overlap=self.config.enable_sequence_overlap,
                **self.config.extra_kwargs,
            )

            # Create ShardFormer instance
            self.shard_former = ShardFormer(shard_config)
            self._initialized = True

            logger.info(f"Shardformer initialized with TP={self.config.tensor_parallel_size}")

        except Exception as e:
            logger.error(f"Failed to initialize Shardformer: {e}")
            self._initialized = False

    def shard_model(
        self,
        model: Union[nn.Module, PreTrainedModel],
        policy: Optional[Any] = None,
    ) -> nn.Module:
        """
        Shard a model for distributed training

        Args:
            model: Model to shard (HuggingFace or custom)
            policy: Optional custom sharding policy

        Returns:
            Sharded model ready for distributed training
        """
        if not self._initialized or not SHARDFORMER_AVAILABLE:
            logger.warning("Shardformer not available, returning original model")
            return model

        try:
            # Auto-detect policy if not provided
            if policy is None:
                policy = self._get_auto_policy(model)

            # Apply sharding
            sharded_model, shared_params = self.shard_former.optimize(model, policy)

            logger.info(f"Successfully sharded model with policy: {policy.__class__.__name__}")
            logger.info(f"Shared parameters: {len(shared_params)}")

            return sharded_model

        except Exception as e:
            logger.error(f"Failed to shard model: {e}")
            logger.warning("Returning original model")
            return model

    def _get_auto_policy(self, model: nn.Module) -> Any:
        """
        Automatically detect the best sharding policy for a model

        Args:
            model: Model to analyze

        Returns:
            Appropriate sharding policy
        """
        if not SHARDFORMER_AVAILABLE:
            return None

        try:
            # Try to get auto policy based on model type
            model_type = getattr(model.config, "model_type", None)

            if model_type:
                policy = get_autopolicy(model, policy_location=PolicyLocation.HuggingFace)
                logger.info(f"Using auto-detected policy for model type: {model_type}")
                return policy

            # Fallback to generic policy
            logger.warning("Could not detect model type, using generic policy")
            return None

        except Exception as e:
            logger.warning(f"Failed to get auto policy: {e}")
            return None

    def prepare_optimizer(
        self,
        optimizer_cls: type,
        model: nn.Module,
        **optimizer_kwargs
    ) -> torch.optim.Optimizer:
        """
        Prepare optimizer for sharded model

        Args:
            optimizer_cls: Optimizer class (e.g., torch.optim.AdamW)
            model: Sharded model
            **optimizer_kwargs: Optimizer arguments

        Returns:
            Optimizer configured for sharded model
        """
        # Get trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]

        # Create optimizer
        optimizer = optimizer_cls(params, **optimizer_kwargs)

        logger.info(f"Created optimizer with {len(params)} parameter groups")

        return optimizer

    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get information about sharded model

        Args:
            model: Sharded model

        Returns:
            Dictionary with model information
        """
        info = {
            "sharded": self._initialized,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
        }

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info["total_parameters"] = total_params
        info["trainable_parameters"] = trainable_params
        info["parameter_size_gb"] = total_params * 4 / (1024**3)  # Assuming float32

        return info


def create_shardformer_integration(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_flash_attention: bool = True,
    **kwargs
) -> ShardformerIntegration:
    """
    Factory function to create Shardformer integration

    Args:
        tensor_parallel_size: Size of tensor parallelism
        pipeline_parallel_size: Size of pipeline parallelism
        enable_flash_attention: Whether to enable Flash Attention
        **kwargs: Additional configuration options

    Returns:
        ShardformerIntegration instance
    """
    config = ShardformerConfig(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enable_flash_attention=enable_flash_attention,
        **kwargs,
    )

    return ShardformerIntegration(config)


# Helper function for easy integration
def auto_shard_huggingface_model(
    model: PreTrainedModel,
    tensor_parallel_size: int = 1,
    enable_optimizations: bool = True,
) -> PreTrainedModel:
    """
    Automatically shard a HuggingFace model with sensible defaults

    Args:
        model: HuggingFace model to shard
        tensor_parallel_size: Tensor parallelism size
        enable_optimizations: Enable all optimizations

    Returns:
        Sharded model
    """
    integration = create_shardformer_integration(
        tensor_parallel_size=tensor_parallel_size,
        enable_flash_attention=enable_optimizations,
        enable_jit_fused=enable_optimizations,
        enable_all_optimization=enable_optimizations,
    )

    return integration.shard_model(model)
