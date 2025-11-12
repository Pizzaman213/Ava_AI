"""
Memory Optimization Profiles for Ava Training

This module defines predefined memory optimization profiles that combine
various memory-saving techniques for different resource constraints.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemoryProfile(Enum):
    """Predefined memory optimization profiles."""

    FULL = "full"                 # No optimizations, maximum quality
    MODERATE = "moderate"          # 30-40% memory reduction
    AGGRESSIVE = "aggressive"      # 60-70% memory reduction
    EXTREME = "extreme"            # 80-90% memory reduction, may impact quality


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""

    # Streaming and batching
    use_streaming_tokenization: bool = False
    streaming_buffer_size: int = 10000

    # Expert optimizations
    use_lora_experts: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Expert offloading
    use_expert_offloading: bool = False
    max_active_experts_gpu: int = 4
    offload_pin_memory: bool = True
    offload_async_transfers: bool = True

    # Quantization
    use_expert_quantization: bool = False
    expert_quantization_bits: int = 8
    use_int8_training: bool = False

    # Gradient optimizations
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1

    # Mixed precision
    mixed_precision: str = "fp32"  # fp32, fp16, bf16, fp8

    # Model sharding
    enable_fsdp: bool = False
    fsdp_sharding_strategy: str = "no_shard"  # full_shard, shard_grad_op, no_shard

    # Optimizer state
    use_8bit_optimizer: bool = False
    offload_optimizer_states: bool = False

    # Activation checkpointing
    activation_checkpointing_layers: Optional[list] = None

    # Memory monitoring
    enable_memory_profiling: bool = False
    memory_limit_gb: Optional[float] = None


def get_memory_profile(profile: MemoryProfile) -> MemoryOptimizationConfig:
    """
    Get memory optimization configuration for a specific profile.

    Args:
        profile: Memory optimization profile

    Returns:
        MemoryOptimizationConfig with appropriate settings
    """

    if profile == MemoryProfile.FULL:
        # No optimizations - maximum quality and speed
        return MemoryOptimizationConfig(
            use_streaming_tokenization=False,
            streaming_buffer_size=10000,
            use_lora_experts=False,
            use_expert_offloading=False,
            use_expert_quantization=False,
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            mixed_precision="fp32",
            enable_fsdp=False,
            use_8bit_optimizer=False,
            offload_optimizer_states=False,
            enable_memory_profiling=False
        )

    elif profile == MemoryProfile.MODERATE:
        # 30-40% memory reduction with minimal quality impact
        return MemoryOptimizationConfig(
            use_streaming_tokenization=True,
            streaming_buffer_size=5000,
            use_lora_experts=True,
            lora_rank=16,
            lora_alpha=32,
            use_expert_offloading=False,
            use_expert_quantization=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
            mixed_precision="bf16",  # or fp16 depending on hardware
            enable_fsdp=False,
            use_8bit_optimizer=False,
            offload_optimizer_states=False,
            enable_memory_profiling=True,
            memory_limit_gb=24.0
        )

    elif profile == MemoryProfile.AGGRESSIVE:
        # 60-70% memory reduction with some quality trade-offs
        return MemoryOptimizationConfig(
            use_streaming_tokenization=True,
            streaming_buffer_size=2000,
            use_lora_experts=True,
            lora_rank=8,
            lora_alpha=16,
            use_expert_offloading=True,
            max_active_experts_gpu=4,
            offload_pin_memory=True,
            offload_async_transfers=True,
            use_expert_quantization=True,
            expert_quantization_bits=8,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            mixed_precision="fp16",
            enable_fsdp=True,
            fsdp_sharding_strategy="shard_grad_op",
            use_8bit_optimizer=True,
            offload_optimizer_states=False,
            enable_memory_profiling=True,
            memory_limit_gb=16.0
        )

    elif profile == MemoryProfile.EXTREME:
        # 80-90% memory reduction - may significantly impact quality
        return MemoryOptimizationConfig(
            use_streaming_tokenization=True,
            streaming_buffer_size=1000,
            use_lora_experts=True,
            lora_rank=4,
            lora_alpha=8,
            use_expert_offloading=True,
            max_active_experts_gpu=2,
            offload_pin_memory=True,
            offload_async_transfers=True,
            use_expert_quantization=True,
            expert_quantization_bits=4,
            use_int8_training=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=8,
            mixed_precision="fp16",
            enable_fsdp=True,
            fsdp_sharding_strategy="full_shard",
            use_8bit_optimizer=True,
            offload_optimizer_states=True,
            activation_checkpointing_layers=list(range(0, 32, 2)),  # Every other layer
            enable_memory_profiling=True,
            memory_limit_gb=8.0
        )

    else:
        raise ValueError(f"Unknown memory profile: {profile}")


def apply_memory_profile(config: Dict[str, Any], profile: MemoryProfile) -> Dict[str, Any]:
    """
    Apply a memory optimization profile to a training configuration.

    Args:
        config: Training configuration dictionary
        profile: Memory optimization profile to apply

    Returns:
        Updated configuration with memory optimizations applied
    """

    logger.info(f"Applying memory profile: {profile.value}")

    mem_config = get_memory_profile(profile)

    # Update data configuration
    if "data" not in config:
        config["data"] = {}
    config["data"]["use_streaming_tokenization"] = mem_config.use_streaming_tokenization
    config["data"]["streaming_buffer_size"] = mem_config.streaming_buffer_size

    # Update model configuration
    if "model" not in config:
        config["model"] = {}
    config["model"]["use_lora_experts"] = mem_config.use_lora_experts
    config["model"]["lora_rank"] = mem_config.lora_rank
    config["model"]["lora_alpha"] = mem_config.lora_alpha
    config["model"]["use_expert_offloading"] = mem_config.use_expert_offloading
    config["model"]["max_active_experts_gpu"] = mem_config.max_active_experts_gpu
    config["model"]["offload_pin_memory"] = mem_config.offload_pin_memory
    config["model"]["offload_async_transfers"] = mem_config.offload_async_transfers
    config["model"]["use_expert_quantization"] = mem_config.use_expert_quantization
    config["model"]["expert_quantization_bits"] = mem_config.expert_quantization_bits

    # Update training configuration
    if "training" not in config:
        config["training"] = {}
    config["training"]["gradient_checkpointing"] = mem_config.gradient_checkpointing
    config["training"]["gradient_accumulation_steps"] = mem_config.gradient_accumulation_steps
    config["training"]["mixed_precision"] = mem_config.mixed_precision

    # Update distributed configuration
    if "distributed" not in config:
        config["distributed"] = {}
    config["distributed"]["enable_fsdp"] = mem_config.enable_fsdp
    config["distributed"]["fsdp_sharding_strategy"] = mem_config.fsdp_sharding_strategy

    # Update optimizer configuration
    if "optimizer" not in config:
        config["optimizer"] = {}
    config["optimizer"]["use_8bit"] = mem_config.use_8bit_optimizer
    config["optimizer"]["offload_states"] = mem_config.offload_optimizer_states

    # Log expected memory savings
    expected_savings = {
        MemoryProfile.FULL: "0%",
        MemoryProfile.MODERATE: "30-40%",
        MemoryProfile.AGGRESSIVE: "60-70%",
        MemoryProfile.EXTREME: "80-90%"
    }

    logger.info(f"Expected memory reduction: {expected_savings[profile]}")

    # Warn about quality trade-offs
    if profile == MemoryProfile.AGGRESSIVE:
        logger.warning("Aggressive profile may impact model quality due to quantization and reduced precision")
    elif profile == MemoryProfile.EXTREME:
        logger.warning("Extreme profile will significantly impact quality - use only for testing or when resources are severely limited")

    return config


def validate_memory_compatibility(config: MemoryOptimizationConfig) -> bool:
    """
    Validate that memory optimization settings are compatible.

    Args:
        config: Memory optimization configuration

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration contains incompatible settings
    """

    # Check LoRA and offloading compatibility
    if config.use_lora_experts and config.use_expert_offloading:
        if config.lora_rank > 16:
            raise ValueError("High LoRA rank (>16) is not recommended with expert offloading due to memory overhead")

    # Check quantization compatibility
    if config.use_expert_quantization and config.expert_quantization_bits < 8:
        if not config.use_lora_experts:
            raise ValueError("4-bit quantization requires LoRA experts to maintain quality")

    # Check mixed precision and quantization
    if config.mixed_precision == "fp8" and config.use_expert_quantization:
        raise ValueError("FP8 mixed precision conflicts with integer quantization")

    # Check gradient accumulation limits
    if config.gradient_accumulation_steps > 16:
        logger.warning("Very high gradient accumulation (>16) may lead to stale gradients")

    # Check FSDP and offloading
    if config.enable_fsdp and config.fsdp_sharding_strategy == "full_shard":
        if config.use_expert_offloading:
            raise ValueError("Full FSDP sharding conflicts with expert offloading")

    return True


def get_recommended_profile(available_memory_gb: float, model_size_gb: float) -> MemoryProfile:
    """
    Get recommended memory profile based on available resources.

    Args:
        available_memory_gb: Available GPU memory in GB
        model_size_gb: Estimated model size in GB

    Returns:
        Recommended MemoryProfile
    """

    memory_ratio = available_memory_gb / model_size_gb

    if memory_ratio >= 4.0:
        # Plenty of memory - no optimization needed
        return MemoryProfile.FULL
    elif memory_ratio >= 2.5:
        # Comfortable memory - light optimizations
        return MemoryProfile.MODERATE
    elif memory_ratio >= 1.5:
        # Tight memory - aggressive optimizations
        return MemoryProfile.AGGRESSIVE
    else:
        # Very limited memory - extreme optimizations
        logger.warning(f"Very low memory ratio ({memory_ratio:.2f}x) - quality will be impacted")
        return MemoryProfile.EXTREME


# Export key components
__all__ = [
    'MemoryProfile',
    'MemoryOptimizationConfig',
    'get_memory_profile',
    'apply_memory_profile',
    'validate_memory_compatibility',
    'get_recommended_profile'
]