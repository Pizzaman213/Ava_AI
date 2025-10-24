"""
Enhanced Colossal-AI Features Module

This module provides advanced Colossal-AI features including:
- Advanced gradient compression strategies
- Auto-parallelism configuration helpers
- Performance monitoring and profiling
- Communication optimization
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import colossalai
    from colossalai.accelerator import get_accelerator
    from colossalai.tensor import ProcessGroup
    from colossalai.utils import get_current_device

    COLOSSALAI_AVAILABLE = True
except ImportError:
    COLOSSALAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression"""

    enabled: bool = True
    compression_ratio: float = 0.1  # Compress to 10% of original size
    compression_type: str = "topk"  # topk, randomk, or threshold
    threshold: float = 0.01
    min_compression_bytes: int = 1024  # Don't compress gradients smaller than 1KB
    warmup_steps: int = 100  # No compression for first N steps


class GradientCompressor:
    """Advanced gradient compression for communication efficiency"""

    def __init__(self, config: GradientCompressionConfig):
        self.config = config
        self.step_count = 0
        self._compression_stats = {
            "compressed_bytes": 0,
            "original_bytes": 0,
            "compression_time": 0.0,
        }

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress gradient tensor

        Args:
            tensor: Gradient tensor to compress

        Returns:
            Tuple of (compressed_tensor, metadata)
        """
        if not self.config.enabled or self.step_count < self.config.warmup_steps:
            return tensor, {"compressed": False}

        start_time = time.time()
        original_size = tensor.numel() * tensor.element_size()

        # Skip small tensors
        if original_size < self.config.min_compression_bytes:
            return tensor, {"compressed": False}

        metadata = {"compressed": True, "original_shape": tensor.shape}

        if self.config.compression_type == "topk":
            compressed, metadata = self._topk_compression(tensor, metadata)
        elif self.config.compression_type == "randomk":
            compressed, metadata = self._randomk_compression(tensor, metadata)
        elif self.config.compression_type == "threshold":
            compressed, metadata = self._threshold_compression(tensor, metadata)
        else:
            logger.warning(f"Unknown compression type: {self.config.compression_type}")
            return tensor, {"compressed": False}

        # Update statistics
        compression_time = time.time() - start_time
        self._compression_stats["compressed_bytes"] += compressed.numel() * compressed.element_size()
        self._compression_stats["original_bytes"] += original_size
        self._compression_stats["compression_time"] += compression_time

        return compressed, metadata

    def _topk_compression(
        self, tensor: torch.Tensor, metadata: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Top-K compression - keep largest K% values"""
        k = int(tensor.numel() * self.config.compression_ratio)
        k = max(1, k)  # Keep at least one value

        flat_tensor = tensor.flatten()
        values, indices = torch.topk(flat_tensor.abs(), k)
        signs = flat_tensor[indices].sign()

        metadata["k"] = k
        metadata["compression_type"] = "topk"

        # Return sparse representation
        compressed = torch.stack([indices.float(), values * signs])
        return compressed, metadata

    def _randomk_compression(
        self, tensor: torch.Tensor, metadata: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Random-K compression - keep random K% values"""
        k = int(tensor.numel() * self.config.compression_ratio)
        k = max(1, k)

        flat_tensor = tensor.flatten()
        indices = torch.randperm(flat_tensor.numel(), device=tensor.device)[:k]
        values = flat_tensor[indices]

        metadata["k"] = k
        metadata["compression_type"] = "randomk"

        compressed = torch.stack([indices.float(), values])
        return compressed, metadata

    def _threshold_compression(
        self, tensor: torch.Tensor, metadata: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Threshold compression - keep values above threshold"""
        threshold = self.config.threshold
        mask = tensor.abs() > threshold

        indices = mask.nonzero(as_tuple=False).squeeze()
        values = tensor.flatten()[indices]

        metadata["threshold"] = threshold
        metadata["compression_type"] = "threshold"

        compressed = torch.stack([indices.float(), values])
        return compressed, metadata

    def decompress(
        self, compressed: torch.Tensor, metadata: Dict, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Decompress gradient tensor

        Args:
            compressed: Compressed tensor
            metadata: Compression metadata

        Returns:
            Decompressed tensor
        """
        if not metadata.get("compressed", False):
            return compressed

        if device is None:
            device = compressed.device

        original_shape = metadata["original_shape"]
        total_elements = torch.prod(torch.tensor(original_shape)).item()

        # Reconstruct sparse tensor
        indices = compressed[0].long()
        values = compressed[1]

        # Create full tensor
        result = torch.zeros(total_elements, device=device, dtype=values.dtype)
        result[indices] = values

        return result.reshape(original_shape)

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        if self._compression_stats["original_bytes"] > 0:
            ratio = (
                self._compression_stats["compressed_bytes"]
                / self._compression_stats["original_bytes"]
            )
        else:
            ratio = 1.0

        return {
            **self._compression_stats,
            "actual_compression_ratio": ratio,
            "compression_savings_gb": (
                self._compression_stats["original_bytes"]
                - self._compression_stats["compressed_bytes"]
            )
            / (1024**3),
        }

    def step(self):
        """Increment step counter"""
        self.step_count += 1


class AutoParallelismHelper:
    """Helper for automatic parallelism configuration"""

    @staticmethod
    def suggest_parallelism_config(
        model_size: int,
        num_gpus: int,
        gpu_memory_gb: float,
        sequence_length: int = 2048,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Suggest optimal parallelism configuration

        Args:
            model_size: Number of model parameters
            num_gpus: Number of available GPUs
            gpu_memory_gb: Memory per GPU in GB
            sequence_length: Maximum sequence length
            batch_size: Batch size per GPU

        Returns:
            Dictionary with parallelism recommendations
        """
        # Calculate memory requirements
        model_memory = model_size * 4 / (1024**3)  # Float32
        activation_memory = (
            batch_size * sequence_length * 1024 * 4 / (1024**3)
        )  # Rough estimate

        logger.info(f"Model size: {model_size/1e9:.2f}B parameters")
        logger.info(f"Estimated model memory: {model_memory:.2f}GB")
        logger.info(f"Estimated activation memory: {activation_memory:.2f}GB per GPU")

        config = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "zero_stage": 0,
            "use_activation_checkpointing": False,
            "use_cpu_offload": False,
            "use_flash_attention": True,
        }

        # Determine if model fits in single GPU
        total_memory_per_gpu = model_memory + activation_memory

        if total_memory_per_gpu > gpu_memory_gb * 0.9:
            # Model doesn't fit, need parallelism

            if num_gpus >= 8 and model_size > 7e9:  # Large model, many GPUs
                # Use pipeline + tensor parallelism
                config["tensor_parallel_size"] = 4
                config["pipeline_parallel_size"] = 2
                config["data_parallel_size"] = num_gpus // 8
                config["zero_stage"] = 2
                logger.info("Recommending hybrid parallelism (TP=4, PP=2)")

            elif num_gpus >= 4:  # Medium model, several GPUs
                # Use tensor parallelism
                config["tensor_parallel_size"] = min(4, num_gpus)
                config["data_parallel_size"] = num_gpus // config["tensor_parallel_size"]
                config["zero_stage"] = 2
                logger.info(f"Recommending tensor parallelism (TP={config['tensor_parallel_size']})")

            else:  # Small GPU count
                # Use ZeRO
                config["data_parallel_size"] = num_gpus
                config["zero_stage"] = 3
                config["use_cpu_offload"] = total_memory_per_gpu > gpu_memory_gb * 1.5
                logger.info(f"Recommending ZeRO-{config['zero_stage']}")

            # Always use activation checkpointing for large models
            if model_size > 1e9:
                config["use_activation_checkpointing"] = True

        elif total_memory_per_gpu > gpu_memory_gb * 0.7:
            # Model fits but tight on memory
            config["zero_stage"] = 2
            config["use_activation_checkpointing"] = model_size > 3e9
            logger.info("Recommending ZeRO-2 for memory efficiency")

        else:
            # Plenty of memory, use simple data parallelism
            config["zero_stage"] = 0
            logger.info("Recommending standard data parallelism")

        # Always enable flash attention for Ampere+ GPUs
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            config["use_flash_attention"] = compute_capability[0] >= 8

        return config

    @staticmethod
    def optimize_for_throughput(num_gpus: int) -> Dict[str, Any]:
        """Configuration optimized for maximum throughput"""
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "zero_stage": 1,  # Minimal memory overhead
            "use_activation_checkpointing": False,
            "use_cpu_offload": False,
            "use_flash_attention": True,
            "overlap_communication": True,
            "use_async_communication": True,
        }

    @staticmethod
    def optimize_for_memory(num_gpus: int) -> Dict[str, Any]:
        """Configuration optimized for maximum model size"""
        return {
            "tensor_parallel_size": min(4, num_gpus),
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus // min(4, num_gpus),
            "zero_stage": 3,
            "use_activation_checkpointing": True,
            "use_cpu_offload": True,
            "use_flash_attention": True,
            "overlap_communication": True,
            "use_async_communication": True,
        }


class PerformanceMonitor:
    """Monitor and profile distributed training performance"""

    def __init__(self):
        self.metrics = {
            "step_times": [],
            "forward_times": [],
            "backward_times": [],
            "optimizer_times": [],
            "communication_times": [],
            "memory_usage": [],
        }
        self.step_start_time = None
        self.phase_start_time = None

    def start_step(self):
        """Mark start of training step"""
        self.step_start_time = time.time()

    def start_phase(self):
        """Mark start of a phase (forward, backward, etc.)"""
        self.phase_start_time = time.time()

    def end_phase(self, phase_name: str):
        """Mark end of a phase and record time"""
        if self.phase_start_time is None:
            return

        elapsed = time.time() - self.phase_start_time
        key = f"{phase_name}_times"

        if key in self.metrics:
            self.metrics[key].append(elapsed)

        self.phase_start_time = None

    def end_step(self):
        """Mark end of training step"""
        if self.step_start_time is None:
            return

        elapsed = time.time() - self.step_start_time
        self.metrics["step_times"].append(elapsed)

        # Record memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            self.metrics["memory_usage"].append(memory_gb)

        self.step_start_time = None

    def get_summary(self, last_n_steps: int = 100) -> Dict[str, Any]:
        """Get performance summary for last N steps"""
        summary = {}

        for key, values in self.metrics.items():
            if not values:
                continue

            recent_values = values[-last_n_steps:]

            summary[key] = {
                "mean": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "last": recent_values[-1] if recent_values else 0,
            }

            # Calculate throughput for step times
            if key == "step_times" and recent_values:
                summary["throughput_steps_per_sec"] = 1.0 / summary[key]["mean"]

        return summary

    def log_summary(self, step: int, last_n_steps: int = 100):
        """Log performance summary"""
        summary = self.get_summary(last_n_steps)

        logger.info(f"Performance Summary (Step {step}, last {last_n_steps} steps):")
        logger.info(f"  Step time: {summary.get('step_times', {}).get('mean', 0):.4f}s")
        logger.info(
            f"  Throughput: {summary.get('throughput_steps_per_sec', 0):.2f} steps/sec"
        )

        if "memory_usage" in summary:
            logger.info(f"  Memory: {summary['memory_usage']['mean']:.2f}GB")


def create_optimal_config(
    model: nn.Module,
    num_gpus: Optional[int] = None,
    gpu_memory_gb: Optional[float] = None,
    optimize_for: str = "balanced",  # balanced, throughput, or memory
) -> Dict[str, Any]:
    """
    Create optimal Colossal-AI configuration

    Args:
        model: Model to train
        num_gpus: Number of GPUs (auto-detect if None)
        gpu_memory_gb: GPU memory in GB (auto-detect if None)
        optimize_for: Optimization target

    Returns:
        Optimal configuration dictionary
    """
    # Auto-detect resources
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Count model parameters
    model_size = sum(p.numel() for p in model.parameters())

    helper = AutoParallelismHelper()

    if optimize_for == "throughput":
        config = helper.optimize_for_throughput(num_gpus)
    elif optimize_for == "memory":
        config = helper.optimize_for_memory(num_gpus)
    else:  # balanced
        config = helper.suggest_parallelism_config(model_size, num_gpus, gpu_memory_gb)

    logger.info(f"Generated {optimize_for} configuration for {model_size/1e9:.2f}B parameter model")
    logger.info(f"Configuration: {config}")

    return config
