"""
Memory Optimization Module for A100 GPUs

This module provides advanced memory management and gradient checkpointing
optimized for NVIDIA A100's high memory bandwidth and large memory capacity.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
import functools
import warnings
import logging
from contextlib import contextmanager
import gc

# DeepSpeed imports (optional)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

logger = logging.getLogger(__name__)


class A100MemoryOptimizer:
    """
    Advanced memory optimizer for NVIDIA A100 GPUs.

    Features:
    - Custom memory pool management
    - Selective gradient checkpointing
    - Memory-efficient attention patterns
    - Activation rematerialization
    - Memory profiling and monitoring
    """

    def __init__(
        self,
        enable_memory_pool: bool = True,
        pool_size_gb: Optional[float] = None,
        enable_gradient_checkpointing: bool = True,
        checkpoint_policy: str = "selective",  # "none", "all", "selective", "adaptive"
        memory_threshold_gb: float = 30.0,  # A100 has 40GB or 80GB
        enable_cpu_offload: bool = False,
        profile_memory: bool = False,
        clear_cache_frequency: int = 100,  # Clear cache every N steps
        # DeepSpeed integration
        deepspeed_engine: Optional[Any] = None,
        deepspeed_zero_stage: int = 0,
        deepspeed_cpu_offload: bool = False,
        deepspeed_nvme_offload: bool = False
    ):
        """
        Initialize memory optimizer.

        Args:
            enable_memory_pool: Enable custom memory pool
            pool_size_gb: Memory pool size in GB (None for auto)
            enable_gradient_checkpointing: Enable gradient checkpointing
            checkpoint_policy: Policy for gradient checkpointing
            memory_threshold_gb: Memory usage threshold for adaptive policies
            enable_cpu_offload: Enable CPU offloading for large models
            profile_memory: Enable memory profiling
            clear_cache_frequency: Frequency of cache clearing
        """
        self.device = self._check_gpu_memory()
        self.enable_memory_pool = enable_memory_pool
        self.pool_size_gb = pool_size_gb
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.checkpoint_policy = checkpoint_policy
        self.memory_threshold_gb = memory_threshold_gb
        self.enable_cpu_offload = enable_cpu_offload
        self.profile_memory = profile_memory
        self.clear_cache_frequency = clear_cache_frequency

        # Memory statistics
        self.memory_stats = {
            "peak_memory_gb": 0.0,
            "current_memory_gb": 0.0,
            "reserved_memory_gb": 0.0,
            "checkpointed_layers": [],
            "offloaded_layers": [],
            "cache_clears": 0
        }

        # Step counter for cache clearing
        self.step_count = 0

        # DeepSpeed integration
        self.deepspeed_engine = deepspeed_engine
        self.deepspeed_zero_stage = deepspeed_zero_stage
        self.deepspeed_cpu_offload = deepspeed_cpu_offload
        self.deepspeed_nvme_offload = deepspeed_nvme_offload
        self.is_deepspeed_enabled = deepspeed_engine is not None

        # GPU info
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.is_a100_80gb = "A100" in self.device_name and self.total_memory > 70e9
        else:
            self.device_name = "CPU"
            self.total_memory = 0
            self.is_a100_80gb = False

        # Setup memory pool if enabled
        if enable_memory_pool and torch.cuda.is_available():
            self._setup_memory_pool()

    def _check_gpu_memory(self) -> str:
        """Check GPU memory capacity."""
        if not torch.cuda.is_available():
            return "cpu"

        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / (1024 ** 3)

        logger.info(f"GPU Memory: {memory_gb:.1f} GB")

        # Check if it's an A100
        if "a100" in props.name.lower():
            if memory_gb > 70:
                logger.info("Detected A100 80GB - Enabling large model optimizations")
                self.is_a100_80gb = True
            else:
                logger.info("Detected A100 40GB")
                self.is_a100_80gb = False
        else:
            self.is_a100_80gb = False

        return f"cuda:{torch.cuda.current_device()}"

    def _setup_memory_pool(self):
        """Setup CUDA memory pool for A100."""
        try:
            if self.pool_size_gb:
                # Set explicit memory pool size
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_memory_gb <= 0:
                    raise ValueError("Invalid GPU memory size detected")
                fraction = self.pool_size_gb / total_memory_gb
                fraction = min(fraction, 0.95)  # Leave some memory for safety
            else:
                # Auto-configure based on GPU
                if self.is_a100_80gb:
                    fraction = 0.9  # Use 90% of 80GB
                else:
                    fraction = 0.85  # Use 85% of 40GB

            torch.cuda.set_per_process_memory_fraction(fraction)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to setup memory pool: {e}")
            # Fall back to default settings

        # Configure memory allocator for A100
        if hasattr(torch.cuda, 'memory_pool'):
            # Set optimal parameters for A100's memory bandwidth (1.5-2 TB/s)
            torch.cuda.memory_pool.set_active_pool_size(int(fraction * torch.cuda.get_device_properties(0).total_memory))

        logger.info(f"Memory pool configured: {fraction:.1%} of GPU memory")

    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to a model.

        Args:
            model: PyTorch model

        Returns:
            Memory-optimized model
        """
        if self.enable_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)

        if self.enable_cpu_offload:
            model = self._setup_cpu_offloading(model)

        # Optimize buffer allocation
        self._optimize_buffers(model)

        return model

    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing based on policy."""
        checkpointed_layers = []

        if self.checkpoint_policy == "none":
            return model

        elif self.checkpoint_policy == "all":
            # Checkpoint all transformer blocks
            for name, module in model.named_modules():
                if self._should_checkpoint_layer(name, module, always=True):
                    self._wrap_with_checkpoint(module)
                    checkpointed_layers.append(name)

        elif self.checkpoint_policy == "selective":
            # Checkpoint compute-intensive layers (attention, FFN)
            for name, module in model.named_modules():
                if self._should_checkpoint_layer(name, module, always=False):
                    self._wrap_with_checkpoint(module)
                    checkpointed_layers.append(name)

        elif self.checkpoint_policy == "adaptive":
            # Checkpoint based on memory usage
            current_memory = self._get_current_memory_gb()
            if current_memory > self.memory_threshold_gb * 0.7:
                # High memory usage - checkpoint more aggressively
                for name, module in model.named_modules():
                    if self._should_checkpoint_layer(name, module, always=False):
                        self._wrap_with_checkpoint(module)
                        checkpointed_layers.append(name)

        self.memory_stats["checkpointed_layers"] = checkpointed_layers
        logger.info(f"Gradient checkpointing applied to {len(checkpointed_layers)} layers")

        return model

    def _should_checkpoint_layer(self, name: str, module: nn.Module, always: bool = False) -> bool:
        """Determine if a layer should be checkpointed."""
        if always:
            # Check if it's a substantial layer worth checkpointing
            param_count = sum(p.numel() for p in module.parameters())
            return param_count > 1_000_000  # Checkpoint layers with >1M parameters

        # Selective checkpointing for specific layer types
        checkpoint_patterns = [
            'attention', 'attn', 'transformer', 'block',
            'ffn', 'feedforward', 'mlp', 'expert'
        ]

        return any(pattern in name.lower() for pattern in checkpoint_patterns)

    def _wrap_with_checkpoint(self, module: nn.Module):
        """Wrap module forward with gradient checkpointing."""
        original_forward = module.forward

        def checkpointed_forward(*args, **kwargs):
            # Use checkpoint with use_reentrant=False for better memory efficiency
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

        module.forward = checkpointed_forward

    def _setup_cpu_offloading(self, model: nn.Module) -> nn.Module:
        """Setup CPU offloading for large models."""
        offloaded_layers = []

        # Identify large layers for offloading
        for name, module in model.named_modules():
            param_size = sum(p.numel() * p.element_size() for p in module.parameters())
            param_size_gb = param_size / (1024 ** 3)

            # Offload layers larger than 1GB
            if param_size_gb > 1.0:
                self._setup_layer_offloading(module)
                offloaded_layers.append(name)

        self.memory_stats["offloaded_layers"] = offloaded_layers
        if offloaded_layers:
            logger.info(f"CPU offloading enabled for {len(offloaded_layers)} large layers")

        return model

    def _setup_layer_offloading(self, module: nn.Module):
        """Setup CPU offloading for a specific layer."""
        original_forward = module.forward

        def offloaded_forward(*args, **kwargs):
            # Move to GPU for computation
            module.cuda()
            output = original_forward(*args, **kwargs)
            # Move back to CPU after computation
            module.cpu()
            torch.cuda.empty_cache()
            return output

        module.forward = offloaded_forward

    def _optimize_buffers(self, model: nn.Module):
        """Optimize buffer allocation for A100."""
        # Set all buffers to half precision if possible
        for buffer_name, buffer in model.named_buffers():
            if buffer.dtype == torch.float32:
                # Convert to bfloat16 for A100 (maintains better precision than fp16)
                if torch.cuda.is_bf16_supported():
                    buffer.data = buffer.data.to(torch.bfloat16)

    @contextmanager
    def memory_efficient_forward(self, clear_cache: bool = True):
        """
        Context manager for memory-efficient forward passes.

        Args:
            clear_cache: Whether to clear cache after forward
        """
        # Record initial memory
        if self.profile_memory:
            initial_memory = self._get_current_memory_gb()

        try:
            yield
        finally:
            # Clear cache if needed
            if clear_cache:
                torch.cuda.empty_cache()
                gc.collect()

            # Update statistics
            if self.profile_memory:
                final_memory = self._get_current_memory_gb()
                self.memory_stats["peak_memory_gb"] = max(
                    self.memory_stats["peak_memory_gb"],
                    final_memory
                )

            # Periodic cache clearing
            self.step_count += 1
            if self.step_count % self.clear_cache_frequency == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.memory_stats["cache_clears"] += 1

    def optimize_batch_size(self, model: nn.Module, target_memory_usage: float = 0.8) -> int:
        """
        Find optimal batch size for given memory constraints.

        Args:
            model: Model to optimize
            target_memory_usage: Target memory usage as fraction

        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 1

        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = total_memory * target_memory_usage

        # Start with a small batch size and increase
        batch_size = 1
        seq_length = 512  # Default sequence length

        while True:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, seq_length, model.config.hidden_size if hasattr(model, 'config') else 768)
                dummy_input = dummy_input.to(self.device)

                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)

                # Check memory usage
                current_memory = torch.cuda.memory_allocated()

                if current_memory > target_memory:
                    # Exceeded target, use previous batch size
                    batch_size = max(1, batch_size - 1)
                    break

                # Try larger batch size
                batch_size *= 2

                # Clear for next iteration
                del dummy_input
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                # OOM, use previous batch size
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                break

        logger.info(f"Optimal batch size: {batch_size} for {target_memory_usage:.0%} memory usage")
        return batch_size

    def _get_current_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get current memory usage: {e}")
            return 0.0

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if not torch.cuda.is_available():
            return {}

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

            return {
                "allocated_gb": allocated / (1024 ** 3),
                "reserved_gb": reserved / (1024 ** 3),
                "free_gb": (total_memory - allocated) / (1024 ** 3),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
                "checkpointed_layers": len(self.memory_stats["checkpointed_layers"]),
                "offloaded_layers": len(self.memory_stats["offloaded_layers"]),
                "cache_clears": self.memory_stats["cache_clears"]
            }
        except Exception as e:
            logger.warning(f"Failed to get memory summary: {e}")
            return {"error": str(e)}

    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.memory_stats["peak_memory_gb"] = 0.0

    def get_deepspeed_memory_stats(self) -> Dict[str, Any]:
        """Get DeepSpeed-specific memory statistics."""
        if not self.is_deepspeed_enabled:
            return self.get_memory_summary()

        stats = self.get_memory_summary()

        # Add DeepSpeed-specific information
        if self.deepspeed_engine and hasattr(self.deepspeed_engine, 'monitor'):
            try:
                monitor = self.deepspeed_engine.monitor
                ds_stats = monitor.get_memory_usage()
                stats.update({
                    "deepspeed_zero_stage": self.deepspeed_zero_stage,
                    "cpu_offload_enabled": self.deepspeed_cpu_offload,
                    "nvme_offload_enabled": self.deepspeed_nvme_offload,
                    "deepspeed_memory": ds_stats
                })
            except Exception as e:
                logger.warning(f"Failed to get DeepSpeed memory stats: {e}")

        return stats

    def optimize_for_deepspeed(self, model: torch.nn.Module) -> None:
        """Apply DeepSpeed-specific memory optimizations."""
        if not self.is_deepspeed_enabled or not DEEPSPEED_AVAILABLE:
            logger.info("DeepSpeed not enabled, skipping DeepSpeed optimizations")
            return

        logger.info("Applying DeepSpeed memory optimizations...")

        # Configure gradient checkpointing for DeepSpeed
        if self.enable_gradient_checkpointing:
            try:
                if hasattr(model, 'enable_deepspeed_checkpointing'):
                    model.enable_deepspeed_checkpointing()
                else:
                    # Apply generic gradient checkpointing
                    self._apply_deepspeed_checkpointing(model)

                logger.info(" DeepSpeed gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"DeepSpeed gradient checkpointing failed: {e}")

        # Optimize memory allocation patterns for ZeRO
        if self.deepspeed_zero_stage > 0:
            self._optimize_zero_memory_patterns()

        logger.info(" DeepSpeed memory optimizations applied")

    def _apply_deepspeed_checkpointing(self, model: torch.nn.Module) -> None:
        """Apply DeepSpeed-style gradient checkpointing."""
        if not DEEPSPEED_AVAILABLE:
            return

        # Apply checkpointing to transformer layers
        for name, module in model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'transformer']):
                if hasattr(module, 'forward'):
                    # Wrap with DeepSpeed checkpointing
                    try:
                        module.forward = deepspeed.checkpointing.checkpoint(module.forward)
                        self.memory_stats["checkpointed_layers"].append(name)
                        logger.debug(f"Applied DeepSpeed checkpointing to {name}")
                    except Exception as e:
                        logger.warning(f"Failed to apply DeepSpeed checkpointing to {name}: {e}")

    def _optimize_zero_memory_patterns(self) -> None:
        """Optimize memory allocation patterns for ZeRO."""
        logger.info(f"Optimizing memory patterns for ZeRO stage {self.deepspeed_zero_stage}")

        # Clear cache to reset memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Adjust cache clearing frequency based on ZeRO stage
        if self.deepspeed_zero_stage >= 2:
            # More aggressive cache clearing for ZeRO-2 and ZeRO-3
            self.clear_cache_frequency = max(50, self.clear_cache_frequency // 2)

        # Reduce memory pool size if using CPU offloading
        if self.deepspeed_cpu_offload:
            logger.info("CPU offload detected, reducing GPU memory pool")
            if self.pool_size_gb:
                self.pool_size_gb = min(self.pool_size_gb, 20.0)  # Reduce to 20GB max

    def handle_deepspeed_oom(self) -> bool:
        """Handle DeepSpeed-specific OOM scenarios."""
        if not self.is_deepspeed_enabled:
            return False

        logger.warning(" DeepSpeed OOM detected, attempting recovery...")

        try:
            # 1. Clear all caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 2. If using ZeRO-3, trigger parameter gathering cleanup
            if self.deepspeed_zero_stage == 3 and self.deepspeed_engine:
                if hasattr(self.deepspeed_engine, 'free_parameter_partitions'):
                    self.deepspeed_engine.free_parameter_partitions()

            # 3. Force garbage collection
            gc.collect()

            # 4. Reset memory pool if enabled
            if self.enable_memory_pool:
                self._setup_memory_pool()

            logger.info(" DeepSpeed OOM recovery attempted")
            return True

        except Exception as e:
            logger.error(f" DeepSpeed OOM recovery failed: {e}")
            return False


class GradientAccumulator:
    """
    Efficient gradient accumulation for large batch training on A100.
    """

    def __init__(
        self,
        accumulation_steps: int = 4,
        use_gradient_scaling: bool = True,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate
            use_gradient_scaling: Use gradient scaling for stability
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.accumulation_steps = accumulation_steps
        self.use_gradient_scaling = use_gradient_scaling
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

        if use_gradient_scaling:
            self.scaler = torch.cuda.amp.GradScaler()

    def should_update_weights(self) -> bool:
        """Check if weights should be updated."""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def accumulate_gradients(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Accumulate gradients with proper scaling.

        Args:
            loss: Loss tensor
            model: Model
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler

        Returns:
            Dictionary of metrics
        """
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps

        # Backward pass
        if self.use_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {"loss": loss.item() * self.accumulation_steps}

        # Update weights if needed
        if self.should_update_weights():
            if self.use_gradient_scaling:
                # Unscale gradients
                self.scaler.unscale_(optimizer)

            # Clip gradients
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                metrics["grad_norm"] = grad_norm.item() if grad_norm is not None else 0.0
            except Exception as e:
                logger.warning(f"Gradient clipping failed: {e}")
                metrics["grad_norm"] = 0.0

            # Optimizer step
            if self.use_gradient_scaling:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient

        self.step_count += 1
        return metrics


class ActivationCheckpointing:
    """
    Advanced activation checkpointing for transformer models on A100.
    """

    @staticmethod
    def apply_to_transformer(
        model: nn.Module,
        checkpoint_ratio: float = 0.5,
        min_layers: int = 4
    ) -> nn.Module:
        """
        Apply checkpointing to transformer layers.

        Args:
            model: Transformer model
            checkpoint_ratio: Ratio of layers to checkpoint
            min_layers: Minimum number of layers to checkpoint

        Returns:
            Model with checkpointing
        """
        # Find all transformer blocks
        transformer_blocks = []
        for name, module in model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                if hasattr(module, 'forward'):
                    transformer_blocks.append((name, module))

        # Calculate number of layers to checkpoint
        num_layers = len(transformer_blocks)
        num_checkpoint = max(min_layers, int(num_layers * checkpoint_ratio))

        # Apply checkpointing to selected layers
        # Checkpoint every other layer for better memory/compute trade-off
        for i, (name, module) in enumerate(transformer_blocks):
            if i % (num_layers // num_checkpoint) == 0:
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

                module.forward = checkpointed_forward
                logger.info(f"Checkpointed layer: {name}")

        return model


def profile_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_steps: int = 10
) -> Dict[str, Any]:
    """
    Profile memory usage of a model on A100.

    Args:
        model: Model to profile
        input_shape: Input tensor shape
        num_steps: Number of steps to profile

    Returns:
        Memory profiling results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = {
        "steps": [],
        "memory_allocated": [],
        "memory_reserved": [],
        "memory_cached": []
    }

    # Profile forward and backward passes
    for step in range(num_steps):
        x = torch.randn(input_shape).to(device)

        # Forward pass
        output = model(x)
        loss = output.mean()  # Dummy loss

        # Backward pass
        loss.backward()

        # Record memory stats
        results["steps"].append(step)
        results["memory_allocated"].append(torch.cuda.memory_allocated() / (1024 ** 3))
        results["memory_reserved"].append(torch.cuda.memory_reserved() / (1024 ** 3))
        results["memory_cached"].append(torch.cuda.memory_reserved() / (1024 ** 3))

        # Clear gradients
        model.zero_grad()

    # Summary statistics
    results["summary"] = {
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "avg_memory_gb": sum(results["memory_allocated"]) / len(results["memory_allocated"]),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    return results