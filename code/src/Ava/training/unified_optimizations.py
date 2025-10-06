"""
Unified Training Optimizations

Brings together all optimization modules into a single unified interface.
Provides one-line training optimization with intelligent defaults.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass, field

# Import all optimization modules
from ..optimization.gradient_optimizations import (
    MixedPrecisionManager,
    GradientCompressor,
    AdaptiveGradientClipper,
    GradientNoiseInjector
)
from ..optimization.fused_optimizers import create_optimizer
from ..optimization.compilation_optimizations import (
    optimize_for_training,
    CompilationManager
)
from ..optimization.hardware_optimizations import auto_optimize_hardware
from ..data.optimized_dataloader import create_production_dataloader
from ..layers.advanced_attention import FlashAttentionWrapper
from ..training.profiling_tools import ThroughputTracker, MemoryProfiler
from ..training.distributed_optimizations import FSDPManager, setup_distributed_training

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for all optimizations."""

    # Gradient optimizations
    use_mixed_precision: bool = True
    mixed_precision_dtype: Optional[str] = None  # None for auto
    use_gradient_compression: bool = False
    compression_method: str = 'powersgd'
    compression_ratio: float = 0.1
    use_adaptive_clipping: bool = True
    clip_type: str = 'adaptive'
    base_clip_value: float = 1.0
    use_gradient_noise: bool = False

    # Optimizer
    optimizer_type: str = 'fused_adam'  # 'fused_adam', 'adam8bit', 'lion', 'sophia'
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_sequence_packing: bool = False
    pack_block_size: int = 2048
    use_dynamic_batching: bool = False
    max_tokens: int = 8192

    # Attention
    use_flash_attention: bool = True
    attention_window_size: Optional[int] = None

    # Compilation
    use_torch_compile: bool = True
    compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    compile_backend: str = 'inductor'

    # Hardware
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    auto_optimize_hardware: bool = True

    # Distributed
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = 'full'
    fsdp_cpu_offload: bool = False

    # Profiling
    enable_profiling: bool = False
    enable_throughput_tracking: bool = True
    enable_memory_profiling: bool = True
    profiling_output_dir: str = './profiler_traces'

    # Loss optimizations
    use_vocab_parallel: bool = False
    use_sampled_softmax: bool = False
    num_sampled_classes: int = 8192


class UnifiedOptimizer:
    """
    Unified optimizer that applies all optimizations automatically.

    One-stop shop for optimizing your training pipeline.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize unified optimizer.

        Args:
            config: Optimization configuration (uses defaults if None)
        """
        self.config = config or OptimizationConfig()

        # Initialize components
        self.mixed_precision: Optional[MixedPrecisionManager] = None
        self.gradient_compressor: Optional[GradientCompressor] = None
        self.gradient_clipper: Optional[AdaptiveGradientClipper] = None
        self.gradient_noise: Optional[GradientNoiseInjector] = None
        self.throughput_tracker: Optional[ThroughputTracker] = None
        self.memory_profiler: Optional[MemoryProfiler] = None
        self.fsdp_manager: Optional[FSDPManager] = None

        logger.info("Unified optimizer initialized")

    def optimize_model(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Optimize model with all enabled optimizations.

        Args:
            model: Model to optimize
            example_inputs: Example inputs for compilation warmup

        Returns:
            Optimized model
        """
        logger.info("=" * 60)
        logger.info("OPTIMIZING MODEL")
        logger.info("=" * 60)

        # Hardware optimizations (first, affects everything else)
        if self.config.auto_optimize_hardware:
            logger.info("\n[1/5] Hardware Optimizations")
            auto_optimize_hardware()

        # Distributed training setup
        if self.config.use_fsdp:
            logger.info("\n[2/5] Distributed Training (FSDP)")
            self.fsdp_manager = FSDPManager(
                sharding_strategy=self.config.fsdp_sharding_strategy,
                cpu_offload=self.config.fsdp_cpu_offload,
                mixed_precision=self.config.use_mixed_precision
            )
            model = self.fsdp_manager.wrap_model(model)

        # Compilation
        if self.config.use_torch_compile and not self.config.use_fsdp:
            logger.info("\n[3/5] Model Compilation (torch.compile)")
            model = optimize_for_training(
                model,
                example_inputs=example_inputs,
                compile_mode=self.config.compile_mode
            )

        logger.info("\n[4/5] Memory Optimizations")
        # Memory optimizations are handled by FSDP or can be added separately

        logger.info("\n[5/5] Gradient Optimizations")
        # Mixed precision
        if self.config.use_mixed_precision:
            self.mixed_precision = MixedPrecisionManager(
                enabled=True,
                dtype=self._get_precision_dtype()
            )
            logger.info(f"  ✓ Mixed precision: {self.mixed_precision.dtype}")

        # Gradient compression
        if self.config.use_gradient_compression:
            self.gradient_compressor = GradientCompressor(
                method=self.config.compression_method,
                compression_ratio=self.config.compression_ratio
            )
            logger.info(f"  ✓ Gradient compression: {self.config.compression_method}")

        # Adaptive clipping
        if self.config.use_adaptive_clipping:
            self.gradient_clipper = AdaptiveGradientClipper(
                clip_type=self.config.clip_type,
                base_clip_value=self.config.base_clip_value
            )
            logger.info(f"  ✓ Adaptive gradient clipping: {self.config.clip_type}")

        # Gradient noise
        if self.config.use_gradient_noise:
            self.gradient_noise = GradientNoiseInjector()
            logger.info("  ✓ Gradient noise injection")

        logger.info("\n" + "=" * 60)
        logger.info("MODEL OPTIMIZATION COMPLETE")
        logger.info("=" * 60 + "\n")

        return model

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create optimized optimizer.

        Args:
            model: Model to optimize

        Returns:
            Configured optimizer
        """
        logger.info(f"Creating optimizer: {self.config.optimizer_type}")

        optimizer = create_optimizer(
            model,
            optimizer_type=self.config.optimizer_type,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas
        )

        return optimizer

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        device: Optional[torch.device] = None
    ) -> DataLoader:
        """
        Create optimized dataloader.

        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Shuffle data
            device: Target device

        Returns:
            Optimized DataLoader
        """
        logger.info("Creating optimized dataloader...")

        dataloader = create_production_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            num_workers=self.config.num_workers,
            use_packing=self.config.use_sequence_packing,
            pack_block_size=self.config.pack_block_size,
            use_dynamic_batching=self.config.use_dynamic_batching,
            max_tokens=self.config.max_tokens,
            shuffle=shuffle
        )

        return dataloader

    def setup_profiling(self):
        """Setup profiling tools."""
        if self.config.enable_throughput_tracking:
            self.throughput_tracker = ThroughputTracker(
                window_size=100,
                log_interval=10
            )
            logger.info("✓ Throughput tracking enabled")

        if self.config.enable_memory_profiling:
            self.memory_profiler = MemoryProfiler(
                check_interval=100
            )
            logger.info("✓ Memory profiling enabled")

    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Optimized training step.

        Args:
            model: Model
            batch: Input batch
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Start timing
        if self.throughput_tracker:
            self.throughput_tracker.start_batch()
            self.throughput_tracker.start_forward()

        # Forward pass with mixed precision
        if self.mixed_precision:
            with self.mixed_precision.autocast():
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        else:
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']

        if self.throughput_tracker:
            self.throughput_tracker.end_forward()
            self.throughput_tracker.start_backward()

        # Backward pass
        if self.mixed_precision:
            scaled_loss = self.mixed_precision.scale_loss(loss)
            scaled_loss.backward()
        else:
            loss.backward()

        if self.throughput_tracker:
            self.throughput_tracker.end_backward()
            self.throughput_tracker.start_optimizer()

        # Gradient clipping
        if self.gradient_clipper:
            clip_metrics = self.gradient_clipper.clip_gradients(model)
            metrics.update(clip_metrics)

        # Gradient noise
        if self.gradient_noise:
            noise_scale = self.gradient_noise.inject_noise(model.parameters())
            metrics['gradient_noise_scale'] = noise_scale

        # Optimizer step
        if self.mixed_precision:
            step_metrics = self.mixed_precision.step_optimizer(optimizer)
            metrics.update(step_metrics)
        else:
            optimizer.step()

        optimizer.zero_grad()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        if self.throughput_tracker:
            self.throughput_tracker.end_optimizer()

            # Compute tokens
            num_samples = batch['input_ids'].shape[0]
            num_tokens = batch['input_ids'].numel()
            self.throughput_tracker.end_batch(num_samples, num_tokens)

        # Memory profiling
        if self.memory_profiler:
            self.memory_profiler.step()

        metrics['loss'] = loss.item()

        return metrics

    def _get_precision_dtype(self) -> torch.dtype:
        """Get precision dtype based on config and hardware."""
        if self.config.mixed_precision_dtype == 'bfloat16':
            return torch.bfloat16
        elif self.config.mixed_precision_dtype == 'float16':
            return torch.float16
        else:
            # Auto-detect
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16

    def get_metrics(self) -> Dict[str, Any]:
        """Get all profiling metrics."""
        metrics = {}

        if self.throughput_tracker:
            throughput_metrics = self.throughput_tracker.get_metrics()
            metrics['throughput'] = {
                'samples_per_sec': throughput_metrics.samples_per_sec,
                'tokens_per_sec': throughput_metrics.tokens_per_sec,
                'mfu': throughput_metrics.model_flops_utilization,
                'avg_batch_time_ms': throughput_metrics.avg_batch_time * 1000
            }

        if self.memory_profiler:
            memory_stats = self.memory_profiler.get_memory_stats()
            metrics['memory'] = memory_stats

        return metrics


def quick_optimize(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    **config_kwargs
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, UnifiedOptimizer]:
    """
    Quick one-line optimization for training.

    Args:
        model: Model to optimize
        dataset: Training dataset
        batch_size: Batch size
        learning_rate: Learning rate
        device: Target device
        **config_kwargs: Additional configuration arguments

    Returns:
        Tuple of (optimized_model, optimizer, dataloader, unified_optimizer)

    Example:
        >>> model, optimizer, dataloader, opt_manager = quick_optimize(
        ...     model, dataset, batch_size=32, learning_rate=1e-3
        ... )
        >>> for batch in dataloader:
        ...     metrics = opt_manager.training_step(model, batch, optimizer)
    """
    # Create config
    config = OptimizationConfig(learning_rate=learning_rate, **config_kwargs)

    # Create unified optimizer
    unified_opt = UnifiedOptimizer(config)

    # Optimize model
    model = unified_opt.optimize_model(model)

    # Create optimizer
    optimizer = unified_opt.create_optimizer(model)

    # Create dataloader
    dataloader = unified_opt.create_dataloader(dataset, batch_size, device=device)

    # Setup profiling
    unified_opt.setup_profiling()

    logger.info("Quick optimization complete! Ready to train.")

    return model, optimizer, dataloader, unified_opt
