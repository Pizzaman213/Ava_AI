"""
Model builder for the Ava pipeline.

Handles model creation, quantization, and optimization setup.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .context import TrainingComponent, TrainingContext

logger = logging.getLogger(__name__)

# Check distributed availability
try:
    from torch.nn.parallel import DistributedDataParallel
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class ModelBuilder(TrainingComponent):
    """
    Builds and configures the model with all optimizations.

    Handles:
        - MoE model creation from config
        - Quantization (INT8, NF4, NVFP4)
        - FP8 training
        - Hybrid caching
        - Overlapped checkpointing
        - Double checkpointing
        - torch.compile
        - DDP wrapping

    Example:
        >>> context = TrainingContext(model=None, device=device)
        >>> builder = ModelBuilder(context)
        >>> builder.initialize()
        >>> model = builder.build_model(config, device)
        >>> model = builder.apply_optimizations(model, config)
        >>> model = builder.wrap_distributed(model, rank)
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the model builder.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self._built_model: Optional[nn.Module] = None
        self.model_config: Optional[Any] = None
        self._total_params: int = 0

    def initialize(self) -> None:
        """Initialize the model builder."""
        self._initialized = True
        self.logger.debug("ModelBuilder initialized")

    def cleanup(self) -> None:
        """Release model resources."""
        if self._built_model is not None:
            del self._built_model
            self._built_model = None
            torch.cuda.empty_cache()

    def build_model(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """
        Build MoE model from configuration.

        Args:
            config: Full config dict with 'model' section
            device: Target device

        Returns:
            Configured model on device

        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model creation fails
        """
        self.assert_initialized()

        # Import model classes
        try:
            from ava.models.moe import EnhancedMoEConfig, EnhancedMoEModel
        except ImportError as e:
            raise RuntimeError(f"Failed to import MoE model: {e}")

        model_config = config.get('model', {})
        is_main = self.context.metadata.get('is_main_process', True)

        if is_main:
            self.logger.info("Creating MoE model...")

        # Build MoE config from YAML config
        moe_config = EnhancedMoEConfig(
            vocab_size=model_config.get('vocab_size', 50680),
            hidden_size=model_config.get('hidden_size', 512),
            num_layers=model_config.get('num_layers', 8),
            num_attention_heads=model_config.get('num_attention_heads', 8),
            intermediate_size=model_config.get('intermediate_size',
                                                model_config.get('hidden_size', 512) * 4),
            num_experts=model_config.get('num_experts', 2),
            num_experts_per_token=model_config.get('num_experts_per_token', 1),
            max_position_embeddings=model_config.get('max_position_embeddings', 2048),
            router_type=model_config.get('router_type', 'switch'),
            expert_capacity_factor=model_config.get('capacity_factor', 1.25),
            attention_dropout=model_config.get('attention_dropout', 0.1),
            dropout=model_config.get('dropout', 0.1),
            use_flash_attention=model_config.get('use_flash_attention', False),
            router_aux_loss_coef=model_config.get('router_z_loss_coef', 0.01),
            router_jitter_noise=model_config.get('router_jitter_noise', 0.01),
            # Performance optimization flags (CRITICAL - these were missing!)
            gradient_checkpointing=model_config.get('gradient_checkpointing', False),
            use_grouped_gemm=model_config.get('use_grouped_gemm', False),
            use_triton_kernels=model_config.get('use_triton_kernels', False),
            use_torch_compile=model_config.get('use_torch_compile', False),
            use_optimized_moe=model_config.get('use_optimized_moe', False),
        )

        self.model_config = moe_config

        # Create model
        model = EnhancedMoEModel(moe_config)
        self._built_model = model

        # Log model info
        self._total_params = sum(p.numel() for p in model.parameters())
        if is_main:
            self.logger.info(f"  Parameters: {self._total_params:,} ({self._total_params/1e6:.1f}M)")

        return model

    def apply_quantization(
        self,
        model: nn.Module,
        quant_config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply quantization to model.

        Args:
            model: Model to quantize
            quant_config: Quantization configuration dict

        Returns:
            Quantized model
        """
        from ..optimizations.quantization import apply_quantization

        is_main = self.context.metadata.get('is_main_process', True)

        if quant_config.get('enabled', False):
            quant_type = quant_config.get('type', 'none')
            if is_main:
                self.logger.info(f"Applying {quant_type} quantization...")
            model = apply_quantization(
                model, quant_config,
                logger=self.logger if is_main else None
            )

        return model

    def apply_optimizations(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply all configured optimizations.

        Args:
            model: Model to optimize
            config: Full configuration dict

        Returns:
            Optimized model
        """
        is_main = self.context.metadata.get('is_main_process', True)

        # Track failed optimizations for summary warning
        failed_optimizations: list = []
        applied_optimizations: list = []

        # Apply FP8 training
        model, success = self._apply_fp8(model, config, is_main)
        if config.get('fp8', {}).get('enabled', False):
            if success:
                applied_optimizations.append('FP8')
            else:
                failed_optimizations.append('FP8')

        # Apply hybrid caching
        model, success = self._apply_hybrid_caching(model, config, is_main)
        if config.get('hybrid_caching', {}).get('enabled', False):
            if success:
                applied_optimizations.append('Hybrid Caching')
            else:
                failed_optimizations.append('Hybrid Caching')

        # Apply overlapped checkpointing
        model, success = self._apply_overlapped_checkpointing(model, config, is_main)
        oc_config = config.get('overlapped_checkpointing', {})
        opt_config = config.get('optimizations', {})
        if oc_config.get('enabled', False) or opt_config.get('use_overlapped_checkpointing', False):
            if success:
                applied_optimizations.append('Overlapped Checkpointing')
            else:
                failed_optimizations.append('Overlapped Checkpointing')

        # Apply double checkpointing
        model, success = self._apply_double_checkpointing(model, config, is_main)
        if config.get('double_checkpointing', {}).get('enabled', False):
            if success:
                applied_optimizations.append('Double Checkpointing')
            else:
                failed_optimizations.append('Double Checkpointing')

        # Apply torch.compile
        model, success = self._apply_torch_compile(model, config, is_main)
        perf_config = config.get('performance', {})
        if perf_config.get('enable_torch_compile', False):
            if success:
                applied_optimizations.append('torch.compile')
            else:
                failed_optimizations.append('torch.compile')

        # PERFORMANCE FIX: Emit prominent warning if any optimizations failed
        if is_main and failed_optimizations:
            self.logger.warning("=" * 60)
            self.logger.warning("⚠️  PERFORMANCE DEGRADATION WARNING")
            self.logger.warning("=" * 60)
            self.logger.warning(f"The following optimizations FAILED to apply:")
            for opt in failed_optimizations:
                self.logger.warning(f"  ❌ {opt}")
            self.logger.warning("")
            self.logger.warning("Your training will run SLOWER than expected!")
            self.logger.warning("Check the error logs above for details.")
            self.logger.warning("=" * 60)

        if is_main and applied_optimizations:
            self.logger.info(f"✅ Applied optimizations: {', '.join(applied_optimizations)}")

        return model

    def _apply_fp8(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply FP8 training if enabled.

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        fp8_config = config.get('fp8', {}) or \
                     config.get('training', {}).get('precision', {}).get('fp8', {})

        if not fp8_config.get('enabled', False):
            return model, True  # Not enabled is not a failure

        try:
            from ava.optimizations.fp8 import (
                apply_fp8_training,
                FP8Config,
            )

            fp8_cfg = FP8Config(
                enabled=True,
                use_transformer_engine=fp8_config.get('use_transformer_engine', True),
                amax_history_len=fp8_config.get('amax_history_len', 1024),
                fp8_format=fp8_config.get('format', 'e4m3'),
            )

            if is_main:
                self.logger.info(f"Applying FP8 training (format={fp8_cfg.fp8_format})...")

            model = apply_fp8_training(model, fp8_cfg)

            if is_main:
                self.logger.info("  FP8 training enabled (2-3x speedup on Hopper/Ada GPUs)")

            return model, True

        except Exception as e:
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - FP8 training: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return model, False

    def _apply_hybrid_caching(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply hybrid caching if enabled.

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        hybrid_config = config.get('hybrid_caching', {})

        if not hybrid_config.get('enabled', False):
            return model, True  # Not enabled is not a failure

        try:
            from ava.optimizations.hybrid_cache import (
                apply_hybrid_caching,
                HybridCacheConfig,
            )

            cache_config = HybridCacheConfig(
                enabled=True,
                max_cache_size_gb=hybrid_config.get('max_cache_size_gb', 0.5),
                kv_cache_ratio=hybrid_config.get('kv_cache_ratio', 0.7),
                eviction_policy=hybrid_config.get('eviction_policy', 'hybrid'),
                prefetch_enabled=hybrid_config.get('prefetch_enabled', False),
                prefetch_lookahead=hybrid_config.get('prefetch_lookahead', 2),
                min_score_threshold=hybrid_config.get('min_score_threshold', 0.1),
            )

            if is_main:
                self.logger.info(f"Applying hybrid caching ({cache_config.max_cache_size_gb}GB)...")

            model, _ = apply_hybrid_caching(model, cache_config)

            if is_main:
                self.logger.info("  Hybrid caching applied (20-30% throughput improvement)")

            return model, True

        except Exception as e:
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Hybrid caching: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return model, False

    def _apply_overlapped_checkpointing(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply overlapped checkpointing if enabled.

        Checks both config locations:
        - overlapped_checkpointing.enabled (YAML config style)
        - optimizations.use_overlapped_checkpointing (legacy style)

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        # Check both config locations for backwards compatibility
        oc_config = config.get('overlapped_checkpointing', {})
        opt_config = config.get('optimizations', {})

        enabled = (
            oc_config.get('enabled', False) or
            opt_config.get('use_overlapped_checkpointing', False)
        )

        if not enabled:
            return model, True  # Not enabled is not a failure

        # Get stream_overlap setting from config (uses optimized StreamPool)
        stream_overlap = oc_config.get('stream_overlap', True)
        layer_pattern = oc_config.get('target_layers', 'layers')

        try:
            from ava.optimizations.overlapped_recomputation import (
                apply_overlapped_checkpointing,
            )

            if is_main:
                self.logger.info("Applying overlapped checkpointing...")
                self.logger.info(f"  stream_overlap={stream_overlap} (uses CUDA StreamPool)")

            model = apply_overlapped_checkpointing(
                model,
                layer_pattern=layer_pattern,
                stream_overlap=stream_overlap
            )

            if is_main:
                self.logger.info("  Overlapped checkpointing applied (10-20% speedup)")

            return model, True

        except Exception as e:
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Overlapped checkpointing: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return model, False

    def _apply_double_checkpointing(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply double checkpointing if enabled.

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        dc_config = config.get('double_checkpointing', {})

        if not dc_config.get('enabled', False):
            return model, True  # Not enabled is not a failure

        try:
            from ava.optimizations.checkpointing import (
                apply_double_checkpointing,
                DoubleCheckpointConfig,
            )

            double_config = DoubleCheckpointConfig(
                enabled=True,
                coarse_checkpoint_interval=dc_config.get('coarse_checkpoint_interval', 8),
                fine_checkpoint_interval=dc_config.get('fine_checkpoint_interval', 2),
                use_cuda_streams=dc_config.get('use_cuda_streams', True),
            )

            if is_main:
                self.logger.info("Applying double checkpointing...")

            model = apply_double_checkpointing(
                model,
                config=double_config,
                target_modules=["layers"]
            )

            if is_main:
                self.logger.info("  Double checkpointing applied (10x longer sequences)")

            return model, True

        except Exception as e:
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Double checkpointing: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return model, False

    def _apply_torch_compile(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply torch.compile if enabled.

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        perf_config = config.get('performance', {})

        if not perf_config.get('enable_torch_compile', False):
            return model, True  # Not enabled is not a failure

        compile_mode = perf_config.get('torch_compile_mode', 'reduce-overhead')
        compile_dynamic = perf_config.get('torch_compile_dynamic', True)
        compile_fullgraph = perf_config.get('torch_compile_fullgraph', False)

        if is_main:
            self.logger.info(f"Applying torch.compile (mode={compile_mode})...")

        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=compile_fullgraph,
            )

            if is_main:
                self.logger.info("  torch.compile applied (15-25% speedup after warmup)")

            return model, True

        except Exception as e:
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - torch.compile: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return model, False

    def wrap_distributed(
        self,
        model: nn.Module,
        rank: int,
        world_size: int = 1,
    ) -> nn.Module:
        """
        Wrap model in DDP for distributed training.

        Args:
            model: Model to wrap
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            DDP-wrapped model if distributed, original model otherwise
        """
        if world_size > 1 and DISTRIBUTED_AVAILABLE:
            model = DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True  # Required for MoE - not all experts used every batch
            )
            if rank == 0:
                self.logger.info(f"Model wrapped in DDP (world_size={world_size})")

        return model

    def move_to_device(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Move model to target device.

        Args:
            model: Model to move
            device: Target device

        Returns:
            Model on target device
        """
        model = model.to(device)
        self.context.model = model
        return model

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[nn.Module] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into (uses self._built_model if None)

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        if model is None:
            model = self._built_model

        if model is None:
            raise RuntimeError("No model available. Call build_model first.")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return model, checkpoint

    def get_total_params(self) -> int:
        """Get total number of model parameters."""
        return self._total_params

    def get_status(self) -> Dict[str, Any]:
        """Return current model builder status."""
        return {
            'total_params': self._total_params,
            'model_config': str(self.model_config) if self.model_config else None,
            'has_model': self._built_model is not None,
        }
