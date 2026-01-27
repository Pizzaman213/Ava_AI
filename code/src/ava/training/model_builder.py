"""
Model builder for the Ava pipeline.

Handles model creation, quantization, and optimization setup.

Optimization Application Order:
    Optimizations must be applied in a specific order due to dependencies:

    1. CREATE MODEL (on CPU)
       └─ EnhancedMoEModel(config)

    2. APPLY QUANTIZATION (on CPU, before device move)
       └─ INT8, NF4, NVFP4 quantization
       └─ Must happen before moving to GPU for memory savings

    3. APPLY PRE-DEVICE OPTIMIZATIONS (on CPU)
       └─ torch.compile
       └─ Graph compilation is more reliable on CPU

    4. MOVE TO DEVICE
       └─ model.to(device)

    5. APPLY POST-DEVICE OPTIMIZATIONS (on GPU)
       └─ FP8 training (requires CUDA compute capability 8.9+)
       └─ Hybrid caching (device-specific memory management)
       └─ Overlapped checkpointing (uses CUDA streams)
       └─ Double checkpointing (uses CUDA streams)

    6. WRAP IN DDP (for distributed training)
       └─ DistributedDataParallel(model, device_ids=[rank])
       └─ Must be LAST because DDP wraps the model

Why Order Matters:
    - Quantization on CPU avoids OOM when loading full-precision weights
    - torch.compile on CPU avoids some CUDA graph issues
    - FP8/checkpointing require GPU context
    - DDP must wrap the final optimized model

Fail-Fast Mode:
    set_fail_on_optimization_error(True) raises on first failure.
    Default: log warning and continue (more robust for optional features).
"""

import logging
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.checkpoint import load_state_dict_with_remapping
from .context import TrainingComponent, TrainingContext

logger = logging.getLogger(__name__)

# Config key aliases for backwards compatibility
# Format: canonical_name -> list of accepted aliases (first is canonical)
CONFIG_KEY_ALIASES: Dict[str, List[str]] = {
    'expert_capacity_factor': ['expert_capacity_factor', 'capacity_factor'],
    'router_aux_loss_coef': ['router_aux_loss_coef', 'router_z_loss_coef', 'aux_loss_coef'],
}

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
        # Fail-fast option for optimization errors
        self._fail_on_optimization_error: bool = False
        self._optimization_errors: List[Tuple[str, Exception]] = []

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

    def set_fail_on_optimization_error(self, fail_fast: bool) -> None:
        """
        Configure whether to raise exceptions on optimization failures.

        Args:
            fail_fast: If True, raise exception on first optimization failure.
                       If False (default), log error and continue.
        """
        self._fail_on_optimization_error = fail_fast

    def get_optimization_errors(self) -> List[Tuple[str, Exception]]:
        """Get list of (optimization_name, exception) for all failed optimizations."""
        return self._optimization_errors.copy()

    def _get_config_value(
        self,
        config: Dict[str, Any],
        canonical_key: str,
        default: Any,
    ) -> Any:
        """
        Get config value with alias support for backwards compatibility.

        Checks canonical key first, then aliases. Logs info if alias used.

        Args:
            config: Config dictionary to search
            canonical_key: The canonical (preferred) key name
            default: Default value if no key found

        Returns:
            Config value or default
        """
        aliases = CONFIG_KEY_ALIASES.get(canonical_key, [canonical_key])

        for alias in aliases:
            if alias in config:
                if alias != canonical_key:
                    self.logger.info(
                        f"Config key '{alias}' is deprecated, use '{canonical_key}' instead."
                    )
                return config[alias]

        return default

    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """
        Validate model configuration values.

        Raises ValueError for critical issues, logs warnings for minor ones.

        Args:
            model_config: The model configuration dictionary

        Raises:
            ValueError: If critical validation fails
        """
        issues = []

        # Check for required fields (warn if missing)
        required_fields = ['vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads']
        missing = [f for f in required_fields if f not in model_config]
        if missing:
            self.logger.warning(
                f"Model config missing recommended fields: {missing}. "
                f"Using defaults which may not be optimal."
            )

        # Validate hidden_size divisible by num_attention_heads
        hidden_size = model_config.get('hidden_size', 512)
        num_heads = model_config.get('num_attention_heads', 8)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_heads})"
            )

        # Validate num_experts_per_token <= num_experts
        num_experts = model_config.get('num_experts', 2)
        num_experts_per_token = model_config.get('num_experts_per_token', 1)
        if num_experts_per_token > num_experts:
            raise ValueError(
                f"num_experts_per_token ({num_experts_per_token}) cannot exceed "
                f"num_experts ({num_experts})"
            )

        # Warn about potentially problematic values
        vocab_size = model_config.get('vocab_size', 50680)
        if vocab_size < 1000:
            issues.append(f"vocab_size={vocab_size} seems too small (typical: 30000-100000)")

        capacity_factor = self._get_config_value(
            model_config, 'expert_capacity_factor', 1.25
        )
        if capacity_factor < 1.0:
            issues.append(
                f"expert_capacity_factor={capacity_factor} < 1.0 will cause token dropping"
            )

        for issue in issues:
            self.logger.warning(f"Config warning: {issue}")

    def _report_optimization_status(
        self,
        phase: str,
        applied: List[str],
        failed: List[str],
        is_main: bool,
    ) -> None:
        """Report optimization status with prominent warnings for failures."""
        if is_main and failed:
            self.logger.warning("=" * 60)
            self.logger.warning(f"PERFORMANCE DEGRADATION WARNING ({phase} phase)")
            self.logger.warning("=" * 60)
            self.logger.warning("The following optimizations FAILED to apply:")
            for opt in failed:
                self.logger.warning(f"  - {opt}")
            self.logger.warning("")
            self.logger.warning("Your training will run SLOWER than expected!")
            self.logger.warning("Check the error logs above for details.")
            if not self._fail_on_optimization_error:
                self.logger.warning("Set model_building.fail_on_optimization_error=true to fail fast.")
            self.logger.warning("=" * 60)

        if is_main and applied:
            self.logger.info(f"Applied {phase} optimizations: {', '.join(applied)}")

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

        # Validate model config exists
        model_config = config.get('model')
        if model_config is None:
            raise ValueError(
                "Configuration missing required 'model' section. "
                "Please ensure your YAML config includes a 'model:' block with "
                "at minimum: vocab_size, hidden_size, num_layers, num_attention_heads"
            )

        if not isinstance(model_config, dict):
            raise ValueError(
                f"'model' config must be a dictionary, got {type(model_config).__name__}"
            )

        # Validate model config values
        self._validate_model_config(model_config)

        is_main = self.context.metadata.get('is_main_process', True)

        if is_main:
            self.logger.info("Creating MoE model...")

        # FIX: Get label_smoothing from training.batching section (critical anti-overfitting)
        training_config = config.get('training', {})
        batching_config = training_config.get('batching', {})
        label_smoothing = batching_config.get('label_smoothing', 0.0)

        # Build MoE config from YAML config (using aliases for backwards compatibility)
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
            expert_capacity_factor=self._get_config_value(
                model_config, 'expert_capacity_factor', 1.25
            ),
            attention_dropout=model_config.get('attention_dropout', 0.1),
            dropout=model_config.get('dropout', 0.1),
            use_flash_attention=model_config.get('use_flash_attention', False),
            router_aux_loss_coef=self._get_config_value(
                model_config, 'router_aux_loss_coef', 0.01
            ),
            router_jitter_noise=model_config.get('router_jitter_noise', 0.01),
            # FIX: Activation function for expert layers (swiglu recommended for performance)
            activation=model_config.get('activation', 'swiglu'),
            # Performance optimization flags (CRITICAL - these were missing!)
            gradient_checkpointing=model_config.get('gradient_checkpointing', False),
            use_grouped_gemm=model_config.get('use_grouped_gemm', False),
            use_triton_kernels=model_config.get('use_triton_kernels', False),
            use_torch_compile=config.get('compute', {}).get('performance', {}).get('enable_torch_compile', False),
            use_optimized_moe=model_config.get('use_optimized_moe', False),
            # Coherence regularization settings (fixes coherence issues)
            entropy_regularization=model_config.get('entropy_regularization', 0.0),
            output_diversity_weight=model_config.get('output_diversity_weight', 0.0),
            eos_logit_bias=model_config.get('eos_logit_bias', 0.0),
            eos_token_id=model_config.get('eos_token_id', 3),
            min_sequence_length=model_config.get('min_sequence_length', 0),
            # FIX: Label smoothing from training config (critical anti-overfitting)
            label_smoothing=label_smoothing,
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

    def apply_pre_device_optimizations(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply optimizations that should happen BEFORE moving to device.

        Currently includes:
        - torch.compile (graph compilation works better on CPU)

        Args:
            model: Model to optimize (on CPU)
            config: Full configuration dict

        Returns:
            Optimized model
        """
        is_main = self.context.metadata.get('is_main_process', True)

        failed_optimizations: list = []
        applied_optimizations: list = []

        # Apply torch.compile BEFORE device move for optimal performance
        model, success = self._apply_torch_compile(model, config, is_main)
        perf_config = config.get('performance', {})
        if perf_config.get('enable_torch_compile', False):
            if success:
                applied_optimizations.append('torch.compile')
            else:
                failed_optimizations.append('torch.compile')

        self._report_optimization_status(
            'pre-device', applied_optimizations, failed_optimizations, is_main
        )

        return model

    def apply_post_device_optimizations(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply optimizations that require the model to be on device.

        Includes:
        - FP8 training (requires CUDA)
        - Hybrid caching (device-specific)
        - Overlapped checkpointing (CUDA streams)
        - Double checkpointing (CUDA streams)

        Args:
            model: Model to optimize (on device)
            config: Full configuration dict

        Returns:
            Optimized model
        """
        is_main = self.context.metadata.get('is_main_process', True)

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

        self._report_optimization_status(
            'post-device', applied_optimizations, failed_optimizations, is_main
        )

        return model

    def apply_optimizations(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply all configured optimizations (legacy method).

        DEPRECATED: Use apply_pre_device_optimizations() before move_to_device()
        and apply_post_device_optimizations() after for optimal performance.

        Args:
            model: Model to optimize
            config: Full configuration dict

        Returns:
            Optimized model
        """
        warnings.warn(
            "apply_optimizations() is deprecated. Use apply_pre_device_optimizations() "
            "before move_to_device() and apply_post_device_optimizations() after.",
            DeprecationWarning,
            stacklevel=2
        )
        model = self.apply_pre_device_optimizations(model, config)
        model = self.apply_post_device_optimizations(model, config)
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
                self.logger.info("  FP8 training enabled (optimized for Hopper/Ada GPUs)")

            return model, True

        except Exception as e:
            self._optimization_errors.append(('FP8', e))
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - FP8 training: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

            if self._fail_on_optimization_error:
                raise RuntimeError(
                    f"Optimization 'FP8' failed and fail_on_optimization_error=True: {e}"
                ) from e

            return model, False

    def _apply_hybrid_caching(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        is_main: bool
    ) -> Tuple[nn.Module, bool]:
        """Apply hybrid caching if enabled.

        The new hybrid caching system provides:
        1. ActivationCache (training) - caches activations for gradient checkpointing
        2. KVCacheManager (generation) - intelligent eviction for long sequences

        Returns:
            Tuple of (model, success) where success indicates if optimization was applied
        """
        hybrid_config = config.get('hybrid_caching', {})

        if not hybrid_config.get('enabled', False):
            return model, True  # Not enabled is not a failure

        try:
            from ava.optimizations.hybrid_cache import (
                HybridCacheManager,
                build_hybrid_cache_config,
            )

            # Build config from YAML dict
            cache_config = build_hybrid_cache_config(hybrid_config)

            if is_main:
                act_size = cache_config.activation_cache.max_size_gb
                kv_size = cache_config.kv_cache.max_size_gb
                self.logger.info(f"Applying hybrid caching (activation: {act_size}GB, kv: {kv_size}GB)...")

            # Create cache manager
            cache_manager = HybridCacheManager(cache_config)

            # Attach manager to model (NO wrapping of attention modules)
            model._hybrid_cache_manager = cache_manager

            # Set activation cache on each transformer layer (for training)
            if cache_manager.activation_cache is not None and hasattr(model, 'layers'):
                for idx, layer in enumerate(model.layers):
                    if hasattr(layer, 'set_activation_cache'):
                        layer.set_activation_cache(cache_manager.activation_cache)
                    # Ensure layer_idx is set
                    if hasattr(layer, 'layer_idx'):
                        layer.layer_idx = idx

            # Set KV cache manager reference (for generation)
            if cache_manager.kv_cache_manager is not None:
                model._kv_cache_manager = cache_manager.kv_cache_manager

            if is_main:
                self.logger.info("  Hybrid caching applied (activation + KV cache management)")

            return model, True

        except Exception as e:
            self._optimization_errors.append(('Hybrid Caching', e))
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Hybrid caching: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

            if self._fail_on_optimization_error:
                raise RuntimeError(
                    f"Optimization 'Hybrid Caching' failed and fail_on_optimization_error=True: {e}"
                ) from e

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
                self.logger.info("  Overlapped checkpointing applied")

            return model, True

        except Exception as e:
            self._optimization_errors.append(('Overlapped Checkpointing', e))
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Overlapped checkpointing: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

            if self._fail_on_optimization_error:
                raise RuntimeError(
                    f"Optimization 'Overlapped Checkpointing' failed and fail_on_optimization_error=True: {e}"
                ) from e

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
                self.logger.info("  Double checkpointing applied (enables longer sequences)")

            return model, True

        except Exception as e:
            self._optimization_errors.append(('Double Checkpointing', e))
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - Double checkpointing: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

            if self._fail_on_optimization_error:
                raise RuntimeError(
                    f"Optimization 'Double Checkpointing' failed and fail_on_optimization_error=True: {e}"
                ) from e

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
        # Check both old path (performance) and new path (compute.performance)
        perf_config = config.get('compute', {}).get('performance', {})
        if not perf_config:
            perf_config = config.get('performance', {})  # Fallback for legacy configs

        if not perf_config.get('enable_torch_compile', False):
            return model, True  # Not enabled is not a failure

        compile_mode = perf_config.get('torch_compile_mode', 'reduce-overhead')
        compile_dynamic = perf_config.get('torch_compile_dynamic', True)
        compile_fullgraph = perf_config.get('torch_compile_fullgraph', False)

        if is_main:
            self.logger.info(f"Applying torch.compile (mode={compile_mode})...")

        try:
            # Configure torch._dynamo BEFORE compilation to avoid recompilation issues
            # 1. allow_unspec_int_on_nn_module: Treat integer module attributes (like _step_counter)
            #    as dynamic to prevent recompilation when they change each step
            # 2. suppress_errors: Continue even if dynamo can't trace a function
            # 3. recompile_limit: Increase from default 8 to handle varying sequence lengths
            #    during autoregressive generation (seq_len goes 1→2→3→...→128+)
            import torch._dynamo
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.recompile_limit = 256

            # Store compile settings on model for layers to use compile-friendly dispatch
            model._use_compile_friendly = True
            model._compile_mode = compile_mode
            model._compile_dynamic = compile_dynamic

            model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=compile_fullgraph,
            )

            if is_main:
                self.logger.info("  torch.compile applied (speedup visible after warmup)")
                self.logger.info(f"  Dynamic shapes: {compile_dynamic}, recompile_limit: 256")
                self.logger.info("  Using compile-friendly expert dispatch")

            return model, True

        except Exception as e:
            self._optimization_errors.append(('torch.compile', e))
            if is_main:
                self.logger.error(f"OPTIMIZATION FAILED - torch.compile: {e}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

            # Fail fast if configured
            if self._fail_on_optimization_error:
                raise RuntimeError(
                    f"Optimization 'torch.compile' failed and fail_on_optimization_error=True: {e}"
                ) from e

            return model, False

    def _extract_optimizer_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract optimizer parameters for DeepSpeed from config.

        Args:
            config: Full Ava config dictionary

        Returns:
            Optimizer params dict compatible with DeepSpeed
        """
        training_cfg = config.get('training', {})
        optimizer_cfg = training_cfg.get('optimizer', {})

        # Handle optimizer config as dict or string
        if isinstance(optimizer_cfg, dict):
            opt_type = optimizer_cfg.get('type', 'AdamW')
            opt_lr = optimizer_cfg.get('learning_rate') or training_cfg.get('learning_rate', 1e-4)
            opt_betas = optimizer_cfg.get('betas', (0.9, 0.999))
            opt_eps = optimizer_cfg.get('eps', 1e-8)
            opt_wd = optimizer_cfg.get('weight_decay') or training_cfg.get('weight_decay', 0.01)
        else:
            # optimizer is a string like 'adamw'
            opt_type = 'AdamW'
            opt_lr = training_cfg.get('learning_rate', 1e-4)
            opt_betas = (0.9, 0.999)
            opt_eps = 1e-8
            opt_wd = training_cfg.get('weight_decay', 0.01)

        return {
            'type': opt_type,
            'params': {
                'lr': opt_lr,
                'weight_decay': opt_wd,
                'betas': list(opt_betas) if isinstance(opt_betas, tuple) else opt_betas,
                'eps': opt_eps,
            }
        }

    def _extract_scheduler_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract scheduler parameters for DeepSpeed from config.

        Args:
            config: Full Ava config dictionary

        Returns:
            Scheduler params dict compatible with DeepSpeed
        """
        training_cfg = config.get('training', {})
        scheduler_cfg = training_cfg.get('schedule', {}) or training_cfg.get('scheduler', {})

        warmup_steps = scheduler_cfg.get('warmup_steps', training_cfg.get('warmup_steps', 1000))
        total_steps = training_cfg.get('max_steps', training_cfg.get('total_steps', 100000))
        learning_rate = training_cfg.get('learning_rate', 1e-4)
        min_lr = scheduler_cfg.get('min_lr', training_cfg.get('min_lr', 0))

        return {
            'type': 'WarmupDecayLR',
            'params': {
                'warmup_min_lr': min_lr,
                'warmup_max_lr': learning_rate,
                'warmup_num_steps': warmup_steps,
                'total_num_steps': total_steps,
            }
        }

    def _wrap_with_deepspeed(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        rank: int,
        world_size: int,
    ) -> tuple:
        """
        Initialize DeepSpeed engine with ZeRO optimization.

        Args:
            model: Model to wrap
            config: Full Ava config dictionary
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            Tuple of (model_engine, optimizer, scheduler)
        """
        try:
            import deepspeed
        except ImportError:
            self.logger.error(
                "DeepSpeed enabled but not installed! "
                "Install with: pip install deepspeed"
            )
            self.logger.warning("Falling back to DDP")
            # Fallback to DDP
            wrapped_model = DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )
            return wrapped_model, None, None

        from .deepspeed import build_deepspeed_config, validate_deepspeed_config, validate_model_for_deepspeed

        # Support both v2.0 path (distributed.deepspeed) and legacy path (deepspeed)
        distributed_cfg = config.get('distributed', {})
        deepspeed_cfg = distributed_cfg.get('deepspeed', {})
        if not deepspeed_cfg:
            deepspeed_cfg = config.get('deepspeed', {})

        # Validate model for DeepSpeed compatibility BEFORE building config
        # This catches tie_word_embeddings=True which causes "parameter already reduced" errors
        try:
            validate_model_for_deepspeed(model, config)
        except ValueError as e:
            self.logger.error(f"Model incompatible with DeepSpeed: {e}")
            raise

        # Build DeepSpeed JSON config from YAML
        try:
            ds_config = build_deepspeed_config(config)
            validate_deepspeed_config(ds_config)
        except Exception as e:
            self.logger.error(f"Failed to build DeepSpeed config: {e}")
            raise

        # Initialize DeepSpeed
        # DeepSpeed will create optimizer and scheduler internally
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )

        zero_stage = deepspeed_cfg.get('zero_stage', 2)
        if rank == 0:
            self.logger.info(
                f"DeepSpeed initialized: ZeRO-{zero_stage}, world_size={world_size}"
            )

            # Log DeepSpeed info
            from .deepspeed import log_deepspeed_info
            log_deepspeed_info(model_engine, rank)

        return model_engine, optimizer, scheduler

    def wrap_distributed(
        self,
        model: nn.Module,
        rank: int,
        world_size: int = 1,
    ) -> nn.Module:
        """
        Wrap model in DDP or DeepSpeed for distributed training.

        Args:
            model: Model to wrap
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            DDP-wrapped model or DeepSpeed engine, or original model if single-GPU
        """
        # Check if DeepSpeed is enabled
        # Support both v2.0 path (distributed.deepspeed) and legacy path (deepspeed)
        deepspeed_config = {}
        if self.context.config:
            distributed_cfg = self.context.config.get('distributed', {})
            deepspeed_config = distributed_cfg.get('deepspeed', {})
            # Fallback to legacy top-level 'deepspeed' key for backward compatibility
            if not deepspeed_config:
                deepspeed_config = self.context.config.get('deepspeed', {})

        # Allow DeepSpeed even with world_size=1 (useful for CPU/NVMe offloading)
        if deepspeed_config.get('enabled', False) and DISTRIBUTED_AVAILABLE:
            # DeepSpeed path - returns engine + optimizer + scheduler
            engine, optimizer, scheduler = self._wrap_with_deepspeed(
                model, self.context.config, rank, world_size
            )

            # Store optimizer/scheduler in context for later retrieval
            self.context.deepspeed_optimizer = optimizer
            self.context.deepspeed_scheduler = scheduler

            return engine
        elif world_size > 1 and DISTRIBUTED_AVAILABLE:
            # DDP path (existing code)
            model = DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False  # Disabled - adds overhead, not needed for this model
            )
            if rank == 0:
                self.logger.info(f"Model wrapped in DDP (world_size={world_size})")

        return model

    def move_to_device(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Move model to target device.

        Note: Does NOT set context.model. The caller (train_pipeline.py) is
        responsible for setting context.model after ALL building steps complete.

        Args:
            model: Model to move
            device: Target device

        Returns:
            Model on target device
        """
        model = model.to(device)
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

        # Handle different checkpoint formats with automatic key remapping
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load with automatic key remapping for backwards compatibility
        load_state_dict_with_remapping(model, state_dict, strict=True)

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
