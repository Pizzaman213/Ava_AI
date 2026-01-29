"""
Phase 2: Configuration loading.

Loads YAML configuration with path resolution and applies
CLI argument overrides.
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class ConfigPhase(TrainingPhase):
    """
    Phase 2: Load and validate configuration.

    This phase:
    1. Loads YAML config with path resolution
    2. Configures Triton kernels if available
    3. Applies CLI argument overrides
    4. Logs optimization status

    Features:
        - YAML path resolution relative to config file
        - Triton kernel configuration
        - Detailed optimization status logging
    """

    name = "config"
    description = "Load configuration"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Load configuration from YAML.

        Args:
            ctx: Phase context with args.config path

        Returns:
            Context with config dict populated
        """
        from ava.config.yaml_loader import load_yaml_with_path_resolution

        # Load YAML config
        config_path = ctx.args.config
        try:
            config = load_yaml_with_path_resolution(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}") from e

        ctx.config = config

        # Configure kernel optimizations
        self._configure_kernels(ctx)

        # Apply CLI overrides
        self._apply_cli_overrides(ctx)

        # Log optimization status
        if ctx.is_main_process:
            self._log_optimization_status(ctx)

        return ctx

    def _configure_kernels(self, ctx: PhaseContext) -> None:
        """Configure Triton kernel optimizations."""
        try:
            from ava.cuda.moe_kernels import (
                KernelConfig, set_kernel_config, TRITON_AVAILABLE
            )

            if not TRITON_AVAILABLE:
                if ctx.is_main_process:
                    self.log(ctx, "Triton not available, using PyTorch fallbacks", "warning")
                return

            # Get kernel config from compute.kernels or kernel_optimization
            compute_config = ctx.config.get('compute', {})
            kernel_opt = (
                ctx.config.get('kernel_optimization', {}) or
                compute_config.get('kernels', {})
            )

            if kernel_opt:
                kernel_config = KernelConfig(
                    use_bitonic_topk=kernel_opt.get('use_fused_softmax_topk', True),
                    use_heap_topk=kernel_opt.get('use_fused_softmax_topk', True),
                    router_block_size=kernel_opt.get('router_block_size', 4),
                    use_fused_softmax_topk=kernel_opt.get('use_fused_softmax_topk', True),
                )
                set_kernel_config(kernel_config)

                if ctx.is_main_process:
                    self.log(ctx, "Triton kernel optimization configured", "debug")

        except ImportError:
            # Triton not available, continue without kernel optimization
            pass

    def _apply_cli_overrides(self, ctx: PhaseContext) -> None:
        """Apply CLI argument overrides to config."""
        args = ctx.args
        config = ctx.config
        training_config = config.get('training', {})
        batching_config = training_config.get('batching', {})
        optimizer_config = training_config.get('optimizer', {})

        # Epochs
        ctx.num_epochs = args.epochs or training_config.get(
            'schedule', {}
        ).get('num_epochs', training_config.get('num_epochs', 3))

        # Batch size (check calibration first)
        calibration_config = config.get('batch_size_calibration', {})
        calibration_enabled = calibration_config.get('enabled', False)

        if args.batch_size is not None and not calibration_enabled:
            ctx.batch_size = args.batch_size
            if ctx.is_main_process:
                self.log(ctx, f"Using CLI batch_size={ctx.batch_size} (calibration disabled)", "warning")
        else:
            ctx.batch_size = (
                batching_config.get('batch_size') or
                training_config.get('batch_size', 8)
            )

        # Learning rate
        if isinstance(optimizer_config, dict):
            opt_lr = optimizer_config.get('learning_rate')
        else:
            opt_lr = None

        ctx.learning_rate = (
            args.learning_rate or
            opt_lr or
            training_config.get('learning_rate') or
            5e-5
        )

        # Validate learning rate
        if not isinstance(ctx.learning_rate, (int, float)):
            raise ValueError(
                f"learning_rate must be numeric, got {type(ctx.learning_rate).__name__}"
            )
        if ctx.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {ctx.learning_rate}")

        # Store in metadata
        ctx.metadata['log_interval'] = self._get_log_interval(ctx)
        ctx.metadata['val_interval'] = getattr(args, 'val_interval', 1)
        ctx.metadata['max_steps'] = getattr(args, 'max_steps', None)

    def _get_log_interval(self, ctx: PhaseContext) -> int:
        """Get log interval from config hierarchy."""
        args = ctx.args
        config = ctx.config
        top_level_logging = config.get('logging', {})
        frequencies = top_level_logging.get('frequencies', {})
        training_logging = config.get('training', {}).get('logging', {})

        return (
            getattr(args, 'log_interval', None) or
            frequencies.get('log_interval') or  # logging.frequencies.log_interval
            top_level_logging.get('log_interval') or
            training_logging.get('logging_steps') or
            config.get('training', {}).get('log_interval', 100)
        )

    def _log_optimization_status(self, ctx: PhaseContext) -> None:
        """Log detailed optimization status."""
        # Import the logging function from train_pipeline
        # This will be refactored later - for now just log basic info
        model_config = ctx.config.get('model', {})
        training_config = ctx.config.get('training', {})

        self.log(ctx, f"Model: hidden_size={model_config.get('hidden_size', 768)}, "
                      f"num_layers={model_config.get('num_layers', 12)}, "
                      f"num_experts={model_config.get('num_experts', 8)}")
        self.log(ctx, f"Training: batch_size={ctx.batch_size}, "
                      f"epochs={ctx.num_epochs}, "
                      f"lr={ctx.learning_rate:.2e}")

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate that config path is provided."""
        errors = []
        if not ctx.args or not hasattr(ctx.args, 'config'):
            errors.append("args.config is required")
        return errors
