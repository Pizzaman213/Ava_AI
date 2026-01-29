"""
Phase 6: Model building.

Builds the model with full optimization pipeline and optional
batch size calibration.
"""

import logging
from typing import List

import torch

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class ModelPhase(TrainingPhase):
    """
    Phase 6: Build model with optimizations.

    This phase:
    1. Builds model (EnhancedMoEModel)
    2. Applies quantization if enabled
    3. Applies pre-device optimizations (torch.compile on CPU)
    4. Moves to GPU
    5. Applies post-device optimizations (FP8, gradient checkpointing)
    6. Wraps in DDP for multi-GPU
    7. Runs batch size calibration if enabled
    """

    name = "model"
    description = "Build and optimize model"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Build model with full optimization pipeline.

        Args:
            ctx: Phase context

        Returns:
            Context with model built and optimized
        """
        model_builder = ctx.model_builder
        model_builder.initialize()

        # Configure fail-fast behavior
        fail_fast = ctx.config.get('model_building', {}).get('fail_on_optimization_error', True)
        model_builder.set_fail_on_optimization_error(fail_fast)

        # Build model
        model = model_builder.build_model(ctx.config, ctx.device)

        # Apply quantization if enabled
        quant_config = ctx.config.get('quantization', {})
        if quant_config.get('enabled', False):
            model = model_builder.apply_quantization(model, quant_config)

        # Apply pre-device optimizations
        model = model_builder.apply_pre_device_optimizations(model, ctx.config)

        # Move to device
        model = model_builder.move_to_device(model, ctx.device)

        # Apply post-device optimizations
        model = model_builder.apply_post_device_optimizations(model, ctx.config)

        # Wrap in DDP
        model = model_builder.wrap_distributed(model, ctx.rank, ctx.world_size)

        # Apply torch.compile after DDP wrapping for maximum throughput
        model = self._apply_torch_compile(ctx, model)

        ctx.model = model
        ctx.context.model = model

        # Run batch size calibration
        self._run_calibration(ctx, model_builder)

        if ctx.is_main_process:
            param_count = sum(p.numel() for p in model.parameters())
            self.log(ctx, f"Model built: {param_count / 1e6:.1f}M parameters")

        return ctx

    def _apply_torch_compile(self, ctx: PhaseContext, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply torch.compile for improved throughput (10-30% speedup).

        torch.compile is applied AFTER DDP wrapping to ensure compatibility
        with distributed training and gradient synchronization.

        Config options (under compute.performance):
            enable_torch_compile: bool (default: False)
            torch_compile_mode: str (default: 'reduce-overhead')
                - 'default': Balanced compilation
                - 'reduce-overhead': Minimizes overhead for variable workloads
                - 'max-autotune': Maximum optimization, longer compile time
            torch_compile_fullgraph: bool (default: False)
            torch_compile_dynamic: bool (default: True)

        Args:
            ctx: Phase context
            model: Model to compile

        Returns:
            Compiled model (or original if compilation disabled/fails)
        """
        # BUG FIX: Check if model is already compiled to prevent double compilation
        # torch.compile sets _orig_mod attribute on compiled models
        # For DDP-wrapped models, check both wrapper and inner module
        inner_model = model.module if hasattr(model, 'module') else model
        if hasattr(inner_model, '_orig_mod'):
            if ctx.is_main_process:
                self.log(ctx, "Model already compiled, skipping duplicate torch.compile")
            return model

        # Check config - support both old and new config paths
        performance_config = ctx.config.get('compute', {}).get('performance', {})
        if not performance_config:
            # Fallback to old config path
            performance_config = ctx.config.get('performance', {})

        enable_compile = performance_config.get('enable_torch_compile', False)

        if not enable_compile:
            return model

        compile_mode = performance_config.get('torch_compile_mode', 'reduce-overhead')
        fullgraph = performance_config.get('torch_compile_fullgraph', False)
        dynamic = performance_config.get('torch_compile_dynamic', True)

        if ctx.is_main_process:
            self.log(ctx, f"Applying torch.compile (mode={compile_mode}, fullgraph={fullgraph}, dynamic={dynamic})")

        try:
            import time
            t0 = time.perf_counter()

            compiled_model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )

            compile_time = time.perf_counter() - t0
            if ctx.is_main_process:
                self.log(ctx, f"torch.compile completed in {compile_time:.1f}s")

            return compiled_model

        except Exception as e:
            if ctx.is_main_process:
                self.log(ctx, f"torch.compile failed: {e}", "warning")
                self.log(ctx, "Continuing without compilation", "warning")
            return model

    def _run_calibration(self, ctx: PhaseContext, model_builder) -> None:
        """
        Run batch size calibration if enabled.

        Updates ctx.batch_size with calibrated value.
        """
        calibration_config = ctx.config.get('batch_size_calibration', {})
        if not calibration_config.get('enabled', False):
            return

        # Check DeepSpeed compatibility (check both v2.0 and legacy paths)
        distributed_cfg = ctx.config.get('distributed', {})
        deepspeed_enabled = (
            distributed_cfg.get('deepspeed', {}).get('enabled', False) or
            ctx.config.get('deepspeed', {}).get('enabled', False)
        )
        if deepspeed_enabled:
            if ctx.is_main_process:
                self.log(ctx, "Batch size calibration disabled (incompatible with DeepSpeed)", "warning")
            return

        # Synchronize ranks before calibration
        if ctx.world_size > 1:
            import torch.distributed as dist
            dist.barrier()

        try:
            from ava.training.calibration import BatchSizeCalibrator

            calibrator = BatchSizeCalibrator(
                config=ctx.config,
                model=ctx.model,
                model_builder=model_builder,
                device=ctx.device,
                rank=ctx.rank,
                world_size=ctx.world_size,
                logger=ctx.logger,
            )

            optimal_batch_size = calibrator.calibrate()

            if optimal_batch_size is not None:
                ctx.batch_size = optimal_batch_size

                # Update config
                if 'training' not in ctx.config:
                    ctx.config['training'] = {}
                if 'batching' not in ctx.config['training']:
                    ctx.config['training']['batching'] = {}
                ctx.config['training']['batching']['batch_size'] = optimal_batch_size
                ctx.config['training']['batch_size'] = optimal_batch_size

                # Store controller for OOM recovery
                ctx.context.batch_controller = calibrator.controller

                if ctx.is_main_process:
                    self.log(ctx, f"Batch size calibrated: {optimal_batch_size}")

        except ImportError as e:
            self.log(ctx, f"Could not import BatchSizeCalibrator: {e}", "warning")
        except Exception as e:
            fail_on_error = calibration_config.get('fail_on_error', False)
            if fail_on_error:
                raise RuntimeError(f"Batch size calibration failed: {e}") from e

            self.log(ctx, f"Batch size calibration failed: {e}", "warning")
            self.log(ctx, "Training will continue with configured batch_size", "warning")

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.model_builder is None:
            errors.append("model_builder must be registered before model building")
        if ctx.device is None:
            errors.append("device must be set before model building")
        return errors
