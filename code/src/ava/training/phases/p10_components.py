"""
Phase 10: Additional component setup.

Sets up validation, generation, diagnostics, and training loop managers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class ComponentsPhase(TrainingPhase):
    """
    Phase 10: Initialize additional training components.

    This phase:
    1. Sets up ValidationManager with quality scoring
    2. Sets up GenerationManager for sample generation
    3. Sets up DiagnosticsManager if enabled
    4. Sets up EpisodicMemoryManager if enabled
    5. Configures TrainingLoopManager
    """

    name = "components"
    description = "Initialize training components"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Initialize remaining components.

        Args:
            ctx: Phase context

        Returns:
            Context with all components configured
        """
        from ava.config.training_config import ModelSelectionConfig, DiagnosticsConfig
        from ava.training import TrainingLoopConfig

        training_config = ctx.config.get('training', {})
        log_dir = ctx.metadata.get('log_dir')

        # Setup validation manager
        self._setup_validation(ctx)

        # Setup generation manager
        self._setup_generation(ctx, training_config, log_dir)

        # Setup diagnostics manager
        self._setup_diagnostics(ctx)

        # Setup episodic memory manager
        self._setup_episodic_memory(ctx)

        # Configure training loop manager
        self._setup_training_loop(ctx, training_config, log_dir)

        return ctx

    def _setup_validation(self, ctx: PhaseContext) -> None:
        """Setup validation manager with quality scoring."""
        from ava.config.training_config import ModelSelectionConfig

        validation_mgr = ctx.validation_mgr
        validation_mgr.initialize()

        model_selection_config = ctx.config.get('model_selection', {})
        if model_selection_config.get('enabled', True):
            ms_config = ModelSelectionConfig(
                enabled=model_selection_config.get('enabled', True),
                val_loss_weight=model_selection_config.get('val_loss_weight', 0.5),
                coherence_score_weight=model_selection_config.get('coherence_score_weight', 0.3),
                perplexity_weight=model_selection_config.get('perplexity_weight', 0.2),
                perplexity_cap=model_selection_config.get('perplexity_cap', 100.0),
                val_loss_cap=model_selection_config.get('val_loss_cap', 10.0),
            )
            validation_mgr.setup_quality_scoring(ms_config)

            if ctx.is_main_process:
                self.log(
                    ctx,
                    f"Quality scoring: val_loss={ms_config.val_loss_weight:.0%}, "
                    f"coherence={ms_config.coherence_score_weight:.0%}",
                    "debug"
                )

    def _setup_generation(self, ctx: PhaseContext, training_config: dict, log_dir) -> None:
        """Setup generation manager."""
        generation_mgr = ctx.generation_mgr
        generation_mgr.initialize()
        generation_mgr.set_log_dir(log_dir)

        generation_config = training_config.get('generation', {})

        # Disable generation if no tokenizer
        if generation_config and ctx.tokenizer is None:
            if ctx.is_main_process:
                self.log(
                    ctx,
                    f"Generation disabled: no tokenizer available. "
                    f"Ignoring settings: {list(generation_config.keys())}",
                    "warning"
                )
            generation_config = {}

        if generation_config:
            generation_mgr.set_generation_config(generation_config)

        ctx.metadata['generation_config'] = generation_config

    def _setup_diagnostics(self, ctx: PhaseContext) -> None:
        """Setup diagnostics manager if enabled."""
        from ava.logging.diagnostics.training import DiagnosticsManager
        from ava.config.training_config import DiagnosticsConfig

        diagnostics_config = ctx.config.get('diagnostics', {})
        if not diagnostics_config.get('enabled', False):
            return

        diag_config = DiagnosticsConfig(
            enabled=True,
            enable_per_layer_gradients=diagnostics_config.get('enable_per_layer_gradients', False),
            per_layer_log_freq=diagnostics_config.get('per_layer_log_freq', 500),
            enable_routing_diagnostics=diagnostics_config.get('enable_routing_diagnostics', False),
            routing_log_freq=diagnostics_config.get('routing_log_freq', 100),
            enable_memory_breakdown=diagnostics_config.get('enable_memory_breakdown', False),
            memory_log_freq=diagnostics_config.get('memory_log_freq', 500),
            enable_timing_profiling=diagnostics_config.get('enable_timing_profiling', False),
            timing_log_freq=diagnostics_config.get('timing_log_freq', 100),
        )

        diagnostics_mgr = DiagnosticsManager(ctx.context)
        diagnostics_mgr.initialize()
        diagnostics_mgr.configure(diag_config)

        ctx.diagnostics_mgr = diagnostics_mgr

        if ctx.is_main_process:
            self.log(ctx, "Diagnostics manager initialized", "debug")

    def _setup_episodic_memory(self, ctx: PhaseContext) -> None:
        """Setup episodic memory manager if enabled."""
        from ava.training.episodic_memory import EpisodicMemoryManager

        episodic_config = ctx.config.get('experimental', {}).get('episodic_memory', {})
        if not episodic_config:
            episodic_config = ctx.config.get('episodic_memory', {})

        if not (episodic_config.get('use_episodic_memory', False) or
                episodic_config.get('enabled', False)):
            return

        # Create config object using dataclass
        @dataclass
        class EpisodicMemoryConfig:
            use_episodic_memory: bool = True
            memory_capacity: int = 10000
            memory_replay_ratio: float = 0.2
            buffer_warmup_steps: int = 100
            memory_selection_strategy: str = 'importance'
            priority_exponent: float = 0.6
            importance_weight_exponent: float = 0.4
            store_aux_info: bool = False
            silent_mode: bool = False

        mem_config = EpisodicMemoryConfig(
            use_episodic_memory=True,
            memory_capacity=episodic_config.get('memory_capacity', 10000),
            memory_replay_ratio=episodic_config.get('memory_replay_ratio', 0.2),
            buffer_warmup_steps=episodic_config.get('buffer_warmup_steps', 100),
            memory_selection_strategy=episodic_config.get('memory_selection_strategy', 'importance'),
            priority_exponent=episodic_config.get('priority_exponent', 0.6),
            importance_weight_exponent=episodic_config.get('importance_weight_exponent', 0.4),
            store_aux_info=episodic_config.get('store_aux_info', False),
            silent_mode=episodic_config.get('silent_mode', False),
        )

        episodic_memory_mgr = EpisodicMemoryManager(mem_config, device=ctx.device)
        ctx.episodic_memory_mgr = episodic_memory_mgr

        if ctx.is_main_process:
            self.log(
                ctx,
                f"Episodic memory: capacity={mem_config.memory_capacity}, "
                f"replay_ratio={mem_config.memory_replay_ratio}",
                "debug"
            )

    def _setup_training_loop(self, ctx: PhaseContext, training_config: dict, log_dir) -> None:
        """Configure training loop manager."""
        from ava.training import TrainingLoopConfig

        training_mgr = ctx.training_mgr
        training_mgr.initialize()

        # Set component managers
        training_mgr.set_components(
            metrics_manager=ctx.metrics_mgr,
            generation_manager=ctx.generation_mgr,
            checkpoint_manager=ctx.checkpoint_manager,
            diagnostics_manager=ctx.diagnostics_mgr,
            episodic_memory_manager=ctx.episodic_memory_mgr,
            validation_manager=ctx.validation_mgr,
            val_loader=ctx.val_loader,
        )

        # Setup gradient sync for multi-GPU
        # Check both v2.0 path (distributed.deepspeed) and legacy path (deepspeed)
        distributed_cfg = ctx.config.get('distributed', {})
        is_deepspeed = (
            distributed_cfg.get('deepspeed', {}).get('enabled', False) or
            ctx.config.get('deepspeed', {}).get('enabled', False)
        )
        training_mgr.setup_gradient_sync(ctx.model, is_deepspeed=is_deepspeed)

        # Set resume info
        if ctx.resume_step > 0:
            ctx.context.metadata['resume_step'] = ctx.resume_step
            ctx.context.metadata['resume_epoch'] = ctx.resume_epoch

        # Create loop config
        loop_config = self._create_loop_config(ctx, training_config, log_dir)
        ctx.metadata['loop_config'] = loop_config

        # Setup profiler if enabled
        if loop_config.enable_profiling and ctx.is_main_process:
            training_mgr.setup_profiler(loop_config)

    def _create_loop_config(self, ctx: PhaseContext, training_config: dict, log_dir) -> 'TrainingLoopConfig':
        """Create training loop configuration."""
        from ava.training import TrainingLoopConfig

        args = ctx.args
        logging_config = ctx.config.get('logging', {})
        generation_config = ctx.metadata.get('generation_config', {})

        # Logging options
        log_mode = logging_config.get('log_mode') or training_config.get('log_mode', 'tqdm')
        frequencies = logging_config.get('frequencies', {})
        verbose_log_interval = (
            frequencies.get('verbose_log_interval') or
            logging_config.get('verbose_log_interval') or
            training_config.get('verbose_log_interval', 500)
        )
        tqdm_update_interval = logging_config.get('tqdm_update_interval', 10)

        # CUDA Graphs config
        compute_config = ctx.config.get('compute', {})
        cuda_config = compute_config.get('cuda', {})
        cuda_graphs_config = cuda_config.get('graphs', {})
        use_cuda_graphs = cuda_graphs_config.get('enabled', False)
        cuda_graph_warmup_steps = cuda_graphs_config.get('warmup_steps', 10)

        # Profile directory
        profile_dir = getattr(args, 'profile_dir', None)
        if profile_dir is None or profile_dir == './profiles':
            profile_dir = str(ctx.run_manager.run_dir / 'profiles') if ctx.run_manager else './profiles'

        # Overlapped accumulation config
        batching_config = training_config.get('batching', {})
        use_overlapped_accum = batching_config.get('use_overlapped_accumulation', True)

        # MoE routing metrics config
        logging_cfg = training_config.get('logging', {})
        routing_metrics_freq = logging_cfg.get('routing_metrics_freq', 0)

        return TrainingLoopConfig(
            gradient_accumulation_steps=ctx.context.gradient_accumulation_steps,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            use_amp=ctx.context.use_amp,
            amp_dtype=ctx.context.amp_dtype,
            log_interval=ctx.metadata.get('log_interval', 1),  # Default to every step
            generate_every_n_steps=generation_config.get('generate_every_n_steps', 500),
            save_steps=logging_cfg.get('save_steps', 500),
            eval_steps=logging_cfg.get('eval_steps', 1000),
            max_val_batches=training_config.get('validation', {}).get('max_batches', 50),
            max_steps=ctx.metadata.get('max_steps'),
            # Profiling
            enable_profiling=getattr(args, 'enable_profiling', False),
            profile_start_step=getattr(args, 'profile_start_step', 0),
            profile_end_step=getattr(args, 'profile_end_step', 999999),
            profile_dir=profile_dir,
            # Logging
            log_mode=log_mode,
            verbose_log_interval=verbose_log_interval,
            # CUDA Graphs
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            # Overlapped accumulation
            use_overlapped_accumulation=use_overlapped_accum,
            # MoE routing metrics
            routing_metrics_freq=routing_metrics_freq,
        )

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.validation_mgr is None:
            errors.append("validation_mgr must be registered")
        if ctx.generation_mgr is None:
            errors.append("generation_mgr must be registered")
        if ctx.training_mgr is None:
            errors.append("training_mgr must be registered")
        return errors
