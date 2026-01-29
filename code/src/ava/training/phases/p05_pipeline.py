"""
Phase 5: TrainingPipeline creation and component registration.

Creates the TrainingPipeline and registers all component managers.
"""

import logging
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class PipelinePhase(TrainingPhase):
    """
    Phase 5: Create TrainingPipeline and register components.

    The TrainingPipeline is the component orchestrator that:
    - Registers all manager components
    - Calls lifecycle hooks (on_epoch_start, on_epoch_end, etc.)
    - Handles component errors and cleanup

    Components registered:
    - model: ModelBuilder
    - optimizer: OptimizerManager
    - data: DataLoaderManager
    - training: TrainingLoopManager
    - validation: ValidationManager
    - generation: GenerationManager
    - metrics: MetricsManager
    """

    name = "pipeline"
    description = "Register training components"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Create TrainingPipeline and register components.

        Args:
            ctx: Phase context

        Returns:
            Context with pipeline and component managers
        """
        from ava.training import (
            TrainingPipeline,
            ModelBuilder,
            OptimizerManager,
            DataLoaderManager,
            TrainingLoopManager,
            ValidationManager,
            GenerationManager,
            MetricsManager,
        )

        # Create pipeline
        pipeline = TrainingPipeline(ctx.context)
        ctx.pipeline = pipeline

        # Create and register components
        model_builder = ModelBuilder(ctx.context)
        optimizer_mgr = OptimizerManager(ctx.context)
        data_mgr = DataLoaderManager(ctx.context)
        training_mgr = TrainingLoopManager(ctx.context)
        validation_mgr = ValidationManager(ctx.context)
        generation_mgr = GenerationManager(ctx.context)
        metrics_mgr = MetricsManager(ctx.context)

        # Register with pipeline
        pipeline.register('model', model_builder)
        pipeline.register('optimizer', optimizer_mgr)
        pipeline.register('data', data_mgr)
        pipeline.register('training', training_mgr)
        pipeline.register('validation', validation_mgr)
        pipeline.register('generation', generation_mgr)
        pipeline.register('metrics', metrics_mgr)

        # Store references in phase context
        ctx.model_builder = model_builder
        ctx.optimizer_mgr = optimizer_mgr
        ctx.data_mgr = data_mgr
        ctx.training_mgr = training_mgr
        ctx.validation_mgr = validation_mgr
        ctx.generation_mgr = generation_mgr
        ctx.metrics_mgr = metrics_mgr

        if ctx.is_main_process:
            self.log(ctx, f"Registered {len(pipeline)} training components")

        return ctx

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.context is None:
            errors.append("TrainingContext must be created before pipeline registration")
        return errors
