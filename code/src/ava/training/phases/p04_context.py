"""
Phase 4: TrainingContext creation.

Creates the shared state hub used by all training components.
"""

import logging
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class ContextPhase(TrainingPhase):
    """
    Phase 4: Create TrainingContext.

    The TrainingContext is the central shared state hub that all
    training component managers access. It holds:
    - Model and optimizer references
    - Device and distributed training info
    - Config and RunManager
    - Training state (epoch, step, loss)

    This phase creates the context and updates it from config.
    """

    name = "context"
    description = "Create training context"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Create TrainingContext.

        Args:
            ctx: Phase context

        Returns:
            Context with TrainingContext created
        """
        from ava.training import TrainingContext

        # Create TrainingContext
        training_context = TrainingContext(
            model=None,  # Will be set by ModelPhase
            device=ctx.device,
            config=ctx.config,
            run_manager=ctx.run_manager,
            rank=ctx.rank,
            world_size=ctx.world_size,
            is_main_process=(ctx.rank == 0),
        )

        # Update from config
        training_context.update_from_config(ctx.config)

        ctx.context = training_context

        if ctx.is_main_process:
            self.log(ctx, f"TrainingContext created (rank={ctx.rank}, world_size={ctx.world_size})")

        return ctx

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.device is None:
            errors.append("device must be set before creating TrainingContext")
        if not ctx.config:
            errors.append("config must be loaded before creating TrainingContext")
        return errors
