"""
Phase 7: Optimizer creation and checkpoint loading.

Creates the optimizer and loads checkpoint if resuming.
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class OptimizerPhase(TrainingPhase):
    """
    Phase 7: Create optimizer and load checkpoint.

    This phase:
    1. Creates optimizer (AdamW, SGD, etc.)
    2. Loads checkpoint if resuming
    3. Restores optimizer state from checkpoint
    """

    name = "optimizer"
    description = "Create optimizer"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Create optimizer and load checkpoint.

        Args:
            ctx: Phase context

        Returns:
            Context with optimizer created
        """
        optimizer_mgr = ctx.optimizer_mgr
        optimizer_mgr.initialize()

        # Get weight decay from config
        training_config = ctx.config.get('training', {})
        weight_decay = training_config.get('weight_decay', 0.01)

        # Create optimizer
        optimizer = optimizer_mgr.create_optimizer(
            ctx.model,
            ctx.config,
            ctx.learning_rate,
            weight_decay=weight_decay
        )

        ctx.optimizer = optimizer
        ctx.context.optimizer = optimizer

        # Load checkpoint if resuming
        resume_path = getattr(ctx.args, 'resume', None)
        if resume_path:
            resume_path = Path(resume_path)
            if resume_path.exists():
                self._load_checkpoint(ctx, resume_path)

        if ctx.is_main_process:
            self.log(ctx, f"Optimizer created: lr={ctx.learning_rate:.2e}, wd={weight_decay}")

        return ctx

    def _load_checkpoint(self, ctx: PhaseContext, resume_path: Path) -> None:
        """Load checkpoint and restore state."""
        if ctx.is_main_process:
            self.log(ctx, f"Loading checkpoint: {resume_path}")

        resume_epoch, resume_step = ctx.checkpoint_manager.load(
            model=ctx.model,
            optimizer=ctx.optimizer,
            checkpoint_path=resume_path
        )

        ctx.resume_epoch = resume_epoch
        ctx.resume_step = resume_step

        if ctx.is_main_process:
            self.log(ctx, f"Resumed from epoch {resume_epoch}, step {resume_step}")

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.model is None:
            errors.append("model must be built before optimizer creation")
        if ctx.optimizer_mgr is None:
            errors.append("optimizer_mgr must be registered")
        return errors
