"""
Phase 9: Metrics setup.

Initializes metrics tracking, WandB integration, and logging.
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class MetricsPhase(TrainingPhase):
    """
    Phase 9: Setup metrics and logging.

    This phase:
    1. Initializes MetricsManager
    2. Sets up WandB integration if enabled
    3. Configures logging directories
    """

    name = "metrics"
    description = "Setup metrics tracking"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Setup metrics tracking.

        Args:
            ctx: Phase context

        Returns:
            Context with metrics configured
        """
        metrics_mgr = ctx.metrics_mgr
        metrics_mgr.initialize()

        # Get logging config
        logging_config = ctx.config.get('logging', {})
        wandb_config = logging_config.get('wandb', {})
        wandb_enabled = wandb_config.get('enabled', False)

        # Get log directory
        log_dir = ctx.metadata.get('log_dir')
        if log_dir is None:
            log_dir = ctx.run_manager.run_dir / 'logs' if ctx.run_manager else Path('./logs')

        # Set wandb directory
        wandb_dir = ctx.run_manager.run_dir / 'wandb' if ctx.run_manager else None

        if ctx.is_main_process:
            self.log(
                ctx,
                f"WandB: enabled={wandb_enabled}, project={wandb_config.get('project', 'N/A')}"
            )

        # Check if all logging is disabled
        if logging_config.get('disabled', False):
            if ctx.is_main_process:
                self.log(ctx, "All logging DISABLED via config for maximum training speed")

        # Setup metrics
        metrics_mgr.setup(
            log_dir=log_dir,
            wandb_config=wandb_config if wandb_enabled else None,
            use_wandb=wandb_enabled,
            wandb_dir=wandb_dir,
            logging_config=logging_config,
        )

        return ctx

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.metrics_mgr is None:
            errors.append("metrics_mgr must be registered")
        return errors
