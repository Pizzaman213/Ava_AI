"""
Phase 12: Finalization.

Handles cleanup, metrics summary, and final reporting.
"""

import logging
from pathlib import Path
from typing import List

import torch

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class FinalizePhase(TrainingPhase):
    """
    Phase 12: Finalize training.

    This phase:
    1. Logs final metrics
    2. Saves metrics summary
    3. Logs generation history
    4. Stops profiler
    5. Marks run as complete
    6. Cleans up resources
    """

    name = "finalize"
    description = "Training complete"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Finalize training.

        Args:
            ctx: Phase context

        Returns:
            Context unchanged
        """
        best_val_loss = ctx.metadata.get('best_val_loss', float('inf'))
        log_dir = ctx.metadata.get('log_dir')

        if ctx.is_main_process:
            self.log(ctx, f"Best validation loss: {best_val_loss:.4f}")

            # Save final metrics
            if ctx.metrics_mgr and log_dir:
                try:
                    ctx.metrics_mgr.save_summary(log_dir / 'metrics_summary.json')
                except Exception as e:
                    self.log(ctx, f"Failed to save metrics summary: {e}", "warning")

            # Log final generations table
            if ctx.generation_mgr:
                try:
                    generations = ctx.generation_mgr.get_generation_history()
                    if generations:
                        ctx.metrics_mgr.log_generation_table(generations)
                except Exception as e:
                    self.log(ctx, f"Failed to log generation history: {e}", "warning")

            # Stop profiler
            if ctx.training_mgr:
                try:
                    profile_dir = ctx.training_mgr.stop_profiler()
                    if profile_dir:
                        self.log(ctx, f"Profile data saved to: {profile_dir}")
                except Exception as e:
                    self.log(ctx, f"Failed to stop profiler: {e}", "warning")

        # Mark run as complete
        if ctx.run_manager:
            ctx.run_manager.finish_run(status='completed')

        return ctx

    def cleanup(self, ctx: PhaseContext) -> None:
        """
        Cleanup resources.

        NOTE: Full cleanup is handled by PhaseExecutor.finalize() to avoid
        double cleanup. This method is intentionally minimal.
        """
        # Cleanup is centralized in PhaseExecutor.finalize() to prevent
        # double cleanup issues (e.g., "already destroyed" errors).
        # See PhaseExecutor.finalize() for the actual cleanup sequence.
        pass

    def validate(self, ctx: PhaseContext) -> List[str]:
        """No validation needed for finalization."""
        return []
