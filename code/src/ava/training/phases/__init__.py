"""
Phase-based training orchestration for Ava.

This module provides a phase-based architecture for training pipeline
orchestration. Each phase is a self-contained unit of work that executes
in sequence, building up the training state progressively.

Architecture:
    PhaseExecutor (Orchestrator)
        ├── PhaseContext (Shared State)
        │
        └── Phases (in order):
            ├── p00: DependencyPhase      - Check required packages
            ├── p01: DistributedPhase     - Setup DDP/single GPU
            ├── p02: ConfigPhase          - Load YAML config
            ├── p03: RunManagerPhase      - Setup output directories
            ├── p04: ContextPhase         - Create TrainingContext
            ├── p05: PipelinePhase        - Register components
            ├── p06: ModelPhase           - Build model + calibration + torch.compile
            ├── p07: OptimizerPhase       - Create optimizer + load checkpoint
            ├── p08: DataPhase            - Create dataloaders + scheduler
            ├── p09: MetricsPhase         - Setup WandB, logging
            ├── p10: ComponentsPhase      - Validation, generation, diagnostics
            ├── p10b: WarmupPhase         - Model warmup (optional, primes CUDA kernels)
            ├── p11: TrainingPhase        - Main training loop
            └── p12: FinalizePhase        - Cleanup and summary

Usage:
    from ava.training.phases import PhaseExecutor, PhaseContext

    ctx = PhaseContext(args=args, config={})
    executor = PhaseExecutor(logger)
    ctx = executor.run_all(ctx)
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Type

from .base import PhaseContext, TrainingPhase

# Import all phase modules
from .p00_dependencies import DependencyPhase
from .p01_distributed import DistributedPhase
from .p02_config import ConfigPhase
from .p03_run_manager import RunManagerPhase
from .p04_context import ContextPhase
from .p05_pipeline import PipelinePhase
from .p06_model import ModelPhase
from .p07_optimizer import OptimizerPhase
from .p08_data import DataPhase
from .p09_metrics import MetricsPhase
from .p10_components import ComponentsPhase
from .p10b_warmup import WarmupPhase
from .p11_training import TrainingLoopPhase
from .p12_finalize import FinalizePhase


# Default phase order
DEFAULT_PHASES: List[Type[TrainingPhase]] = [
    DependencyPhase,
    DistributedPhase,
    ConfigPhase,
    RunManagerPhase,
    ContextPhase,
    PipelinePhase,
    ModelPhase,
    OptimizerPhase,
    DataPhase,
    MetricsPhase,
    ComponentsPhase,
    WarmupPhase,  # Phase 10b: Model warmup (optional, controlled by config)
    TrainingLoopPhase,
    FinalizePhase,
]


class PhaseExecutor:
    """
    Orchestrates execution of training phases.

    The PhaseExecutor:
    1. Runs phases in order, passing PhaseContext between them
    2. Logs phase progress with timing
    3. Handles errors with proper cleanup
    4. Supports skipping phases (e.g., --skip-dep-check)

    Attributes:
        logger: Logger for phase execution messages
        phases: List of phase classes to execute
        executed_phases: Phases that have been executed (for cleanup)

    Example:
        executor = PhaseExecutor(logger)
        ctx = executor.run_all(PhaseContext(args=args))
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        phases: Optional[List[Type[TrainingPhase]]] = None,
    ):
        """
        Initialize the phase executor.

        Args:
            logger: Logger for execution messages (creates default if None)
            phases: List of phase classes (uses DEFAULT_PHASES if None)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.phases = phases or DEFAULT_PHASES
        self.executed_phases: List[TrainingPhase] = []
        self._current_phase: Optional[TrainingPhase] = None

    def run_all(self, ctx: PhaseContext) -> PhaseContext:
        """
        Execute all phases in order.

        Args:
            ctx: Initial phase context with args

        Returns:
            Final phase context with all state populated

        Raises:
            RuntimeError: If a phase fails and cannot recover
        """
        import time

        total_start = time.time()

        for i, phase_class in enumerate(self.phases):
            phase = phase_class()
            self._current_phase = phase

            # Check if phase should be skipped
            if self._should_skip_phase(phase, ctx):
                self._log_skip(phase, ctx)
                continue

            # Run validation
            errors = phase.validate(ctx)
            if errors:
                error_msg = f"Phase '{phase.name}' validation failed: {errors}"
                self._log_error(error_msg, ctx)
                raise RuntimeError(error_msg)

            # Execute phase with timing
            self._log_phase_start(i, phase, ctx)
            start_time = time.time()

            try:
                ctx = phase.execute(ctx)
                self.executed_phases.append(phase)
            except Exception as e:
                self._log_phase_error(phase, e, ctx)
                raise

            elapsed = time.time() - start_time
            self._log_phase_complete(phase, elapsed, ctx)

        total_elapsed = time.time() - total_start
        self._log_all_complete(total_elapsed, ctx)

        return ctx

    def _should_skip_phase(self, phase: TrainingPhase, ctx: PhaseContext) -> bool:
        """Check if a phase should be skipped based on args/config."""
        if phase.name == "dependencies":
            return getattr(ctx.args, 'skip_dep_check', False)
        return False

    def _log_phase_start(
        self, index: int, phase: TrainingPhase, ctx: PhaseContext
    ) -> None:
        """Log phase start (rank 0 only)."""
        if ctx.rank != 0:
            return

        desc = phase.description or phase.name.replace('_', ' ').title()
        self.logger.info(f"Phase {index}: {desc}")

    def _log_phase_complete(
        self, phase: TrainingPhase, elapsed: float, ctx: PhaseContext
    ) -> None:
        """Log phase completion (rank 0 only)."""
        if ctx.rank != 0:
            return

        self.logger.debug(f"Phase '{phase.name}' completed in {elapsed:.2f}s")

    def _log_phase_error(
        self, phase: TrainingPhase, error: Exception, ctx: PhaseContext
    ) -> None:
        """Log phase error."""
        self.logger.error(f"Phase '{phase.name}' failed: {error}")
        self.logger.debug(traceback.format_exc())

    def _log_skip(self, phase: TrainingPhase, ctx: PhaseContext) -> None:
        """Log phase skip."""
        if ctx.rank != 0:
            return
        self.logger.debug(f"Skipping phase: {phase.name}")

    def _log_error(self, message: str, ctx: PhaseContext) -> None:
        """Log error message."""
        self.logger.error(message)

    def _log_all_complete(self, elapsed: float, ctx: PhaseContext) -> None:
        """Log completion of all phases."""
        if ctx.rank != 0:
            return
        self.logger.info(f"All phases completed in {elapsed:.1f}s")

    def handle_error(self, ctx: PhaseContext, error: Exception) -> None:
        """
        Handle error during phase execution.

        Calls cleanup on all executed phases in reverse order.

        Args:
            ctx: Current phase context
            error: The exception that occurred
        """
        self.logger.error(f"Training failed: {error}")

        # Notify pipeline if available
        if ctx.pipeline is not None:
            try:
                ctx.pipeline.on_error(error)
            except Exception as e:
                self.logger.warning(f"Pipeline error notification failed: {e}")

        # Cleanup executed phases in reverse order
        for phase in reversed(self.executed_phases):
            try:
                phase.cleanup(ctx)
            except Exception as e:
                self.logger.warning(f"Cleanup failed for '{phase.name}': {e}")

    def finalize(self, ctx: PhaseContext) -> None:
        """
        Finalize after training (success or failure).

        Performs cleanup operations that should happen regardless of
        training outcome.

        Args:
            ctx: Final phase context
        """
        import torch

        # Cleanup pipeline components
        if ctx.pipeline is not None:
            try:
                ctx.pipeline.cleanup_all()
            except Exception as e:
                self.logger.warning(f"Pipeline cleanup error: {e}")

        # Shutdown checkpoint manager
        if ctx.checkpoint_manager is not None:
            try:
                ctx.checkpoint_manager.shutdown()
            except Exception as e:
                self.logger.warning(f"Checkpoint manager shutdown error: {e}")

        # Cleanup CUDA resources
        if torch.cuda.is_available():
            try:
                # Clear async logger
                from ava.logging.metrics.async_logger import shutdown_async_logger
                shutdown_async_logger()
            except Exception:
                pass

            try:
                # Clear buffer pool
                from ava.cuda.streams import clear_buffer_pool
                clear_buffer_pool()
            except Exception:
                pass

            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Cleanup distributed
        if ctx.world_size > 1:
            try:
                from ava.training import cleanup_distributed
                cleanup_distributed(ctx.rank, ctx.world_size)
            except Exception as e:
                self.logger.warning(f"Distributed cleanup error: {e}")


__all__ = [
    'PhaseContext',
    'TrainingPhase',
    'PhaseExecutor',
    'DEFAULT_PHASES',
    # Individual phases
    'DependencyPhase',
    'DistributedPhase',
    'ConfigPhase',
    'RunManagerPhase',
    'ContextPhase',
    'PipelinePhase',
    'ModelPhase',
    'OptimizerPhase',
    'DataPhase',
    'MetricsPhase',
    'ComponentsPhase',
    'WarmupPhase',
    'TrainingLoopPhase',
    'FinalizePhase',
]
