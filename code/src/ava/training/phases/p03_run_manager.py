"""
Phase 3: RunManager setup.

Creates output directory structure and initializes logging.
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class RunManagerPhase(TrainingPhase):
    """
    Phase 3: Setup RunManager and output directories.

    This phase:
    1. Creates RunManager for output organization
    2. Sets up directory structure (checkpoints, logs, configs)
    3. Saves configuration copies
    4. Initializes logging
    5. Creates CheckpointManager

    Output structure:
        {output_dir}/pretraining/{run_name}_YYYYMMDD_HHMMSS/
        ├── checkpoints/
        ├── logs/
        ├── wandb/
        ├── full_config.yaml
        ├── model_config.yaml
        └── training_config.yaml
    """

    name = "run_manager"
    description = "Setup output directories"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Create RunManager and setup directories.

        Args:
            ctx: Phase context with config

        Returns:
            Context with run_manager and checkpoint_manager
        """
        from ava.training import RunManager
        from ava.core.checkpoint import CheckpointManager
        from ava.core.logging import configure_root_logger, ColoredFormatter

        # Configure root logger
        configure_root_logger(level=logging.WARNING)

        # Get output directory
        output_config = ctx.config.get('output', {})
        args = ctx.args
        output_dir = output_config.get(
            'output_dir',
            getattr(args, 'save_dir', 'outputs')
        )

        # Create RunManager
        run_manager = RunManager(
            base_output_dir=str(output_dir),
            run_name=ctx.config.get('experiment_name', 'ava_training'),
            description=f"Training with config: {args.config}"
        )
        ctx.run_manager = run_manager

        # Save configurations (rank 0 only)
        if ctx.is_main_process:
            run_manager.save_config(ctx.config, 'full')

            model_config = ctx.config.get('model', {})
            if model_config:
                run_manager.save_config(model_config, 'model')

            training_config = ctx.config.get('training', {})
            if training_config:
                run_manager.save_config(training_config, 'training')

            data_config = ctx.config.get('data', {})
            if data_config:
                run_manager.save_config(data_config, 'data')

            # Save CLI args
            run_manager.save_args(args)

        # Setup logging
        log_dir = (
            run_manager.run_dir / 'logs'
            if hasattr(run_manager, 'run_dir')
            else Path(getattr(args, 'log_dir', './logs'))
        )
        ctx.logger = self._setup_logging(log_dir, ctx.rank)

        # Create CheckpointManager
        checkpoint_dir = (
            run_manager.run_dir / 'checkpoints'
            if hasattr(run_manager, 'run_dir')
            else Path(output_dir) / 'checkpoints'
        )
        training_config = ctx.config.get('training', {})
        ctx.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_dir,
            max_keep=training_config.get('max_checkpoints', 3),
            config=ctx.config,
            async_save=True
        )

        # Check for resume checkpoint
        self._check_resume(ctx)

        # Store log directory in metadata
        ctx.metadata['log_dir'] = log_dir

        if ctx.is_main_process:
            self.log(ctx, f"Output directory: {run_manager.run_dir}")

        return ctx

    def _setup_logging(self, log_dir: Path, rank: int) -> logging.Logger:
        """Setup logging for training."""
        from ava.core.logging import ColoredFormatter

        train_logger = logging.getLogger('train_pipeline')
        train_logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        train_logger.handlers.clear()

        if rank == 0:
            # Console handler with colors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(ColoredFormatter(show_level=False, show_icons=True))
            train_logger.addHandler(console_handler)

            # File handler (plain format)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / 'training.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            )
            train_logger.addHandler(file_handler)

        train_logger.propagate = False
        return train_logger

    def _check_resume(self, ctx: PhaseContext) -> None:
        """Check and validate resume checkpoint."""
        args = ctx.args
        resume_path = getattr(args, 'resume', None)

        if not resume_path:
            return

        resume_path = Path(resume_path)
        if resume_path.exists():
            if ctx.is_main_process:
                self.log(ctx, f"Will resume from checkpoint: {resume_path}")
        else:
            # Check if resume is required
            require_resume = (
                getattr(args, 'require_resume', False) or
                ctx.config.get('training', {}).get('require_resume', False)
            )

            if require_resume:
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path} "
                    f"(--require-resume set, not starting fresh)"
                )

            if ctx.is_main_process:
                self.log(ctx, f"Resume checkpoint not found: {resume_path}, starting fresh", "warning")
            ctx.args.resume = None

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate that config is loaded."""
        errors = []
        if not ctx.config:
            errors.append("config must be loaded before RunManager setup")
        return errors
