"""
Phase 8: DataLoader creation and scheduler setup.

Creates training and validation dataloaders, then creates
the learning rate scheduler (requires total_steps).
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class DataPhase(TrainingPhase):
    """
    Phase 8: Create dataloaders and scheduler.

    This phase:
    1. Loads tokenizer
    2. Creates train and validation dataloaders
    3. Calculates total training steps
    4. Creates learning rate scheduler
    5. Synchronizes ranks after dataloader creation
    """

    name = "data"
    description = "Create dataloaders"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Create dataloaders and scheduler.

        Args:
            ctx: Phase context

        Returns:
            Context with dataloaders and scheduler
        """
        from ava.config.training_config import DynamicConfig
        from ava.core.paths import get_tokenizer_path

        data_mgr = ctx.data_mgr
        data_mgr.initialize()

        data_config = ctx.config.get('data', {})
        training_config = ctx.config.get('training', {})

        # Load tokenizer
        tokenizer = self._load_tokenizer(ctx, data_config)
        ctx.tokenizer = tokenizer
        ctx.context.tokenizer = tokenizer

        # Convert config for DataLoaderManager
        config_obj = DynamicConfig(ctx.config) if isinstance(ctx.config, dict) else ctx.config

        # Get num_workers from config
        num_workers = data_config.get('num_workers', 6)

        # Create dataloaders
        train_loader, val_loader = data_mgr.create_dataloaders(
            training_config=config_obj,
            tokenizer=tokenizer,
            config_dict=ctx.config,
            batch_size=ctx.batch_size,
            num_workers=num_workers
        )

        ctx.train_loader = train_loader
        ctx.val_loader = val_loader

        # Synchronize ranks
        if ctx.world_size > 1:
            import torch.distributed as dist
            dist.barrier()

        # Calculate total steps
        try:
            loader_len = len(train_loader)
        except TypeError:
            # Infinite/streaming datasets don't have length
            loader_len = training_config.get('steps_per_epoch', 1000)

        # Validate gradient_accumulation_steps
        grad_accum = ctx.context.gradient_accumulation_steps
        if grad_accum <= 0:
            raise ValueError(
                f"Invalid gradient_accumulation_steps: {grad_accum}. Must be >= 1."
            )

        # Calculate steps per epoch, then multiply by epochs
        steps_per_epoch = (loader_len + grad_accum - 1) // grad_accum
        ctx.total_steps = ctx.num_epochs * steps_per_epoch

        if ctx.is_main_process:
            self.log(ctx, f"DataLoader: {loader_len} batches, {ctx.total_steps} total steps")

        # Create scheduler
        self._create_scheduler(ctx)

        return ctx

    def _load_tokenizer(self, ctx: PhaseContext, data_config: dict):
        """Load tokenizer from config path."""
        from ava.core.paths import get_tokenizer_path

        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast

            tokenizer_path = (
                data_config.get('tokenizer_path') or
                data_config.get('tokenizer_name') or
                str(get_tokenizer_path())
            )

            tokenizer_path_obj = Path(tokenizer_path)

            if tokenizer_path_obj.exists() and tokenizer_path_obj.suffix == '.json':
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

            elif tokenizer_path_obj.is_dir() and (tokenizer_path_obj / 'tokenizer.json').exists():
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_path_obj / 'tokenizer.json')
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            if ctx.is_main_process:
                self.log(ctx, f"Loaded tokenizer: {tokenizer_path} (vocab_size={len(tokenizer)})")

            return tokenizer

        except Exception as e:
            if ctx.is_main_process:
                self.log(
                    ctx,
                    f"Tokenizer load FAILED: {e}. Generation and token-based logging disabled.",
                    "error"
                )
            return None

    def _create_scheduler(self, ctx: PhaseContext) -> None:
        """Create learning rate scheduler."""
        training_config = ctx.config.get('training', {})
        schedule_config = training_config.get('schedule', {})

        warmup_steps = schedule_config.get(
            'warmup_steps',
            training_config.get('warmup_steps', 1000)
        )
        min_lr = schedule_config.get(
            'min_lr',
            training_config.get('min_lr', 0.0)
        )
        scheduler_type = schedule_config.get('scheduler_type', 'cosine')
        num_cycles = schedule_config.get('num_cycles', 1)

        scheduler = ctx.optimizer_mgr.create_scheduler(
            ctx.optimizer,
            warmup_steps,
            ctx.total_steps,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            num_cycles=num_cycles,
        )

        ctx.scheduler = scheduler
        ctx.context.scheduler = scheduler

        if ctx.is_main_process:
            self.log(
                ctx,
                f"Scheduler: {scheduler_type}, warmup={warmup_steps}, "
                f"total={ctx.total_steps}, min_lr={min_lr:.2e}"
            )

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.data_mgr is None:
            errors.append("data_mgr must be registered")
        if ctx.optimizer is None:
            errors.append("optimizer must be created before scheduler")
        return errors
