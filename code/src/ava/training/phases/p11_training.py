"""
Phase 11: Training loop execution.

Runs the main training loop with validation and checkpointing.
"""

import logging
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class TrainingLoopPhase(TrainingPhase):
    """
    Phase 11: Execute the training loop.

    This phase:
    1. Iterates through epochs
    2. Calls train_epoch() for each epoch
    3. Runs validation periodically
    4. Saves checkpoints and best models
    5. Handles max_steps termination
    """

    name = "training_loop"
    description = "Training loop"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Run the training loop.

        Args:
            ctx: Phase context

        Returns:
            Context with training complete
        """
        training_config = ctx.config.get('training', {})
        model_config = ctx.config.get('model', {})
        loop_config = ctx.metadata.get('loop_config')
        generation_config = ctx.metadata.get('generation_config', {})

        # Define coherence config at training loop scope
        coherence_config = training_config.get('coherence', ctx.config.get('coherence', {}))

        val_interval = ctx.metadata.get('val_interval', 1)
        best_val_loss = float('inf')

        # Warn if validation dataloader is missing
        if ctx.val_loader is None and ctx.is_main_process:
            self.log(
                ctx,
                "Validation dataloader is None - validation will be skipped. "
                "Check data.val_split_ratio in config if this is unexpected.",
                "warning"
            )

        # Training loop
        for epoch in range(ctx.resume_epoch, ctx.num_epochs):
            # Notify components of epoch start
            ctx.pipeline.on_epoch_start(epoch)

            # Train one epoch
            train_loss = ctx.training_mgr.train_epoch(
                model=ctx.model,
                train_loader=ctx.train_loader,
                optimizer=ctx.optimizer,
                scheduler=ctx.scheduler,
                epoch=epoch,
                config=loop_config,
                vocab_size=model_config.get('vocab_size', 50680),
                tokenizer=ctx.tokenizer,
                generation_config=generation_config,
                coherence_config=coherence_config,
            )

            if ctx.is_main_process:
                self.log(ctx, f"Epoch {epoch + 1}/{ctx.num_epochs}: train_loss={train_loss:.4f}")

            # Check max_steps
            if ctx.training_mgr.reached_max_steps(loop_config):
                if ctx.is_main_process:
                    max_steps_str = loop_config.max_steps if loop_config.max_steps is not None else 'limit'
                    self.log(ctx, f"Reached max_steps ({max_steps_str}), stopping")
                break

            # Validation
            best_val_loss = self._run_validation(
                ctx, epoch, val_interval, train_loss, coherence_config, best_val_loss
            )

            # Notify components of epoch end
            ctx.pipeline.on_epoch_end(epoch)

        # Store best validation loss
        ctx.metadata['best_val_loss'] = best_val_loss

        return ctx

    def _run_validation(
        self,
        ctx: PhaseContext,
        epoch: int,
        val_interval: int,
        train_loss: float,
        coherence_config: dict,
        best_val_loss: float
    ) -> float:
        """Run validation if appropriate."""
        if ctx.val_loader is None:
            return best_val_loss

        if (epoch + 1) % val_interval != 0:
            return best_val_loss

        # Run validation
        val_loss = ctx.validation_mgr.validate(
            model=ctx.model,
            val_loader=ctx.val_loader,
            use_amp=ctx.context.use_amp,
            amp_dtype=ctx.context.amp_dtype,
        )

        # Log validation metrics
        current_step = ctx.training_mgr.get_global_step()
        ctx.metrics_mgr.log_validation(
            step=current_step,
            epoch=epoch,
            val_loss=val_loss
        )

        # Get coherence metrics
        coherence_metrics = None
        if coherence_config and coherence_config.get('enabled', False):
            if hasattr(ctx.generation_mgr, 'measure_coherence'):
                try:
                    coherence_metrics = ctx.generation_mgr.measure_coherence(
                        ctx.model, current_step, coherence_config
                    )
                except Exception as e:
                    self.log(ctx, f"Coherence measurement failed: {e}", "warning")

        # Check if best model
        if ctx.validation_mgr.is_best(
            val_loss,
            coherence_metrics=coherence_metrics,
            step=current_step,
            epoch=epoch,
        ):
            best_val_loss = val_loss
            quality_score = ctx.validation_mgr.get_best_quality_score()

            if ctx.is_main_process:
                if quality_score:
                    self.log(
                        ctx,
                        f"New best model! Quality={quality_score.quality_score:.4f} "
                        f"(val_loss={val_loss:.4f})"
                    )
                    ctx.metrics_mgr.log_quality_score(current_step, quality_score)
                else:
                    self.log(ctx, f"New best validation loss: {val_loss:.4f}")

                # Save best checkpoint
                try:
                    ctx.run_manager.save_checkpoint(
                        model_state=ctx.model.state_dict(),
                        optimizer_state=ctx.optimizer.state_dict(),
                        epoch=epoch,
                        step=current_step,
                        loss=val_loss,
                        is_best=True,
                        additional_data={
                            'train_loss': train_loss,
                            'quality_score': quality_score.to_dict() if quality_score else None,
                        }
                    )
                except Exception as e:
                    self.log(ctx, f"Failed to save best checkpoint: {e}", "error")

        # Synchronize ranks after validation
        if ctx.world_size > 1:
            import torch.distributed as dist
            dist.barrier()

        return best_val_loss

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        if ctx.model is None:
            errors.append("model must be built before training")
        if ctx.train_loader is None:
            errors.append("train_loader must be created before training")
        if ctx.training_mgr is None:
            errors.append("training_mgr must be initialized before training")
        return errors
