"""
Validation manager for the Ava pipeline.

Handles model validation during training with proper error tracking.

Accumulates losses on GPU and syncs once at the end for efficiency.
Supports multi-metric model selection using quality scores.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .context import ManagerInterface, TrainingContext
from .quality_score import ModelQualityScore, QualityScorer

if TYPE_CHECKING:
    from ..config.training_config import ModelSelectionConfig

logger = logging.getLogger(__name__)


class ValidationManager(ManagerInterface):
    """
    Manages validation during training.

    Tracks validation loss and determines if current model is best.
    Integrates with the Ava pipeline through TrainingContext.

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> val_manager = ValidationManager(context)
        >>> val_manager.initialize()
        >>> val_loss = val_manager.validate(model, val_loader)
        >>> if val_manager.is_best(val_loss):
        ...     save_checkpoint()
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the validation manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self.best_val_loss = float('inf')
        self._validation_count = 0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10

        # Multi-metric model selection
        self._quality_scorer: Optional[QualityScorer] = None
        self._best_quality_score: Optional[ModelQualityScore] = None
        self._quality_scoring_enabled = False

    def initialize(self) -> None:
        """Initialize the validation manager."""
        self._initialized = True
        self.logger.debug("ValidationManager initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> float:
        """
        Run validation loop and compute average loss.

        Args:
            model: Model to validate
            val_loader: Validation data loader
            use_amp: Whether to use automatic mixed precision
            amp_dtype: Data type for AMP (default: bfloat16)

        Returns:
            Average validation loss

        Raises:
            RuntimeError: If all batches fail during validation
        """
        self.assert_initialized()

        model.eval()
        num_batches = 0
        failed_batches = 0

        device = self.device

        # Accumulate losses on GPU, sync once at end
        loss_tensors: List[torch.Tensor] = []

        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc="Validation",
                leave=False,
                disable=not self.context.metadata.get('is_main_process', True)
            )

            for batch in progress_bar:
                try:
                    # Move batch to device with non_blocking transfers
                    # for async DMA transfers
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    # Forward pass with optional AMP
                    if use_amp and device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=amp_dtype):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                    # Extract loss - CRITICAL: Do NOT use logits as fallback for loss
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss')
                        if loss is None:
                            # Log what we actually got to help debug
                            available_keys = list(outputs.keys())
                            self.logger.error(
                                f"Model output missing 'loss' key. Available keys: {available_keys}. "
                                f"Ensure model is passed labels and returns loss."
                            )
                            failed_batches += 1
                            continue
                    elif isinstance(outputs, tuple):
                        loss = outputs[0]
                    else:
                        loss = outputs

                    if loss is None:
                        failed_batches += 1
                        self.logger.warning("Validation batch returned None loss")
                        continue

                    if torch.isnan(loss):
                        failed_batches += 1
                        self.logger.warning("Validation batch returned NaN loss")
                        continue

                    # Keep loss on GPU, use .detach().clone() to disconnect from graph
                    loss_tensors.append(loss.detach().clone())
                    num_batches += 1
                    # Show batch count in progress bar, loss displayed at end
                    if num_batches % 10 == 0:
                        progress_bar.set_postfix({'batches': num_batches})

                except Exception as e:
                    failed_batches += 1
                    self.logger.warning(f"Validation batch failed: {e}")

                    if failed_batches > self._max_consecutive_failures:
                        raise RuntimeError(
                            f"Too many validation failures ({failed_batches}). "
                            f"Last error: {e}"
                        )

        model.train()
        self._validation_count += 1

        if num_batches == 0:
            raise RuntimeError(
                f"All validation batches failed ({failed_batches} failures). "
                f"Check your validation data."
            )

        # FIX: Handle edge case where loss_tensors might be empty after NaN filtering
        if not loss_tensors:
            self.logger.error(
                f"No valid loss tensors collected ({failed_batches} failed, "
                f"{num_batches} processed but all had NaN/None loss)"
            )
            raise RuntimeError(
                f"Validation failed: all {num_batches} batches produced invalid losses. "
                f"Model may be outputting NaN values."
            )

        # Single sync point: compute mean on GPU, then transfer
        # FIX: Ensure consistent dtype before stacking to avoid precision issues
        loss_dtype = loss_tensors[0].dtype
        normalized_tensors = [t.to(loss_dtype) for t in loss_tensors]
        stacked_losses = torch.stack(normalized_tensors)
        avg_loss = stacked_losses.mean().item()  # Single .item() call for all batches

        if failed_batches > 0:
            self.logger.warning(
                f"Validation completed with {failed_batches} failed batches "
                f"out of {num_batches + failed_batches} total"
            )

        self.logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def setup_quality_scoring(self, config: 'ModelSelectionConfig') -> None:
        """
        Configure multi-metric quality scoring.

        Args:
            config: ModelSelectionConfig with weights and normalization settings
        """
        if config.enabled:
            self._quality_scorer = QualityScorer(config)
            self._quality_scoring_enabled = True
            self.logger.info(
                f"Quality scoring enabled: val_loss={config.val_loss_weight:.1%}, "
                f"coherence={config.coherence_score_weight:.1%}, "
                f"perplexity={config.perplexity_weight:.1%}"
            )
        else:
            self.logger.info("Quality scoring disabled, using val_loss only")

    def compute_quality_score(
        self,
        val_loss: float,
        coherence_metrics: Optional[Dict[str, Any]] = None,
        step: int = 0,
        epoch: int = 0,
    ) -> ModelQualityScore:
        """
        Compute quality score from validation results.

        Args:
            val_loss: Validation loss
            coherence_metrics: Optional dict with 'coherence_score' and 'perplexity'
            step: Current training step
            epoch: Current training epoch

        Returns:
            ModelQualityScore with composite quality score
        """
        if self._quality_scorer is not None:
            return self._quality_scorer.compute_from_coherence_metrics(
                val_loss=val_loss,
                coherence_metrics=coherence_metrics,
                step=step,
                epoch=epoch,
            )
        else:
            # Fallback: use val_loss only (lower is better -> invert for score)
            score = ModelQualityScore(val_loss=val_loss, step=step, epoch=epoch)
            score.val_loss_normalized = max(0.0, 1.0 - min(val_loss / 10.0, 1.0))
            score.quality_score = score.val_loss_normalized
            score.metrics_present = {'val_loss': True, 'coherence_score': False, 'perplexity': False}
            return score

    def is_best(
        self,
        val_loss: float,
        coherence_metrics: Optional[Dict[str, Any]] = None,
        step: int = 0,
        epoch: int = 0,
    ) -> bool:
        """
        Check if this is the best model so far using quality score.

        When quality scoring is enabled, uses a weighted combination of
        val_loss, coherence_score, and perplexity. Otherwise falls back
        to val_loss only.

        Args:
            val_loss: Current validation loss
            coherence_metrics: Optional dict with coherence metrics
            step: Current training step
            epoch: Current training epoch

        Returns:
            True if this is the best model so far
        """
        # Always track best val_loss for backward compatibility
        is_best_val_loss = val_loss < self.best_val_loss
        if is_best_val_loss:
            self.best_val_loss = val_loss

        # Use quality scoring if enabled
        if self._quality_scoring_enabled and self._quality_scorer is not None:
            current_score = self.compute_quality_score(
                val_loss, coherence_metrics, step, epoch
            )

            if self._quality_scorer.is_improvement(current_score, self._best_quality_score):
                self._best_quality_score = current_score
                self.logger.info(
                    f"New best model! Quality={current_score.quality_score:.4f} "
                    f"(val_loss={val_loss:.4f})"
                )
                return True
            return False
        else:
            # Fallback to val_loss only
            if is_best_val_loss:
                self.logger.info(f"New best validation loss: {val_loss:.4f}")
                return True
            return False

    def get_best_quality_score(self) -> Optional[ModelQualityScore]:
        """
        Get the best quality score seen so far.

        Returns:
            Best ModelQualityScore, or None if no validation has been done
        """
        return self._best_quality_score

    def get_status(self) -> Dict[str, Any]:
        """Return current validation status."""
        status = {
            'best_val_loss': self.best_val_loss,
            'validation_count': self._validation_count,
            'quality_scoring_enabled': self._quality_scoring_enabled,
        }
        if self._best_quality_score is not None:
            status['best_quality_score'] = self._best_quality_score.quality_score
            status['best_quality_details'] = self._best_quality_score.to_dict()
        return status

    def on_epoch_end(self, epoch: int) -> None:
        """Called at end of each epoch."""
        pass

    def on_error(self, error: Exception) -> None:
        """Handle validation errors."""
        self.logger.error(f"Validation error: {error}", exc_info=True)
