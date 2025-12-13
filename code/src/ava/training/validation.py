"""
Validation manager for the Ava pipeline.

Handles model validation during training with proper error tracking.

GPU SYNC OPTIMIZATION: Accumulates losses on GPU and syncs once at the end,
instead of calling .item() per batch which causes N separate cudaStreamSynchronize calls.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .context import ManagerInterface, TrainingContext

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

        # GPU SYNC FIX: Accumulate losses on GPU, sync once at end
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
                    # GPU SYNC FIX: Use non_blocking=True for async DMA transfers
                    # This allows CPU to continue preparing next batch while transfer happens
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

                    # Extract loss
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss', outputs.get('logits'))
                    else:
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs

                    if loss is not None and not torch.isnan(loss):
                        # GPU SYNC FIX: Keep loss on GPU, don't call .item()
                        # FIX: Use .detach().clone() to fully disconnect from computation graph
                        # and release any references to intermediate activations for GC
                        loss_tensors.append(loss.detach().clone())
                        num_batches += 1
                        # GPU SYNC FIX: Removed .item() call from progress bar update
                        # Previously: progress_bar.set_postfix({'loss': f'{loss_tensors[-1].item():.4f}'})
                        # This was causing cudaStreamSynchronize every 5 batches
                        # Now we only show batch count, loss is displayed at the end
                        if num_batches % 10 == 0:
                            progress_bar.set_postfix({'batches': num_batches})
                    else:
                        failed_batches += 1
                        self.logger.warning(f"Validation batch returned invalid loss")

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

        # GPU SYNC FIX: Single sync point - compute mean on GPU, then transfer
        stacked_losses = torch.stack(loss_tensors)
        avg_loss = stacked_losses.mean().item()  # Single .item() call for all batches

        if failed_batches > 0:
            self.logger.warning(
                f"Validation completed with {failed_batches} failed batches "
                f"out of {num_batches + failed_batches} total"
            )

        self.logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def is_best(self, val_loss: float) -> bool:
        """
        Check if this is the best validation loss so far.

        Args:
            val_loss: Current validation loss

        Returns:
            True if val_loss is the best so far
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.logger.info(f"New best validation loss: {val_loss:.4f}")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return current validation status."""
        return {
            'best_val_loss': self.best_val_loss,
            'validation_count': self._validation_count,
        }

    def on_epoch_end(self, epoch: int) -> None:
        """Called at end of each epoch."""
        pass

    def on_error(self, error: Exception) -> None:
        """Handle validation errors."""
        self.logger.error(f"Validation error: {error}", exc_info=True)
