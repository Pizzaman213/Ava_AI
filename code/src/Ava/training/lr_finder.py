"""
Learning Rate Finder (LR Range Test)

Implementation of Leslie N. Smith's LR Range Test for finding optimal learning rates.
This method incrementally increases the learning rate and tracks the training loss
to identify the "sweet spot" where the model learns most efficiently.

Reference: https://arxiv.org/abs/1506.01186
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from torch.optim import Optimizer  # type: ignore[import]

logger = logging.getLogger(__name__)


@dataclass
class LRFinderConfig:
    """Configuration for LR Finder."""
    # LR range to test
    start_lr: float = 1e-8  # Very low starting LR
    end_lr: float = 10.0  # High ending LR

    # Number of iterations to run
    num_iter: int = 100  # Number of mini-batches to test

    # Smoothing for loss curve
    beta: float = 0.98  # Exponential moving average beta for loss smoothing

    # Stopping criteria
    stop_div_threshold: float = 4.0  # Stop if loss > best_loss * threshold

    # Search mode
    mode: str = "exponential"  # "exponential" or "linear" LR increase

    # Plot configuration
    save_plot: bool = True
    plot_path: Optional[str] = None

    # Suggestion strategy
    suggestion_method: str = "steepest"  # "steepest", "minimum", "valley"


class LRFinder:
    """
    Learning Rate Finder using the LR Range Test method.

    Incrementally increases the learning rate and records the training loss
    to help identify the optimal learning rate range.

    Usage:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1, num_iter=100)
        lr_finder.plot()  # Visualize results
        suggested_lr = lr_finder.suggest_lr()  # Get suggested LR
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: torch.device,
        config: Optional[LRFinderConfig] = None
    ):
        """
        Initialize LR Finder.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to run on
            config: LR Finder configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config or LRFinderConfig()

        # Store original model state
        self.model_state = None
        self.optimizer_state = None

        # Results storage
        self.history: Dict[str, List[float]] = {
            'lr': [],
            'loss': [],
            'smooth_loss': []
        }

        # Best results
        self.best_loss = float('inf')
        self.best_lr = None

    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr: Optional[float] = None,
        end_lr: Optional[float] = None,
        num_iter: Optional[int] = None,
        smooth_beta: Optional[float] = None,
        accumulation_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Perform the LR range test.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader for additional metrics
            start_lr: Starting learning rate (overrides config)
            end_lr: Ending learning rate (overrides config)
            num_iter: Number of iterations (overrides config)
            smooth_beta: Smoothing factor (overrides config)
            accumulation_steps: Gradient accumulation steps

        Returns:
            Dictionary with results and suggested learning rate
        """
        # Use config defaults if not provided
        start_lr = start_lr or self.config.start_lr
        end_lr = end_lr or self.config.end_lr
        num_iter = num_iter or self.config.num_iter
        smooth_beta = smooth_beta or self.config.beta

        logger.info("=" * 80)
        logger.info("Starting LR Range Test")
        logger.info("=" * 80)
        logger.info(f"  Start LR: {start_lr:.2e}")
        logger.info(f"  End LR: {end_lr:.2e}")
        logger.info(f"  Iterations: {num_iter}")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  Accumulation steps: {accumulation_steps}")
        logger.info("-" * 80)

        # Save original state
        self._save_state()

        # Reset history
        self.history = {'lr': [], 'loss': [], 'smooth_loss': []}
        self.best_loss = float('inf')

        # Set model to training mode
        self.model.train()

        # Calculate LR schedule
        if self.config.mode == "exponential":
            lr_schedule = self._exponential_schedule(start_lr, end_lr, num_iter)
        else:
            lr_schedule = self._linear_schedule(start_lr, end_lr, num_iter)

        # Create infinite iterator from train_loader
        train_iter = iter(train_loader)

        # Variables for smoothed loss
        avg_loss = 0.0
        best_loss = float('inf')
        iteration = 0
        accumulation_counter = 0

        # Run the test
        for i in range(num_iter):
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Update learning rate
            current_lr = lr_schedule[i]
            self._set_lr(current_lr)

            # Forward pass
            loss = self._train_batch(batch, accumulation_steps)

            # Update iteration counter only after accumulation
            accumulation_counter += 1
            if accumulation_counter >= accumulation_steps:
                iteration += 1
                accumulation_counter = 0

            # Compute smoothed loss
            if i == 0:
                avg_loss = loss
            else:
                avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss

            # Bias correction for early iterations
            smoothed_loss = avg_loss / (1 - smooth_beta ** (i + 1))

            # Record history
            self.history['lr'].append(current_lr)
            self.history['loss'].append(loss)
            self.history['smooth_loss'].append(smoothed_loss)

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                self.best_lr = current_lr

            # Log progress
            if (i + 1) % max(1, num_iter // 10) == 0:
                logger.info(
                    f"  Iter {i+1}/{num_iter} | LR: {current_lr:.2e} | "
                    f"Loss: {loss:.4f} | Smooth: {smoothed_loss:.4f}"
                )

            # Check for divergence
            if smoothed_loss > best_loss * self.config.stop_div_threshold:
                logger.warning(
                    f"  Stopping early at iteration {i+1}: "
                    f"Loss diverged (smooth_loss={smoothed_loss:.4f} > "
                    f"best_loss={best_loss:.4f} * {self.config.stop_div_threshold})"
                )
                break

        # Restore original state
        self._restore_state()

        # Find suggested LR
        suggested_lr = self.suggest_lr()

        logger.info("-" * 80)
        logger.info(f"LR Range Test Complete!")
        logger.info(f"  Best Loss: {best_loss:.6f} at LR: {self.best_lr:.2e}")
        logger.info(f"  Suggested LR: {suggested_lr:.2e}")
        logger.info("=" * 80)

        # Plot results if requested
        if self.config.save_plot:
            self.plot()

        return {
            'suggested_lr': suggested_lr,
            'best_lr': self.best_lr,
            'best_loss': best_loss,
            'history': self.history,
            'num_iterations': len(self.history['lr'])
        }

    def _train_batch(self, batch, accumulation_steps: int = 1) -> float:
        """
        Train on a single batch.

        Args:
            batch: Training batch
            accumulation_steps: Number of accumulation steps

        Returns:
            Loss value
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            # Dictionary format (common in HuggingFace)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**inputs)

            # Extract loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Manual loss calculation
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                labels = inputs.get('labels', inputs.get('input_ids'))
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        elif isinstance(batch, (tuple, list)):
            # Tuple format (input, target)
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                # All items are inputs
                inputs = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch)
                targets = None

            # Forward pass
            outputs = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)

            # Calculate loss
            if targets is not None:
                loss = self.criterion(outputs, targets)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                raise ValueError("Cannot determine loss from outputs")

        else:
            # Single tensor (assumes labels are part of the model output)
            inputs = batch.to(self.device)
            outputs = self.model(inputs)

            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                raise ValueError("Cannot determine loss from outputs")

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Optimizer step (only if we've accumulated enough gradients)
        # Note: In LR finder, we step on every iteration for simplicity
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item() * accumulation_steps  # Return unscaled loss

    def _exponential_schedule(self, start_lr: float, end_lr: float, num_iter: int) -> List[float]:
        """Generate exponential LR schedule."""
        gamma = (end_lr / start_lr) ** (1 / (num_iter - 1))
        return [start_lr * (gamma ** i) for i in range(num_iter)]

    def _linear_schedule(self, start_lr: float, end_lr: float, num_iter: int) -> List[float]:
        """Generate linear LR schedule."""
        return [start_lr + (end_lr - start_lr) * i / (num_iter - 1) for i in range(num_iter)]

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _save_state(self):
        """Save model and optimizer state."""
        self.model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.optimizer_state = self.optimizer.state_dict()

    def _restore_state(self):
        """Restore model and optimizer to original state."""
        if self.model_state is not None:
            # Move state dict back to device
            state_dict = {k: v.to(self.device) for k, v in self.model_state.items()}
            self.model.load_state_dict(state_dict)

        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate based on the loss curve.

        Args:
            skip_start: Number of initial points to skip
            skip_end: Number of final points to skip

        Returns:
            Suggested learning rate
        """
        if len(self.history['lr']) < skip_start + skip_end:
            logger.warning("Not enough data points for LR suggestion, returning best LR")
            return self.best_lr or self.config.start_lr

        # Use smoothed losses for suggestion
        losses = self.history['smooth_loss'][skip_start:-skip_end] if skip_end > 0 else self.history['smooth_loss'][skip_start:]
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]

        if self.config.suggestion_method == "steepest":
            # Find point with steepest negative gradient
            gradients = []
            for i in range(1, len(losses)):
                # Calculate gradient in log-log space
                grad = (losses[i] - losses[i-1]) / (math.log10(lrs[i]) - math.log10(lrs[i-1]))
                gradients.append(grad)

            # Find steepest descent
            min_gradient_idx = gradients.index(min(gradients))
            suggested_lr = lrs[min_gradient_idx]

        elif self.config.suggestion_method == "minimum":
            # Find minimum loss point
            min_loss_idx = losses.index(min(losses))
            suggested_lr = lrs[min_loss_idx]

        elif self.config.suggestion_method == "valley":
            # Find the LR at 1/10th the value where loss starts increasing significantly
            min_loss = min(losses)
            min_loss_idx = losses.index(min_loss)

            # Use LR at 1/10 of the way to minimum
            valley_idx = max(0, min_loss_idx // 10)
            suggested_lr = lrs[valley_idx]

        else:
            # Default to steepest
            logger.warning(f"Unknown suggestion method '{self.config.suggestion_method}', using 'steepest'")
            return self.suggest_lr(skip_start, skip_end)

        logger.info(f"Suggested LR ({self.config.suggestion_method} method): {suggested_lr:.2e}")
        return suggested_lr

    def plot(self, log_lr: bool = True, skip_start: int = 10, skip_end: int = 5,
             save_path: Optional[str] = None):
        """
        Plot the learning rate range test results.

        Args:
            log_lr: Use log scale for LR axis
            skip_start: Skip initial points
            skip_end: Skip final points
            save_path: Path to save plot (overrides config)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot results")
            return

        # Prepare data
        lrs = self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:]
        smooth_losses = self.history['smooth_loss'][skip_start:]

        if skip_end > 0:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]
            smooth_losses = smooth_losses[:-skip_end]

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot loss vs LR
        plt.subplot(1, 2, 1)
        if log_lr:
            plt.semilogx(lrs, losses, alpha=0.3, label='Loss')
            plt.semilogx(lrs, smooth_losses, linewidth=2, label='Smoothed Loss')
        else:
            plt.plot(lrs, losses, alpha=0.3, label='Loss')
            plt.plot(lrs, smooth_losses, linewidth=2, label='Smoothed Loss')

        # Mark suggested LR
        suggested_lr = self.suggest_lr(skip_start, skip_end)
        plt.axvline(x=suggested_lr, color='r', linestyle='--',
                   label=f'Suggested LR: {suggested_lr:.2e}')

        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Finder: Loss vs Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot gradient
        plt.subplot(1, 2, 2)
        if len(smooth_losses) > 1:
            gradients = []
            gradient_lrs = []
            for i in range(1, len(smooth_losses)):
                grad = (smooth_losses[i] - smooth_losses[i-1]) / (lrs[i] - lrs[i-1])
                gradients.append(grad)
                gradient_lrs.append(lrs[i])

            if log_lr:
                plt.semilogx(gradient_lrs, gradients, linewidth=2)
            else:
                plt.plot(gradient_lrs, gradients, linewidth=2)

            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=suggested_lr, color='r', linestyle='--',
                       label=f'Suggested LR: {suggested_lr:.2e}')

            plt.xlabel('Learning Rate')
            plt.ylabel('Loss Gradient')
            plt.title('LR Finder: Loss Gradient')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = save_path or self.config.plot_path or 'lr_finder_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"LR Finder plot saved to: {save_path}")

        # Also try to show if in interactive mode
        try:
            plt.show()
        except:
            pass

        plt.close()

    def get_results(self) -> Dict[str, Any]:
        """Get complete results from the LR range test."""
        return {
            'history': self.history,
            'best_loss': self.best_loss,
            'best_lr': self.best_lr,
            'suggested_lr': self.suggest_lr(),
            'config': {
                'start_lr': self.config.start_lr,
                'end_lr': self.config.end_lr,
                'num_iter': self.config.num_iter,
                'mode': self.config.mode,
                'suggestion_method': self.config.suggestion_method
            }
        }


def find_lr(
    model: nn.Module,
    train_loader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    smooth_beta: float = 0.98,
    accumulation_steps: int = 1,
    plot: bool = True,
    plot_path: Optional[str] = None
) -> float:
    """
    Convenience function for running LR finder.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        smooth_beta: Smoothing factor for loss
        accumulation_steps: Gradient accumulation steps
        plot: Whether to plot results
        plot_path: Path to save plot

    Returns:
        Suggested learning rate
    """
    config = LRFinderConfig(
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        beta=smooth_beta,
        save_plot=plot,
        plot_path=plot_path
    )

    finder = LRFinder(model, optimizer, criterion, device, config)
    results = finder.range_test(
        train_loader,
        accumulation_steps=accumulation_steps
    )

    return results['suggested_lr']
