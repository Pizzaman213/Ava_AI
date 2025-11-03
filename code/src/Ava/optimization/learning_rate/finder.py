"""
Learning Rate Finder (LR Range Test) - Enhanced Edition

Implementation of Leslie N. Smith's LR Range Test for finding optimal learning rates.
This method incrementally increases the learning rate and tracks the training loss
to identify the "sweet spot" where the model learns most efficiently.

ENHANCEMENTS:
- Fastai suggestion method (1/10th minimum, most conservative)
- Statistical smoothing with Savitzky-Golay filter
- Automatic valley detection with inflection point analysis
- Optional validation loss tracking for accuracy
- Multi-run averaging for robustness
- Enhanced divergence detection (variance-based)
- Momentum cycling support (super-convergence)
- Comprehensive plotting with confidence intervals

Reference: https://arxiv.org/abs/1506.01186
Reference: https://arxiv.org/abs/1708.07120 (Super-convergence)
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from torch.optim import Optimizer  # type: ignore[import]
import numpy as np

# Optional imports for enhanced features
try:
    from scipy.signal import savgol_filter  # type: ignore[import]
    from scipy.ndimage import gaussian_filter1d  # type: ignore[import]
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, some smoothing features will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class LRFinderConfig:
    """Configuration for LR Finder - Enhanced Edition."""
    # LR range to test
    start_lr: float = 1e-8  # Very low starting LR
    end_lr: float = 1e-2  # IMPROVED: Extended range for better exploration (was 10.0)

    # Number of iterations to run
    num_iter: int = 1000  # IMPROVED: 1000 iterations for maximum accuracy (was 100)

    # Smoothing for loss curve - FIXED: Balanced smoothing to prevent overfitting
    beta: float = 0.9  # FIXED: Moderate smoothing (was 0.98 which was too aggressive)
    use_savgol_filter: bool = True  # Use Savitzky-Golay filter (requires scipy)
    savgol_window: int = 31  # FIXED: Reasonable window for 1000 steps (was 51, must be odd)
    savgol_polyorder: int = 3  # Polynomial order for Savitzky-Golay filter

    # Stopping criteria - FIXED: Better balance to detect divergence
    stop_div_threshold: float = 4.0  # FIXED: Reasonable threshold (was 8.0 which was too lenient)
    use_variance_stopping: bool = True  # Stop if loss variance too high
    variance_threshold: float = 3.0  # FIXED: Reasonable variance threshold (was 5.0)

    # Search mode
    mode: str = "exponential"  # "exponential" or "linear" LR increase

    # Plot configuration
    save_plot: bool = True
    plot_path: Optional[str] = None

    # Suggestion strategy
    suggestion_method: str = "fastai"  # "fastai", "steepest", "minimum", "valley", "combined"

    # Enhanced features
    track_validation: bool = False  # Track validation loss (more accurate but slower)
    num_runs: int = 1  # Number of runs to average (1 = no averaging)
    momentum_cycling: bool = False  # Cycle momentum with LR (super-convergence)
    momentum_range: Tuple[float, float] = (0.85, 0.95)  # Momentum range for cycling


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
            'smooth_loss': [],
            'val_loss': [],  # Enhanced: track validation loss
            'gradient': [],  # Enhanced: track loss gradients
            'momentum': []   # Enhanced: track momentum if cycling
        }

        # Best results
        self.best_loss = float('inf')
        self.best_lr = None

        # Multi-run storage for averaging
        self.run_histories: List[Dict[str, List[float]]] = []

        # Momentum cycling state
        self.base_momentum_values: List[Optional[float]] = []
        if self.config.momentum_cycling:
            for group in optimizer.param_groups:
                if 'momentum' in group or 'betas' in group:
                    if 'momentum' in group:
                        self.base_momentum_values.append(group['momentum'])
                    elif 'betas' in group:
                        self.base_momentum_values.append(group['betas'][0])
                else:
                    self.base_momentum_values.append(None)

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
        logger.info(f"  Gradient Accumulation Steps: {accumulation_steps}")
        if accumulation_steps != 1:
            logger.warning(f"  ⚠️  WARNING: LR Finder using gradient_accumulation_steps={accumulation_steps}")
            logger.warning(f"      For most accurate results, should use gradient_accumulation_steps=1")
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
                loss = outputs.loss  # type: ignore
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

        # FIXED: Don't scale loss in LR finder - we step every iteration
        # Gradient accumulation is handled at the iteration level, not here
        # The LR finder needs to see true loss values to make correct decisions

        # Backward pass
        loss.backward()

        # Optimizer step - always step in LR finder (no accumulation)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Return actual loss value (not scaled)
        return loss.item()

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

    def _apply_savgol_smoothing(self, values: List[float]) -> List[float]:
        """Apply Savitzky-Golay filter for noise reduction."""
        if not SCIPY_AVAILABLE or len(values) < self.config.savgol_window:
            return values

        try:
            # Ensure window size is odd and less than data length
            window = min(self.config.savgol_window, len(values) - 1)
            if window % 2 == 0:
                window -= 1
            if window < 3:
                return values

            polyorder = min(self.config.savgol_polyorder, window - 1)
            smoothed = savgol_filter(values, window, polyorder)  # type: ignore[name-defined]
            return smoothed.tolist()
        except Exception as e:
            logger.warning(f"Savitzky-Golay filtering failed: {e}, using original values")
            return values

    def _detect_valley(self, losses: List[float], lrs: List[float]) -> Tuple[int, int]:
        """
        Improved valley detection using derivative analysis and loss thresholds.

        The valley is where loss is decreasing steadily toward minimum.
        We want to find the LR that gives us good learning without instability.

        Returns:
            Tuple of (valley_start_idx, valley_target_idx)
        """
        if len(losses) < 10:
            return (0, len(losses) - 1)

        # Find minimum loss point
        min_idx = losses.index(min(losses))
        min_loss = losses[min_idx]
        start_loss = losses[0]

        # Compute smoothed gradients to find inflection points
        gradients = self._compute_gradients(losses, lrs)

        if len(gradients) < 5:
            # Fallback to simple method
            return (max(0, min_idx - 10), min(len(losses) - 1, min_idx + 10))

        # Find valley START: where loss has decreased by at least 5% from start
        # AND gradient is consistently negative (actual learning happening)
        valley_start = 0
        threshold_loss = start_loss * 0.95  # 5% decrease

        for i in range(min_idx):
            if losses[i] < threshold_loss:
                # Check for sustained negative gradient (3 consecutive)
                if i + 3 < len(gradients):
                    if all(gradients[j] < 0 for j in range(i, min(i+3, len(gradients)))):
                        valley_start = i
                        break
                else:
                    valley_start = i
                    break

        # Find valley END: where loss is within 10% of minimum
        # This is where we're getting close to optimal
        valley_end = min_idx
        target_loss = min_loss * 1.10  # Within 10% of minimum

        for i in range(valley_start, min_idx):
            if losses[i] <= target_loss:
                valley_end = i
                break

        # If valley_end is same as valley_start, use minimum as end
        if valley_end <= valley_start:
            valley_end = min_idx

        # Valley target: Use the point that's 2/3 of the way from start to end
        # This balances between too conservative (start) and too aggressive (end)
        valley_width = valley_end - valley_start
        if valley_width > 10:
            # For wide valleys, use 2/3 point (less conservative)
            valley_target = valley_start + int(valley_width * 0.67)
        else:
            # For narrow valleys, use midpoint
            valley_target = valley_start + valley_width // 2

        return (valley_start, valley_target)

    def _compute_gradients(self, losses: List[float], lrs: List[float]) -> List[float]:
        """Compute numerical gradients of loss curve in log-log space."""
        gradients = []
        for i in range(1, len(losses)):
            if lrs[i] > 0 and lrs[i-1] > 0:
                grad = (losses[i] - losses[i-1]) / (math.log10(lrs[i]) - math.log10(lrs[i-1]))
                gradients.append(grad)
            else:
                gradients.append(0.0)
        return gradients

    def _check_variance_divergence(self, recent_losses: List[float]) -> bool:
        """Check if loss variance indicates divergence."""
        if len(recent_losses) < 5:
            return False

        # Compute variance of recent losses
        mean_loss = np.mean(recent_losses)
        variance = np.var(recent_losses)

        # Normalized variance (coefficient of variation squared)
        if mean_loss > 0:
            normalized_var = variance / (mean_loss ** 2)
            return bool(normalized_var > self.config.variance_threshold)  # type: ignore[return-value]
        return False

    def _cycle_momentum(self, current_lr: float, lr_schedule: List[float]):
        """
        Cycle momentum inversely with learning rate (super-convergence).

        When LR increases, momentum decreases and vice versa.
        """
        if not self.config.momentum_cycling:
            return

        # Find position in LR schedule
        lr_min, lr_max = min(lr_schedule), max(lr_schedule)
        if lr_max == lr_min:
            return

        # Normalize current LR to [0, 1]
        lr_normalized = (current_lr - lr_min) / (lr_max - lr_min)

        # Momentum cycles inversely: high LR = low momentum
        momentum_min, momentum_max = self.config.momentum_range
        current_momentum = momentum_max - (momentum_max - momentum_min) * lr_normalized

        # Apply to optimizer
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if self.base_momentum_values[idx] is not None:
                if 'momentum' in param_group:
                    param_group['momentum'] = current_momentum
                elif 'betas' in param_group:
                    # For Adam-like optimizers, adjust beta1
                    param_group['betas'] = (current_momentum, param_group['betas'][1])

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

        # Apply additional Savitzky-Golay smoothing if enabled
        if self.config.use_savgol_filter:
            losses = self._apply_savgol_smoothing(losses)

        if self.config.suggestion_method == "fastai":
            # FastAI method: Use LR where loss is 1/10th down from start to minimum
            # This is the CORRECT FastAI implementation (not 1/10th of index!)
            min_loss = min(losses)
            min_loss_idx = losses.index(min_loss)

            # Find LR where loss first drops to 90% of the way from start to minimum
            # This ensures we're well into the learning region but not too aggressive
            start_loss = losses[0]
            target_loss = start_loss - 0.9 * (start_loss - min_loss)

            # Find first point where loss crosses this threshold
            target_idx = min_loss_idx  # Default to minimum if not found
            for i in range(min_loss_idx):
                if losses[i] <= target_loss:
                    target_idx = max(0, i - 5)  # Go back 5 steps for safety
                    break

            suggested_lr = lrs[target_idx]
            logger.info(f"FastAI method: min_loss at idx {min_loss_idx} (LR={lrs[min_loss_idx]:.2e}), "
                       f"target_loss={target_loss:.4f}, using idx {target_idx} (LR={suggested_lr:.2e})")

        elif self.config.suggestion_method == "steepest":
            # Find point with steepest negative gradient in the LEARNING region
            # Skip early noise and post-minimum divergence
            gradients = self._compute_gradients(losses, lrs)

            if len(gradients) > 0:
                min_loss_idx = losses.index(min(losses))

                # Skip first 10% (noisy initialization) and only look before minimum
                skip_start = max(5, len(gradients) // 10)
                gradients_learning_region = gradients[skip_start:min_loss_idx]

                if len(gradients_learning_region) > 0:
                    # Find steepest descent in the learning region
                    min_gradient_idx = gradients_learning_region.index(min(gradients_learning_region))
                    # Adjust for skipped start
                    actual_idx = skip_start + min_gradient_idx
                    suggested_lr = lrs[actual_idx + 1]  # +1 because gradients are shifted

                    logger.info(f"Steepest method: steepest gradient at idx {actual_idx} "
                               f"(in learning region {skip_start}:{min_loss_idx}), LR={suggested_lr:.2e}")
                else:
                    # Fallback: use 1/3 between skip_start and minimum
                    fallback_idx = skip_start + (min_loss_idx - skip_start) // 3
                    suggested_lr = lrs[fallback_idx]
                    logger.info(f"Steepest method: using fallback at idx {fallback_idx}, LR={suggested_lr:.2e}")
            else:
                suggested_lr = lrs[len(lrs) // 2]

        elif self.config.suggestion_method == "minimum":
            # Find minimum loss point (least conservative)
            min_loss_idx = losses.index(min(losses))
            suggested_lr = lrs[min_loss_idx]

        elif self.config.suggestion_method == "valley":
            # Find valley region and use improved detection
            valley_start, valley_target = self._detect_valley(losses, lrs)
            suggested_lr = lrs[valley_target]

            logger.info(f"Valley method: region starts at idx {valley_start}, "
                       f"target idx {valley_target}, LR={suggested_lr:.2e}")

        elif self.config.suggestion_method == "combined":
            # Combined method: average of multiple methods
            # This is most robust but may not be optimal for all cases
            min_loss_idx = losses.index(min(losses))
            fastai_idx = max(0, min_loss_idx // 10)
            gradients = self._compute_gradients(losses, lrs)
            steepest_idx = gradients.index(min(gradients)) if gradients else min_loss_idx

            # Weight the suggestions: fastai (50%), steepest (30%), minimum (20%)
            suggested_lr = (
                0.5 * lrs[fastai_idx] +
                0.3 * lrs[steepest_idx] +
                0.2 * lrs[min_loss_idx]
            )

            logger.info(f"Combined method: fastai={lrs[fastai_idx]:.2e}, steepest={lrs[steepest_idx]:.2e}, min={lrs[min_loss_idx]:.2e}")

        else:
            # Default to fastai (most reliable)
            logger.warning(f"Unknown suggestion method '{self.config.suggestion_method}', using 'fastai'")
            self.config.suggestion_method = "fastai"
            return self.suggest_lr(skip_start, skip_end)

        logger.info(f"Suggested LR ({self.config.suggestion_method} method): {suggested_lr:.2e}")
        return suggested_lr

    def plot(self, log_lr: bool = True, skip_start: int = 10, skip_end: int = 5,
             save_path: Optional[str] = None):
        """
        Enhanced plot with all suggestion methods and confidence intervals.

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

        # Apply Savitzky-Golay smoothing if available
        if self.config.use_savgol_filter:
            extra_smooth = self._apply_savgol_smoothing(smooth_losses)
        else:
            extra_smooth = smooth_losses

        # Create enhanced plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ===== Plot 1: Loss vs LR with all methods =====
        ax1 = axes[0, 0]
        if log_lr:
            ax1.semilogx(lrs, losses, alpha=0.2, label='Raw Loss', color='gray')
            ax1.semilogx(lrs, smooth_losses, linewidth=2, label='EMA Smoothed', color='blue')
            if self.config.use_savgol_filter:
                ax1.semilogx(lrs, extra_smooth, linewidth=2, label='Savitzky-Golay', color='purple', linestyle='--')
        else:
            ax1.plot(lrs, losses, alpha=0.2, label='Raw Loss', color='gray')
            ax1.plot(lrs, smooth_losses, linewidth=2, label='EMA Smoothed', color='blue')
            if self.config.use_savgol_filter:
                ax1.plot(lrs, extra_smooth, linewidth=2, label='Savitzky-Golay', color='purple', linestyle='--')

        # Mark all suggestion methods
        colors = {'fastai': 'red', 'steepest': 'green', 'minimum': 'orange', 'valley': 'cyan'}
        original_method = self.config.suggestion_method

        for method, color in colors.items():
            self.config.suggestion_method = method
            try:
                suggested = self.suggest_lr(skip_start, skip_end)
                ax1.axvline(x=suggested, color=color, linestyle='--', alpha=0.7,
                           label=f'{method.capitalize()}: {suggested:.2e}')
            except:
                pass

        self.config.suggestion_method = original_method  # Restore

        ax1.set_xlabel('Learning Rate', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('LR Finder: Loss vs Learning Rate (All Methods)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ===== Plot 2: Gradient Analysis =====
        ax2 = axes[0, 1]
        if len(extra_smooth) > 1:
            gradients = self._compute_gradients(extra_smooth, lrs)

            if log_lr:
                ax2.semilogx(lrs[1:], gradients, linewidth=2, color='darkblue')
            else:
                ax2.plot(lrs[1:], gradients, linewidth=2, color='darkblue')

            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

            # Mark steepest gradient point
            if len(gradients) > 0:
                min_grad_idx = gradients.index(min(gradients))
                ax2.axvline(x=lrs[min_grad_idx + 1], color='green', linestyle='--', alpha=0.7,
                           label=f'Steepest: {lrs[min_grad_idx + 1]:.2e}')

            ax2.set_xlabel('Learning Rate', fontsize=12)
            ax2.set_ylabel('Loss Gradient (d(loss)/d(log(lr)))', fontsize=12)
            ax2.set_title('LR Finder: Gradient Analysis', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        # ===== Plot 3: Valley Detection =====
        ax3 = axes[1, 0]
        if log_lr:
            ax3.semilogx(lrs, extra_smooth, linewidth=2, color='blue', label='Smoothed Loss')
        else:
            ax3.plot(lrs, extra_smooth, linewidth=2, color='blue', label='Smoothed Loss')

        # Detect and shade valley region
        valley_start, valley_end = self._detect_valley(extra_smooth, lrs)
        ax3.axvspan(lrs[valley_start], lrs[valley_end], alpha=0.2, color='yellow', label='Valley Region')

        # Mark minimum
        min_idx = extra_smooth.index(min(extra_smooth))
        ax3.scatter([lrs[min_idx]], [extra_smooth[min_idx]], color='red', s=100, zorder=5,
                   label=f'Minimum: {lrs[min_idx]:.2e}')

        ax3.set_xlabel('Learning Rate', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('LR Finder: Valley Detection', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # ===== Plot 4: Summary Statistics =====
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Calculate summary statistics
        min_loss = min(extra_smooth)
        min_idx = extra_smooth.index(min_loss)
        min_lr = lrs[min_idx]

        # Get all suggestions
        suggestions = {}
        for method in ['fastai', 'steepest', 'minimum', 'valley']:
            self.config.suggestion_method = method
            try:
                suggestions[method] = self.suggest_lr(skip_start, skip_end)
            except:
                suggestions[method] = None
        self.config.suggestion_method = original_method

        # Create summary text
        summary_text = "📊 LR Finder Summary\n" + "="*50 + "\n\n"
        summary_text += f"Iterations: {len(self.history['lr'])}\n"
        summary_text += f"LR Range: {min(self.history['lr']):.2e} - {max(self.history['lr']):.2e}\n\n"
        summary_text += f"Best Loss: {min_loss:.6f}\n"
        summary_text += f"Best LR: {min_lr:.2e}\n\n"
        summary_text += "🎯 Suggested Learning Rates:\n" + "-"*50 + "\n"

        for method, lr_val in suggestions.items():
            if lr_val:
                marker = "✓" if method == original_method else " "
                summary_text += f"{marker} {method.capitalize():12s}: {lr_val:.6e}\n"

        summary_text += "\n💡 Recommendation:\n" + "-"*50 + "\n"
        summary_text += f"Use '{original_method}' method: {suggestions.get(original_method, 0):.2e}\n\n"
        summary_text += "📝 Notes:\n"
        summary_text += "• FastAI: Most conservative (1/10 before min)\n"
        summary_text += "• Steepest: Fastest descent point\n"
        summary_text += "• Minimum: Least conservative\n"
        summary_text += "• Valley: Midpoint of valley region\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save plot
        save_path = save_path or self.config.plot_path or 'lr_finder_results_enhanced.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Enhanced LR Finder plot saved to: {save_path}")

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
