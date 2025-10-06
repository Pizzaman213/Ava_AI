"""
Advanced Warmup and Scheduling Features

Implements:
- Gradient noise scale for adaptive batch sizing
- Learning rate finder (one-cycle style)
- Cyclical batch sizes
- Adaptive warmup strategies
- Performance-based scheduling
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)


class GradientNoiseScale:
    """
    Gradient noise scale for adaptive batch size selection.

    Implements method from "An Empirical Model of Large-Batch Training"
    (McCandlish et al., 2018)
    """

    def __init__(
        self,
        window_size: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize gradient noise scale tracker.

        Args:
            window_size: Window for noise scale estimation
            batch_size: Current batch size
        """
        self.window_size = window_size
        self.batch_size = batch_size

        self.grad_norms: deque = deque(maxlen=window_size)
        self.noise_scale = 0.0

        logger.info(f"Gradient noise scale initialized (window={window_size})")

    def update(self, grad_norm: float):
        """
        Update with gradient norm.

        Args:
            grad_norm: Gradient norm for this step
        """
        self.grad_norms.append(grad_norm)

        if len(self.grad_norms) >= 10:
            # Estimate noise scale: B_noise = (grad_std / grad_mean)^2
            norms = list(self.grad_norms)
            mean = np.mean(norms)
            std = np.std(norms)

            if mean > 0:
                noise_scale = (std / mean) ** 2
                self.noise_scale = noise_scale

    def get_optimal_batch_size(
        self,
        current_batch_size: int,
        target_efficiency: float = 0.9
    ) -> int:
        """
        Get optimal batch size based on noise scale.

        Args:
            current_batch_size: Current batch size
            target_efficiency: Target training efficiency (0-1)

        Returns:
            Recommended batch size
        """
        if self.noise_scale == 0:
            return current_batch_size

        # Simple batch size = noise_scale / (1 - efficiency)
        optimal = self.noise_scale / (1 - target_efficiency)

        # Clamp to reasonable values
        optimal = max(1, min(optimal, current_batch_size * 4))

        return int(optimal)


class LearningRateFinder:
    """
    Learning rate finder using exponential growth.

    Implements range test from "Cyclical Learning Rates for Training Neural Networks"
    (Smith, 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        smoothing: float = 0.05
    ):
        """
        Initialize learning rate finder.

        Args:
            model: Model to test
            optimizer: Optimizer
            criterion: Loss criterion
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_steps: Number of test steps
            smoothing: Loss smoothing factor
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smoothing = smoothing

        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []

        logger.info(f"LR Finder initialized: range=[{min_lr}, {max_lr}], steps={num_steps}")

    def find(
        self,
        train_loader,
        device: str = 'cuda'
    ) -> Tuple[float, List[float], List[float]]:
        """
        Run learning rate range test.

        Args:
            train_loader: Training data loader
            device: Device for computation

        Returns:
            Tuple of (suggested_lr, lr_list, loss_list)
        """
        # Save initial state
        initial_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        # Exponential growth factor
        gamma = (self.max_lr / self.min_lr) ** (1 / self.num_steps)

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

        current_lr = self.min_lr
        best_loss = float('inf')
        smoothed_loss = 0.0

        self.model.train()

        # Training loop
        step = 0
        iterator = iter(train_loader)

        while step < self.num_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

            if isinstance(outputs, dict):
                loss = outputs.get('loss', self.criterion(outputs, batch.get('labels')))
            else:
                loss = self.criterion(outputs, batch.get('labels') if isinstance(batch, dict) else batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            loss_value = loss.item()

            # Smooth loss
            if step == 0:
                smoothed_loss = loss_value
            else:
                smoothed_loss = self.smoothing * loss_value + (1 - self.smoothing) * smoothed_loss

            self.lrs.append(current_lr)
            self.losses.append(smoothed_loss)

            # Check for divergence
            if smoothed_loss > 4 * best_loss or math.isnan(smoothed_loss):
                logger.warning(f"Loss diverged at lr={current_lr}, stopping")
                break

            best_loss = min(best_loss, smoothed_loss)

            # Increase learning rate
            current_lr *= gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            step += 1

            if step % 10 == 0:
                logger.info(f"LR Finder step {step}/{self.num_steps}: lr={current_lr:.2e}, loss={smoothed_loss:.4f}")

        # Restore initial state
        self.model.load_state_dict(initial_state['model'])
        self.optimizer.load_state_dict(initial_state['optimizer'])

        # Suggest learning rate (steepest descent point)
        suggested_lr = self._suggest_lr()

        logger.info(f"LR Finder complete. Suggested LR: {suggested_lr:.2e}")

        return suggested_lr, self.lrs, self.losses

    def _suggest_lr(self) -> float:
        """Suggest optimal learning rate from results."""
        if len(self.losses) < 2:
            return self.min_lr

        # Find steepest gradient
        gradients = np.gradient(self.losses)
        min_gradient_idx = np.argmin(gradients)

        # Use LR slightly before minimum gradient (more conservative)
        suggested_idx = max(0, min_gradient_idx - len(self.losses) // 20)

        return self.lrs[suggested_idx]


class CyclicalBatchScheduler:
    """
    Cyclical batch size scheduling.

    Varies batch size during training for better generalization.
    """

    def __init__(
        self,
        min_batch_size: int = 8,
        max_batch_size: int = 64,
        cycle_length: int = 1000,
        mode: str = 'triangular'
    ):
        """
        Initialize cyclical batch scheduler.

        Args:
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            cycle_length: Length of one cycle (steps)
            mode: Cycle mode ('triangular', 'triangular2', 'exp_range')
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.cycle_length = cycle_length
        self.mode = mode

        self.step = 0

        logger.info(f"Cyclical batch scheduling: [{min_batch_size}, {max_batch_size}], "
                   f"cycle={cycle_length}, mode={mode}")

    def get_batch_size(self) -> int:
        """Get current batch size."""
        cycle_pos = (self.step % self.cycle_length) / self.cycle_length

        if self.mode == 'triangular':
            # Linear increase then decrease
            if cycle_pos < 0.5:
                scale = cycle_pos * 2
            else:
                scale = 2 - cycle_pos * 2

        elif self.mode == 'triangular2':
            # Triangular with decreasing amplitude
            cycle_num = self.step // self.cycle_length
            amplitude = 1.0 / (2 ** cycle_num)

            if cycle_pos < 0.5:
                scale = cycle_pos * 2 * amplitude
            else:
                scale = (2 - cycle_pos * 2) * amplitude

        elif self.mode == 'exp_range':
            # Exponential decay
            gamma = 0.99994
            scale = (gamma ** self.step)

        else:
            scale = 1.0

        # Calculate batch size
        batch_size = self.min_batch_size + int(
            (self.max_batch_size - self.min_batch_size) * scale
        )

        self.step += 1

        return batch_size


class AdaptiveWarmupScheduler(_LRScheduler):
    """
    Adaptive warmup scheduler that adjusts based on training dynamics.

    Automatically determines warmup length based on gradient statistics.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        min_warmup_steps: int = 100,
        max_warmup_steps: int = 10000,
        grad_clip_threshold: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Initialize adaptive warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            base_scheduler: Base scheduler after warmup
            min_warmup_steps: Minimum warmup steps
            max_warmup_steps: Maximum warmup steps
            grad_clip_threshold: Threshold for stable gradients
            last_epoch: Last epoch
        """
        self.base_scheduler = base_scheduler
        self.min_warmup_steps = min_warmup_steps
        self.max_warmup_steps = max_warmup_steps
        self.grad_clip_threshold = grad_clip_threshold

        # Gradient tracking
        self.grad_norms: List[float] = []
        self.warmup_complete = False
        self.actual_warmup_steps = min_warmup_steps

        super().__init__(optimizer, last_epoch)

        logger.info(f"Adaptive warmup: range=[{min_warmup_steps}, {max_warmup_steps}]")

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if not self.warmup_complete:
            # Still in warmup
            warmup_factor = (self.last_epoch + 1) / self.actual_warmup_steps
            warmup_factor = min(warmup_factor, 1.0)

            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            return self.base_scheduler.get_lr()

    def step(self, grad_norm: Optional[float] = None):
        """
        Step the scheduler.

        Args:
            grad_norm: Current gradient norm
        """
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

            # Check if warmup should complete
            if not self.warmup_complete and self.last_epoch >= self.min_warmup_steps:
                if self._check_gradient_stability():
                    self.warmup_complete = True
                    self.actual_warmup_steps = self.last_epoch
                    logger.info(f"Warmup completed at step {self.actual_warmup_steps}")

            # Force complete after max steps
            if self.last_epoch >= self.max_warmup_steps:
                self.warmup_complete = True

        super().step()

        if self.warmup_complete:
            self.base_scheduler.step()

    def _check_gradient_stability(self) -> bool:
        """Check if gradients are stable enough to end warmup."""
        if len(self.grad_norms) < 20:
            return False

        recent_norms = self.grad_norms[-20:]

        # Check if gradient norms are below threshold
        avg_norm = np.mean(recent_norms)
        std_norm = np.std(recent_norms)

        # Stable if within threshold and low variance
        is_stable = (avg_norm < self.grad_clip_threshold and
                    std_norm / (avg_norm + 1e-6) < 0.5)

        return is_stable


class PerformanceBasedScheduler(_LRScheduler):
    """
    Schedule learning rate based on actual training performance.

    Increases LR when training is stable, decreases when unstable.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        patience: int = 10,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        min_lr: float = 1e-7,
        max_lr: float = 1e-2,
        metric_window: int = 50,
        last_epoch: int = -1
    ):
        """
        Initialize performance-based scheduler.

        Args:
            optimizer: Wrapped optimizer
            patience: Patience for performance degradation
            increase_factor: Factor to increase LR by
            decrease_factor: Factor to decrease LR by
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            metric_window: Window for performance metrics
            last_epoch: Last epoch
        """
        self.patience = patience
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.metric_window = metric_window

        # Performance tracking
        self.loss_history: deque = deque(maxlen=metric_window)
        self.best_loss = float('inf')
        self.bad_epochs = 0

        super().__init__(optimizer, last_epoch)

        logger.info(f"Performance-based scheduling: patience={patience}")

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, loss: float):
        """
        Step scheduler with loss value.

        Args:
            loss: Current loss value
        """
        self.loss_history.append(loss)

        if len(self.loss_history) < self.metric_window // 2:
            return

        # Calculate smoothed loss
        smoothed_loss = np.mean(list(self.loss_history)[-10:])

        # Check if improving
        if smoothed_loss < self.best_loss * 0.99:  # 1% improvement threshold
            self.best_loss = smoothed_loss
            self.bad_epochs = 0

            # Increase learning rate (training is stable)
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = min(old_lr * self.increase_factor, self.max_lr)
                param_group['lr'] = new_lr

                if new_lr != old_lr:
                    logger.info(f"Increasing LR: {old_lr:.2e} -> {new_lr:.2e}")

        else:
            self.bad_epochs += 1

            # Decrease learning rate after patience
            if self.bad_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.decrease_factor, self.min_lr)
                    param_group['lr'] = new_lr

                    if new_lr != old_lr:
                        logger.info(f"Decreasing LR: {old_lr:.2e} -> {new_lr:.2e}")

                self.bad_epochs = 0

        super().step()


def find_optimal_learning_rate(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer_class = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    device: str = 'cuda',
    num_steps: int = 100
) -> float:
    """
    Find optimal learning rate using LR range test.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer arguments
        device: Device for training
        num_steps: Number of test steps

    Returns:
        Suggested optimal learning rate
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Create temporary optimizer
    temp_optimizer = optimizer_class(model.parameters(), lr=1e-7, **optimizer_kwargs)

    # Run LR finder
    lr_finder = LearningRateFinder(
        model=model,
        optimizer=temp_optimizer,
        criterion=criterion,
        num_steps=num_steps
    )

    suggested_lr, lrs, losses = lr_finder.find(train_loader, device)

    logger.info(f"Optimal learning rate found: {suggested_lr:.2e}")

    return suggested_lr
