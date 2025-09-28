"""
Intelligent Learning Rate Management System

Replaces hardcoded values with proper calculation based on dataset size,
gradient accumulation, and training configuration.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
import math
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LRConfig:
    """Configuration for learning rate management."""
    # Warmup configuration
    warmup_ratio: float = 0.03  # 3% of total steps for warmup
    warmup_min_ratio: float = 0.01  # Start at 1% of target LR
    warmup_schedule: str = "linear"  # "linear", "cosine", "polynomial"

    # Main scheduler configuration
    main_schedule: str = "cosine"  # "cosine", "linear_decay", "polynomial", "constant"
    min_lr_ratio: float = 0.01  # Minimum LR as fraction of initial LR

    # Adaptive LR configuration
    enable_adaptive: bool = False
    plateau_patience: int = 10  # Steps to wait before reducing LR
    plateau_threshold: float = 0.01  # Minimum improvement required
    plateau_factor: float = 0.5  # Factor to reduce LR by
    plateau_min_lr: float = 1e-8  # Absolute minimum LR

    # Gradient accumulation awareness
    gradient_accumulation_steps: int = 1

    # Recovery configuration
    enable_lr_recovery: bool = True
    recovery_warmup_steps: int = 100  # Steps to warmup after LR reduction


class IntelligentLRManager:
    """
    Intelligent learning rate manager that adapts to actual training conditions.

    Calculates optimal warmup based on dataset size, handles plateau detection,
    and provides recovery mechanisms.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: LRConfig,
        total_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Initialize LR manager.

        Args:
            optimizer: PyTorch optimizer
            config: LR configuration
            total_steps: Total training steps (calculated if None)
            steps_per_epoch: Steps per epoch for calculation
        """
        self.optimizer = optimizer
        self.config = config
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

        # Calculate training schedule
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

        # Calculate warmup and main training steps
        if self.total_steps:
            self.warmup_steps = max(1, int(self.total_steps * config.warmup_ratio))
            self.main_training_steps = self.total_steps - self.warmup_steps
        else:
            self.warmup_steps = 1000  # Fallback
            self.main_training_steps = 10000  # Fallback

        # Adaptive LR state
        self.plateau_detector = PlateauDetector(
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            factor=config.plateau_factor,
            min_lr=config.plateau_min_lr
        ) if config.enable_adaptive else None

        # Recovery state
        self.in_recovery = False
        self.recovery_start_step = 0
        self.pre_recovery_lr = None

        # Statistics
        self.lr_history = deque(maxlen=1000)
        self.reduction_count = 0
        self.recovery_count = 0

        logger.info(f"LR Manager initialized:")
        logger.info(f"  Total steps: {self.total_steps}")

        # Handle None total_steps for percentage calculation
        if self.total_steps is not None:
            warmup_percentage = self.warmup_steps / max(self.total_steps, 1)
            logger.info(f"  Warmup steps: {self.warmup_steps} ({warmup_percentage:.1%})")
        else:
            logger.info(f"  Warmup steps: {self.warmup_steps} (percentage unknown - streaming dataset)")

        logger.info(f"  Main training steps: {self.main_training_steps}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Adaptive LR: {'Enabled' if config.enable_adaptive else 'Disabled'}")

    def calculate_total_steps(
        self,
        num_epochs: int,
        dataset_size: Optional[int] = None,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1
    ) -> int:
        """
        Calculate total training steps based on actual dataset parameters.

        Args:
            num_epochs: Number of training epochs
            dataset_size: Size of training dataset
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Total number of optimizer steps
        """
        if dataset_size is None:
            # Fallback estimation
            estimated_dataset_size = 100000
            logger.warning(f"Dataset size unknown, estimating {estimated_dataset_size:,} samples")
            dataset_size = estimated_dataset_size

        # Calculate steps per epoch
        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = math.ceil(dataset_size / effective_batch_size)

        # Total steps
        total_steps = steps_per_epoch * num_epochs

        # Update internal state
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
        self.main_training_steps = total_steps - self.warmup_steps

        logger.info(f"Training schedule calculated:")
        logger.info(f"  Dataset size: {dataset_size:,} samples")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {self.warmup_steps:,} ({self.warmup_steps/total_steps:.1%})")

        return total_steps

    def get_lr(self, step: int) -> float:
        """
        Get learning rate for current step.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        self.current_step = step

        # Handle recovery mode
        if self.in_recovery:
            return self._get_recovery_lr(step)

        # Warmup phase
        if step < self.warmup_steps:
            return self._get_warmup_lr(step)

        # Main training phase
        return self._get_main_lr(step)

    def _get_warmup_lr(self, step: int) -> float:
        """Calculate warmup learning rate."""
        progress = step / self.warmup_steps
        initial_lr = self.initial_lrs[0]
        min_lr = initial_lr * self.config.warmup_min_ratio

        if self.config.warmup_schedule == "linear":
            lr = min_lr + (initial_lr - min_lr) * progress
        elif self.config.warmup_schedule == "cosine":
            lr = min_lr + (initial_lr - min_lr) * (1 - math.cos(math.pi * progress)) / 2
        elif self.config.warmup_schedule == "polynomial":
            lr = min_lr + (initial_lr - min_lr) * (progress ** 2)
        else:
            lr = initial_lr  # Fallback

        return lr

    def _get_main_lr(self, step: int) -> float:
        """Calculate main training learning rate."""
        # Progress through main training (after warmup)
        main_step = step - self.warmup_steps
        progress = main_step / max(self.main_training_steps, 1)
        progress = min(1.0, progress)  # Clamp to [0, 1]

        initial_lr = self.initial_lrs[0]
        min_lr = initial_lr * self.config.min_lr_ratio

        if self.config.main_schedule == "cosine":
            lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
        elif self.config.main_schedule == "linear_decay":
            lr = initial_lr - (initial_lr - min_lr) * progress
        elif self.config.main_schedule == "polynomial":
            lr = min_lr + (initial_lr - min_lr) * ((1 - progress) ** 2)
        elif self.config.main_schedule == "constant":
            lr = initial_lr
        else:
            lr = initial_lr  # Fallback

        return lr

    def _get_recovery_lr(self, step: int) -> float:
        """Calculate recovery learning rate after plateau reduction."""
        recovery_progress = (step - self.recovery_start_step) / self.config.recovery_warmup_steps
        recovery_progress = min(1.0, recovery_progress)

        # Warmup from reduced LR back to normal schedule
        current_normal_lr = self._get_main_lr(step)
        recovery_lr = self.pre_recovery_lr + (current_normal_lr - self.pre_recovery_lr) * recovery_progress

        if recovery_progress >= 1.0:
            self.in_recovery = False
            self.recovery_count += 1
            logger.info(f"LR recovery completed at step {step}")

        return recovery_lr

    def step(self, validation_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform LR schedule step.

        Args:
            validation_loss: Current validation loss for plateau detection

        Returns:
            Dictionary with step information
        """
        # Get current LR
        current_lr = self.get_lr(self.current_step)

        # Apply LR to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        # Track LR history
        self.lr_history.append(current_lr)

        # Handle adaptive LR (plateau detection)
        step_info = {'lr': current_lr, 'phase': self._get_current_phase()}

        if self.plateau_detector and validation_loss is not None:
            plateau_result = self.plateau_detector.step(validation_loss)

            if plateau_result['reduce_lr']:
                # Plateau detected - reduce LR
                reduced_lr = current_lr * self.config.plateau_factor
                reduced_lr = max(reduced_lr, self.config.plateau_min_lr)

                logger.info(f"Plateau detected at step {self.current_step}")
                logger.info(f"Reducing LR: {current_lr:.2e} -> {reduced_lr:.2e}")

                # Apply reduction immediately
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = reduced_lr

                # Start recovery if enabled
                if self.config.enable_lr_recovery:
                    self.in_recovery = True
                    self.recovery_start_step = self.current_step
                    self.pre_recovery_lr = reduced_lr

                self.reduction_count += 1
                step_info.update({
                    'plateau_detected': True,
                    'lr_reduced': True,
                    'reduction_factor': self.config.plateau_factor
                })

            step_info.update({
                'validation_loss': validation_loss,
                'plateau_patience': plateau_result['patience_remaining'],
                'best_loss': plateau_result['best_loss']
            })

        self.current_step += 1
        return step_info

    def _get_current_phase(self) -> str:
        """Get current training phase name."""
        if self.in_recovery:
            return "recovery"
        elif self.current_step < self.warmup_steps:
            return "warmup"
        else:
            return "main"

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive LR statistics."""
        recent_lrs = list(self.lr_history)[-100:]  # Last 100 steps

        return {
            'current_step': self.current_step,
            'current_lr': recent_lrs[-1] if recent_lrs else 0.0,
            'initial_lr': self.initial_lrs[0],
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'reduction_count': self.reduction_count,
            'recovery_count': self.recovery_count,
            'current_phase': self._get_current_phase(),
            'recent_lr_mean': sum(recent_lrs) / len(recent_lrs) if recent_lrs else 0.0,
            'recent_lr_std': (
                (sum((lr - sum(recent_lrs)/len(recent_lrs))**2 for lr in recent_lrs) / len(recent_lrs))**0.5
                if len(recent_lrs) > 1 else 0.0
            ),
            'adaptive_enabled': self.plateau_detector is not None,
            'in_recovery': self.in_recovery
        }


class PlateauDetector:
    """Detects training plateaus and triggers LR reductions."""

    def __init__(self, patience: int, threshold: float, factor: float, min_lr: float):
        self.patience = patience
        self.threshold = threshold
        self.factor = factor
        self.min_lr = min_lr

        self.best_loss = float('inf')
        self.patience_remaining = patience
        self.step_count = 0

    def step(self, loss: float) -> Dict[str, Any]:
        """
        Check for plateau and determine if LR should be reduced.

        Args:
            loss: Current validation loss

        Returns:
            Dictionary with plateau detection results
        """
        self.step_count += 1
        reduce_lr = False

        # Check if we have improvement
        if loss < self.best_loss - self.threshold:
            # Significant improvement
            self.best_loss = loss
            self.patience_remaining = self.patience
        else:
            # No significant improvement
            self.patience_remaining -= 1

            if self.patience_remaining <= 0:
                # Plateau detected
                reduce_lr = True
                self.patience_remaining = self.patience  # Reset for next plateau

        return {
            'reduce_lr': reduce_lr,
            'best_loss': self.best_loss,
            'patience_remaining': self.patience_remaining,
            'step_count': self.step_count
        }