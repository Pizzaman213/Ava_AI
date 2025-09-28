"""
Advanced Warmup System for Enhanced LLM Training

This module provides sophisticated warmup scheduling with multiple warmup types,
gradient-norm based completion, restart mechanisms, and adaptive parameter tuning.
"""

import torch
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class WarmupSchedule(Enum):
    """Available warmup schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


@dataclass
class WarmupConfig:
    """Configuration for advanced warmup system."""
    warmup_steps: int = 2000                    # Total warmup steps
    schedule: WarmupSchedule = WarmupSchedule.COSINE  # Warmup schedule type
    power: float = 2.0                          # Power for polynomial warmup
    start_ratio: float = 0.01                   # Starting LR ratio
    restart_threshold: float = 3.0              # Loss spike threshold for restart
    use_gradient_norm: bool = True              # Enable gradient-adaptive warmup
    gradient_threshold: float = 1.0             # Gradient norm threshold for completion
    decay_factor: float = 1.0                   # Decay factor after warmup
    decay_steps: int = 0                        # Steps to apply decay after warmup


class AdvancedWarmupScheduler:
    """
    Advanced warmup scheduler with multiple warmup types and adaptive features.

    Features:
    - Multiple warmup schedules (linear, cosine, polynomial, exponential)
    - Gradient-norm based early completion
    - Loss spike detection and warmup restart
    - Configurable warmup parameters
    - Decay scheduling after warmup completion
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: WarmupConfig):
        """
        Initialize advanced warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Warmup configuration
        """
        self.optimizer = optimizer
        self.config = config

        # Store initial learning rates
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.target_lrs = self.initial_lrs.copy()

        # Warmup state
        self.current_step = 0
        self.in_warmup = True
        self.warmup_completed = False
        self.warmup_restart_count = 0

        # Loss tracking for restart detection
        self.warmup_loss_baseline = None

        # Decay tracking
        self.decay_applied = False
        self.warmup_completion_step = None

        # Statistics
        self.warmup_stats = {
            'total_restarts': 0,
            'early_completions': 0,
            'gradient_norm_completions': 0
        }

    def step(self, loss: Optional[float] = None, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
        """
        Perform warmup step with optional loss and model for adaptive features.

        Args:
            loss: Current training loss (for restart detection)
            model: Model for gradient norm computation

        Returns:
            Dictionary with warmup information
        """
        if not self.in_warmup:
            return self._handle_post_warmup()

        # Check for warmup restart
        if loss is not None and self._should_restart_warmup(loss):
            self._restart_warmup()
            return self._get_warmup_info()

        # Calculate warmup progress
        progress = min(self.current_step / self.config.warmup_steps, 1.0)

        # Check for early completion
        if self.config.use_gradient_norm and model is not None:
            if self._should_complete_warmup_early(model):
                self._complete_warmup_early()
                return self._get_warmup_info()

        # Calculate and apply warmup learning rate
        warmup_lrs = self._calculate_warmup_lrs(progress)
        self._apply_learning_rates(warmup_lrs)

        # Update baseline loss
        if loss is not None:
            self._update_loss_baseline(loss)

        # Check for warmup completion
        if progress >= 1.0:
            self._complete_warmup()

        self.current_step += 1
        return self._get_warmup_info()

    def _calculate_warmup_lrs(self, progress: float) -> list:
        """Calculate learning rates based on warmup schedule and progress."""
        warmup_lrs = []

        for target_lr in self.target_lrs:
            lr_ratio = self._calculate_lr_ratio(progress)
            warmup_lr = target_lr * lr_ratio
            warmup_lrs.append(warmup_lr)

        return warmup_lrs

    def _calculate_lr_ratio(self, progress: float) -> float:
        """Calculate learning rate ratio based on warmup schedule."""
        start_ratio = self.config.start_ratio

        if self.config.schedule == WarmupSchedule.LINEAR:
            # Linear warmup: simple linear interpolation
            lr_ratio = start_ratio + (1.0 - start_ratio) * progress

        elif self.config.schedule == WarmupSchedule.COSINE:
            # Cosine warmup: smoother transition with cosine function
            lr_ratio = start_ratio + (1.0 - start_ratio) * (1 - math.cos(progress * math.pi)) / 2

        elif self.config.schedule == WarmupSchedule.POLYNOMIAL:
            # Polynomial warmup: configurable power for different curves
            lr_ratio = start_ratio + (1.0 - start_ratio) * (progress ** self.config.power)

        elif self.config.schedule == WarmupSchedule.EXPONENTIAL:
            # Exponential warmup: rapid initial increase, then slower
            lr_ratio = start_ratio + (1.0 - start_ratio) * (1 - math.exp(-progress * 3))

        else:
            # Default to linear if unknown schedule
            lr_ratio = start_ratio + (1.0 - start_ratio) * progress

        return lr_ratio

    def _apply_learning_rates(self, learning_rates: list) -> None:
        """Apply learning rates to optimizer parameter groups."""
        for param_group, lr in zip(self.optimizer.param_groups, learning_rates):
            param_group['lr'] = lr

    def _should_restart_warmup(self, current_loss: float) -> bool:
        """Check if warmup should be restarted due to loss spike."""
        if self.warmup_loss_baseline is None:
            return False

        # Restart if loss spikes beyond threshold
        if current_loss > self.warmup_loss_baseline * self.config.restart_threshold:
            print(f"   Warmup restart triggered: loss {current_loss:.4f} > {self.warmup_loss_baseline * self.config.restart_threshold:.4f}")
            return True

        return False

    def _restart_warmup(self) -> None:
        """Restart the warmup process."""
        self.in_warmup = True
        self.warmup_completed = False
        self.warmup_restart_count += 1
        self.current_step = 0
        self.warmup_stats['total_restarts'] += 1

        # Reset to initial warmup LR
        warmup_start_lrs = [lr * self.config.start_ratio for lr in self.target_lrs]
        self._apply_learning_rates(warmup_start_lrs)

        print(f"   Warmup restarted (#{self.warmup_restart_count}) - LR reset to {warmup_start_lrs[0]:.2e}")

    def _compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """Compute gradient norm for adaptive warmup completion."""
        try:
            total_norm = 0.0
            param_count = 0

            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm

        except Exception:
            pass

        return float('inf')

    def _should_complete_warmup_early(self, model: torch.nn.Module) -> bool:
        """Check if warmup should be completed early based on gradient norm."""
        if self.current_step < self.config.warmup_steps * 0.1:  # Don't complete too early
            return False

        gradient_norm = self._compute_gradient_norm(model)

        # Complete early if gradient norm is stable and below threshold
        if gradient_norm <= self.config.gradient_threshold:
            print(f"   Early warmup completion: gradient norm {gradient_norm:.4f} <= {self.config.gradient_threshold}")
            return True

        return False

    def _complete_warmup_early(self) -> None:
        """Complete warmup early due to gradient norm."""
        self.in_warmup = False
        self.warmup_completed = True
        self.warmup_completion_step = self.current_step
        self.warmup_stats['early_completions'] += 1
        self.warmup_stats['gradient_norm_completions'] += 1

        # Apply target learning rates
        self._apply_learning_rates(self.target_lrs)

        print(f"   Warmup completed early at step {self.current_step} (target: {self.config.warmup_steps})")

    def _complete_warmup(self) -> None:
        """Complete warmup normally."""
        self.in_warmup = False
        self.warmup_completed = True
        self.warmup_completion_step = self.current_step

        # Apply target learning rates
        self._apply_learning_rates(self.target_lrs)

        print(f"   Warmup completed at step {self.current_step}")

    def _update_loss_baseline(self, loss: float) -> None:
        """Update loss baseline for restart detection."""
        if self.warmup_loss_baseline is None:
            self.warmup_loss_baseline = loss
        else:
            # Exponential moving average
            alpha = 0.1
            self.warmup_loss_baseline = alpha * loss + (1 - alpha) * self.warmup_loss_baseline

    def _handle_post_warmup(self) -> Dict[str, Any]:
        """Handle post-warmup decay if configured."""
        if not self.decay_applied and self.config.decay_steps > 0:
            steps_since_warmup = self.current_step - self.warmup_completion_step
            if steps_since_warmup <= self.config.decay_steps:
                self._apply_warmup_decay(steps_since_warmup)
            else:
                self.decay_applied = True

        self.current_step += 1
        return self._get_warmup_info()

    def _apply_warmup_decay(self, steps_since_warmup: int) -> None:
        """Apply decay after warmup completion."""
        decay_progress = steps_since_warmup / self.config.decay_steps
        decay_factor = 1.0 - (1.0 - self.config.decay_factor) * decay_progress

        decayed_lrs = [lr * decay_factor for lr in self.target_lrs]
        self._apply_learning_rates(decayed_lrs)

    def _get_warmup_info(self) -> Dict[str, Any]:
        """Get current warmup information."""
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        return {
            'in_warmup': self.in_warmup,
            'warmup_completed': self.warmup_completed,
            'current_step': self.current_step,
            'target_steps': self.config.warmup_steps,
            'progress': min(self.current_step / self.config.warmup_steps, 1.0),
            'current_lrs': current_lrs,
            'target_lrs': self.target_lrs,
            'restart_count': self.warmup_restart_count,
            'schedule_type': self.config.schedule if isinstance(self.config.schedule, str) else self.config.schedule.value,
            'warmup_stats': self.warmup_stats
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'in_warmup': self.in_warmup,
            'warmup_completed': self.warmup_completed,
            'warmup_restart_count': self.warmup_restart_count,
            'warmup_loss_baseline': self.warmup_loss_baseline,
            'decay_applied': self.decay_applied,
            'warmup_completion_step': self.warmup_completion_step,
            'warmup_stats': self.warmup_stats,
            'target_lrs': self.target_lrs
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.in_warmup = state_dict['in_warmup']
        self.warmup_completed = state_dict['warmup_completed']
        self.warmup_restart_count = state_dict['warmup_restart_count']
        self.warmup_loss_baseline = state_dict['warmup_loss_baseline']
        self.decay_applied = state_dict['decay_applied']
        self.warmup_completion_step = state_dict['warmup_completion_step']
        self.warmup_stats = state_dict['warmup_stats']
        self.target_lrs = state_dict['target_lrs']

    def reset(self) -> None:
        """Reset warmup scheduler to initial state."""
        self.current_step = 0
        self.in_warmup = True
        self.warmup_completed = False
        self.warmup_restart_count = 0
        self.warmup_loss_baseline = None
        self.decay_applied = False
        self.warmup_completion_step = None
        self.warmup_stats = {
            'total_restarts': 0,
            'early_completions': 0,
            'gradient_norm_completions': 0
        }

    def set_target_learning_rates(self, target_lrs: list) -> None:
        """Update target learning rates."""
        self.target_lrs = target_lrs.copy()


# Convenience functions for creating common warmup configurations
def create_linear_warmup_config(warmup_steps: int = 2000, start_ratio: float = 0.01) -> WarmupConfig:
    """Create linear warmup configuration."""
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.LINEAR,
        start_ratio=start_ratio
    )


def create_cosine_warmup_config(warmup_steps: int = 2000, start_ratio: float = 0.01) -> WarmupConfig:
    """Create cosine warmup configuration."""
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.COSINE,
        start_ratio=start_ratio
    )


def create_polynomial_warmup_config(warmup_steps: int = 2000, power: float = 2.0, start_ratio: float = 0.01) -> WarmupConfig:
    """Create polynomial warmup configuration."""
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.POLYNOMIAL,
        power=power,
        start_ratio=start_ratio
    )


def create_adaptive_warmup_config(
    warmup_steps: int = 2000,
    gradient_threshold: float = 1.0,
    restart_threshold: float = 3.0,
    start_ratio: float = 0.01
) -> WarmupConfig:
    """Create adaptive warmup configuration with gradient-based completion."""
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.COSINE,
        start_ratio=start_ratio,
        use_gradient_norm=True,
        gradient_threshold=gradient_threshold,
        restart_threshold=restart_threshold
    )