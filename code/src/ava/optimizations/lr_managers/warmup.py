"""
Advanced Warmup Scheduler - Comprehensive warmup system.

This module provides AdvancedWarmupScheduler for sophisticated warmup
scheduling with multiple schedule types and adaptive features.

Features:
- Multiple warmup schedules (linear, cosine, polynomial, exponential)
- Gradient-norm based early completion
- Loss spike detection and warmup restart
- Configurable warmup parameters
- Decay scheduling after warmup completion
"""

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configs import WarmupConfig, WarmupSchedule

logger = logging.getLogger(__name__)


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

    def step(self, loss: Optional[float] = None, model: Optional[nn.Module] = None) -> Dict[str, Any]:
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

    def _calculate_warmup_lrs(self, progress: float) -> List[float]:
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
            lr_ratio = start_ratio + (1.0 - start_ratio) * progress
        elif self.config.schedule == WarmupSchedule.COSINE:
            lr_ratio = start_ratio + (1.0 - start_ratio) * (1 - math.cos(progress * math.pi)) / 2
        elif self.config.schedule == WarmupSchedule.POLYNOMIAL:
            lr_ratio = start_ratio + (1.0 - start_ratio) * (progress ** self.config.power)
        elif self.config.schedule == WarmupSchedule.EXPONENTIAL:
            lr_ratio = start_ratio + (1.0 - start_ratio) * (1 - math.exp(-progress * 3))
        else:
            lr_ratio = start_ratio + (1.0 - start_ratio) * progress

        return lr_ratio

    def _apply_learning_rates(self, learning_rates: List[float]) -> None:
        """Apply learning rates to optimizer parameter groups."""
        for param_group, lr in zip(self.optimizer.param_groups, learning_rates):
            param_group['lr'] = lr

    def _should_restart_warmup(self, current_loss: float) -> bool:
        """Check if warmup should be restarted due to loss spike."""
        if self.warmup_loss_baseline is None:
            return False

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

        warmup_start_lrs = [lr * self.config.start_ratio for lr in self.target_lrs]
        self._apply_learning_rates(warmup_start_lrs)

        print(f"   Warmup restarted (#{self.warmup_restart_count}) - LR reset to {warmup_start_lrs[0]:.2e}")

    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """
        Compute gradient norm for adaptive warmup completion.

        Accumulates norms on GPU with single .item() call at the end
        to avoid synchronization overhead.
        """
        try:
            # Collect all gradient norms on GPU first
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.data.norm(2))

            if grad_norms:
                # Stack and compute total norm on GPU - single sync at .item()
                stacked_norms = torch.stack(grad_norms)
                total_norm = torch.sqrt((stacked_norms ** 2).sum())
                return total_norm.item()  # Single cudaStreamSynchronize here

        except Exception as e:
            logger.debug(f"Gradient norm computation failed: {e}")

        return float('inf')

    def _should_complete_warmup_early(self, model: nn.Module) -> bool:
        """Check if warmup should be completed early based on gradient norm."""
        if self.current_step < self.config.warmup_steps * 0.1:
            return False

        gradient_norm = self._compute_gradient_norm(model)

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

        self._apply_learning_rates(self.target_lrs)

        print(f"   Warmup completed early at step {self.current_step} (target: {self.config.warmup_steps})")

    def _complete_warmup(self) -> None:
        """Complete warmup normally."""
        self.in_warmup = False
        self.warmup_completed = True
        self.warmup_completion_step = self.current_step

        self._apply_learning_rates(self.target_lrs)

        print(f"   Warmup completed at step {self.current_step}")

    def _update_loss_baseline(self, loss: float) -> None:
        """Update loss baseline for restart detection."""
        if self.warmup_loss_baseline is None:
            self.warmup_loss_baseline = loss
        else:
            alpha = 0.1
            self.warmup_loss_baseline = alpha * loss + (1 - alpha) * self.warmup_loss_baseline

    def _handle_post_warmup(self) -> Dict[str, Any]:
        """Handle post-warmup decay if configured."""
        if not self.decay_applied and self.config.decay_steps > 0 and self.warmup_completion_step is not None:
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

    def set_target_learning_rates(self, target_lrs: List[float]) -> None:
        """Update target learning rates."""
        self.target_lrs = target_lrs.copy()


__all__ = ['AdvancedWarmupScheduler']
