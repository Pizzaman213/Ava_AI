"""
Adaptive Learning Rate Manager - Real-time LR adaptation.

This module provides AdaptiveLearningRateManager for real-time monitoring
and adjustment of learning rates during training.

Features:
- Real-time loss divergence detection
- Plateau detection and LR reduction
- Stability-based LR increases
- Emergency spike handling
- Comprehensive tracking and logging
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

import torch
import torch.optim

from .configs import AdaptiveLRConfig

logger = logging.getLogger(__name__)


class AdaptiveLearningRateManager:
    """
    Adaptive learning rate manager with real-time monitoring and adjustment.

    Features:
    - Real-time loss divergence detection
    - Plateau detection and LR reduction
    - Stability-based LR increases
    - Emergency spike handling
    - Comprehensive tracking and logging
    """

    # Maximum entries to keep in lr_history/loss_history to prevent memory leak
    MAX_STATS_HISTORY = 1000

    def __init__(self, optimizer: torch.optim.Optimizer, config: AdaptiveLRConfig):
        """
        Initialize adaptive learning rate manager.

        Args:
            optimizer: PyTorch optimizer
            config: Adaptive LR configuration
        """
        self.optimizer = optimizer
        self.config = config

        # Store target LR for warmup
        self.target_lr = optimizer.param_groups[0]['lr']

        # Set initial LR to warmup start if warmup is enabled
        if config.warmup_steps > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.warmup_start_lr

        # Loss tracking - SEPARATED for training and validation
        self.batch_losses = deque(maxlen=config.batch_loss_window)
        self.best_training_loss = float('inf')  # Best training loss
        self.best_validation_loss = float('inf')  # Best validation loss
        self.recent_best_loss = float('inf')  # For spike detection

        # Counters and tracking
        self.step_count = 0
        self.batches_since_improvement = 0
        self.stable_improvement_count = 0
        self.lr_reductions = 0
        self.last_lr_reduction_step = 0
        self.last_lr_increase_step = 0

        # Spike handling
        self.lr_before_spike = None

        # Statistics
        # Use deques with maxlen to prevent unbounded memory growth
        # This fixes OOM (Killed) after ~20k steps due to RAM exhaustion
        self.lr_stats = {
            'total_reductions': 0,
            'total_increases': 0,
            'emergency_reductions': 0,
            'plateau_reductions': 0,
            'stability_increases': 0,
            'warmup_steps_completed': 0,
            'lr_history': deque(maxlen=self.MAX_STATS_HISTORY),
            'loss_history': deque(maxlen=self.MAX_STATS_HISTORY)
        }

    def step(self, loss: float, batch_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a training step and potentially adjust learning rate.

        Args:
            loss: Current batch loss
            batch_idx: Current batch index (optional)

        Returns:
            Dictionary with LR adjustment information
        """
        self.step_count += 1
        current_lr = self.optimizer.param_groups[0]['lr']

        # Add loss to tracking
        self.batch_losses.append(loss)

        # Update statistics
        self.lr_stats['lr_history'].append(current_lr)
        self.lr_stats['loss_history'].append(loss)

        # Get recent loss average
        if len(self.batch_losses) >= 10:  # Need minimum samples
            avg_recent_loss = sum(list(self.batch_losses)[-min(20, len(self.batch_losses)):]) / min(20, len(self.batch_losses))
        else:
            avg_recent_loss = loss

        # Determine current phase
        if self.config.warmup_steps > 0 and self.step_count <= self.config.warmup_steps:
            phase = "warmup"
        else:
            phase = "main"

        lr_info = {
            'step': self.step_count,
            'current_lr': current_lr,
            'current_loss': loss,
            'avg_recent_loss': avg_recent_loss,
            'best_training_loss': self.best_training_loss,
            'best_validation_loss': self.best_validation_loss,
            'lr_adjusted': False,
            'adjustment_type': None,
            'adjustment_reason': None,
            'phase': phase
        }

        # Emergency spike detection BEFORE warmup handling
        if self._detect_loss_spike(avg_recent_loss):
            adjustment = self._handle_loss_spike(avg_recent_loss)
            lr_info.update(adjustment)
            if self.config.warmup_steps > 0 and self.step_count <= self.config.warmup_steps:
                lr_info['emergency_during_warmup'] = True
            return lr_info

        # Handle warmup phase
        if self.config.warmup_steps > 0 and self.step_count <= self.config.warmup_steps:
            warmup_adjustment = self._handle_warmup()
            lr_info.update(warmup_adjustment)
            self.lr_stats['warmup_steps_completed'] = self.step_count
            return lr_info

        # Regular interval checks - USING TRAINING LOSS
        if self.step_count % self.config.lr_check_interval == 0:
            # FIX: Initialize best_training_loss on first check if still at infinity
            # This allows first batch to trigger improvement detection properly
            if self.best_training_loss == float('inf'):
                if len(self.batch_losses) > 0:
                    self.best_training_loss = avg_recent_loss
                    # Also initialize recent_best_loss for spike detection
                    self.recent_best_loss = avg_recent_loss

            # Check for improvement in training loss
            if avg_recent_loss < self.best_training_loss - self.config.min_improvement:
                adjustment = self._handle_improvement(avg_recent_loss)
                lr_info.update(adjustment)
            else:
                adjustment = self._handle_no_improvement(avg_recent_loss)
                lr_info.update(adjustment)

        return lr_info

    def _handle_warmup(self) -> Dict[str, Any]:
        """Handle learning rate warmup phase with linear scaling."""
        # Linear warmup from warmup_start_lr to target_lr
        progress = self.step_count / self.config.warmup_steps
        new_lr = self.config.warmup_start_lr + (self.target_lr - self.config.warmup_start_lr) * progress

        current_lr = self.optimizer.param_groups[0]['lr']

        # Update LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        return {
            'lr_adjusted': True,
            'adjustment_type': 'warmup',
            'adjustment_reason': f'Warmup step {self.step_count}/{self.config.warmup_steps}',
            'old_lr': current_lr,
            'new_lr': new_lr,
            'warmup_progress': progress
        }

    def _detect_loss_spike(self, current_loss: float) -> bool:
        """Detect if current loss represents a spike requiring immediate action."""
        if len(self.batch_losses) < 10:  # Need some history
            return False

        # Check against recent best
        if (self.recent_best_loss != float('inf') and
            current_loss > self.recent_best_loss * self.config.spike_threshold):
            return True

        # Check against overall best training loss
        if (self.best_training_loss != float('inf') and
            current_loss > self.best_training_loss * self.config.divergence_threshold and
            len(self.batch_losses) >= self.config.batch_loss_window // 2):
            return True

        return False

    def _handle_loss_spike(self, current_loss: float) -> Dict[str, Any]:
        """Handle detected loss spike with emergency LR reduction."""
        current_lr = self.optimizer.param_groups[0]['lr']

        # Store LR before spike for potential rollback
        if self.lr_before_spike is None:
            self.lr_before_spike = current_lr

        # Emergency reduction
        new_lr = max(current_lr * self.config.emergency_factor, self.config.min_lr)

        if new_lr < current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.lr_reductions += 1
            self.last_lr_reduction_step = self.step_count
            self.batches_since_improvement = 0

            self.lr_stats['total_reductions'] += 1
            self.lr_stats['emergency_reductions'] += 1

            return {
                'lr_adjusted': True,
                'adjustment_type': 'emergency_reduction',
                'adjustment_reason': f'Loss spike detected: {current_loss:.4f} >> {self.best_training_loss:.4f}',
                'old_lr': current_lr,
                'new_lr': new_lr
            }

        return {'lr_adjusted': False}

    def _handle_improvement(self, current_loss: float) -> Dict[str, Any]:
        """Handle detected improvement in training loss."""
        current_lr = self.optimizer.param_groups[0]['lr']
        improvement_ratio = (self.best_training_loss - current_loss) / max(self.best_training_loss, 0.001)

        # Update best training loss
        self.best_training_loss = current_loss
        self.batches_since_improvement = 0
        self.lr_before_spike = None  # Reset spike tracking on improvement

        # Track consecutive improvements for stability
        self.stable_improvement_count += 1

        # Allow LR increases after sufficient time since last reduction
        steps_since_reduction = self.step_count - self.last_lr_reduction_step

        # Consider increasing LR if training is very stable
        if (self.stable_improvement_count >= self.config.stability_threshold and
            improvement_ratio > 0.005 and  # At least 0.5% improvement
            steps_since_reduction > self.config.increase_min_gap and
            current_lr < self.config.max_lr and
            self.step_count - self.last_lr_increase_step > self.config.increase_min_gap):

            new_lr = min(current_lr * self.config.increase_factor, self.config.max_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            improvement_count = self.stable_improvement_count
            self.stable_improvement_count = 0  # Reset counter
            self.last_lr_increase_step = self.step_count

            self.lr_stats['total_increases'] += 1
            self.lr_stats['stability_increases'] += 1

            return {
                'lr_adjusted': True,
                'adjustment_type': 'stability_increase',
                'adjustment_reason': f'{improvement_count} consecutive improvements, {improvement_ratio*100:.2f}% better',
                'old_lr': current_lr,
                'new_lr': new_lr
            }

        # Update recent best for spike detection
        self.recent_best_loss = min(self.recent_best_loss, current_loss)

        return {
            'lr_adjusted': False,
            'improvement_ratio': improvement_ratio,
            'stable_count': self.stable_improvement_count
        }

    def _handle_no_improvement(self, current_loss: float) -> Dict[str, Any]:
        """Handle no improvement detected."""
        current_lr = self.optimizer.param_groups[0]['lr']

        # Increment no-improvement counter
        self.batches_since_improvement += self.config.lr_check_interval
        self.stable_improvement_count = 0  # Reset improvement counter

        # Update recent_best_loss to prevent stale values
        self.recent_best_loss = min(self.recent_best_loss, current_loss)

        # Add grace period after warmup before reducing LR
        # Allow 2x plateau_patience steps after warmup for model to start learning
        # FIX: plateau_patience is in batches (incremented by lr_check_interval),
        # so multiply by lr_check_interval to get equivalent step count
        grace_batches = 2 * self.config.plateau_patience
        grace_steps = grace_batches * self.config.lr_check_interval
        grace_period = self.config.warmup_steps + grace_steps
        in_post_warmup_grace = self.step_count < grace_period

        # Check for plateau (but not during post-warmup grace period)
        if (not in_post_warmup_grace and
            self.batches_since_improvement >= self.config.plateau_patience and
            self.step_count - self.last_lr_reduction_step > self.config.plateau_patience):

            new_lr = max(current_lr * self.config.plateau_factor, self.config.min_lr)

            if new_lr < current_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                self.lr_reductions += 1
                self.last_lr_reduction_step = self.step_count
                self.batches_since_improvement = 0

                self.lr_stats['total_reductions'] += 1
                self.lr_stats['plateau_reductions'] += 1

                plateau_warning = ""
                if new_lr <= self.config.min_lr:
                    plateau_warning = " (minimum LR reached)"
                elif self.lr_reductions >= self.config.max_reductions:
                    plateau_warning = f" (max reductions {self.config.max_reductions} reached)"

                return {
                    'lr_adjusted': True,
                    'adjustment_type': 'plateau_reduction',
                    'adjustment_reason': f'No improvement for {self.config.plateau_patience} batches{plateau_warning}',
                    'old_lr': current_lr,
                    'new_lr': new_lr
                }

        return {
            'lr_adjusted': False,
            'batches_since_improvement': self.batches_since_improvement
        }

    def update_validation_loss(self, val_loss: float) -> None:
        """
        Update manager with validation loss for tracking.

        Only updates validation loss tracking, does NOT affect training loss decisions.

        Args:
            val_loss: Current validation loss
        """
        # Update best validation loss (separate from training loss)
        if val_loss < self.best_validation_loss:
            self.best_validation_loss = val_loss

            # Track validation improvements
            if 'validation_improvements' not in self.lr_stats:
                self.lr_stats['validation_improvements'] = 0
            self.lr_stats['validation_improvements'] += 1

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning rate statistics."""
        return self.get_lr_statistics()

    def get_lr_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning rate statistics."""
        current_lr = self.get_current_lr()

        return {
            'current_lr': current_lr,
            'best_training_loss': self.best_training_loss,
            'best_validation_loss': self.best_validation_loss,
            'recent_best_loss': self.recent_best_loss,
            'batches_since_improvement': self.batches_since_improvement,
            'stable_improvement_count': self.stable_improvement_count,
            'lr_reductions': self.lr_reductions,
            'steps_since_last_reduction': self.step_count - self.last_lr_reduction_step,
            'steps_since_last_increase': self.step_count - self.last_lr_increase_step,
            'at_min_lr': current_lr <= self.config.min_lr,
            'at_max_lr': current_lr >= self.config.max_lr,
            'max_reductions_reached': self.lr_reductions >= self.config.max_reductions,
            'avg_recent_loss': sum(self.batch_losses) / len(self.batch_losses) if self.batch_losses else 0.0,
            'loss_trend': self._calculate_loss_trend(),
            'lr_stats': self.lr_stats
        }

    def _calculate_loss_trend(self) -> str:
        """Calculate current loss trend."""
        if len(self.batch_losses) < 20:
            return "insufficient_data"

        recent_losses = list(self.batch_losses)[-20:]
        first_half = sum(recent_losses[:10]) / 10
        second_half = sum(recent_losses[10:]) / 10

        if second_half < first_half * 0.95:
            return "improving"
        elif second_half > first_half * 1.05:
            return "worsening"
        else:
            return "stable"

    def reset_spike_tracking(self) -> None:
        """Reset spike tracking (useful after successful recovery)."""
        self.lr_before_spike = None
        self.recent_best_loss = self.best_training_loss

    def force_lr_reduction(self, factor: float = 0.5, reason: str = "manual") -> Dict[str, Any]:
        """Force learning rate reduction."""
        current_lr = self.get_current_lr()
        new_lr = max(current_lr * factor, self.config.min_lr)

        if new_lr < current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.lr_reductions += 1
            self.last_lr_reduction_step = self.step_count
            self.lr_stats['total_reductions'] += 1

            return {
                'lr_adjusted': True,
                'adjustment_type': 'forced_reduction',
                'adjustment_reason': reason,
                'old_lr': current_lr,
                'new_lr': new_lr
            }

        return {'lr_adjusted': False}

    def get_state_dict(self) -> Dict[str, Any]:
        """Get manager state for checkpointing."""
        # Convert deques to lists for serialization, keeping only last N entries
        lr_stats_serializable = {
            'total_reductions': self.lr_stats['total_reductions'],
            'total_increases': self.lr_stats['total_increases'],
            'emergency_reductions': self.lr_stats['emergency_reductions'],
            'plateau_reductions': self.lr_stats['plateau_reductions'],
            'stability_increases': self.lr_stats['stability_increases'],
            'warmup_steps_completed': self.lr_stats['warmup_steps_completed'],
            'lr_history': list(self.lr_stats['lr_history']),
            'loss_history': list(self.lr_stats['loss_history'])
        }
        # Preserve any extra keys that may have been added (e.g., validation_improvements)
        for key in self.lr_stats:
            if key not in lr_stats_serializable:
                lr_stats_serializable[key] = self.lr_stats[key]

        return {
            'batch_losses': list(self.batch_losses),
            'best_training_loss': self.best_training_loss,
            'best_validation_loss': self.best_validation_loss,
            'recent_best_loss': self.recent_best_loss,
            'step_count': self.step_count,
            'batches_since_improvement': self.batches_since_improvement,
            'stable_improvement_count': self.stable_improvement_count,
            'lr_reductions': self.lr_reductions,
            'last_lr_reduction_step': self.last_lr_reduction_step,
            'last_lr_increase_step': self.last_lr_increase_step,
            'lr_before_spike': self.lr_before_spike,
            'target_lr': self.target_lr,
            'lr_stats': lr_stats_serializable
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load manager state from checkpoint with backward compatibility."""
        self.batch_losses = deque(state_dict['batch_losses'], maxlen=self.config.batch_loss_window)

        # Backward compatibility - handle old checkpoints with 'best_loss'
        if 'best_training_loss' in state_dict:
            self.best_training_loss = state_dict['best_training_loss']
            self.best_validation_loss = state_dict.get('best_validation_loss', float('inf'))
        else:
            # Old checkpoint format - use 'best_loss' for both
            self.best_training_loss = state_dict.get('best_loss', float('inf'))
            self.best_validation_loss = state_dict.get('best_loss', float('inf'))

        self.recent_best_loss = state_dict['recent_best_loss']
        self.step_count = state_dict['step_count']
        self.batches_since_improvement = state_dict['batches_since_improvement']
        self.stable_improvement_count = state_dict['stable_improvement_count']
        self.lr_reductions = state_dict['lr_reductions']
        self.last_lr_reduction_step = state_dict['last_lr_reduction_step']
        self.last_lr_increase_step = state_dict['last_lr_increase_step']
        self.lr_before_spike = state_dict['lr_before_spike']
        self.target_lr = state_dict.get('target_lr', self.optimizer.param_groups[0]['lr'])

        # Reconstruct lr_stats with deques for lr_history and loss_history
        # This handles both old (list) and new (already list from serialization) formats
        loaded_stats = state_dict['lr_stats']
        self.lr_stats = {
            'total_reductions': loaded_stats.get('total_reductions', 0),
            'total_increases': loaded_stats.get('total_increases', 0),
            'emergency_reductions': loaded_stats.get('emergency_reductions', 0),
            'plateau_reductions': loaded_stats.get('plateau_reductions', 0),
            'stability_increases': loaded_stats.get('stability_increases', 0),
            'warmup_steps_completed': loaded_stats.get('warmup_steps_completed', 0),
            'lr_history': deque(loaded_stats.get('lr_history', []), maxlen=self.MAX_STATS_HISTORY),
            'loss_history': deque(loaded_stats.get('loss_history', []), maxlen=self.MAX_STATS_HISTORY)
        }
        # Preserve any extra keys (e.g., validation_improvements)
        for key in loaded_stats:
            if key not in self.lr_stats:
                self.lr_stats[key] = loaded_stats[key]


__all__ = ['AdaptiveLearningRateManager']
