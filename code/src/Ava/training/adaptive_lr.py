"""
Adaptive Learning Rate Manager

This module provides sophisticated learning rate management with real-time
loss monitoring, plateau detection, spike detection, and stability-based adjustments.
"""

import torch  # type: ignore[import]
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate management."""
    # Warmup configuration
    warmup_steps: int = 0                  # Number of warmup steps (0 = no warmup)
    warmup_start_lr: float = 1e-8          # Starting LR for warmup

    # Loss tracking
    batch_loss_window: int = 100           # Window size for loss averaging
    min_improvement: float = 0.001         # Minimum improvement threshold

    # Plateau detection
    plateau_patience: int = 500            # Batches to wait before LR reduction
    plateau_factor: float = 0.5            # Factor to reduce LR on plateau
    lr_check_interval: int = 100           # Check interval in batches

    # Spike detection
    divergence_threshold: float = 1.5      # Divergence threshold multiplier
    spike_threshold: float = 2.0           # Emergency reduction threshold
    emergency_factor: float = 0.1          # Emergency reduction factor

    # Stability-based increases
    stability_threshold: int = 5           # Consecutive improvements needed
    increase_factor: float = 1.1           # Factor to increase LR when stable
    max_lr: float = 1e-3                   # Maximum allowed learning rate
    increase_min_gap: int = 1000           # Minimum steps between increases

    # General limits
    min_lr: float = 1e-7                   # Minimum learning rate
    max_reductions: int = 5                # Maximum number of reductions


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
        self.lr_stats = {
            'total_reductions': 0,
            'total_increases': 0,
            'emergency_reductions': 0,
            'plateau_reductions': 0,
            'stability_increases': 0,
            'warmup_steps_completed': 0,
            'lr_history': [],
            'loss_history': []
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

        # FIXED: Emergency spike detection BEFORE warmup handling
        # This protects against divergence even during warmup
        if self._detect_loss_spike(avg_recent_loss):
            adjustment = self._handle_loss_spike(avg_recent_loss)
            lr_info.update(adjustment)
            # Mark as emergency during warmup for logging
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

        # FIXED: Allow LR increases after sufficient time since last reduction
        # This prevents permanent blocking after reductions
        steps_since_reduction = self.step_count - self.last_lr_reduction_step

        # Consider increasing LR if training is very stable
        if (self.stable_improvement_count >= self.config.stability_threshold and
            improvement_ratio > 0.005 and  # At least 0.5% improvement
            steps_since_reduction > self.config.increase_min_gap and  # Sufficient gap since reduction
            current_lr < self.config.max_lr and
            self.step_count - self.last_lr_increase_step > self.config.increase_min_gap):

            new_lr = min(current_lr * self.config.increase_factor, self.config.max_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # FIXED: Store count before resetting for accurate logging
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

        # FIXED: Update recent_best_loss to prevent stale values
        # This ensures spike detection uses current loss trajectory
        self.recent_best_loss = min(self.recent_best_loss, current_loss)

        # Check for plateau
        if (self.batches_since_improvement >= self.config.plateau_patience and
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

        FIXED: Only updates validation loss tracking, does NOT affect training loss decisions.
        This prevents validation loss from interfering with training loss plateau detection.

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
        """Get comprehensive learning rate statistics (alias for get_lr_statistics)."""
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
            'lr_stats': self.lr_stats
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load manager state from checkpoint with backward compatibility."""
        self.batch_losses = deque(state_dict['batch_losses'], maxlen=self.config.batch_loss_window)

        # FIXED: Backward compatibility - handle old checkpoints with 'best_loss'
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
        self.lr_stats = state_dict['lr_stats']


# Convenience functions for creating common configurations
def create_conservative_lr_config() -> AdaptiveLRConfig:
    """Create conservative LR management configuration."""
    return AdaptiveLRConfig(
        plateau_patience=1000,          # Wait longer before reducing
        plateau_factor=0.7,             # Smaller reductions
        divergence_threshold=2.0,       # Higher spike threshold
        emergency_factor=0.3,           # Less aggressive emergency reduction
        stability_threshold=8,          # More improvements needed for increase
        increase_factor=1.05            # Smaller increases
    )


def create_aggressive_lr_config() -> AdaptiveLRConfig:
    """Create aggressive LR management configuration."""
    return AdaptiveLRConfig(
        plateau_patience=250,           # Quick to reduce
        plateau_factor=0.3,             # Large reductions
        divergence_threshold=1.2,       # Lower spike threshold
        emergency_factor=0.05,          # Very aggressive emergency reduction
        stability_threshold=3,          # Few improvements needed for increase
        increase_factor=1.2             # Larger increases
    )


def create_balanced_lr_config() -> AdaptiveLRConfig:
    """Create balanced LR management configuration (default)."""
    return AdaptiveLRConfig()  # Use defaults