"""
Adaptive Learning Rate Management System

This module consolidates all core adaptive learning rate management functionality,
including real-time adaptation, warmup scheduling, plateau detection, and intelligent
LR coordination for optimal training.

This file replaces:
- adaptive_lr.py (AdaptiveLearningRateManager - real-time monitoring)
- lr_manager.py (IntelligentLRManager + PlateauDetector - high-level coordination)
- advanced_warmup.py (AdvancedWarmupScheduler - comprehensive warmup system)
"""

import torch  # type: ignore[import]
import torch.optim  # type: ignore[import]
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

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

    # Progressive training LR scaling (sequence length aware)
    enable_progressive_scaling: bool = True  # Scale LR based on sequence length
    progressive_scaling_method: str = "sqrt"  # "sqrt", "linear", "none"

    # Gradient accumulation awareness
    gradient_accumulation_steps: int = 1

    # Recovery configuration
    enable_lr_recovery: bool = True
    recovery_warmup_steps: int = 100  # Steps to warmup after LR reduction


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


# ============================================================================
# AdaptiveLearningRateManager - Real-time Adaptive LR Management
# ============================================================================

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
        self.lr_stats = state_dict['lr_stats']


# ============================================================================
# PlateauDetector - Validation-based Plateau Detection
# ============================================================================

class PlateauDetector:
    """Detects training plateaus and triggers LR reductions."""

    def __init__(self, patience: int, threshold: float, factor: float, min_lr: float,
                 min_checks_between_reductions: int = 5):
        """
        Initialize plateau detector.

        Args:
            patience: Number of validation checks without improvement before reducing LR
            threshold: Minimum improvement required
            factor: Factor to reduce LR by
            min_lr: Minimum learning rate
            min_checks_between_reductions: Minimum validation checks before allowing another reduction
        """
        self.patience = patience
        self.threshold = threshold
        self.factor = factor
        self.min_lr = min_lr
        self.min_checks_between_reductions = min_checks_between_reductions

        self.best_loss = float('inf')
        self.patience_remaining = patience
        self.validation_check_count = 0
        self.last_reduction_check = -min_checks_between_reductions
        self.checks_without_improvement = 0

    def step(self, loss: float, current_step: Optional[int] = None) -> Dict[str, Any]:
        """
        Check for plateau and determine if LR should be reduced.

        Args:
            loss: Current validation loss
            current_step: Current training step (for logging purposes only)

        Returns:
            Dictionary with plateau detection results
        """
        self.validation_check_count += 1
        reduce_lr = False

        # Check if enough validation checks have passed since last reduction
        checks_since_last_reduction = self.validation_check_count - self.last_reduction_check
        if checks_since_last_reduction < self.min_checks_between_reductions:
            return {
                'reduce_lr': False,
                'best_loss': self.best_loss,
                'patience_remaining': self.patience_remaining,
                'validation_checks': self.validation_check_count,
                'reason': f'Too soon since last reduction ({checks_since_last_reduction}/{self.min_checks_between_reductions} validation checks)'
            }

        # Check if we have improvement
        if loss < self.best_loss - self.threshold:
            # Significant improvement
            self.best_loss = loss
            self.patience_remaining = self.patience
            self.checks_without_improvement = 0
            logger.info(f"Plateau detector: Improvement detected at validation check {self.validation_check_count}, loss: {loss:.6f}")
        else:
            # No significant improvement
            self.checks_without_improvement += 1
            self.patience_remaining -= 1

            if self.patience_remaining <= 0:
                # Plateau detected
                reduce_lr = True
                self.patience_remaining = self.patience
                self.last_reduction_check = self.validation_check_count
                self.checks_without_improvement = 0
                step_info = f" (training step {current_step})" if current_step else ""
                logger.info(f"Plateau detected after {self.patience} validation checks without improvement{step_info}")

        return {
            'reduce_lr': reduce_lr,
            'best_loss': self.best_loss,
            'patience_remaining': self.patience_remaining,
            'validation_checks': self.validation_check_count,
            'checks_without_improvement': self.checks_without_improvement
        }


# ============================================================================
# IntelligentLRManager - High-level LR Coordinator
# ============================================================================

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
        steps_per_epoch: Optional[int] = None,
        warmup_steps: Optional[int] = None
    ):
        """
        Initialize LR manager.

        Args:
            optimizer: PyTorch optimizer
            config: LR configuration
            total_steps: Total training steps (calculated if None)
            steps_per_epoch: Steps per epoch for calculation
            warmup_steps: Explicit warmup steps (overrides config.warmup_ratio if provided)
        """
        self.optimizer = optimizer
        self.config = config
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.explicit_warmup_steps = warmup_steps

        # Calculate training schedule
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

        # Calculate warmup and main training steps
        if self.total_steps:
            if self.explicit_warmup_steps is not None:
                self.warmup_steps = self.explicit_warmup_steps
            else:
                self.warmup_steps = max(1, int(self.total_steps * config.warmup_ratio))
            self.main_training_steps = self.total_steps - self.warmup_steps
        else:
            self.warmup_steps = warmup_steps if warmup_steps is not None else 1000
            if config.warmup_ratio > 0:
                estimated_total = int(self.warmup_steps / config.warmup_ratio)
                self.main_training_steps = estimated_total - self.warmup_steps
            else:
                self.main_training_steps = self.warmup_steps * 20

        # Adaptive LR state
        self.plateau_detector = PlateauDetector(
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            factor=config.plateau_factor,
            min_lr=config.plateau_min_lr,
            min_checks_between_reductions=5
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
        if self.explicit_warmup_steps is not None:
            self.warmup_steps = self.explicit_warmup_steps
        else:
            self.warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
        self.main_training_steps = total_steps - self.warmup_steps

        logger.info(f"Training schedule calculated:")
        logger.info(f"  Dataset size: {dataset_size:,} samples")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {self.warmup_steps:,} ({self.warmup_steps/total_steps:.1%})")

        return total_steps

    def get_lr(self, step: int, current_seq_length: Optional[int] = None) -> float:
        """
        Get learning rate for current step.

        Args:
            step: Current training step
            current_seq_length: Current sequence length for progressive scaling

        Returns:
            Learning rate for this step
        """
        self.current_step = step

        # Handle recovery mode
        if self.in_recovery:
            base_lr = self._get_recovery_lr(step)
        # Warmup phase
        elif step < self.warmup_steps:
            base_lr = self._get_warmup_lr(step)
        # Main training phase
        else:
            base_lr = self._get_main_lr(step)

        # Apply progressive scaling if enabled
        if self.config.enable_progressive_scaling and current_seq_length is not None:
            base_lr = self._apply_progressive_scaling(base_lr, current_seq_length)

        return base_lr

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
            lr = initial_lr

        return lr

    def _get_main_lr(self, step: int) -> float:
        """Calculate main training learning rate."""
        main_step = step - self.warmup_steps
        progress = main_step / max(self.main_training_steps, 1)
        progress = min(1.0, progress)

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
            lr = initial_lr

        return lr

    def _get_recovery_lr(self, step: int) -> float:
        """Calculate recovery learning rate after plateau reduction."""
        recovery_progress = (step - self.recovery_start_step) / self.config.recovery_warmup_steps
        recovery_progress = min(1.0, recovery_progress)

        current_normal_lr = self._get_main_lr(step)
        recovery_lr = (self.pre_recovery_lr or 0.0) + (current_normal_lr - (self.pre_recovery_lr or 0.0)) * recovery_progress

        if recovery_progress >= 1.0:
            self.in_recovery = False
            self.recovery_count += 1
            logger.info(f"LR recovery completed at step {step}")

        return recovery_lr

    def _apply_progressive_scaling(self, base_lr: float, current_seq_length: int) -> float:
        """
        Apply progressive LR scaling based on current sequence length.

        Args:
            base_lr: Base learning rate
            current_seq_length: Current sequence length

        Returns:
            Scaled learning rate
        """
        if self.config.progressive_scaling_method == "sqrt":
            reference_length = 512
            scale_factor = math.sqrt(reference_length / max(current_seq_length, 1))
            scaled_lr = base_lr * scale_factor
        elif self.config.progressive_scaling_method == "linear":
            reference_length = 512
            scale_factor = reference_length / max(current_seq_length, 1)
            scaled_lr = base_lr * scale_factor
        else:
            scaled_lr = base_lr

        return scaled_lr

    def step(self, validation_loss: Optional[float] = None, training_step: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform LR schedule step.

        Args:
            validation_loss: Current validation loss for plateau detection
            training_step: Current training step (if provided, updates current_step)

        Returns:
            Dictionary with step information
        """
        if training_step is not None:
            self.current_step = training_step

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
            plateau_result = self.plateau_detector.step(validation_loss, current_step=self.current_step)

            if plateau_result['reduce_lr']:
                reduced_lr = current_lr * self.config.plateau_factor
                reduced_lr = max(reduced_lr, self.config.plateau_min_lr)

                logger.info(f"Plateau detected at step {self.current_step}")
                logger.info(f"Reducing LR: {current_lr:.2e} -> {reduced_lr:.2e}")

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = reduced_lr

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
        recent_lrs = list(self.lr_history)[-100:]

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


# ============================================================================
# AdvancedWarmupScheduler - Comprehensive Warmup System
# ============================================================================

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


# ============================================================================
# Convenience Functions
# ============================================================================

def create_conservative_lr_config() -> AdaptiveLRConfig:
    """Create conservative LR management configuration."""
    return AdaptiveLRConfig(
        plateau_patience=1000,
        plateau_factor=0.7,
        divergence_threshold=2.0,
        emergency_factor=0.3,
        stability_threshold=8,
        increase_factor=1.05
    )


def create_aggressive_lr_config() -> AdaptiveLRConfig:
    """Create aggressive LR management configuration."""
    return AdaptiveLRConfig(
        plateau_patience=250,
        plateau_factor=0.3,
        divergence_threshold=1.2,
        emergency_factor=0.05,
        stability_threshold=3,
        increase_factor=1.2
    )


def create_balanced_lr_config() -> AdaptiveLRConfig:
    """Create balanced LR management configuration (default)."""
    return AdaptiveLRConfig()


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
