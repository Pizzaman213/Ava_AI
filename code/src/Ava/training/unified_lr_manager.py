"""
Unified Learning Rate Management System

This module consolidates all learning rate functionality from:
- lr_manager.py
- adaptive_lr.py
- advanced_warmup.py
- advanced_warmup_scheduling.py
- advanced_schedulers.py
- lr_finder.py

Provides a single, unified interface for all LR management needs.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try to import scipy for advanced features
try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some smoothing features will be disabled")


# ============================================================================
# Enums and Constants
# ============================================================================

class WarmupScheduleType(Enum):
    """Available warmup schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


class MainScheduleType(Enum):
    """Available main schedule types."""
    COSINE = "cosine"
    LINEAR_DECAY = "linear_decay"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    COSINE_RESTARTS = "cosine_restarts"
    ONECYCLE = "onecycle"


# ============================================================================
# Unified Configuration
# ============================================================================

@dataclass
class UnifiedLRConfig:
    """
    Unified configuration for all learning rate management features.

    Consolidates settings from:
    - LRConfig (lr_manager.py)
    - AdaptiveLRConfig (adaptive_lr.py)
    - WarmupConfig (advanced_warmup.py)
    - LRFinderConfig (lr_finder.py)
    """

    # ========== Warmup Configuration ==========
    warmup_steps: Optional[int] = None          # Explicit warmup steps (overrides warmup_ratio)
    warmup_ratio: float = 0.03                   # Warmup ratio of total steps (3%)
    warmup_schedule: WarmupScheduleType = WarmupScheduleType.COSINE
    warmup_start_lr: float = 1e-8                # Starting LR for warmup
    warmup_min_ratio: float = 0.01               # Start at 1% of target LR
    warmup_power: float = 2.0                    # Power for polynomial warmup

    # Advanced warmup features
    use_gradient_norm_warmup: bool = False       # Enable gradient-adaptive warmup
    gradient_threshold: float = 1.0              # Gradient norm threshold for early completion
    warmup_restart_threshold: float = 3.0        # Loss spike threshold for warmup restart

    # ========== Main Schedule Configuration ==========
    main_schedule: MainScheduleType = MainScheduleType.COSINE
    min_lr_ratio: float = 0.01                   # Minimum LR as fraction of initial LR
    min_lr: float = 1e-8                         # Absolute minimum learning rate
    max_lr: float = 1e-3                         # Maximum allowed learning rate

    # ========== Adaptive/Plateau Detection ==========
    enable_adaptive: bool = False                # Enable adaptive LR adjustments
    plateau_patience: int = 500                  # Steps to wait before reducing LR on plateau
    plateau_threshold: float = 0.01              # Minimum improvement required
    plateau_factor: float = 0.5                  # Factor to reduce LR on plateau

    # ========== Loss Tracking ==========
    batch_loss_window: int = 100                 # Window size for loss averaging
    min_improvement: float = 0.001               # Minimum improvement threshold

    # ========== Spike Detection & Emergency Handling ==========
    divergence_threshold: float = 1.5            # Divergence threshold multiplier
    spike_threshold: float = 2.0                 # Emergency reduction threshold
    emergency_factor: float = 0.1                # Emergency reduction factor

    # ========== Stability-Based Adjustments ==========
    stability_threshold: int = 5                 # Consecutive improvements needed
    increase_factor: float = 1.1                 # Factor to increase LR when stable
    increase_min_gap: int = 1000                 # Minimum steps between increases

    # ========== Recovery & Restart ==========
    enable_lr_recovery: bool = True              # Enable LR recovery after reduction
    recovery_warmup_steps: int = 100             # Steps to warmup after LR reduction
    max_reductions: int = 5                      # Maximum number of LR reductions

    # ========== Progressive Training ==========
    enable_progressive_scaling: bool = False     # Scale LR based on sequence length
    progressive_scaling_method: str = "sqrt"     # "sqrt", "linear", "none"

    # ========== Cosine Restarts (SGDR) ==========
    restart_T_0: int = 1000                      # Initial restart period
    restart_T_mult: int = 2                      # Restart period multiplier
    restart_eta_max_mult: float = 1.0            # Max LR multiplier after restart
    restart_temperature: float = 1.0             # Temperature for exploration

    # ========== OneCycle Settings ==========
    onecycle_max_lr_mult: float = 10.0           # Max LR multiplier for onecycle
    onecycle_div_factor: float = 25.0            # Initial LR divisor
    onecycle_final_div_factor: float = 1e4       # Final LR divisor
    onecycle_pct_start: float = 0.3              # Percentage of cycle for LR increase

    # ========== LR Finder Configuration ==========
    lr_finder_start_lr: float = 1e-8             # LR finder starting LR
    lr_finder_end_lr: float = 1e-2               # LR finder ending LR
    lr_finder_num_iter: int = 1000               # LR finder iterations
    lr_finder_beta: float = 0.9                  # Loss smoothing factor
    lr_finder_suggestion_method: str = "fastai"  # "fastai", "steepest", "minimum", "valley"

    # ========== General Settings ==========
    gradient_accumulation_steps: int = 1         # Gradient accumulation steps
    lr_check_interval: int = 100                 # Interval to check LR adjustments
    verbose: bool = True                         # Enable logging


# ============================================================================
# Warmup Schedulers
# ============================================================================

class UnifiedWarmupScheduler:
    """
    Unified warmup scheduler consolidating all warmup implementations.

    Consolidates:
    - IntelligentLRManager warmup (lr_manager.py)
    - AdaptiveLearningRateManager warmup (adaptive_lr.py)
    - AdvancedWarmupScheduler (advanced_warmup.py)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: UnifiedLRConfig,
        total_steps: Optional[int] = None
    ):
        """
        Initialize unified warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Unified LR configuration
            total_steps: Total training steps (for ratio calculation)
        """
        self.optimizer = optimizer
        self.config = config

        # Store target LRs (after warmup)
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]

        # Calculate warmup steps
        if config.warmup_steps is not None:
            self.warmup_steps = config.warmup_steps
        elif total_steps is not None:
            self.warmup_steps = max(1, int(total_steps * config.warmup_ratio))
        else:
            self.warmup_steps = 1000  # Default fallback

        # Initialize state
        self.current_step = 0
        self.in_warmup = True
        self.warmup_completed = False
        self.warmup_restart_count = 0
        self.warmup_completion_step = None

        # Loss tracking for restart
        self.warmup_loss_baseline = None

        # Set initial LR to warmup start
        start_lr = config.warmup_start_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr

        # Statistics
        self.stats = {
            'total_restarts': 0,
            'early_completions': 0,
            'gradient_norm_completions': 0
        }

    def step(
        self,
        loss: Optional[float] = None,
        model: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Perform warmup step.

        Args:
            loss: Current loss for restart detection
            model: Model for gradient norm computation

        Returns:
            Dictionary with warmup information
        """
        if not self.in_warmup:
            return {'in_warmup': False, 'warmup_completed': True}

        # Check for warmup restart
        if loss is not None and self._should_restart_warmup(loss):
            self._restart_warmup()
            return {'in_warmup': True, 'warmup_restarted': True}

        # Check for gradient-based early completion
        if self.config.use_gradient_norm_warmup and model is not None:
            if self._check_gradient_completion(model):
                self._complete_warmup()
                self.stats['gradient_norm_completions'] += 1
                return {'in_warmup': False, 'early_completion': True}

        # Calculate warmup LR
        progress = min(1.0, self.current_step / self.warmup_steps)

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._calculate_warmup_lr(
                progress,
                self.config.warmup_start_lr,
                self.target_lrs[i]
            )

        self.current_step += 1

        # Check if warmup completed
        if self.current_step >= self.warmup_steps:
            self._complete_warmup()

        return {
            'in_warmup': True,
            'warmup_progress': progress,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }

    def _calculate_warmup_lr(self, progress: float, start_lr: float, target_lr: float) -> float:
        """Calculate LR for current warmup progress."""
        schedule = self.config.warmup_schedule

        if schedule == WarmupScheduleType.LINEAR:
            return start_lr + (target_lr - start_lr) * progress

        elif schedule == WarmupScheduleType.COSINE:
            return start_lr + (target_lr - start_lr) * (1 - math.cos(progress * math.pi)) / 2

        elif schedule == WarmupScheduleType.POLYNOMIAL:
            power = self.config.warmup_power
            return start_lr + (target_lr - start_lr) * (progress ** power)

        elif schedule == WarmupScheduleType.EXPONENTIAL:
            return start_lr * (target_lr / start_lr) ** progress

        else:
            return start_lr + (target_lr - start_lr) * progress

    def _should_restart_warmup(self, loss: float) -> bool:
        """Check if warmup should restart due to loss spike."""
        if self.warmup_loss_baseline is None:
            self.warmup_loss_baseline = loss
            return False

        threshold = self.config.warmup_restart_threshold
        if loss > self.warmup_loss_baseline * threshold:
            logger.warning(f"Loss spike detected: {loss:.4f} > {self.warmup_loss_baseline * threshold:.4f}")
            return True

        # Update baseline with exponential moving average
        self.warmup_loss_baseline = 0.9 * self.warmup_loss_baseline + 0.1 * loss
        return False

    def _restart_warmup(self):
        """Restart warmup from beginning."""
        logger.info("Restarting warmup...")
        self.current_step = 0
        self.warmup_restart_count += 1
        self.stats['total_restarts'] += 1

        # Reset to warmup start LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.warmup_start_lr

    def _check_gradient_completion(self, model: nn.Module) -> bool:
        """Check if gradients indicate warmup can complete early."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        return total_norm < self.config.gradient_threshold

    def _complete_warmup(self):
        """Complete warmup phase."""
        self.in_warmup = False
        self.warmup_completed = True
        self.warmup_completion_step = self.current_step

        # Set to target LRs
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.target_lrs[i]

        logger.info(f"Warmup completed at step {self.current_step}")


# ============================================================================
# Adaptive LR Manager
# ============================================================================

class AdaptiveLRManager:
    """
    Adaptive LR manager with plateau detection and stability-based adjustments.

    Consolidates:
    - IntelligentLRManager adaptive features (lr_manager.py)
    - AdaptiveLearningRateManager (adaptive_lr.py)
    """

    def __init__(self, optimizer: Optimizer, config: UnifiedLRConfig):
        """
        Initialize adaptive LR manager.

        Args:
            optimizer: PyTorch optimizer
            config: Unified LR configuration
        """
        self.optimizer = optimizer
        self.config = config

        # Loss tracking
        self.batch_losses = deque(maxlen=config.batch_loss_window)
        self.best_loss = float('inf')
        self.recent_best_loss = float('inf')

        # Counters
        self.step_count = 0
        self.batches_since_improvement = 0
        self.stable_improvement_count = 0
        self.lr_reductions = 0
        self.last_lr_reduction_step = 0
        self.last_lr_increase_step = 0

        # Spike handling
        self.lr_before_spike = None

        # Statistics
        self.stats = {
            'total_reductions': 0,
            'total_increases': 0,
            'emergency_reductions': 0,
            'plateau_reductions': 0,
            'stability_increases': 0
        }

    def step(self, loss: float) -> Dict[str, Any]:
        """
        Update adaptive LR based on loss.

        Args:
            loss: Current training loss

        Returns:
            Dictionary with LR adjustment information
        """
        if not self.config.enable_adaptive:
            return {'adaptive_enabled': False}

        self.batch_losses.append(loss)
        self.step_count += 1

        # Check for loss spike (emergency)
        if self._detect_loss_spike(loss):
            self._handle_emergency_reduction()
            return {'emergency_reduction': True}

        # Check for divergence
        if self._detect_divergence(loss):
            self._handle_divergence()
            return {'divergence_detected': True}

        # Check for plateau
        if self.step_count % self.config.lr_check_interval == 0:
            if self._detect_plateau():
                self._handle_plateau()
                return {'plateau_reduction': True}

            # Check for stability-based increase
            if self._detect_stable_improvement():
                self._handle_stability_increase()
                return {'stability_increase': True}

        # Update best loss tracking
        if loss < self.best_loss:
            self.best_loss = loss
            self.batches_since_improvement = 0
            self.stable_improvement_count += 1
        else:
            self.batches_since_improvement += 1
            self.stable_improvement_count = 0

        return {'status': 'normal'}

    def _detect_loss_spike(self, loss: float) -> bool:
        """Detect emergency loss spike."""
        if len(self.batch_losses) < 10:
            return False

        recent_avg = np.mean(list(self.batch_losses)[-10:])
        return loss > recent_avg * self.config.spike_threshold

    def _detect_divergence(self, loss: float) -> bool:
        """Detect loss divergence."""
        if len(self.batch_losses) < 20:
            return False

        recent_avg = np.mean(list(self.batch_losses))
        return loss > recent_avg * self.config.divergence_threshold

    def _detect_plateau(self) -> bool:
        """Detect learning plateau."""
        if len(self.batch_losses) < self.config.batch_loss_window:
            return False

        # Check if no improvement for plateau_patience steps
        if self.batches_since_improvement < self.config.plateau_patience:
            return False

        # Check if recent loss not improving
        recent_losses = list(self.batch_losses)[-self.config.batch_loss_window // 2:]
        recent_avg = np.mean(recent_losses)

        improvement = (self.recent_best_loss - recent_avg) / (self.recent_best_loss + 1e-8)

        if improvement < self.config.plateau_threshold:
            logger.info(f"Plateau detected: improvement={improvement:.6f} < {self.config.plateau_threshold}")
            return True

        self.recent_best_loss = min(self.recent_best_loss, recent_avg)
        return False

    def _detect_stable_improvement(self) -> bool:
        """Detect stable improvement for LR increase."""
        if self.stable_improvement_count < self.config.stability_threshold:
            return False

        # Don't increase too soon after last increase
        steps_since_increase = self.step_count - self.last_lr_increase_step
        if steps_since_increase < self.config.increase_min_gap:
            return False

        # Don't increase if already at max
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr >= self.config.max_lr:
            return False

        return True

    def _handle_emergency_reduction(self):
        """Handle emergency LR reduction."""
        logger.warning("Emergency LR reduction triggered!")

        # Store LR before spike for potential recovery
        if self.lr_before_spike is None:
            self.lr_before_spike = self.optimizer.param_groups[0]['lr']

        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * self.config.emergency_factor
            param_group['lr'] = max(new_lr, self.config.min_lr)

        self.stats['emergency_reductions'] += 1
        self.stats['total_reductions'] += 1

    def _handle_divergence(self):
        """Handle divergence detection."""
        logger.warning("Divergence detected, reducing LR")

        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * 0.5
            param_group['lr'] = max(new_lr, self.config.min_lr)

        self.stats['total_reductions'] += 1

    def _handle_plateau(self):
        """Handle plateau by reducing LR."""
        if self.lr_reductions >= self.config.max_reductions:
            logger.warning(f"Maximum LR reductions ({self.config.max_reductions}) reached")
            return

        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * self.config.plateau_factor
            param_group['lr'] = max(new_lr, self.config.min_lr)
            logger.info(f"Plateau reduction: LR {param_group['lr']:.2e} -> {new_lr:.2e}")

        self.lr_reductions += 1
        self.last_lr_reduction_step = self.step_count
        self.batches_since_improvement = 0
        self.stats['plateau_reductions'] += 1
        self.stats['total_reductions'] += 1

    def _handle_stability_increase(self):
        """Handle stability-based LR increase."""
        for param_group in self.optimizer.param_groups:
            new_lr = min(param_group['lr'] * self.config.increase_factor, self.config.max_lr)
            logger.info(f"Stability increase: LR {param_group['lr']:.2e} -> {new_lr:.2e}")
            param_group['lr'] = new_lr

        self.last_lr_increase_step = self.step_count
        self.stable_improvement_count = 0
        self.stats['stability_increases'] += 1
        self.stats['total_increases'] += 1


# ============================================================================
# Main Schedule Schedulers
# ============================================================================

class UnifiedMainScheduler(_LRScheduler):
    """
    Unified main scheduler supporting multiple schedule types.

    Consolidates:
    - Cosine annealing
    - Linear decay
    - Polynomial decay
    - Cosine restarts (SGDR)
    - OneCycle policy
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: UnifiedLRConfig,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        """
        Initialize unified main scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Unified LR configuration
            total_steps: Total training steps
            warmup_steps: Number of warmup steps (to adjust total_steps)
            last_epoch: Last epoch index
        """
        self.config = config
        self.total_steps = total_steps - warmup_steps
        self.warmup_steps = warmup_steps

        # Cosine restarts state
        if config.main_schedule == MainScheduleType.COSINE_RESTARTS:
            self.T_0 = config.restart_T_0
            self.T_mult = config.restart_T_mult
            self.T_cur = 0
            self.T_i = self.T_0
            self.cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        schedule = self.config.main_schedule

        if schedule == MainScheduleType.COSINE:
            return self._get_cosine_lr()
        elif schedule == MainScheduleType.LINEAR_DECAY:
            return self._get_linear_decay_lr()
        elif schedule == MainScheduleType.POLYNOMIAL:
            return self._get_polynomial_lr()
        elif schedule == MainScheduleType.CONSTANT:
            return self.base_lrs
        elif schedule == MainScheduleType.COSINE_RESTARTS:
            return self._get_cosine_restarts_lr()
        elif schedule == MainScheduleType.ONECYCLE:
            return self._get_onecycle_lr()
        else:
            return self.base_lrs

    def _get_cosine_lr(self):
        """Cosine annealing schedule."""
        lrs = []
        progress = self.last_epoch / max(1, self.total_steps)

        for base_lr in self.base_lrs:
            min_lr = base_lr * self.config.min_lr_ratio
            lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            lrs.append(lr)

        return lrs

    def _get_linear_decay_lr(self):
        """Linear decay schedule."""
        lrs = []
        progress = self.last_epoch / max(1, self.total_steps)

        for base_lr in self.base_lrs:
            min_lr = base_lr * self.config.min_lr_ratio
            lr = base_lr - (base_lr - min_lr) * progress
            lrs.append(lr)

        return lrs

    def _get_polynomial_lr(self):
        """Polynomial decay schedule."""
        lrs = []
        power = self.config.warmup_power
        progress = self.last_epoch / max(1, self.total_steps)

        for base_lr in self.base_lrs:
            min_lr = base_lr * self.config.min_lr_ratio
            lr = min_lr + (base_lr - min_lr) * ((1 - progress) ** power)
            lrs.append(lr)

        return lrs

    def _get_cosine_restarts_lr(self):
        """Cosine annealing with warm restarts (SGDR)."""
        lrs = []
        t = self.T_cur / self.T_i

        for base_lr in self.base_lrs:
            min_lr = base_lr * self.config.min_lr_ratio
            max_lr = base_lr * (self.config.restart_eta_max_mult ** self.cycle)
            lr = min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * t)) / 2
            lrs.append(lr)

        # Update restart state
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.cycle += 1
            self.T_i *= self.config.restart_T_mult

        return lrs

    def _get_onecycle_lr(self):
        """OneCycle learning rate policy."""
        lrs = []
        pct = self.last_epoch / max(1, self.total_steps)
        pct_start = self.config.onecycle_pct_start

        for base_lr in self.base_lrs:
            max_lr = base_lr * self.config.onecycle_max_lr_mult
            start_lr = base_lr / self.config.onecycle_div_factor
            end_lr = base_lr / self.config.onecycle_final_div_factor

            if pct < pct_start:
                # Increasing phase
                progress = pct / pct_start
                lr = start_lr + (max_lr - start_lr) * progress
            else:
                # Decreasing phase
                progress = (pct - pct_start) / (1 - pct_start)
                lr = max_lr - (max_lr - end_lr) * progress

            lrs.append(lr)

        return lrs


# ============================================================================
# Unified LR Manager (Main Interface)
# ============================================================================

class UnifiedLearningRateManager:
    """
    Main unified learning rate manager interface.

    This is the primary class that users should interact with.
    It orchestrates warmup, main scheduling, and adaptive adjustments.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: UnifiedLRConfig,
        total_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Initialize unified LR manager.

        Args:
            optimizer: PyTorch optimizer
            config: Unified LR configuration
            total_steps: Total training steps
            steps_per_epoch: Steps per epoch
        """
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        # Initialize warmup scheduler
        self.warmup_scheduler = UnifiedWarmupScheduler(optimizer, config, total_steps)

        # Initialize adaptive manager if enabled
        self.adaptive_manager = AdaptiveLRManager(optimizer, config) if config.enable_adaptive else None

        # Initialize main scheduler (will be used after warmup)
        if total_steps is not None:
            self.main_scheduler = UnifiedMainScheduler(
                optimizer,
                config,
                total_steps,
                warmup_steps=self.warmup_scheduler.warmup_steps
            )
        else:
            self.main_scheduler = None

        self.current_step = 0

    def step(self, loss: Optional[float] = None, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Perform LR update step.

        Args:
            loss: Current training loss
            model: Model for gradient-based features

        Returns:
            Dictionary with step information
        """
        self.current_step += 1
        info = {}

        # Warmup phase
        if self.warmup_scheduler.in_warmup:
            warmup_info = self.warmup_scheduler.step(loss, model)
            info.update(warmup_info)
        else:
            # Main scheduling phase
            if self.main_scheduler is not None:
                self.main_scheduler.step()
                info['main_scheduler_active'] = True

        # Adaptive adjustments (runs throughout training)
        if self.adaptive_manager is not None and loss is not None:
            adaptive_info = self.adaptive_manager.step(loss)
            info.update(adaptive_info)

        info['current_step'] = self.current_step
        info['current_lr'] = self.optimizer.param_groups[0]['lr']

        return info

    def get_last_lr(self) -> List[float]:
        """Get current learning rates for all param groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state = {
            'current_step': self.current_step,
            'warmup_scheduler': {
                'current_step': self.warmup_scheduler.current_step,
                'in_warmup': self.warmup_scheduler.in_warmup,
                'warmup_completed': self.warmup_scheduler.warmup_completed,
                'stats': self.warmup_scheduler.stats
            }
        }

        if self.adaptive_manager is not None:
            state['adaptive_manager'] = {
                'step_count': self.adaptive_manager.step_count,
                'best_loss': self.adaptive_manager.best_loss,
                'lr_reductions': self.adaptive_manager.lr_reductions,
                'stats': self.adaptive_manager.stats
            }

        if self.main_scheduler is not None:
            state['main_scheduler'] = self.main_scheduler.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        self.current_step = state['current_step']

        warmup_state = state['warmup_scheduler']
        self.warmup_scheduler.current_step = warmup_state['current_step']
        self.warmup_scheduler.in_warmup = warmup_state['in_warmup']
        self.warmup_scheduler.warmup_completed = warmup_state['warmup_completed']
        self.warmup_scheduler.stats = warmup_state['stats']

        if 'adaptive_manager' in state and self.adaptive_manager is not None:
            adaptive_state = state['adaptive_manager']
            self.adaptive_manager.step_count = adaptive_state['step_count']
            self.adaptive_manager.best_loss = adaptive_state['best_loss']
            self.adaptive_manager.lr_reductions = adaptive_state['lr_reductions']
            self.adaptive_manager.stats = adaptive_state['stats']

        if 'main_scheduler' in state and self.main_scheduler is not None:
            self.main_scheduler.load_state_dict(state['main_scheduler'])


# ============================================================================
# Convenience Functions
# ============================================================================

def create_lr_manager(
    optimizer: Optimizer,
    total_steps: Optional[int] = None,
    warmup_ratio: float = 0.03,
    main_schedule: str = "cosine",
    enable_adaptive: bool = False,
    **kwargs
) -> UnifiedLearningRateManager:
    """
    Convenience function to create a unified LR manager with common settings.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_ratio: Warmup ratio (default: 0.03 = 3%)
        main_schedule: Main schedule type ("cosine", "linear_decay", etc.)
        enable_adaptive: Enable adaptive LR adjustments
        **kwargs: Additional config parameters

    Returns:
        UnifiedLearningRateManager instance
    """
    config = UnifiedLRConfig(
        warmup_ratio=warmup_ratio,
        main_schedule=MainScheduleType(main_schedule),
        enable_adaptive=enable_adaptive,
        **kwargs
    )

    return UnifiedLearningRateManager(optimizer, config, total_steps)
