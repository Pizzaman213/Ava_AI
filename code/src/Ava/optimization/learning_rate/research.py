"""
Research & Experimental Learning Rate Schedulers

This module contains advanced learning rate scheduling strategies from research papers.
These schedulers are optional and provided for experimentation and research purposes.

For production training, use the adaptive_manager.py module instead.

Schedulers included:
- Cosine Annealing with Warm Restarts (SGDR) - https://arxiv.org/abs/1608.03983
- OneCycle Learning Rate Policy - https://arxiv.org/abs/1708.07120
- Polynomial Decay with Warmup
- Adaptive Learning Rate Scheduling (performance-based)
- Noisy Student Scheduling
- SchedulerFactory for easy instantiation

References:
- SGDR: https://arxiv.org/abs/1608.03983
- OneCycle: https://arxiv.org/abs/1708.07120
- Super-convergence: https://arxiv.org/abs/1506.01186
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from torch.optim.lr_scheduler import _LRScheduler  # type: ignore[import]
import math
import numpy as np  # type: ignore[import]
from typing import Dict, List, Optional, Any, Union
import logging
from collections import deque

logger = logging.getLogger(__name__)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Enhanced Cosine Annealing with Warm Restarts (SGDR).

    This scheduler implements cosine annealing with periodic restarts,
    which helps escape local minima and achieve better convergence.

    Features:
    - Periodic restarts with exponentially increasing cycles
    - Temperature scaling for exploration
    - Momentum annealing coordination
    - Multi-cycle convergence tracking

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_mult: Factor for increasing cycle length (default: 2)
        eta_min: Minimum learning rate (default: 0)
        eta_max_mult: Multiplier for max LR after restart (default: 1.0)
        temperature: Temperature for exploration enhancement (default: 1.0)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 0,
        eta_max_mult: float = 1.0,
        temperature: float = 1.0,
        momentum_annealing: bool = True,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max_mult = eta_max_mult
        self.temperature = temperature
        self.momentum_annealing = momentum_annealing

        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0

        # Store base values
        self.base_momentum = []
        for group in optimizer.param_groups:
            if 'momentum' in group:
                self.base_momentum.append(group['momentum'])
            else:
                self.base_momentum.append(None)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        lrs = []

        for base_lr in self.base_lrs:
            # Current position in cycle
            t = self.T_cur / self.T_i

            # Cosine annealing formula with temperature
            eta_t = self.eta_min + (base_lr * (self.eta_max_mult ** self.cycle) - self.eta_min) * \
                    (1 + math.cos(math.pi * t)) / 2

            # Apply temperature scaling
            if self.temperature != 1.0:
                eta_t = eta_t * (self.temperature ** t)

            lrs.append(eta_t)

        return lrs

    def get_momentum(self):
        """Compute momentum for current step (if applicable)."""
        if not self.momentum_annealing:
            return [group.get('momentum', None) for group in self.optimizer.param_groups]

        momentums = []
        t = self.T_cur / self.T_i

        for base_momentum in self.base_momentum:
            if base_momentum is not None:
                # Inverse cosine annealing for momentum
                momentum = base_momentum - (base_momentum - 0.85) * (1 + math.cos(math.pi * t)) / 2
                momentums.append(momentum)
            else:
                momentums.append(None)

        return momentums

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.T_cur += 1

        # Check for restart
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i *= self.T_mult

            logger.info(f"SGDR Restart: Cycle {self.cycle}, Next cycle length: {self.T_i}")

        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        # Update momentum if applicable
        if self.momentum_annealing:
            for param_group, momentum in zip(self.optimizer.param_groups, self.get_momentum()):
                if momentum is not None:
                    param_group['momentum'] = momentum


class OneCycleLR(_LRScheduler):
    """
    Enhanced One Cycle Learning Rate Policy.

    This scheduler implements the "super-convergence" approach with one cycle
    of learning rate and momentum scheduling for extremely fast convergence.

    Features:
    - Coordinated LR and momentum annealing
    - Three-phase training (warmup, annealing, fine-tuning)
    - Automatic phase transition
    - Performance-based adjustments

    Args:
        optimizer: Wrapped optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of cycle for warmup (default: 0.3)
        anneal_strategy: Annealing strategy ('cos' or 'linear')
        cycle_momentum: Whether to cycle momentum
        base_momentum: Lower momentum boundary
        max_momentum: Upper momentum boundary
        div_factor: Initial LR divisor (default: 25)
        final_div_factor: Final LR divisor (default: 1e4)
        three_phase: Whether to use three-phase training
    """

    def __init__(
        self,
        optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.85,
        max_momentum: Union[float, List[float]] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1
    ):
        # Validate parameters and ensure lists
        if not isinstance(max_lr, (list, tuple)):
            max_lrs_list: List[float] = [float(max_lr)] * len(optimizer.param_groups)
        else:
            max_lrs_list = [float(lr) for lr in max_lr]

        if not isinstance(base_momentum, (list, tuple)):
            base_momentum_list: List[float] = [float(base_momentum)] * len(optimizer.param_groups)
        else:
            base_momentum_list = [float(m) for m in base_momentum]

        if not isinstance(max_momentum, (list, tuple)):
            max_momentum_list: List[float] = [float(max_momentum)] * len(optimizer.param_groups)
        else:
            max_momentum_list = [float(m) for m in max_momentum]

        self.max_lrs: List[float] = max_lrs_list
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum: List[float] = base_momentum_list
        self.max_momentum: List[float] = max_momentum_list
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        # Calculate phase boundaries
        if three_phase:
            self.phase1_steps = int(total_steps * pct_start)
            self.phase2_steps = int(total_steps * 0.6)  # 60% for main training
            self.phase3_steps = total_steps - self.phase1_steps - self.phase2_steps
        else:
            self.phase1_steps = int(total_steps * pct_start)
            self.phase2_steps = total_steps - self.phase1_steps
            self.phase3_steps = 0

        # Calculate initial learning rates
        self.initial_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.min_lrs = [max_lr / final_div_factor for max_lr in self.max_lrs]

        super().__init__(optimizer, last_epoch)

    def _annealing_function(self, start: float, end: float, pct: float) -> float:
        """Apply annealing function."""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * pct)) / 2
        elif self.anneal_strategy == 'linear':
            return start + pct * (end - start)
        else:
            raise ValueError(f"Unknown annealing strategy: {self.anneal_strategy}")

    def get_lr(self):
        """Compute current learning rates."""
        step_num = self.last_epoch

        if step_num <= self.phase1_steps:
            # Phase 1: Warmup
            pct = step_num / self.phase1_steps
            return [self._annealing_function(initial_lr, max_lr, pct)
                    for initial_lr, max_lr in zip(self.initial_lrs, self.max_lrs)]

        elif step_num <= self.phase1_steps + self.phase2_steps:
            # Phase 2: Annealing
            pct = (step_num - self.phase1_steps) / self.phase2_steps
            if self.three_phase:
                # Anneal to intermediate level
                target_lrs = [max_lr * 0.1 for max_lr in self.max_lrs]
            else:
                # Anneal to minimum
                target_lrs = self.min_lrs

            return [self._annealing_function(max_lr, target_lr, pct)
                    for max_lr, target_lr in zip(self.max_lrs, target_lrs)]

        else:
            # Phase 3: Fine-tuning (three-phase only)
            pct = (step_num - self.phase1_steps - self.phase2_steps) / self.phase3_steps
            start_lrs = [max_lr * 0.1 for max_lr in self.max_lrs]
            return [self._annealing_function(start_lr, min_lr, pct)
                    for start_lr, min_lr in zip(start_lrs, self.min_lrs)]

    def get_momentum(self):
        """Compute current momentum values."""
        if not self.cycle_momentum:
            return [group.get('momentum', None) for group in self.optimizer.param_groups]

        step_num = self.last_epoch

        if step_num <= self.phase1_steps:
            # Phase 1: Decrease momentum as LR increases
            pct = step_num / self.phase1_steps
            return [self._annealing_function(max_mom, base_mom, pct)
                    for base_mom, max_mom in zip(self.base_momentum, self.max_momentum)]

        elif step_num <= self.phase1_steps + self.phase2_steps:
            # Phase 2: Increase momentum as LR decreases
            pct = (step_num - self.phase1_steps) / self.phase2_steps
            if self.three_phase:
                target_momentum = [(base_mom + max_mom) / 2 for base_mom, max_mom in
                                 zip(self.base_momentum, self.max_momentum)]
            else:
                target_momentum = self.max_momentum

            return [self._annealing_function(base_mom, target_mom, pct)
                    for base_mom, target_mom in zip(self.base_momentum, target_momentum)]

        else:
            # Phase 3: Fine-tuning momentum
            pct = (step_num - self.phase1_steps - self.phase2_steps) / self.phase3_steps
            start_momentum = [(base_mom + max_mom) / 2 for base_mom, max_mom in
                            zip(self.base_momentum, self.max_momentum)]
            return [self._annealing_function(start_mom, max_mom, pct)
                    for start_mom, max_mom in zip(start_momentum, self.max_momentum)]

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        # Update momentum
        if self.cycle_momentum:
            for param_group, momentum in zip(self.optimizer.param_groups, self.get_momentum()):
                if momentum is not None and 'momentum' in param_group:
                    param_group['momentum'] = momentum


class PolynomialDecayLR(_LRScheduler):
    """
    Polynomial learning rate decay with warmup.

    Provides smooth learning rate transitions with configurable polynomial decay.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        lr_end: Final learning rate
        power: Polynomial power (default: 1.0 for linear)
        cycle: Whether to cycle the schedule
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_end: float = 0.0,
        power: float = 1.0,
        cycle: bool = False,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_end = lr_end
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rates."""
        step_num = self.last_epoch

        if step_num < self.warmup_steps:
            # Linear warmup
            return [base_lr * step_num / self.warmup_steps for base_lr in self.base_lrs]

        # Polynomial decay
        if self.cycle:
            # Cycling polynomial decay
            cycle_len = self.total_steps - self.warmup_steps
            step_in_cycle = (step_num - self.warmup_steps) % cycle_len
            progress = step_in_cycle / cycle_len
        else:
            # Standard polynomial decay
            progress = min((step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps), 1.0)

        return [
            self.lr_end + (base_lr - self.lr_end) * (1 - progress) ** self.power
            for base_lr in self.base_lrs
        ]


class AdaptiveLRScheduler(_LRScheduler):
    """
    Performance-based adaptive learning rate scheduler.

    Automatically adjusts learning rate based on training dynamics including
    loss plateaus, gradient norms, and training stability metrics.

    Features:
    - Loss plateau detection with configurable patience
    - Gradient norm monitoring
    - Learning rate recovery mechanisms
    - Automatic scaling factor adjustment

    Args:
        optimizer: Wrapped optimizer
        patience: Steps to wait before reducing LR
        factor: Factor to reduce LR by
        threshold: Threshold for measuring improvement
        cooldown: Cooldown period after LR reduction
        min_lr: Minimum learning rate
        mode: 'min' for loss, 'max' for accuracy
        gradient_clip_threshold: Threshold for gradient explosion detection
        recovery_factor: Factor for LR recovery
    """

    def __init__(
        self,
        optimizer,
        patience: int = 10,
        factor: float = 0.5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        mode: str = 'min',
        gradient_clip_threshold: float = 1.0,
        recovery_factor: float = 1.5,
        last_epoch: int = -1
    ):
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.mode = mode
        self.gradient_clip_threshold = gradient_clip_threshold
        self.recovery_factor = recovery_factor

        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # State tracking
        self.best_metric = None
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.gradient_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=50)
        self.lr_reductions = 0

        super().__init__(optimizer, last_epoch)

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best - self.threshold
        else:
            return current > best + self.threshold

    def _detect_gradient_explosion(self, grad_norm: float) -> bool:
        """Detect gradient explosion."""
        if len(self.gradient_history) == 0:
            return False

        # Check if gradient norm is significantly higher than recent history
        recent_avg = float(np.mean(list(self.gradient_history)[-10:]))
        return bool(grad_norm > recent_avg * 3.0 and grad_norm > self.gradient_clip_threshold)

    def _detect_learning_rate_recovery_opportunity(self) -> bool:
        """Detect if learning rate can be recovered."""
        if len(self.loss_history) < 20 or self.lr_reductions == 0:
            return False

        # Check if loss has been steadily improving
        recent_losses = list(self.loss_history)[-10:]
        if len(recent_losses) >= 5:
            # Check for consistent improvement
            improvements = sum(
                1 for i in range(1, len(recent_losses))
                if recent_losses[i] < recent_losses[i-1]
            )
            return improvements >= len(recent_losses) * 0.7

        return False

    def step(self, metrics: Dict[str, float]):  # type: ignore[override]
        """
        Step the scheduler with current metrics.

        Args:
            metrics: Dictionary containing 'loss', 'grad_norm', and optionally other metrics
        """
        current_metric = metrics.get('loss' if self.mode == 'min' else 'accuracy')
        grad_norm = metrics.get('grad_norm', 0.0)

        if current_metric is None:
            logger.warning("Required metric not found in metrics dictionary")
            return

        # Update history
        self.loss_history.append(metrics.get('loss', 0.0))
        self.gradient_history.append(grad_norm)

        # Handle gradient explosion
        if self._detect_gradient_explosion(grad_norm):
            logger.warning(f"Gradient explosion detected (norm: {grad_norm:.2f}). Reducing LR.")
            self._reduce_lr(emergency=True)
            return

        # Check for LR recovery opportunity
        if self._detect_learning_rate_recovery_opportunity():
            logger.info("LR recovery opportunity detected. Increasing LR.")
            self._increase_lr()
            return

        # Standard plateau detection
        if self.best_metric is None:
            self.best_metric = current_metric
        elif self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Check if in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Reduce LR if plateau detected
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()

    def _reduce_lr(self, emergency: bool = False):
        """Reduce learning rate."""
        factor = self.factor if not emergency else self.factor ** 2

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * factor, self.min_lrs[i])

            if new_lr < old_lr:
                param_group['lr'] = new_lr
                logger.info(f"Reducing LR: {old_lr:.2e} -> {new_lr:.2e}")

        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0
        self.lr_reductions += 1

    def _increase_lr(self):
        """Increase learning rate (recovery)."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = old_lr * self.recovery_factor

            # Don't exceed original base LR
            max_allowed = min(new_lr, self.base_lrs[i])
            param_group['lr'] = max_allowed

            logger.info(f"Recovering LR: {old_lr:.2e} -> {max_allowed:.2e}")

        self.num_bad_epochs = 0


class NoisyStudentScheduler(_LRScheduler):
    """
    Noisy Student Training scheduler with periodic noise injection.

    Implements periodic learning rate and optimization noise injection
    to improve model robustness and generalization.

    Args:
        optimizer: Wrapped optimizer
        base_scheduler: Base scheduler to wrap
        noise_cycle_length: Length of noise injection cycles
        noise_factor: Factor for LR noise (default: 0.1)
        momentum_noise: Factor for momentum noise (default: 0.05)
        weight_noise: Factor for weight noise (default: 1e-5)
    """

    def __init__(
        self,
        optimizer,
        base_scheduler: _LRScheduler,
        noise_cycle_length: int = 1000,
        noise_factor: float = 0.1,
        momentum_noise: float = 0.05,
        weight_noise: float = 1e-5,
        last_epoch: int = -1
    ):
        self.base_scheduler = base_scheduler
        self.noise_cycle_length = noise_cycle_length
        self.noise_factor = noise_factor
        self.momentum_noise = momentum_noise
        self.weight_noise = weight_noise

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rates with noise injection."""
        base_lrs = self.base_scheduler.get_lr()

        # Determine if in noise phase
        cycle_position = self.last_epoch % self.noise_cycle_length
        is_noise_phase = cycle_position < self.noise_cycle_length * 0.1  # 10% of cycle

        if is_noise_phase:
            # Add noise to learning rates
            noisy_lrs = []
            for lr in base_lrs:
                noise = np.random.normal(0, lr * self.noise_factor)
                noisy_lr = max(lr + noise, lr * 0.1)  # Ensure minimum LR
                noisy_lrs.append(noisy_lr)
            return noisy_lrs
        else:
            return base_lrs

    def step(self, epoch=None):
        """Step both schedulers."""
        # Step base scheduler
        self.base_scheduler.step(epoch)

        # Step this scheduler
        super().step(epoch)

        # Apply weight noise if in noise phase
        cycle_position = self.last_epoch % self.noise_cycle_length
        is_noise_phase = cycle_position < self.noise_cycle_length * 0.1

        if is_noise_phase and self.weight_noise > 0:
            self._add_weight_noise()

        # Apply momentum noise
        if is_noise_phase and self.momentum_noise > 0:
            self._add_momentum_noise()

    def _add_weight_noise(self):
        """Add noise to model weights."""
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.weight_noise
                    param.data.add_(noise)

    def _add_momentum_noise(self):
        """Add noise to momentum values."""
        for group in self.optimizer.param_groups:
            if 'momentum' in group:
                base_momentum = group['momentum']
                noise = np.random.normal(0, self.momentum_noise)
                noisy_momentum = np.clip(base_momentum + noise, 0.0, 0.99)
                group['momentum'] = noisy_momentum


class SchedulerFactory:
    """
    Factory for creating and managing advanced learning rate schedulers.
    """

    @staticmethod
    def create_scheduler(
        scheduler_name: str,
        optimizer,
        total_steps: int,
        **kwargs
    ) -> _LRScheduler:
        """
        Create a scheduler with recommended settings.

        Args:
            scheduler_name: Name of scheduler
            optimizer: PyTorch optimizer
            total_steps: Total training steps
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Configured scheduler instance
        """
        scheduler_name = scheduler_name.lower()

        if scheduler_name == 'cosine_restarts':
            T_0 = kwargs.get('T_0', total_steps // 4)
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 0),
                **{k: v for k, v in kwargs.items() if k not in ['T_0', 'T_mult', 'eta_min']}
            )

        elif scheduler_name == 'onecycle':
            max_lr = kwargs.get('max_lr', 1e-3)
            return OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=kwargs.get('pct_start', 0.3),
                **{k: v for k, v in kwargs.items() if k not in ['max_lr', 'pct_start']}
            )

        elif scheduler_name == 'polynomial':
            warmup_steps = kwargs.get('warmup_steps', total_steps // 10)
            return PolynomialDecayLR(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                **{k: v for k, v in kwargs.items() if k != 'warmup_steps'}
            )

        elif scheduler_name == 'adaptive':
            return AdaptiveLRScheduler(
                optimizer,
                patience=kwargs.get('patience', total_steps // 100),
                **{k: v for k, v in kwargs.items() if k != 'patience'}
            )

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    @staticmethod
    def get_recommended_config(scheduler_name: str, training_type: str) -> Dict[str, Any]:
        """
        Get recommended configuration for different training scenarios.

        Args:
            scheduler_name: Name of scheduler
            training_type: Type of training ('pretraining', 'finetuning', 'few_shot')

        Returns:
            Dictionary of recommended hyperparameters
        """
        configs = {
            'cosine_restarts': {
                'pretraining': {'T_mult': 2, 'eta_min': 1e-6, 'temperature': 1.2},
                'finetuning': {'T_mult': 1, 'eta_min': 1e-7, 'temperature': 1.0},
                'few_shot': {'T_mult': 1, 'eta_min': 1e-8, 'temperature': 0.8}
            },
            'onecycle': {
                'pretraining': {'pct_start': 0.1, 'div_factor': 25, 'three_phase': True},
                'finetuning': {'pct_start': 0.3, 'div_factor': 10, 'three_phase': False},
                'few_shot': {'pct_start': 0.5, 'div_factor': 5, 'three_phase': False}
            },
            'polynomial': {
                'pretraining': {'power': 1.0, 'cycle': False},
                'finetuning': {'power': 0.5, 'cycle': False},
                'few_shot': {'power': 2.0, 'cycle': True}
            },
            'adaptive': {
                'pretraining': {'patience': 1000, 'factor': 0.5, 'threshold': 1e-3},
                'finetuning': {'patience': 100, 'factor': 0.7, 'threshold': 1e-4},
                'few_shot': {'patience': 10, 'factor': 0.8, 'threshold': 1e-5}
            }
        }

        return configs.get(scheduler_name, {}).get(training_type, {})
