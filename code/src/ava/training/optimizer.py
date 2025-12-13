"""
Optimizer manager for the Ava pipeline.

Handles optimizer and learning rate scheduler creation with support for
multiple optimizer types and proper fallback handling.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .context import ManagerInterface, TrainingContext

logger = logging.getLogger(__name__)

# Check optional optimizer availability
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

try:
    from galore_torch import GaLoreAdamW
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False

try:
    import bitsandbytes as bnb
    ADAMW8BIT_AVAILABLE = True
except ImportError:
    ADAMW8BIT_AVAILABLE = False

try:
    from transformers import Adafactor
    ADAFACTOR_AVAILABLE = True
except ImportError:
    ADAFACTOR_AVAILABLE = False


class OptimizerManager(ManagerInterface):
    """
    Manages optimizer and learning rate scheduler creation.

    Supports multiple optimizer types with automatic fallback:
        - AdamW (default, always available)
        - Lion (requires lion-pytorch)
        - GaLore (requires galore-torch)
        - AdamW8bit (requires bitsandbytes)
        - Adafactor (requires transformers)

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> opt_manager = OptimizerManager(context)
        >>> opt_manager.initialize()
        >>> optimizer = opt_manager.create_optimizer(model, config, lr=1e-4)
        >>> scheduler = opt_manager.create_scheduler(optimizer, warmup_steps=1000, total_steps=10000)
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the optimizer manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self._warmup_steps = 0
        self._total_steps = 0

    def initialize(self) -> None:
        """Initialize the optimizer manager."""
        self._initialized = True
        self._log_available_optimizers()

    def cleanup(self) -> None:
        """Cleanup optimizer resources."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def _log_available_optimizers(self) -> None:
        """Log which optimizers are available."""
        available = ['AdamW (default)']
        unavailable = []

        if LION_AVAILABLE:
            available.append('Lion')
        else:
            unavailable.append('Lion (pip install lion-pytorch)')

        if GALORE_AVAILABLE:
            available.append('GaLore')
        else:
            unavailable.append('GaLore (pip install galore-torch)')

        if ADAMW8BIT_AVAILABLE:
            available.append('AdamW8bit')
        else:
            unavailable.append('AdamW8bit (pip install bitsandbytes)')

        if ADAFACTOR_AVAILABLE:
            available.append('Adafactor')
        else:
            unavailable.append('Adafactor (pip install transformers)')

        self.logger.debug(f"Available optimizers: {', '.join(available)}")
        if unavailable:
            self.logger.debug(f"Unavailable optimizers: {', '.join(unavailable)}")

    def create_optimizer(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        learning_rate: float,
        weight_decay: float = 0.01,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.

        Args:
            model: Model to optimize
            config: Configuration dict with optimizer settings
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization

        Returns:
            Configured optimizer

        Raises:
            ValueError: If optimizer creation fails completely
        """
        self.assert_initialized()

        # Get optimizer type from config (check both root level and training.optimizer)
        optimizer_config = config.get('optimizer', {}) or config.get('training', {}).get('optimizer', {})
        if isinstance(optimizer_config, dict):
            optimizer_type = optimizer_config.get('type', 'adamw').lower()
        else:
            optimizer_type = str(optimizer_config).lower() if optimizer_config else 'adamw'

        # Get parameters
        params = list(model.parameters())

        self.logger.info(f"Creating optimizer: {optimizer_type} (lr={learning_rate}, wd={weight_decay})")

        # Try to create requested optimizer with fallback chain
        optimizer = self._try_create_optimizer(
            optimizer_type, params, learning_rate, weight_decay, config
        )

        if optimizer is None:
            # Final fallback to AdamW
            self.logger.warning(f"Falling back to AdamW optimizer")
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay
            )

        self.optimizer = optimizer
        self.context.optimizer = optimizer

        return optimizer

    def _try_create_optimizer(
        self,
        optimizer_type: str,
        params: List[torch.nn.Parameter],
        learning_rate: float,
        weight_decay: float,
        config: Dict[str, Any],
    ) -> Optional[torch.optim.Optimizer]:
        """
        Try to create the requested optimizer.

        Args:
            optimizer_type: Type of optimizer to create
            params: Model parameters
            learning_rate: Learning rate
            weight_decay: Weight decay
            config: Full configuration dict

        Returns:
            Optimizer if successful, None otherwise
        """
        optimizer_config = config.get('optimizer', {})
        if not isinstance(optimizer_config, dict):
            optimizer_config = {}

        try:
            if optimizer_type == 'lion':
                if not LION_AVAILABLE:
                    self.logger.warning(
                        "Lion optimizer requested but not available. "
                        "Install with: pip install lion-pytorch"
                    )
                    return None
                return Lion(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )

            elif optimizer_type == 'galore':
                if not GALORE_AVAILABLE:
                    self.logger.warning(
                        "GaLore optimizer requested but not available. "
                        "Install with: pip install galore-torch"
                    )
                    return None
                # GaLoreAdamW handles low-rank projection internally
                betas = optimizer_config.get('betas', (0.9, 0.999))
                eps = optimizer_config.get('eps', 1e-6)
                return GaLoreAdamW(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )

            elif optimizer_type in ('adamw8bit', 'adamw_8bit'):
                if not ADAMW8BIT_AVAILABLE:
                    self.logger.warning(
                        "AdamW8bit optimizer requested but not available. "
                        "Install with: pip install bitsandbytes"
                    )
                    return None
                betas = optimizer_config.get('betas', (0.9, 0.95))
                eps = optimizer_config.get('eps', 1e-8)
                return bnb.optim.AdamW8bit(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay
                )

            elif optimizer_type == 'adafactor':
                if not ADAFACTOR_AVAILABLE:
                    self.logger.warning(
                        "Adafactor optimizer requested but not available. "
                        "Install with: pip install transformers"
                    )
                    return None
                return Adafactor(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    scale_parameter=False,
                    relative_step=False
                )

            elif optimizer_type == 'adamw':
                betas = optimizer_config.get('betas', (0.9, 0.999))
                eps = optimizer_config.get('eps', 1e-8)
                return torch.optim.AdamW(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps
                )

            else:
                self.logger.warning(f"Unknown optimizer type: {optimizer_type}")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to create {optimizer_type} optimizer: {e}")
            return None

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        lr_decay_steps: Optional[int] = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler with warmup and cosine decay.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate at end of decay
            lr_decay_steps: Number of steps for LR decay (defaults to total_steps)

        Returns:
            Sequential LR scheduler with warmup and cosine decay
        """
        self.assert_initialized()

        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

        if lr_decay_steps is None:
            lr_decay_steps = total_steps - warmup_steps

        # Warmup scheduler (linear increase)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Decay scheduler (cosine annealing)
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=lr_decay_steps,
            eta_min=min_lr
        )

        # Combine with sequential scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )

        self.scheduler = scheduler
        self.context.scheduler = scheduler

        self.logger.info(
            f"Created LR scheduler: warmup={warmup_steps} steps, "
            f"decay={lr_decay_steps} steps, min_lr={min_lr}"
        )

        return scheduler

    def on_step_end(self, step: int, loss: float) -> None:
        """
        Update scheduler after optimizer step.

        Args:
            step: Current training step
            loss: Current loss value
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            return self.optimizer.param_groups[0]['lr']
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return current optimizer status."""
        return {
            'learning_rate': self.get_current_lr(),
            'warmup_complete': self.context.step >= self._warmup_steps,
            'warmup_steps': self._warmup_steps,
            'total_steps': self._total_steps,
        }

    def on_error(self, error: Exception) -> None:
        """Handle optimizer errors."""
        self.logger.error(f"Optimizer error: {error}", exc_info=True)


def get_optimizer_availability() -> Dict[str, bool]:
    """
    Get availability status for all supported optimizers.

    Returns:
        Dictionary mapping optimizer names to availability status.
    """
    return {
        'adamw': True,  # Always available
        'lion': LION_AVAILABLE,
        'galore': GALORE_AVAILABLE,
        'adamw8bit': ADAMW8BIT_AVAILABLE,
        'adafactor': ADAFACTOR_AVAILABLE,
    }
