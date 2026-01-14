"""
Optimizer manager for the Ava pipeline.

Handles optimizer and learning rate scheduler creation with support for
multiple optimizer types and proper fallback handling.

Optimizer Fallback Chain:
    When a requested optimizer is unavailable (package not installed),
    the manager falls back gracefully:

    Requested     → Check availability → If unavailable → Fallback
    ─────────────────────────────────────────────────────────────────
    lion          → LION_AVAILABLE     → False          → AdamW
    galore        → GALORE_AVAILABLE   → False          → AdamW
    adamw8bit     → ADAMW8BIT_AVAILABLE → False         → AdamW
    adafactor     → ADAFACTOR_AVAILABLE → False         → AdamW
    adamw         → Always available   → (default)

Supported Optimizers:
    - AdamW (torch.optim.AdamW): Default, always available
      Best for: General training, stable convergence

    - Lion (lion_pytorch.Lion): Memory efficient, requires pip install lion-pytorch
      Best for: Large models, memory constrained, faster convergence

    - GaLore (galore_torch.GaLoreAdamW): Low-rank projection, requires pip install galore-torch
      Best for: Training with low-rank gradients

    - AdamW8bit (bitsandbytes.AdamW8bit): 8-bit optimizer, requires pip install bitsandbytes
      Best for: Memory savings on GPU, may need epsilon adjustment for bf16

    - Adafactor (transformers.Adafactor): Memory efficient, requires pip install transformers
      Best for: Very large models, no momentum storage overhead

Epsilon Adjustment:
    For bf16 training, epsilon is increased from default 1e-8 to 1e-6
    to prevent denormalized values which cause numerical instability.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

from .context import ManagerInterface, TrainingContext
from ava.config.training_config import get_mixed_precision

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

# SYMI optimizer decoupling for MoE (arXiv 2504.19925)
try:
    from ava.optimizations.symi_optimizer import (
        SYMIOptimizerWrapper,
        SYMIConfig as SYMIOptimizerConfig,
        wrap_optimizer_with_symi,
    )
    SYMI_AVAILABLE = True
except ImportError:
    SYMI_AVAILABLE = False
    SYMIOptimizerWrapper = None
    SYMIOptimizerConfig = None
    wrap_optimizer_with_symi = None


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

        # Check if DeepSpeed engine exists and manages optimizer
        # DeepSpeed creates its own optimizer when initialized - we should use it, not create another
        if hasattr(self.context, 'model'):
            try:
                from .deepspeed import is_deepspeed_engine
                if is_deepspeed_engine(self.context.model):
                    # DeepSpeed engine has its own optimizer - extract and use it
                    engine = self.context.model
                    if hasattr(engine, 'optimizer') and engine.optimizer is not None:
                        self.logger.info("Using DeepSpeed-managed optimizer from engine")
                        self.optimizer = engine.optimizer
                        self.context.optimizer = self.optimizer
                        return self.optimizer
            except ImportError:
                pass  # DeepSpeed not available

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
                weight_decay=weight_decay,
                fused=torch.cuda.is_available()  # 15-30% faster on CUDA
            )

        # SYMI optimizer decoupling for MoE (arXiv 2504.19925)
        # Provides ~30% speedup through optimizer state partitioning
        symi_config = config.get('symi', {})
        if symi_config.get('enabled', False) and SYMI_AVAILABLE:
            self.logger.info("Wrapping optimizer with SYMI decoupling for MoE speedup")
            symi_cfg = SYMIOptimizerConfig(
                enabled=True,
                num_partitions=symi_config.get('num_partitions', 0),
                sync_frequency=symi_config.get('sync_frequency', 100),
                use_async_sync=symi_config.get('use_async_sync', True),
                gradient_averaging=symi_config.get('gradient_averaging', 'partition'),
                state_precision=symi_config.get('state_precision', 'fp32'),
                enable_checkpointing=symi_config.get('enable_checkpointing', True),
            )
            optimizer = wrap_optimizer_with_symi(optimizer, model, symi_cfg)
            self.logger.info(f"SYMI enabled: partitions={symi_cfg.num_partitions}, sync_freq={symi_cfg.sync_frequency}")
        elif symi_config.get('enabled', False) and not SYMI_AVAILABLE:
            self.logger.warning("SYMI requested but not available, using standard optimizer")

        self.optimizer = optimizer
        self.context.optimizer = optimizer

        return optimizer

    def create_optimizer_factory(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        learning_rate: float,
        weight_decay: float = 0.01,
        support_model_arg: bool = False,
    ) -> Callable[..., torch.optim.Optimizer]:
        """
        Create a factory function that produces optimizer instances.

        This is useful for batch size calibration where we need to create temporary
        optimizers for memory testing without affecting the main optimizer.

        Args:
            model: Model to optimize (used when factory is called without args)
            config: Configuration dict with optimizer settings
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            support_model_arg: If True, factory accepts optional model parameter.
                              When called with a model arg, uses that model.
                              When called without args, uses the default model.
                              This is required when using model_factory for fresh
                              models per calibration test.

        Returns:
            Factory function that creates optimizer instances
        """
        if support_model_arg:
            def factory(target_model: Optional[torch.nn.Module] = None) -> torch.optim.Optimizer:
                """Create optimizer for given model or default model."""
                use_model = target_model if target_model is not None else model
                return self.create_optimizer(
                    model=use_model,
                    config=config,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay
                )
        else:
            def factory() -> torch.optim.Optimizer:
                return self.create_optimizer(
                    model=model,
                    config=config,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay
                )

        return factory

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
                # FIX: Add BF16 epsilon detection for AdamW8bit
                mixed_precision = get_mixed_precision(config)
                if mixed_precision in ('bf16', 'bfloat16'):
                    default_eps = 1e-6  # Safe for bfloat16
                else:
                    default_eps = 1e-8  # Standard for fp32/fp16
                eps = optimizer_config.get('eps', default_eps)
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
                # FIX: Detect bfloat16 precision and use appropriate epsilon
                # bfloat16 has ~7.8 bits mantissa, so eps=1e-8 is effectively zero
                # Use 1e-6 as default for bfloat16, 1e-8 for fp32/fp16
                mixed_precision = get_mixed_precision(config)
                if mixed_precision in ('bf16', 'bfloat16'):
                    default_eps = 1e-6  # Safe for bfloat16
                else:
                    default_eps = 1e-8  # Standard for fp32/fp16
                eps = optimizer_config.get('eps', default_eps)
                return torch.optim.AdamW(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                    fused=torch.cuda.is_available()  # 15-30% faster on CUDA
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
        scheduler_type: str = 'cosine',
        num_cycles: int = 1,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler with warmup and decay.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate at end of decay
            lr_decay_steps: Number of steps for LR decay (defaults to total_steps - warmup_steps)
            scheduler_type: Type of decay scheduler ('cosine', 'cosine_with_restarts', 'linear')
            num_cycles: Number of cycles for cosine_with_restarts (default 1)

        Returns:
            LR scheduler with warmup and decay
        """
        self.assert_initialized()

        # Check if DeepSpeed engine manages the scheduler
        # DeepSpeed can create its own scheduler when initialized - we should use it
        if hasattr(self.context, 'model'):
            try:
                from .deepspeed import is_deepspeed_engine
                if is_deepspeed_engine(self.context.model):
                    engine = self.context.model
                    if hasattr(engine, 'lr_scheduler') and engine.lr_scheduler is not None:
                        self.logger.info("Using DeepSpeed-managed scheduler from engine")
                        self.scheduler = engine.lr_scheduler
                        self.context.scheduler = self.scheduler
                        return self.scheduler
            except ImportError:
                pass  # DeepSpeed not available

        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

        if lr_decay_steps is None:
            lr_decay_steps = total_steps - warmup_steps

        # Ensure we have at least 1 decay step to avoid division by zero
        lr_decay_steps = max(lr_decay_steps, 1)

        # Warmup scheduler (linear increase from 1% to 100% of base LR)
        # Using start_factor=0.01 for smoother warmup start
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Create decay scheduler based on type
        if scheduler_type == 'cosine_with_restarts':
            # CosineAnnealingWarmRestarts: T_0 is the period of the first restart
            # After warmup, we want num_cycles restarts over lr_decay_steps
            t_0 = max(lr_decay_steps // num_cycles, 1)
            decay_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=1,  # Keep same period for each restart
                eta_min=min_lr
            )
            self.logger.info(
                f"Using CosineAnnealingWarmRestarts: T_0={t_0}, num_cycles={num_cycles}"
            )
        elif scheduler_type == 'linear':
            # Linear decay from base LR to min_lr
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr / optimizer.param_groups[0]['lr'] if optimizer.param_groups[0]['lr'] > 0 else 0.01,
                total_iters=lr_decay_steps
            )
        else:
            # Default: cosine annealing (standard cosine decay)
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
            f"Created LR scheduler: type={scheduler_type}, warmup={warmup_steps} steps, "
            f"decay={lr_decay_steps} steps, min_lr={min_lr:.2e}"
        )

        return scheduler

    def on_step_end(self, step: int, loss: float) -> None:
        """
        Callback after optimizer step completes.

        NOTE: Scheduler stepping is handled in the training loop (loop.py line 841),
        NOT here. This prevents double-stepping which would cause the LR schedule
        to run 2x too fast. This method is kept for other post-step operations.

        Args:
            step: Current training step
            loss: Current loss value
        """
        # FIX: Removed scheduler.step() - it's already called in the training loop
        # Having it here caused LR warmup/decay to happen 2x too fast
        pass

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

    def create_adaptive_lr_manager(
        self,
        enable_plateau_detection: bool = True,
        enable_spike_detection: bool = True,
        plateau_patience: int = 500,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
    ) -> Optional[Any]:
        """
        Create an adaptive learning rate manager that wraps the optimizer.

        This integrates with ava.optimizations.lr_managers.AdaptiveLearningRateManager
        to provide real-time LR adaptation based on training dynamics.

        Args:
            enable_plateau_detection: Reduce LR on loss plateau
            enable_spike_detection: Emergency LR reduction on loss spikes
            plateau_patience: Steps before plateau reduction
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate

        Returns:
            AdaptiveLearningRateManager instance or None if unavailable
        """
        self.assert_initialized()

        if self.optimizer is None:
            self.logger.warning("Cannot create adaptive LR manager: optimizer not created")
            return None

        try:
            from ava.optimizations.lr_managers import (
                AdaptiveLearningRateManager,
                AdaptiveLRConfig,
            )

            config = AdaptiveLRConfig(
                plateau_patience=plateau_patience,
                min_lr=min_lr,
                max_lr=max_lr,
            )

            adaptive_manager = AdaptiveLearningRateManager(
                optimizer=self.optimizer,
                config=config,
            )

            self.logger.info(
                f"Created adaptive LR manager: plateau_patience={plateau_patience}, "
                f"min_lr={min_lr:.2e}, max_lr={max_lr:.2e}"
            )

            return adaptive_manager

        except ImportError:
            self.logger.warning(
                "AdaptiveLearningRateManager not available. "
                "Using standard scheduler only."
            )
            return None


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
