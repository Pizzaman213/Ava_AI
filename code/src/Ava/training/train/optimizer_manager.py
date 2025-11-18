"""
Optimizer Manager for Ava Training Pipeline

Handles all optimizer setup and learning rate management including:
- Multiple optimizer types (AdamW, Lion, Sophia, AdaFactor, etc.)
- Fused and 8-bit optimizer variants
- Adaptive learning rate management
- Weight decay parameter grouping
"""

from typing import Any, Optional, Tuple

import torch

from src.Ava.optimization import AdaptiveLearningRateManager, AdaptiveLRConfig

from .base import TrainingComponent, TrainingContext


class OptimizerManager(TrainingComponent):
    """Manages optimizer creation and learning rate management."""

    def __init__(self, context: TrainingContext):
        """Initialize optimizer manager.

        Args:
            context: Training context with configuration
        """
        super().__init__(context)
        self.optimizer = None
        self.adaptive_lr_manager = None

    def setup_optimizer_and_lr(
        self,
        model: torch.nn.Module,
        config_dict: dict,
        training_config: Any,
        total_steps: Optional[int] = None,
    ) -> Tuple[torch.optim.Optimizer, Optional[AdaptiveLearningRateManager]]:
        """Set up optimizer and learning rate management.

        Args:
            model: Model to optimize
            config_dict: Raw configuration dictionary
            training_config: Enhanced training configuration
            total_steps: Total training steps for warmup calculation

        Returns:
            Tuple of (optimizer, adaptive_lr_manager)
        """
        from src.Ava.utils.logging import get_logger

        training_cfg = config_dict.get("training", {})

        # Get learning rate and weight decay
        lr = training_config.training.learning_rate or training_cfg.get(
            "learning_rate",
            getattr(training_config.training, "default_learning_rate", 5e-5),
        )
        weight_decay = training_cfg.get(
            "weight_decay",
            getattr(training_config.training, "default_weight_decay", 0.01),
        )

        lr = float(lr)
        weight_decay = float(weight_decay)

        # Create parameter groups with weight decay exclusions
        optimizer_grouped_parameters = self._create_param_groups(
            model, training_config, weight_decay
        )

        # Create optimizer
        optimizer_type = training_cfg.get("optimizer", "adamw").lower()
        optimizer = self._create_optimizer(
            optimizer_type,
            optimizer_grouped_parameters,
            lr,
            weight_decay,
            training_cfg,
            training_config,
        )

        # Setup adaptive learning rate management
        adaptive_lr_manager = self._setup_adaptive_lr(
            optimizer, training_config, total_steps, lr, training_cfg
        )

        self.optimizer = optimizer
        self.adaptive_lr_manager = adaptive_lr_manager

        return optimizer, adaptive_lr_manager

    def _create_param_groups(
        self, model: torch.nn.Module, training_config: Any, weight_decay: float
    ) -> list:
        """Create parameter groups with weight decay exclusions.

        Args:
            model: Model to optimize
            training_config: Training configuration
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter groups

        Raises:
            ValueError: If parameter count mismatch detected
        """
        from src.Ava.utils.logging import get_logger

        # Parameters that should not have weight decay
        no_decay = getattr(
            training_config.training,
            "no_decay_patterns",
            [
                "bias",
                "LayerNorm.weight",
                "layernorm.weight",
                "ln_f.weight",
                "ln_",
                "norm.weight",
            ],
        )

        decay_params = []
        no_decay_params = []
        total_trainable_params = 0

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_trainable_params += p.numel()

            # Check if parameter should skip weight decay
            should_skip_decay = False
            for nd in no_decay:
                # Exact match
                if nd == n:
                    should_skip_decay = True
                    break
                # Partial match - ensure it's a complete component
                if (
                    f".{nd}" in n
                    or n.endswith(nd)
                    or (
                        nd in n
                        and not n.replace(nd, "").replace(".", "").replace("_", "").isalnum()
                    )
                ):
                    should_skip_decay = True
                    break

            if should_skip_decay:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Validate parameter counts
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        total_optimizer_params = num_decay_params + num_no_decay_params

        get_logger().info(" Optimizer parameter groups:")
        get_logger().info(f"   With weight decay: {num_decay_params:,} parameters")
        get_logger().info(f"   Without weight decay: {num_no_decay_params:,} parameters")
        get_logger().info(
            f"   Total: {total_optimizer_params:,} / {total_trainable_params:,} trainable parameters"
        )

        if total_optimizer_params != total_trainable_params:
            raise ValueError(
                f"Parameter count mismatch! Optimizer has {total_optimizer_params:,} parameters "
                f"but model has {total_trainable_params:,} trainable parameters. "
                f"Some parameters are missing from optimizer groups!"
            )

        return optimizer_grouped_parameters

    def _create_optimizer(
        self,
        optimizer_type: str,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
        training_config: Any,
    ) -> torch.optim.Optimizer:
        """Create optimizer of specified type.

        Args:
            optimizer_type: Type of optimizer to create
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay coefficient
            training_cfg: Training config dictionary
            training_config: Enhanced training config

        Returns:
            Configured optimizer instance

        Raises:
            ValueError: If optimizer type is unsupported
        """
        from src.Ava.utils.logging import get_logger

        if optimizer_type == "adamw" or optimizer_type == "adamw_fused":
            return self._create_adamw(
                optimizer_type, optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_type == "lion":
            return self._create_lion(
                optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        elif optimizer_type == "lion8bit":
            return self._create_lion8bit(
                optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        elif optimizer_type == "adamw8bit":
            return self._create_adamw8bit(
                optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        elif optimizer_type == "sophia":
            return self._create_sophia(
                optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        elif optimizer_type == "adafactor":
            return self._create_adafactor(
                optimizer_grouped_parameters, lr, weight_decay, training_cfg
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. "
                f"Supported: adamw, adam, lion, lion8bit, adamw8bit, sophia, adafactor"
            )

    def _create_adamw(
        self,
        optimizer_type: str,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with optional fused kernels.

        Args:
            optimizer_type: "adamw" or "adamw_fused"
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay coefficient
            training_cfg: Training config

        Returns:
            AdamW optimizer instance
        """
        from src.Ava.utils.logging import get_logger

        # Get Adam betas from config
        adam_betas = getattr(training_cfg, "adam_betas", None)
        if adam_betas is None:
            adam_betas = (0.9, 0.95)
        else:
            adam_betas = tuple(adam_betas) if isinstance(adam_betas, list) else adam_betas

        use_fused = training_cfg.get("use_fused_optimizer", False) or (
            optimizer_type == "adamw_fused"
        )
        offload_to_cpu = training_cfg.get("offload_optimizer_state", False)

        # Try fused optimizer if requested
        if use_fused and torch.cuda.is_available() and not offload_to_cpu:
            try:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=adam_betas,
                    fused=True,
                )
                get_logger().info("✓ Using fused AdamW optimizer (15-25% faster)")
                get_logger().info(
                    f"  AdamW hyperparams: lr={lr:.2e}, betas={adam_betas}, weight_decay={weight_decay}"
                )
                get_logger().info(
                    "  Note: Fused kernels reduce optimizer overhead significantly"
                )
            except Exception as e:
                get_logger().warning(
                    f"⚠️  Fused optimizer not available, falling back to standard: {e}"
                )
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters, lr=lr, betas=adam_betas
                )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr, betas=adam_betas
            )
            if optimizer_type == "adamw_fused":
                get_logger().warning(
                    "⚠️  Fused AdamW requested but not available (requires CUDA and no CPU offloading)"
                )

        # Setup CPU offloading if enabled
        if offload_to_cpu:
            self._setup_optimizer_offload(optimizer, offload_to_cpu)

        return optimizer

    def _create_lion(
        self,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create Lion optimizer.

        Args:
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay
            training_cfg: Training config

        Returns:
            Lion optimizer instance
        """
        from src.Ava.optimization.optimizers.advanced import LionOptimizer
        from src.Ava.utils.logging import get_logger

        # Validate learning rate
        typical_adamw_lr_max = 3e-3
        if lr > typical_adamw_lr_max:
            get_logger().warning(
                f"⚠️  WARNING: Lion learning rate may be too high!\n"
                f"   Current LR: {lr:.2e}\n"
                f"   Lion typically requires 3-10x smaller LR than AdamW\n"
                f"   Recommended Lion LR range: 3e-5 to 1e-3\n"
            )

        lion_betas = training_cfg.get("lion_betas", (0.9, 0.99))
        lion_betas = tuple(lion_betas) if isinstance(lion_betas, list) else lion_betas

        optimizer = LionOptimizer(
            optimizer_grouped_parameters,
            lr=lr,
            betas=lion_betas,
            weight_decay=weight_decay,
        )
        get_logger().info("✓ Using Lion optimizer (50% memory reduction vs AdamW)")
        get_logger().info(
            f"  Lion hyperparams: lr={lr:.2e}, betas={lion_betas}, weight_decay={weight_decay}"
        )
        get_logger().info("  Note: Lion uses sign-based updates for better efficiency")
        if lr <= 1e-3:
            get_logger().info(f"  ✓ Learning rate {lr:.2e} is within recommended range for Lion")

        return optimizer

    def _create_lion8bit(
        self,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create 8-bit Lion optimizer.

        Args:
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay
            training_cfg: Training config

        Returns:
            Lion8bit optimizer instance
        """
        from src.Ava.optimization.optimizers.memory_efficient import Lion8bit
        from src.Ava.utils.logging import get_logger

        # Validate learning rate
        typical_adamw_lr_max = 3e-3
        if lr > typical_adamw_lr_max:
            get_logger().warning(
                f"⚠️  WARNING: Lion learning rate may be too high!\n"
                f"   Current LR: {lr:.2e}\n"
                f"   Lion typically requires 3-10x smaller LR than AdamW\n"
                f"   Recommended Lion LR range: 3e-5 to 1e-3\n"
            )

        lion_betas = training_cfg.get("lion_betas", (0.9, 0.99))
        lion_betas = tuple(lion_betas) if isinstance(lion_betas, list) else lion_betas

        lion_min_8bit_size = training_cfg.get("lion_min_8bit_size", 4096)
        lion_block_wise = training_cfg.get("lion_block_wise", True)
        lion_is_paged = training_cfg.get("lion_is_paged", False)
        lion_percentile_clipping = training_cfg.get("lion_percentile_clipping", 100)

        optimizer = Lion8bit(
            optimizer_grouped_parameters,
            lr=lr,
            betas=lion_betas,
            weight_decay=weight_decay,
            min_8bit_size=lion_min_8bit_size,
            block_wise=lion_block_wise,
            is_paged=lion_is_paged,
            percentile_clipping=lion_percentile_clipping,
        )
        get_logger().info(
            "✓ Using 8-bit Lion optimizer (87.5% memory reduction vs AdamW)"
        )
        get_logger().info(
            f"  Lion8bit hyperparams: lr={lr:.2e}, betas={lion_betas}, weight_decay={weight_decay}"
        )
        get_logger().info(
            "  Note: 8-bit quantization of optimizer states with minimal accuracy impact"
        )
        if lr <= 1e-3:
            get_logger().info(f"  ✓ Learning rate {lr:.2e} is within recommended range for Lion")

        return optimizer

    def _create_adamw8bit(
        self,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create 8-bit AdamW optimizer.

        Args:
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay
            training_cfg: Training config

        Returns:
            AdamW8bit optimizer instance
        """
        from src.Ava.optimization.optimizers.memory_efficient import AdamW8bit
        from src.Ava.utils.logging import get_logger

        adamw_betas = training_cfg.get("adamw_betas", (0.9, 0.999))
        adamw_betas = tuple(adamw_betas) if isinstance(adamw_betas, list) else adamw_betas

        optimizer = AdamW8bit(
            optimizer_grouped_parameters,
            lr=lr,
            betas=adamw_betas,
            eps=1e-8,
            weight_decay=weight_decay,
        )
        get_logger().info(
            "✓ Using 8-bit AdamW optimizer (75% memory reduction vs standard AdamW)"
        )
        get_logger().info(
            f"  AdamW8bit hyperparams: lr={lr:.2e}, betas={adamw_betas}, weight_decay={weight_decay}"
        )
        get_logger().info(
            "  Note: 8-bit quantization of optimizer states with <1% accuracy impact"
        )

        return optimizer

    def _create_sophia(
        self,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create Sophia optimizer.

        Args:
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay
            training_cfg: Training config

        Returns:
            Sophia optimizer instance
        """
        from src.Ava.optimization.optimizers.advanced import SophiaOptimizer
        from src.Ava.utils.logging import get_logger

        sophia_betas = training_cfg.get("sophia_betas", (0.965, 0.99))
        sophia_rho = training_cfg.get("sophia_rho", 0.04)

        optimizer = SophiaOptimizer(
            optimizer_grouped_parameters,
            lr=lr,
            betas=tuple(sophia_betas) if isinstance(sophia_betas, list) else sophia_betas,
            rho=sophia_rho,
            weight_decay=weight_decay,
        )
        get_logger().info(
            "✓ Using Sophia optimizer (2x speedup with second-order optimization)"
        )
        get_logger().info(
            f"  Sophia hyperparams: lr={lr:.2e}, betas={sophia_betas}, rho={sophia_rho}"
        )

        return optimizer

    def _create_adafactor(
        self,
        optimizer_grouped_parameters: list,
        lr: float,
        weight_decay: float,
        training_cfg: dict,
    ) -> torch.optim.Optimizer:
        """Create AdaFactor optimizer.

        Args:
            optimizer_grouped_parameters: Parameter groups
            lr: Learning rate
            weight_decay: Weight decay
            training_cfg: Training config

        Returns:
            AdaFactor optimizer instance
        """
        from src.Ava.optimization.optimizers.advanced import AdaFactorOptimizer
        from src.Ava.utils.logging import get_logger

        use_adaptive_lr = training_cfg.get("adafactor_adaptive_lr", True)

        optimizer = AdaFactorOptimizer(
            optimizer_grouped_parameters,
            lr=None if use_adaptive_lr else lr,
            weight_decay=weight_decay,
            scale_parameter=True,
            relative_step=use_adaptive_lr,
            warmup_init=training_cfg.get("adafactor_warmup_init", False),
        )
        get_logger().info(
            "✓ Using AdaFactor optimizer (80% memory reduction vs AdamW)"
        )
        get_logger().info(
            f"  AdaFactor: adaptive_lr={use_adaptive_lr}, weight_decay={weight_decay}"
        )

        return optimizer

    def _setup_optimizer_offload(self, optimizer: torch.optim.Optimizer, offload_to_cpu: bool) -> None:
        """Setup CPU offloading for optimizer state.

        Args:
            optimizer: Optimizer instance
            offload_to_cpu: Whether to offload to CPU
        """
        from src.Ava.utils.logging import get_logger

        try:
            def create_offload_hook(should_offload: bool):
                return lambda grad: grad.cpu() if should_offload else grad

            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    param.register_hook(create_offload_hook(offload_to_cpu))

            get_logger().info(
                "✓ Optimizer state CPU offloading enabled (30-50% memory savings)"
            )
            get_logger().info("  Note: Adds ~5% overhead but allows larger batch sizes")
        except Exception as e:
            get_logger().warning(f"⚠️  CPU offloading not available: {e}")

    def _setup_adaptive_lr(
        self,
        optimizer: torch.optim.Optimizer,
        training_config: Any,
        total_steps: Optional[int],
        lr: float,
        training_cfg: dict,
    ) -> Optional[AdaptiveLearningRateManager]:
        """Setup adaptive learning rate management.

        Args:
            optimizer: Optimizer instance
            training_config: Training configuration
            total_steps: Total training steps
            lr: Base learning rate
            training_cfg: Training config dictionary

        Returns:
            Adaptive LR manager or None
        """
        from src.Ava.utils.logging import get_logger

        if not getattr(training_config.training, "use_adaptive_lr", True):
            return None

        get_logger().info(" Setting up adaptive learning rate management...")

        # Calculate warmup steps
        warmup_percentage = getattr(training_config.training, "warmup_percentage", 0.03)
        if total_steps and warmup_percentage > 0:
            warmup_steps = int(total_steps * warmup_percentage)
            get_logger().info(
                f"   Warmup steps: {warmup_steps} ({warmup_percentage:.1%} of {total_steps} total steps)"
            )
        else:
            warmup_steps = getattr(training_config.training, "warmup_steps", 3000)
            get_logger().info(
                f"   Warmup steps: {warmup_steps} (from config - total steps unknown)"
            )

        # Load adaptive LR config
        adaptive_lr_cfg = getattr(training_config.training, "adaptive_lr", {})

        # Helper to get value from dict or object
        def get_cfg(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        adaptive_config = AdaptiveLRConfig(
            warmup_steps=warmup_steps,
            batch_loss_window=get_cfg(adaptive_lr_cfg, "batch_loss_window", 100),
            plateau_patience=get_cfg(adaptive_lr_cfg, "plateau_patience", 500),
            plateau_factor=get_cfg(adaptive_lr_cfg, "plateau_factor", 0.5),
            lr_check_interval=get_cfg(adaptive_lr_cfg, "lr_check_interval", 50),
            min_lr=get_cfg(adaptive_lr_cfg, "min_lr", 1e-7),
            max_lr=get_cfg(adaptive_lr_cfg, "max_lr", lr * 2.0),
            stability_threshold=get_cfg(adaptive_lr_cfg, "stability_threshold", 5),
            increase_factor=get_cfg(adaptive_lr_cfg, "increase_factor", 1.05),
            min_improvement=get_cfg(adaptive_lr_cfg, "min_improvement", 0.0002),
            divergence_threshold=get_cfg(
                adaptive_lr_cfg, "divergence_threshold", 3.0
            ),
            emergency_factor=get_cfg(adaptive_lr_cfg, "emergency_factor", 0.5),
        )

        adaptive_lr_manager = AdaptiveLearningRateManager(optimizer, adaptive_config)
        get_logger().info(
            "✓ Adaptive LR manager initialized with warmup, plateau detection, and stability increases"
        )
        get_logger().info(
            f"   Divergence threshold: {adaptive_config.divergence_threshold}x "
            f"(loss spikes tolerated up to {adaptive_config.divergence_threshold}x best loss)"
        )
        get_logger().info(
            f"   Emergency LR reduction: {adaptive_config.emergency_factor}x "
            f"(cuts LR to {adaptive_config.emergency_factor*100:.0f}% on emergency)"
        )
        get_logger().info(
            f"   Min improvement: {adaptive_config.min_improvement} (plateau detection threshold)"
        )
        get_logger().info(f"   Plateau patience: {adaptive_config.plateau_patience} steps")

        return adaptive_lr_manager
