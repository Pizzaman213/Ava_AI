"""
Factory functions for creating LR management configurations.

Provides convenience functions for common configuration presets:
- Conservative: Slower adaptation, more stable
- Aggressive: Fast adaptation, may oscillate
- Balanced: Default middle-ground
- Various warmup schedule presets
"""

from .configs import AdaptiveLRConfig, WarmupConfig, WarmupSchedule


def create_conservative_lr_config() -> AdaptiveLRConfig:
    """Create conservative LR management configuration.

    Conservative settings are suitable for:
    - Large models that are sensitive to LR changes
    - When training from scratch with uncertain hyperparameters
    - Production training where stability is prioritized
    """
    return AdaptiveLRConfig(
        plateau_patience=1000,
        plateau_factor=0.7,
        divergence_threshold=2.0,
        emergency_factor=0.3,
        stability_threshold=8,
        increase_factor=1.05
    )


def create_aggressive_lr_config() -> AdaptiveLRConfig:
    """Create aggressive LR management configuration.

    Aggressive settings are suitable for:
    - Quick experimentation and hyperparameter search
    - Small models that are less sensitive to LR changes
    - When you want faster convergence at the risk of instability
    """
    return AdaptiveLRConfig(
        plateau_patience=250,
        plateau_factor=0.3,
        divergence_threshold=1.2,
        emergency_factor=0.05,
        stability_threshold=3,
        increase_factor=1.2
    )


def create_balanced_lr_config() -> AdaptiveLRConfig:
    """Create balanced LR management configuration (default).

    Balanced settings provide a good starting point for most cases.
    """
    return AdaptiveLRConfig()


def create_linear_warmup_config(warmup_steps: int = 2000, start_ratio: float = 0.01) -> WarmupConfig:
    """Create linear warmup configuration.

    Linear warmup increases LR linearly from start_ratio to 1.0.
    Simple and widely used.

    Args:
        warmup_steps: Number of steps for warmup
        start_ratio: Starting LR as fraction of target LR
    """
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.LINEAR,
        start_ratio=start_ratio
    )


def create_cosine_warmup_config(warmup_steps: int = 2000, start_ratio: float = 0.01) -> WarmupConfig:
    """Create cosine warmup configuration.

    Cosine warmup provides smoother LR increase, especially at the beginning
    and end of warmup. Often works well for transformers.

    Args:
        warmup_steps: Number of steps for warmup
        start_ratio: Starting LR as fraction of target LR
    """
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.COSINE,
        start_ratio=start_ratio
    )


def create_polynomial_warmup_config(warmup_steps: int = 2000, power: float = 2.0, start_ratio: float = 0.01) -> WarmupConfig:
    """Create polynomial warmup configuration.

    Polynomial warmup with configurable power. Higher power = slower initial
    increase, faster final approach to target.

    Args:
        warmup_steps: Number of steps for warmup
        power: Polynomial power (2.0 = quadratic)
        start_ratio: Starting LR as fraction of target LR
    """
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
    """Create adaptive warmup configuration with gradient-based completion.

    Adaptive warmup can complete early if gradient norms stabilize,
    or restart if a loss spike is detected.

    Args:
        warmup_steps: Maximum number of steps for warmup
        gradient_threshold: Gradient norm threshold for early completion
        restart_threshold: Loss multiplier that triggers warmup restart
        start_ratio: Starting LR as fraction of target LR
    """
    return WarmupConfig(
        warmup_steps=warmup_steps,
        schedule=WarmupSchedule.COSINE,
        start_ratio=start_ratio,
        use_gradient_norm=True,
        gradient_threshold=gradient_threshold,
        restart_threshold=restart_threshold
    )


__all__ = [
    'create_conservative_lr_config',
    'create_aggressive_lr_config',
    'create_balanced_lr_config',
    'create_linear_warmup_config',
    'create_cosine_warmup_config',
    'create_polynomial_warmup_config',
    'create_adaptive_warmup_config',
]
