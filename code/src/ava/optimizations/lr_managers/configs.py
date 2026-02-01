"""
Configuration classes for learning rate management.

This module contains all configuration dataclasses for the lr_managers package:
- AdaptiveLRConfig: Real-time adaptive LR configuration
- LRConfig: General LR management configuration
- WarmupConfig: Advanced warmup system configuration
- WarmupSchedule: Enum for warmup schedule types
"""

from dataclasses import dataclass
from enum import Enum


class WarmupSchedule(Enum):
    """Available warmup schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate management."""
    # Warmup configuration
    warmup_steps: int = 0                  # Number of warmup steps (0 = no warmup)
    warmup_start_lr: float = 1e-5          # Starting LR for warmup (must be >= optimizer epsilon for BF16)

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
    min_lr: float = 1e-6                   # Minimum learning rate (must be >= optimizer epsilon for BF16)
    max_reductions: int = 5                # Maximum number of reductions


@dataclass
class LRConfig:
    """Configuration for learning rate management."""
    # Warmup configuration
    warmup_ratio: float = 0.03  # Fraction of total steps for warmup
    warmup_min_ratio: float = 0.01  # Starting LR as fraction of target LR
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


__all__ = [
    'WarmupSchedule',
    'AdaptiveLRConfig',
    'LRConfig',
    'WarmupConfig',
]
