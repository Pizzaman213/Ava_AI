"""
Performance Mode Manager

This module provides different performance modes for training optimization,
including ultra-fast mode, progress display modes, and speed optimizations.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class PerformanceMode(Enum):
    """Available performance modes."""
    STANDARD = "standard"
    ULTRA_FAST = "ultra_fast"
    FAST_PROGRESS = "fast_progress"
    MINIMAL_PROGRESS = "minimal_progress"
    EXPRESS_MODE = "express_mode"
    NO_SYNC = "no_sync"


@dataclass
class PerformanceModeConfig:
    """Configuration for performance modes."""
    mode: PerformanceMode = PerformanceMode.STANDARD
    disable_wandb: bool = False             # Disable WandB logging
    disable_progress_bar: bool = False      # Disable progress bars
    disable_cuda_sync: bool = False         # Disable CUDA synchronization
    minimal_logging: bool = False           # Minimal logging output
    async_logging: bool = True              # Use async logging
    log_frequency: int = 100                # Logging frequency
    progress_update_frequency: int = 50     # Progress update frequency
    enable_profiling: bool = False          # Enable performance profiling
    cache_metrics: bool = True              # Cache metrics for batch processing


class PerformanceModeManager:
    """
    Manager for different performance modes to optimize training speed.

    Performance Modes:
    - STANDARD: Normal training with all features
    - ULTRA_FAST: Maximum speed, all logging disabled
    - FAST_PROGRESS: Real-time progress with enhanced updates
    - MINIMAL_PROGRESS: Compact progress display
    - EXPRESS_MODE: Optimized async logging with reduced frequency
    - NO_SYNC: Disable CUDA sync for maximum speed
    """

    def __init__(self, config: PerformanceModeConfig):
        """
        Initialize performance mode manager.

        Args:
            config: Performance mode configuration
        """
        self.config = config
        self.active_optimizations = []
        self._apply_mode_settings()

    def _apply_mode_settings(self) -> None:
        """Apply settings based on selected performance mode."""
        mode = self.config.mode

        if mode == PerformanceMode.ULTRA_FAST:
            self._configure_ultra_fast_mode()
        elif mode == PerformanceMode.FAST_PROGRESS:
            self._configure_fast_progress_mode()
        elif mode == PerformanceMode.MINIMAL_PROGRESS:
            self._configure_minimal_progress_mode()
        elif mode == PerformanceMode.EXPRESS_MODE:
            self._configure_express_mode()
        elif mode == PerformanceMode.NO_SYNC:
            self._configure_no_sync_mode()
        else:  # STANDARD
            self._configure_standard_mode()

    def _configure_ultra_fast_mode(self) -> None:
        """Configure ultra-fast mode - maximum speed, minimal overhead."""
        self.config.disable_wandb = True
        self.config.disable_progress_bar = True
        self.config.disable_cuda_sync = True
        self.config.minimal_logging = True
        self.config.async_logging = False
        self.config.log_frequency = 1000
        self.config.progress_update_frequency = 1000
        self.config.cache_metrics = False

        self.active_optimizations.extend([
            "All logging disabled",
            "Progress bars disabled",
            "CUDA sync disabled",
            "Metrics caching disabled",
            "Maximum training speed"
        ])

    def _configure_fast_progress_mode(self) -> None:
        """Configure fast progress mode - enhanced real-time updates."""
        self.config.disable_wandb = False
        self.config.disable_progress_bar = False
        self.config.disable_cuda_sync = False
        self.config.minimal_logging = False
        self.config.async_logging = True
        self.config.log_frequency = 10
        self.config.progress_update_frequency = 1

        self.active_optimizations.extend([
            "Enhanced progress bar",
            "Real-time loss updates",
            "Frequent WandB logging",
            "Async logging enabled"
        ])

    def _configure_minimal_progress_mode(self) -> None:
        """Configure minimal progress mode - compact display."""
        self.config.disable_wandb = False
        self.config.disable_progress_bar = False
        self.config.disable_cuda_sync = False
        self.config.minimal_logging = True
        self.config.async_logging = True
        self.config.log_frequency = 100
        self.config.progress_update_frequency = 100

        self.active_optimizations.extend([
            "Ultra-compact progress display",
            "Minimal console output",
            "Reduced logging frequency"
        ])

    def _configure_express_mode(self) -> None:
        """Configure express mode - optimized async logging."""
        self.config.disable_wandb = False
        self.config.disable_progress_bar = False
        self.config.disable_cuda_sync = False
        self.config.minimal_logging = False
        self.config.async_logging = True
        self.config.log_frequency = 50
        self.config.progress_update_frequency = 50
        self.config.cache_metrics = True

        self.active_optimizations.extend([
            "Optimized async logging",
            "Reduced logging frequency",
            "Batch metric processing",
            "Network-resilient caching"
        ])

    def _configure_no_sync_mode(self) -> None:
        """Configure no-sync mode - disable CUDA synchronization."""
        self.config.disable_wandb = False
        self.config.disable_progress_bar = False
        self.config.disable_cuda_sync = True
        self.config.minimal_logging = False
        self.config.async_logging = True
        self.config.log_frequency = 100
        self.config.progress_update_frequency = 50

        self.active_optimizations.extend([
            "CUDA synchronization disabled",
            "Reduced GPU-CPU sync overhead",
            "Faster batch processing"
        ])

    def _configure_standard_mode(self) -> None:
        """Configure standard mode - normal training with all features."""
        # Keep default settings
        self.active_optimizations.append("Standard mode with all features")

    def should_log_step(self, step: int) -> bool:
        """Check if this step should be logged based on frequency settings."""
        return step % self.config.log_frequency == 0

    def should_update_progress(self, step: int) -> bool:
        """Check if progress should be updated at this step."""
        if self.config.disable_progress_bar:
            return False
        return step % self.config.progress_update_frequency == 0

    def should_sync_cuda(self) -> bool:
        """Check if CUDA synchronization should be performed."""
        return not self.config.disable_cuda_sync

    def should_use_wandb(self) -> bool:
        """Check if WandB logging should be used."""
        return not self.config.disable_wandb

    def should_use_async_logging(self) -> bool:
        """Check if async logging should be used."""
        return self.config.async_logging

    def should_cache_metrics(self) -> bool:
        """Check if metrics should be cached for batch processing."""
        return self.config.cache_metrics

    def get_progress_format(self) -> Dict[str, Any]:
        """Get progress bar format configuration."""
        if self.config.mode == PerformanceMode.ULTRA_FAST:
            return {'enabled': False}

        elif self.config.mode == PerformanceMode.FAST_PROGRESS:
            return {
                'enabled': True,
                'format': 'detailed',
                'update_frequency': 1,
                'show_timing': True,
                'show_memory': True,
                'show_lr': True,
                'show_loss': True
            }

        elif self.config.mode == PerformanceMode.MINIMAL_PROGRESS:
            return {
                'enabled': True,
                'format': 'compact',
                'update_frequency': 100,
                'show_timing': False,
                'show_memory': False,
                'show_lr': True,
                'show_loss': True
            }

        else:  # STANDARD, EXPRESS_MODE, NO_SYNC
            return {
                'enabled': True,
                'format': 'standard',
                'update_frequency': self.config.progress_update_frequency,
                'show_timing': True,
                'show_memory': False,
                'show_lr': True,
                'show_loss': True
            }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration for the current mode."""
        return {
            'enabled': not self.config.minimal_logging,
            'frequency': self.config.log_frequency,
            'async_enabled': self.config.async_logging,
            'wandb_enabled': not self.config.disable_wandb,
            'cache_enabled': self.config.cache_metrics,
            'minimal': self.config.minimal_logging
        }

    def get_training_optimizations(self) -> Dict[str, Any]:
        """Get training-specific optimizations."""
        return {
            'cuda_sync_disabled': self.config.disable_cuda_sync,
            'skip_evaluation': self.config.mode == PerformanceMode.ULTRA_FAST,
            'skip_checkpointing': self.config.mode == PerformanceMode.ULTRA_FAST,
            'minimal_validation': self.config.minimal_logging,
            'batch_processing': self.config.cache_metrics,
            'profiling_enabled': self.config.enable_profiling
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance configuration."""
        return {
            'mode': self.config.mode.value,
            'active_optimizations': self.active_optimizations,
            'settings': {
                'wandb_disabled': self.config.disable_wandb,
                'progress_disabled': self.config.disable_progress_bar,
                'cuda_sync_disabled': self.config.disable_cuda_sync,
                'minimal_logging': self.config.minimal_logging,
                'async_logging': self.config.async_logging,
                'log_frequency': self.config.log_frequency,
                'progress_frequency': self.config.progress_update_frequency
            },
            'expected_benefits': self._get_expected_benefits()
        }

    def _get_expected_benefits(self) -> List[str]:
        """Get expected performance benefits for current mode."""
        mode = self.config.mode

        benefits = {
            PerformanceMode.ULTRA_FAST: [
                "Maximum training speed",
                "Minimal memory overhead",
                "No I/O bottlenecks",
                "Fastest convergence"
            ],
            PerformanceMode.FAST_PROGRESS: [
                "Real-time monitoring",
                "Immediate feedback",
                "Enhanced debugging",
                "Live loss tracking"
            ],
            PerformanceMode.MINIMAL_PROGRESS: [
                "Clean console output",
                "Reduced visual clutter",
                "Lower memory usage",
                "Focused on essentials"
            ],
            PerformanceMode.EXPRESS_MODE: [
                "Optimized logging",
                "Network resilience",
                "Batch processing",
                "Balanced speed/monitoring"
            ],
            PerformanceMode.NO_SYNC: [
                "Reduced GPU-CPU sync",
                "Faster batch processing",
                "Lower latency",
                "Better GPU utilization"
            ],
            PerformanceMode.STANDARD: [
                "Full feature set",
                "Complete monitoring",
                "Maximum visibility",
                "Best for debugging"
            ]
        }

        return benefits.get(mode, ["Standard performance"])

    def enable_profiling(self) -> None:
        """Enable performance profiling."""
        self.config.enable_profiling = True
        if "Performance profiling enabled" not in self.active_optimizations:
            self.active_optimizations.append("Performance profiling enabled")

    def disable_profiling(self) -> None:
        """Disable performance profiling."""
        self.config.enable_profiling = False
        self.active_optimizations = [
            opt for opt in self.active_optimizations
            if opt != "Performance profiling enabled"
        ]

    def adjust_frequency(self, log_freq: Optional[int] = None, progress_freq: Optional[int] = None) -> None:
        """Adjust logging and progress frequencies."""
        if log_freq is not None:
            self.config.log_frequency = log_freq
        if progress_freq is not None:
            self.config.progress_update_frequency = progress_freq

    def get_state_dict(self) -> Dict[str, Any]:
        """Get manager state for checkpointing."""
        return {
            'mode': self.config.mode.value,
            'config': {
                'disable_wandb': self.config.disable_wandb,
                'disable_progress_bar': self.config.disable_progress_bar,
                'disable_cuda_sync': self.config.disable_cuda_sync,
                'minimal_logging': self.config.minimal_logging,
                'async_logging': self.config.async_logging,
                'log_frequency': self.config.log_frequency,
                'progress_update_frequency': self.config.progress_update_frequency,
                'enable_profiling': self.config.enable_profiling,
                'cache_metrics': self.config.cache_metrics
            },
            'active_optimizations': self.active_optimizations
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load manager state from checkpoint."""
        mode_str = state_dict['mode']
        self.config.mode = PerformanceMode(mode_str)

        config_data = state_dict['config']
        for key, value in config_data.items():
            setattr(self.config, key, value)

        self.active_optimizations = state_dict.get('active_optimizations', [])


# Convenience functions for creating performance mode configurations
def create_ultra_fast_config() -> PerformanceModeConfig:
    """Create ultra-fast performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.ULTRA_FAST)


def create_fast_progress_config() -> PerformanceModeConfig:
    """Create fast progress performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.FAST_PROGRESS)


def create_minimal_progress_config() -> PerformanceModeConfig:
    """Create minimal progress performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.MINIMAL_PROGRESS)


def create_express_config() -> PerformanceModeConfig:
    """Create express mode performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.EXPRESS_MODE)


def create_no_sync_config() -> PerformanceModeConfig:
    """Create no-sync performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.NO_SYNC)


def create_standard_config() -> PerformanceModeConfig:
    """Create standard performance configuration."""
    return PerformanceModeConfig(mode=PerformanceMode.STANDARD)


def detect_performance_mode_from_args(args) -> PerformanceMode:
    """Detect performance mode from command line arguments."""
    if getattr(args, 'ultra_fast_mode', False):
        return PerformanceMode.ULTRA_FAST
    elif getattr(args, 'fast_progress', False):
        return PerformanceMode.FAST_PROGRESS
    elif getattr(args, 'minimal_progress', False):
        return PerformanceMode.MINIMAL_PROGRESS
    elif getattr(args, 'express_mode', False):
        return PerformanceMode.EXPRESS_MODE
    elif getattr(args, 'no_sync', False):
        return PerformanceMode.NO_SYNC
    else:
        return PerformanceMode.STANDARD


def create_config_from_args(args) -> PerformanceModeConfig:
    """Create performance mode configuration from command line arguments."""
    mode = detect_performance_mode_from_args(args)
    config = PerformanceModeConfig(mode=mode)

    # Apply any additional overrides from args
    if hasattr(args, 'disable_wandb') and args.disable_wandb:
        config.disable_wandb = True

    if hasattr(args, 'wandb_log_freq'):
        config.log_frequency = args.wandb_log_freq

    return config