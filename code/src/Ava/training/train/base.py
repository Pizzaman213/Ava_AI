"""
Base classes and interfaces for modular training components.

Provides common interfaces and data structures for all training components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TrainingContext:
    """
    Shared context passed to all training components.

    This allows components to access training state without tight coupling.
    The context serves as the central hub for all shared state in the Ava pipeline.

    Attributes:
        model: The neural network model being trained
        optimizer: The optimizer for model parameters
        scheduler: Learning rate scheduler
        device: Target device (cuda/cpu)
        config: Training configuration dictionary or object
        run_manager: RunManager for output organization
        tokenizer: Tokenizer for text encoding/decoding

        epoch: Current epoch number
        step: Current optimizer step (after gradient accumulation)
        micro_step: Current micro-batch step (before accumulation)

        current_loss: Loss from most recent step
        best_loss: Best loss seen so far

        rank: Process rank for distributed training
        world_size: Total number of processes
        is_main_process: Whether this is rank 0

        gradient_accumulation_steps: Number of accumulation steps
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (bfloat16 or float16)

        metadata: Custom attributes for extension
    """
    # Core components
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None  # LRScheduler
    device: Optional[torch.device] = None
    config: Optional[Any] = None  # EnhancedTrainingConfig or dict
    run_manager: Optional[Any] = None
    tokenizer: Optional[Any] = None

    # Training state
    epoch: int = 0
    step: int = 0
    micro_step: int = 0

    # Loss tracking
    current_loss: float = 0.0
    best_loss: float = float("inf")

    # Distributed training info
    rank: int = 0
    world_size: int = 1
    is_main_process: bool = True

    # Training configuration shortcuts
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    # Batch size management
    batch_controller: Optional[Any] = None  # BatchSizeController for dynamic batching

    # Custom attributes for extension
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure is_main_process is consistent with rank
        if self.rank != 0:
            self.is_main_process = False

    def update_from_config(self, config: Dict[str, Any]) -> None:
        """
        Update context fields from a configuration dictionary.

        Args:
            config: Configuration dictionary with training settings
        """
        training = config.get('training', {})

        # Update training settings
        self.gradient_accumulation_steps = training.get(
            'gradient_accumulation_steps',
            self.gradient_accumulation_steps
        )

        # Update precision settings
        precision = training.get('precision', {})
        mixed_precision = precision.get('mixed_precision', 'bf16')

        if mixed_precision == 'bf16':
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
        elif mixed_precision == 'fp16':
            self.use_amp = True
            self.amp_dtype = torch.float16
        elif mixed_precision in ('fp32', 'none', False):
            self.use_amp = False
            self.amp_dtype = torch.float32


class TrainingComponent(ABC):
    """
    Base class for all modular training components.

    Components should handle initialization, cleanup, and provide
    a clean interface for their responsibility.
    """

    def __init__(self, context: TrainingContext):
        """Initialize component with training context."""
        self.context = context
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize component. Called once at startup."""
        self._initialized = True

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources. Called on shutdown or error."""
        pass

    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    @property
    def logger(self):
        """Get logger for this component."""
        import logging
        return logging.getLogger(self.__class__.__name__)

    @property
    def model(self) -> nn.Module:
        """Shortcut to model from context."""
        return self.context.model

    @property
    def device(self) -> torch.device:
        """Shortcut to device from context."""
        return self.context.device or torch.device("cpu")

    @property
    def config(self):
        """Shortcut to config from context."""
        return self.context.config

    def assert_initialized(self):
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. "
                f"Call initialize() first."
            )


class ManagerInterface(TrainingComponent):
    """
    Extended interface for manager components that integrate into
    the main training loop.
    """

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at the start of each training step."""
        pass

    def on_step_end(self, step: int, loss: float) -> None:
        """Called at the end of each training step."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs during training."""
        self.logger.error(f"Error in {self.__class__.__name__}: {error}")

    def get_status(self) -> Dict[str, Any]:
        """Return current status/statistics."""
        return {}
