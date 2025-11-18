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
    """
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    device: Optional[torch.device] = None
    config: Optional[Any] = None  # EnhancedTrainingConfig
    run_manager: Optional[Any] = None

    # Training state
    epoch: int = 0
    step: int = 0
    micro_step: int = 0

    # Loss tracking
    current_loss: float = 0.0
    best_loss: float = float("inf")

    # Custom attributes for extension
    metadata: Dict[str, Any] = field(default_factory=dict)


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
