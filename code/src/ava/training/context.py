"""
Base classes and interfaces for modular training components.

This module provides the shared state hub and component interfaces for the
Ava training pipeline. The design separates concerns:

Design Pattern - Shared Context:
    TrainingContext is a dataclass that acts as a "shared state hub" between
    components. Rather than components directly referencing each other (tight
    coupling), they all reference the context (loose coupling).

    Before (tight coupling):
        optimizer = Adam(model.parameters())  # Optimizer knows about Model
        scheduler = CosineScheduler(optimizer)  # Scheduler knows about Optimizer

    After (loose coupling):
        context.model = model
        context.optimizer = optimizer
        # Components access via context.model, context.optimizer

Why Metadata Dict vs Direct Fields:
    The `metadata` dictionary provides extensibility for config values that
    don't warrant a dedicated field. This avoids frequent dataclass changes.

    Use direct fields for:
    - Core training state (epoch, step, loss)
    - Components that multiple managers need (model, optimizer, tokenizer)

    Use metadata for:
    - Config values that only one manager reads (learning_rate, warmup_steps)
    - Experimental features not yet stable enough for direct fields
    - Custom user extensions

Component Lifecycle:
    1. __init__(context) - Store context reference, set _initialized=False
    2. initialize() - Allocate resources, connect to services, set _initialized=True
    3. [training loop] - Components called via hooks
    4. cleanup() - Release resources, close connections
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

    # OPTIMIZATION: Persistent pinned buffer pool for async GPU transfers
    # Survives across epochs to avoid reallocation overhead (2-5% speedup on epochs 2+)
    pinned_buffer_pool: Optional[Any] = None  # PersistentPinnedBufferPool

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

        This method bridges YAML config → TrainingContext, ensuring that
        config changes actually affect training behavior. Values go to either
        direct fields (if widely used) or metadata dict (if component-specific).

        Flow: YAML file → dict → this method → context fields/metadata → components

        Args:
            config: Configuration dictionary with training settings
        """
        training = config.get('training', {})

        # Core training settings
        self.gradient_accumulation_steps = training.get(
            'gradient_accumulation_steps',
            self.gradient_accumulation_steps
        )

        # Propagate training config values to metadata dict
        # These go to metadata (not direct fields) because:
        # 1. Only specific managers need them (OptimizerManager, TrainingLoopManager)
        # 2. Avoiding dataclass field explosion for 50+ config values
        # 3. Allows easy extension without modifying TrainingContext signature
        config_keys = [
            'batch_size', 'learning_rate', 'warmup_steps', 'max_steps',
            'num_epochs', 'weight_decay', 'max_grad_norm',
            'log_interval', 'save_interval', 'eval_interval'
        ]
        for key in config_keys:
            if key in training:
                self.metadata[key] = training[key]

        # Data config
        data = config.get('data', {})
        if 'max_length' in data:
            self.metadata['max_length'] = data['max_length']
        if 'num_workers' in data:
            self.metadata['num_workers'] = data['num_workers']

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
        """Get logger for this component (inherits from ava hierarchy)."""
        import logging
        # Use full module path so loggers inherit config from 'ava' parent
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

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

    Lifecycle hooks are called by TrainingPipeline at specific points:

    Training Flow:
        pipeline.initialize_all()  →  initialize()
        for epoch in range(num_epochs):
            pipeline.on_epoch_start(epoch)  →  on_epoch_start(epoch)
            for step, batch in enumerate(dataloader):
                pipeline.on_step_start(step)  →  on_step_start(step)
                loss = train_step(batch)
                pipeline.on_step_end(step, loss)  →  on_step_end(step, loss)
            pipeline.on_epoch_end(epoch)  →  on_epoch_end(epoch)
        pipeline.cleanup_all()  →  cleanup()

    Error Recovery:
        If training fails, pipeline.on_error() is called on all components
        BEFORE cleanup, allowing emergency checkpoint saves.

    All hooks are optional (default implementation is pass). Override only
    the hooks your component needs.
    """

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch. Use to reset epoch-level state."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch. Use to log epoch metrics, save checkpoints."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at the start of each training step. Use to prepare batch resources."""
        pass

    def on_step_end(self, step: int, loss: float) -> None:
        """Called at the end of each training step. Use to log step metrics."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when training fails. Use for emergency saves before cleanup."""
        self.logger.error(f"Error in {self.__class__.__name__}: {error}")

    def get_status(self) -> Dict[str, Any]:
        """Return current status/statistics for monitoring dashboards."""
        return {}
