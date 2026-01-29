"""
Base classes for training phases.

This module provides the foundational infrastructure for phase-based training:
- PhaseContext: Dataclass for passing state between phases
- TrainingPhase: Abstract base class for all training phases

Design Pattern:
    Each phase is a class that:
    1. Receives PhaseContext with current state
    2. Executes its specific logic (can modify context)
    3. Returns the (potentially modified) context
    4. Raises exceptions on failure (handled by PhaseExecutor)

Example:
    class ConfigPhase(TrainingPhase):
        name = "config"

        def execute(self, ctx: PhaseContext) -> PhaseContext:
            ctx.config = load_yaml(ctx.args.config)
            return ctx
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch


@dataclass
class PhaseContext:
    """
    Shared context passed between training phases.

    This dataclass acts as the "state bag" that flows through all phases.
    Each phase can read from and write to the context, building up the
    complete training state progressively.

    Design Decisions:
        - args: CLI arguments (set once, read by many phases)
        - config: YAML config dict (loaded in Phase 2)
        - Components: Set progressively (model in Phase 6, optimizer in Phase 7)
        - State tracking: rank, world_size, device for distributed training
        - Metadata: Flexible dict for phase-specific data that doesn't warrant a field

    Attributes:
        args: Parsed command line arguments (argparse.Namespace)
        config: Loaded YAML configuration dict

        # Distributed training
        rank: Process rank (0 for single GPU)
        world_size: Total processes (1 for single GPU)
        device: Target torch device

        # Core components (set progressively by phases)
        model: Neural network model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        tokenizer: Tokenizer for encoding/decoding

        # Training infrastructure
        run_manager: RunManager for output organization
        checkpoint_manager: CheckpointManager for saving/loading
        context: TrainingContext (shared state hub for components)
        pipeline: TrainingPipeline (component orchestrator)

        # Data loading
        train_loader: Training dataloader
        val_loader: Validation dataloader

        # Training state
        resume_epoch: Epoch to resume from (0 if fresh start)
        resume_step: Step to resume from (0 if fresh start)
        batch_size: Current batch size (may be calibrated)
        total_steps: Total training steps

        # Metadata for extensibility
        metadata: Flexible dict for phase-specific data
    """

    # CLI arguments
    args: Any = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Distributed training state
    rank: int = 0
    world_size: int = 1
    device: Optional[torch.device] = None

    # Core components (set by phases)
    model: Optional[Any] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None
    tokenizer: Optional[Any] = None

    # Training infrastructure
    run_manager: Optional[Any] = None
    checkpoint_manager: Optional[Any] = None
    context: Optional[Any] = None  # TrainingContext
    pipeline: Optional[Any] = None  # TrainingPipeline

    # Data loading
    train_loader: Optional[Any] = None
    val_loader: Optional[Any] = None

    # Training state
    resume_epoch: int = 0
    resume_step: int = 0
    batch_size: int = 8
    total_steps: int = 0
    num_epochs: int = 1
    learning_rate: float = 5e-5

    # Managers (set by phases)
    model_builder: Optional[Any] = None
    optimizer_mgr: Optional[Any] = None
    data_mgr: Optional[Any] = None
    training_mgr: Optional[Any] = None
    validation_mgr: Optional[Any] = None
    generation_mgr: Optional[Any] = None
    metrics_mgr: Optional[Any] = None
    diagnostics_mgr: Optional[Any] = None
    episodic_memory_mgr: Optional[Any] = None

    # Logger for rank-aware logging
    logger: Optional[Any] = None

    # Metadata for extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def get_config(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested config value using dot notation or multiple keys.

        Args:
            keys: Config keys (e.g., 'training', 'batch_size')
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            ctx.get_config('training', 'optimizer', 'learning_rate', default=1e-4)
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class TrainingPhase(ABC):
    """
    Abstract base class for training phases.

    Each phase represents a distinct step in the training pipeline setup
    or execution. Phases are executed in order by PhaseExecutor.

    Subclasses must define:
        - name: Unique phase identifier (e.g., 'config', 'model')
        - execute(): Main phase logic

    Subclasses may override:
        - description: Human-readable description
        - validate(): Pre-execution validation
        - cleanup(): Cleanup logic (called on error)

    Example:
        class ConfigPhase(TrainingPhase):
            name = "config"
            description = "Load and validate configuration"

            def execute(self, ctx: PhaseContext) -> PhaseContext:
                ctx.config = load_yaml(ctx.args.config)
                return ctx
    """

    # Must be overridden by subclasses
    name: str = "base"
    description: str = ""

    def __init__(self):
        """Initialize the phase."""
        pass

    @abstractmethod
    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Execute the phase logic.

        Args:
            ctx: Current phase context with accumulated state

        Returns:
            Updated phase context (may be same object)

        Raises:
            Any exception will be caught by PhaseExecutor
        """
        pass

    def validate(self, ctx: PhaseContext) -> List[str]:
        """
        Validate phase preconditions.

        Override this to check that required context fields are set
        before execute() runs.

        Args:
            ctx: Current phase context

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def cleanup(self, ctx: PhaseContext) -> None:
        """
        Cleanup resources on phase failure.

        Override this to release resources if execute() fails partway.
        Called by PhaseExecutor when an exception occurs.

        Args:
            ctx: Current phase context
        """
        pass

    def log(self, ctx: PhaseContext, message: str, level: str = "info") -> None:
        """
        Log a message (rank-aware).

        Only logs on rank 0 unless level is 'error'.

        Args:
            ctx: Phase context (for rank info)
            message: Message to log
            level: Log level ('debug', 'info', 'warning', 'error')
        """
        if ctx.rank != 0 and level != 'error':
            return

        if ctx.logger:
            log_fn = getattr(ctx.logger, level, ctx.logger.info)
            log_fn(message)
        else:
            print(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
