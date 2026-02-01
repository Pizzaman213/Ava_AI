"""
Protocol interfaces for decoupled training components.

This module defines Protocol interfaces that allow training components to
communicate without tight coupling. Any object implementing the protocol
can be used interchangeably.

Protocols defined:
- MetricsLoggerProtocol: For metrics logging during training
- GenerationProviderProtocol: For text generation during training
- CheckpointSaverProtocol: For checkpoint saving
- EpisodicMemoryProtocol: For episodic memory/experience replay

Example:
    # Any class implementing these methods satisfies the protocol
    class MyMetricsLogger:
        def log_training_step(self, step, loss, lr, batch_size, **extra):
            ...

    loop_manager.set_components(metrics_manager=MyMetricsLogger())
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class MetricsLoggerProtocol(Protocol):
    """Protocol for metrics logging during training."""

    def log_training_step(
        self, step: int, loss: float, lr: float, batch_size: int, **extra_metrics
    ) -> None:
        """Log metrics for a training step (includes avg_loss and smoothed_loss)."""
        ...

    def log_gradients(self, step: int, grad_stats: Dict[str, Any]) -> None:
        """Log gradient statistics."""
        ...

    def log_generation(self, step: int, gen_data: Dict[str, Any]) -> None:
        """Log generated text samples."""
        ...

    def log_coherence(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log coherence metrics."""
        ...


@runtime_checkable
class GenerationProviderProtocol(Protocol):
    """Protocol for text generation during training."""

    def is_generation_pending(self) -> bool:
        """Check if a generation is currently running."""
        ...

    def generate_async(
        self,
        model: nn.Module,
        step: int,
        vocab_size: int,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        """Start async generation."""
        ...

    def process_completed_generations(self) -> list:
        """Get completed generation results."""
        ...

    def measure_coherence(
        self, model: nn.Module, step: int, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Measure model coherence."""
        ...


@runtime_checkable
class CheckpointSaverProtocol(Protocol):
    """Protocol for checkpoint saving during training."""

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> Any:
        """Save a checkpoint."""
        ...


@runtime_checkable
class EpisodicMemoryProtocol(Protocol):
    """Protocol for episodic memory integration during training.

    Episodic memory enables experience replay for continual learning.
    Implementations should manage a buffer of past experiences and
    provide methods to augment training batches with replay samples.
    """

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        global_step: int,
        current_phase: str = "training",
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """
        Augment current batch with replay samples from memory.

        Args:
            batch: Current training batch
            batch_idx: Current batch index
            global_step: Current global training step
            current_phase: 'training' or 'accumulating'

        Returns:
            Tuple of (augmented_batch, replay_indices)
        """
        ...

    def get_sample_weights(
        self,
        batch_idx: int,
        outputs: Dict[str, Any],
        aux_info: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Get per-sample weights for importance-weighted loss."""
        ...

    def store_batch(
        self,
        batch: Dict[str, torch.Tensor],
        losses: torch.Tensor,
        aux_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store current batch samples in memory buffer."""
        ...

    def update_replay_priorities(
        self, indices: Any, losses: torch.Tensor
    ) -> None:
        """Update priorities for replayed samples."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get episodic memory metrics for logging."""
        ...


__all__ = [
    'MetricsLoggerProtocol',
    'GenerationProviderProtocol',
    'CheckpointSaverProtocol',
    'EpisodicMemoryProtocol',
]
