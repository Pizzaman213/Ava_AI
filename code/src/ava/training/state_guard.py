"""Context managers for safe model state transitions.

This module provides context managers to ensure model state is properly
preserved and restored during evaluation or inference operations.

Key Features:
- Guaranteed state restoration even on exceptions
- Preserves training/eval mode
- Preserves gradient computation state
- Comprehensive error handling

Example:
    >>> with model_eval_mode(model):
    ...     outputs = model(inputs)  # Model in eval mode
    ... # Model automatically restored to training mode
"""

import logging
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def model_eval_mode(model: nn.Module, restore_mode: bool = True):
    """
    Context manager for safe model eval mode transitions.

    This ensures that the model's training mode is properly restored even if
    an exception occurs during evaluation. This is critical for preventing
    training state corruption when measuring coherence, running validation,
    or performing inference during training.

    Args:
        model: PyTorch model to put into eval mode
        restore_mode: Whether to restore original training mode on exit (default: True)

    Yields:
        The model in eval mode

    Example:
        >>> model = MyModel()
        >>> model.train()  # Set to training mode
        >>> with model_eval_mode(model):
        ...     # Model is now in eval mode
        ...     outputs = model(inputs)
        ... # Model automatically restored to training mode

        >>> # Even on exception, mode is restored:
        >>> model.train()
        >>> try:
        ...     with model_eval_mode(model):
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass
        >>> assert model.training  # Still in training mode!
    """
    was_training = model.training

    try:
        model.eval()
        logger.debug(f"Model set to eval mode (was_training={was_training})")
        yield model
    finally:
        if restore_mode and was_training:
            model.train()
            logger.debug("Model restored to training mode")
        elif restore_mode and not was_training:
            logger.debug("Model kept in eval mode (was already in eval)")


@contextmanager
def model_state_guard(
    model: nn.Module,
    preserve_grad_state: bool = True,
    preserve_training_mode: bool = True
):
    """
    Comprehensive model state guard that preserves multiple state aspects.

    This context manager provides complete state preservation:
    - Training/eval mode
    - Gradient computation state (torch.is_grad_enabled())
    - Any custom model attributes

    Args:
        model: PyTorch model
        preserve_grad_state: Whether to preserve gradient computation state (default: True)
        preserve_training_mode: Whether to preserve training/eval mode (default: True)

    Yields:
        The model

    Example:
        >>> with model_state_guard(model):
        ...     model.eval()
        ...     torch.set_grad_enabled(False)
        ...     # Do inference
        ... # Both training mode and grad state restored
    """
    # Save current state
    was_training = model.training
    grad_enabled = torch.is_grad_enabled() if preserve_grad_state else None

    logger.debug(
        f"Saving model state: training={was_training}, "
        f"grad_enabled={grad_enabled}"
    )

    try:
        yield model
    finally:
        # Restore training mode
        if preserve_training_mode:
            if was_training:
                model.train()
            else:
                model.eval()
            logger.debug(f"Restored training mode to: {was_training}")

        # Restore gradient state
        if preserve_grad_state and grad_enabled is not None:
            torch.set_grad_enabled(grad_enabled)
            logger.debug(f"Restored grad_enabled to: {grad_enabled}")


@contextmanager
def no_grad_eval_mode(model: nn.Module, restore_mode: bool = True):
    """
    Combined context manager for eval mode + no gradients.

    This is a convenience wrapper that combines model_eval_mode with torch.no_grad(),
    which is a common pattern for validation and inference.

    Args:
        model: PyTorch model
        restore_mode: Whether to restore original mode on exit (default: True)

    Yields:
        The model in eval mode with gradients disabled

    Example:
        >>> with no_grad_eval_mode(model):
        ...     outputs = model(inputs)  # No gradients, eval mode
        ... # Training mode and gradients restored
    """
    was_training = model.training
    was_grad_enabled = torch.is_grad_enabled()

    try:
        model.eval()
        with torch.no_grad():
            logger.debug("Model in eval mode with gradients disabled")
            yield model
    finally:
        if restore_mode:
            if was_training:
                model.train()
            torch.set_grad_enabled(was_grad_enabled)
            logger.debug(
                f"Restored: training={was_training}, "
                f"grad_enabled={was_grad_enabled}"
            )


@contextmanager
def preserve_rng_state(device: Optional[torch.device] = None):
    """
    Preserve and restore RNG state for reproducibility.

    This preserves both CPU and CUDA RNG states, ensuring that operations
    inside the context don't affect reproducibility outside.

    Args:
        device: CUDA device to preserve state for (default: current device)

    Yields:
        None

    Example:
        >>> with preserve_rng_state():
        ...     # Random operations here don't affect external state
        ...     x = torch.rand(10)
    """
    # Save CPU RNG state
    cpu_rng_state = torch.get_rng_state()

    # Save CUDA RNG state if available
    cuda_rng_state = None
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        cuda_rng_state = torch.cuda.get_rng_state(device)

    logger.debug(f"Saved RNG state (device={device})")

    try:
        yield
    finally:
        # Restore CPU RNG state
        torch.set_rng_state(cpu_rng_state)

        # Restore CUDA RNG state
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

        logger.debug("Restored RNG state")


@contextmanager
def safe_inference_mode(
    model: nn.Module,
    preserve_training_mode: bool = True,
    preserve_grad_state: bool = True,
    preserve_rng: bool = False
):
    """
    Comprehensive safe inference context manager.

    Combines all safety features:
    - Eval mode
    - No gradients
    - Training mode restoration
    - Optional RNG preservation

    Args:
        model: PyTorch model
        preserve_training_mode: Restore training mode on exit (default: True)
        preserve_grad_state: Restore gradient state on exit (default: True)
        preserve_rng: Preserve RNG state (default: False, expensive)

    Yields:
        The model in safe inference mode

    Example:
        >>> with safe_inference_mode(model):
        ...     outputs = model(inputs)  # Completely safe inference
        ... # Everything restored
    """
    # Save state
    was_training = model.training
    was_grad_enabled = torch.is_grad_enabled()

    # Optional RNG preservation
    rng_context = preserve_rng_state() if preserve_rng else None

    logger.debug(
        f"Entering safe inference mode: training={was_training}, "
        f"grad={was_grad_enabled}, preserve_rng={preserve_rng}"
    )

    try:
        model.eval()
        with torch.no_grad():
            if rng_context is not None:
                with rng_context:
                    yield model
            else:
                yield model
    finally:
        # Restore states
        if preserve_training_mode:
            if was_training:
                model.train()
            else:
                model.eval()

        if preserve_grad_state:
            torch.set_grad_enabled(was_grad_enabled)

        logger.debug("Exited safe inference mode, state restored")
