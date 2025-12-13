"""
Mixed Precision Training Utilities.

This module provides utilities for mixed precision (AMP) training:
- MixedPrecisionConfig: Configuration class for precision settings
- setup_mixed_precision: Quick setup function
- clip_gradients_and_step: Combined gradient clipping and optimizer step
- save_checkpoint/load_checkpoint: Checkpoint utilities

Supported precision modes:
- fp32: Full precision (default)
- fp16: Half precision with gradient scaling
- bf16: Brain floating point (no scaling needed)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MixedPrecisionConfig:
    """
    Configuration for mixed precision training.

    Attributes:
        mixed_precision: Precision mode ('fp32', 'fp16', or 'bf16')
        use_amp: Whether automatic mixed precision is enabled
        amp_dtype: PyTorch dtype for autocast
        use_scaler: Whether gradient scaling is needed
        scaler: GradScaler instance (if needed)
    """

    def __init__(
        self,
        mixed_precision: str = 'fp32',
        device: Optional[torch.device] = None,
    ):
        """
        Configure mixed precision training settings.

        Args:
            mixed_precision: 'fp32', 'fp16', or 'bf16'
            device: Device for training (used to init GradScaler)
        """
        self.mixed_precision = mixed_precision.lower()

        if self.mixed_precision == 'bf16':
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False  # bf16 doesn't need scaler (same dynamic range as fp32)
        elif self.mixed_precision == 'fp16':
            self.use_amp = True
            self.amp_dtype = torch.float16
            self.use_scaler = True  # fp16 needs scaler for gradient scaling
        else:  # fp32 or anything else
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.use_scaler = False

        # Initialize gradient scaler if needed
        if self.use_scaler and device is not None and device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    def get_autocast_context(self, device_type: str = 'cuda'):
        """Get autocast context manager for forward pass."""
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp)


def setup_mixed_precision(
    mixed_precision: str,
    device: Optional[torch.device] = None,
) -> Tuple[bool, torch.dtype, bool, Optional[torch.amp.GradScaler]]:
    """
    Setup mixed precision training configuration.

    Args:
        mixed_precision: 'fp32', 'fp16', or 'bf16'
        device: Device for training

    Returns:
        Tuple of (use_amp, amp_dtype, use_scaler, scaler)
    """
    config = MixedPrecisionConfig(mixed_precision, device)
    return config.use_amp, config.amp_dtype, config.use_scaler, config.scaler


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = float('inf'),
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        scaler: Gradient scaler to save (optional)
        epoch: Current epoch
        step: Current step/global_step
        loss: Current loss value
        metrics: Optional metrics dictionary
        config: Optional configuration object
        extra_state: Additional state to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'global_step': step,  # Alias for compatibility
        'loss': loss,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        checkpoint['config'] = config

    if extra_state is not None:
        checkpoint.update(extra_state)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Supports both current and legacy checkpoint formats:
    - Current: 'optimizer_state_dict', 'model_state_dict'
    - Legacy: 'optimizer_state', 'model_state'

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scaler: Gradient scaler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle both current and legacy model state keys
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=strict)
    else:
        # Assume checkpoint is raw state dict
        model.load_state_dict(checkpoint, strict=strict)

    # Handle both current and legacy optimizer state keys
    if optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif 'optimizer_state' in checkpoint:
            # Legacy format from CheckpointManager
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', checkpoint.get('global_step', 0))

    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, step {step})")

    return checkpoint


# =============================================================================
# GRADIENT UTILITIES
# =============================================================================

def clip_gradients_and_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    return_tensor: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Clip gradients and perform optimizer step with optional AMP scaling.

    Args:
        optimizer: Optimizer instance
        model: Model with parameters
        scaler: Optional gradient scaler for AMP
        max_grad_norm: Maximum gradient norm for clipping
        return_tensor: If True, return grad_norm as tensor (avoids cudaStreamSynchronize).
                      Caller should only call .item() at log intervals.

    Returns:
        Gradient norm before clipping (as float or tensor based on return_tensor)
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    optimizer.zero_grad()

    # GPU SYNC FIX: Allow callers to defer .item() sync to log intervals
    if return_tensor:
        return grad_norm
    return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


__all__ = [
    'MixedPrecisionConfig',
    'setup_mixed_precision',
    'save_checkpoint',
    'load_checkpoint',
    'clip_gradients_and_step',
]
