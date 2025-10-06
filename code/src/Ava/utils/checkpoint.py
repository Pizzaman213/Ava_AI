"""
Checkpoint management utilities for model saving and loading.

This module handles checkpoint operations including saving model states,
optimizer states, and training metadata. It ensures robust checkpoint
management for resuming training and model deployment.
"""

import torch  # type: ignore[import]
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_path: str,
    is_best: bool = False
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch (int): Current epoch
        step (int): Current training step
        metrics (Dict[str, float]): Current metrics
        config (Dict[str, Any]): Model configuration
        checkpoint_path (str): Path to save checkpoint
        is_best (bool): Whether this is the best model so far

    Returns:
        Path to saved checkpoint

    Example:
        >>> checkpoint_path = save_checkpoint(
        ...     model, optimizer, epoch=5, step=1000,
        ...     metrics={'loss': 2.3}, config=model_config,
        ...     checkpoint_path='checkpoints/model.pt'
        ... )
    """
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    # Save main checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save best model separately if this is the best
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)

    # Save metadata
    metadata_path = checkpoint_dir / 'checkpoint_metadata.json'
    metadata = {
        'latest_checkpoint': str(checkpoint_path),
        'best_checkpoint': str(checkpoint_dir / 'best_model.pt') if is_best else None,
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file
        model: Model to load state into
        optimizer (optional): Optimizer to load state into
        device (torch.device, optional): Device to load checkpoint to
        strict (bool): Whether to strictly enforce state dict matching

    Returns:
        Dictionary containing checkpoint metadata

    Example:
        >>> checkpoint_data = load_checkpoint(
        ...     'checkpoints/best_model.pt',
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>> print(f"Resumed from epoch {checkpoint_data['epoch']}")
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return metadata
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
        'timestamp': checkpoint.get('timestamp', None)
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir (str): Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if not found

    Example:
        >>> latest = find_latest_checkpoint('checkpoints/')
        >>> if latest:
        ...     load_checkpoint(latest, model)
    """
    checkpoint_dir_path = Path(checkpoint_dir)

    # Check for metadata file
    metadata_path = checkpoint_dir_path / 'checkpoint_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'latest_checkpoint' in metadata:
                return metadata['latest_checkpoint']

    # Fall back to finding newest .pt file
    checkpoints = list(checkpoint_dir_path.glob('*.pt'))
    if checkpoints:
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    return None


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in a directory.

    Args:
        checkpoint_dir (str): Directory containing checkpoints

    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir_path = Path(checkpoint_dir)

    # Check for best model file
    best_path = checkpoint_dir_path / 'best_model.pt'
    if best_path.exists():
        return str(best_path)

    # Check metadata
    metadata_path = checkpoint_dir_path / 'checkpoint_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'best_checkpoint' in metadata:
                return metadata['best_checkpoint']

    return None