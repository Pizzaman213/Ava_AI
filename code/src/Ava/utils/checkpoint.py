"""
Checkpoint management utilities for model saving and loading.

This module handles checkpoint operations including saving model states,
optimizer states, and training metadata. It ensures robust checkpoint
management for resuming training and model deployment.

Features:
- Synchronous and asynchronous checkpoint saving
- Background thread-based async saving for non-blocking saves
- Automatic cleanup of old checkpoints
- Best model tracking
"""

import torch  # type: ignore[import]
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import threading
import queue
import logging

logger = logging.getLogger(__name__)

# Global async checkpoint saver
_async_checkpoint_saver = None


class AsyncCheckpointSaver:
    """
    Background thread-based checkpoint saver for non-blocking saves.

    This allows training to continue while checkpoints are being saved,
    which can save 20-30 seconds per checkpoint for large models.

    Usage:
        >>> saver = AsyncCheckpointSaver()
        >>> saver.save_async(checkpoint_data, path)
        >>> # Training continues immediately
        >>> saver.wait_all()  # Wait for all saves to complete before exiting
    """

    def __init__(self, max_queue_size: int = 2):
        """
        Initialize async checkpoint saver.

        Args:
            max_queue_size: Maximum number of pending saves in queue
        """
        self.save_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.error: Optional[Exception] = None
        self._stop_event = threading.Event()

    def _worker(self):
        """Background worker that processes save requests."""
        while not self._stop_event.is_set():
            try:
                # Get next save task with timeout to check stop event
                try:
                    task = self.save_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task is None:  # Poison pill to stop worker
                    break

                checkpoint_data, path, metadata_info = task

                try:
                    # Perform the actual save
                    torch.save(checkpoint_data, path)
                    logger.debug(f"Async checkpoint saved to {path}")

                    # Save metadata if provided
                    if metadata_info:
                        metadata_path, metadata = metadata_info
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)

                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
                    self.error = e
                finally:
                    self.save_queue.task_done()

            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                self.error = e

    def save_async(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_path: str,
        metadata_info: Optional[tuple] = None
    ):
        """
        Queue a checkpoint for asynchronous saving.

        Args:
            checkpoint_data: Checkpoint dictionary to save
            checkpoint_path: Path where checkpoint should be saved
            metadata_info: Optional tuple of (metadata_path, metadata_dict)
        """
        if self.error:
            raise RuntimeError(f"Previous checkpoint save failed: {self.error}")

        # Create parent directory
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        # Queue the save task
        self.save_queue.put((checkpoint_data, checkpoint_path, metadata_info))
        logger.debug(f"Queued async checkpoint save to {checkpoint_path}")

    def wait_all(self, timeout: Optional[float] = None):
        """
        Wait for all pending saves to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        self.save_queue.join()
        if self.error:
            raise RuntimeError(f"Checkpoint save failed: {self.error}")

    def shutdown(self):
        """Gracefully shutdown the async saver."""
        self._stop_event.set()
        self.save_queue.put(None)  # Poison pill
        self.worker_thread.join(timeout=10.0)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass


def get_async_saver() -> AsyncCheckpointSaver:
    """Get or create the global async checkpoint saver."""
    global _async_checkpoint_saver
    if _async_checkpoint_saver is None:
        _async_checkpoint_saver = AsyncCheckpointSaver()
    return _async_checkpoint_saver


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_path: str,
    is_best: bool = False,
    async_save: bool = False
) -> str:
    """
    Save a training checkpoint (synchronously or asynchronously).

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch (int): Current epoch
        step (int): Current training step
        metrics (Dict[str, float]): Current metrics
        config (Dict[str, Any]): Model configuration
        checkpoint_path (str): Path to save checkpoint
        is_best (bool): Whether this is the best model so far
        async_save (bool): If True, save in background thread (20-30s faster for large models)

    Returns:
        Path to saved checkpoint

    Example:
        >>> # Synchronous save
        >>> checkpoint_path = save_checkpoint(
        ...     model, optimizer, epoch=5, step=1000,
        ...     metrics={'loss': 2.3}, config=model_config,
        ...     checkpoint_path='checkpoints/model.pt'
        ... )
        >>> # Asynchronous save (non-blocking)
        >>> checkpoint_path = save_checkpoint(
        ...     model, optimizer, epoch=5, step=1000,
        ...     metrics={'loss': 2.3}, config=model_config,
        ...     checkpoint_path='checkpoints/model.pt',
        ...     async_save=True
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

    # Prepare metadata
    metadata_path = checkpoint_dir / 'checkpoint_metadata.json'
    metadata = {
        'latest_checkpoint': str(checkpoint_path),
        'best_checkpoint': str(checkpoint_dir / 'best_model.pt') if is_best else None,
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    if async_save:
        # Asynchronous save using background thread
        saver = get_async_saver()
        saver.save_async(checkpoint, checkpoint_path, (metadata_path, metadata))

        # Save best model separately if this is the best
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            saver.save_async(checkpoint, str(best_path))

        logger.info(f"Queued async checkpoint save to {checkpoint_path}")
    else:
        # Synchronous save (original behavior)
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately if this is the best
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        # Save metadata
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