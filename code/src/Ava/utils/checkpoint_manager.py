"""
Unified Checkpoint Manager for Ava Training Framework.

Provides async-capable checkpoint saving/loading with:
- CUDA stream-based non-blocking GPU->CPU transfers
- Pinned memory double buffering for overlap
- Background disk I/O via thread pool
- Support for both GPU and CPU-only training
- Automatic best/latest model tracking

Usage:
    from Ava.utils.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(save_dir=Path("checkpoints"), max_keep=3)
    manager.save(model, optimizer, epoch=1, step=1000, metrics={"loss": 0.5})
    epoch, step = manager.load(model, optimizer, checkpoint_path)
    manager.shutdown()  # Clean up when done
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

# Import optimized CUDA stream utilities
try:
    from Ava.utils.cuda_streams import (
        StreamPool, NonBlockingTransfer, get_stream_pool,
        wait_stream, efficient_sync
    )
    _CUDA_STREAMS_AVAILABLE = True
except ImportError:
    _CUDA_STREAMS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with truly async saving using CUDA streams and pinned memory.

    This is the unified checkpoint manager for the Ava training framework, extracted
    from train_100m_full.py for reuse across different training scripts.
    """

    def __init__(
        self,
        save_dir: Path,
        max_keep: int = 3,
        config: Optional[Dict] = None,
        async_save: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep (oldest deleted first)
            config: Optional config dict to save with checkpoints
            async_save: Enable async saving (recommended for GPU training)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints: list = []
        self.config = config
        self.async_save = async_save

        # Thread pool for async disk I/O
        self.save_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="AsyncCheckpoint"
        ) if async_save else None
        self.pending_saves: list = []  # Track multiple ongoing saves

        # CUDA stream for non-blocking GPU->CPU transfers
        self._checkpoint_stream: Optional[torch.cuda.Stream] = None
        if async_save and torch.cuda.is_available():
            self._checkpoint_stream = torch.cuda.Stream()

        # Double buffer for pinned memory (allows overlap)
        self._pinned_buffers: list = [{}, {}]
        self._current_buffer = 0
        self._buffer_in_use = [False, False]
        self._buffer_lock = threading.Lock()  # Protects buffer selection

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float]
    ) -> Path:
        """
        Save checkpoint in format compatible with generate.py.

        Args:
            model: Model to save (handles DDP automatically)
            optimizer: Optimizer to save
            epoch: Current epoch number
            step: Current step number
            metrics: Dictionary of metrics to save

        Returns:
            Path to saved checkpoint
        """
        if self.async_save and self.save_executor is not None and torch.cuda.is_available():
            return self._save_truly_async(model, optimizer, epoch, step, metrics)
        elif self.async_save and self.save_executor is not None:
            return self._save_async_cpu(model, optimizer, epoch, step, metrics)
        else:
            return self._save_sync(model, optimizer, epoch, step, metrics)

    def _save_sync(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float]
    ) -> Path:
        """Synchronous checkpoint save (original behavior)."""
        # Handle DDP models
        actual_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': step,  # Alias for compatibility with shared.py
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # Standardized key
            'metrics': metrics,
            'config': self.config,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)

        # Always save latest
        latest_path = self.save_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)

        # Save best if validation loss improved
        if 'val_loss' in metrics:
            best_path = self.save_dir / 'best_model.pt'
            if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                torch.save(checkpoint, best_path)

        # Cleanup old checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        return path

    def _save_truly_async(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float]
    ) -> Path:
        """
        Truly asynchronous checkpoint save using CUDA streams and pinned memory.

        This implementation:
        1. Uses a separate CUDA stream for GPU->CPU transfers (non-blocking)
        2. Uses pinned memory for fast async transfers
        3. Only blocks briefly to record the stream, not for the full transfer
        4. Disk I/O happens in a background thread after transfer completes
        """
        # Clean up completed saves
        self._cleanup_completed_saves()

        # Get the model state dict reference (this is fast, just gets references)
        actual_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        # Select buffer with lock to prevent race conditions
        with self._buffer_lock:
            buffer_idx = self._current_buffer
            self._current_buffer = 1 - self._current_buffer

            # Wait for buffer to be free (from previous save 2 saves ago)
            if self._buffer_in_use[buffer_idx]:
                for future, buf_idx, _ in list(self.pending_saves):
                    if buf_idx == buffer_idx and not future.done():
                        try:
                            future.result(timeout=300)  # 5 minute timeout
                        except Exception as e:
                            logger.warning(f"Previous checkpoint save failed: {e}")

            self._buffer_in_use[buffer_idx] = True
        pinned_buffer = self._pinned_buffers[buffer_idx]

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'

        # GPU SYNC FIX: Use event-based stream ordering instead of blocking sync
        # The old approach called efficient_sync() which blocked the CPU until current stream
        # completed. Instead, we record an event on the compute stream and make the checkpoint
        # stream wait for that event. This allows:
        # 1. CPU to continue immediately (no blocking)
        # 2. Checkpoint stream waits only on compute stream (not all GPU work)
        # 3. Compute stream can continue with next batch in parallel
        compute_complete_event = torch.cuda.Event()
        compute_complete_event.record(torch.cuda.current_stream())

        # Record CUDA events for timing
        transfer_end_event = torch.cuda.Event(enable_timing=True)

        # Use checkpoint stream for non-blocking transfers
        # The checkpoint stream waits for compute to finish before starting transfers
        self._checkpoint_stream.wait_event(compute_complete_event)
        with torch.cuda.stream(self._checkpoint_stream):
            # Copy model state to pinned memory asynchronously
            model_state = {}
            for name, param in actual_model.state_dict().items():
                if param.is_cuda:
                    if name not in pinned_buffer or pinned_buffer[name].shape != param.shape:
                        pinned_buffer[name] = torch.empty(
                            param.shape, dtype=param.dtype,
                            pin_memory=True, device='cpu'
                        )
                    pinned_buffer[name].copy_(param, non_blocking=True)
                    model_state[name] = pinned_buffer[name]
                else:
                    model_state[name] = param.cpu().clone()

            # Copy optimizer state similarly
            opt_state = self._copy_optimizer_state_async(optimizer, pinned_buffer)

            transfer_end_event.record()

        # Create checkpoint structure (references to pinned memory)
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'global_step': step,  # Alias for compatibility with shared.py
            'model_state_dict': model_state,
            'optimizer_state_dict': opt_state,  # Standardized key
            'metrics': metrics.copy(),
            'config': self.config,
        }

        # Capture variables for closure
        save_dir = self.save_dir
        buffer_in_use = self._buffer_in_use

        def _async_save_worker():
            """Background worker: waits for GPU transfer, then saves to disk."""
            try:
                transfer_end_event.synchronize()

                # Clone from pinned memory to regular memory for saving
                final_checkpoint = {
                    'epoch': checkpoint_data['epoch'],
                    'step': checkpoint_data['step'],
                    'global_step': checkpoint_data['step'],  # Alias for compatibility
                    'model_state_dict': {k: v.clone() for k, v in checkpoint_data['model_state_dict'].items()},
                    'optimizer_state_dict': self._clone_optimizer_state(checkpoint_data['optimizer_state_dict']),
                    'metrics': checkpoint_data['metrics'],
                    'config': checkpoint_data['config'],
                }

                buffer_in_use[buffer_idx] = False

                # Save to disk
                torch.save(final_checkpoint, path)

                # Save latest
                latest_path = save_dir / 'latest_model.pt'
                torch.save(final_checkpoint, latest_path)

                # Update best if needed
                if 'val_loss' in metrics:
                    best_path = save_dir / 'best_model.pt'
                    if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                        torch.save(final_checkpoint, best_path)

            except Exception as e:
                buffer_in_use[buffer_idx] = False
                logger.error(f"Async checkpoint save failed: {e}")

        # Submit to background thread and return immediately
        future = self.save_executor.submit(_async_save_worker)
        self.pending_saves.append((future, buffer_idx, path))
        self.checkpoints.append(path)

        # Cleanup old checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            self.save_executor.submit(lambda p=old_path: p.unlink() if p.exists() else None)

        return path

    def _copy_optimizer_state_async(self, optimizer: torch.optim.Optimizer, pinned_buffer: Dict) -> Dict:
        """Copy optimizer state to pinned memory asynchronously."""
        opt_state_dict = optimizer.state_dict()
        result: Dict[str, Any] = {'state': {}, 'param_groups': opt_state_dict.get('param_groups', [])}

        for param_id, state in opt_state_dict.get('state', {}).items():
            result['state'][param_id] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    buf_key = f"opt_{param_id}_{key}"
                    if buf_key not in pinned_buffer or pinned_buffer[buf_key].shape != value.shape:
                        pinned_buffer[buf_key] = torch.empty(
                            value.shape, dtype=value.dtype,
                            pin_memory=True, device='cpu'
                        )
                    pinned_buffer[buf_key].copy_(value, non_blocking=True)
                    result['state'][param_id][key] = pinned_buffer[buf_key]
                elif isinstance(value, torch.Tensor):
                    result['state'][param_id][key] = value.cpu().clone()
                else:
                    result['state'][param_id][key] = value

        return result

    def _clone_optimizer_state(self, opt_state: Dict) -> Dict:
        """Clone optimizer state from pinned memory."""
        result: Dict[str, Any] = {'state': {}, 'param_groups': opt_state.get('param_groups', [])}

        for param_id, state in opt_state.get('state', {}).items():
            result['state'][param_id] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    result['state'][param_id][key] = value.clone()
                else:
                    result['state'][param_id][key] = value

        return result

    def _save_async_cpu(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float]
    ) -> Path:
        """Async save for CPU-only training."""
        self._cleanup_completed_saves()

        actual_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': step,  # Alias for compatibility with shared.py
            'model_state_dict': {k: v.clone() for k, v in actual_model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),  # Standardized key
            'metrics': metrics.copy(),
            'config': self.config,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        save_dir = self.save_dir

        def _save_worker():
            try:
                torch.save(checkpoint, path)
                latest_path = save_dir / 'latest_model.pt'
                torch.save(checkpoint, latest_path)
                if 'val_loss' in metrics:
                    best_path = save_dir / 'best_model.pt'
                    if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                        torch.save(checkpoint, best_path)
            except Exception as e:
                logger.error(f"Async checkpoint save failed: {e}")

        future = self.save_executor.submit(_save_worker)
        self.pending_saves.append((future, -1, path))
        self.checkpoints.append(path)

        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        return path

    def _cleanup_completed_saves(self):
        """Remove completed saves from pending list."""
        self.pending_saves = [(f, b, p) for f, b, p in self.pending_saves if not f.done()]

    def wait_for_pending_saves(self):
        """Wait for all pending async saves to complete."""
        for future, buffer_idx, _ in self.pending_saves:
            if not future.done():
                future.result()
        self.pending_saves = []

    def shutdown(self):
        """Shutdown async save executor. Call this when training is complete."""
        if self.save_executor is not None:
            self.wait_for_pending_saves()
            self.save_executor.shutdown(wait=True)
        if self._checkpoint_stream is not None:
            self._checkpoint_stream.synchronize()

    def _get_best_loss(self, best_path: Path) -> float:
        """Get best loss from existing checkpoint."""
        try:
            checkpoint = torch.load(best_path, weights_only=False)
            return checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        except Exception:
            return float('inf')

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Path
    ) -> Tuple[int, int]:
        """
        Load checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (epoch, step) from the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Handle both old and new checkpoint formats for model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        # Handle both old and new checkpoint formats for optimizer
        # Standardized key is 'optimizer_state_dict', legacy key is 'optimizer_state'
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif 'optimizer_state' in checkpoint:
            # Legacy format support
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        return checkpoint.get('epoch', 0), checkpoint.get('step', checkpoint.get('global_step', 0))

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.save_dir / 'latest_model.pt'
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.save_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
