"""
Unified Checkpoint Manager for Ava Training Framework.

Provides async-capable checkpoint saving/loading with:
- CUDA stream-based non-blocking GPU->CPU transfers
- Pinned memory double buffering for overlap
- Background disk I/O via thread pool
- Support for both GPU and CPU-only training
- Automatic best/latest model tracking

Usage:
    from ava.core.checkpoint import CheckpointManager

    manager = CheckpointManager(save_dir=Path("checkpoints"), max_keep=3)
    manager.save(model, optimizer, epoch=1, step=1000, metrics={"loss": 0.5})
    epoch, step = manager.load(model, optimizer, checkpoint_path)
    manager.shutdown()  # Clean up when done
"""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ava.training.validation import ModelQualityScore

# Import optimized CUDA stream utilities
try:
    from ava.cuda.streams import (
        StreamPool, NonBlockingTransfer, get_stream_pool,
        wait_stream, efficient_sync
    )
    _CUDA_STREAMS_AVAILABLE = True
except ImportError:
    _CUDA_STREAMS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DDP and torch.compile wrappers to get the base model for saving.

    When saving checkpoints, we need the underlying model without wrappers so that:
    1. DDP's 'module.' prefix is not included in state dict keys
    2. torch.compile's '_orig_mod.' prefix is not included in state dict keys

    This ensures checkpoints are compatible with non-distributed, non-compiled models.

    Args:
        model: Model that may be wrapped with DDP and/or torch.compile

    Returns:
        The unwrapped base model
    """
    # Unwrap DDP (DistributedDataParallel wraps model in .module)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    # Unwrap torch.compile (OptimizedModule stores original model at ._orig_mod)
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def remap_state_dict_keys(
    state_dict: Dict[str, Any],
    expected_keys: Optional[set] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Remap corrupted checkpoint keys to match expected model architecture.

    This handles common key corruption patterns from architecture changes:
    - layers.X.attention.attention.Y -> layers.X.attention.Y (double attention)
    - layers.X.attention.{q,k,v,o}_proj.attention.Y -> layers.X.attention.{q,k,v,o}_proj.Y
    - layers.X.attention.rope.attention.Y -> layers.X.attention.rope.Y
    - layers.X.attention._wrapped_attn.Y -> layers.X.attention.Y (hybrid caching wrapper)

    Args:
        state_dict: The checkpoint state dict with potentially corrupted keys
        expected_keys: Optional set of expected keys for filtering. If None, all remapped keys are kept.

    Returns:
        Tuple of (remapped_state_dict, remapped_count)
    """
    remapped_state_dict = {}
    remapped_count = 0

    for old_key, value in state_dict.items():
        new_key = old_key

        # Fix: layers.X.attention._wrapped_attn.Y -> layers.X.attention.Y
        # This handles keys from hybrid caching wrapper
        if '._wrapped_attn.' in old_key:
            new_key = old_key.replace('._wrapped_attn.', '.')
            remapped_count += 1

        # Fix: layers.X.attention.attention.Y -> layers.X.attention.Y
        elif '.attention.attention.' in old_key:
            new_key = old_key.replace('.attention.attention.', '.attention.')
            remapped_count += 1

        # Fix: layers.X.attention.{q,k,v,o}_proj.attention.Y -> layers.X.attention.{q,k,v,o}_proj.Y
        # This handles corrupted keys like: layers.0.attention.q_proj.attention.weight
        elif '.attention.' in old_key and old_key.count('.attention.') > 1:
            # Remove spurious '.attention.' after projection names
            new_key = re.sub(
                r'\.(q_proj|k_proj|v_proj|o_proj)\.attention\.',
                r'.\1.',
                old_key
            )
            if new_key != old_key:
                remapped_count += 1
            else:
                # Fallback: remove duplicate adjacent 'attention' segments
                parts = old_key.split('.')
                cleaned_parts = []
                prev_was_attention = False
                for part in parts:
                    if part == 'attention' and prev_was_attention:
                        continue  # Skip duplicate
                    cleaned_parts.append(part)
                    prev_was_attention = (part == 'attention')
                new_key = '.'.join(cleaned_parts)
                if new_key != old_key:
                    remapped_count += 1

        # Fix: layers.X.attention.rope.attention.inv_freq -> layers.X.attention.rope.inv_freq
        if '.rope.attention.' in new_key:
            new_key = new_key.replace('.rope.attention.', '.rope.')
            remapped_count += 1

        # Keep the key if it matches expected keys or if no expected keys provided
        if expected_keys is None:
            remapped_state_dict[new_key] = value
        elif new_key in expected_keys:
            remapped_state_dict[new_key] = value
        elif old_key in expected_keys:
            remapped_state_dict[old_key] = value

    return remapped_state_dict, remapped_count


def load_state_dict_with_remapping(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = True
) -> Tuple[bool, Optional[Any]]:
    """
    Load state dict into model with automatic key remapping for backwards compatibility.

    This is a utility function that can be used by any code loading checkpoints.
    - If strict=True: tries strict loading first, then remaps on failure
    - If strict=False: always remaps first (handles _wrapped_attn, corrupted keys, etc)
      then loads with non-strict mode

    Handles common key corruption patterns:
    - layers.X.attention._wrapped_attn.Y -> layers.X.attention.Y (hybrid caching)
    - layers.X.attention.attention.Y -> layers.X.attention.Y (double attention)
    - layers.X.attention.{q,k,v,o}_proj.attention.Y -> layers.X.attention.{q,k,v,o}_proj.Y
    - And other architecture change mismatches

    Args:
        model: The model to load state into
        state_dict: The state dict to load
        strict: If True, try strict loading first then remap. If False, always remap first.

    Returns:
        Tuple of (success: bool, load_result: IncompatibleKeys or None)
    """
    # If strict=False, always remap first to handle _wrapped_attn and other key mismatches
    if not strict:
        expected_keys = set(model.state_dict().keys())
        remapped_state_dict, remapped_count = remap_state_dict_keys(state_dict, expected_keys)

        if remapped_count > 0:
            logger.debug(f"Remapped {remapped_count} keys for backwards compatibility")
            state_dict = remapped_state_dict

        # Load with non-strict mode
        result = model.load_state_dict(state_dict, strict=False)
        return True, result

    # If strict=True, try strict loading first
    try:
        model.load_state_dict(state_dict, strict=True)
        return True, None
    except RuntimeError as e:
        logger.warning(
            f"Checkpoint loading with strict=True failed: {e}\n"
            f"Attempting to remap keys for backwards compatibility..."
        )

        # Get expected keys and remap
        expected_keys = set(model.state_dict().keys())
        remapped_state_dict, remapped_count = remap_state_dict_keys(state_dict, expected_keys)

        if remapped_count > 0:
            logger.info(f"Remapped {remapped_count} keys for backwards compatibility")

        # Load remapped state dict
        result = model.load_state_dict(remapped_state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys after remap: {len(result.missing_keys)} keys")
            logger.debug(f"Missing keys: {result.missing_keys[:10]}...")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys after remap: {len(result.unexpected_keys)} keys")
            logger.debug(f"Unexpected keys: {result.unexpected_keys[:10]}...")

        return True, result


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
        async_save: bool = True,
        raise_on_failure: bool = False,
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep (oldest deleted first)
            config: Optional config dict to save with checkpoints
            async_save: Enable async saving (recommended for GPU training)
            raise_on_failure: If True, raise exceptions on save failures instead of logging
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints: list = []
        self.config = config
        self.async_save = async_save
        self.raise_on_failure = raise_on_failure
        self._save_errors: list = []  # Track errors for later inspection

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
        # FIX: Add per-buffer locks for proper synchronization during async saves
        self._buffer_write_locks = [threading.Lock(), threading.Lock()]

        # Cache best loss/quality to avoid loading checkpoint from disk on every save
        self._cached_best_loss: Optional[float] = None
        self._cached_best_quality: Optional[float] = None
        self._best_cache_lock = threading.Lock()
        self._init_best_cache()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        quality_score: Optional['ModelQualityScore'] = None,
    ) -> Path:
        """
        Save checkpoint in format compatible with generate.py.

        Args:
            model: Model to save (handles DDP automatically)
            optimizer: Optimizer to save
            epoch: Current epoch number
            step: Current step number
            metrics: Dictionary of metrics to save
            quality_score: Optional ModelQualityScore for multi-metric selection

        Returns:
            Path to saved checkpoint
        """
        if self.async_save and self.save_executor is not None and torch.cuda.is_available():
            return self._save_truly_async(model, optimizer, epoch, step, metrics, quality_score)
        elif self.async_save and self.save_executor is not None:
            return self._save_async_cpu(model, optimizer, epoch, step, metrics, quality_score)
        else:
            return self._save_sync(model, optimizer, epoch, step, metrics, quality_score)

    def _save_sync(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        quality_score: Optional['ModelQualityScore'] = None,
    ) -> Path:
        """Synchronous checkpoint save (original behavior)."""
        # Handle DDP models
        actual_model = _unwrap_model(model)

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': step,  # Alias for compatibility with shared.py
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # Standardized key
            'metrics': metrics,
            'config': self.config,
            'quality_score': quality_score.to_dict() if quality_score else None,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)

        # Always save latest
        latest_path = self.save_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)

        # Save best if quality score improved (or fallback to val_loss)
        best_path = self.save_dir / 'best_model.pt'
        should_save_best = False

        if quality_score is not None:
            # Use quality score for best model selection
            current_quality = quality_score.quality_score
            best_quality = self._get_best_quality_score(best_path)
            should_save_best = current_quality > best_quality
        elif 'val_loss' in metrics:
            # Fallback to val_loss
            should_save_best = not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path)

        if should_save_best:
            torch.save(checkpoint, best_path)
            # Update cache with new best values
            self._update_best_cache(
                loss=metrics.get('val_loss'),
                quality=quality_score.quality_score if quality_score else None
            )

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
        metrics: Dict[str, float],
        quality_score: Optional['ModelQualityScore'] = None,
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
        actual_model = _unwrap_model(model)

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
                            error_msg = f"Previous checkpoint save failed: {e}"
                            self._save_errors.append(e)
                            if self.raise_on_failure:
                                raise RuntimeError(error_msg) from e
                            logger.error(error_msg)

            self._buffer_in_use[buffer_idx] = True

        # FIX: Acquire per-buffer write lock before accessing buffer
        # This prevents race conditions where async save is still reading buffer
        # while main thread starts writing new data
        self._buffer_write_locks[buffer_idx].acquire()
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
            'quality_score': quality_score.to_dict() if quality_score else None,
        }

        # Capture variables for closure
        save_dir = self.save_dir
        buffer_in_use = self._buffer_in_use
        buffer_write_locks = self._buffer_write_locks

        # Capture quality_score for closure
        qs = quality_score

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
                    'quality_score': checkpoint_data['quality_score'],
                }

                # FIX: Release write lock AFTER cloning from pinned buffer
                # This ensures main thread can't overwrite buffer while we're reading
                buffer_write_locks[buffer_idx].release()
                buffer_in_use[buffer_idx] = False

                # Save to disk
                torch.save(final_checkpoint, path)

                # Save latest
                latest_path = save_dir / 'latest_model.pt'
                torch.save(final_checkpoint, latest_path)

                # Update best if needed (use quality score if available)
                best_path = save_dir / 'best_model.pt'
                should_save_best = False

                if qs is not None:
                    current_quality = qs.quality_score
                    best_quality = self._get_best_quality_score(best_path)
                    should_save_best = current_quality > best_quality
                elif 'val_loss' in metrics:
                    should_save_best = not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path)

                if should_save_best:
                    torch.save(final_checkpoint, best_path)
                    # Update cache with new best values
                    self._update_best_cache(
                        loss=metrics.get('val_loss'),
                        quality=qs.quality_score if qs else None
                    )

            except Exception as e:
                # FIX: Always release lock on error
                try:
                    buffer_write_locks[buffer_idx].release()
                except RuntimeError:
                    pass  # Already released
                buffer_in_use[buffer_idx] = False
                self._save_errors.append(e)
                logger.error(f"Async checkpoint save failed: {e}")
                # Re-raise to propagate to future.result()
                raise

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
        metrics: Dict[str, float],
        quality_score: Optional['ModelQualityScore'] = None,
    ) -> Path:
        """Async save for CPU-only training."""
        self._cleanup_completed_saves()

        actual_model = _unwrap_model(model)

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': step,  # Alias for compatibility with shared.py
            'model_state_dict': {k: v.clone() for k, v in actual_model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),  # Standardized key
            'metrics': metrics.copy(),
            'config': self.config,
            'quality_score': quality_score.to_dict() if quality_score else None,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        save_dir = self.save_dir
        qs = quality_score

        def _save_worker():
            try:
                torch.save(checkpoint, path)
                latest_path = save_dir / 'latest_model.pt'
                torch.save(checkpoint, latest_path)

                # Update best if needed (use quality score if available)
                best_path = save_dir / 'best_model.pt'
                should_save_best = False

                if qs is not None:
                    current_quality = qs.quality_score
                    best_quality = self._get_best_quality_score(best_path)
                    should_save_best = current_quality > best_quality
                elif 'val_loss' in metrics:
                    should_save_best = not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path)

                if should_save_best:
                    torch.save(checkpoint, best_path)
                    # Update cache with new best values
                    self._update_best_cache(
                        loss=metrics.get('val_loss'),
                        quality=qs.quality_score if qs else None
                    )
            except Exception as e:
                self._save_errors.append(e)
                logger.error(f"Async checkpoint save failed: {e}")
                # Re-raise to propagate to future.result()
                raise

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

    def flush_pending_saves(self, timeout: float = 60.0) -> bool:
        """
        Wait for all pending async checkpoint saves to complete with timeout.

        This method ensures checkpoint durability by blocking until all async
        save operations complete. Should be called before validation, training
        termination, or any operation that depends on checkpoint consistency.

        Args:
            timeout: Maximum seconds to wait for all saves to complete (default: 60s)

        Returns:
            True if all saves completed successfully, False if timeout or errors

        Example:
            >>> checkpoint_manager.save(model, optimizer, epoch=1)
            >>> # Ensure save completes before validation
            >>> if not checkpoint_manager.flush_pending_saves(timeout=30.0):
            ...     logger.error("Checkpoint save incomplete!")
        """
        import time

        if not self.pending_saves:
            return True  # No pending saves

        start_time = time.time()
        pending_count = len(self.pending_saves)

        logger.info(f"Waiting for {pending_count} pending checkpoint saves...")

        completed_saves = []
        failed_saves = []

        for future, buffer_idx, save_path in self.pending_saves:
            remaining_time = timeout - (time.time() - start_time)

            if remaining_time <= 0:
                logger.error(
                    f"Timeout waiting for checkpoint saves! "
                    f"{len(completed_saves)}/{pending_count} completed, "
                    f"{len(failed_saves)} failed"
                )
                return False

            try:
                # Wait for this save to complete with remaining timeout
                future.result(timeout=remaining_time)
                completed_saves.append(save_path)
                logger.debug(f"Checkpoint save completed: {save_path}")
            except TimeoutError:
                logger.error(f"Checkpoint save timeout after {timeout}s: {save_path}")
                failed_saves.append((save_path, "timeout"))
            except Exception as e:
                logger.error(f"Checkpoint save failed: {save_path} - {e}")
                failed_saves.append((save_path, str(e)))

        # Clear pending saves list
        self.pending_saves = []

        elapsed = time.time() - start_time

        if failed_saves:
            logger.error(
                f"Checkpoint save failures: {len(failed_saves)}/{pending_count} failed, "
                f"{len(completed_saves)} completed in {elapsed:.2f}s"
            )
            for save_path, error in failed_saves:
                logger.error(f"  - {save_path}: {error}")
            return False

        logger.info(
            f"All {pending_count} checkpoint saves completed successfully in {elapsed:.2f}s"
        )
        return True

    def shutdown(self):
        """Shutdown async save executor. Call this when training is complete."""
        if self.save_executor is not None:
            self.wait_for_pending_saves()
            self.save_executor.shutdown(wait=True)
        if self._checkpoint_stream is not None:
            self._checkpoint_stream.synchronize()

    def _init_best_cache(self) -> None:
        """Initialize the best loss/quality cache from existing checkpoint on disk."""
        best_path = self.save_dir / 'best_model.pt'
        if not best_path.exists():
            self._cached_best_loss = float('inf')
            self._cached_best_quality = 0.0
            return

        try:
            checkpoint = torch.load(best_path, weights_only=False)

            # Cache best loss
            self._cached_best_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))

            # Cache best quality
            qs = checkpoint.get('quality_score')
            if qs and isinstance(qs, dict) and 'quality_score' in qs:
                self._cached_best_quality = qs['quality_score']
            elif self._cached_best_loss != float('inf'):
                # Fallback: compute from val_loss
                self._cached_best_quality = max(0.0, 1.0 - min(self._cached_best_loss / 10.0, 1.0))
            else:
                self._cached_best_quality = 0.0

            logger.debug(f"Initialized best cache: loss={self._cached_best_loss}, quality={self._cached_best_quality}")
        except Exception as e:
            logger.warning(f"Could not load best checkpoint for cache initialization: {e}")
            self._cached_best_loss = float('inf')
            self._cached_best_quality = 0.0

    def _update_best_cache(self, loss: Optional[float], quality: Optional[float]) -> None:
        """Update the cached best loss/quality values (thread-safe)."""
        with self._best_cache_lock:
            if loss is not None:
                self._cached_best_loss = loss
            if quality is not None:
                self._cached_best_quality = quality

    def _get_best_loss(self, best_path: Path) -> float:
        """Get best loss from cache (fast, no disk I/O)."""
        with self._best_cache_lock:
            if self._cached_best_loss is not None:
                return self._cached_best_loss
        # Fallback: load from disk if cache not initialized
        return float('inf')

    def _get_best_quality_score(self, best_path: Path) -> float:
        """Get best quality score from cache (fast, no disk I/O)."""
        with self._best_cache_lock:
            if self._cached_best_quality is not None:
                return self._cached_best_quality
        # Fallback: return default if cache not initialized
        return 0.0

    def get_save_errors(self) -> list:
        """Get list of save errors that occurred during async saves."""
        return list(self._save_errors)

    def has_save_errors(self) -> bool:
        """Check if any save errors occurred."""
        return len(self._save_errors) > 0

    def clear_save_errors(self) -> None:
        """Clear the list of save errors."""
        self._save_errors.clear()

    def _validate_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        opt_state_dict: dict
    ) -> bool:
        """
        Validate that optimizer state matches model parameters.

        CRITICAL: This prevents the bug where optimizer state has different number
        of parameters than the model, causing non-coherent generation.

        Args:
            model: The model being trained
            optimizer: The optimizer
            opt_state_dict: Optimizer state dict to validate

        Returns:
            True if validation passes, False otherwise
        """
        # Count model parameters that require gradients
        model_params = [p for p in model.parameters() if p.requires_grad]
        num_model_params = len(model_params)

        # Count optimizer state entries
        num_opt_states = len(opt_state_dict.get('state', {}))

        # Check if counts match
        if num_opt_states != num_model_params:
            logger.error(
                f"❌ OPTIMIZER STATE MISMATCH DETECTED:\n"
                f"   Model parameters (requires_grad): {num_model_params}\n"
                f"   Optimizer state entries: {num_opt_states}\n"
                f"   Missing: {num_model_params - num_opt_states} parameters\n"
                f"\n"
                f"   This mismatch causes NON-COHERENT generation because:\n"
                f"   - Missing parameters have no momentum buffers\n"
                f"   - They train without gradient history\n"
                f"   - Model becomes internally inconsistent\n"
                f"\n"
                f"   SOLUTION: Rejecting corrupted optimizer state and using fresh state."
            )
            return False

        # Validate param_groups structure
        if 'param_groups' not in opt_state_dict:
            logger.error("❌ Optimizer state missing 'param_groups' - state is corrupted")
            return False

        # Check for NaN/Inf in optimizer state (8-bit optimizer corruption check)
        state_dict = opt_state_dict.get('state', {})
        for param_id, state in state_dict.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        logger.error(f"❌ NaN detected in optimizer state[{param_id}][{key}]")
                        return False
                    if torch.isinf(value).any():
                        logger.error(f"❌ Inf detected in optimizer state[{param_id}][{key}]")
                        return False

        logger.info(f"✓ Optimizer state validation passed ({num_opt_states} parameters)")
        return True

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

        # Get the state dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # Handle DDP module prefix mismatch
        # DDP wraps models with 'module.' prefix - we need to match checkpoint format to model format
        model_is_ddp = hasattr(model, 'module')
        checkpoint_is_ddp = any(k.startswith('module.') for k in state_dict.keys())

        if model_is_ddp and not checkpoint_is_ddp:
            # Model is DDP but checkpoint is not - add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            logger.info("Checkpoint loaded: Added 'module.' prefix for DDP model")
        elif not model_is_ddp and checkpoint_is_ddp:
            # Model is not DDP but checkpoint is - remove 'module.' prefix
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            logger.info("Checkpoint loaded: Removed 'module.' prefix for non-DDP model")

        # Load with automatic key remapping for backwards compatibility
        load_state_dict_with_remapping(model, state_dict, strict=True)

        # Handle both old and new checkpoint formats for optimizer
        # Standardized key is 'optimizer_state_dict', legacy key is 'optimizer_state'
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            # CRITICAL FIX: Validate optimizer state matches model before loading
            if self._validate_optimizer_state(model, optimizer, opt_state):
                try:
                    optimizer.load_state_dict(opt_state)
                    logger.info("✓ Optimizer state loaded and validated successfully")
                except Exception as e:
                    logger.error(f"Failed to load optimizer state: {e}")
                    logger.warning("⚠ Resetting optimizer state - training will continue with fresh optimizer")
            else:
                logger.warning("⚠ Optimizer state validation failed - using fresh optimizer state")
                logger.warning("  This will cause the model to 'forget' momentum and require warmup")
        elif 'optimizer_state' in checkpoint:
            # Legacy format support
            opt_state = checkpoint['optimizer_state']
            if self._validate_optimizer_state(model, optimizer, opt_state):
                try:
                    optimizer.load_state_dict(opt_state)
                    logger.info("✓ Optimizer state loaded (legacy format) and validated successfully")
                except Exception as e:
                    logger.error(f"Failed to load optimizer state: {e}")
                    logger.warning("⚠ Resetting optimizer state - training will continue with fresh optimizer")
            else:
                logger.warning("⚠ Optimizer state validation failed - using fresh optimizer state")

        return checkpoint.get('epoch', 0), checkpoint.get('step', checkpoint.get('global_step', 0))

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.save_dir / 'latest_model.pt'
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.save_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
