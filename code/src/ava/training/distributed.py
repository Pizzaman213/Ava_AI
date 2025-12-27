"""
Distributed training utilities for the Ava pipeline.

Provides setup and cleanup functions for multi-GPU training with proper
error handling and graceful degradation.

Also provides OverlappedGradientSync for overlapping gradient
all-reduce with backward computation in DDP scenarios.
"""

import datetime
import logging
import os
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn

# Check distributed availability
try:
    import torch.distributed as dist
    from torch.distributed import destroy_process_group, init_process_group
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    dist = None

logger = logging.getLogger(__name__)

# Default timeout for distributed barriers (shorter for cleanup)
_CLEANUP_BARRIER_TIMEOUT = timedelta(minutes=5)


class DistributedSetupError(Exception):
    """Raised when distributed training setup fails in strict mode."""
    pass


def setup_distributed(strict: bool = False) -> Tuple[int, int]:
    """
    Setup distributed training if available.

    Args:
        strict: If True, raise DistributedSetupError on failure instead of falling back.
                Default is False for backward compatibility.

    Returns:
        Tuple of (rank, world_size). Returns (0, 1) for single-GPU training.

    Raises:
        DistributedSetupError: If strict=True and distributed setup fails.

    Note:
        - Requires RANK and WORLD_SIZE environment variables for distributed mode
        - Sets NCCL_TIMEOUT to 30 minutes if not already set
        - Falls back to single-GPU training on failure (unless strict=True)
    """
    if not DISTRIBUTED_AVAILABLE:
        msg = "Distributed training not available (torch.distributed not importable)"
        if strict:
            raise DistributedSetupError(msg)
        logger.debug(msg)
        return 0, 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Set NCCL timeout environment variable if not set
        if 'NCCL_TIMEOUT' not in os.environ:
            os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes

        try:
            # Initialize with explicit timeout to prevent hanging forever
            init_process_group(
                backend='nccl',
                timeout=datetime.timedelta(minutes=30)
            )
            logger.info(f"[Rank {rank}] Distributed setup complete: world_size={world_size}")
            return rank, world_size

        except Exception as e:
            # CRITICAL WARNING: User may not realize distributed training failed
            error_msg = (
                f"\n{'='*60}\n"
                f"DISTRIBUTED TRAINING FAILURE\n"
                f"{'='*60}\n"
                f"Distributed setup failed with error: {e}\n"
                f"Expected: world_size={world_size}, rank={rank}\n"
            )

            if strict:
                logger.error(error_msg + f"{'='*60}")
                raise DistributedSetupError(
                    f"Distributed setup failed: {e}. "
                    f"Expected world_size={world_size}, rank={rank}"
                ) from e

            logger.error(
                error_msg +
                f"Actual: Falling back to SINGLE-GPU training (world_size=1)\n"
                f"\n"
                f"This means your training will NOT use multiple GPUs!\n"
                f"Check your distributed setup if this was unexpected.\n"
                f"Set strict=True to raise an exception instead of falling back.\n"
                f"{'='*60}"
            )
            return 0, 1

    logger.debug("No RANK/WORLD_SIZE environment variables, using single-GPU mode")
    return 0, 1


def cleanup_distributed(rank: int, world_size: int) -> None:
    """
    Cleanup distributed training resources with proper synchronization.

    This ensures all CUDA operations complete and all processes synchronize
    before destroying the process group, preventing NCCL errors on shutdown.

    Args:
        rank: Current process rank
        world_size: Total number of processes

    Note:
        Only performs cleanup if world_size > 1 and distributed is available.
    """
    if DISTRIBUTED_AVAILABLE and world_size > 1:
        try:
            if dist.is_initialized():
                # Sync CUDA first to ensure all GPU operations complete
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.debug(f"[Rank {rank}] CUDA sync during cleanup failed: {e}")

                # Barrier to sync all processes before cleanup
                # May fail if other processes already crashed or timeout occurs
                try:
                    dist.barrier(timeout=_CLEANUP_BARRIER_TIMEOUT)
                except Exception as e:
                    logger.debug(f"[Rank {rank}] Barrier during cleanup failed (timeout or other processes may have exited): {e}")

                destroy_process_group()
                logger.debug(f"[Rank {rank}] Distributed cleanup complete")
        except Exception as e:
            logger.warning(f"[Rank {rank}] Error during distributed cleanup: {e}")


def is_main_process(rank: int = 0) -> bool:
    """
    Check if this is the main process (rank 0).

    Args:
        rank: Process rank to check

    Returns:
        True if rank is 0 (main process)
    """
    return rank == 0


def get_world_size() -> int:
    """
    Get the world size for distributed training.

    Returns:
        Number of processes in the distributed group, or 1 if not distributed.
    """
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Rank of current process, or 0 if not distributed.
    """
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return dist.get_rank()
    return 0


class OverlappedGradientSync:
    """
    Overlap gradient all-reduce with backward computation.

    In standard DDP, gradient synchronization happens after the backward pass
    completes, leaving the GPU idle during network communication. This class
    enables overlapping by:

    1. Registering backward hooks on model parameters
    2. Bucketing gradients by size for efficient communication
    3. Starting async all-reduce as soon as each bucket is full
    4. Overlapping communication with remaining backward computation

    Usage:
        # Create and attach to model BEFORE DDP wrapping
        gradient_sync = OverlappedGradientSync(model, bucket_size_mb=25)
        gradient_sync.register_hooks()

        # During training:
        loss.backward()  # Backward computes gradients, hooks start async all-reduce
        gradient_sync.wait_for_all_reduce()  # Wait before optimizer.step()
        optimizer.step()
        gradient_sync.clear_handles()  # Clear completed handles

    Args:
        model: Model to attach hooks to (before DDP wrapping)
        bucket_size_mb: Size of gradient buckets in MB (default 25MB)
        enabled: Whether overlapping is enabled (default True if distributed)

    Note:
        This is complementary to DDP's built-in gradient bucketing. For most
        cases, DDP's bucket_cap_mb parameter is sufficient. Use this class
        when you need finer control over gradient synchronization timing.
    """

    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: float = 25.0,
        enabled: Optional[bool] = None,
    ):
        self.model = model
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.enabled = enabled if enabled is not None else (DISTRIBUTED_AVAILABLE and dist.is_initialized())

        # Bucket management
        self._buckets: List[List[torch.Tensor]] = []  # List of buckets, each bucket is list of grads
        self._bucket_sizes: List[int] = []  # Size of each bucket in bytes
        self._current_bucket: List[torch.Tensor] = []
        self._current_bucket_size = 0

        # Parameter to bucket mapping
        self._param_to_bucket: Dict[int, int] = {}  # param_id -> bucket_index

        # Async operation handles
        self._async_handles: List[Any] = []

        # Thread safety
        self._lock = threading.Lock()

        # Stats
        self._total_syncs = 0
        self._overlapped_time_ms = 0.0

        # Hooks
        self._hooks: List[Any] = []

        if self.enabled:
            logger.info(
                f"OverlappedGradientSync initialized: bucket_size={bucket_size_mb}MB, "
                f"world_size={get_world_size()}"
            )

    def _get_param_size_bytes(self, param: torch.Tensor) -> int:
        """Get parameter size in bytes."""
        return param.numel() * param.element_size()

    def _assign_buckets(self) -> None:
        """Assign parameters to buckets based on size."""
        self._buckets = []
        self._bucket_sizes = []
        self._current_bucket = []
        self._current_bucket_size = 0
        self._param_to_bucket = {}

        # Iterate parameters in reverse order (DDP convention for backward order)
        params_with_grad = [
            (name, param) for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

        for name, param in reversed(params_with_grad):
            param_size = self._get_param_size_bytes(param)

            # If adding this param would exceed bucket size, start new bucket
            if self._current_bucket_size + param_size > self.bucket_size_bytes and self._current_bucket:
                self._buckets.append(self._current_bucket)
                self._bucket_sizes.append(self._current_bucket_size)
                self._current_bucket = []
                self._current_bucket_size = 0

            # Track bucket assignment (by param id)
            bucket_idx = len(self._buckets)
            self._param_to_bucket[id(param)] = bucket_idx

            self._current_bucket.append(param)
            self._current_bucket_size += param_size

        # Don't forget the last bucket
        if self._current_bucket:
            self._buckets.append(self._current_bucket)
            self._bucket_sizes.append(self._current_bucket_size)

        logger.debug(
            f"Assigned {sum(len(b) for b in self._buckets)} parameters to "
            f"{len(self._buckets)} buckets"
        )

    def _create_backward_hook(self, param: torch.Tensor, bucket_idx: int):
        """Create backward hook for a parameter."""
        def hook(grad: torch.Tensor) -> None:
            if not self.enabled or dist is None:
                return

            # Check if this completes a bucket
            # For simplicity, we launch async all-reduce per parameter
            # A more sophisticated implementation would track bucket completion
            with self._lock:
                # Launch async all-reduce for this gradient
                handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                if handle is not None:
                    self._async_handles.append(handle)
                self._total_syncs += 1

        return hook

    def register_hooks(self) -> None:
        """Register backward hooks on all parameters."""
        if not self.enabled:
            logger.debug("OverlappedGradientSync disabled, not registering hooks")
            return

        # Assign parameters to buckets
        self._assign_buckets()

        # Register hooks
        for bucket_idx, bucket_params in enumerate(self._buckets):
            for param in bucket_params:
                hook = self._create_backward_hook(param, bucket_idx)
                handle = param.register_post_accumulate_grad_hook(hook)
                self._hooks.append(handle)

        logger.info(f"Registered {len(self._hooks)} gradient overlap hooks")

    def unregister_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("Unregistered gradient overlap hooks")

    def wait_for_all_reduce(self) -> None:
        """Wait for all pending all-reduce operations to complete."""
        if not self.enabled:
            return

        with self._lock:
            for handle in self._async_handles:
                handle.wait()
            self._async_handles.clear()

        # Average gradients (all_reduce sums, we need mean for DDP compatibility)
        world_size = get_world_size()
        if world_size > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(world_size)

    def clear_handles(self) -> None:
        """Clear completed async handles."""
        with self._lock:
            self._async_handles = [h for h in self._async_handles if not h.is_completed()]

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            'total_syncs': self._total_syncs,
            'num_buckets': len(self._buckets),
            'bucket_sizes_mb': [s / (1024 * 1024) for s in self._bucket_sizes],
            'pending_handles': len(self._async_handles),
        }

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unregister_hooks()
        except Exception as e:
            # Log at DEBUG level since __del__ errors are expected during shutdown
            logger.debug(f"OverlappedGradientSync cleanup warning: {e}")


def create_overlapped_gradient_sync(
    model: nn.Module,
    bucket_size_mb: float = 25.0,
    enabled: bool = True,
) -> Optional[OverlappedGradientSync]:
    """
    Factory function to create OverlappedGradientSync if distributed is available.

    Args:
        model: Model to attach hooks to
        bucket_size_mb: Bucket size in MB
        enabled: Whether to enable overlapping

    Returns:
        OverlappedGradientSync instance or None if not in distributed mode
    """
    if not enabled:
        return None

    if not DISTRIBUTED_AVAILABLE or not dist.is_initialized():
        logger.debug("Distributed not initialized, skipping OverlappedGradientSync")
        return None

    world_size = get_world_size()
    if world_size <= 1:
        logger.debug("Single GPU mode, skipping OverlappedGradientSync")
        return None

    return OverlappedGradientSync(model, bucket_size_mb=bucket_size_mb, enabled=True)
