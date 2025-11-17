"""
CUDA stream utilities for async GPU-CPU transfers.

This module provides utilities for managing CUDA streams to enable
overlapping computation and data transfer operations.

Key features:
- Dual-stream management (compute + transfer)
- Pinned memory allocation for fast transfers
- Stream synchronization helpers
- Transfer queue management
"""

import torch
from typing import Optional, List, Dict, Any
from collections import deque
import threading


class CUDAStreamManager:
    """
    Manages CUDA streams for async operations.

    Provides separate streams for:
    - Computation (main work)
    - Transfers (GPU↔CPU data movement)

    This allows overlapping compute and transfer to hide latency.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize CUDA stream manager.

        Args:
            device: CUDA device to use (default: current device)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.is_cuda = device.type == 'cuda'

        if self.is_cuda:
            # Create streams
            self.compute_stream = torch.cuda.current_stream(device)
            self.transfer_stream = torch.cuda.Stream(device)

            # Transfer queue for async operations
            self.pending_transfers = deque()
            self.transfer_lock = threading.Lock()
        else:
            # CPU fallback - no stream management needed
            self.compute_stream = None
            self.transfer_stream = None

    def to_gpu_async(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        Transfer tensor from CPU to GPU asynchronously.

        Args:
            tensor: CPU tensor to transfer
            non_blocking: Use non-blocking transfer (requires pinned memory)

        Returns:
            GPU tensor (transfer may still be in progress)
        """
        if not self.is_cuda:
            return tensor

        with torch.cuda.stream(self.transfer_stream):
            gpu_tensor = tensor.to(self.device, non_blocking=non_blocking)

        return gpu_tensor

    def to_cpu_async(
        self,
        tensor: torch.Tensor,
        pin_memory: bool = True
    ) -> torch.Tensor:
        """
        Transfer tensor from GPU to CPU asynchronously.

        Args:
            tensor: GPU tensor to transfer
            pin_memory: Pin the CPU tensor for faster future transfers

        Returns:
            CPU tensor (transfer may still be in progress)
        """
        if not self.is_cuda:
            return tensor

        with torch.cuda.stream(self.transfer_stream):
            cpu_tensor = tensor.cpu()
            if pin_memory:
                cpu_tensor = cpu_tensor.pin_memory()

        return cpu_tensor

    def synchronize_transfer(self):
        """Wait for all pending transfers to complete."""
        if self.is_cuda and self.transfer_stream is not None:
            self.transfer_stream.synchronize()

    def synchronize_compute(self):
        """Wait for all pending compute operations to complete."""
        if self.is_cuda and self.compute_stream is not None:
            self.compute_stream.synchronize()

    def wait_for_transfers(self):
        """Make compute stream wait for transfer stream to complete."""
        if self.is_cuda:
            self.compute_stream.wait_stream(self.transfer_stream)

    def wait_for_compute(self):
        """Make transfer stream wait for compute stream to complete."""
        if self.is_cuda:
            self.transfer_stream.wait_stream(self.compute_stream)


class PinnedMemoryPool:
    """
    Pool of pinned CPU memory for fast GPU transfers.

    Pinned (page-locked) memory enables:
    - DMA transfers (Direct Memory Access)
    - Non-blocking GPU↔CPU transfers
    - 2-3x faster transfer speeds

    But it's a limited resource, so we pool and reuse it.
    """

    def __init__(self, max_size_mb: int = 512):
        """
        Initialize pinned memory pool.

        Args:
            max_size_mb: Maximum pool size in MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.pool: Dict[tuple, List[torch.Tensor]] = {}
        self.current_size_bytes = 0
        self.lock = threading.Lock()

    def allocate(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate pinned tensor from pool.

        Args:
            shape: Tensor shape
            dtype: Tensor dtype

        Returns:
            Pinned CPU tensor
        """
        key = (shape, dtype)

        with self.lock:
            # Try to reuse from pool
            if key in self.pool and len(self.pool[key]) > 0:
                tensor = self.pool[key].pop()
                tensor.zero_()  # Clear data
                return tensor

            # Allocate new pinned tensor
            tensor = torch.zeros(shape, dtype=dtype).pin_memory()
            tensor_size = tensor.numel() * tensor.element_size()

            # Check pool size limit
            if self.current_size_bytes + tensor_size > self.max_size_bytes:
                # Pool full - don't track this tensor
                return tensor

            self.current_size_bytes += tensor_size
            return tensor

    def release(self, tensor: torch.Tensor):
        """
        Return tensor to pool for reuse.

        Args:
            tensor: Pinned tensor to return
        """
        if not tensor.is_pinned():
            return  # Not a pinned tensor, can't pool it

        key = (tuple(tensor.shape), tensor.dtype)

        with self.lock:
            if key not in self.pool:
                self.pool[key] = []

            # Limit pool entries per key
            if len(self.pool[key]) < 10:
                self.pool[key].append(tensor)

    def clear(self):
        """Clear the pool and free memory."""
        with self.lock:
            self.pool.clear()
            self.current_size_bytes = 0
            torch.cuda.empty_cache()


# Global instances for convenience
_default_stream_manager: Optional[CUDAStreamManager] = None
_default_pinned_pool: Optional[PinnedMemoryPool] = None


def get_stream_manager() -> CUDAStreamManager:
    """Get or create the default CUDA stream manager."""
    global _default_stream_manager
    if _default_stream_manager is None:
        _default_stream_manager = CUDAStreamManager()
    return _default_stream_manager


def get_pinned_pool() -> PinnedMemoryPool:
    """Get or create the default pinned memory pool."""
    global _default_pinned_pool
    if _default_pinned_pool is None:
        _default_pinned_pool = PinnedMemoryPool()
    return _default_pinned_pool


def async_to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convenience function: Transfer tensor to GPU asynchronously.

    Args:
        tensor: CPU tensor

    Returns:
        GPU tensor
    """
    return get_stream_manager().to_gpu_async(tensor)


def async_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convenience function: Transfer tensor to CPU asynchronously.

    Args:
        tensor: GPU tensor

    Returns:
        CPU tensor (pinned)
    """
    return get_stream_manager().to_cpu_async(tensor)


def sync_transfers():
    """Convenience function: Wait for all async transfers to complete."""
    get_stream_manager().synchronize_transfer()
