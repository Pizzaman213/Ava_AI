"""
CUDA Stream Optimization Utilities

This module provides optimized CUDA stream management to minimize synchronization overhead:
- StreamPool: Reusable stream pool to avoid stream creation overhead
- CUDATimer: Event-based timing without full synchronization
- Non-blocking transfer utilities
- Batch synchronization helpers

Key principles:
1. Avoid torch.cuda.synchronize() - it blocks CPU until ALL GPU work completes
2. Use CUDA events for timing - they only sync specific operations
3. Use stream.wait_stream() for inter-stream dependencies
4. Reuse streams via pools - stream creation has overhead
5. Use non_blocking=True for all CPU<->GPU transfers
"""

import torch
import torch.cuda
from typing import Optional, List, Dict, Any, TypeVar
from contextlib import contextmanager
import threading
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StreamPool:
    """
    Pool of reusable CUDA streams to avoid stream creation overhead.

    Creating a new CUDA stream has non-trivial overhead. This pool maintains
    a fixed set of streams that are reused in round-robin fashion.

    Thread-safe for multi-threaded usage.

    Example:
        >>> pool = StreamPool(num_streams=4)
        >>> with pool.get_stream() as stream:
        ...     with torch.cuda.stream(stream):
        ...         output = model(input)
    """

    def __init__(self, num_streams: int = 4, high_priority: bool = False):
        """
        Initialize stream pool.

        Args:
            num_streams: Number of streams to maintain in pool
            high_priority: Use high-priority streams (for latency-critical work)
        """
        self.num_streams = num_streams
        self.high_priority = high_priority
        self._streams: List[torch.cuda.Stream] = []
        self._idx = 0
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize streams on first use (must be on CUDA device)."""
        if self._initialized or not torch.cuda.is_available():
            return

        with self._lock:
            if self._initialized:
                return
            priority = -1 if self.high_priority else 0
            self._streams = [
                torch.cuda.Stream(priority=priority)
                for _ in range(self.num_streams)
            ]
            self._initialized = True
            logger.debug(f"Initialized StreamPool with {self.num_streams} streams")

    def get_stream(self) -> torch.cuda.Stream:
        """
        Get next available stream from pool (round-robin).

        Returns:
            CUDA stream from pool
        """
        self._ensure_initialized()
        if not self._streams:
            return torch.cuda.default_stream()

        with self._lock:
            stream = self._streams[self._idx]
            self._idx = (self._idx + 1) % self.num_streams
        return stream

    @contextmanager
    def stream_context(self):
        """
        Context manager for using a pooled stream.

        Example:
            >>> with pool.stream_context() as stream:
            ...     # Operations run on pooled stream
            ...     output = model(input)
        """
        stream = self.get_stream()
        with torch.cuda.stream(stream):
            yield stream

    def synchronize_all(self):
        """Synchronize all streams in pool."""
        for stream in self._streams:
            stream.synchronize()


class CUDATimer:
    """
    Event-based CUDA timing without full synchronization.

    Uses CUDA events to measure GPU time accurately without blocking
    the CPU unnecessarily. Much more efficient than torch.cuda.synchronize()
    + time.time().

    Example:
        >>> timer = CUDATimer()
        >>> timer.start()
        >>> output = model(input)
        >>> elapsed_ms = timer.stop()  # Returns time in milliseconds
    """

    def __init__(self, enable_timing: bool = True):
        """
        Initialize CUDA timer.

        Args:
            enable_timing: Enable timing on events (small overhead)
        """
        self.enable_timing = enable_timing
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._elapsed_ms: Optional[float] = None

    def start(self, stream: Optional[torch.cuda.Stream] = None):
        """
        Record start time.

        Args:
            stream: Stream to record on (default: current stream)
        """
        if not torch.cuda.is_available():
            return

        self._start_event = torch.cuda.Event(enable_timing=self.enable_timing)
        self._end_event = torch.cuda.Event(enable_timing=self.enable_timing)
        self._elapsed_ms = None

        if stream is not None:
            self._start_event.record(stream)
        else:
            self._start_event.record()

    def stop(self, stream: Optional[torch.cuda.Stream] = None, sync: bool = True) -> float:
        """
        Record end time and optionally compute elapsed time.

        Args:
            stream: Stream to record on (default: current stream)
            sync: Whether to synchronize and return elapsed time immediately

        Returns:
            Elapsed time in milliseconds (if sync=True), else 0.0
        """
        if not torch.cuda.is_available() or self._start_event is None:
            return 0.0

        if stream is not None:
            self._end_event.record(stream)
        else:
            self._end_event.record()

        if sync:
            self._end_event.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
            return self._elapsed_ms
        return 0.0

    def elapsed_time(self) -> float:
        """
        Get elapsed time (synchronizes if not already done).

        Returns:
            Elapsed time in milliseconds
        """
        if self._elapsed_ms is not None:
            return self._elapsed_ms

        if self._end_event is None or self._start_event is None:
            return 0.0

        self._end_event.synchronize()
        self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        return self._elapsed_ms

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_time() / 1000.0


@contextmanager
def cuda_timed_region(name: str = "", log: bool = False):
    """
    Context manager for timing a CUDA region with events.

    More efficient than synchronize() + time.time().

    Args:
        name: Name for logging
        log: Whether to log the elapsed time

    Yields:
        CUDATimer instance (call .elapsed_time() after context exits)

    Example:
        >>> with cuda_timed_region("forward_pass", log=True) as timer:
        ...     output = model(input)
        >>> print(f"Took {timer.elapsed_ms:.2f} ms")
    """
    timer = CUDATimer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop(sync=True)
        if log and name:
            logger.info(f"[CUDA Timer] {name}: {timer.elapsed_time():.2f} ms")


class NonBlockingTransfer:
    """
    Utilities for efficient non-blocking CPU<->GPU transfers.

    Key optimizations:
    1. Always use non_blocking=True
    2. Use pinned memory for faster transfers
    3. Use dedicated transfer stream to overlap with compute
    """

    def __init__(self, use_pinned_memory: bool = True):
        """
        Initialize transfer utilities.

        Args:
            use_pinned_memory: Pin CPU tensors for faster transfer
        """
        self.use_pinned_memory = use_pinned_memory
        self._transfer_stream: Optional[torch.cuda.Stream] = None

    def _ensure_stream(self):
        """Lazily create transfer stream."""
        if self._transfer_stream is None and torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream(priority=-1)

    def to_gpu(
        self,
        tensor: torch.Tensor,
        device: Optional[torch.device] = None,
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        Transfer tensor to GPU with optimal settings.

        Args:
            tensor: CPU tensor to transfer
            device: Target device (default: current CUDA device)
            non_blocking: Use non-blocking transfer

        Returns:
            GPU tensor
        """
        if not torch.cuda.is_available():
            return tensor

        if device is None:
            device = torch.cuda.current_device()

        if tensor.device.type == 'cpu' and self.use_pinned_memory and not tensor.is_pinned():
            tensor = tensor.pin_memory()

        return tensor.to(device, non_blocking=non_blocking)

    def to_cpu(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True,
        pin_memory: bool = False
    ) -> torch.Tensor:
        """
        Transfer tensor to CPU with optimal settings.

        Args:
            tensor: GPU tensor to transfer
            non_blocking: Use non-blocking transfer
            pin_memory: Pin the resulting CPU tensor

        Returns:
            CPU tensor
        """
        cpu_tensor = tensor.to('cpu', non_blocking=non_blocking)

        if pin_memory and not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()

        return cpu_tensor

    def transfer_dict_to_cpu(
        self,
        state_dict: Dict[str, Any],
        non_blocking: bool = True
    ) -> Dict[str, Any]:
        """
        Transfer entire state dict to CPU efficiently.

        Uses dedicated stream to overlap transfers.

        Args:
            state_dict: Dictionary with tensor values
            non_blocking: Use non-blocking transfers

        Returns:
            State dict with CPU tensors
        """
        self._ensure_stream()

        cpu_dict = {}

        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        cpu_dict[key] = value.to('cpu', non_blocking=non_blocking)
                    elif isinstance(value, dict):
                        cpu_dict[key] = self.transfer_dict_to_cpu(value, non_blocking)
                    else:
                        cpu_dict[key] = value
        else:
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    cpu_dict[key] = value.to('cpu', non_blocking=non_blocking)
                elif isinstance(value, dict):
                    cpu_dict[key] = self.transfer_dict_to_cpu(value, non_blocking)
                else:
                    cpu_dict[key] = value

        return cpu_dict

    def wait_for_transfer(self):
        """Wait for all pending transfers to complete."""
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()


class BatchSynchronizer:
    """
    Batch multiple operations before synchronizing.

    Instead of synchronizing after each metric/value extraction,
    batch them together and sync once at the end.

    Example:
        >>> sync = BatchSynchronizer()
        >>> sync.add(loss)
        >>> sync.add(accuracy)
        >>> sync.add(perplexity)
        >>> values = sync.get_values()  # Single sync for all values
    """

    def __init__(self):
        self._tensors: List[torch.Tensor] = []
        self._synced = False
        self._values: List[float] = []

    def add(self, tensor: torch.Tensor):
        """Add a tensor to batch (will be detached)."""
        self._tensors.append(tensor.detach())
        self._synced = False

    def get_values(self) -> List[float]:
        """
        Get all values with single synchronization.

        GPU SYNC FIX: Stack all tensors and transfer with single .tolist() call
        instead of N separate .item() calls. This reduces N cudaStreamSynchronize
        calls to just 1.

        Returns:
            List of scalar values
        """
        if not self._synced:
            if self._tensors:
                # Stack all scalar tensors and transfer in single operation
                # This is O(1) syncs instead of O(N) syncs
                stacked = torch.stack(self._tensors)
                self._values = stacked.tolist()
            else:
                self._values = []
            self._synced = True
        return self._values

    def clear(self):
        """Clear batch for reuse."""
        self._tensors.clear()
        self._values.clear()
        self._synced = False


def wait_stream(target_stream: torch.cuda.Stream, source_stream: torch.cuda.Stream):
    """
    Make target stream wait for source stream (instead of full sync).

    This is much more efficient than torch.cuda.synchronize() when you
    only need to ensure one stream's work completes before another starts.

    Args:
        target_stream: Stream that should wait
        source_stream: Stream to wait for
    """
    target_stream.wait_stream(source_stream)


def record_and_wait(stream: torch.cuda.Stream) -> torch.cuda.Event:
    """
    Record event on stream and return it for later waiting.

    Args:
        stream: Stream to record on

    Returns:
        Event that can be waited on
    """
    event = torch.cuda.Event()
    event.record(stream)
    return event


@contextmanager
def optimal_sync_context():
    """
    Context manager that ensures GPU work completes only when needed.

    Uses an event to track completion rather than synchronizing immediately.

    Example:
        >>> with optimal_sync_context():
        ...     output = model(input)
        ...     # Work continues without blocking
        ... # Syncs only when exiting context
    """
    if not torch.cuda.is_available():
        yield
        return

    event = torch.cuda.Event()
    try:
        yield
    finally:
        event.record()
        event.synchronize()


# Global instances for convenience
_global_stream_pool: Optional[StreamPool] = None
_global_transfer: Optional[NonBlockingTransfer] = None


def get_stream_pool(num_streams: int = 4) -> StreamPool:
    """Get or create global stream pool."""
    global _global_stream_pool
    if _global_stream_pool is None:
        _global_stream_pool = StreamPool(num_streams=num_streams)
    return _global_stream_pool


def get_transfer_utils() -> NonBlockingTransfer:
    """Get or create global transfer utilities."""
    global _global_transfer
    if _global_transfer is None:
        _global_transfer = NonBlockingTransfer()
    return _global_transfer


def efficient_sync():
    """
    Minimal synchronization - prefer this over torch.cuda.synchronize().

    Only use when absolutely necessary (e.g., before CPU timing).
    """
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


def sync_if_needed(tensor: torch.Tensor):
    """Synchronize only if tensor is on GPU."""
    if tensor.is_cuda:
        torch.cuda.current_stream().synchronize()
