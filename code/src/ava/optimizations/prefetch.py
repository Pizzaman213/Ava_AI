"""
Async batch prefetcher for the Ava pipeline.

Eliminates GPU starvation by prefetching batches in a background thread
while the main thread performs GPU computations.
"""

import logging
import threading
from collections import deque
from queue import Empty, Queue
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AsyncBatchPrefetcher:
    """
    Prefetches batches in a background thread for continuous GPU feeding.

    This class solves GPU starvation by completely decoupling data loading from
    GPU computation. The DataLoader runs in a separate thread, continuously
    loading batches while the main thread focuses solely on GPU operations.

    Key features:
        - Background thread for DataLoader iteration (never blocks main thread)
        - Dedicated CUDA stream for CPU->GPU transfers (overlaps with compute)
        - Queue-based buffering with configurable prefetch depth
        - CUDA event synchronization for correctness

    Example:
        >>> prefetcher = AsyncBatchPrefetcher(train_loader, device, prefetch_count=3)
        >>> for batch_idx, batch in prefetcher:
        ...     output = model(batch['input_ids'])
        >>> prefetcher.stop()
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_count: int = 3,
        memory_threshold: float = 0.90,
        preallocate_buffers: bool = True,
    ):
        """
        Initialize the async batch prefetcher.

        Args:
            dataloader: PyTorch DataLoader to iterate over
            device: Target device for batch transfer
            prefetch_count: Number of batches to keep ready (default: 3)
            memory_threshold: GPU memory utilization threshold (0.0-1.0) above which
                            prefetching pauses to prevent OOM (default: 0.90)
            preallocate_buffers: Pre-allocate pinned buffers for faster transfers (default: True)
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_count = prefetch_count
        self.memory_threshold = memory_threshold
        self.queue: Queue = Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.transfer_stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._total_batches = 0
        self._error: Optional[Exception] = None
        self._pinned_buffers: List[torch.Tensor] = []  # Track pinned buffers for cleanup
        self._initialized = threading.Event()  # Initialization barrier
        self._memory_pressure_count = 0  # Track memory pressure events
        # Store (pinned_sources, event) tuples and synchronize on the CUDA event
        # before evicting. This guarantees DMA completion before freeing pinned memory.
        self._pinned_history: Deque[Tuple[List[torch.Tensor], Optional[torch.cuda.Event]]] = deque(maxlen=prefetch_count + 2)

        # Pre-allocated pinned buffer pool for zero-copy reuse (nanoGPT optimization)
        # OPTIMIZATION: Uses lock-free deques keyed by (key, shape, dtype) tuples.
        # Python's deque operations are GIL-protected, eliminating lock contention.
        # MEMORY FIX: Limit total pinned memory to prevent 6GB+ unbounded growth
        self._preallocate_buffers = preallocate_buffers
        self._pinned_buffer_pool: Dict[Tuple, Deque[torch.Tensor]] = {}  # (shape, dtype) -> deque of buffers
        self._max_pinned_pools = 4  # Max number of different (shape, dtype) pools (reduced from 8 for memory savings)
        self._max_buffers_per_pool = prefetch_count + 1  # Max buffers per pool (reduced from +2)
        self._pinned_pool_access_count: Dict[Tuple, int] = {}  # Track access frequency for eviction

        # Memory pressure check caching (OPTIMIZATION: reduces sync overhead)
        # Cache total memory at init - never changes
        self._total_memory = 0
        if torch.cuda.is_available() and device.type == 'cuda':
            try:
                self._total_memory = torch.cuda.get_device_properties(device).total_memory
            except Exception:
                pass
        self._memory_cache_time = 0.0
        self._memory_cache_value = 0.0
        self._memory_cache_interval = 2.0  # OPTIMIZED: Check every 2 seconds (faster response to memory pressure)

        # Start prefetch thread
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

        # Wait for thread to signal it has started (with timeout to detect immediate crashes)
        if not self._initialized.wait(timeout=5.0):
            if self._error is not None:
                raise RuntimeError(f"Prefetch thread failed to start: {self._error}")
            logger.warning("Prefetch thread did not signal initialization, continuing anyway")

    def _check_memory_pressure(self) -> bool:
        """
        Check if GPU memory is under pressure.

        OPTIMIZATION: Uses time-based caching to reduce GPU sync overhead.
        Memory stats are only queried every _memory_cache_interval seconds,
        not on every batch. This eliminates 10-100ms delays from
        torch.cuda.memory_allocated() syncs.

        Returns:
            True if memory utilization exceeds threshold, False otherwise
        """
        import time

        if not torch.cuda.is_available() or self.device.type != 'cuda':
            return False

        # Skip check if total memory unknown
        if self._total_memory == 0:
            return False

        # Use cached value if fresh enough (OPTIMIZATION: avoids GPU sync)
        now = time.monotonic()
        if now - self._memory_cache_time < self._memory_cache_interval:
            # FIX: Return cached result, not stale comparison
            # The cached value IS the utilization ratio, compare against threshold
            return self._memory_cache_value > self.memory_threshold

        # Cache is stale - refresh it below

        # Cache is stale - refresh
        # OPTIMIZATION: Use memory_reserved() instead of memory_allocated()
        # memory_reserved() queries CUDA allocator state (no GPU sync)
        # memory_allocated() requires cudaMemGetInfo (GPU sync)
        try:
            reserved = torch.cuda.memory_reserved(self.device)
            self._memory_cache_value = reserved / self._total_memory
            self._memory_cache_time = now
            return self._memory_cache_value > self.memory_threshold
        except Exception:
            return False  # On error, don't block

    def _get_pinned_buffer(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Get a pre-allocated pinned buffer, creating one if needed.

        This eliminates per-batch pin_memory() calls which add ~1ms latency each.
        Buffers are reused across batches for zero allocation overhead.

        OPTIMIZATION: Uses lock-free deque operations. Python's deque.popleft()
        and deque.append() are GIL-protected and thread-safe, eliminating
        lock contention at high throughput (>1000 batches/min).

        MEMORY FIX: Limits total pools to _max_pinned_pools to prevent unbounded
        growth. With many different batch shapes, this could consume 6GB+ of
        pinned memory without limits.

        Args:
            key: Batch key (e.g., 'input_ids', 'attention_mask') - unused, kept for API compat
            shape: Required buffer shape
            dtype: Required buffer dtype

        Returns:
            Pinned memory buffer (may be larger than requested, slice as needed)
        """
        # Pool key is (shape, dtype) only - buffers of same shape/dtype are interchangeable
        pool_key = (shape, dtype)

        # Track access for LRU eviction
        self._pinned_pool_access_count[pool_key] = self._pinned_pool_access_count.get(pool_key, 0) + 1

        # Lock-free path: try to get from existing deque
        if pool_key in self._pinned_buffer_pool:
            pool = self._pinned_buffer_pool[pool_key]
            try:
                return pool.popleft()  # Thread-safe O(1) operation
            except IndexError:
                pass  # Pool empty, fall through to create new

        # MEMORY FIX: Evict pools if we have too many
        # VRAM OPTIMIZATION: Use hybrid eviction score = size / (access_count + 1)
        # This prioritizes evicting large, infrequently-used buffers
        if len(self._pinned_buffer_pool) >= self._max_pinned_pools:
            def compute_eviction_score(pool_key_candidate):
                """Higher score = more likely to evict (large + infrequent)."""
                shape, dtype = pool_key_candidate
                # Compute size in bytes
                element_count = 1
                for dim in shape:
                    element_count *= dim
                bytes_per_element = 4  # Assume float32, adjust if needed
                if dtype in (torch.float16, torch.bfloat16):
                    bytes_per_element = 2
                elif dtype == torch.float64:
                    bytes_per_element = 8
                size_bytes = element_count * bytes_per_element
                # Score: size / (access_count + 1)
                access_count = self._pinned_pool_access_count.get(pool_key_candidate, 0)
                return size_bytes / (access_count + 1)

            # Find pool with highest eviction score
            sorted_pools = sorted(
                [(k, compute_eviction_score(k)) for k in self._pinned_pool_access_count.keys() if k != pool_key],
                key=lambda x: x[1],
                reverse=True  # Highest score first
            )
            if sorted_pools:
                evict_key = sorted_pools[0][0]
                if evict_key in self._pinned_buffer_pool:
                    # Clear the pool - buffers will be GC'd
                    evicted_pool = self._pinned_buffer_pool.pop(evict_key, None)
                    if evicted_pool:
                        evicted_pool.clear()
                    self._pinned_pool_access_count.pop(evict_key, None)
                    logger.debug(f"Evicted pinned buffer pool for shape {evict_key[0]} (size-based)")

        # Create new pinned buffer (no lock needed - worst case is duplicate pools)
        try:
            buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
            # Lazily create pool for this key (atomic dict assignment in Python)
            if pool_key not in self._pinned_buffer_pool:
                self._pinned_buffer_pool[pool_key] = deque(maxlen=self._max_buffers_per_pool)
            return buffer
        except Exception:
            # Fallback: return None and let caller use regular pin_memory()
            return None

    def _return_pinned_buffer(self, key: str, buffer: torch.Tensor) -> None:
        """
        Return a pinned buffer to the pool for reuse.

        OPTIMIZATION: Lock-free append to deque. The deque has maxlen set,
        so excess buffers are automatically dropped (no memory leak).

        MEMORY FIX: Don't create new pools on return - only reuse existing ones.
        This prevents unbounded pool growth from varied batch shapes.

        Args:
            key: Batch key - unused, kept for API compatibility
            buffer: Pinned buffer to return
        """
        if not buffer.is_pinned():
            return  # Don't pool non-pinned buffers

        # Pool key is (shape, dtype) only - buffers of same shape/dtype are interchangeable
        pool_key = (tuple(buffer.shape), buffer.dtype)

        # MEMORY FIX: Only return to existing pools, don't create new ones
        # This prevents unbounded growth from varied batch shapes
        if pool_key not in self._pinned_buffer_pool:
            return  # Drop buffer - pool doesn't exist or was evicted

        # Thread-safe O(1) append - drops oldest if at maxlen
        try:
            self._pinned_buffer_pool[pool_key].append(buffer)
        except Exception:
            pass  # Drop buffer silently on any error

    def _prefetch_loop(self) -> None:
        """Background thread that continuously loads batches and transfers to GPU."""
        import time

        # Signal that thread has started successfully
        self._initialized.set()
        try:
            for batch_idx, batch in enumerate(self.dataloader):
                if self.stop_event.is_set():
                    break

                # Memory-aware backpressure with HYBRID backoff strategy
                # - Linear for first 3 iterations (quick recovery for transient pressure)
                # - Exponential after that (CPU-efficient for persistent pressure)
                if self._check_memory_pressure():
                    wait_time = 0.01   # Start with 10ms
                    max_wait = 0.1     # Cap at 100ms
                    total_waited = 0.0
                    pressure_iterations = 0

                    while self._check_memory_pressure():
                        self._memory_pressure_count += 1
                        pressure_iterations += 1
                        if self._memory_pressure_count % 10 == 0:
                            logger.warning(
                                f"Prefetcher paused due to memory pressure "
                                f"(>{self.memory_threshold:.0%}), count: {self._memory_pressure_count}, "
                                f"waited: {total_waited:.2f}s"
                            )
                        total_waited += wait_time
                        # HYBRID backoff: Linear for quick recovery, exponential for persistent pressure
                        if pressure_iterations <= 3:
                            # Linear for transient pressure (quick recovery)
                            wait_time = min(wait_time + 0.01, max_wait)
                        else:
                            # Exponential for persistent pressure (saves CPU cycles)
                            wait_time = min(wait_time * 1.5, max_wait)
                        if self.stop_event.is_set():
                            return
                        # Safety limit: don't wait forever
                        if total_waited > 30.0:
                            logger.error(f"Memory pressure persisted for {total_waited:.1f}s, continuing anyway")
                            break

                self._total_batches = batch_idx + 1

                # Transfer to GPU using dedicated stream (overlaps with main thread's compute)
                if self.transfer_stream is not None:
                    # Record event INSIDE stream context to ensure proper ordering
                    event = torch.cuda.Event()
                    with torch.cuda.stream(self.transfer_stream):
                        gpu_batch, pinned_sources = self._transfer_batch_to_device(batch)
                        # Record event inside stream context after transfers are enqueued
                        event.record(self.transfer_stream)
                else:
                    # CPU path - direct transfer
                    gpu_batch, pinned_sources = self._transfer_batch_to_device(batch)
                    event = None

                # Put batch in queue (blocks if queue is full - backpressure)
                # CRITICAL: Include pinned_sources to keep them alive until consumer waits on event
                self.queue.put((batch_idx, gpu_batch, event, pinned_sources))

        except Exception as e:
            # Signal error to main thread
            self._error = e
            logger.error(f"AsyncBatchPrefetcher error: {e}", exc_info=True)
            self.queue.put(('ERROR', e, None, None))
        finally:
            # Signal end of iteration
            self.queue.put(None)

    def _transfer_batch_to_device(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Transfer a batch to the target device using NVIDIA-recommended async pattern.

        For true async CPU->GPU transfer without stream synchronization:
        1. Tensor MUST be in pinned (page-locked) memory
        2. Use non_blocking=True with the dedicated transfer stream
        3. Synchronize via CUDA events, not stream.synchronize()

        OPTIMIZATION (nanoGPT-style): Uses pre-allocated pinned buffer pool to avoid
        per-batch pin_memory() calls which add ~1ms latency each.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Tuple of (gpu_batch dict, list of pinned source tensors to keep alive)
        """
        gpu_batch = {}
        # Keep pinned source tensors alive until DMA completes.
        # Without this, the pinned tensor can be garbage collected while async
        # transfer is in progress, causing CUDA errors.
        pinned_sources: List[torch.Tensor] = []

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # NVIDIA async transfer pattern:
                # 1. Ensure tensor is in pinned memory for true async DMA
                # 2. Transfer with non_blocking=True on dedicated stream
                if value.device.type == 'cpu':
                    if not value.is_pinned():
                        # OPTIMIZATION: Try to use pre-allocated pinned buffer pool
                        if self._preallocate_buffers:
                            pinned_buffer = self._get_pinned_buffer(
                                key, tuple(value.shape), value.dtype
                            )
                            if pinned_buffer is not None:
                                # Copy data to pre-allocated pinned buffer
                                pinned_buffer.copy_(value)
                                value = pinned_buffer
                            else:
                                # Fallback: regular pin_memory() if pool fails
                                value = value.pin_memory()
                        else:
                            # Original path: pin_memory() per batch
                            value = value.pin_memory()
                    # Keep reference to pinned source until transfer completes
                    pinned_sources.append(value)
                    # Enqueue DMA on the transfer_stream with record_stream() to ensure
                    # the allocator does not free GPU memory while transfer is in progress.
                    if self.transfer_stream is not None:
                        with torch.cuda.stream(self.transfer_stream):
                            gpu_tensor = value.to(self.device, non_blocking=True)
                            gpu_tensor.record_stream(self.transfer_stream)
                    else:
                        gpu_tensor = value.to(self.device, non_blocking=True)
                    gpu_batch[key] = gpu_tensor
                else:
                    # Already on device, just copy reference
                    gpu_batch[key] = value
            else:
                gpu_batch[key] = value
        return gpu_batch, pinned_sources

    def __iter__(self) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Get next batch from queue, waiting for GPU transfer if needed."""
        item = self.queue.get()

        if item is None:
            raise StopIteration

        if item[0] == 'ERROR':
            raise item[1]

        batch_idx, gpu_batch, event, pinned_sources = item

        # Use stream.wait_event() instead of event.synchronize() to avoid blocking CPU.
        # current_stream().wait_event() makes GPU wait for transfer without blocking CPU.
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

        # RACE CONDITION FIX: Event-based eviction guarantees DMA completion before freeing pinned memory.
        # CRITICAL: Must synchronize BEFORE deque evicts to prevent accessing freed memory.
        # The issue: deque.append() with maxlen can evict oldest item DURING the append,
        # so we must synchronize the oldest event BEFORE appending to prevent race condition.
        if pinned_sources:
            # Check if deque is full and will evict on next append
            # FIX: Use >= maxlen (not maxlen - 1) since deque evicts when len == maxlen
            if len(self._pinned_history) >= self._pinned_history.maxlen:
                # Deque will evict on next append - return oldest buffers to pool
                if self._pinned_history:
                    oldest_sources, oldest_event = self._pinned_history[0]
                    if oldest_event is not None:
                        # OPTIMIZATION: Use event.query() to check if DMA complete without blocking
                        # If not complete, use wait_event on GPU side (non-blocking)
                        if not oldest_event.query():
                            # DMA not yet complete - GPU-side wait on current stream
                            # wait_event ensures proper ordering without CPU stall
                            torch.cuda.current_stream().wait_event(oldest_event)
                    # Return buffers to pool for reuse (nanoGPT optimization)
                    if self._preallocate_buffers and oldest_sources:
                        for buf in oldest_sources:
                            if buf.is_pinned():
                                # Return buffer to pool - key name is used to construct pool_key
                                # Since pool_key = (key, shape, dtype), we use a generic key
                                # The _return_pinned_buffer method will create the proper key
                                self._return_pinned_buffer('buffer', buf)
            # Now safe to append - either not full, or oldest DMA is complete
            self._pinned_history.append((pinned_sources, event))

        return batch_idx, gpu_batch

    def __len__(self) -> int:
        """Return estimated length from underlying dataloader."""
        if hasattr(self.dataloader, 'get_dynamic_total'):
            return self.dataloader.get_dynamic_total()
        try:
            return len(self.dataloader)
        except TypeError:
            # Dataset doesn't support __len__ (e.g., IterableDataset)
            return self._total_batches

    def get_dynamic_total(self) -> int:
        """Support dynamic batch size tracking."""
        if hasattr(self.dataloader, 'get_dynamic_total'):
            return self.dataloader.get_dynamic_total()
        try:
            return len(self.dataloader)
        except TypeError:
            # Dataset doesn't support __len__ (e.g., IterableDataset)
            return self._total_batches

    def clear_queue(self) -> int:
        """
        Clear all prefetched batches from the queue.

        Call this after OOM recovery to discard batches that were generated
        with the old (too large) batch size. The iterator will generate new
        batches with the updated (smaller) batch size.

        Returns:
            Number of batches that were discarded
        """
        discarded = 0
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item is not None and item[0] != 'ERROR':
                    discarded += 1
            except Empty:
                break

        if discarded > 0:
            logger.info(f"AsyncBatchPrefetcher: Cleared {discarded} stale batches after OOM")

        return discarded

    def stop(self) -> None:
        """Stop the prefetch thread and clean up resources."""
        self.stop_event.set()

        # Drain queue to unblock thread if it's waiting on queue.put()
        # Also collect any pinned buffers that need cleanup
        drained_items = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item is not None and item[0] != 'ERROR':
                    drained_items.append(item)
            except Empty:
                break

        # Wait for thread to finish (with longer timeout for graceful shutdown)
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

        if self.thread.is_alive():
            logger.warning("AsyncBatchPrefetcher thread did not terminate cleanly after 5s")
            # Try one more time
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.error("AsyncBatchPrefetcher thread still alive after 7s - potential resource leak")

        # Explicitly clean up pinned buffers to prevent memory leaks
        # Wait for any pending CUDA operations before releasing
        if self.transfer_stream is not None:
            try:
                self.transfer_stream.synchronize()
            except Exception as e:
                logger.debug(f"CUDA stream sync during cleanup: {e}")

        # Clear references to pinned tensors
        self._pinned_buffers.clear()
        # Sync all pending events before clearing to ensure DMA is complete
        for item in self._pinned_history:
            if item[1] is not None:  # item[1] is the CUDA event
                try:
                    item[1].synchronize()
                except Exception:
                    pass  # Ignore errors during cleanup
        self._pinned_history.clear()
        # Clear pinned buffer pools
        for pool in self._pinned_buffer_pool.values():
            pool.clear()
        self._pinned_buffer_pool.clear()
        self._pinned_pool_access_count.clear()
        for item in drained_items:
            # item format: (batch_idx, gpu_batch, event, pinned_sources)
            if len(item) >= 4 and item[3] is not None:
                item[3].clear() if hasattr(item[3], 'clear') else None

        logger.debug("AsyncBatchPrefetcher stopped and cleaned up")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.stop()
        except Exception as e:
            # Log cleanup failures to help debug thread termination issues
            logger.debug(f"AsyncBatchPrefetcher cleanup warning: {e}")
