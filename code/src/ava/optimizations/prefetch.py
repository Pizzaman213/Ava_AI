"""
Async batch prefetcher for the Ava pipeline.

Eliminates GPU starvation by prefetching batches in a background thread
while the main thread performs GPU computations.
"""

import logging
import threading
from collections import deque
from queue import Empty, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
        prefetch_count: int = 3
    ):
        """
        Initialize the async batch prefetcher.

        Args:
            dataloader: PyTorch DataLoader to iterate over
            device: Target device for batch transfer
            prefetch_count: Number of batches to keep ready (default: 3)
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_count = prefetch_count
        self.queue: Queue = Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.transfer_stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._total_batches = 0
        self._error: Optional[Exception] = None
        self._pinned_buffers: List[torch.Tensor] = []  # Track pinned buffers for cleanup
        self._initialized = threading.Event()  # Initialization barrier
        # RACE CONDITION FIX: Track recent pinned sources in a deque instead of
        # synchronizing immediately. This allows DMA to complete naturally without
        # blocking the CPU. Use prefetch_count * 10 for large safety margin - this
        # ensures pinned memory stays alive even with slow batch processing or
        # variable GPU computation times. The previous value of *3 was insufficient
        # and caused "illegal memory access" errors around batch 9.
        self._pinned_history: deque = deque(maxlen=prefetch_count * 10)

        # Start prefetch thread
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

        # Wait for thread to signal it has started (with timeout to detect immediate crashes)
        if not self._initialized.wait(timeout=5.0):
            if self._error is not None:
                raise RuntimeError(f"Prefetch thread failed to start: {self._error}")
            logger.warning("Prefetch thread did not signal initialization, continuing anyway")

    def _prefetch_loop(self) -> None:
        """Background thread that continuously loads batches and transfers to GPU."""
        # Signal that thread has started successfully
        self._initialized.set()
        try:
            for batch_idx, batch in enumerate(self.dataloader):
                if self.stop_event.is_set():
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

        Without pinned memory, .to(device, non_blocking=True) still triggers
        an implicit sync because PyTorch must first copy to internal pinned
        staging buffer before DMA transfer can begin.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Tuple of (gpu_batch dict, list of pinned source tensors to keep alive)
        """
        gpu_batch = {}
        # CRITICAL FIX: Keep pinned source tensors alive until DMA completes
        # Without this, the pinned tensor can be GC'd while async transfer is in progress,
        # causing "illegal memory access" CUDA errors.
        pinned_sources: List[torch.Tensor] = []

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # NVIDIA async transfer pattern:
                # 1. Ensure tensor is in pinned memory for true async DMA
                # 2. Transfer with non_blocking=True on dedicated stream
                if value.device.type == 'cpu':
                    if not value.is_pinned():
                        # Pin memory to enable async DMA transfer
                        # This avoids implicit sync from pageable->pinned copy
                        value = value.pin_memory()
                    # Keep reference to pinned source until transfer completes
                    pinned_sources.append(value)
                    # CRITICAL FIX: .to() enqueues DMA on the CURRENT stream, not the
                    # surrounding context. We must explicitly use transfer_stream context
                    # so that record_stream() matches the actual DMA stream.
                    # Without this, the allocator may free GPU memory while DMA is in progress.
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

        # GPU SYNC FIX: Use stream.wait_event() instead of event.synchronize()
        # event.synchronize() blocks the CPU until the transfer completes (bad!)
        # current_stream().wait_event() makes GPU wait for transfer without blocking CPU
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            # SAFETY SYNC: When deque is about to evict entries, ensure the oldest
            # DMA transfers have completed by syncing the transfer stream. This only
            # happens when we're at capacity, avoiding constant sync overhead.
            if self.transfer_stream is not None and len(self._pinned_history) >= self._pinned_history.maxlen - 1:
                self.transfer_stream.synchronize()

        # RACE CONDITION FIX: Instead of blocking with synchronize(), keep pinned
        # sources alive in a deque. The deque's maxlen ensures old entries are
        # automatically cleaned up after enough batches have passed (DMA complete).
        # This avoids CPU blocking while still preventing premature GC of pinned memory.
        if pinned_sources:
            self._pinned_history.append(pinned_sources)

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
        self._pinned_history.clear()  # Clear the deque tracking recent pinned sources
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
