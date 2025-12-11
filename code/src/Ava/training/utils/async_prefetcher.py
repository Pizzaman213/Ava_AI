"""
Async batch prefetcher for the Ava pipeline.

Eliminates GPU starvation by prefetching batches in a background thread
while the main thread performs GPU computations.
"""

import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Iterator, Optional, Tuple

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

        # Start prefetch thread
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self) -> None:
        """Background thread that continuously loads batches and transfers to GPU."""
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
                        gpu_batch = self._transfer_batch_to_device(batch)
                        # Record event inside stream context after transfers are enqueued
                        event.record(self.transfer_stream)
                else:
                    # CPU path - direct transfer
                    gpu_batch = self._transfer_batch_to_device(batch)
                    event = None

                # Put batch in queue (blocks if queue is full - backpressure)
                self.queue.put((batch_idx, gpu_batch, event))

        except Exception as e:
            # Signal error to main thread
            self._error = e
            logger.error(f"AsyncBatchPrefetcher error: {e}", exc_info=True)
            self.queue.put(('ERROR', e, None))
        finally:
            # Signal end of iteration
            self.queue.put(None)

    def _transfer_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
            Dictionary with tensors transferred to device
        """
        gpu_batch = {}
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
                    gpu_tensor = value.to(self.device, non_blocking=True)
                    # GPU SYNC FIX: record_stream() prevents PyTorch memory allocator from
                    # reusing this tensor's memory before the async transfer completes.
                    # Without this, a race condition can occur where the allocator frees
                    # the memory while DMA transfer is still in progress.
                    if self.transfer_stream is not None:
                        gpu_tensor.record_stream(self.transfer_stream)
                    gpu_batch[key] = gpu_tensor
                else:
                    # Already on device, just copy reference
                    gpu_batch[key] = value
            else:
                gpu_batch[key] = value
        return gpu_batch

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

        batch_idx, gpu_batch, event = item

        # GPU SYNC FIX: Use stream.wait_event() instead of event.synchronize()
        # event.synchronize() blocks the CPU until the transfer completes (bad!)
        # current_stream().wait_event() makes GPU wait for transfer without blocking CPU
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

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

    def stop(self) -> None:
        """Stop the prefetch thread and clean up."""
        self.stop_event.set()

        # Drain queue to unblock thread if it's waiting on queue.put()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

        # Wait for thread to finish (with longer timeout for graceful shutdown)
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

        if self.thread.is_alive():
            logger.warning("AsyncBatchPrefetcher thread did not terminate cleanly, forcing stop")
            # Try one more time with stop event
            self.stop_event.set()
            self.thread.join(timeout=2.0)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.stop()
        except Exception as e:
            # FIX: Log cleanup failures instead of silently ignoring
            # This helps debug issues where threads don't terminate properly
            logger.debug(f"AsyncBatchPrefetcher cleanup warning: {e}")
