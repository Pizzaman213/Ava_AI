"""
Distributed data loading utilities.

This module provides distributed training wrappers for streaming datasets:
- DistributedStreamingDataset: Token-balanced sharding across GPUs

Usage:
    from ava.data.distributed import DistributedStreamingDataset

    distributed_dataset = DistributedStreamingDataset(
        base_dataset=streaming_dataset,
        world_size=4,
        rank=0,
        token_balanced=True,
    )
"""

import logging
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

# Import centralized constants
from ..config.constants import DATA_CONSTANTS

# Check if distributed training is available
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    dist = None

logger = logging.getLogger(__name__)


class DistributedStreamingDataset(IterableDataset):
    """
    Wrapper for distributed streaming dataset with token-balanced sharding.

    PHASE 2.1 OPTIMIZATION: Token-balanced distribution ensures each GPU gets
    equal total tokens (not just samples), preventing GPU idle time from length imbalance.
    Expected improvement: 10-20% better multi-GPU utilization.
    """

    def __init__(
        self,
        base_dataset: IterableDataset,
        world_size: int,
        rank: int,
        load_aware: bool = False,
        memory_monitor: Any = None,
        token_balanced: bool = True
    ):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        self.load_aware = load_aware
        self.memory_monitor = memory_monitor
        self.token_balanced = token_balanced

        # PHASE 2 OPTIMIZATION: Load balancing state
        self._sample_count = 0
        self._skip_next = 0

        # PHASE 2.1: Token balancing state
        self._token_counts = [0] * world_size
        self._batch_buffer: list = []
        self._batch_buffer_size = DATA_CONSTANTS.TOKEN_BALANCE_BATCH_SIZE

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate with token-balanced distribution strategy.

        PHASE 2.1: Balances by total tokens, not samples, for better GPU utilization.
        PHASE 2: If load_aware=True, adjusts based on memory pressure.
        """
        base_iter = iter(self.base_dataset)

        for i, sample in enumerate(base_iter):
            # PHASE 2.1: Token-balanced sharding
            if self.token_balanced:
                self._batch_buffer.append(sample)

                if len(self._batch_buffer) >= self._batch_buffer_size:
                    # Sort buffer by sequence length for better packing
                    self._batch_buffer.sort(key=lambda x: len(x.get('input_ids', [])), reverse=True)

                    # Distribute to rank with fewest tokens
                    for buffered_sample in self._batch_buffer:
                        sample_tokens = len(buffered_sample.get('input_ids', []))

                        # Find rank with minimum tokens
                        min_rank = self._token_counts.index(min(self._token_counts))

                        # Assign to that rank
                        self._token_counts[min_rank] += sample_tokens

                        # Yield if this sample belongs to our rank
                        if min_rank == self.rank:
                            self._sample_count += 1
                            yield buffered_sample

                    self._batch_buffer.clear()
                continue

            # PHASE 2: Load-aware distribution (fallback if token balancing disabled)
            if self.load_aware and self.memory_monitor is not None and i % DATA_CONSTANTS.LOAD_BALANCE_CHECK_INTERVAL == 0:
                try:
                    coordination = self.memory_monitor.coordinate_oom_prevention()

                    if coordination.get('needs_coordination', False):
                        max_util_rank = coordination.get('max_util_rank', -1)
                        min_util_rank = coordination.get('min_util_rank', -1)

                        if self.rank == max_util_rank:
                            self._skip_next = min(DATA_CONSTANTS.LOAD_BALANCE_MAX_SKIP, self._sample_count // 1000)
                        elif self.rank == min_util_rank and self._skip_next == 0:
                            pass
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        f"Worker coordination fallback to round-robin (rank {self.rank}): {e}"
                    )

            # Apply skip logic
            if self._skip_next > 0:
                self._skip_next -= 1
                continue

            # Standard round-robin distribution
            if i % self.world_size == self.rank:
                self._sample_count += 1
                yield sample

        # PHASE 2.1: Flush remaining buffer with synchronization
        if self.token_balanced and self._batch_buffer:
            # SYNC FIX: Synchronize ranks before final flush to prevent imbalance
            # This ensures all ranks finish main iteration before flushing
            if DISTRIBUTED_AVAILABLE and dist is not None and dist.is_initialized():
                try:
                    dist.barrier()
                    logger.debug(f"Rank {self.rank}: synchronized before buffer flush")
                except Exception as e:
                    logger.warning(f"Rank {self.rank}: barrier failed, continuing: {e}")

            # Process remaining buffer
            for buffered_sample in self._batch_buffer:
                sample_tokens = len(buffered_sample.get('input_ids', []))
                min_rank = self._token_counts.index(min(self._token_counts))
                self._token_counts[min_rank] += sample_tokens

                if min_rank == self.rank:
                    yield buffered_sample

            # Clear buffer after flush
            self._batch_buffer.clear()


__all__ = [
    'DistributedStreamingDataset',
]
