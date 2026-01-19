"""
Distributed data loading utilities.

This module provides distributed training wrappers and samplers:
- DistributedStreamingDataset: Token-balanced sharding for IterableDataset
- AdvancedDistributedSampler: Load-balanced sampler for map-style Dataset

Usage:
    from ava.data.distributed import DistributedStreamingDataset, AdvancedDistributedSampler

    # For streaming datasets
    distributed_dataset = DistributedStreamingDataset(
        base_dataset=streaming_dataset,
        world_size=4,
        rank=0,
        token_balanced=True,
    )

    # For map-style datasets
    sampler = AdvancedDistributedSampler(
        dataset=my_dataset,
        shuffle=True,
        enable_load_balancing=True,
    )
"""

import logging
import math
from datetime import timedelta
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

# Import centralized constants
from ..config.constants import DATA_CONSTANTS

# Default timeout for distributed barriers (30 minutes)
_BARRIER_TIMEOUT = timedelta(minutes=30)

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

    OPTIMIZATION: Enhanced with configurable buffer size and balance threshold
    for more precise token distribution across ranks.
    """

    def __init__(
        self,
        base_dataset: IterableDataset,
        world_size: int,
        rank: int,
        load_aware: bool = False,
        memory_monitor: Any = None,
        token_balanced: bool = True,
        enable_length_sorting: bool = True,
        token_buffer_size: int = 2000,  # OPTIMIZATION: Configurable buffer size
        balance_threshold: float = 0.05,  # OPTIMIZATION: Skip rank if >5% above target
    ):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        self.load_aware = load_aware
        self.memory_monitor = memory_monitor
        self.token_balanced = token_balanced
        self.enable_length_sorting = enable_length_sorting
        self.token_buffer_size = token_buffer_size
        self.balance_threshold = balance_threshold

        # PHASE 2 OPTIMIZATION: Load balancing state
        self._sample_count = 0
        self._skip_next = 0

        # PHASE 2.1: Token balancing state with enhanced tracking
        self._token_counts = [0] * world_size
        self._batch_buffer: list = []
        self._batch_buffer_size = token_buffer_size  # Use configurable size

        # OPTIMIZATION: Track target tokens per rank for better balancing
        self._total_tokens_seen = 0
        self._target_tokens_per_rank = 0

        # OPTIMIZATION: Track recent assignment pattern for weighted scoring
        # Helps detect and correct systematic imbalances
        self._recent_assignments: List[int] = []  # Last N rank assignments
        self._assignment_window = 100  # Window size for pattern detection

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate with token-balanced distribution strategy.

        PHASE 2.1: Balances by total tokens, not samples, for better GPU utilization.
        PHASE 2: If load_aware=True, adjusts based on memory pressure.

        OPTIMIZATION: Enhanced with target tracking and weighted scoring to prevent
        systematic imbalances in long training runs.
        """
        base_iter = iter(self.base_dataset)

        for i, sample in enumerate(base_iter):
            # PHASE 2.1: Token-balanced sharding
            if self.token_balanced:
                self._batch_buffer.append(sample)

                if len(self._batch_buffer) >= self._batch_buffer_size:
                    # Optionally sort buffer by sequence length for better packing
                    if self.enable_length_sorting:
                        self._batch_buffer.sort(key=lambda x: len(x.get('input_ids', [])), reverse=True)
                    # else: keep original random order for maximum diversity

                    # OPTIMIZATION: Distribute with enhanced scoring
                    for buffered_sample in self._batch_buffer:
                        sample_tokens = len(buffered_sample.get('input_ids', []))

                        # Update target tokens per rank
                        self._total_tokens_seen += sample_tokens
                        self._target_tokens_per_rank = self._total_tokens_seen // self.world_size

                        # OPTIMIZATION: Find best rank using weighted scoring
                        # Score considers: current tokens, deviation from target, recent assignment pattern
                        best_rank = self._select_best_rank(sample_tokens)

                        # Assign to selected rank
                        self._token_counts[best_rank] += sample_tokens

                        # Track assignment pattern
                        self._recent_assignments.append(best_rank)
                        if len(self._recent_assignments) > self._assignment_window:
                            self._recent_assignments.pop(0)

                        # Yield if this sample belongs to our rank
                        if best_rank == self.rank:
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
                    dist.barrier(timeout=_BARRIER_TIMEOUT)
                    logger.debug(f"Rank {self.rank}: synchronized before buffer flush")
                except Exception as e:
                    logger.warning(f"Rank {self.rank}: barrier failed (timeout or process crash), continuing: {e}")

            # Process remaining buffer using enhanced scoring
            for buffered_sample in self._batch_buffer:
                sample_tokens = len(buffered_sample.get('input_ids', []))
                best_rank = self._select_best_rank(sample_tokens)
                self._token_counts[best_rank] += sample_tokens

                if best_rank == self.rank:
                    yield buffered_sample

            # Clear buffer after flush
            self._batch_buffer.clear()

    def _select_best_rank(self, sample_tokens: int) -> int:
        """
        Select the best rank for a sample using weighted scoring.

        OPTIMIZATION: Uses multi-factor scoring to prevent systematic imbalances:
        1. Current token count (primary factor)
        2. Deviation from target (correction factor)
        3. Recent assignment frequency (anti-bias factor)

        Args:
            sample_tokens: Number of tokens in the sample

        Returns:
            Best rank index for this sample
        """
        if self.world_size == 1:
            return 0

        best_rank = 0
        best_score = float('inf')

        # Calculate recent assignment counts for anti-bias weighting
        recent_counts = [0] * self.world_size
        for r in self._recent_assignments:
            recent_counts[r] += 1

        for rank in range(self.world_size):
            # Base score: current token count (lower is better)
            score = self._token_counts[rank]

            # OPTIMIZATION: Skip ranks that are significantly overloaded
            # This prevents any single rank from getting too far ahead
            if self._target_tokens_per_rank > 0:
                deviation = (self._token_counts[rank] - self._target_tokens_per_rank) / self._target_tokens_per_rank
                if deviation > self.balance_threshold:
                    # Penalize overloaded ranks heavily
                    score += sample_tokens * 2

            # OPTIMIZATION: Add anti-bias factor from recent assignments
            # Prevents same ranks from always being selected in ties
            if self._recent_assignments:
                recent_bias = recent_counts[rank] / len(self._recent_assignments)
                # Add small penalty proportional to recent assignment frequency
                score += sample_tokens * recent_bias * 0.1

            if score < best_score:
                best_score = score
                best_rank = rank

        return best_rank

    def get_balance_stats(self) -> Dict[str, Any]:
        """
        Get token balancing statistics.

        Returns:
            Dictionary with token distribution stats across ranks
        """
        total_tokens = sum(self._token_counts)
        avg_tokens = total_tokens / self.world_size if self.world_size > 0 else 0

        imbalances = []
        for i, count in enumerate(self._token_counts):
            if avg_tokens > 0:
                imbalance = abs(count - avg_tokens) / avg_tokens
                imbalances.append(imbalance)

        return {
            'token_counts': list(self._token_counts),
            'total_tokens': total_tokens,
            'avg_tokens_per_rank': avg_tokens,
            'max_imbalance': max(imbalances) if imbalances else 0.0,
            'samples_this_rank': self._sample_count,
            'balance_threshold': self.balance_threshold,
        }


class AdvancedDistributedSampler(Sampler):
    """
    Advanced distributed sampler with proper sharding, load balancing, and fault tolerance.

    Features:
    - Balanced data distribution across ranks
    - Dynamic resharding for failed ranks
    - Load balancing monitoring
    - Deterministic shuffling with proper epoch seeding
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        enable_load_balancing: bool = True,
        balancing_tolerance: float = 0.05  # 5% tolerance for load imbalance
    ):
        if num_replicas is None:
            if not DISTRIBUTED_AVAILABLE or dist is None or not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not DISTRIBUTED_AVAILABLE or dist is None or not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank is not None and num_replicas is not None:
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.enable_load_balancing = enable_load_balancing
        self.balancing_tolerance = balancing_tolerance

        # Calculate dataset size and samples per rank
        if self.num_replicas is not None:
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = 0
            self.total_size = 0

        # Track load balancing statistics
        self.samples_processed = 0
        self.load_stats: Dict[str, Any] = {
            'samples_assigned': self.num_samples,
            'samples_processed': 0,
            'load_ratio': 0.0
        }

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        if len(indices) != self.total_size:
            raise ValueError(
                f"Index mismatch in distributed sampler: expected {self.total_size} indices "
                f"but got {len(indices)}. This indicates a data sharding issue. "
                f"Rank: {self.rank}, Num replicas: {self.num_replicas}"
            )

        # Subsample for this rank with proper sharding
        rank_indices = self._get_rank_indices(indices)

        # Update load statistics
        self.load_stats['samples_assigned'] = len(rank_indices)

        return iter(rank_indices)

    def _get_rank_indices(self, indices: List[int]) -> List[int]:
        """Get indices for this rank with advanced sharding."""
        if not self.enable_load_balancing:
            # Standard sharding
            return indices[self.rank:self.total_size:self.num_replicas]

        # Advanced load-balanced sharding
        chunk_size = len(indices) // self.num_replicas
        remainder = len(indices) % self.num_replicas

        # Calculate start and end indices for this rank
        if self.rank < remainder:
            # First `remainder` ranks get one extra sample
            start_idx = self.rank * (chunk_size + 1)
            end_idx = start_idx + chunk_size + 1
        else:
            # Remaining ranks get standard chunk size
            start_idx = remainder * (chunk_size + 1) + (self.rank - remainder) * chunk_size
            end_idx = start_idx + chunk_size

        rank_indices = indices[start_idx:end_idx]

        # Monitor load balance
        expected_samples = len(indices) / self.num_replicas
        actual_samples = len(rank_indices)
        load_imbalance = abs(actual_samples - expected_samples) / expected_samples

        if load_imbalance > self.balancing_tolerance:
            logger.warning(
                f"Load imbalance detected on rank {self.rank}: "
                f"{actual_samples} samples vs {expected_samples:.1f} expected "
                f"(imbalance: {load_imbalance:.1%})"
            )

        self.load_stats['load_ratio'] = actual_samples / expected_samples

        return rank_indices

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler."""
        self.epoch = epoch

    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics for this rank."""
        return self.load_stats.copy()

    def coordinate_resharding(self, failed_ranks: List[int]) -> bool:
        """
        Coordinate resharding when ranks fail.

        Args:
            failed_ranks: List of failed rank IDs

        Returns:
            bool: True if resharding successful
        """
        if not failed_ranks:
            return True

        active_ranks = [r for r in range(self.num_replicas) if r not in failed_ranks]

        if self.rank in failed_ranks:
            logger.error(f"Rank {self.rank} is marked as failed - cannot reshard")
            return False

        if self.rank not in active_ranks:
            logger.error(f"Rank {self.rank} not in active ranks: {active_ranks}")
            return False

        # Update num_replicas for resharding
        old_replicas = self.num_replicas
        self.num_replicas = len(active_ranks)

        # Recalculate samples per rank
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        logger.info(
            f"Rank {self.rank}: Resharded from {old_replicas} to {self.num_replicas} replicas, "
            f"new samples per rank: {self.num_samples}"
        )

        return True


__all__ = [
    'DistributedStreamingDataset',
    'AdvancedDistributedSampler',
    'DISTRIBUTED_AVAILABLE',
]
