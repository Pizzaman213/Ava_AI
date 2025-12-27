"""Distributed state synchronization utilities.

This module provides utilities for safe distributed state updates across ranks
to prevent race conditions and desynchronization issues in multi-GPU training.

Key Features:
- Two-phase commit pattern for state updates
- Automatic barrier synchronization
- Validation hooks for synchronized values
- Context managers for config updates

Example:
    >>> dist_manager = DistributedStateManager(rank=0, world_size=4)
    >>> synced_value = dist_manager.synchronized_update(
    ...     value=128,
    ...     reduction_op=dist.ReduceOp.MIN,
    ...     validate_fn=lambda x: x > 0
    ... )
    >>> with dist_manager.config_update_context("batch_size"):
    ...     config['training']['batch_size'] = synced_value
"""

import torch
import torch.distributed as dist
from contextlib import contextmanager
from typing import TypeVar, Optional, Callable, Union
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', int, float)


class DistributedStateManager:
    """Manages synchronized state updates across distributed ranks.

    This class implements a two-phase commit pattern to ensure all ranks
    see the same value before proceeding with state updates. This prevents
    race conditions where different ranks might have different configuration
    values, leading to training hangs or crashes.

    Attributes:
        rank: Current process rank
        world_size: Total number of processes
        timeout: Timeout for barrier operations
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        timeout_seconds: int = 300
    ):
        """Initialize the distributed state manager.

        Args:
            rank: Current process rank (0 to world_size-1)
            world_size: Total number of processes in distributed group
            timeout_seconds: Timeout for barrier operations (default: 300s)
        """
        self.rank = rank
        self.world_size = world_size
        self.timeout = timedelta(seconds=timeout_seconds)
        self._is_distributed = world_size > 1

        logger.debug(
            f"DistributedStateManager initialized: "
            f"rank={rank}, world_size={world_size}, "
            f"distributed={self._is_distributed}"
        )

    def synchronized_update(
        self,
        value: T,
        reduction_op: dist.ReduceOp = dist.ReduceOp.MIN,
        validate_fn: Optional[Callable[[T], bool]] = None,
        value_name: str = "value"
    ) -> T:
        """Synchronize a value across all ranks using two-phase commit.

        This method implements a two-phase commit pattern:
        1. Phase 1: All-reduce operation to compute synchronized value
        2. Phase 2: Barrier to ensure all ranks proceed together
        3. Phase 3: Optional validation of synchronized value

        The synchronized value is determined by the reduction operation:
        - MIN: Use minimum value across all ranks (safe for batch sizes)
        - MAX: Use maximum value across all ranks
        - SUM: Sum values across all ranks
        - AVG: Average values across all ranks

        Args:
            value: Local value to synchronize
            reduction_op: Reduction operation (default: MIN for safety)
            validate_fn: Optional validation function, should return True if valid
            value_name: Name of value for logging (default: "value")

        Returns:
            Synchronized value that is identical across all ranks

        Raises:
            RuntimeError: If validation fails on synchronized value

        Example:
            >>> # All ranks propose batch sizes, use minimum for safety
            >>> synced_bs = dist_manager.synchronized_update(
            ...     value=local_batch_size,
            ...     reduction_op=dist.ReduceOp.MIN,
            ...     validate_fn=lambda x: x > 0,
            ...     value_name="batch_size"
            ... )
        """
        if not self._is_distributed:
            # Single GPU - no synchronization needed
            if validate_fn and not validate_fn(value):
                raise RuntimeError(
                    f"Validation failed for {value_name}={value} (single GPU)"
                )
            return value

        # Phase 1: All-reduce to get synchronized value
        # Use int64 for integers, float32 for floats
        if isinstance(value, int):
            tensor = torch.tensor([value], dtype=torch.int64, device='cuda')
        else:
            tensor = torch.tensor([value], dtype=torch.float32, device='cuda')

        try:
            dist.all_reduce(tensor, op=reduction_op)
        except RuntimeError as e:
            logger.error(
                f"Rank {self.rank}: All-reduce failed for {value_name}: {e}"
            )
            raise

        # Extract synchronized value
        if isinstance(value, int):
            synced_value = int(tensor.item())
        else:
            synced_value = float(tensor.item())

        # Log if values differ across ranks
        if synced_value != value and self.rank == 0:
            logger.warning(
                f"Rank {self.rank}: {value_name} synchronized "
                f"from {value} to {synced_value} (op={reduction_op})"
            )

        # Phase 2: Barrier ensures all ranks proceed together
        try:
            dist.barrier(timeout=self.timeout)
        except RuntimeError as e:
            logger.error(
                f"Rank {self.rank}: Barrier timeout during {value_name} sync: {e}"
            )
            raise

        # Phase 3: Optional validation
        if validate_fn and not validate_fn(synced_value):
            raise RuntimeError(
                f"Rank {self.rank}: Synchronized {value_name}={synced_value} "
                f"failed validation (proposed: {value})"
            )

        logger.debug(
            f"Rank {self.rank}: {value_name} synchronized to {synced_value}"
        )

        return synced_value

    @contextmanager
    def config_update_context(self, config_name: str):
        """Context manager for safe distributed config updates.

        This ensures that config mutations happen atomically across all ranks:
        1. Barrier before update (all ranks ready)
        2. Config update happens
        3. Barrier after update (all ranks synchronized)

        This prevents race conditions where some ranks might read old values
        while others have already updated to new values.

        Args:
            config_name: Name of config field being updated (for logging)

        Yields:
            None

        Example:
            >>> with dist_manager.config_update_context("batch_size"):
            ...     config['training']['batch_size'] = new_batch_size
            # All ranks now have updated config
        """
        if self._is_distributed:
            try:
                # Barrier before update
                dist.barrier(timeout=self.timeout)
                logger.debug(
                    f"Rank {self.rank}: Entering config update for {config_name}"
                )
            except RuntimeError as e:
                logger.error(
                    f"Rank {self.rank}: Barrier timeout before {config_name} update: {e}"
                )
                raise

        try:
            yield
        finally:
            if self._is_distributed:
                try:
                    # Barrier after update
                    dist.barrier(timeout=self.timeout)
                    logger.debug(
                        f"Rank {self.rank}: Exiting config update for {config_name}"
                    )
                except RuntimeError as e:
                    logger.error(
                        f"Rank {self.rank}: Barrier timeout after {config_name} update: {e}"
                    )
                    raise

    def barrier(self, tag: str = ""):
        """Explicit barrier with logging.

        Args:
            tag: Optional tag for logging (e.g., "pre_training", "post_validation")
        """
        if not self._is_distributed:
            return

        logger.debug(f"Rank {self.rank}: Barrier {tag}")
        try:
            dist.barrier(timeout=self.timeout)
        except RuntimeError as e:
            logger.error(f"Rank {self.rank}: Barrier timeout ({tag}): {e}")
            raise

    def is_main_rank(self) -> bool:
        """Check if this is the main rank (rank 0).

        Returns:
            True if rank 0, False otherwise
        """
        return self.rank == 0

    def all_gather_values(
        self,
        value: Union[int, float],
        value_name: str = "value"
    ) -> list:
        """Gather values from all ranks to rank 0.

        Args:
            value: Local value to gather
            value_name: Name of value for logging

        Returns:
            List of values from all ranks (only valid on rank 0)
        """
        if not self._is_distributed:
            return [value]

        if isinstance(value, int):
            tensor = torch.tensor([value], dtype=torch.int64, device='cuda')
        else:
            tensor = torch.tensor([value], dtype=torch.float32, device='cuda')

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)

        if isinstance(value, int):
            values = [int(t.item()) for t in gathered]
        else:
            values = [float(t.item()) for t in gathered]

        if self.rank == 0:
            logger.debug(f"{value_name} across ranks: {values}")

        return values
