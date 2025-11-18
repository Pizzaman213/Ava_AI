"""
DistributedTrainingManager - Handles distributed training setup and synchronization.

Responsibilities:
- Distributed training initialization (NCCL, gloo, etc.)
- Rank/world size management
- Barrier synchronization with retry logic
- Rank failure detection and handling
- Health checking
- Cleanup and barrier coordination
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from .base import TrainingContext, ManagerInterface

logger = logging.getLogger(__name__)


class DistributedTrainingManager(ManagerInterface):
    """
    Manages all aspects of distributed training.

    Handles initialization, synchronization, failure detection, and cleanup.
    """

    def __init__(self, context: TrainingContext):
        """Initialize distributed manager."""
        super().__init__(context)
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.backend = "nccl"
        self._cleanup_handlers: List[Callable] = []
        self._barrier_timeout = 300  # 5 minutes
        self._health_check_interval = 30
        self._max_barrier_retries = 3

    def initialize(self) -> None:
        """Initialize distributed training if available."""
        try:
            if not dist.is_available():
                logger.info("Distributed training not available")
                super().initialize()
                return

            if not dist.is_initialized():
                logger.info("Distributed training not initialized yet")
                super().initialize()
                return

            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.backend = dist.get_backend()
            self.local_rank = int(
                torch.distributed.get_local_rank()
                if hasattr(dist, "get_local_rank")
                else 0
            )

            logger.info(
                f"Distributed training initialized: "
                f"rank={self.rank}, world_size={self.world_size}, "
                f"backend={self.backend}"
            )

            # Set device for this rank
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                if self.context.device is None:
                    self.context.device = torch.device(f"cuda:{self.local_rank}")

            super().initialize()

        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            super().initialize()

    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        try:
            # Call registered cleanup handlers
            for handler in reversed(self._cleanup_handlers):
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Error in cleanup handler: {e}")

            # Final barrier before destroying process group
            if self.is_distributed and dist.is_initialized():
                try:
                    self.barrier(timeout=10)
                except Exception as e:
                    logger.warning(f"Final barrier failed: {e}")

                dist.destroy_process_group()
                logger.info("Distributed training cleaned up")

        except Exception as e:
            logger.error(f"Error during distributed cleanup: {e}")

    def register_cleanup_handler(self, handler: Callable) -> None:
        """Register a cleanup handler to be called on shutdown."""
        self._cleanup_handlers.append(handler)

    def barrier(self, timeout: int = 300) -> None:
        """
        Synchronize all ranks with retry logic.

        Args:
            timeout: Timeout in seconds for barrier

        Raises:
            RuntimeError: If barrier fails after max retries
        """
        if not self.is_distributed:
            return

        for attempt in range(self._max_barrier_retries):
            try:
                dist.barrier(timeout=timeout)
                return
            except Exception as e:
                if attempt == self._max_barrier_retries - 1:
                    logger.error(f"Barrier failed after {self._max_barrier_retries} attempts: {e}")
                    raise RuntimeError(f"Distributed barrier failed: {e}") from e

                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Barrier attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                import time
                time.sleep(wait_time)

    def broadcast_tensor(
        self, tensor: torch.Tensor, src: int = 0
    ) -> torch.Tensor:
        """
        Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Tensor to broadcast
            src: Source rank

        Returns:
            Broadcasted tensor
        """
        if not self.is_distributed:
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def allreduce(
        self, tensor: torch.Tensor, op: str = "sum"
    ) -> torch.Tensor:
        """
        All-reduce operation across ranks.

        Args:
            tensor: Tensor to reduce
            op: Operation (sum, mean, max, min)

        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Manual mean after
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }

        if op not in op_map:
            raise ValueError(f"Unknown reduce operation: {op}")

        dist.all_reduce(tensor, op=op_map[op])

        if op == "mean":
            tensor /= self.world_size

        return tensor

    def gather(
        self, tensor: torch.Tensor, dst: int = 0
    ) -> Optional[List[torch.Tensor]]:
        """
        Gather tensors from all ranks to destination rank.

        Args:
            tensor: Tensor to gather
            dst: Destination rank

        Returns:
            List of tensors on destination rank, None on other ranks
        """
        if not self.is_distributed:
            return [tensor]

        gathered = [None] * self.world_size if self.rank == dst else None
        dist.gather(tensor, gather_list=gathered, dst=dst)
        return gathered

    def is_main_rank(self) -> bool:
        """Check if this is the main rank."""
        return self.rank == 0

    def should_log(self) -> bool:
        """Check if this rank should log (only rank 0)."""
        return self.is_main_rank()

    def synchronize_state_dict(
        self, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensure all ranks have the same state dict (synchronize from rank 0).

        Args:
            state_dict: State dictionary

        Returns:
            Synchronized state dict
        """
        if not self.is_distributed:
            return state_dict

        # Serialize state dict on rank 0
        if self.is_main_rank():
            import pickle
            serialized = pickle.dumps(state_dict)
            size = len(serialized)
            size_tensor = torch.tensor(size, device=self.device)
        else:
            size_tensor = torch.tensor(0, device=self.device)

        # Broadcast size
        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        # Broadcast data
        if self.is_main_rank():
            data_buffer = torch.frombuffer(
                serialized, dtype=torch.uint8, device=self.device
            )
        else:
            data_buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)

        dist.broadcast(data_buffer, src=0)

        # Deserialize on other ranks
        if not self.is_main_rank():
            import pickle
            state_dict = pickle.loads(data_buffer.cpu().numpy().tobytes())

        return state_dict

    def on_rank_failure(self, failed_rank: int, error: Exception) -> None:
        """
        Handle failure of a rank.

        Args:
            failed_rank: Rank that failed
            error: Error that occurred
        """
        logger.error(
            f"Rank {failed_rank} failed with error: {error}. "
            f"This trainer does not support fault tolerance yet."
        )
        # TODO: Implement fault tolerance (checkpoint restore on failed rank)

    def get_status(self) -> Dict[str, Any]:
        """Return distributed training status."""
        return {
            "is_distributed": self.is_distributed,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "backend": self.backend,
            "is_main_rank": self.is_main_rank(),
        }
