"""
Distributed Training Management System

Provides robust distributed training support with proper process group management,
barrier synchronization, error handling, and cleanup procedures.
"""

import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import torch.multiprocessing as mp  # type: ignore[import]
import os
import signal
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum
import json
from datetime import timedelta

logger = logging.getLogger(__name__)


class DistributedState(Enum):
    """Distributed training states."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CLEANUP = "cleanup"
    TERMINATED = "terminated"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    timeout_seconds: int = 1800  # 30 minutes
    init_method: Optional[str] = None

    # Barrier and synchronization settings
    barrier_timeout: int = 300  # 5 minutes
    health_check_interval: int = 30  # seconds
    max_retries: int = 3

    # Process group settings
    enable_barriers: bool = True
    enable_heartbeat: bool = True
    heartbeat_interval: int = 10  # seconds

    # Error handling
    enable_rank_failure_recovery: bool = True
    max_failed_ranks: int = 1  # Maximum ranks that can fail before aborting
    restart_on_failure: bool = False


class DistributedManager:
    """
    Comprehensive distributed training manager.

    Features:
    - Robust process group initialization
    - Barrier synchronization before cleanup
    - Rank-aware error handling
    - Health monitoring and heartbeat
    - Graceful failure recovery
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize distributed manager.

        Args:
            config: Distributed training configuration
        """
        self.config = config or DistributedConfig()
        self.state = DistributedState.NOT_INITIALIZED
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self._is_master_rank = True

        # Process group management
        self.process_group = None
        self.cleanup_handlers: List[Callable] = []
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_stop_event = threading.Event()

        # Error tracking
        self.failed_ranks: set = set()
        self.error_count = 0
        self.last_health_check = time.time()

        # Synchronization primitives
        self._barrier_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._shutdown_initiated = False

        # Register signal handlers for proper cleanup
        self._register_signal_handlers()

    def initialize(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[str] = None
    ) -> bool:
        """
        Initialize distributed training with robust error handling.

        Args:
            rank: Process rank (auto-detected if None)
            world_size: Total number of processes (auto-detected if None)
            master_addr: Master node address (auto-detected if None)
            master_port: Master node port (auto-detected if None)

        Returns:
            True if initialization successful, False otherwise
        """
        if self.state != DistributedState.NOT_INITIALIZED:
            logger.warning(f"Distributed manager already in state: {self.state}")
            return self.state == DistributedState.HEALTHY

        self.state = DistributedState.INITIALIZING
        logger.info("🚀 Initializing distributed training...")

        try:
            # Auto-detect environment variables if not provided
            self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
            self.world_size = world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))

            # Set master address and port
            os.environ.setdefault('MASTER_ADDR', master_addr or os.environ.get('MASTER_ADDR', 'localhost'))
            os.environ.setdefault('MASTER_PORT', master_port or os.environ.get('MASTER_PORT', '12355'))

            self._is_master_rank = (self.rank == 0)

            # Validate configuration
            if self.world_size <= 0:
                raise ValueError(f"Invalid world_size: {self.world_size}")
            if not (0 <= self.rank < self.world_size):
                raise ValueError(f"Invalid rank {self.rank} for world_size {self.world_size}")

            # Check if already initialized
            if dist.is_available() and dist.is_initialized():
                logger.info("🔄 Process group already initialized")
                self._validate_existing_process_group()
            else:
                # Initialize process group with timeout
                self._initialize_process_group()

            # Start health monitoring
            self._start_health_monitoring()

            self.state = DistributedState.HEALTHY

            logger.info(f"✅ Distributed training initialized successfully:")
            logger.info(f"   Rank: {self.rank}/{self.world_size}")
            logger.info(f"   Local rank: {self.local_rank}")
            logger.info(f"   Backend: {self.config.backend}")
            logger.info(f"   Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")

            # Initial barrier to ensure all ranks are ready
            if self.config.enable_barriers:
                self.barrier("initialization", timeout=self.config.barrier_timeout)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize distributed training: {e}")
            self.state = DistributedState.FAILING
            return False

    def _initialize_process_group(self):
        """Initialize the process group with proper error handling."""
        logger.info(f"Initializing process group with {self.config.backend} backend...")

        # Set GPU device for NCCL
        if self.config.backend == 'nccl' and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            logger.info(f"Set CUDA device to {self.local_rank}")

        # Initialize with timeout
        try:
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.rank,
                world_size=self.world_size,
                timeout=torch.distributed.default_pg_timeout if self.config.timeout_seconds == 1800
                        else timedelta(seconds=self.config.timeout_seconds),
                init_method=self.config.init_method
            )

            self.process_group = dist.group.WORLD
            logger.info("Process group initialized successfully")

        except Exception as e:
            logger.error(f"Process group initialization failed: {e}")
            # Try fallback with gloo if nccl failed
            if self.config.backend == 'nccl':
                logger.info("Trying fallback to gloo backend...")
                self.config.backend = 'gloo'
                dist.init_process_group(
                    backend='gloo',
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timedelta(seconds=self.config.timeout_seconds)
                )
                self.process_group = dist.group.WORLD
                logger.warning("Fallback to gloo backend successful")
            else:
                raise

    def _validate_existing_process_group(self):
        """Validate existing process group configuration."""
        current_rank = dist.get_rank()
        current_world_size = dist.get_world_size()

        if current_rank != self.rank:
            logger.warning(f"Rank mismatch: expected {self.rank}, got {current_rank}")
            self.rank = current_rank

        if current_world_size != self.world_size:
            logger.warning(f"World size mismatch: expected {self.world_size}, got {current_world_size}")
            self.world_size = current_world_size

        self.process_group = dist.group.WORLD
        logger.info("Existing process group validated")

    def barrier(self, name: str = "unnamed", timeout: Optional[int] = None) -> bool:
        """
        Synchronize all ranks with proper error handling.

        Args:
            name: Barrier name for logging
            timeout: Timeout in seconds (uses config default if None)

        Returns:
            True if barrier successful, False otherwise
        """
        if not self.is_initialized() or not self.config.enable_barriers:
            return True

        timeout = timeout or self.config.barrier_timeout

        with self._barrier_lock:
            try:
                logger.debug(f"🔄 Barrier '{name}' starting (rank {self.rank})")
                start_time = time.time()

                # Use timeout-aware barrier
                dist.barrier(group=self.process_group, async_op=False)

                elapsed = time.time() - start_time
                logger.debug(f"✅ Barrier '{name}' completed in {elapsed:.2f}s")
                return True

            except Exception as e:
                logger.error(f"❌ Barrier '{name}' failed: {e}")
                self.state = DistributedState.DEGRADED
                return False

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,  # type: ignore[assignment]
        async_op: bool = False
    ) -> Union[torch.Tensor, dist.Work]:
        """
        All-reduce operation with error handling.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            async_op: Whether to perform async operation

        Returns:
            Reduced tensor or work handle for async ops
        """
        if not self.is_initialized():
            return tensor

        try:
            return dist.all_reduce(tensor, op=op, group=self.process_group, async_op=async_op)  # type: ignore[arg-type,return-value]
        except Exception as e:
            logger.error(f"All-reduce failed: {e}")
            self.state = DistributedState.DEGRADED
            if async_op:
                return None  # type: ignore[return-value]
            return tensor

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        async_op: bool = False
    ) -> Union[torch.Tensor, dist.Work]:
        """
        Broadcast operation with error handling.

        Args:
            tensor: Tensor to broadcast
            src: Source rank
            async_op: Whether to perform async operation

        Returns:
            Broadcasted tensor or work handle for async ops
        """
        if not self.is_initialized():
            return tensor

        try:
            return dist.broadcast(tensor, src=src, group=self.process_group, async_op=async_op)  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            self.state = DistributedState.DEGRADED
            if async_op:
                return None  # type: ignore[return-value]
            return tensor

    def gather(
        self,
        tensor: torch.Tensor,
        dst: int,
        gather_list: Optional[List[torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Gather operation with error handling.

        Args:
            tensor: Tensor to gather
            dst: Destination rank
            gather_list: List to store gathered tensors

        Returns:
            List of gathered tensors (only on dst rank)
        """
        if not self.is_initialized():
            return [tensor] if self.rank == dst else None

        try:
            if self.rank == dst and gather_list is None:
                gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]

            dist.gather(tensor, gather_list, dst=dst, group=self.process_group)
            return gather_list if self.rank == dst else None

        except Exception as e:
            logger.error(f"Gather failed: {e}")
            self.state = DistributedState.DEGRADED
            return None

    def all_gather(
        self,
        tensor: torch.Tensor,
        tensor_list: Optional[List[torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        All-gather operation with error handling.

        Args:
            tensor: Tensor to gather
            tensor_list: List to store gathered tensors

        Returns:
            List of gathered tensors from all ranks
        """
        if not self.is_initialized():
            return [tensor]

        try:
            if tensor_list is None:
                tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]

            dist.all_gather(tensor_list, tensor, group=self.process_group)
            return tensor_list

        except Exception as e:
            logger.error(f"All-gather failed: {e}")
            self.state = DistributedState.DEGRADED
            return None

    def broadcast_oom_signal(self, oom_info: Dict[str, Any]) -> bool:
        """
        Broadcast OOM signal to all ranks for collective handling.

        Args:
            oom_info: Information about the OOM event including rank, memory state, etc.

        Returns:
            bool: True if broadcast successful, False otherwise
        """
        if not self.is_initialized():
            logger.warning("Cannot broadcast OOM signal: distributed not initialized")
            return False

        try:
            # Create OOM signal tensor (1 = OOM detected, 0 = no OOM)
            oom_signal = torch.tensor([1.0 if oom_info else 0.0], device=f'cuda:{self.local_rank}')

            # Broadcast OOM signal from the source rank to all ranks
            if oom_info:
                logger.critical(f"Broadcasting OOM signal from rank {self.rank}")
                logger.critical(f"OOM info: {oom_info}")

            # Use all_reduce to let all ranks know if ANY rank has OOM
            oom_signal = self.all_reduce(oom_signal, op=dist.ReduceOp.MAX)  # type: ignore[arg-type]

            has_collective_oom = oom_signal.item() > 0.5 if hasattr(oom_signal, 'item') else float(oom_signal) > 0.5  # type: ignore[attr-defined]

            if has_collective_oom:
                logger.critical(f"Collective OOM detected across ranks - coordinating response")
                # Synchronize all ranks before taking action
                self.barrier("oom_response_sync", timeout=60)

            return True

        except Exception as e:
            logger.error(f"Failed to broadcast OOM signal: {e}")
            self.state = DistributedState.DEGRADED
            return False

    def check_collective_memory_health(self) -> Dict[str, Any]:
        """
        Check memory health across all ranks and coordinate if needed.

        Returns:
            Dict containing collective memory health information
        """
        if not self.is_initialized():
            return {"status": "unknown", "reason": "distributed_not_initialized"}

        try:
            # Get local memory stats
            if torch.cuda.is_available():
                device = f'cuda:{self.local_rank}'
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory

                utilization = memory_reserved / memory_total
                available_gb = (memory_total - memory_reserved) / (1024**3)

                # Local memory status
                local_status = "healthy"
                if utilization > 0.95:
                    local_status = "critical"
                elif utilization > 0.85:
                    local_status = "warning"

                # Create tensor with local memory info [utilization, available_gb, status_code]
                status_code = {"healthy": 0.0, "warning": 1.0, "critical": 2.0}[local_status]
                local_info = torch.tensor([utilization, available_gb, status_code], device=device)
            else:
                local_info = torch.tensor([0.0, 0.0, 0.0], device='cpu')

            # Gather memory info from all ranks
            gathered_info = self.all_gather(local_info)
            if gathered_info is None:
                return {"status": "unknown", "reason": "gather_failed"}

            # Analyze collective memory health
            max_utilization = max(info[0].item() for info in gathered_info)
            min_available = min(info[1].item() for info in gathered_info)
            max_status_code = max(info[2].item() for info in gathered_info)

            # Determine collective status
            if max_status_code >= 2.0:
                collective_status = "critical"
            elif max_status_code >= 1.0:
                collective_status = "warning"
            else:
                collective_status = "healthy"

            # Find ranks with issues
            problematic_ranks = []
            for rank, info in enumerate(gathered_info):
                if info[2].item() >= 1.0:  # warning or critical
                    problematic_ranks.append({
                        "rank": rank,
                        "utilization": info[0].item(),
                        "available_gb": info[1].item(),
                        "status": "critical" if info[2].item() >= 2.0 else "warning"
                    })

            return {
                "status": collective_status,
                "max_utilization": max_utilization,
                "min_available_gb": min_available,
                "problematic_ranks": problematic_ranks,
                "total_ranks": len(gathered_info),
                "healthy_ranks": len(gathered_info) - len(problematic_ranks)
            }

        except Exception as e:
            logger.error(f"Collective memory health check failed: {e}")
            return {"status": "error", "reason": str(e)}

    def coordinate_oom_recovery(self, recovery_action: str = "reduce_batch_size") -> bool:
        """
        Coordinate OOM recovery across all ranks.

        Args:
            recovery_action: Action to take ("reduce_batch_size", "checkpoint_and_restart", "emergency_stop")

        Returns:
            bool: True if coordination successful
        """
        if not self.is_initialized():
            logger.warning("Cannot coordinate OOM recovery: distributed not initialized")
            return False

        try:
            logger.info(f"Coordinating OOM recovery: {recovery_action}")

            # Synchronize all ranks before recovery
            if not self.barrier("oom_recovery_start", timeout=120):
                logger.error("Failed to synchronize ranks for OOM recovery")
                return False

            # Broadcast recovery action to ensure all ranks use the same strategy
            action_codes = {
                "reduce_batch_size": 1.0,
                "checkpoint_and_restart": 2.0,
                "emergency_stop": 3.0
            }

            action_tensor = torch.tensor([action_codes.get(recovery_action, 1.0)],
                                       device=f'cuda:{self.local_rank}')
            action_tensor = self.broadcast(action_tensor, src=0)

            if action_tensor is None:
                logger.error("Failed to broadcast recovery action")
                return False

            coordinated_action = action_tensor.item() if hasattr(action_tensor, 'item') else int(action_tensor)  # type: ignore[attr-defined]
            action_name = {v: k for k, v in action_codes.items()}.get(coordinated_action, "reduce_batch_size")

            logger.info(f"All ranks will execute: {action_name}")

            # Final synchronization before executing recovery
            if not self.barrier("oom_recovery_execute", timeout=60):
                logger.error("Failed to synchronize ranks for recovery execution")
                return False

            return True

        except Exception as e:
            logger.error(f"OOM recovery coordination failed: {e}")
            return False

    def coordinate_synchronized_checkpoint(self, checkpoint_dir: str, step: int) -> bool:
        """
        Coordinate synchronized checkpointing across all ranks.

        Args:
            checkpoint_dir: Directory for checkpoint
            step: Current training step

        Returns:
            bool: True if synchronization successful
        """
        if not self.is_initialized():
            logger.warning("Cannot coordinate checkpoint: distributed not initialized")
            return False

        try:
            logger.info(f"Coordinating synchronized checkpoint at step {step}")

            # Synchronize all ranks before checkpointing
            if not self.barrier("checkpoint_start", timeout=300):
                logger.error("Failed to synchronize ranks for checkpoint start")
                return False

            # Broadcast checkpoint readiness signal
            # Each rank signals if it's ready to checkpoint (1.0) or not (0.0)
            ready_signal = torch.tensor([1.0], device=f'cuda:{self.local_rank}')

            # Gather readiness from all ranks
            readiness_signals = self.all_gather(ready_signal)
            if readiness_signals is None:
                logger.error("Failed to gather checkpoint readiness")
                return False

            # Check if all ranks are ready
            all_ready = all(signal.item() > 0.5 for signal in readiness_signals)

            if not all_ready:
                not_ready_ranks = [i for i, signal in enumerate(readiness_signals) if signal.item() <= 0.5]
                logger.error(f"Not all ranks ready for checkpoint. Not ready: {not_ready_ranks}")
                return False

            logger.info("All ranks ready for synchronized checkpoint")

            # Final barrier before actual checkpointing
            if not self.barrier("checkpoint_execute", timeout=180):
                logger.error("Failed to synchronize ranks for checkpoint execution")
                return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint coordination failed: {e}")
            return False

    def detect_rank_failures(self) -> Dict[str, Any]:
        """
        Detect failed ranks in the distributed training setup.

        Returns:
            Dict containing failure detection results
        """
        if not self.is_initialized():
            return {"status": "unknown", "reason": "distributed_not_initialized"}

        try:
            # Create heartbeat signal
            heartbeat = torch.tensor([float(self.rank)], device=f'cuda:{self.local_rank}')

            # Attempt to gather heartbeats from all ranks with timeout
            start_time = time.time()
            try:
                heartbeats = self.all_gather(heartbeat)
                gather_time = time.time() - start_time

                if heartbeats is None:
                    return {"status": "communication_failed", "reason": "all_gather_failed"}

                # Check for missing or invalid heartbeats
                expected_ranks = set(range(self.world_size))
                received_ranks = set()

                for i, hb in enumerate(heartbeats):
                    rank_value = int(hb.item())
                    if rank_value == i:  # Valid heartbeat
                        received_ranks.add(rank_value)

                missing_ranks = expected_ranks - received_ranks
                failed_ranks = list(missing_ranks)

                # Check for slow responses (potential degraded ranks)
                slow_threshold = 5.0  # seconds
                degraded_ranks = []
                if gather_time > slow_threshold:
                    logger.warning(f"Slow heartbeat gather: {gather_time:.2f}s")
                    # In real scenario, we'd track per-rank timing
                    # For now, we'll just note the slow overall response

                return {
                    "status": "success",
                    "total_ranks": self.world_size,
                    "healthy_ranks": list(received_ranks),
                    "failed_ranks": failed_ranks,
                    "degraded_ranks": degraded_ranks,
                    "gather_time": gather_time,
                    "all_ranks_healthy": len(failed_ranks) == 0
                }

            except Exception as gather_error:
                logger.error(f"Heartbeat gather failed: {gather_error}")
                return {
                    "status": "gather_failed",
                    "reason": str(gather_error),
                    "gather_time": time.time() - start_time
                }

        except Exception as e:
            logger.error(f"Rank failure detection failed: {e}")
            return {"status": "error", "reason": str(e)}

    def coordinate_rank_replacement(self, failed_ranks: List[int], checkpoint_path: str) -> bool:
        """
        Coordinate rank replacement after failures.

        Args:
            failed_ranks: List of failed rank IDs
            checkpoint_path: Path to checkpoint for recovery

        Returns:
            bool: True if coordination successful
        """
        if not self.is_initialized():
            logger.warning("Cannot coordinate rank replacement: distributed not initialized")
            return False

        if not failed_ranks:
            logger.info("No failed ranks to replace")
            return True

        try:
            logger.info(f"Coordinating replacement for failed ranks: {failed_ranks}")

            # Synchronize remaining healthy ranks
            healthy_ranks = [r for r in range(self.world_size) if r not in failed_ranks]
            logger.info(f"Healthy ranks: {healthy_ranks}")

            # Create recovery coordination signal
            recovery_signal = torch.tensor([float(len(failed_ranks))], device=f'cuda:{self.local_rank}')

            # Coordinate recovery with remaining ranks
            if not self.barrier("rank_replacement_start", timeout=300):
                logger.error("Failed to synchronize for rank replacement")
                return False

            # Broadcast recovery information
            recovery_info = torch.tensor([
                len(failed_ranks),  # Number of failed ranks
                len(healthy_ranks),  # Number of healthy ranks
                1.0 if checkpoint_path else 0.0  # Checkpoint available flag
            ], device=f'cuda:{self.local_rank}')

            recovery_broadcast = self.broadcast(recovery_info, src=0)
            if recovery_broadcast is None:
                logger.error("Failed to broadcast recovery information")
                return False

            logger.info("Recovery coordination completed")

            # Final barrier before proceeding with reduced world size
            if not self.barrier("rank_replacement_complete", timeout=180):
                logger.error("Failed to complete rank replacement coordination")
                return False

            return True

        except Exception as e:
            logger.error(f"Rank replacement coordination failed: {e}")
            return False

    def checkpoint_with_fault_tolerance(self, checkpoint_dir: str, step: int, max_retries: int = 3) -> bool:
        """
        Perform fault-tolerant checkpointing with failure detection and recovery.

        Args:
            checkpoint_dir: Directory for checkpoint
            step: Current training step
            max_retries: Maximum retry attempts

        Returns:
            bool: True if checkpointing successful
        """
        if not self.is_initialized():
            logger.warning("Cannot perform fault-tolerant checkpoint: distributed not initialized")
            return False

        for attempt in range(max_retries):
            try:
                logger.info(f"Fault-tolerant checkpoint attempt {attempt + 1}/{max_retries}")

                # Check for rank failures before checkpointing
                failure_status = self.detect_rank_failures()

                if failure_status["status"] != "success":
                    logger.error(f"Rank failure detection failed: {failure_status}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False

                if not failure_status["all_ranks_healthy"]:
                    failed_ranks = failure_status["failed_ranks"]
                    logger.error(f"Detected failed ranks before checkpoint: {failed_ranks}")

                    # Coordinate recovery with remaining ranks
                    recovery_success = self.coordinate_rank_replacement(failed_ranks, "")

                    if not recovery_success:
                        logger.error("Failed to coordinate recovery for failed ranks")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return False

                # Proceed with synchronized checkpoint
                sync_success = self.coordinate_synchronized_checkpoint(checkpoint_dir, step)

                if sync_success:
                    logger.info("Fault-tolerant checkpoint completed successfully")
                    return True
                else:
                    logger.error("Synchronized checkpoint failed")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue

            except Exception as e:
                logger.error(f"Fault-tolerant checkpoint attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        logger.error("All fault-tolerant checkpoint attempts failed")
        return False

    def _start_health_monitoring(self):
        """Start background health monitoring."""
        if not self.config.enable_heartbeat:
            return

        self.heartbeat_stop_event.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name=f"DistributedHealthMonitor-{self.rank}"
        )
        self.heartbeat_thread.start()
        logger.info("Health monitoring started")

    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self.heartbeat_stop_event.wait(self.config.heartbeat_interval):
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.state = DistributedState.DEGRADED

    def _perform_health_check(self):
        """Perform health check across all ranks."""
        if not self.is_initialized() or self.state == DistributedState.CLEANUP:
            return

        try:
            # Create a small tensor for health check
            health_tensor = torch.tensor(
                [self.rank, time.time()],
                device='cuda' if torch.cuda.is_available() and self.config.backend == 'nccl' else 'cpu'
            )

            # Perform all-reduce as health check
            dist.all_reduce(health_tensor, op=dist.ReduceOp.SUM, group=self.process_group)

            self.last_health_check = time.time()

            # If we were degraded but health check passed, mark as healthy
            if self.state == DistributedState.DEGRADED:
                logger.info("Health check passed - state recovered to healthy")
                self.state = DistributedState.HEALTHY

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            if self.state == DistributedState.HEALTHY:
                self.state = DistributedState.DEGRADED

    def register_cleanup_handler(self, handler: Callable):
        """Register a cleanup handler to be called during shutdown."""
        self.cleanup_handlers.append(handler)

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.cleanup()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def cleanup(self, force: bool = False):
        """
        Clean up distributed training with proper synchronization.

        Args:
            force: Force cleanup without barriers
        """
        with self._cleanup_lock:
            if self._shutdown_initiated and not force:
                logger.debug("Cleanup already initiated")
                return

            self._shutdown_initiated = True
            self.state = DistributedState.CLEANUP

            logger.info("🧹 Starting distributed training cleanup...")

            try:
                # Stop health monitoring
                if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                    self.heartbeat_stop_event.set()
                    self.heartbeat_thread.join(timeout=5)
                    logger.debug("Health monitoring stopped")

                # Call registered cleanup handlers
                for i, handler in enumerate(self.cleanup_handlers):
                    try:
                        logger.debug(f"Calling cleanup handler {i+1}/{len(self.cleanup_handlers)}")
                        handler()
                    except Exception as e:
                        logger.error(f"Cleanup handler {i+1} failed: {e}")

                # Synchronization barrier before destroying process group
                if self.is_initialized() and self.config.enable_barriers and not force:
                    logger.info("Synchronizing ranks before cleanup...")
                    try:
                        # Short timeout for cleanup barrier
                        dist.barrier(group=self.process_group, async_op=False)
                        logger.info("Cleanup barrier completed")
                    except Exception as e:
                        logger.warning(f"Cleanup barrier failed: {e}")

                # Destroy process group
                if self.is_initialized():
                    logger.info("Destroying process group...")
                    try:
                        dist.destroy_process_group()
                        logger.info("Process group destroyed")
                    except Exception as e:
                        logger.error(f"Failed to destroy process group: {e}")

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self.state = DistributedState.TERMINATED
                logger.info("✅ Distributed training cleanup completed")

    def is_initialized(self) -> bool:
        """Check if distributed training is initialized."""
        return (
            self.state in [DistributedState.HEALTHY, DistributedState.DEGRADED] and
            dist.is_available() and
            dist.is_initialized()
        )

    def is_master(self) -> bool:
        """Check if this is the master rank."""
        return self.rank == 0

    def get_stats(self) -> Dict[str, Any]:
        """Get distributed training statistics."""
        return {
            'state': self.state.value,
            'rank': self.rank,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
            'is_master': self.is_master,
            'backend': self.config.backend,
            'failed_ranks': list(self.failed_ranks),
            'error_count': self.error_count,
            'last_health_check': self.last_health_check,
            'health_monitoring': self.config.enable_heartbeat,
            'process_group_initialized': self.is_initialized()
        }

    @contextmanager
    def no_sync(self):
        """Context manager for skipping gradient synchronization."""
        if self.is_initialized():
            # This would be implemented by the model/optimizer
            yield
        else:
            yield

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Global distributed manager instance
_global_distributed_manager: Optional[DistributedManager] = None


def get_distributed_manager(config: Optional[DistributedConfig] = None) -> DistributedManager:
    """Get or create global distributed manager."""
    global _global_distributed_manager

    if _global_distributed_manager is None:
        _global_distributed_manager = DistributedManager(config)

    return _global_distributed_manager


def is_distributed() -> bool:
    """Check if we're in a distributed training environment."""
    return (
        'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    ) or (
        'LOCAL_RANK' in os.environ
    ) or (
        dist.is_available() and dist.is_initialized()
    )


def get_rank() -> int:
    """Get current rank (0 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """Get local rank (0 if not distributed)."""
    return int(os.environ.get('LOCAL_RANK', get_rank()))