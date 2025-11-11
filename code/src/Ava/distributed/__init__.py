"""
Distributed Training and Memory Management Module

Consolidated distributed computing functionality including:
- Distributed manager (process group management)
- Unified distributed manager (DDP interface)
- Distributed health checker (health monitoring)
- Rank-aware error handler (error handling across ranks)
- Memory monitor (memory management for distributed training)
- GPU load balancer (intelligent expert placement across GPUs)
- Expert parallelism (MoE expert sharding across GPUs)
- GPU utilities (GPU monitoring and management)
"""

from .distributed_manager import DistributedManager
from .unified_distributed_manager import UnifiedDistributedManager
from .distributed_health_checker import DistributedHealthChecker
from .rank_aware_error_handler import RankAwareErrorHandler
from .memory_monitor import MemoryMonitor
from .gpu_load_balancer import GPULoadBalancer, GPUStats, ExpertPlacement
from .expert_parallel import ExpertParallelManager, create_expert_parallel_manager
from .gpu_utils import (
    get_gpu_memory_stats,
    get_all_gpu_memory_stats,
    get_gpu_compute_utilization,
    balance_experts_across_gpus,
    estimate_expert_memory,
    check_gpu_availability,
    log_gpu_load_balance_status,
)

__all__ = [
    "DistributedManager",
    "UnifiedDistributedManager",
    "DistributedHealthChecker",
    "RankAwareErrorHandler",
    "MemoryMonitor",
    # GPU Load Balancing
    "GPULoadBalancer",
    "GPUStats",
    "ExpertPlacement",
    # Expert Parallelism
    "ExpertParallelManager",
    "create_expert_parallel_manager",
    # GPU Utilities
    "get_gpu_memory_stats",
    "get_all_gpu_memory_stats",
    "get_gpu_compute_utilization",
    "balance_experts_across_gpus",
    "estimate_expert_memory",
    "check_gpu_availability",
    "log_gpu_load_balance_status",
]
