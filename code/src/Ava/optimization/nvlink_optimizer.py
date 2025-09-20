"""
NVLink Topology Optimization for A100 Multi-GPU Systems

This module provides optimized communication patterns for NVIDIA A100 GPUs
connected via NVLink, enabling efficient distributed training.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class NVLinkOptimizer:
    """
    Optimizer for NVLink communication patterns on A100 systems.

    Features:
    - Topology-aware communication
    - Hierarchical AllReduce
    - GPU Direct RDMA support
    - Optimized gradient synchronization
    """

    def __init__(
        self,
        enable_nvlink_optimization: bool = True,
        enable_gpu_direct: bool = True,
        hierarchical_allreduce: bool = True,
        fusion_buffer_size_mb: int = 64,
        profile_communication: bool = False
    ):
        """
        Initialize NVLink optimizer.

        Args:
            enable_nvlink_optimization: Enable NVLink-specific optimizations
            enable_gpu_direct: Enable GPU Direct RDMA
            hierarchical_allreduce: Use hierarchical AllReduce
            fusion_buffer_size_mb: Size of fusion buffer in MB
            profile_communication: Enable communication profiling
        """
        self.enable_nvlink_optimization = enable_nvlink_optimization
        self.enable_gpu_direct = enable_gpu_direct
        self.hierarchical_allreduce = hierarchical_allreduce
        self.fusion_buffer_size_mb = fusion_buffer_size_mb
        self.profile_communication = profile_communication

        # Communication statistics
        self.comm_stats = {
            "total_allreduce_time": 0.0,
            "total_bytes_transferred": 0,
            "nvlink_bandwidth_gbps": 0.0,
            "communication_efficiency": 0.0
        }

        # Detect topology
        self.topology = self._detect_nvlink_topology()

    def _detect_nvlink_topology(self) -> Dict[str, Any]:
        """Detect NVLink topology for optimized communication."""
        topology = {
            "num_gpus": torch.cuda.device_count(),
            "nvlink_available": False,
            "nvlink_version": None,
            "topology_type": "unknown",
            "bandwidth_gbps": 0
        }

        if not torch.cuda.is_available():
            return topology

        # Check for NVLink availability
        # In production, this would use nvidia-ml-py or nvml
        device_props = torch.cuda.get_device_properties(0)

        # A100 has NVLink 3.0 with 600 GB/s bandwidth
        if "a100" in device_props.name.lower():
            topology["nvlink_available"] = True
            topology["nvlink_version"] = "3.0"
            topology["bandwidth_gbps"] = 600

            # Detect topology type based on GPU count
            num_gpus = topology["num_gpus"]
            if num_gpus == 2:
                topology["topology_type"] = "dual"
            elif num_gpus == 4:
                topology["topology_type"] = "quad"
            elif num_gpus == 8:
                topology["topology_type"] = "dgx_a100"  # DGX A100 configuration
            else:
                topology["topology_type"] = "custom"

        logger.info(f"Detected topology: {topology}")
        return topology

    @lru_cache(maxsize=1)
    def get_nvlink_matrix(self) -> np.ndarray:
        """
        Get NVLink connectivity matrix between GPUs.

        Returns:
            Connectivity matrix where matrix[i][j] = bandwidth between GPU i and j
        """
        num_gpus = self.topology["num_gpus"]
        matrix = np.zeros((num_gpus, num_gpus))

        if not self.topology["nvlink_available"]:
            # No NVLink, use PCIe bandwidth (16 GB/s)
            matrix.fill(16)
        else:
            # A100 NVLink topology
            if self.topology["topology_type"] == "dual":
                # 2 GPUs fully connected
                matrix = np.array([[0, 600],
                                  [600, 0]])
            elif self.topology["topology_type"] == "quad":
                # 4 GPUs in mesh topology
                matrix = np.array([[0, 600, 600, 300],
                                  [600, 0, 300, 600],
                                  [600, 300, 0, 600],
                                  [300, 600, 600, 0]])
            elif self.topology["topology_type"] == "dgx_a100":
                # 8 GPUs in DGX A100 topology (NVSwitch)
                # All GPUs connected with 600 GB/s via NVSwitch
                matrix = np.full((8, 8), 600)
            else:
                # Default: assume partial connectivity
                matrix = np.full((num_gpus, num_gpus), 300)

            # Diagonal is 0 (self-connection)
            np.fill_diagonal(matrix, 0)

        return matrix

    def optimize_ddp_model(
        self,
        model: nn.Module,
        device_ids: Optional[List[int]] = None,
        broadcast_buffers: bool = True
    ) -> DDP:
        """
        Optimize DistributedDataParallel model for NVLink.

        Args:
            model: PyTorch model
            device_ids: GPU device IDs
            broadcast_buffers: Whether to broadcast buffers

        Returns:
            Optimized DDP model
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized. Returning original model.")
            return model

        # Configure DDP for NVLink optimization
        ddp_config = {
            "device_ids": device_ids or [torch.cuda.current_device()],
            "broadcast_buffers": broadcast_buffers,
            "bucket_cap_mb": self.fusion_buffer_size_mb,
            "gradient_as_bucket_view": True,  # Memory optimization
            "static_graph": True  # Enable static graph optimization
        }

        # Add NVLink-specific optimizations
        if self.topology["nvlink_available"]:
            # Use larger buckets for NVLink's high bandwidth
            ddp_config["bucket_cap_mb"] = min(self.fusion_buffer_size_mb * 2, 128)

            # Enable find_unused_parameters only if necessary
            ddp_config["find_unused_parameters"] = False

        model = DDP(model, **ddp_config)

        # Configure NCCL for NVLink
        if self.enable_nvlink_optimization:
            self._configure_nccl_for_nvlink()

        logger.info(f"DDP model optimized with bucket_cap_mb={ddp_config['bucket_cap_mb']}")
        return model

    def _configure_nccl_for_nvlink(self):
        """Configure NCCL parameters for optimal NVLink performance."""
        # Set NCCL environment variables for A100 NVLink
        nccl_config = {
            "NCCL_TREE_THRESHOLD": "0",  # Always use tree algorithm
            "NCCL_LL_THRESHOLD": "0",  # Use low-latency algorithm
            "NCCL_NSOCKS_PERTHREAD": "8",  # Increase socket threads
            "NCCL_SOCKET_NTHREADS": "16",  # Increase network threads
            "NCCL_MIN_NCHANNELS": "16",  # Minimum channels for A100
            "NCCL_MAX_NCHANNELS": "16",  # Maximum channels for A100
            "NCCL_P2P_LEVEL": "NVL",  # Use NVLink for P2P
            "NCCL_SHM_DISABLE": "0",  # Enable shared memory
            "NCCL_CUMEM_ENABLE": "1"  # Enable CU memory pool
        }

        for key, value in nccl_config.items():
            os.environ[key] = value

        logger.info("NCCL configured for NVLink optimization")

    def hierarchical_allreduce(
        self,
        tensor: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        """
        Perform hierarchical AllReduce optimized for NVLink topology.

        Args:
            tensor: Tensor to reduce
            group: Process group

        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(group)

        if world_size <= 2 or not self.hierarchical_allreduce:
            # Small scale or disabled, use standard AllReduce
            dist.all_reduce(tensor, group=group)
            return tensor

        # Hierarchical reduction for large scale
        # Phase 1: Local reduction within NVLink-connected groups
        local_groups = self._get_local_nvlink_groups()

        for local_group in local_groups:
            if dist.get_rank() in local_group:
                # Create subgroup for local reduction
                local_pg = dist.new_group(ranks=local_group)
                dist.all_reduce(tensor, group=local_pg)
                break

        # Phase 2: Inter-group reduction (across NVLink domains)
        leader_ranks = [g[0] for g in local_groups]
        if dist.get_rank() in leader_ranks:
            leader_pg = dist.new_group(ranks=leader_ranks)
            dist.all_reduce(tensor, group=leader_pg)

        # Phase 3: Broadcast within local groups
        for local_group in local_groups:
            if dist.get_rank() in local_group:
                src = local_group[0]
                local_pg = dist.new_group(ranks=local_group)
                dist.broadcast(tensor, src=src, group=local_pg)
                break

        return tensor

    def _get_local_nvlink_groups(self) -> List[List[int]]:
        """Get groups of GPUs connected via NVLink."""
        world_size = dist.get_world_size() if dist.is_initialized() else torch.cuda.device_count()

        if self.topology["topology_type"] == "dgx_a100":
            # DGX A100: all 8 GPUs connected via NVSwitch
            return [list(range(8))]
        elif self.topology["topology_type"] == "quad":
            # 4 GPUs: two groups of 2
            return [[0, 1], [2, 3]]
        elif self.topology["topology_type"] == "dual":
            # 2 GPUs: single group
            return [[0, 1]]
        else:
            # Default: groups of 2
            groups = []
            for i in range(0, world_size, 2):
                if i + 1 < world_size:
                    groups.append([i, i + 1])
                else:
                    groups.append([i])
            return groups

    def optimize_gradient_synchronization(
        self,
        model: nn.Module,
        compression: str = "none",  # "none", "fp16", "powersgd"
        compression_ratio: float = 0.1
    ) -> None:
        """
        Optimize gradient synchronization for NVLink.

        Args:
            model: Model with gradients
            compression: Compression type
            compression_ratio: Compression ratio for PowerSGD
        """
        if compression == "fp16":
            # Convert gradients to FP16 for reduced bandwidth
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.half()

        elif compression == "powersgd":
            # PowerSGD compression (simplified version)
            # Real implementation would use proper PowerSGD algorithm
            for param in model.parameters():
                if param.grad is not None and param.grad.numel() > 1000:
                    # Only compress large gradients
                    grad_shape = param.grad.shape
                    grad_flat = param.grad.flatten()

                    # Simple compression via random projection
                    compressed_size = int(grad_flat.numel() * compression_ratio)
                    projection = torch.randn(grad_flat.numel(), compressed_size, device=grad_flat.device)
                    compressed = torch.matmul(grad_flat, projection)

                    # Decompress
                    decompressed = torch.matmul(compressed, projection.t())
                    param.grad = decompressed.reshape(grad_shape)

    def profile_communication_step(
        self,
        tensor_size_mb: float,
        operation: str = "allreduce"
    ) -> Dict[str, float]:
        """
        Profile a communication operation.

        Args:
            tensor_size_mb: Size of tensor in MB
            operation: Type of operation

        Returns:
            Profiling results
        """
        if not torch.cuda.is_available() or not dist.is_initialized():
            return {}

        tensor_size_bytes = int(tensor_size_mb * 1024 * 1024)
        tensor = torch.randn(tensor_size_bytes // 4).cuda()  # FP32

        # Warm up
        for _ in range(5):
            if operation == "allreduce":
                dist.all_reduce(tensor)
            elif operation == "broadcast":
                dist.broadcast(tensor, src=0)

        # Measure
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(10):
            if operation == "allreduce":
                dist.all_reduce(tensor)
            elif operation == "broadcast":
                dist.broadcast(tensor, src=0)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

        # Calculate bandwidth
        total_bytes = tensor_size_bytes * 10  # 10 iterations
        bandwidth_gbps = (total_bytes / elapsed_time) / (1024**3) * 8  # Convert to Gbps

        results = {
            "operation": operation,
            "tensor_size_mb": tensor_size_mb,
            "elapsed_time_s": elapsed_time,
            "bandwidth_gbps": bandwidth_gbps,
            "efficiency": bandwidth_gbps / self.topology["bandwidth_gbps"] if self.topology["bandwidth_gbps"] > 0 else 0
        }

        # Update stats
        self.comm_stats["total_allreduce_time"] += elapsed_time
        self.comm_stats["total_bytes_transferred"] += total_bytes
        self.comm_stats["nvlink_bandwidth_gbps"] = bandwidth_gbps
        self.comm_stats["communication_efficiency"] = results["efficiency"]

        return results

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.comm_stats


class GPUDirectOptimizer:
    """
    GPU Direct RDMA optimization for A100 clusters.
    """

    def __init__(self):
        """Initialize GPU Direct optimizer."""
        self.gpu_direct_available = self._check_gpu_direct()

    def _check_gpu_direct(self) -> bool:
        """Check if GPU Direct is available."""
        # Check for InfiniBand or RoCE support
        # In production, this would check actual RDMA devices
        try:
            # Check environment variables that indicate GPU Direct
            if os.environ.get("NCCL_IB_DISABLE") == "0":
                logger.info("GPU Direct RDMA detected")
                return True
        except:
            pass

        return False

    def configure_gpu_direct(self):
        """Configure GPU Direct RDMA for optimal performance."""
        if not self.gpu_direct_available:
            return

        gpu_direct_config = {
            "NCCL_IB_DISABLE": "0",  # Enable InfiniBand
            "NCCL_IB_GID_INDEX": "3",  # Use RoCEv2
            "NCCL_IB_QPS_PER_CONNECTION": "4",  # Queue pairs per connection
            "NCCL_IB_TC": "160",  # Traffic class for RoCE
            "NCCL_IB_SL": "0",  # Service level
            "NCCL_IB_CUDA_SUPPORT": "1",  # Enable CUDA support for IB
            "NCCL_NET_GDR_LEVEL": "5",  # GPU Direct RDMA level
            "NCCL_NET_GDR_READ": "1"  # Enable GPU Direct RDMA read
        }

        for key, value in gpu_direct_config.items():
            os.environ[key] = value

        logger.info("GPU Direct RDMA configured")


class TopologyAwareDataLoader:
    """
    Data loader optimized for NVLink topology.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        topology_optimizer: NVLinkOptimizer
    ):
        """
        Initialize topology-aware data loader.

        Args:
            dataloader: Original data loader
            topology_optimizer: NVLink topology optimizer
        """
        self.dataloader = dataloader
        self.topology_optimizer = topology_optimizer

    def optimize_for_nvlink(self) -> torch.utils.data.DataLoader:
        """
        Optimize data loader for NVLink bandwidth.

        Returns:
            Optimized data loader
        """
        # Adjust prefetch factor based on NVLink bandwidth
        if self.topology_optimizer.topology["nvlink_available"]:
            # Higher prefetch for NVLink's bandwidth
            self.dataloader.prefetch_factor = 4
            self.dataloader.persistent_workers = True

        # Pin memory for faster GPU transfer
        self.dataloader.pin_memory = True

        # Adjust number of workers based on topology
        num_gpus = self.topology_optimizer.topology["num_gpus"]
        self.dataloader.num_workers = min(8 * num_gpus, 64)

        return self.dataloader


def benchmark_nvlink_communication(
    tensor_sizes_mb: List[int] = [1, 10, 100, 1000]
) -> Dict[str, Any]:
    """
    Benchmark NVLink communication performance.

    Args:
        tensor_sizes_mb: List of tensor sizes to benchmark

    Returns:
        Benchmark results
    """
    results = {}

    # Initialize distributed if not already
    if not dist.is_initialized():
        logger.warning("Distributed not initialized. Skipping benchmark.")
        return results

    optimizer = NVLinkOptimizer(profile_communication=True)

    for size_mb in tensor_sizes_mb:
        # AllReduce benchmark
        allreduce_results = optimizer.profile_communication_step(size_mb, "allreduce")
        results[f"allreduce_{size_mb}mb"] = allreduce_results

        # Broadcast benchmark
        broadcast_results = optimizer.profile_communication_step(size_mb, "broadcast")
        results[f"broadcast_{size_mb}mb"] = broadcast_results

    # Summary
    results["summary"] = {
        "topology": optimizer.topology,
        "avg_bandwidth_gbps": np.mean([r["bandwidth_gbps"] for r in results.values() if isinstance(r, dict)]),
        "avg_efficiency": np.mean([r["efficiency"] for r in results.values() if isinstance(r, dict) and "efficiency" in r])
    }

    return results