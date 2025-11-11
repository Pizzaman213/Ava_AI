"""
GPU Utilities for Load Balancing and Monitoring

Helper functions for GPU load balancing, monitoring, and expert placement.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_gpu_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Args:
        device: Target device (defaults to current device)

    Returns:
        Dictionary with memory stats in MB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'free_mb': 0.0,
            'total_mb': 0.0,
            'utilization': 0.0,
        }

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.type == 'cuda' else 0

    # Get memory stats
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # MB
    free = total - allocated

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'free_mb': free,
        'total_mb': total,
        'utilization': allocated / total if total > 0 else 0.0,
    }


def get_all_gpu_memory_stats() -> Dict[int, Dict[str, float]]:
    """
    Get memory statistics for all available GPUs.

    Returns:
        Dictionary mapping GPU ID to memory stats
    """
    if not torch.cuda.is_available():
        return {}

    num_gpus = torch.cuda.device_count()
    stats = {}

    for gpu_id in range(num_gpus):
        stats[gpu_id] = get_gpu_memory_stats(gpu_id)

    return stats


def get_gpu_compute_utilization(device: Optional[torch.device] = None) -> float:
    """
    Get GPU compute utilization (0-1).

    Attempts to use pynvml for accurate metrics, falls back to nvidia-smi,
    and finally to memory utilization as a last resort.

    Args:
        device: Target device

    Returns:
        Estimated compute utilization (0-1)
    """
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.type == 'cuda' else 0

    # Try pynvml first (most accurate)
    try:
        import pynvml
        if not hasattr(get_gpu_compute_utilization, '_nvml_initialized'):
            pynvml.nvmlInit()
            get_gpu_compute_utilization._nvml_initialized = True

        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu / 100.0
    except (ImportError, Exception):
        pass

    # Try nvidia-smi as fallback
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={device}'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 100.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fallback to memory utilization as proxy
    stats = get_gpu_memory_stats(device)
    return stats['utilization']


def balance_experts_across_gpus(
    num_experts: int,
    num_gpus: int,
    strategy: str = 'even'
) -> Dict[int, List[int]]:
    """
    Compute expert distribution across GPUs.

    Args:
        num_experts: Total number of experts
        num_gpus: Number of available GPUs
        strategy: Distribution strategy
            - 'even': Distribute evenly (default)
            - 'power_of_2': Align to powers of 2 for efficiency

    Returns:
        Dictionary mapping gpu_id -> list of expert_ids
    """
    distribution: Dict[int, List[int]] = {i: [] for i in range(num_gpus)}

    if strategy == 'even':
        # Simple round-robin distribution
        for expert_id in range(num_experts):
            gpu_id = expert_id % num_gpus
            distribution[gpu_id].append(expert_id)

    elif strategy == 'power_of_2':
        # Align expert counts to powers of 2 for better kernel efficiency
        experts_per_gpu = num_experts // num_gpus
        remainder = num_experts % num_gpus

        expert_id = 0
        for gpu_id in range(num_gpus):
            count = experts_per_gpu + (1 if gpu_id < remainder else 0)
            distribution[gpu_id] = list(range(expert_id, expert_id + count))
            expert_id += count

    return distribution


def estimate_expert_memory(
    hidden_size: int,
    intermediate_size: int,
    use_lora: bool = False,
    lora_rank: int = 8,
    use_quantization: bool = False,
    quantization_bits: int = 8,
    activation: str = 'swiglu',
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Estimate memory usage for a single expert in MB.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        use_lora: Whether LoRA is used
        lora_rank: LoRA rank if used
        use_quantization: Whether quantization is used
        quantization_bits: Quantization bits if used
        activation: Activation type
        dtype: Parameter dtype

    Returns:
        Estimated memory in MB
    """
    # Bytes per parameter
    if use_quantization:
        bytes_per_param = quantization_bits / 8
    else:
        bytes_per_param = torch.finfo(dtype).bits / 8

    if use_lora:
        # LoRA: shared base (frozen) + low-rank adapters
        # Only count adapters for per-expert memory
        if activation in ['swiglu', 'geglu']:
            # gate_up: hidden -> intermediate*2
            adapter_params = hidden_size * lora_rank + lora_rank * intermediate_size * 2
        else:
            adapter_params = hidden_size * lora_rank + lora_rank * intermediate_size

        # down projection adapter
        adapter_params += intermediate_size * lora_rank + lora_rank * hidden_size

        memory_bytes = adapter_params * bytes_per_param
    else:
        # Full expert
        if activation in ['swiglu', 'geglu']:
            params = hidden_size * intermediate_size * 2  # gate_up
        else:
            params = hidden_size * intermediate_size  # up

        params += intermediate_size * hidden_size  # down
        memory_bytes = params * bytes_per_param

    # Add overhead for optimizer states, gradients, etc. (2x for Adam)
    memory_bytes *= 3  # params + grads + optimizer states

    return memory_bytes / (1024 ** 2)  # Convert to MB


def check_gpu_availability(min_memory_mb: float = 1024.0) -> List[int]:
    """
    Check which GPUs have sufficient free memory.

    Args:
        min_memory_mb: Minimum required free memory in MB

    Returns:
        List of available GPU IDs
    """
    if not torch.cuda.is_available():
        return []

    available_gpus = []
    all_stats = get_all_gpu_memory_stats()

    for gpu_id, stats in all_stats.items():
        if stats['free_mb'] >= min_memory_mb:
            available_gpus.append(gpu_id)

    return available_gpus


def sync_expert_placement_across_ranks(
    local_placement: Dict[int, int],
    world_size: int,
    rank: int,
) -> Dict[int, int]:
    """
    Synchronize expert placement across all distributed ranks.

    Args:
        local_placement: Local expert-to-GPU mapping
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        Globally synchronized placement
    """
    if not dist.is_initialized() or world_size == 1:
        return local_placement

    # Convert to tensor for all_gather
    expert_ids = sorted(local_placement.keys())
    gpu_ids = [local_placement[e] for e in expert_ids]

    # Create tensors
    num_experts = len(expert_ids)
    local_tensor = torch.tensor(
        [[e, g] for e, g in zip(expert_ids, gpu_ids)],
        dtype=torch.long
    )

    # Gather from all ranks
    gathered = [
        torch.zeros_like(local_tensor) for _ in range(world_size)
    ]

    if dist.is_initialized():
        dist.all_gather(gathered, local_tensor)

    # Merge into global placement (rank 0 wins on conflicts)
    global_placement = {}
    for rank_tensor in gathered:
        for row in rank_tensor:
            expert_id = row[0].item()
            gpu_id = row[1].item()
            if expert_id not in global_placement:
                global_placement[expert_id] = gpu_id

    return global_placement


def log_gpu_load_balance_status(
    balancer,
    logger: Optional[logging.Logger] = None
):
    """
    Log current GPU load balancing status.

    Args:
        balancer: GPULoadBalancer instance
        logger: Optional logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    metrics = balancer.get_load_balance_metrics()

    logger.info("=" * 60)
    logger.info("GPU Load Balance Status")
    logger.info("=" * 60)
    logger.info(f"Strategy: {metrics.get('balancing_strategy', 'N/A')}")
    logger.info(f"Number of GPUs: {metrics.get('num_gpus', 0)}")
    logger.info(f"Average Load Score: {metrics.get('avg_load_score', 0):.3f}")
    logger.info(f"Load Imbalance: {metrics.get('load_imbalance', 0):.3f}")
    logger.info(f"Avg Memory Utilization: {metrics.get('avg_memory_util', 0):.1%}")
    logger.info(f"Avg Compute Utilization: {metrics.get('avg_compute_util', 0):.1%}")
    logger.info(f"Experts per GPU: {metrics.get('min_experts_per_gpu', 0)}-{metrics.get('max_experts_per_gpu', 0)}")
    logger.info(f"Total Rebalances: {metrics.get('num_rebalances', 0)}")
    logger.info("=" * 60)
