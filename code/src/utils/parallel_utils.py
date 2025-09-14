"""
Distributed training utilities for MoE++ model
"""
import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl"
) -> Tuple[int, int]:
    """
    Setup distributed training environment
    
    Args:
        rank: Process rank (if None, read from environment)
        world_size: Total number of processes (if None, read from environment)
        backend: Communication backend (nccl, gloo, mpi)
    
    Returns:
        Tuple of (rank, world_size)
    """
    if torch.distributed.is_initialized():
        logger.info("Distributed already initialized")
        return dist.get_rank(), dist.get_world_size()
    
    # Get rank and world size from environment if not provided
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Single GPU case
    if world_size == 1:
        logger.info("Running in single GPU mode")
        return 0, 1
    
    # Initialize process group
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    logger.info(f"Initializing distributed: rank={rank}, world_size={world_size}")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    return rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")

def is_main_process() -> bool:
    """Check if current process is the main process"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_world_size() -> int:
    """Get world size"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce tensor across all processes"""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    dist.all_reduce(tensor, op=op)
    return tensor

def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensors from all processes"""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor.unsqueeze(0)
    
    world_size = get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.stack(tensors)

def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source to all processes"""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor

def barrier():
    """Synchronize all processes"""
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()

def print_rank_0(*args, **kwargs):
    """Print only on rank 0"""
    if is_main_process():
        print(*args, **kwargs)