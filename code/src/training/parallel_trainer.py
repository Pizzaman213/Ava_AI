"""
Parallel Training with MTP/DDP Coordination
Advanced distributed training strategies
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """Configuration for parallel training"""
    # Basic settings
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env"  # env, tcp, file
    
    # Parallelism dimensions
    data_parallel_size: int = 1
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    # FSDP settings
    use_fsdp: bool = False
    fsdp_cpu_offload: bool = True
    fsdp_backward_prefetch: bool = True
    fsdp_mixed_precision: bool = True
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    
    # Communication optimization
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    # Performance
    enable_async_collective: bool = True
    enable_collective_determinism: bool = False

class ProcessGroupManager:
    """Manages process groups for different parallelism types"""
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Process groups
        self.world_group = None
        self.dp_group = None
        self.mp_group = None
        self.pp_group = None
        self.ep_group = None
        
        # Ranks
        self.dp_rank = 0
        self.mp_rank = 0
        self.pp_rank = 0
        self.ep_rank = 0
        
        if dist.is_initialized():
            self._create_process_groups()
    
    def _create_process_groups(self):
        """Create process groups for different parallelism dimensions"""
        # Validate configuration
        total_size = (self.config.data_parallel_size * 
                     self.config.model_parallel_size * 
                     self.config.pipeline_parallel_size * 
                     self.config.expert_parallel_size)
        
        if total_size != self.world_size:
            raise ValueError(
                f"Product of parallel sizes ({total_size}) != world_size ({self.world_size})"
            )
        
        # Calculate strides
        ep_stride = 1
        pp_stride = ep_stride * self.config.expert_parallel_size
        mp_stride = pp_stride * self.config.pipeline_parallel_size
        dp_stride = mp_stride * self.config.model_parallel_size
        
        # Create process groups
        logger.info("Creating process groups...")
        
        # Data parallel groups
        for i in range(self.config.model_parallel_size * 
                      self.config.pipeline_parallel_size * 
                      self.config.expert_parallel_size):
            ranks = [i + j * dp_stride for j in range(self.config.data_parallel_size)]
            group = dist.new_group(ranks)
            
            if self.rank in ranks:
                self.dp_group = group
                self.dp_rank = ranks.index(self.rank)
        
        # Model parallel groups
        for i in range(self.config.data_parallel_size * 
                      self.config.pipeline_parallel_size * 
                      self.config.expert_parallel_size):
            start = (i // mp_stride) * mp_stride + i % pp_stride
            ranks = [start + j * pp_stride for j in range(self.config.model_parallel_size)]
            group = dist.new_group(ranks)
            
            if self.rank in ranks:
                self.mp_group = group
                self.mp_rank = ranks.index(self.rank)
        
        # Pipeline parallel groups
        for i in range(self.config.data_parallel_size * 
                      self.config.model_parallel_size * 
                      self.config.expert_parallel_size):
            start = (i // pp_stride) * pp_stride + i % ep_stride
            ranks = [start + j * ep_stride for j in range(self.config.pipeline_parallel_size)]
            group = dist.new_group(ranks)
            
            if self.rank in ranks:
                self.pp_group = group
                self.pp_rank = ranks.index(self.rank)
        
        # Expert parallel groups
        for i in range(self.config.data_parallel_size * 
                      self.config.model_parallel_size * 
                      self.config.pipeline_parallel_size):
            start = i * ep_stride
            ranks = [start + j for j in range(self.config.expert_parallel_size)]
            group = dist.new_group(ranks)
            
            if self.rank in ranks:
                self.ep_group = group
                self.ep_rank = ranks.index(self.rank)
        
        logger.info(f"Process groups created - DP rank: {self.dp_rank}, "
                   f"MP rank: {self.mp_rank}, PP rank: {self.pp_rank}, "
                   f"EP rank: {self.ep_rank}")

class ParallelTrainer:
    """Advanced parallel trainer with multiple parallelism strategies"""
    def __init__(
        self,
        model: nn.Module,
        config: ParallelConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize process groups
        self.pg_manager = ProcessGroupManager(config)
        
        # Wrap model based on configuration
        self.wrapped_model = self._wrap_model()
        
        # Performance monitoring
        self.communication_time = 0.0
        self.computation_time = 0.0
    
    def _wrap_model(self) -> nn.Module:
        """Wrap model with appropriate parallelism strategy"""
        model = self.model
        
        # Apply model parallelism first (if needed)
        if self.config.model_parallel_size > 1:
            model = self._apply_model_parallelism(model)
        
        # Apply expert parallelism
        if self.config.expert_parallel_size > 1:
            model = self._apply_expert_parallelism(model)
        
        # Apply pipeline parallelism
        if self.config.pipeline_parallel_size > 1:
            model = self._apply_pipeline_parallelism(model)
        
        # Apply data parallelism (DDP or FSDP)
        if self.config.data_parallel_size > 1:
            if self.config.use_fsdp:
                model = self._wrap_with_fsdp(model)
            else:
                model = self._wrap_with_ddp(model)
        
        return model
    
    def _apply_model_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor model parallelism"""
        # This would integrate with MTP implementation
        # For now, return model as-is
        logger.info(f"Model parallelism with size {self.config.model_parallel_size}")
        return model
    
    def _apply_expert_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply expert parallelism"""
        # This would modify expert layers for distributed experts
        logger.info(f"Expert parallelism with size {self.config.expert_parallel_size}")
        return model
    
    def _apply_pipeline_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply pipeline parallelism"""
        # This would split model into pipeline stages
        logger.info(f"Pipeline parallelism with size {self.config.pipeline_parallel_size}")
        return model
    
    def _wrap_with_ddp(self, model: nn.Module) -> DDP:
        """Wrap model with DistributedDataParallel"""
        logger.info("Wrapping model with DDP")
        
        ddp_model = DDP(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            process_group=self.pg_manager.dp_group,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            bucket_cap_mb=self.config.bucket_cap_mb,
        )
        
        return ddp_model
    
    def _wrap_with_fsdp(self, model: nn.Module) -> FSDP:
        """Wrap model with FullyShardedDataParallel"""
        logger.info("Wrapping model with FSDP")
        
        # Configure FSDP
        cpu_offload = CPUOffload(offload_params=self.config.fsdp_cpu_offload)
        
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE if self.config.fsdp_backward_prefetch else None
        
        mixed_precision = None
        if self.config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        
        # Wrap model
        fsdp_model = FSDP(
            model,
            process_group=self.pg_manager.dp_group,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            sharding_strategy=self.config.fsdp_sharding_strategy,
            device_id=self.device,
        )
        
        return fsdp_model
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Execute single training step with timing"""
        # Forward pass
        comp_start = time.time()
        
        if grad_scaler:
            with torch.cuda.amp.autocast():
                outputs = self.wrapped_model(**batch)
                loss = outputs["loss"]
        else:
            outputs = self.wrapped_model(**batch)
            loss = outputs["loss"]
        
        # Add auxiliary losses
        if "aux_loss" in outputs:
            loss = loss + outputs["aux_loss"]
        
        comp_time = time.time() - comp_start
        
        # Backward pass
        comm_start = time.time()
        
        if grad_scaler:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        comm_time = time.time() - comm_start
        
        # Update timings
        self.computation_time += comp_time
        self.communication_time += comm_time
        
        # Optimizer step
        if grad_scaler:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Gather metrics
        metrics = {
            "loss": loss.item(),
            "computation_time": comp_time,
            "communication_time": comm_time,
        }
        
        if "aux_loss" in outputs:
            metrics["aux_loss"] = outputs["aux_loss"].item()
        
        return metrics
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce metrics across data parallel group"""
        if self.config.data_parallel_size == 1:
            return metrics
        
        reduced_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, group=self.pg_manager.dp_group)
            reduced_metrics[key] = tensor.item() / self.config.data_parallel_size
        
        return reduced_metrics
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ):
        """Save distributed checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Only save on rank 0 of each model parallel group
        if self.pg_manager.dp_rank == 0:
            # Get model state dict
            if isinstance(self.wrapped_model, (DDP, FSDP)):
                state_dict = self.wrapped_model.module.state_dict()
            else:
                state_dict = self.wrapped_model.state_dict()
            
            # Add model parallel rank to filename if using model parallelism
            if self.config.model_parallel_size > 1:
                model_file = checkpoint_dir / f"model_mp{self.pg_manager.mp_rank}.pt"
            else:
                model_file = checkpoint_dir / "model.pt"
            
            torch.save(state_dict, model_file)
            
            # Save optimizer and training state on global rank 0
            if self.pg_manager.rank == 0:
                training_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": self.config,
                }
                
                if scheduler:
                    training_state["scheduler_state_dict"] = scheduler.state_dict()
                
                torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Synchronize
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load distributed checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model state
        if self.config.model_parallel_size > 1:
            model_file = checkpoint_dir / f"model_mp{self.pg_manager.mp_rank}.pt"
        else:
            model_file = checkpoint_dir / "model.pt"
        
        state_dict = torch.load(model_file, map_location=self.device)
        
        if isinstance(self.wrapped_model, (DDP, FSDP)):
            self.wrapped_model.module.load_state_dict(state_dict)
        else:
            self.wrapped_model.load_state_dict(state_dict)
        
        # Load training state
        training_state_file = checkpoint_dir / "training_state.pt"
        if training_state_file.exists():
            training_state = torch.load(training_state_file, map_location=self.device)
            
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            
            if scheduler and "scheduler_state_dict" in training_state:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
            
            return training_state
        
        return {}

def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env",
    world_size: int = -1,
    rank: int = -1,
) -> Tuple[int, int]:
    """
    Setup distributed training environment
    
    Args:
        backend: Backend to use (nccl, gloo, mpi)
        init_method: Initialization method (env, tcp, file)
        world_size: Total number of processes
        rank: Rank of current process
        
    Returns:
        Tuple of (rank, world_size)
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    
    if init_method == "env":
        # Get from environment variables
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        elif "SLURM_PROCID" in os.environ:
            # SLURM environment
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
        else:
            # Single process
            rank = 0
            world_size = 1
    
    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method if init_method != "env" else None,
            world_size=world_size,
            rank=rank,
        )
        
        # Set device
        if torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
        
        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")
    
    return rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def all_gather_object(obj: Any, world_size: int) -> List[Any]:
    """Gather objects from all ranks"""
    if not dist.is_initialized() or world_size == 1:
        return [obj]
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    
    return gathered

def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source rank"""
    if not dist.is_initialized():
        return obj
    
    objects = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    
    return objects[0]