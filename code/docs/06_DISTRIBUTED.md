# Distributed Training Guide

Complete guide to multi-GPU and multi-node training with Ava.

## Overview

Ava supports multiple distributed training strategies:

| Strategy | Use Case | Memory Savings | Communication |
|----------|----------|----------------|---------------|
| DDP | Multi-GPU, fits in memory | None | Gradient sync |
| FSDP | Large models | High | Parameter sharding |
| DeepSpeed ZeRO-1 | Optimizer memory | Moderate | Optimizer sharding |
| DeepSpeed ZeRO-2 | Gradient + Optimizer | High | Gradient sharding |
| DeepSpeed ZeRO-3 | Full sharding | Maximum | All parameters |

## Quick Start

### Multi-GPU with DDP

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 \
    code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

### Multi-GPU with DeepSpeed

```bash
torchrun --nproc_per_node=4 \
    code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml \
    --use-deepspeed \
    --zero-stage 2
```

### Multi-Node Training

```bash
# On node 0 (master)
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="node0.example.com" \
    --master_port=29500 \
    code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml

# On node 1
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="node0.example.com" \
    --master_port=29500 \
    code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

## Distributed Data Parallel (DDP)

Basic multi-GPU strategy where each GPU has a full model copy.

### Configuration

```yaml
# code/configs/moe/large_Multy.yaml
hardware:
  num_gpus: 4

training:
  batch_size: 32              # Per-GPU batch size
  gradient_accumulation_steps: 2
```

### How It Works

1. Each GPU maintains a full model replica
2. Data is sharded across GPUs
3. Gradients are synchronized after backward pass
4. Effective batch size = batch_size × num_gpus × gradient_accumulation

### Best Practices

- Use `persistent_workers: true` to avoid worker restart overhead
- Set `batch_size` to fit single GPU memory
- Use gradient accumulation for larger effective batch sizes

## DeepSpeed Integration

### ZeRO Stage 1: Optimizer State Partitioning

Partitions optimizer states across GPUs.

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 1
  precision_type: 'bf16'
```

**Memory savings**: ~4x optimizer memory reduction

### ZeRO Stage 2: Gradient Partitioning

Adds gradient partitioning to ZeRO-1.

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2
  precision_type: 'bf16'
```

**Memory savings**: ~8x optimizer + gradient memory reduction

### ZeRO Stage 3: Parameter Partitioning

Full model sharding across GPUs.

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 3
  cpu_offload: true           # Recommended for ZeRO-3
  precision_type: 'bf16'
```

**Memory savings**: Linear scaling with GPU count

### DeepSpeed Configuration Files

Pre-configured DeepSpeed configs in `code/configs/distributed/`:

```yaml
# code/configs/distributed/deepspeed_zero2.yaml
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0
}
```

Use with:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --use-deepspeed \
    --deepspeed-config code/configs/distributed/deepspeed_zero2.yaml
```

### CPU Offloading

For very large models:

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 3
  cpu_offload: true
  nvme_offload: false         # Enable for extreme sizes

  # Communication settings
  zero_reduce_bucket_size: 500000000
  zero_allgather_bucket_size: 500000000
```

## Expert Parallelism

Distribute MoE experts across GPUs.

### Configuration

```yaml
model:
  num_experts: 8
  expert_parallel_size: 4     # Experts distributed over 4 GPUs

hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'adaptive'
```

### How It Works

```
GPU 0: Experts 0, 1
GPU 1: Experts 2, 3
GPU 2: Experts 4, 5
GPU 3: Experts 6, 7
```

Tokens are routed to the GPU holding their selected expert.

### Load Balancing

```yaml
hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'memory_aware'  # Consider GPU memory
  rebalance_interval: 1000            # Check every 1000 steps
  enable_expert_migration: true       # Move experts if imbalanced
  migration_threshold: 0.2            # 20% imbalance triggers migration
```

## Data Parallelism

### Distributed Data Loading

The data pipeline automatically shards data across workers:

```python
# code/src/ava/data/distributed.py
class DistributedStreamingDataset:
    def __init__(self, data_dir, rank, world_size, ...):
        # Shard files across ranks
        all_files = glob.glob(f"{data_dir}/*.arrow")
        self.files = all_files[rank::world_size]
```

### Configuration

```yaml
data:
  num_workers: 4              # Per-GPU workers
  prefetch_factor: 2
  persistent_workers: true
  dataloader_pin_memory: true
```

## Gradient Synchronization

### Gradient Accumulation with DDP

```yaml
training:
  gradient_accumulation_steps: 4

# Gradients synced only on accumulation boundary
# Reduces communication overhead
```

### No-Sync Mode

For debugging or special cases:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --no-sync
```

## Checkpointing in Distributed Training

### Saving

Only rank 0 saves checkpoints:

```python
if rank == 0:
    torch.save(checkpoint, path)
torch.distributed.barrier()  # Ensure all ranks wait
```

### Loading

All ranks load the same checkpoint:

```python
torch.distributed.barrier()  # Ensure checkpoint exists
checkpoint = torch.load(path, map_location=f'cuda:{rank}')
model.load_state_dict(checkpoint['model_state_dict'])
```

### DeepSpeed Checkpointing

DeepSpeed handles checkpointing automatically:

```python
# Saves sharded checkpoint
model_engine.save_checkpoint(output_dir)

# Loads and reconstructs
model_engine.load_checkpoint(checkpoint_dir)
```

## Memory Optimization for Large Models

### Strategy Selection Guide

| Model Size | GPUs | Strategy |
|------------|------|----------|
| <1B | 1-2 | DDP |
| 1-10B | 2-8 | DeepSpeed ZeRO-2 |
| 10-100B | 8+ | DeepSpeed ZeRO-3 |
| >100B | 16+ | ZeRO-3 + CPU/NVMe |

### Memory Calculation

```
Per-GPU Memory ≈ Model + Optimizer + Gradients + Activations

With ZeRO-3:
Per-GPU Memory ≈ Model/N + Optimizer/N + Gradients/N + Activations
                 (N = number of GPUs)
```

### Activation Checkpointing

```yaml
deepspeed:
  activation_checkpointing: true
  partition_activations: true   # For ZeRO-3
  cpu_checkpointing: true       # Offload to CPU
```

## Communication Optimization

### NCCL Settings

```bash
# Optimize NCCL for your network
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_IB_GID_INDEX=3        # Set GID for IB
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
export NCCL_DEBUG=INFO            # Debug output
```

### Bucket Sizes

```yaml
deepspeed:
  zero_reduce_bucket_size: 500000000      # 500MB
  zero_allgather_bucket_size: 500000000   # 500MB
  overlap_comm: true                       # Overlap with compute
```

## Troubleshooting

### NCCL Timeout

```bash
# Increase timeout
export NCCL_TIMEOUT=1800  # 30 minutes
```

### Out of Memory

1. Reduce per-GPU batch size
2. Increase gradient accumulation
3. Use ZeRO-3 with CPU offload
4. Enable activation checkpointing

### Slow Training

1. Check network bandwidth (use `ib_read_bw`)
2. Enable communication overlap
3. Increase bucket sizes for fewer, larger transfers

### Hanging at Barrier

1. Check all processes started correctly
2. Verify master address accessible from all nodes
3. Check firewall rules

## Example: Large-Scale Training

```yaml
# Multi-node, multi-GPU configuration
hardware:
  num_gpus: 8

model:
  num_experts: 16
  expert_parallel_size: 8
  gradient_checkpointing: true

training:
  batch_size: 16              # Small per-GPU batch
  gradient_accumulation_steps: 8

deepspeed:
  use_deepspeed: true
  zero_stage: 3
  cpu_offload: true
  activation_checkpointing: true
  precision_type: 'bf16'

  # Communication optimization
  overlap_comm: true
  zero_reduce_bucket_size: 1000000000
  zero_allgather_bucket_size: 1000000000
```

Launch command:

```bash
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    code/scripts/5_training/train_pipeline.py \
    --config code/configs/distributed/large_scale.yaml
```

## Next Steps

- [Training Guide](./03_TRAINING_GUIDE.md) - Training basics
- [Performance Tuning](./10_PERFORMANCE.md) - Optimization
- [Troubleshooting](./09_TROUBLESHOOTING.md) - Common issues
