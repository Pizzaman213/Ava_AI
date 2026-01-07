# Configuration Reference

Complete reference for all 110+ configuration parameters in Ava.

## Configuration System

Ava uses YAML configuration files with hierarchical structure:

```yaml
# Example: code/configs/moe/large.yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  ...

training:
  batch_size: 128
  learning_rate: 0.0006
  ...

data:
  data_dir: "code/data/processed"
  ...
```

## Loading Configuration

```python
from ava.config.training_config import TrainingConfigManager

manager = TrainingConfigManager()
config = manager.load_yaml_config("code/configs/moe/large.yaml")

# Access via dot notation
batch_size = config.training.batch_size
hidden_size = config.model.hidden_size
```

---

## Model Configuration

### Core Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 50680 | Vocabulary size |
| `hidden_size` | int | 1024 | Hidden dimension |
| `num_layers` | int | 6 | Number of transformer layers |
| `num_attention_heads` | int | 16 | Attention heads |
| `intermediate_size` | int | 8192 | FFN intermediate size |
| `max_position_embeddings` | int | 512 | Maximum sequence length |

### MoE Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | int | 4 | Total number of experts |
| `num_experts_per_token` | int | 1 | Experts active per token |
| `router_type` | str | 'mixtral' | Router: 'mixtral', 'deepseek', 'switch' |
| `capacity_factor` | float | 1.25 | Expert capacity buffer |
| `expert_dropout` | float | 0.0 | Expert dropout rate |

### Auxiliary Losses

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `router_z_loss_coef` | float | 0.0001 | Router logit regularization |
| `load_balance_loss_coef` | float | 0.01 | Load balancing loss weight |
| `diversity_loss_coef` | float | 0.0001 | Routing diversity loss |
| `router_jitter_noise` | float | 0.01 | Exploration noise |

### Activations & Attention

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activation` | str | 'swiglu' | Activation: 'swiglu', 'geglu', 'gelu', 'relu' |
| `use_flash_attention` | bool | true | Enable Flash Attention |
| `use_alibi` | bool | false | ALiBi positional encoding |
| `rope_theta` | float | 10000.0 | RoPE base theta |

### Performance Optimizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_checkpointing` | bool | true | Enable gradient checkpointing |
| `use_grouped_gemm` | bool | true | Grouped GEMM for experts |
| `use_triton_kernels` | bool | true | Custom Triton kernels |
| `use_torch_compile` | bool | false | Enable torch.compile |
| `quantize_kv_cache` | bool | false | Quantize attention cache |

### Regularization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dropout` | float | 0.0 | General dropout |
| `attention_dropout` | float | 0.0 | Attention dropout |
| `layer_norm_eps` | float | 1e-5 | LayerNorm epsilon |
| `initializer_range` | float | 0.01 | Weight init std |

---

## Training Configuration

### Core Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | None | Training batch size |
| `epochs` | int | None | Number of epochs |
| `learning_rate` | float | None | Learning rate |
| `warmup_steps` | int | 2000 | LR warmup steps |
| `max_steps` | int | None | Max training steps |

### Gradient Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation |
| `max_gradient_norm` | float | 1.0 | Gradient clipping threshold |

### Adaptive LR

```yaml
training:
  adaptive_lr:
    scheduler: 'cosine'      # 'cosine', 'linear', 'constant'
    min_lr_ratio: 0.1        # Min LR as ratio of base LR
```

---

## Data Configuration

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | auto | Data directory path |
| `max_length` | int | 512 | Maximum sequence length |
| `tokenizer_name` | str | None | Tokenizer path or name |
| `max_samples` | int | None | Limit samples (testing) |

### Streaming Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming` | bool | false | Enable streaming loader |
| `buffer_size` | int | 50000 | Shuffle buffer size |
| `num_workers` | int | 0 | Data loading workers |
| `prefetch_factor` | int | 4 | Batches to prefetch |
| `persistent_workers` | bool | false | Keep workers alive |

### Sequence Packing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sequence_packing` | bool | false | Enable packing |
| `packing_strategy` | str | 'greedy' | 'greedy' or 'adaptive' |
| `use_dynamic_batching` | bool | false | Dynamic batch sizing |

### Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_split` | str | 'train' | Training split name |
| `eval_split` | str | 'validation' | Validation split name |
| `auto_create_validation_split` | bool | true | Auto-create val split |
| `validation_split_ratio` | float | 0.1 | Val split ratio |

### Randomization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shuffle_seed` | int | None | Shuffle seed (None=random) |
| `enable_length_sorting` | bool | true | Sort by length |
| `enable_bucketing` | bool | true | Length bucketing |

---

## Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | 'cuda' | Device: 'cuda', 'cpu', 'mps' |
| `mixed_precision` | str | 'fp32' | 'fp32', 'fp16', 'bf16' |
| `compile` | bool | false | Enable torch.compile |
| `num_gpus` | int | 1 | Number of GPUs |

### GPU Load Balancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gpu_load_balancing` | bool | false | Enable load balancing |
| `balancing_strategy` | str | 'adaptive' | 'round_robin', 'memory_aware', 'adaptive' |
| `rebalance_interval` | int | 1000 | Steps between rebalancing |
| `enable_expert_migration` | bool | true | Allow expert migration |
| `migration_threshold` | float | 0.2 | Imbalance threshold |

---

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | auto | Output directory |
| `save_every` | int | 100 | Checkpoint frequency |
| `resume` | str | None | Resume checkpoint path |
| `fresh_start` | bool | false | Ignore existing checkpoints |

---

## Weights & Biases Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | bool | false | Enable WandB |
| `wandb_offline` | bool | false | Force offline mode |
| `wandb_project` | str | 'Ava' | Project name |
| `wandb_name` | str | None | Run name |
| `wandb_tags` | list | ['moe', 'training'] | Run tags |
| `wandb_log_freq` | int | 10 | Logging frequency |

---

## DeepSpeed Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_deepspeed` | bool | false | Enable DeepSpeed |
| `config_file` | str | None | DeepSpeed JSON config |
| `zero_stage` | int | 2 | ZeRO stage (0-3) |
| `cpu_offload` | bool | false | CPU offloading |
| `nvme_offload` | bool | false | NVMe offloading |
| `precision_type` | str | 'fp16' | 'fp16', 'bf16', 'fp32' |

---

## Performance Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ultra_fast_mode` | bool | false | Disable all logging |
| `fast_progress` | bool | false | Enhanced progress bar |
| `express_mode` | bool | false | Optimized async logging |
| `enable_tf32` | bool | true | TF32 on Ampere+ |
| `enable_cudnn_benchmark` | bool | true | cuDNN auto-tuning |

---

## MoE Memory Optimization

### Expert Offloading

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_expert_offloading` | bool | false | CPU offloading |
| `max_active_experts_gpu` | int | 4 | Max experts on GPU |
| `offload_eviction_policy` | str | 'lru' | 'lru', 'frequency', 'hybrid' |
| `offload_prefetch_lookahead` | int | 2 | Prefetch ahead |

### LoRA Experts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora_experts` | bool | false | Enable LoRA experts |
| `lora_rank` | int | 8 | LoRA rank |
| `lora_alpha` | int | 16 | LoRA alpha |
| `freeze_lora_base` | bool | false | Freeze base weights |

### Expert Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_expert_quantization` | bool | false | Enable quantization |
| `expert_quantization_bits` | int | 8 | Bits (8 or 4) |
| `quantize_inactive_experts` | bool | true | Only inactive experts |

---

## Coherence Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable coherence eval |
| `eval_every_n_steps` | int | 500 | Evaluation frequency |
| `num_samples` | int | 10 | Samples to generate |
| `max_generation_length` | int | 256 | Max generation length |
| `temperature` | float | 0.8 | Generation temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `top_k` | int | 50 | Top-k sampling |

---

## Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbosity` | str | 'info' | Log level |
| `metrics_log_freq` | int | 500 | Metrics frequency |
| `memory_check_freq` | int | 2000 | Memory check frequency |
| `health_summary_freq` | int | 500 | Health summary frequency |
| `enable_timing_breakdown` | bool | true | Step timing logs |
| `enable_memory_profiling` | bool | true | Memory profiling |

---

## Progressive Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_progressive_training` | bool | false | Enable progressive |
| `enable_sequence_scaling` | bool | false | Sequence length scaling |
| `initial_seq_length` | int | 128 | Starting length |
| `final_seq_length` | int | 2048 | Final length |
| `length_schedule` | str | 'linear' | Scaling schedule |
| `enable_curriculum` | bool | false | Curriculum learning |

---

## Dynamic Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable dynamic batching |
| `min_batch_size` | int | 16 | Minimum batch size |
| `max_batch_size` | int | 256 | Maximum batch size |
| `target_memory` | float | 0.75 | Target memory usage |
| `high_memory` | float | 0.85 | Start decreasing |
| `critical_memory` | float | 0.92 | Emergency threshold |

---

## FP8 Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable FP8 |
| `use_transformer_engine` | bool | false | Use TE library |
| `format` | str | 'e4m3' | 'e4m3' or 'e5m2' |
| `margin` | int | 0 | Scale margin |

---

## Example Configurations

### Development/Testing

```yaml
# code/configs/moe/minimal_working.yaml
model:
  vocab_size: 50680
  hidden_size: 512
  num_layers: 6
  num_experts: 4
  num_experts_per_token: 1
  gradient_checkpointing: true

training:
  batch_size: 32
  learning_rate: 0.001
  warmup_steps: 100
  epochs: 1

data:
  max_length: 256
  max_samples: 10000
```

### Production Training

```yaml
# code/configs/moe/large.yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_experts: 8
  num_experts_per_token: 2
  use_flash_attention: true
  gradient_checkpointing: true

training:
  batch_size: 128
  learning_rate: 0.0006
  warmup_steps: 2000
  gradient_accumulation_steps: 4
  max_gradient_norm: 1.0

hardware:
  mixed_precision: 'bf16'

data:
  max_length: 512
  num_workers: 8
  use_sequence_packing: true
```

### Multi-GPU Training

```yaml
# code/configs/moe/large_Multy.yaml
model:
  # ... same as large.yaml

training:
  batch_size: 32         # Per-GPU batch size
  gradient_accumulation_steps: 2

hardware:
  num_gpus: 4
  use_gpu_load_balancing: true

deepspeed:
  use_deepspeed: true
  zero_stage: 2
  precision_type: 'bf16'
```

## Configuration Validation

The config system validates settings at load time:

```python
# Automatic validation
config = manager.load_yaml_config("config.yaml")
errors = manager.validate_dynamic_config(config)

if errors:
    for error in errors:
        print(f"Warning: {error}")
```

## Command-Line Overrides

Override YAML settings from command line:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --epochs 10
```
