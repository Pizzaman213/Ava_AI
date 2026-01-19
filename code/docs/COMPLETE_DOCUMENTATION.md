# Ava LLM Training Framework - Complete Documentation

Welcome to the complete Ava documentation. This comprehensive guide covers everything from getting started to advanced optimization techniques.

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Getting Started](#2-getting-started)
3. [Architecture](#3-architecture)
4. [Configuration Reference](#4-configuration-reference)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Guide](#6-training-guide)
7. [Distributed Training](#7-distributed-training)
8. [RLHF Training](#8-rlhf-training)
9. [API Reference](#9-api-reference)
10. [Performance Tuning](#10-performance-tuning)
11. [Troubleshooting](#11-troubleshooting)

---

# 1. Framework Overview

Ava is an advanced LLM training framework featuring:

- **MoE++ Architecture**: 8-32 experts with Mixtral/DeepSeek/Switch routing
- **Memory Efficiency**: Gradient checkpointing, expert offloading, FP8 support
- **Scalability**: Single GPU to multi-node distributed training
- **Production Ready**: WandB integration, checkpointing, evaluation metrics

## Quick Start

```bash
# Single GPU training
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU training
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

## Documentation Conventions

- **Code blocks** show exact commands to run
- **Configuration examples** use YAML format
- **File paths** are relative to `/root/Ava_AI/`
- **Line references** use `file.py:123` format

---

# 2. Getting Started

This section walks you through setting up and running your first training job with Ava.

## 2.1 Prerequisites

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB+ recommended)
- **CUDA**: 11.8+

## 2.2 Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Ava_AI
```

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: DeepSpeed for distributed training
pip install deepspeed

# Optional: Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

### Verify Installation

```bash
python code/scripts/check_dependencies.py
```

## 2.3 Project Structure

```
/root/Ava_AI/
├── code/
│   ├── src/ava/           # Core framework
│   ├── scripts/           # Training scripts
│   ├── configs/           # Configuration files
│   └── outputs/           # Training outputs
├── data/                  # Training data
└── models/                # Model checkpoints
```

## 2.4 Your First Training Run

### Step 1: Prepare Data

Download and preprocess training data:

```bash
python code/scripts/1_data_download/unified_download.py
```

This downloads datasets and creates pre-tokenized Arrow files in `code/data/`.

### Step 2: Choose a Configuration

| Config | Parameters | GPU Memory | Use Case |
|--------|------------|------------|----------|
| `minimal_working.yaml` | 62M | 8GB | Testing/Development |
| `large.yaml` | 200M+ | 24GB | Production |
| `large_Multy.yaml` | 200M+ | 24GB x N | Multi-GPU |

### Step 3: Start Training

```bash
# Development/testing (smaller model)
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/minimal_working.yaml

# Production training
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml
```

### Step 4: Monitor Progress

Training logs appear in the terminal. For detailed monitoring:

```bash
# Enable Weights & Biases
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --use-wandb
```

View runs at [wandb.ai](https://wandb.ai).

## 2.5 Understanding the Output

Training creates a run directory:

```
code/outputs/pretraining/ava_training_YYYYMMDD_HHMMSS_ID/
├── checkpoints/           # Model checkpoints
│   ├── checkpoint_step_1000.pt
│   └── best_model.pt
├── logs/                  # Training logs
├── metrics/               # JSON metrics
└── config.yaml            # Run configuration
```

## 2.6 Key Concepts

### Mixture of Experts (MoE)

Ava uses sparse MoE architecture where each token is processed by only `k` experts out of `N` total:

```yaml
model:
  num_experts: 8          # Total experts
  num_experts_per_token: 2  # Active per token
  router_type: 'mixtral'  # Routing algorithm
```

### Training Pipeline

The pipeline manages components via lifecycle hooks:

```
Initialize → Epoch Loop → Step Loop → Cleanup
                ↓
    [DataLoader → Forward → Loss → Backward → Optimizer]
```

### Gradient Checkpointing

Reduces memory by recomputing activations during backward pass:

```yaml
model:
  gradient_checkpointing: true
```

## 2.7 Quick Reference

```bash
# Training commands
python code/scripts/5_training/train_pipeline.py --config <config.yaml>
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config <config.yaml>

# Fine-tuning
python code/scripts/5_training/finetune.py --checkpoint <path>

# Generation
python code/scripts/7_generation/generate.py --prompt "Once upon a time"

# RLHF
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config <config.yaml>
```

---

# 3. Architecture

This section provides a deep dive into the Ava MoE++ architecture, covering model components, routing strategies, and design decisions.

## 3.1 Architecture Overview

Ava implements a **Mixture of Experts Plus Plus (MoE++)** architecture that combines:

- Sparse expert routing for compute efficiency
- Flash Attention for memory-efficient attention
- Rotary Position Embeddings (RoPE) for position encoding
- SwiGLU/GeGLU gated activations

```
Input Tokens
     ↓
Token Embeddings + RoPE
     ↓
┌─────────────────────────────────────┐
│         Transformer Block x N        │
│  ┌─────────────────────────────────┐ │
│  │   Multi-Head Attention (MHA)    │ │
│  │   with Flash Attention          │ │
│  └─────────────────────────────────┘ │
│              ↓                       │
│  ┌─────────────────────────────────┐ │
│  │      Sparse MoE Layer           │ │
│  │  Router → Top-K Experts → Merge │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
     ↓
Output Logits
```

## 3.2 Core Components

### Token Embeddings

Located in `code/src/ava/models/moe.py`:

```python
class EnhancedMoEModel:
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = RotaryEmbedding(config.hidden_size // config.num_attention_heads)
```

Configuration:
```yaml
model:
  vocab_size: 50680       # Vocabulary size
  hidden_size: 1024       # Embedding dimension
  max_position_embeddings: 512
```

### Attention Mechanism

Supports multiple attention variants:

| Type | Description | Config |
|------|-------------|--------|
| MHA | Multi-Head Attention | Default |
| MQA | Multi-Query Attention | `num_kv_heads: 1` |
| GQA | Grouped-Query Attention | `num_kv_heads: 4` |

Flash Attention integration:

```yaml
model:
  use_flash_attention: true  # Requires flash-attn package
```

Implementation in `code/src/ava/models/moe.py:EnhancedMoEModel`:

```python
# Flash Attention path
if self.use_flash_attention and flash_attn_available:
    attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout)
else:
    # Standard attention fallback
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

### Sparse MoE Layer

The core innovation: each token is routed to only `k` experts.

Located in `code/src/ava/models/moe_layer.py`:

```python
class SparseMoELayer:
    def forward(self, hidden_states):
        # 1. Compute routing probabilities
        router_logits = self.router(hidden_states)

        # 2. Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token
        )

        # 3. Process through selected experts
        expert_outputs = self.dispatch_to_experts(
            hidden_states, selected_experts, routing_weights
        )

        return expert_outputs
```

### Router Architectures

Three routing strategies available:

#### Mixtral Router
Standard learned gating with load balancing:

```python
class MixtralRouter(nn.Module):
    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        weights = F.softmax(logits, dim=-1)
        return weights
```

#### DeepSeek Router
Hybrid routing with shared experts:

```python
class DeepSeekRouter(nn.Module):
    def forward(self, hidden_states):
        # Some experts always active (shared)
        # Others selected via routing
        shared_output = self.shared_experts(hidden_states)
        routed_output = self.route_to_experts(hidden_states)
        return shared_output + routed_output
```

#### Switch Router
Simplified single-expert routing:

```python
class SwitchRouter(nn.Module):
    def forward(self, hidden_states):
        # Route each token to exactly one expert
        logits = self.gate(hidden_states)
        expert_idx = logits.argmax(dim=-1)
        return expert_idx
```

Configuration:

```yaml
model:
  router_type: 'mixtral'  # 'mixtral', 'deepseek', or 'switch'
  num_experts: 8
  num_experts_per_token: 2
  capacity_factor: 1.25   # Expert capacity buffer
```

### Expert Networks

High-performance FFN experts with gated activations:

```python
class HighPerformanceExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation='swiglu'):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = get_activation(activation)

    def forward(self, x):
        # SwiGLU: gate * up, then project down
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### Auxiliary Losses

MoE training requires auxiliary losses for load balancing:

```yaml
model:
  load_balance_loss_coef: 0.01    # Expert load balancing
  router_z_loss_coef: 0.0001      # Router logit regularization
  diversity_loss_coef: 0.0001     # Encourage diverse routing
```

**Load Balance Loss**: Penalizes uneven expert utilization
```python
# Encourages uniform expert selection across batch
balance_loss = num_experts * (expert_fraction * router_prob_fraction).sum()
```

**Router Z-Loss**: Prevents router logits from growing too large
```python
z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
```

## 3.3 Model Configuration

### EnhancedMoEConfig

Located in `code/src/ava/models/moe.py`:

```python
@dataclass
class EnhancedMoEConfig:
    # Architecture
    vocab_size: int = 50680
    hidden_size: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    router_type: str = 'mixtral'
    capacity_factor: float = 1.25

    # Optimizations
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    activation: str = 'swiglu'
```

### Parameter Counts

| Component | Formula | Example (1024 hidden) |
|-----------|---------|----------------------|
| Embeddings | vocab × hidden | 50M |
| Attention | 4 × hidden² × layers | 67M |
| Experts | 3 × hidden × intermediate × experts | 100M+ |
| Router | hidden × experts × layers | 0.5M |

## 3.4 Memory Optimizations

### Gradient Checkpointing

Recomputes activations during backward pass:

```yaml
model:
  gradient_checkpointing: true
```

Memory reduction: ~60% at cost of ~30% slower training.

### KV Cache Quantization

Reduce attention memory for long sequences:

```yaml
model:
  quantize_kv_cache: true
```

## 3.5 Triton Kernels

Custom CUDA kernels for performance:

- `code/src/ava/kernels/moe.py`: Fused gating and top-k
- `code/src/ava/kernels/activations.py`: Fused SwiGLU/GeGLU
- `code/src/ava/kernels/fused_experts.py`: Batched expert computation

Enable via:

```yaml
model:
  use_triton_kernels: true
  use_fused_activations: true
```

## 3.6 Design Decisions

### Why MoE?

1. **Compute efficiency**: Only k/N experts active per token
2. **Scaling**: Add experts without proportional compute increase
3. **Specialization**: Experts learn different features

### Why Flash Attention?

1. **Memory**: O(N) vs O(N²) for sequence length N
2. **Speed**: Fused CUDA kernels, no materialized attention matrix
3. **Long contexts**: Enables 8K+ sequence lengths

### Why SwiGLU?

1. **Performance**: Better than ReLU/GELU on LLM benchmarks
2. **Gating**: Learned gating improves expressivity
3. **Stability**: Smoother gradients than ReLU

---

# 4. Configuration Reference

Complete reference for all 110+ configuration parameters in Ava.

## 4.1 Configuration System

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

## 4.2 Loading Configuration

```python
from ava.config.training_config import TrainingConfigManager

manager = TrainingConfigManager()
config = manager.load_yaml_config("code/configs/moe/large.yaml")

# Access via dot notation
batch_size = config.training.batch_size
hidden_size = config.model.hidden_size
```

## 4.3 Model Configuration

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

## 4.4 Training Configuration

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

## 4.5 Data Configuration

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

## 4.6 Hardware Configuration

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

## 4.7 Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | auto | Output directory |
| `save_every` | int | 100 | Checkpoint frequency |
| `resume` | str | None | Resume checkpoint path |
| `fresh_start` | bool | false | Ignore existing checkpoints |

## 4.8 Weights & Biases Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | bool | false | Enable WandB |
| `wandb_offline` | bool | false | Force offline mode |
| `wandb_project` | str | 'Ava' | Project name |
| `wandb_name` | str | None | Run name |
| `wandb_tags` | list | ['moe', 'training'] | Run tags |
| `wandb_log_freq` | int | 10 | Logging frequency |

## 4.9 DeepSpeed Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_deepspeed` | bool | false | Enable DeepSpeed |
| `config_file` | str | None | DeepSpeed JSON config |
| `zero_stage` | int | 2 | ZeRO stage (0-3) |
| `cpu_offload` | bool | false | CPU offloading |
| `nvme_offload` | bool | false | NVMe offloading |
| `precision_type` | str | 'fp16' | 'fp16', 'bf16', 'fp32' |

## 4.10 Performance Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ultra_fast_mode` | bool | false | Disable all logging |
| `fast_progress` | bool | false | Enhanced progress bar |
| `express_mode` | bool | false | Optimized async logging |
| `enable_tf32` | bool | true | TF32 on Ampere+ |
| `enable_cudnn_benchmark` | bool | true | cuDNN auto-tuning |

## 4.11 Coherence Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable coherence eval |
| `eval_every_n_steps` | int | 500 | Evaluation frequency |
| `num_samples` | int | 10 | Samples to generate |
| `max_generation_length` | int | 256 | Max generation length |
| `temperature` | float | 0.8 | Generation temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `top_k` | int | 50 | Top-k sampling |

## 4.12 Logging Configuration

### Logging Intervals

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_interval` | int | 100 | Main logging interval (GPU->CPU sync for loss) |
| `verbose_log_interval` | int | 500 | Detailed INFO logs every N steps (0 = never) |
| `tqdm_update_interval` | int | 10 | Progress bar update frequency |
| `log_mode` | str | 'tqdm' | 'tqdm' (progress bar only) or 'verbose' (tqdm + INFO) |

### Monitoring Frequencies

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbosity` | str | 'info' | Log level |
| `metrics_log_freq` | int | 500 | Metrics logging frequency |
| `memory_check_freq` | int | 2000 | Memory check frequency |
| `health_summary_freq` | int | 500 | Health summary frequency |
| `moe_metrics_freq` | int | 5000 | MoE metrics frequency |
| `enable_timing_breakdown` | bool | true | Step timing logs |
| `enable_memory_profiling` | bool | true | Memory profiling |

## 4.13 Progressive Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_progressive_training` | bool | false | Enable progressive |
| `enable_sequence_scaling` | bool | false | Sequence length scaling |
| `initial_seq_length` | int | 128 | Starting length |
| `final_seq_length` | int | 2048 | Final length |
| `length_schedule` | str | 'linear' | Scaling schedule |
| `enable_curriculum` | bool | false | Curriculum learning |

## 4.14 Dynamic Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable dynamic batching |
| `min_batch_size` | int | 16 | Minimum batch size |
| `max_batch_size` | int | 256 | Maximum batch size |
| `target_memory` | float | 0.75 | Target memory usage |
| `high_memory` | float | 0.85 | Start decreasing |
| `critical_memory` | float | 0.92 | Emergency threshold |

## 4.15 FP8 Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable FP8 |
| `use_transformer_engine` | bool | false | Use TE library |
| `format` | str | 'e4m3' | 'e4m3' or 'e5m2' |
| `margin` | int | 0 | Scale margin |

## 4.16 Example Configurations

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

## 4.17 Command-Line Overrides

Override YAML settings from command line:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --epochs 10
```

---

# 5. Data Pipeline

Complete guide to data loading, preprocessing, and efficient data handling in Ava.

## 5.1 Overview

Ava supports multiple data loading strategies:

| Strategy | Memory | Throughput | Use Case |
|----------|--------|------------|----------|
| Streaming | Low | High | Large datasets |
| Pre-tokenized | Medium | Very High | Production |
| Indexed | High | Highest | Random access needed |

## 5.2 Data Formats

### Supported Formats

- **Arrow/Parquet**: Recommended for large datasets
- **JSONL**: Text data with metadata
- **HuggingFace Datasets**: Direct HF integration

### Data Structure

Pre-tokenized Arrow files should contain:

```python
{
    'input_ids': List[int],      # Token IDs
    'attention_mask': List[int],  # 1 for tokens, 0 for padding
    'labels': List[int],          # Target token IDs (optional)
}
```

## 5.3 Data Download

### Using Unified Download

```bash
# Download all supported datasets
python code/scripts/1_data_download/unified_download.py

# Limit partitions for testing
python code/scripts/1_data_download/unified_download.py --max-partitions 10
```

### Pre-tokenization

For maximum training speed, pre-tokenize your data:

```bash
python code/scripts/1_data_download/build_pretokenized_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --tokenizer code/data/Ava_Ai/tokenizer \
    --max-length 512
```

## 5.4 Streaming Dataset

Located in `code/src/ava/data/streaming.py`:

```python
class StreamingDataset:
    """Memory-efficient streaming from Arrow files."""

    def __init__(
        self,
        data_dir: str,
        buffer_size: int = 50000,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
```

### Configuration

```yaml
data:
  streaming: true
  buffer_size: 50000      # Shuffle buffer size
  num_workers: 8          # Parallel workers
  prefetch_factor: 4      # Batches to prefetch
  persistent_workers: true
```

### How It Works

1. Scans `data_dir` for Arrow files
2. Loads files lazily using memory-mapped I/O
3. Fills shuffle buffer from multiple files
4. Yields shuffled batches

## 5.5 Pre-tokenized Dataset

Located in `code/src/ava/data/pretokenized.py`:

```python
class PreTokenizedDataset:
    """High-performance loading from pre-tokenized Arrow files."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        use_mmap: bool = True,
    ):
        # Memory-map for zero-copy access
        self.table = pa.ipc.open_file(pa.memory_map(data_path, 'r')).read_all()
```

### Benefits

- **Zero-copy loading**: No tokenization overhead
- **Memory mapping**: OS handles caching
- **Consistent sequences**: Reproducible training

## 5.6 Distributed Dataset

For multi-GPU training:

```python
# code/src/ava/data/distributed.py
class DistributedStreamingDataset:
    """Shards data across distributed workers."""

    def __init__(
        self,
        data_dir: str,
        rank: int,
        world_size: int,
        ...
    ):
        # Assign file shards to each rank
        self.files = self._shard_files(all_files, rank, world_size)
```

## 5.7 Data Factory

The recommended way to create dataloaders:

```python
from ava.data.factory import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1,
)
```

## 5.8 Sequence Packing

Combine short sequences to eliminate padding waste:

```yaml
data:
  use_sequence_packing: true
  packing_strategy: 'greedy'   # or 'adaptive'
```

### How Packing Works

```
Before packing:
[A, A, A, PAD, PAD, PAD]  # 50% waste
[B, B, PAD, PAD, PAD, PAD]  # 67% waste

After packing:
[A, A, A, SEP, B, B]        # 0% waste
```

Implementation in `code/src/ava/data/packing.py`:

```python
class SequencePacker:
    def pack_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Greedy bin-packing of sequences."""
        packed = []
        current_bin = []
        current_length = 0

        for seq in sorted(sequences, key=len, reverse=True):
            if current_length + len(seq) + 1 <= self.max_length:
                current_bin.extend(seq + [self.sep_token])
                current_length += len(seq) + 1
            else:
                packed.append(current_bin)
                current_bin = seq + [self.sep_token]
                current_length = len(seq) + 1

        return packed
```

## 5.9 Length-Based Bucketing

Group similar-length sequences for efficient batching:

```yaml
data:
  enable_bucketing: true
```

Located in `code/src/ava/data/bucketing.py`:

```python
class LengthBasedBucketing:
    """Groups sequences by length to minimize padding."""

    def __init__(self, num_buckets: int = 8):
        self.buckets = [[] for _ in range(num_buckets)]

    def add(self, sequence, length):
        bucket_idx = self._get_bucket(length)
        self.buckets[bucket_idx].append(sequence)
```

## 5.10 Conversation Data

Turn-aware loading for dialogue:

```yaml
data:
  use_conversation_format: true
  turn_separator: "<|turn|>"
```

Located in `code/src/ava/data/conversation.py`:

```python
class ConversationDataset:
    """Preserves dialogue structure with turn markers."""

    def format_conversation(self, turns: List[Dict]) -> str:
        formatted = []
        for turn in turns:
            role = turn['role']
            content = turn['content']
            formatted.append(f"<|{role}|>{content}")
        return self.turn_separator.join(formatted)
```

## 5.11 Multi-Column Data

Load datasets with multiple columns:

```yaml
multi_column_data:
  use_multi_column: true
  hf_dataset: "openai/gsm8k"
  column_names: "question,answer"
  column_types: "text,text"
  combine_strategy: 'template'
  column_template: "Question: {question}\nAnswer: {answer}"
```

## 5.12 Data Randomization

### Deterministic Training

```yaml
data:
  shuffle_seed: 42
  randomization:
    deterministic: true
```

### Maximum Randomness

```yaml
data:
  shuffle_seed: null     # Random seed each run
  buffer_size: 100000    # Large shuffle buffer
  randomization:
    deterministic: false
```

## 5.13 Performance Optimization

### Parallel Loading

```yaml
data:
  num_workers: 8              # CPU cores for loading
  prefetch_factor: 4          # Batches ahead to load
  persistent_workers: true    # Keep workers between epochs
  dataloader_pin_memory: true # Faster GPU transfer
```

### Worker Configuration

| GPUs | Recommended Workers | Prefetch |
|------|-------------------|----------|
| 1 | 4-8 | 2-4 |
| 4 | 4 per GPU | 2 |
| 8 | 2-4 per GPU | 2 |

### Memory-Mapped Loading

For large datasets:

```yaml
data:
  use_mmap: true              # Memory-map files
  dataloader_drop_last: true  # Avoid partial batches
```

## 5.14 Validation Split

### Automatic Split

```yaml
data:
  auto_create_validation_split: true
  validation_split_ratio: 0.1
  val_max_samples: 1000       # Limit validation size
```

### Separate Files

```yaml
data:
  train_split: 'train'
  eval_split: 'validation'
```

## 5.15 Tokenizer Integration

### Custom Tokenizer

```yaml
data:
  tokenizer_name: "code/data/Ava_Ai/tokenizer"
  padding_side: 'right'
  truncation: true
  max_length: 512
```

### Special Tokens

Tokens are configured in model config:

```yaml
model:
  pad_token_id: 0
  eos_token_id: 1
  bos_token_id: 2
```

## 5.16 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DataLoaderManager                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 create_dataloaders()                     │ │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │ │
│  │  │ StreamingDS  │  │ Pretokenized  │  │ ConversationDS │ │ │
│  │  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘  │ │
│  │         │                  │                 │          │ │
│  │         └──────────────────┼─────────────────┘          │ │
│  │                            ▼                            │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │   LengthBasedBucketing  │                │ │
│  │              └────────────┬────────────┘                │ │
│  │                           ▼                             │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │    SequencePacker       │                │ │
│  │              └────────────┬────────────┘                │ │
│  │                           ▼                             │ │
│  │              ┌─────────────────────────┐                │ │
│  │              │      DataLoader         │                │ │
│  │              └─────────────────────────┘                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

# 6. Training Guide

Comprehensive guide to training models with Ava, from basic runs to advanced configurations.

## 6.1 Training Overview

The training pipeline follows this flow:

```
Config Loading → Model Building → Data Loading → Training Loop → Checkpointing
                                                      ↓
                              [Forward → Loss → Backward → Optimizer Step]
```

## 6.2 Basic Training

### Single GPU Training

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

### Resuming Training

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --resume code/outputs/pretraining/run_xyz/checkpoints/checkpoint_step_5000.pt
```

## 6.3 Configuration Structure

Training is controlled by YAML configuration files:

```yaml
# code/configs/moe/large.yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_experts: 8
  num_experts_per_token: 2

training:
  batch_size: 128
  learning_rate: 0.0006
  warmup_steps: 1000
  num_epochs: 5
  gradient_accumulation_steps: 4
  max_gradient_norm: 1.0

data:
  data_dir: "code/data/processed"
  max_length: 512
  num_workers: 8

output:
  output_dir: "code/outputs/pretraining"
  save_every: 500
```

## 6.4 Training Pipeline Components

The `TrainingPipeline` class orchestrates training through registered components:

```python
# code/src/ava/training/pipeline.py
pipeline = TrainingPipeline(context)
pipeline.register('model', ModelBuilder(context))
pipeline.register('optimizer', OptimizerManager(context))
pipeline.register('data', DataLoaderManager(context))
pipeline.register('training', TrainingLoopManager(context))
pipeline.register('validation', ValidationManager(context))
pipeline.register('metrics', MetricsManager(context))

pipeline.initialize_all()
# ... training loop ...
pipeline.cleanup_all()
```

### Component Lifecycle

Each component implements lifecycle hooks:

| Hook | When Called | Severity |
|------|-------------|----------|
| `initialize()` | Before training | FATAL |
| `cleanup()` | After training | WARNING |
| `on_epoch_start(epoch)` | Start of epoch | WARNING |
| `on_epoch_end(epoch)` | End of epoch | WARNING |
| `on_step_start(step)` | Start of step | WARNING |
| `on_step_end(step, loss)` | End of step | WARNING |
| `on_error(error)` | On training error | WARNING |

## 6.5 Learning Rate Scheduling

### Warmup + Cosine Decay

```yaml
training:
  learning_rate: 0.0006
  warmup_steps: 1000
  max_steps: 100000

  adaptive_lr:
    scheduler: 'cosine'
    min_lr_ratio: 0.1
```

### Learning Rate Finder

Find optimal learning rate before training:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --run-lr-finder \
    --lr-finder-use-suggested
```

Configuration:
```yaml
lr_finder:
  run_lr_finder: true
  start_lr: 1e-8
  end_lr: 1.0
  num_iterations: 100
  suggestion_method: 'steepest'
```

## 6.6 Gradient Management

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  batch_size: 32                    # Micro batch
  gradient_accumulation_steps: 4    # Effective: 32 × 4 = 128
```

### Gradient Clipping

Prevent exploding gradients:

```yaml
training:
  max_gradient_norm: 1.0
```

### Gradient Checkpointing

Trade compute for memory:

```yaml
model:
  gradient_checkpointing: true
```

## 6.7 Mixed Precision Training

### BF16 (Recommended for Ampere+)

```yaml
hardware:
  mixed_precision: 'bf16'
```

### FP16 (Older GPUs)

```yaml
hardware:
  mixed_precision: 'fp16'
```

### FP8 (Hopper/Ada GPUs)

```yaml
fp8:
  enabled: true
  format: 'e4m3'
```

## 6.8 Checkpointing

### Automatic Checkpointing

```yaml
output:
  save_every: 500           # Save every N steps
  output_dir: "code/outputs/pretraining"
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': current_epoch,
    'step': global_step,
    'best_loss': best_loss,
    'config': config,
}
```

### Async Checkpointing

Save checkpoints without blocking training:

```yaml
optimizations:
  checkpoint:
    async_saving: true
```

## 6.9 Validation During Training

### Periodic Validation

```yaml
evaluation:
  eval_during_training: true
  eval_frequency: 500
  eval_metrics: 'loss,perplexity,coherence'
```

### Coherence Evaluation

Generate samples and measure quality:

```yaml
coherence:
  enabled: true
  eval_every_n_steps: 500
  num_samples: 10
  max_generation_length: 256
```

## 6.10 Monitoring & Logging

### Weights & Biases

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --use-wandb \
    --wandb-project "Ava" \
    --wandb-tags "moe,training"
```

Configuration:
```yaml
wandb:
  use_wandb: true
  wandb_project: 'Ava'
  wandb_log_freq: 10
```

### Console Logging

```yaml
logging:
  verbosity: 'info'
  metrics_log_freq: 100
  health_summary_freq: 500
```

## 6.11 Performance Modes

### Ultra Fast Mode

Disable all logging for maximum speed:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --ultra-fast-mode
```

### Express Mode

Optimized async logging:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --express-mode
```

## 6.12 Fine-Tuning

### From Pretrained Checkpoint

```bash
python code/scripts/5_training/finetune.py \
    --checkpoint code/outputs/pretraining/run_xyz/checkpoints/best_model.pt \
    --data-dir code/data/fine-tuning
```

### Auto-Discovery

```bash
# Automatically finds latest checkpoint and data
python code/scripts/5_training/finetune.py
```

## 6.13 Training Strategies

### Progressive Training

Start with shorter sequences, increase over training:

```yaml
training:
  progressive:
    enable_progressive_training: true
    enable_sequence_scaling: true
    initial_seq_length: 128
    final_seq_length: 2048
    length_schedule: 'linear'
```

### Curriculum Learning

Train on easier examples first:

```yaml
training:
  progressive:
    enable_curriculum: true
    curriculum_metric: 'loss'
```

## 6.14 Common Training Patterns

### Memory-Efficient Training

```yaml
model:
  gradient_checkpointing: true
  use_flash_attention: true

hardware:
  mixed_precision: 'bf16'

training:
  gradient_accumulation_steps: 8
```

### Fast Iteration

```yaml
model:
  gradient_checkpointing: false
  use_triton_kernels: true

data:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
```

### Maximum Quality

```yaml
training:
  learning_rate: 0.0003       # Lower LR
  warmup_steps: 2000          # Longer warmup
  max_gradient_norm: 0.5      # Stricter clipping
  gradient_accumulation_steps: 8

coherence:
  enabled: true
  eval_every_n_steps: 250     # Frequent evaluation

model_selection:
  enabled: true
  val_loss_weight: 0.5
  coherence_score_weight: 0.3
```

---

# 7. Distributed Training

Complete guide to multi-GPU and multi-node training with Ava.

## 7.1 Overview

Ava supports multiple distributed training strategies:

| Strategy | Use Case | Memory Savings | Communication |
|----------|----------|----------------|---------------|
| DDP | Multi-GPU, fits in memory | None | Gradient sync |
| FSDP | Large models | High | Parameter sharding |
| DeepSpeed ZeRO-1 | Optimizer memory | Moderate | Optimizer sharding |
| DeepSpeed ZeRO-2 | Gradient + Optimizer | High | Gradient sharding |
| DeepSpeed ZeRO-3 | Full sharding | Maximum | All parameters |

## 7.2 Quick Start

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

## 7.3 Distributed Data Parallel (DDP)

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

## 7.4 DeepSpeed Integration

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

## 7.5 Expert Parallelism

Distribute MoE experts across GPUs using DeepSpeed or FSDP.

### Configuration

```yaml
model:
  num_experts: 8

hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'adaptive'

deepspeed:
  use_deepspeed: true
  zero_stage: 3  # Full sharding for expert distribution
```

### Load Balancing

```yaml
hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'memory_aware'  # Consider GPU memory
  rebalance_interval: 1000            # Check every 1000 steps
  enable_expert_migration: true       # Move experts if imbalanced
  migration_threshold: 0.2            # 20% imbalance triggers migration
```

## 7.6 Data Parallelism

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

## 7.7 Gradient Synchronization

### Gradient Accumulation with DDP

```yaml
training:
  gradient_accumulation_steps: 4

# Gradients synced only on accumulation boundary
# Reduces communication overhead
```

## 7.8 Checkpointing in Distributed Training

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

## 7.9 Memory Optimization for Large Models

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

## 7.10 Communication Optimization

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

## 7.11 Example: Large-Scale Training

```yaml
# Multi-node, multi-GPU configuration
hardware:
  num_gpus: 8

model:
  num_experts: 16
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

---

# 8. RLHF Training

Reinforcement Learning from Human Feedback (RLHF) training with Ava.

## 8.1 Overview

RLHF aligns language models with human preferences through:

1. **Supervised Fine-Tuning (SFT)**: Train on curated examples
2. **Reward Model Training**: Learn human preferences
3. **PPO Training**: Optimize policy against reward model

```
SFT Model → Reward Model → PPO Training → Aligned Model
               ↑                ↓
         Human Preferences    Feedback Loop
```

## 8.2 Quick Start

### Prepare Prompts

```bash
python code/scripts/6_rhlf_Finetuning/prepare_prompts.py \
    --input data/raw_prompts.jsonl \
    --output data/rlhf_prompts.jsonl
```

### Train RLHF

```bash
python code/scripts/6_rhlf_Finetuning/train_rlhf.py \
    --config code/configs/rlhf/rlhf_config.yaml
```

## 8.3 RLHF Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      RLHF Pipeline                          │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  Policy Model │  │ Reward Model  │  │ Reference Model │  │
│  │  (Trainable)  │  │  (Frozen)     │  │   (Frozen)      │  │
│  └───────┬───────┘  └───────┬───────┘  └────────┬────────┘  │
│          │                  │                   │           │
│          ▼                  ▼                   ▼           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    PPO Trainer                         │  │
│  │  Generate → Score → Compute Advantages → Update       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Policy Model

The model being optimized:

```python
# code/src/rlhf/ppo_trainer.py
class PPOTrainer:
    def __init__(self, policy_model, reward_model, reference_model):
        self.policy = policy_model      # Updated by PPO
        self.reward = reward_model      # Frozen
        self.reference = reference_model # Frozen, for KL penalty
```

### Reward Model

Scores generations for quality:

```python
# code/src/rlhf/reward_model.py
class RewardModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Output: scalar reward per sequence
        hidden = self.backbone(input_ids, attention_mask)
        reward = self.reward_head(hidden[:, -1, :])
        return reward
```

## 8.4 Configuration

### RLHF Config

```yaml
# code/configs/rlhf/rlhf_config.yaml
rlhf:
  # Model paths
  policy_model_path: "code/outputs/pretraining/best_model.pt"
  reward_model_path: "code/outputs/reward_model/best.pt"
  reference_model_path: null  # Uses policy model snapshot

  # PPO hyperparameters
  ppo_epochs: 4
  batch_size: 32
  mini_batch_size: 8
  learning_rate: 1e-5
  kl_coef: 0.1              # KL penalty coefficient
  clip_range: 0.2           # PPO clip range
  value_coef: 0.5           # Value loss coefficient
  entropy_coef: 0.01        # Entropy bonus

  # Generation settings
  max_new_tokens: 128
  temperature: 0.8
  top_p: 0.9

  # Training settings
  num_episodes: 10000
  save_frequency: 500
  eval_frequency: 100
```

### Minimal Config

```yaml
# code/configs/rlhf/rlhf_minimal.yaml
rlhf:
  policy_model_path: "path/to/model.pt"
  reward_model_path: "path/to/reward.pt"

  ppo_epochs: 2
  batch_size: 16
  learning_rate: 5e-6

  max_new_tokens: 64
  num_episodes: 1000
```

## 8.5 Training Process

### PPO Training Loop

```python
for episode in range(num_episodes):
    # 1. Sample prompts from dataset
    prompts = dataloader.sample(batch_size)

    # 2. Generate responses with policy
    responses = policy.generate(prompts, max_new_tokens=128)

    # 3. Score with reward model
    rewards = reward_model(prompts + responses)

    # 4. Compute KL penalty (stay close to reference)
    kl_penalty = compute_kl(policy, reference, prompts + responses)
    adjusted_rewards = rewards - kl_coef * kl_penalty

    # 5. Compute advantages using GAE
    advantages = compute_gae(adjusted_rewards, values)

    # 6. PPO update
    for _ in range(ppo_epochs):
        policy_loss = ppo_loss(advantages, old_probs, new_probs)
        value_loss = mse_loss(values, returns)
        entropy_loss = -entropy(new_probs)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
        loss.backward()
        optimizer.step()
```

### Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `kl_coef` | 0.01-0.5 | Higher = more conservative updates |
| `clip_range` | 0.1-0.3 | PPO clipping range |
| `ppo_epochs` | 1-8 | Optimization passes per batch |
| `learning_rate` | 1e-6 to 1e-4 | Lower than SFT typically |

## 8.6 Reward Model Training

### Data Format

```json
{
    "prompt": "Write a poem about nature",
    "chosen": "The gentle breeze whispers through trees...",
    "rejected": "Nature is nice. Trees are green..."
}
```

### Training

```bash
python code/scripts/6_rhlf_Finetuning/train_reward_model.py \
    --data data/preference_pairs.jsonl \
    --base-model code/outputs/pretraining/best_model.pt \
    --output code/outputs/reward_model
```

### Loss Function

```python
def reward_loss(chosen_rewards, rejected_rewards):
    # Bradley-Terry model
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

## 8.7 Best Practices

### Reward Hacking Prevention

1. **KL Penalty**: Keep model close to reference

```yaml
rlhf:
  kl_coef: 0.2  # Increase if model diverges too much
```

2. **Reward Clipping**: Prevent extreme rewards

```python
rewards = torch.clamp(rewards, -10, 10)
```

3. **Diverse Prompts**: Train on varied prompt distribution

### Stability Tips

1. **Lower Learning Rate**: 1e-6 to 1e-5 typical
2. **Warmup**: Gradual learning rate increase
3. **Gradient Clipping**: Max norm 0.5-1.0
4. **Small Batch Updates**: Multiple mini-batches per batch

### Memory Optimization

For large models:

```yaml
rlhf:
  # Use gradient checkpointing
  gradient_checkpointing: true

  # Offload reference model to CPU
  offload_reference: true

  # Reduce generation batch size
  generation_batch_size: 8
```

## 8.8 Common Issues

### Reward Collapse

Symptoms: All generations receive same reward

Solutions:
1. Check reward model quality
2. Increase prompt diversity
3. Reduce learning rate

### KL Explosion

Symptoms: KL divergence grows unboundedly

Solutions:
1. Increase `kl_coef`
2. Use adaptive KL controller
3. Reduce PPO epochs

### Poor Generation Quality

Symptoms: Model outputs degrade

Solutions:
1. Verify reward model alignment
2. Add entropy bonus
3. Check for mode collapse

## 8.9 Advanced Topics

### PPO-ptx (Mixed Training)

Combine RLHF with continued pretraining:

```yaml
rlhf:
  use_ptx: true
  ptx_coef: 0.1
  ptx_data: "data/pretraining_samples.jsonl"
```

### Direct Preference Optimization (DPO)

Simpler alternative to PPO:

```python
# Skip reward model, train directly on preferences
loss = -log_sigmoid(beta * (log_pi_chosen - log_pi_rejected))
```

## 8.10 Testing

### CPU Testing

```bash
python code/scripts/6_rhlf_Finetuning/test_rlhf_cpu.py
```

### Training Test

```bash
python code/scripts/6_rhlf_Finetuning/test_training_cpu.py
```

---

# 9. API Reference

Module-by-module documentation of the Ava framework.

## 9.1 Package Structure

```
code/src/ava/
├── config/           # Configuration management
├── core/             # Core utilities
├── cuda/             # CUDA utilities
├── data/             # Data loading
├── nn/               # Neural network layers
├── kernels/          # Triton kernels
├── models/           # Model architectures
├── optim/            # Optimizers
├── training/         # Training pipeline
├── optimizations/    # Training optimizations
├── strategies/       # Training strategies
└── eval/             # Evaluation
```

## 9.2 Configuration Module

### `ava.config.training_config`

#### DynamicConfig

Flexible configuration with dot-notation access.

```python
from ava.config.training_config import DynamicConfig

config = DynamicConfig({
    'model': {'hidden_size': 1024},
    'training': {'batch_size': 32}
})

# Access via dot notation
hidden = config.model.hidden_size  # 1024

# Dictionary-style access
batch = config['training']['batch_size']  # 32

# Safe access with default
lr = config.get('training.learning_rate', 0.001)
```

**Methods**:
- `to_dict()` - Convert to dictionary
- `validate()` - Check for circular references
- `validate_required_fields(fields)` - Verify required fields exist
- `validate_schema(schema)` - Validate against schema

#### TrainingConfigManager

Manages configuration loading and validation.

```python
from ava.config.training_config import TrainingConfigManager

manager = TrainingConfigManager()
config = manager.load_yaml_config("config.yaml")

# Validate
errors = manager.validate_dynamic_config(config)

# Get feature summary
summary = manager.get_feature_summary(config)
```

## 9.3 Models Module

### `ava.models.moe`

#### EnhancedMoEConfig

Model configuration dataclass.

```python
from ava.models.moe import EnhancedMoEConfig

config = EnhancedMoEConfig(
    vocab_size=50680,
    hidden_size=1024,
    num_layers=16,
    num_attention_heads=16,
    intermediate_size=4096,
    num_experts=8,
    num_experts_per_token=2,
    router_type='mixtral',
)
```

#### EnhancedMoEModel

Main model class.

```python
from ava.models.moe import EnhancedMoEModel

model = EnhancedMoEModel(config)

# Forward pass
outputs = model(
    input_ids,           # [batch, seq_len]
    attention_mask=mask, # [batch, seq_len]
    labels=labels,       # [batch, seq_len] optional
)

# Outputs
logits = outputs.logits          # [batch, seq_len, vocab]
loss = outputs.loss              # scalar if labels provided
aux_loss = outputs.aux_loss      # MoE auxiliary loss
```

**Key Methods**:
- `forward(input_ids, attention_mask, labels)` - Forward pass
- `generate(input_ids, max_length, ...)` - Text generation
- `get_num_parameters()` - Total parameter count

### `ava.models.moe_layer`

#### SparseMoELayer

Mixture of Experts layer.

```python
from ava.models.moe_layer import SparseMoELayer

layer = SparseMoELayer(
    hidden_size=1024,
    intermediate_size=4096,
    num_experts=8,
    num_experts_per_token=2,
    router_type='mixtral',
)

output, aux_loss = layer(hidden_states)
```

## 9.4 Neural Network Layers

### `ava.models.routing`

#### MixtralRouter

Standard learned gating router.

```python
from ava.models.routing import MixtralRouter

router = MixtralRouter(
    hidden_size=1024,
    num_experts=8,
    num_experts_per_token=2,
)

routing_weights, selected_experts = router(hidden_states)
# routing_weights: [batch, seq, k]
# selected_experts: [batch, seq, k]
```

#### DeepSeekRouter

Hybrid router with shared experts.

```python
from ava.models.routing import DeepSeekRouter

router = DeepSeekRouter(
    hidden_size=1024,
    num_experts=8,
    num_shared_experts=2,
    num_experts_per_token=2,
)
```

### `ava.models.experts`

#### HighPerformanceExpert

Optimized FFN expert.

```python
from ava.models.experts import HighPerformanceExpert

expert = HighPerformanceExpert(
    hidden_size=1024,
    intermediate_size=4096,
    activation='swiglu',
)

output = expert(hidden_states)
```

## 9.5 Training Module

### `ava.training.pipeline`

#### TrainingPipeline

Central orchestrator for training.

```python
from ava.training.pipeline import TrainingPipeline
from ava.training.context import TrainingContext

context = TrainingContext(model=model, device=device)
pipeline = TrainingPipeline(context)

# Register components
pipeline.register('model', model_builder)
pipeline.register('optimizer', optimizer_manager)
pipeline.register('data', data_manager)
pipeline.register('training', training_loop)

# Lifecycle
pipeline.initialize_all()

for epoch in range(num_epochs):
    pipeline.on_epoch_start(epoch)
    # ... training loop ...
    pipeline.on_epoch_end(epoch)

pipeline.cleanup_all()
```

**Lifecycle Methods**:
- `initialize_all()` - Initialize components (FATAL on error)
- `cleanup_all()` - Cleanup in reverse order
- `on_epoch_start(epoch)` - Epoch start hook
- `on_epoch_end(epoch)` - Epoch end hook
- `on_step_start(step)` - Step start hook
- `on_step_end(step, loss)` - Step end hook
- `on_error(error)` - Error handling hook

### `ava.training.context`

#### TrainingContext

Shared state container.

```python
from ava.training.context import TrainingContext

context = TrainingContext(
    model=model,
    device=device,
    config=config,
)

# Access shared state
context.epoch       # Current epoch
context.step        # Current step
context.current_loss  # Latest loss
```

#### TrainingComponent

Base class for pipeline components.

```python
from ava.training.context import TrainingComponent

class MyComponent(TrainingComponent):
    def initialize(self):
        """Called before training."""
        pass

    def cleanup(self):
        """Called after training."""
        pass

    def is_initialized(self) -> bool:
        """Check initialization status."""
        return self._initialized
```

### `ava.training.loop`

#### TrainingLoopManager

Core training loop implementation.

```python
from ava.training.loop import TrainingLoopManager

loop = TrainingLoopManager(context)

# Train one epoch
avg_loss = loop.train_epoch(
    train_loader,
    optimizer,
    scheduler,
)
```

## 9.6 Data Module

### `ava.data.streaming`

#### StreamingDataset

Memory-efficient streaming dataset.

```python
from ava.data.streaming import StreamingDataset

dataset = StreamingDataset(
    data_dir="data/processed",
    buffer_size=50000,
    shuffle=True,
    seed=42,
)

for batch in dataset:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
```

### `ava.data.factory`

#### create_streaming_dataloaders

Factory function for dataloaders.

```python
from ava.data.factory import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1,
)
```

### `ava.data.pretokenized`

#### UltraFastPretokenizedDataset

High-performance pre-tokenized loading with zero-copy Arrow access.

```python
from ava.data.pretokenized import UltraFastPretokenizedDataset

dataset = UltraFastPretokenizedDataset(
    data_dir="data/tokenized",
    split="train",
    max_length=512,
)
```

## 9.7 Optimizations Module

### `ava.optimizations.checkpointing`

#### GradientCheckpointer

Activation checkpointing utilities.

```python
from ava.optimizations.checkpointing import checkpoint

# Wrap expensive forward pass
output = checkpoint(expensive_forward, input, use_reentrant=False)
```

### `ava.optimizations.fp8`

#### FP8Training

FP8 mixed precision support.

```python
from ava.optimizations.fp8 import FP8Config, enable_fp8

fp8_config = FP8Config(enabled=True, format='e4m3')
model = enable_fp8(model, fp8_config)
```

## 9.8 CUDA Utilities

### `ava.cuda.streams`

#### StreamPool

CUDA stream management.

```python
from ava.cuda.streams import StreamPool

pool = StreamPool(num_streams=4)

with pool.get_stream() as stream:
    # Operations on this stream
    tensor.copy_(other)
```

### `ava.cuda.metrics`

#### AsyncMetricsTracker

Non-blocking metrics tracking.

```python
from ava.cuda.metrics import AsyncMetricsTracker

tracker = AsyncMetricsTracker()

tracker.log({'loss': 0.5, 'lr': 1e-4})
metrics = tracker.get_metrics()
```

## 9.9 Optimizer Module

### `ava.optim.lr_managers`

#### AdaptiveLearningRateManager

Dynamic learning rate scheduling.

```python
from ava.optimizations.lr_managers import AdaptiveLearningRateManager

lr_manager = AdaptiveLearningRateManager(
    optimizer,
    warmup_steps=1000,
    total_steps=100000,
    min_lr_ratio=0.1,
)

for step in range(total_steps):
    lr_manager.step()
    current_lr = lr_manager.get_lr()
```

## 9.10 Evaluation Module

### `ava.eval.coherence`

#### CoherenceEvaluator

Text coherence metrics.

```python
from ava.training.coherence import CoherenceEvaluator

evaluator = CoherenceEvaluator(config)

scores = evaluator.evaluate(
    model,
    tokenizer,
    prompts=["Once upon a time"],
    num_samples=10,
)

print(f"Coherence: {scores['coherence_score']:.3f}")
print(f"Perplexity: {scores['perplexity']:.2f}")
print(f"Repetition: {scores['repetition_score']:.3f}")
```

## 9.11 Kernels Module

### `ava.kernels.moe`

Triton kernels for MoE operations.

```python
from ava.cuda.moe_kernels import fused_softmax_topk

# Fused softmax and top-k selection
weights, indices = fused_softmax_topk(logits, k=2)
```

### `ava.kernels.activations`

Fused activation kernels.

```python
from ava.cuda.kernel_activations import fused_swiglu

output = fused_swiglu(gate_proj, up_proj)
```

## 9.12 Utilities

### `ava.core.paths`

Path management utilities.

```python
from ava.core.paths import get_project_root, get_data_dir, get_outputs_dir

root = get_project_root()      # /root/Ava_AI
data = get_data_dir()          # /root/Ava_AI/code/data
outputs = get_outputs_dir()    # /root/Ava_AI/code/outputs
```

### `ava.core.checkpoint`

Checkpoint utilities.

```python
from ava.core.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    path="checkpoint.pt",
)

checkpoint = load_checkpoint("checkpoint.pt", device="cuda")
```

## 9.13 Extending Ava

### Custom Component

```python
from ava.training.context import TrainingComponent, ManagerInterface

class CustomMetrics(TrainingComponent, ManagerInterface):
    def initialize(self):
        self.metrics = {}
        self._initialized = True

    def cleanup(self):
        self.save_metrics()
        self._initialized = False

    def on_step_end(self, step: int, loss: float):
        self.metrics[step] = loss

    def get_status(self) -> Dict[str, Any]:
        return {'num_metrics': len(self.metrics)}
```

### Custom Router

```python
from ava.models.routing import BaseRouter

class CustomRouter(BaseRouter):
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.gate(hidden_states)
        # Custom routing logic
        weights, indices = self.custom_selection(logits)
        return weights, indices
```

---

# 10. Performance Tuning

Optimization strategies to maximize training throughput and efficiency.

## 10.1 Performance Overview

Key metrics to optimize:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| GPU Utilization | >90% | `nvidia-smi` |
| Memory Usage | 80-95% | `nvidia-smi` |
| Throughput | tokens/sec | Training logs |
| Time per Step | minimize | Training logs |

## 10.2 Quick Wins

### Enable TF32 (Ampere+ GPUs)

```yaml
performance:
  enable_tf32: true
```

**Impact**: 2-3x faster matrix operations with minimal precision loss.

### Enable Flash Attention

```yaml
model:
  use_flash_attention: true
```

**Impact**: 2-4x faster attention, O(N) memory vs O(N²).

### Use BF16 Mixed Precision

```yaml
hardware:
  mixed_precision: 'bf16'
```

**Impact**: 2x memory reduction, faster compute.

### Enable cuDNN Benchmark

```yaml
performance:
  enable_cudnn_benchmark: true
```

**Impact**: Auto-selects fastest convolution algorithms.

## 10.3 Data Loading Optimization

### Parallel Workers

```yaml
data:
  num_workers: 8          # Match CPU cores
  prefetch_factor: 4      # Batches ahead to load
  persistent_workers: true  # Don't restart workers
  dataloader_pin_memory: true
```

### Pre-tokenized Data

```bash
# Pre-tokenize once, load fast forever
python code/scripts/1_data_download/build_pretokenized_data.py \
    --output data/pretokenized
```

```yaml
data:
  use_pretokenized: true
  data_dir: "data/pretokenized"
```

### Sequence Packing

Eliminate padding waste:

```yaml
data:
  use_sequence_packing: true
  packing_strategy: 'greedy'
```

**Impact**: Up to 30-50% throughput improvement on variable-length data.

### Memory-Mapped Loading

```yaml
data:
  use_mmap: true
```

**Impact**: Faster initial loading, lower memory footprint.

## 10.4 Model Optimization

### Triton Kernels

```yaml
model:
  use_triton_kernels: true
  use_grouped_gemm: true
```

### Torch Compile

```yaml
model:
  use_torch_compile: true
hardware:
  compile: true
```

**Note**: First epoch slower due to compilation.

### Expert Optimization

```yaml
kernel_optimization:
  use_fused_softmax_topk: true
  use_fused_activations: true
  use_vectorized_capacity: true
```

## 10.5 Memory Optimization

### Gradient Checkpointing

Trade compute for memory:

```yaml
model:
  gradient_checkpointing: true
```

**Impact**: ~60% memory reduction, ~30% slower.

### KV Cache Quantization

```yaml
model:
  quantize_kv_cache: true
```

### Gradient Accumulation

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 8
  # Effective batch: 16 × 8 = 128
```

## 10.6 Training Optimization

### Optimizer Selection

**AdamW** (default): Good convergence, high memory
**8-bit Adam**: 2x optimizer memory reduction
**Lion**: Lower memory, competitive performance

```yaml
optimizer:
  name: 'adamw'
  # or: 'adam8bit', 'lion'
```

### Learning Rate

Use LR finder:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml \
    --run-lr-finder \
    --lr-finder-use-suggested
```

### Warmup Strategy

```yaml
training:
  warmup_steps: 2000
  adaptive_lr:
    scheduler: 'cosine'
    min_lr_ratio: 0.1
```

## 10.7 Distributed Optimization

### Gradient Compression

Reduce communication bandwidth:

```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2
  overlap_comm: true
  reduce_bucket_size: 1000000000  # 1GB buckets
```

### Load Balancing

```yaml
hardware:
  use_gpu_load_balancing: true
  balancing_strategy: 'adaptive'
```

## 10.8 Profiling

### Built-in Profiling

```yaml
dev_log:
  enabled: true
  show_step_breakdown: true
  report_interval: 100
```

### Nsight Systems

```bash
nsys profile -o ava_profile \
    python code/scripts/5_training/train_pipeline.py \
        --config config.yaml
```

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    # Training step
    loss = model(batch)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 10.9 Configuration Profiles

### Maximum Throughput

```yaml
model:
  use_flash_attention: true
  use_triton_kernels: true
  use_grouped_gemm: true
  gradient_checkpointing: false  # Trade memory for speed

hardware:
  mixed_precision: 'bf16'

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
  ultra_fast_mode: true

data:
  num_workers: 8
  persistent_workers: true
  use_sequence_packing: true
```

### Maximum Memory Efficiency

```yaml
model:
  gradient_checkpointing: true
  quantize_kv_cache: true
  use_flash_attention: true

hardware:
  mixed_precision: 'bf16'

training:
  gradient_accumulation_steps: 16
```

### Balanced (Recommended)

```yaml
model:
  use_flash_attention: true
  gradient_checkpointing: true
  use_triton_kernels: true

hardware:
  mixed_precision: 'bf16'

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true

data:
  num_workers: 4
  persistent_workers: true
  use_sequence_packing: true

training:
  gradient_accumulation_steps: 4
```

## 10.10 Benchmarks

### Expected Performance

| Config | GPU | Batch | Seq Len | Tokens/sec |
|--------|-----|-------|---------|------------|
| minimal | A100 40GB | 64 | 512 | ~50K |
| large | A100 40GB | 32 | 512 | ~30K |
| large | A100 80GB | 64 | 1024 | ~40K |
| large (4x) | 4×A100 | 128 | 512 | ~100K |

### Bottleneck Analysis

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low GPU util | Data loading | More workers, prefetch |
| High GPU util, slow | Memory bound | Reduce batch, enable checkpoint |
| Variable step time | GC pauses | Reduce object creation |
| First epoch slow | Compilation | Expected with torch.compile |

## 10.11 Hardware Recommendations

### Single GPU

| GPU | VRAM | Best Config |
|-----|------|-------------|
| RTX 3090 | 24GB | minimal + bf16 |
| RTX 4090 | 24GB | large + bf16 + Flash |
| A100 40GB | 40GB | large + Flash |
| A100 80GB | 80GB | large + large batch |
| H100 | 80GB | large + FP8 |

### Multi-GPU

| GPUs | Strategy | Config |
|------|----------|--------|
| 2-4 | DDP | large_Multy |
| 4-8 | DeepSpeed ZeRO-2 | + deepspeed_zero2 |
| 8+ | DeepSpeed ZeRO-3 | + cpu_offload |

## 10.12 Common Pitfalls

### Too Many Workers

```yaml
# Bad: More workers than CPU cores
data:
  num_workers: 32

# Good: Match CPU cores
data:
  num_workers: 8
```

### Ignoring Memory Headroom

```yaml
# Bad: Using 100% GPU memory
training:
  batch_size: 128  # Causes OOM on long sequences

# Good: Leave headroom
training:
  batch_size: 96
```

### Not Using Flash Attention

```yaml
# Bad: Standard attention
model:
  use_flash_attention: false

# Good: Always use on supported hardware
model:
  use_flash_attention: true
```

### Synchronous Checkpointing

```yaml
# Bad: Blocks training
output:
  save_every: 100

# Good: Async saves
optimizations:
  checkpoint:
    async_saving: true
```

## 10.13 Monitoring

### WandB Dashboard

```yaml
wandb:
  use_wandb: true
  wandb_log_freq: 10
```

Track:
- `train/throughput` - Tokens per second
- `system/gpu_utilization` - GPU usage
- `system/gpu_memory` - Memory usage
- `train/step_time` - Time per step

### Console Metrics

```yaml
logging:
  metrics_log_freq: 100
  health_summary_freq: 500
  enable_timing_breakdown: true
```

---

# 11. Troubleshooting

Common issues and solutions when using Ava.

## 11.1 Quick Diagnosis

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Verify installation
python code/scripts/check_dependencies.py
```

## 11.2 Memory Issues

### Out of Memory (OOM)

**Symptoms**:
```
CUDA out of memory. Tried to allocate X GiB
RuntimeError: CUDA error: out of memory
```

**Solutions**:

1. **Reduce batch size**
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 4  # Compensate
```

2. **Enable gradient checkpointing**
```yaml
model:
  gradient_checkpointing: true
```

3. **Use mixed precision**
```yaml
hardware:
  mixed_precision: 'bf16'
```

4. **Reduce sequence length**
```yaml
data:
  max_length: 256  # Reduce from 512
```

### Memory Leak

**Symptoms**:
- GPU memory grows over time
- Eventually OOM after many steps

**Solutions**:

1. **Clear cache periodically**
```python
import torch
torch.cuda.empty_cache()
```

2. **Disable persistent workers**
```yaml
data:
  persistent_workers: false
```

3. **Check for tensor accumulation**
```python
# Don't accumulate tensors
total_loss += loss.item()  # Use .item()
# Not: total_loss += loss
```

## 11.3 Training Issues

### Loss Not Decreasing

**Symptoms**:
- Loss stuck at high value
- No improvement after many steps

**Solutions**:

1. **Check learning rate**
```bash
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml \
    --run-lr-finder
```

2. **Increase warmup**
```yaml
training:
  warmup_steps: 2000  # Increase from 1000
```

3. **Verify data loading**
```python
# Debug: print first batch
for batch in train_loader:
    print(batch['input_ids'].shape)
    print(batch['input_ids'][:2])
    break
```

4. **Check for all-padding sequences**
```python
# Ensure sequences have content
mask_sum = batch['attention_mask'].sum(dim=1)
assert (mask_sum > 0).all(), "Found all-padding sequences"
```

### Loss Exploding (NaN/Inf)

**Symptoms**:
```
Loss: nan or inf
RuntimeError: Loss is nan
```

**Solutions**:

1. **Lower learning rate**
```yaml
training:
  learning_rate: 0.0001  # Reduce from 0.001
```

2. **Enable gradient clipping**
```yaml
training:
  max_gradient_norm: 0.5  # Stricter clipping
```

3. **Check for numerical issues**
```python
# Add to training loop
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Bad loss at step {step}")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN grad in {name}")
```

4. **Use stable softmax**
```yaml
model:
  router_z_loss_coef: 0.001  # Add z-loss
```

### Training Too Slow

**Symptoms**:
- Low GPU utilization
- Slow iterations

**Solutions**:

1. **Enable Flash Attention**
```yaml
model:
  use_flash_attention: true
```

2. **Increase data workers**
```yaml
data:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true
```

3. **Enable TF32**
```yaml
performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
```

4. **Use Triton kernels**
```yaml
model:
  use_triton_kernels: true
```

5. **Check data loading bottleneck**
```yaml
dev_log:
  enabled: true
  show_step_breakdown: true
```

## 11.4 Data Issues

### Data Not Found

**Symptoms**:
```
FileNotFoundError: Data directory not found
No Arrow files found in directory
```

**Solutions**:

1. **Check path**
```bash
ls -la code/data/processed/*.arrow
```

2. **Verify config path**
```yaml
data:
  data_dir: "code/data/processed"  # Absolute or relative to project root
```

3. **Download data**
```bash
python code/scripts/1_data_download/unified_download.py
```

### Tokenizer Mismatch

**Symptoms**:
- Garbled output
- `IndexError: index out of range`

**Solutions**:

1. **Verify tokenizer path**
```yaml
data:
  tokenizer_name: "code/data/Ava_Ai/tokenizer"
```

2. **Check vocab size match**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
print(f"Vocab size: {len(tokenizer)}")
# Should match model.vocab_size
```

3. **Re-tokenize data**
```bash
python code/scripts/1_data_download/build_pretokenized_data.py \
    --tokenizer code/data/Ava_Ai/tokenizer
```

### Empty Batches

**Symptoms**:
```
RuntimeError: Expected non-empty tensor
Zero-size tensor
```

**Solutions**:

1. **Check data file**
```python
import pyarrow as pa
table = pa.ipc.open_file(pa.memory_map("data.arrow", 'r')).read_all()
print(f"Rows: {len(table)}")
print(f"Columns: {table.column_names}")
```

2. **Add drop_last**
```yaml
data:
  dataloader_drop_last: true
```

## 11.5 Distributed Training Issues

### NCCL Timeout

**Symptoms**:
```
NCCL timeout
RuntimeError: NCCL watchdog timeout
```

**Solutions**:

1. **Increase timeout**
```bash
export NCCL_TIMEOUT=1800
```

2. **Check network**
```bash
# Test connectivity between nodes
ping other-node
```

3. **Verify NCCL settings**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
```

### Rank Mismatch

**Symptoms**:
```
All processes must use same world size
Rank out of bounds
```

**Solutions**:

1. **Consistent launch**
```bash
# All nodes must use same nnodes
torchrun --nproc_per_node=4 --nnodes=2 ...
```

2. **Check environment**
```python
import os
print(f"RANK: {os.environ.get('RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
```

### Hanging at Barrier

**Symptoms**:
- Training hangs
- No progress on one or more GPUs

**Solutions**:

1. **Check all processes started**
```bash
ps aux | grep python
```

2. **Verify master address**
```bash
# All nodes must reach master
ping $MASTER_ADDR
```

3. **Debug with NCCL**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## 11.6 Generation Issues

### Repetitive Output

**Symptoms**:
- Model repeats same phrase
- Degenerate output

**Solutions**:

1. **Add repetition penalty**
```yaml
generation:
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3
```

2. **Adjust temperature**
```yaml
generation:
  temperature: 0.8
  top_p: 0.9
```

3. **Check training quality**
```bash
python code/scripts/7_generation/generate.py \
    --prompt "Test prompt" \
    --checkpoint path/to/best_model.pt
```

### Incoherent Output

**Symptoms**:
- Nonsense text
- Grammatically incorrect

**Solutions**:

1. **Verify checkpoint loaded**
```python
checkpoint = torch.load("model.pt")
print(f"Step: {checkpoint.get('step')}")
print(f"Loss: {checkpoint.get('best_loss')}")
```

2. **Check tokenizer**
```python
text = "Hello world"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert text == decoded, "Tokenizer roundtrip failed"
```

3. **Train longer**
```yaml
training:
  epochs: 10  # Increase epochs
```

## 11.7 Configuration Issues

### Config Not Loading

**Symptoms**:
```
FileNotFoundError: Config file not found
yaml.scanner.ScannerError
```

**Solutions**:

1. **Check path**
```bash
ls -la code/configs/moe/large.yaml
```

2. **Validate YAML**
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

3. **Check indentation**
```yaml
# Correct (2 spaces)
model:
  hidden_size: 1024

# Wrong (mixed tabs/spaces)
model:
	hidden_size: 1024
```

### Missing Config Key

**Symptoms**:
```
AttributeError: 'NoneType' object has no attribute
KeyError: 'learning_rate'
```

**Solutions**:

1. **Use defaults**
```python
lr = config.training.learning_rate or 0.001
```

2. **Enable strict mode**
```python
DynamicConfig.enable_strict_mode()
# Will raise error for missing keys
```

## 11.8 Checkpoint Issues

### Cannot Load Checkpoint

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict
Missing keys / Unexpected keys
```

**Solutions**:

1. **Check model architecture**
```python
# Compare keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(checkpoint['model_state_dict'].keys())
print(f"Missing: {model_keys - ckpt_keys}")
print(f"Unexpected: {ckpt_keys - model_keys}")
```

2. **Load partial checkpoint**
```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

3. **Convert checkpoint**
```python
# Remove 'module.' prefix from DDP
new_state = {k.replace('module.', ''): v
             for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(new_state)
```

### Checkpoint Corruption

**Symptoms**:
```
pickle.UnpicklingError
RuntimeError: storage offset
```

**Solutions**:

1. **Load with weights_only**
```python
checkpoint = torch.load("model.pt", weights_only=True)
```

2. **Use backup checkpoint**
```bash
ls -la outputs/checkpoints/
# Use previous checkpoint
```

3. **Enable checkpoint validation**
```yaml
logging:
  log_checkpoint_validation: true
```

## 11.9 Environment Issues

### CUDA Not Available

**Symptoms**:
```
RuntimeError: CUDA is not available
torch.cuda.is_available() returns False
```

**Solutions**:

1. **Check NVIDIA driver**
```bash
nvidia-smi
```

2. **Check PyTorch CUDA**
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

3. **Reinstall PyTorch**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Package Version Conflicts

**Symptoms**:
```
ImportError: cannot import name
ModuleNotFoundError
```

**Solutions**:

1. **Check versions**
```bash
pip list | grep -E "torch|transformers|flash"
```

2. **Reinstall requirements**
```bash
pip install -r requirements.txt --force-reinstall
```

## 11.10 Getting Help

### Debug Mode

```yaml
logging:
  verbosity: 'debug'

dev_log:
  enabled: true
  show_step_breakdown: true
```

### Collect Diagnostics

```bash
# System info
nvidia-smi
python --version
pip list

# Training log
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml 2>&1 | tee training.log
```

### Report Issue

Include:
1. Error message and full traceback
2. Configuration file (sanitized)
3. System info (GPU, CUDA version)
4. Steps to reproduce

---

# Appendix

## A. File Locations Quick Reference

| Component | Location |
|-----------|----------|
| Main training script | `code/scripts/5_training/train_pipeline.py` |
| Model definition | `code/src/ava/models/moe.py` |
| MoE layer | `code/src/ava/models/moe_layer.py` |
| Routing | `code/src/ava/models/routing.py` |
| Config system | `code/src/ava/config/training_config.py` |
| Data loading | `code/src/ava/data/pretokenized.py` |
| Training loop | `code/src/ava/training/loop.py` |
| Training pipeline | `code/src/ava/training/pipeline.py` |
| CUDA kernels | `code/src/ava/cuda/moe_kernels.py` |

## B. Configuration Checklist

Before training, verify:

- [ ] `model.vocab_size` matches tokenizer vocabulary
- [ ] `data.data_dir` points to valid data directory
- [ ] `data.tokenizer_name` points to valid tokenizer
- [ ] `hardware.mixed_precision` matches GPU capabilities
- [ ] `model.use_flash_attention` is enabled if flash-attn is installed
- [ ] `training.batch_size` × `gradient_accumulation_steps` gives desired effective batch

## C. Common Commands

```bash
# Training
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config config.yaml

# Fine-tuning
python code/scripts/5_training/finetune.py

# Generation
python code/scripts/7_generation/generate.py --prompt "Hello"

# RLHF
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config code/configs/rlhf/rlhf_config.yaml

# Data preparation
python code/scripts/1_data_download/build_pretokenized_data.py

# Tests
python code/tests/test_all.py
```

---

*Document generated from Ava LLM Training Framework documentation files.*
