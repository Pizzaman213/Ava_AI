# Ava LLM Training Framework

Advanced LLM training framework with Mixture of Experts (MoE++) architecture, optimized for memory efficiency and scalability.

## Quick Start

```bash
# Main training (single GPU)
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU training
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config code/configs/moe/Min_multy.yaml

# Fine-tuning from checkpoint
python code/scripts/5_training/finetune.py

# RLHF training
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config code/configs/rlhf/rlhf_config.yaml

# Build pre-tokenized data
python code/scripts/1_data_download/build_pretokenized_data.py

# Train custom tokenizer
python code/scripts/1_data_download/train_custom_tokenizer.py

# Text generation
python code/scripts/7_generation/generate.py --prompt "Once upon a time"
```

## Project Structure

```
/root/Ava_AI/
├── code/                           # Main codebase
│   ├── src/                        # Source code
│   │   ├── ava/                    # Core framework package
│   │   │   ├── config/             # Configuration management
│   │   │   │   ├── training_config.py   # 39 dataclasses, 110+ parameters
│   │   │   │   ├── yaml_loader.py       # YAML parsing & validation
│   │   │   │   ├── validator.py         # ConfigValidator with path resolution
│   │   │   │   ├── path_mapping.py      # Backward compatibility mappings
│   │   │   │   └── constants.py         # DataPipelineConstants, TrainerConstants
│   │   │   │
│   │   │   ├── core/               # Core utilities
│   │   │   │   ├── paths.py             # Path management
│   │   │   │   ├── checkpoint.py        # Async checkpoint saving
│   │   │   │   ├── mixed_precision.py   # AMP training utilities
│   │   │   │   ├── data_utils.py        # Collation, data helpers
│   │   │   │   ├── activations.py       # Activation factory
│   │   │   │   ├── logging.py           # Colored logging system
│   │   │   │   ├── script_utils.py      # Script helpers
│   │   │   │   ├── wandb_logger.py      # WandB integration
│   │   │   │   ├── error_tracking.py    # Error tracking utilities
│   │   │   │   └── errors.py            # Custom exception definitions
│   │   │   │
│   │   │   ├── cuda/               # CUDA & Triton kernels
│   │   │   │   ├── profiler.py          # Nsight profiling
│   │   │   │   ├── moe_kernels.py       # Fused gating/topk kernels
│   │   │   │   ├── fused_experts.py     # Fused expert computation
│   │   │   │   ├── kernel_activations.py # Fused SwiGLU/GeGLU
│   │   │   │   ├── streams.py           # StreamPool, CUDATimer
│   │   │   │   ├── metrics.py           # Async metrics tracking
│   │   │   │   └── rope_kernel.py       # RoPE embeddings kernel
│   │   │   │
│   │   │   ├── data/               # Data loading & processing
│   │   │   │   ├── pretokenized.py      # Memory-mapped Arrow loading
│   │   │   │   ├── indexed.py           # Index-based with true shuffle
│   │   │   │   ├── multi_column.py      # Multi-modal data support
│   │   │   │   ├── packing.py           # Sequence packing (20-35% speedup)
│   │   │   │   ├── conversation.py      # Turn-aware dialogue loading
│   │   │   │   ├── bucketing.py         # Length-based batching
│   │   │   │   ├── distributed.py       # Distributed data loading
│   │   │   │   ├── collators.py         # Batch collation
│   │   │   │   ├── arrow_io.py          # Arrow I/O utilities
│   │   │   │   ├── validation.py        # Data validation
│   │   │   │   ├── profiling.py         # Pipeline profiling
│   │   │   │   ├── base_dataset.py      # Base dataset class
│   │   │   │   └── factory.py           # create_dataloaders factory
│   │   │   │
│   │   │   ├── models/             # Model architectures & components
│   │   │   │   ├── moe.py               # EnhancedMoEModel, EnhancedMoEConfig
│   │   │   │   ├── moe_layer.py         # SparseMoELayer
│   │   │   │   ├── experts.py           # HighPerformanceExpert, ExpertParallelGroup
│   │   │   │   └── routing.py           # MixtralRouter, DeepSeekRouter, UnifiedMoERouter
│   │   │   │
│   │   │   ├── optimizations/      # Training optimizations
│   │   │   │   ├── batch_controller.py  # Dynamic batch size control
│   │   │   │   ├── lr_managers.py       # AdaptiveLearningRateManager
│   │   │   │   ├── prefetch.py          # Async batch prefetching
│   │   │   │   ├── gradients.py         # Gradient utilities
│   │   │   │   ├── checkpointing.py     # Gradient checkpointing
│   │   │   │   ├── overlapped_recomputation.py  # Parallel backward pass
│   │   │   │   ├── hybrid_cache.py      # KV + activation caching
│   │   │   │   ├── fp8.py               # FP8 quantization
│   │   │   │   ├── quantization.py      # Quantization helpers
│   │   │   │   └── oom_recovery.py      # OOM recovery utilities
│   │   │   │
│   │   │   ├── training/           # Training pipeline
│   │   │   │   ├── loop.py              # TrainingLoopManager
│   │   │   │   ├── data_manager.py      # DataLoaderManager
│   │   │   │   ├── model_builder.py     # ModelBuilder
│   │   │   │   ├── optimizer.py         # OptimizerManager
│   │   │   │   ├── validation.py        # ValidationManager
│   │   │   │   ├── generation.py        # GenerationManager
│   │   │   │   ├── run_manager.py       # Experiment tracking
│   │   │   │   ├── pipeline.py          # TrainingPipeline orchestration
│   │   │   │   ├── distributed.py       # DDP/FSDP setup
│   │   │   │   ├── deepspeed.py         # DeepSpeed integration
│   │   │   │   ├── diagnostics.py       # Training diagnostics
│   │   │   │   ├── progressive.py       # Curriculum learning
│   │   │   │   ├── coherence.py         # Coherence metrics
│   │   │   │   ├── quality_evaluator.py # Output quality scoring
│   │   │   │   ├── context.py           # TrainingContext, TrainingComponent
│   │   │   │   └── state_guard.py       # Training state management
│   │   │   │
│   │   │   └── utils.py            # General utilities
│   │   │
│   │   ├── generation/             # Text generation
│   │   │   └── generator.py             # Generation utilities
│   │   │
│   │   └── rlhf/                   # RLHF training
│   │       ├── rlhf_trainer.py          # RLHF training loop
│   │       ├── ppo_trainer.py           # PPO implementation
│   │       └── reward_model.py          # Reward model
│   │
│   ├── scripts/                    # Executable scripts
│   │   ├── 1_data_download/        # Data preparation
│   │   │   ├── build_pretokenized_data.py   # Build pre-tokenized datasets
│   │   │   └── train_custom_tokenizer.py    # Train custom tokenizer
│   │   │
│   │   ├── 5_training/             # Training scripts
│   │   │   ├── train_pipeline.py        # Main training script
│   │   │   └── finetune.py              # Auto-discovery fine-tuning
│   │   │
│   │   ├── 6_rhlf_Finetuning/      # RLHF training
│   │   │   ├── train_rlhf.py            # RLHF training script
│   │   │   ├── prepare_prompts.py       # Prompt preparation
│   │   │   ├── test_rlhf_cpu.py         # RLHF CPU testing
│   │   │   └── test_training_cpu.py     # Training CPU testing
│   │   │
│   │   └── 7_generation/           # Text generation
│   │       └── generate.py              # Generation interface
│   │
│   ├── configs/                    # Configuration files
│   │   ├── moe/                    # MoE configurations
│   │   │   ├── large.yaml               # Production (446M params, RTX 3090 Ti)
│   │   │   ├── minimal_working.yaml     # Development/testing (62M params)
│   │   │   ├── minimal_working_fixed.yaml   # Fixed minimal config
│   │   │   ├── Min_multy.yaml           # Multi-GPU configuration
│   │   │   ├── Full_TEST.yaml           # Full test configuration
│   │   │   ├── fast.yaml                # Fast iteration config
│   │   │   └── stable_debug.yaml        # Stable debug config
│   │   │
│   │   ├── rlhf/                   # RLHF configurations
│   │   │   ├── rlhf_config.yaml         # Full RLHF config
│   │   │   └── rlhf_minimal.yaml        # Minimal RLHF config
│   │   │
│   │   ├── distributed/            # Distributed training
│   │   │   ├── deepspeed_zero1.yaml     # ZeRO-1 optimizer sharding
│   │   │   ├── deepspeed_zero2.yaml     # ZeRO-2 optimizer+gradient
│   │   │   └── deepspeed_zero3.yaml     # ZeRO-3 full parameter sharding
│   │   │
│   │   └── reference/              # Reference configurations
│   │       └── all_options.yaml         # Comprehensive config reference
│   │
│   ├── data/                       # Training data (local)
│   │   ├── fine-tuning/            # Fine-tuning datasets (Arrow format)
│   │   └── Ava_Ai/                 # Tokenizer & processed data
│   │
│   ├── outputs/                    # Training outputs
│   │   └── pretraining/            # Pre-training runs
│   │
│   ├── docs/                       # Documentation
│   │   ├── 01_GETTING_STARTED.md
│   │   ├── 02_ARCHITECTURE.md
│   │   ├── 03_TRAINING_GUIDE.md
│   │   ├── 04_CONFIGURATION.md
│   │   ├── 05_DATA_PIPELINE.md
│   │   ├── 06_DISTRIBUTED.md
│   │   ├── 07_RLHF.md
│   │   ├── 08_API_REFERENCE.md
│   │   ├── 09_TROUBLESHOOTING.md
│   │   └── 10_PERFORMANCE.md
│   │
│   └── tests/                      # Test suite
│       ├── test_all.py                  # Unified test runner
│       ├── test_pipeline_fixes.py       # Pipeline tests
│       └── fuzzing/                     # Fuzzing test suite
│           ├── config_fuzzer.py
│           ├── parameter_registry.py
│           └── mutation_strategies.py
│
├── pretokenized_data/              # Pre-tokenized datasets
├── data/                           # Root data directory
├── models/                         # Root model checkpoints
├── wandb/                          # WandB tracking
├── CLAUDE.md                       # This file
├── requirements.txt                # Python dependencies
└── apt.txt                         # System dependencies
```

## Configuration Files

| Config | Use Case |
|--------|----------|
| `moe/large.yaml` | Production training (446M params, optimized for RTX 3090 Ti) |
| `moe/minimal_working.yaml` | Development and testing (62M params) |
| `moe/minimal_working_fixed.yaml` | Fixed minimal config for testing |
| `moe/Min_multy.yaml` | Multi-GPU distributed training |
| `moe/fast.yaml` | Fast iteration config |
| `moe/stable_debug.yaml` | Stable debugging config |
| `moe/Full_TEST.yaml` | Full test configuration |
| `rlhf/rlhf_config.yaml` | Full RLHF training configuration |
| `rlhf/rlhf_minimal.yaml` | Minimal RLHF for testing |
| `distributed/deepspeed_zero*.yaml` | DeepSpeed ZeRO configurations |

## Key Features

### Model Architecture
- **MoE++ Architecture**: 4-32 experts with configurable top-k routing
- **Router Types**: Mixtral, DeepSeek (hybrid), Switch
- **Attention**: Flash Attention v2+, Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- **Embeddings**: Rotary Position Embeddings (RoPE) with custom Triton kernel
- **Activations**: SwiGLU/GeGLU gated activations
- **Auxiliary Losses**: Load balancing, router z-loss, diversity loss

### Special Token Configuration

All models use token IDs from config (not from tokenizer):

```yaml
model:
  vocab_size: 50680
  pad_token_id: 0    # Padding token
  eos_token_id: 1    # End-of-sequence
  bos_token_id: 2    # Beginning-of-sequence
```

**Important**: These IDs must match your tokenizer's vocabulary.

**Verification**:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
print(tokenizer.convert_tokens_to_ids(['[PAD]', '[EOS]', '[BOS]']))
# Should match: [0, 1, 2]
```

The training scripts automatically handle missing tokenizer special tokens by using model config IDs as fallback.

### Training Optimizations
- **Parallel Data Loading**: Configurable workers for optimal throughput
- **Gradient Checkpointing**: O(sqrt(n)) memory for extended sequences
- **Mixed Precision**: FP16/BF16/FP8 support
- **Hybrid Caching**: KV + activation caching
- **Overlapped Recomputation**: Parallel backward pass computation
- **Progressive Training**: Curriculum learning, sequence length scaling
- **Grouped GEMM**: Efficient expert computation
- **OOM Recovery**: Automatic recovery with batch size reduction
- **Async Batch Prefetching**: Overlapped data loading

### Data Loading
- **Pre-tokenized Loading**: Memory-mapped Arrow/Parquet files
- **Indexed Datasets**: True random shuffle with index-based access
- **Turn-Aware Conversation**: Preserves dialogue structure
- **Sequence Packing**: 20-35% throughput improvement
- **Multi-Column Datasets**: Text, numeric, categorical, image, tensor support
- **Length-Based Bucketing**: Efficient batch construction

### Distributed Training
- **DDP**: Data Distributed Parallel for multi-GPU
- **FSDP**: Fully Sharded Data Parallel for large models
- **DeepSpeed**: ZeRO-1/2/3 optimizer sharding
- **Expert Parallelism**: Distributed MoE experts across GPUs

## Key Files Reference

| File | Purpose |
|------|---------|
| `code/scripts/5_training/train_pipeline.py` | Main training script |
| `code/src/ava/models/moe.py` | EnhancedMoEModel, EnhancedMoEConfig |
| `code/src/ava/models/moe_layer.py` | SparseMoELayer |
| `code/src/ava/models/routing.py` | MixtralRouter, DeepSeekRouter, UnifiedMoERouter |
| `code/src/ava/models/experts.py` | HighPerformanceExpert, ExpertParallelGroup |
| `code/src/ava/config/training_config.py` | All configuration dataclasses (39 classes) |
| `code/src/ava/config/constants.py` | DataPipelineConstants, TrainerConstants |
| `code/src/ava/data/pretokenized.py` | PreTokenizedDataset |
| `code/src/ava/data/indexed.py` | IndexedDataset with true shuffle |
| `code/src/ava/data/packing.py` | Sequence packing implementation |
| `code/src/ava/optimizations/lr_managers.py` | AdaptiveLearningRateManager |
| `code/src/ava/optimizations/batch_controller.py` | Dynamic batch size control |
| `code/src/ava/training/loop.py` | TrainingLoopManager |
| `code/src/ava/training/pipeline.py` | TrainingPipeline |
| `code/src/ava/training/progressive.py` | ProgressiveTrainer (curriculum learning) |
| `code/src/ava/cuda/moe_kernels.py` | Fused MoE gating/topk kernels |
| `code/src/ava/cuda/fused_experts.py` | Fused expert computation |

## Common Tasks

### Training
```bash
# Single GPU
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/Min_multy.yaml

# Resume from checkpoint
python code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/large.yaml \
  --resume /path/to/checkpoint.pt

# Fast iteration (smaller config)
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/fast.yaml
```

### Fine-tuning
```bash
# Auto-discovers latest checkpoint and data
python code/scripts/5_training/finetune.py

# Specific checkpoint
python code/scripts/5_training/finetune.py --checkpoint /path/to/model.pt

# Use all available data files
python code/scripts/5_training/finetune.py --use-all-files
```

### RLHF Training
```bash
# Full RLHF training
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config code/configs/rlhf/rlhf_config.yaml

# Minimal RLHF for testing
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config code/configs/rlhf/rlhf_minimal.yaml
```

### Text Generation
```bash
# Auto-discover latest model
python code/scripts/7_generation/generate.py --prompt "Once upon a time"

# Interactive mode
python code/scripts/7_generation/generate.py --interactive --temperature 0.8

# Batch processing
python code/scripts/7_generation/generate.py --input-file prompts.txt --output-file responses.txt
```

### Data Preparation
```bash
# Build pre-tokenized dataset
python code/scripts/1_data_download/build_pretokenized_data.py

# Train custom tokenizer
python code/scripts/1_data_download/train_custom_tokenizer.py
```

### Testing
```bash
python code/tests/test_all.py
```

## Outputs Location
- **Training Runs**: `code/outputs/pretraining/ava_training_YYYYMMDD_HHMMSS_*/`
- **Checkpoints**: `code/outputs/pretraining/*/checkpoints/`
- **WandB Logs**: `code/scripts/5_training/wandb/`

## Configuration Reference

Full reference: `code/configs/reference/all_options.yaml`

### Configuration Structure (v2.0)

Configuration is organized into 8 main categories:

| Category | Purpose |
|----------|---------|
| `model` | Architecture, MoE, tokens, regularization |
| `training` | Optimizer, schedule, batching, validation |
| `data` | Loading, streaming, packing, splits |
| `compute` | Device, precision, CUDA, kernels, memory |
| `distributed` | Multi-GPU, DeepSpeed, load balancing |
| `logging` | Console, WandB, TensorBoard, diagnostics |
| `checkpoints` | Save/load, model selection |
| `experimental` | Advanced features (disabled by default) |

**Backward Compatibility:** Old config paths (e.g., `hardware.device`) still work and are automatically resolved to new paths with deprecation warnings. Set `SUPPRESS_CONFIG_DEPRECATION=1` to silence.

**Migration:** Use the migration tool to convert old configs:
```bash
python code/scripts/migrate_config.py old_config.yaml --output new_config.yaml
python code/scripts/migrate_config.py old_config.yaml --validate  # Check for deprecated paths
```

### Model Architecture

```yaml
model:
  # Core Architecture
  vocab_size: 50680                # Must match tokenizer
  hidden_size: 1024                # Divisible by num_attention_heads
  num_layers: 16                   # Transformer layers
  num_attention_heads: 16          # Attention heads
  intermediate_size: 4096          # FFN size (4x hidden_size)
  max_position_embeddings: 512     # Max sequence length
  activation: swiglu               # swiglu, geglu, gelu, relu

  # Special Tokens
  pad_token_id: 0
  eos_token_id: 1
  bos_token_id: 2

  # MoE Settings
  num_experts: 8                   # 4, 8, 16, 32
  num_experts_per_token: 2         # Top-k routing
  router_type: mixtral             # mixtral, deepseek, switch
  capacity_factor: 1.25            # 1.0-2.0

  # Performance
  use_flash_attention: true        # [RECOMMENDED] 40% memory savings
  gradient_checkpointing: true     # [RECOMMENDED] 70% memory savings
  use_grouped_gemm: true           # [RECOMMENDED] 15-25% speedup
  use_triton_kernels: true         # [RECOMMENDED] 2-3x speedup

  # Regularization
  attention_dropout: 0.0           # 0.0-0.1
  dropout: 0.0
  label_smoothing: 0.0             # 0.0-0.1

  # MoE Auxiliary Losses (sum < 10% of CE loss)
  router_z_loss_coef: 0.0001
  load_balance_loss_coef: 0.01
  diversity_loss_coef: 0.0001
```

### Training Parameters

```yaml
training:
  # Batch Size
  batching:
    batch_size: 32                 # Per-GPU batch size
    gradient_accumulation_steps: 4 # Effective = batch_size × steps

  # Optimizer
  optimizer:
    type: adamw                    # adamw, adam, sgd, lion
    learning_rate: 0.0001
    betas: [0.9, 0.95]
    weight_decay: 0.01
    max_grad_norm: 1.0
    use_fused: true                # 5-10% speedup

  # Schedule
  schedule:
    num_epochs: 3
    warmup_steps: 2000
    scheduler_type: cosine         # cosine, linear, constant
    min_lr: 0.0

  # Precision
  precision:
    mixed_precision: bf16          # fp32, fp16, bf16

  # Logging
  logging:
    save_steps: 500
    eval_steps: 500
    logging_steps: 10
    use_wandb: true

  # Validation
  validation:
    enabled: true
    batch_size: 16
    max_batches: 20

  # Generation During Training
  generation:
    enabled: true
    generate_every_n_steps: 1000
    temperature: 0.8
    top_p: 0.9
    repetition_penalty: 1.2
```

### Data Pipeline

```yaml
data:
  # Source
  data_dir: code/data/processed
  max_length: 512
  max_samples: null                # null = all

  # Workers
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true

  # Sequence Packing (20-35% speedup)
  use_sequence_packing: false
  packing_strategy: greedy         # greedy, adaptive
  packing_target_ratio: 0.95

  # Randomization
  shuffle_seed: null               # null = random each run
  enable_length_sorting: true
  enable_bucketing: true

  # Splits
  train_split: train
  eval_split: validation
  validation_split_ratio: 0.1
```

### Hardware & Performance

```yaml
hardware:
  device: cuda
  mixed_precision: bf16            # fp32, fp16, bf16
  num_gpus: 1

performance:
  enable_tf32: true                # [RECOMMENDED] Ampere+
  enable_cudnn_benchmark: true     # [RECOMMENDED]
  float32_matmul_precision: high   # highest, high, medium

optimizations:
  memory_headroom_gb: 3.0
  memory_cleanup_thresholds:
    warning: 0.85
    critical: 0.90
    emergency: 0.95

  expert_prefetch:
    enabled: true
    lookahead: 2

  checkpoint:
    async_saving: true
```

### Distributed Training

```yaml
# DDP/FSDP (via torchrun)
hardware:
  num_gpus: 4
  use_gpu_load_balancing: true
  balancing_strategy: adaptive     # round_robin, memory_aware, compute_aware, adaptive

# DeepSpeed
deepspeed:
  enabled: false
  zero_stage: 2                    # 0, 1, 2, 3
  cpu_offload: false
  precision_type: bf16
  gradient_accumulation_steps: 1
  activation_checkpointing: false

# Distributed Barrier Optimization (nanoGPT-style zero-barrier training)
# Eliminates 20-200ms/epoch overhead from distributed barriers
training:
  distributed:
    minimize_distributed_barriers: true  # Skip epoch barriers (default: true)
    sync_loss_across_ranks: false        # Log local loss only (default: false)
```

### Logging & Monitoring

```yaml
wandb:
  enabled: true
  project: Ava
  tags: [moe, training]
  log_freq: 10

logging:
  tensorboard:
    enabled: false

  # Logging intervals (control when logs are emitted)
  log_interval: 100               # Main logging interval (GPU->CPU sync)
  verbose_log_interval: 500       # Detailed INFO logs every N steps (0 = never)
  tqdm_update_interval: 10        # Progress bar update frequency
  log_mode: tqdm                  # 'tqdm' or 'verbose'

  # Monitoring frequencies (control how often metrics are computed)
  metrics_log_freq: 500           # Metrics logging frequency
  memory_check_freq: 2000         # Memory check frequency
  health_summary_freq: 500        # Health summary frequency
  moe_metrics_freq: 5000          # MoE metrics frequency

  # Feature flags
  enable_timing_breakdown: true
  enable_memory_profiling: true

diagnostics:
  enabled: false
  enable_per_layer_gradients: false
  enable_routing_diagnostics: false
  enable_memory_breakdown: false
```

### Advanced Features

```yaml
# Progressive Training (Curriculum Learning)
progressive:
  enable_progressive_training: false
  enable_sequence_scaling: false
  initial_seq_length: 128
  final_seq_length: 2048
  length_schedule: linear          # linear, exponential, step

# FP8 Training (H100, L40, RTX 4090)
fp8:
  enabled: false
  format: e4m3                     # e4m3, e5m2

# Hybrid Caching
hybrid_caching:
  enabled: false
  activation_cache:
    max_size_gb: 2.0
  kv_cache:
    max_size_gb: 4.0

# Batch Size Calibration
batch_size_calibration:
  enabled: false
  target_memory: 0.85
  max_batch_size: 128

# RLHF
rlhf:
  policy_model_path: null
  judge_model_path: null
  use_model_to_model_reward: true
  ppo:
    learning_rate: 1.0e-6
    ppo_epochs: 4
    clip_range: 0.2
```

## Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (for GPU)
- **Key Libraries**: transformers, datasets, wandb, deepspeed (optional)

## Documentation

Comprehensive guides in `code/docs/`:
- `01_GETTING_STARTED.md` - Quick start guide
- `02_ARCHITECTURE.md` - System design
- `03_TRAINING_GUIDE.md` - Training procedures
- `04_CONFIGURATION.md` - Config reference
- `05_DATA_PIPELINE.md` - Data loading system
- `06_DISTRIBUTED.md` - Distributed training
- `07_RLHF.md` - RLHF training guide
- `08_API_REFERENCE.md` - API documentation
- `09_TROUBLESHOOTING.md` - Common issues
- `10_PERFORMANCE.md` - Performance optimization
