# Ava LLM Training Framework

Advanced LLM training framework with Mixture of Experts (MoE++), optimized for high performance and memory efficiency.

## Quick Start

```bash
# Main training (single GPU)
python code/scripts/5_training/train_100m_full.py --config code/configs/moe/large.yaml

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 code/scripts/5_training/train_100m_full.py --config code/configs/moe/4x_a6000_max_speed.yaml

# Fine-tuning from checkpoint
python code/scripts/5_training/finetune.py

# RLHF training
python code/scripts/6_rhlf_Finetuning/train_rlhf.py

# Data download
python code/scripts/1_data_download/unified_download.py
```

## Project Structure

```
/project/
├── .claude/                        # Claude Code configuration
├── .git/                           # Git repository
├── .vscode/                        # VS Code settings
│
├── code/                           # Main codebase
│   ├── src/                        # Source code
│   │   ├── Ava/                    # Core framework package
│   │   │   ├── config/             # Configuration management
│   │   │   │   ├── training_config.py   # Main config dataclasses (110+ params)
│   │   │   │   ├── yaml_loader.py       # YAML parsing & validation
│   │   │   │   └── constants.py         # DataPipelineConstants, TrainerConstants
│   │   │   │
│   │   │   ├── data/               # Data loading & processing
│   │   │   │   ├── dataloader.py        # StreamingDataset, DynamicTokenBatcher
│   │   │   │   ├── pretokenized_loader.py    # 60x faster Arrow loading
│   │   │   │   ├── dynamic_batch_iterator.py # Memory-aware batch sizing
│   │   │   │   ├── multi_column_data.py      # Multi-modal data support
│   │   │   │   └── conversation_turn_loader.py # Turn-aware dialogue loading
│   │   │   │
│   │   │   ├── layers/             # Neural network components
│   │   │   │   ├── experts.py           # HighPerformanceExpert, ExpertParallelGroup
│   │   │   │   └── routing.py           # MixtralRouter, DeepSeekRouter
│   │   │   │
│   │   │   ├── models/             # Model architectures
│   │   │   │   ├── moe_model.py         # EnhancedMoEModel (61KB, main model)
│   │   │   │   └── moe_layer.py         # SparseMoELayer (32KB)
│   │   │   │
│   │   │   ├── optimization/       # Optimizer & LR management
│   │   │   │   └── learning_rate/
│   │   │   │       └── managers.py      # AdaptiveLearningRateManager
│   │   │   │
│   │   │   ├── training/           # Training pipeline
│   │   │   │   ├── train/          # Core trainer
│   │   │   │   │   ├── trainer.py       # SimplifiedEnhancedTrainer
│   │   │   │   │   ├── base.py          # TrainingContext, TrainingComponent
│   │   │   │   │   └── data_loader_manager.py
│   │   │   │   │
│   │   │   │   ├── optimizations/  # Training speedups
│   │   │   │   │   ├── dynamic_batching.py       # 15-25% throughput gain
│   │   │   │   │   ├── overlapped_recomputation.py
│   │   │   │   │   ├── double_checkpointing.py   # 10x longer sequences
│   │   │   │   │   └── hybrid_cache.py           # 2.19x throughput
│   │   │   │   │
│   │   │   │   ├── strategies/     # Training strategies
│   │   │   │   │   └── progressive_training.py   # Curriculum learning
│   │   │   │   │
│   │   │   │   └── orchestration/  # Run management
│   │   │   │       └── run_manager.py    # Experiment tracking, logging
│   │   │   │
│   │   │   └── utils/              # Utilities
│   │   │       └── paths.py             # Path management
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
│   │   ├── 1_data_download/        # Data downloading
│   │   │   └── unified_download.py      # Multi-dataset downloader
│   │   │
│   │   ├── 5_training/             # Training scripts
│   │   │   ├── train_100m_full.py       # Main training script
│   │   │   ├── finetune.py              # Fine-tuning script
│   │   │   └── wandb/                   # WandB logs
│   │   │
│   │   ├── 6_rhlf_Finetuning/      # RLHF training
│   │   │   └── train_rlhf.py            # RLHF training script
│   │   │
│   │   └── 7_generation/           # Text generation
│   │       └── generate.py              # Generation interface
│   │
│   ├── configs/                    # Configuration files
│   │   ├── moe/                    # MoE configurations
│   │   │   ├── large.yaml               # 200M+ params, production
│   │   │   ├── minimal_working.yaml     # 62M params, 24GB GPU
│   │   │   ├── 4x_a6000_max_speed.yaml  # Multi-GPU optimized
│   │   │   ├── finetune_from_checkpoint.yaml
│   │   │   └── optimized_batching.yaml
│   │   │
│   │   ├── distributed/            # Distributed training
│   │   │   ├── deepspeed_zero1.yaml
│   │   │   ├── deepspeed_zero2.yaml
│   │   │   └── deepspeed_zero3.yaml
│   │   │
│   │   ├── memory/                 # Memory optimization
│   │   │   └── galore_memory_efficient.yaml
│   │   │
│   │   └── examples/               # Example configs
│   │       └── complete_config_reference.yaml  # All 200+ parameters
│   │
│   ├── data/                       # Training data (local)
│   │   ├── fine-tuning/            # Fine-tuning datasets
│   │   │   ├── CodeAlpaca/arrow/
│   │   │   ├── OpenAssistant/arrow/
│   │   │   └── OpenOrca/arrow/
│   │   └── Ava_Ai/                 # Tokenizer & processed data
│   │       ├── tokenizer/
│   │       └── data/train/
│   │
│   ├── outputs/                    # Training outputs
│   │   ├── runs/                   # Training runs
│   │   │   └── run_YYYYMMDD_HHMMSS/
│   │   │       ├── checkpoints/         # Model checkpoints
│   │   │       ├── configs/             # Saved configs
│   │   │       └── logs/                # Training logs
│   │   │
│   │   └── finetune_runs/          # Fine-tuning runs
│   │       └── finetune_run_*/
│   │
│   ├── docs/                       # Documentation (30+ guides)
│   │   ├── 01_ARCHITECTURE.md
│   │   ├── 02_TRAINING_GUIDE.md
│   │   ├── 03_MEMORY_OPTIMIZATION.md
│   │   ├── 05_OPTIMIZATION_GUIDE.md
│   │   ├── 07_CONFIGURATION_SYSTEM.md
│   │   └── TURN_AWARE_DATA_LOADING.md
│   │
│   └── tests/                      # Test suite
│       ├── test_all_codebase_features.py
│       └── test_dynamic_batch_iterator.py
│
├── data/                           # Root data directory
├── models/                         # Root model checkpoints
├── logs/                           # Root training logs
├── claude_docs/                    # Claude documentation
│
├── CLAUDE.md                       # This file (Claude Code reference)
├── Claude.md                       # Changelog (98KB)
├── requirements.txt                # Python dependencies
├── apt.txt                         # System dependencies
└── find_dependencies.py            # Dependency analysis
```

## Configuration Files

| Config | Model Size | GPU Memory | Use Case |
|--------|-----------|------------|----------|
| `configs/moe/minimal_working.yaml` | 62M params | 24GB | Development, testing |
| `configs/moe/large.yaml` | 200M+ params | 40GB+ | Production training |
| `configs/moe/4x_a6000_max_speed.yaml` | 200M+ params | 4x 48GB | Multi-GPU maximum speed |
| `configs/moe/finetune_from_checkpoint.yaml` | Variable | Variable | Fine-tuning from saved model |
| `configs/moe/optimized_batching.yaml` | Variable | Variable | Sequence packing optimization |

## Key Features

### Model Architecture
- **MoE++ Architecture**: 8-16 experts with top-k routing
- **Router Types**: Mixtral (top-k), DeepSeek (hybrid shared + routed)
- **Attention**: Multi-Query (MQA), Grouped Query (GQA), Flash Attention v2+
- **Embeddings**: Rotary Position Embeddings (RoPE) with scaling
- **Activations**: SwiGLU/GeGLU gated activations
- **Auxiliary Losses**: Load balancing, router z-loss, diversity loss

### Training Optimizations
- **Dynamic Batching**: Memory-aware batch sizing (15-25% throughput improvement)
- **Gradient Checkpointing**: 70-80% memory reduction
- **Mixed Precision**: FP16/BF16/FP8 support
- **Double Checkpointing**: O(n) to O(sqrt(n)) memory for 10x longer sequences
- **Hybrid Caching**: KV + activation caching (2.19x throughput)
- **Overlapped Recomputation**: Parallel backward pass
- **Progressive Training**: Curriculum learning, sequence length scaling

### Data Loading
- **Pre-tokenized Loading**: 60x speedup with memory-mapped Arrow files
- **Turn-Aware Conversation**: Preserves dialogue structure with speaker markers
- **Dynamic Batch Iterator**: Variable batch sizes based on GPU memory
- **Multi-Column Datasets**: Text, numeric, categorical, image, tensor support
- **HuggingFace Integration**: Direct loading from HF Hub
- **Streaming**: Memory-efficient large dataset processing

### Distributed Training
- **DDP**: Data Distributed Parallel for multi-GPU
- **FSDP**: Fully Sharded Data Parallel for large models
- **DeepSpeed**: ZeRO-1/2/3 optimizer sharding
- **Expert Parallelism**: Distributed MoE experts across GPUs

## Key Files Reference

| File | Purpose |
|------|---------|
| `code/src/Ava/models/moe_model.py` | Main MoE transformer model, EnhancedMoEConfig |
| `code/src/Ava/models/moe_layer.py` | SparseMoELayer, routing strategies, auxiliary losses |
| `code/src/Ava/layers/routing.py` | MixtralRouter, DeepSeekRouter, UnifiedMoERouter |
| `code/src/Ava/layers/experts.py` | HighPerformanceExpert, ExpertParallelGroup |
| `code/src/Ava/training/train/trainer.py` | SimplifiedEnhancedTrainer main loop |
| `code/src/Ava/config/training_config.py` | All configuration dataclasses |
| `code/src/Ava/config/constants.py` | DataPipelineConstants, TrainerConstants, MoEConstants |
| `code/src/Ava/data/dataloader.py` | StreamingDataset, DynamicTokenBatcher |
| `code/src/Ava/data/pretokenized_loader.py` | PreTokenizedDataset, ultra-fast loading |
| `code/src/Ava/data/dynamic_batch_iterator.py` | DynamicBatchIterator for memory-aware batching |
| `code/src/Ava/optimization/learning_rate/managers.py` | AdaptiveLearningRateManager |

## Common Tasks

### Training
```bash
# Single GPU with large config
python code/scripts/5_training/train_100m_full.py --config code/configs/moe/large.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/4x_a6000_max_speed.yaml

# Resume from checkpoint
python code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/large.yaml \
  --resume /path/to/checkpoint.pt
```

### Fine-tuning
```bash
# Auto-discovers latest checkpoint and data
python code/scripts/5_training/finetune.py

# With specific checkpoint
python code/scripts/5_training/finetune.py --checkpoint /path/to/model.pt
```

### Data Download
```bash
# Download multiple datasets
python code/scripts/1_data_download/unified_download.py
```

### Testing
```bash
# Run comprehensive tests
python code/tests/test_all_codebase_features.py

# Test dynamic batching
python code/tests/test_dynamic_batch_iterator.py
```

### Checking Outputs
- **WandB Logs**: `code/scripts/5_training/wandb/`
- **Training Outputs**: `code/outputs/runs/`
- **Checkpoints**: `code/outputs/runs/*/checkpoints/`
- **Metrics**: `code/outputs/runs/*/metrics/`

## Performance Benchmarks

### Throughput
| Configuration | Hardware | Samples/sec |
|--------------|----------|-------------|
| Minimal (62M) | Single 24GB GPU | 800-1,200 |
| Large (200M) | Single 40GB GPU | 1,600-3,000 |
| Multi-GPU | 4x A6000 (192GB) | 15,000-25,000 |

### Memory Savings
| Optimization | Memory Reduction |
|--------------|-----------------|
| Gradient Checkpointing | 70-80% |
| Flash Attention | 40% |
| Mixed Precision (BF16) | 50% |
| Double Checkpointing | O(n) to O(sqrt(n)) |

### Speedups
| Feature | Improvement |
|---------|-------------|
| Pre-tokenized Loading | 60x faster |
| Dynamic Batching | 15-25% throughput |
| Hybrid Caching | 2.19x throughput |
| Grouped GEMM Experts | 5-10x expert computation |
| Progressive Training | 30-50% faster convergence |

## Configuration Options

### Model Architecture (`architecture:`)
```yaml
architecture:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_attention_heads: 16
  intermediate_size: 4096
  num_experts: 8
  num_experts_per_token: 2
  router_type: 'mixtral'  # or 'deepseek', 'switch'
  use_flash_attention: true
  use_rotary_embeddings: true
  activation: 'swiglu'
```

### Training (`training:`)
```yaml
training:
  batch_size: 128
  gradient_accumulation_steps: 4
  learning_rate: 0.0006
  warmup_steps: 1000
  num_epochs: 5
  max_steps: null
  weight_decay: 0.01
  max_gradient_norm: 1.0
  mixed_precision: 'bf16'
  save_steps: 1000
  eval_steps: 500
  logging_steps: 10
```

### Dynamic Batching (`dynamic_batching:`)
```yaml
dynamic_batching:
  enabled: true
  min_batch_size: 32
  max_batch_size: 512
  memory_thresholds:
    low_memory_threshold: 0.5
    target_memory_threshold: 0.7
    high_memory_threshold: 0.85
    critical_memory_threshold: 0.95
```

### Optimizations (`optimizations:`)
```yaml
optimizations:
  gradient_checkpointing: true
  torchinductor_autotune: 1
  memory_headroom_gb: 1
  checkpoint:
    async_saving: true
  router:
    compile_routers: true
    enable_router_caching: true
```

## Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (optional, for GPU)
- **Key Libraries**: transformers, datasets, wandb, deepspeed (optional)

## Documentation

Comprehensive guides in `code/docs/`:
- `01_ARCHITECTURE.md` - System design overview
- `02_TRAINING_GUIDE.md` - Training procedures
- `03_MEMORY_OPTIMIZATION.md` - Memory efficiency techniques
- `05_OPTIMIZATION_GUIDE.md` - Optimization strategies
- `07_CONFIGURATION_SYSTEM.md` - Config system reference
- `TURN_AWARE_DATA_LOADING.md` - Conversation data handling
