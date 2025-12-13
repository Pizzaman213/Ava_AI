# Ava LLM Training Framework

Advanced LLM training framework with Mixture of Experts (MoE++), optimized for high performance and memory efficiency.

## Quick Start

```bash
# Main training (single GPU)
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config code/configs/moe/4x_a6000_max_speed.yaml

# Fine-tuning from checkpoint
python code/scripts/5_training/finetune.py

# RLHF training
python code/scripts/6_rhlf_Finetuning/train_rlhf.py

# Data download
python code/scripts/1_data_download/unified_download.py

# Text generation
python code/scripts/7_generation/generate.py --prompt "Once upon a time"
```

## Project Structure

```
/project/
├── code/                           # Main codebase
│   ├── src/                        # Source code
│   │   ├── ava/                    # Core framework package (lowercase, PEP8)
│   │   │   ├── config/             # Configuration management
│   │   │   │   ├── training_config.py   # Main config dataclasses (110+ params)
│   │   │   │   ├── yaml_loader.py       # YAML parsing & validation
│   │   │   │   ├── validator.py         # ConfigValidator
│   │   │   │   └── constants.py         # DataPipelineConstants, TrainerConstants
│   │   │   │
│   │   │   ├── core/               # Core utilities
│   │   │   │   ├── paths.py             # Path management
│   │   │   │   ├── checkpoint.py        # Async checkpoint saving
│   │   │   │   ├── mixed_precision.py   # AMP training utilities
│   │   │   │   ├── data_utils.py        # Collation, data helpers
│   │   │   │   ├── activations.py       # Activation factory
│   │   │   │   ├── logging.py           # Colored logging
│   │   │   │   └── script_utils.py      # Script helpers
│   │   │   │
│   │   │   ├── cuda/               # CUDA utilities
│   │   │   │   ├── streams.py           # StreamPool, CUDATimer
│   │   │   │   ├── buffers.py           # Pinned buffer management
│   │   │   │   ├── metrics.py           # Async metrics tracking
│   │   │   │   └── profiler.py          # Nsight profiling
│   │   │   │
│   │   │   ├── data/               # Data loading & processing
│   │   │   │   ├── streaming.py         # StreamingDataset
│   │   │   │   ├── distributed.py       # DistributedStreamingDataset
│   │   │   │   ├── bucketing.py         # DynamicTokenBatcher, LengthBasedBucketing
│   │   │   │   ├── factory.py           # create_streaming_dataloaders
│   │   │   │   ├── pretokenized.py      # 60x faster Arrow loading
│   │   │   │   ├── batch_iterator.py    # DynamicBatchIterator
│   │   │   │   ├── multi_column.py      # Multi-modal data support
│   │   │   │   ├── conversation.py      # Turn-aware dialogue loading
│   │   │   │   └── packing.py           # Sequence packing
│   │   │   │
│   │   │   ├── nn/                 # Neural network layers
│   │   │   │   ├── experts.py           # HighPerformanceExpert, ExpertParallelGroup
│   │   │   │   └── routing.py           # MixtralRouter, DeepSeekRouter, UnifiedMoERouter
│   │   │   │
│   │   │   ├── kernels/            # Triton kernels
│   │   │   │   ├── moe.py               # Fused gating/topk kernels
│   │   │   │   └── activations.py       # Fused SwiGLU/GeGLU
│   │   │   │
│   │   │   ├── models/             # Model architectures
│   │   │   │   ├── moe.py               # EnhancedMoEModel, EnhancedMoEConfig
│   │   │   │   └── moe_layer.py         # SparseMoELayer
│   │   │   │
│   │   │   ├── optim/              # Optimizers & LR scheduling
│   │   │   │   └── lr_managers.py       # AdaptiveLearningRateManager
│   │   │   │
│   │   │   ├── training/           # Training pipeline (flattened)
│   │   │   │   ├── context.py           # TrainingContext, TrainingComponent
│   │   │   │   ├── loop.py              # TrainingLoopManager
│   │   │   │   ├── data_manager.py      # DataLoaderManager
│   │   │   │   ├── model_builder.py     # ModelBuilder
│   │   │   │   ├── optimizer.py         # OptimizerManager
│   │   │   │   ├── validation.py        # ValidationManager
│   │   │   │   ├── generation.py        # GenerationManager
│   │   │   │   ├── metrics.py           # MetricsManager
│   │   │   │   ├── run_manager.py       # Experiment tracking, logging
│   │   │   │   ├── pipeline.py          # TrainingPipeline
│   │   │   │   └── distributed.py       # DDP/FSDP setup
│   │   │   │
│   │   │   ├── optimizations/      # Training speedups
│   │   │   │   ├── dynamic_batching.py  # 15-25% throughput gain
│   │   │   │   ├── checkpointing.py     # Double checkpointing (10x sequences)
│   │   │   │   ├── overlapped_recomputation.py
│   │   │   │   ├── hybrid_cache.py      # 2.19x throughput
│   │   │   │   ├── fp8.py               # FP8 quantization
│   │   │   │   ├── batch_controller.py  # Batch size control
│   │   │   │   ├── prefetch.py          # Async batch prefetching
│   │   │   │   ├── gradients.py         # Gradient utilities
│   │   │   │   └── quantization.py      # Quantization helpers
│   │   │   │
│   │   │   ├── strategies/         # Training strategies
│   │   │   │   └── progressive.py       # Curriculum learning
│   │   │   │
│   │   │   └── eval/               # Model evaluation
│   │   │       └── coherence.py         # Coherence metrics
│   │   │
│   │   ├── generation/             # Text generation
│   │   │   └── generator.py             # Generation utilities
│   │   │
│   │   └── rlhf/                   # RLHF training
│   │       ├── trainer.py               # RLHF training loop
│   │       ├── ppo.py                   # PPO implementation
│   │       └── reward.py                # Reward model
│   │
│   ├── scripts/                    # Executable scripts
│   │   ├── 1_data_download/        # Data downloading
│   │   │   ├── unified_download.py      # Multi-dataset downloader
│   │   │   └── download_openorca.py     # OpenOrca downloader
│   │   │
│   │   ├── 5_training/             # Training scripts
│   │   │   ├── train_pipeline.py        # Main training script
│   │   │   ├── finetune.py              # Auto-discovery fine-tuning
│   │   │   └── wandb/                   # WandB logs
│   │   │
│   │   ├── 6_rhlf_Finetuning/      # RLHF training
│   │   │   ├── train_rlhf.py            # RLHF training script
│   │   │   └── prepare_prompts.py       # Prompt preparation
│   │   │
│   │   └── 7_generation/           # Text generation
│   │       └── generate.py              # Generation interface
│   │
│   ├── configs/                    # Configuration files
│   │   ├── moe/                    # MoE configurations
│   │   │   ├── large.yaml               # 200M+ params, production
│   │   │   ├── minimal_working.yaml     # 62M params, testing
│   │   │   ├── 4x_a6000_max_speed.yaml  # Multi-GPU optimized
│   │   │   ├── finetune_from_checkpoint.yaml
│   │   │   └── optimized_batching.yaml
│   │   │
│   │   ├── distributed/            # Distributed training
│   │   │   ├── deepspeed_zero1.yaml
│   │   │   ├── deepspeed_zero2.yaml
│   │   │   └── deepspeed_zero3.yaml
│   │   │
│   │   └── memory/                 # Memory optimization
│   │       └── memory_efficient_optimizers.yaml
│   │
│   ├── data/                       # Training data (local)
│   │   ├── fine-tuning/            # Fine-tuning datasets (Arrow format)
│   │   └── Ava_Ai/                 # Tokenizer & processed data
│   │
│   ├── outputs/                    # Training outputs
│   │   ├── runs/                   # Training runs
│   │   └── finetune_runs/          # Fine-tuning runs
│   │
│   ├── docs/                       # Documentation (50+ guides)
│   │
│   └── tests/                      # Test suite
│       ├── test_dynamic_batch_iterator.py
│       └── test_coherence.py
│
├── data/                           # Root data directory
├── models/                         # Root model checkpoints
├── wandb/                          # WandB tracking (152+ runs)
├── CLAUDE.md                       # This file
├── requirements.txt                # Python dependencies
└── apt.txt                         # System dependencies
```

## Configuration Files

| Config | Model Size | GPU Memory | Use Case |
|--------|-----------|------------|----------|
| `moe/minimal_working.yaml` | 62M params | 11GB | Development, testing |
| `moe/large.yaml` | 200M+ params | 24GB | Production training |
| `moe/4x_a6000_max_speed.yaml` | 200M+ params | 4x 48GB | Multi-GPU maximum speed |
| `moe/finetune_from_checkpoint.yaml` | Variable | Variable | Fine-tuning |
| `moe/optimized_batching.yaml` | Variable | Variable | Sequence packing |

## Key Features

### Model Architecture
- **MoE++ Architecture**: 8-32 experts with top-k routing
- **Router Types**: Mixtral, DeepSeek (hybrid), Switch
- **Attention**: Flash Attention v2+, MQA, GQA
- **Embeddings**: Rotary Position Embeddings (RoPE)
- **Activations**: SwiGLU/GeGLU gated activations
- **Auxiliary Losses**: Load balancing, router z-loss, diversity loss

### Training Optimizations
- **Dynamic Batching**: Memory-aware batch sizing (15-25% throughput gain)
- **Gradient Checkpointing**: 70-80% memory reduction
- **Mixed Precision**: FP16/BF16/FP8 support
- **Double Checkpointing**: O(sqrt(n)) memory for 10x longer sequences
- **Hybrid Caching**: KV + activation caching (2.19x throughput)
- **Overlapped Recomputation**: Parallel backward pass
- **Progressive Training**: Curriculum learning, sequence length scaling
- **Grouped GEMM**: 5-10x expert computation speedup

### Data Loading
- **Pre-tokenized Loading**: 60x speedup with memory-mapped Arrow files
- **Turn-Aware Conversation**: Preserves dialogue structure
- **Dynamic Batch Iterator**: Variable batch sizes based on GPU memory
- **Sequence Packing**: 20-35% speedup by eliminating padding
- **Multi-Column Datasets**: Text, numeric, categorical, image, tensor
- **Streaming**: Memory-efficient large dataset processing

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
| `code/src/ava/models/moe_layer.py` | SparseMoELayer, routing strategies |
| `code/src/ava/nn/routing.py` | MixtralRouter, DeepSeekRouter, UnifiedMoERouter |
| `code/src/ava/nn/experts.py` | HighPerformanceExpert, ExpertParallelGroup |
| `code/src/ava/config/training_config.py` | All configuration dataclasses |
| `code/src/ava/config/constants.py` | DataPipelineConstants, TrainerConstants |
| `code/src/ava/data/streaming.py` | StreamingDataset |
| `code/src/ava/data/bucketing.py` | DynamicTokenBatcher, LengthBasedBucketing |
| `code/src/ava/data/pretokenized.py` | PreTokenizedDataset (60x faster) |
| `code/src/ava/data/batch_iterator.py` | DynamicBatchIterator |
| `code/src/ava/optimizations/dynamic_batching.py` | DynamicBatchScheduler |
| `code/src/ava/optim/lr_managers.py` | AdaptiveLearningRateManager |

## Common Tasks

### Training
```bash
# Single GPU
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/4x_a6000_max_speed.yaml

# Resume from checkpoint
python code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/large.yaml \
  --resume /path/to/checkpoint.pt
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
python code/scripts/6_rhlf_Finetuning/train_rlhf.py --config <config_file>
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

### Data Download
```bash
# Full download
python code/scripts/1_data_download/unified_download.py

# Limited partitions
python code/scripts/1_data_download/unified_download.py --max-partitions 10
```

### Testing
```bash
python code/tests/test_dynamic_batch_iterator.py
python code/tests/test_coherence.py
```

## Outputs Location
- **Training Runs**: `code/outputs/runs/run_YYYYMMDD_HHMMSS/`
- **Checkpoints**: `code/outputs/runs/*/checkpoints/`
- **Fine-tune Runs**: `code/outputs/finetune_runs/`
- **WandB Logs**: `code/scripts/5_training/wandb/`

## Performance Benchmarks

| Configuration | Hardware | Throughput |
|--------------|----------|------------|
| Minimal (62M) | Single 24GB GPU | 800-1,200 samples/sec |
| Large (200M) | Single 24GB GPU | 1,600-3,000 samples/sec |
| Multi-GPU | 4x A6000 | 15,000-25,000 samples/sec |

| Optimization | Improvement |
|-------------|-------------|
| Pre-tokenized Loading | 60x faster |
| Dynamic Batching | 15-25% throughput |
| Hybrid Caching | 2.19x throughput |
| Sequence Packing | 20-35% speedup |
| Grouped GEMM Experts | 5-10x computation |
| Gradient Checkpointing | 70-80% memory savings |
| Flash Attention | 40% memory savings |

## Configuration Examples

### Model Architecture
```yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_attention_heads: 16
  intermediate_size: 4096
  num_experts: 8
  num_experts_per_token: 2
  router_type: 'mixtral'  # or 'deepseek', 'switch'
  use_flash_attention: true
  activation: 'swiglu'
```

### Training
```yaml
training:
  batch_size: 128
  gradient_accumulation_steps: 4
  learning_rate: 0.0006
  warmup_steps: 1000
  num_epochs: 5
  weight_decay: 0.01
  mixed_precision: 'bf16'
```

### Dynamic Batching
```yaml
dynamic_batching:
  enabled: true
  min_batch_size: 32
  max_batch_size: 512
  target_memory_threshold: 0.7
  token_budget:
    enabled: true
    target_tokens_per_batch: 8192
```

## Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (for GPU)
- **Key Libraries**: transformers, datasets, wandb, deepspeed (optional)

## Documentation

Comprehensive guides in `code/docs/`:
- `01_ARCHITECTURE.md` - System design
- `02_TRAINING_GUIDE.md` - Training procedures
- `03_MEMORY_OPTIMIZATION.md` - Memory techniques
- `05_OPTIMIZATION_GUIDE.md` - Performance optimization
- `07_CONFIGURATION_SYSTEM.md` - Config reference
- `TURN_AWARE_DATA_LOADING.md` - Conversation handling
- `DYNAMIC_BATCHING.md` - Memory-aware batching
