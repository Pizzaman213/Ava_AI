# Ava LLM Training Framework

Advanced LLM training framework with Mixture of Experts (MoE++) architecture, optimized for memory efficiency and scalability.

## Quick Start

```bash
# Main training (single GPU)
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU training
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py --config code/configs/moe/large_Multy.yaml

# Fine-tuning from checkpoint
python code/scripts/5_training/finetune.py

# RLHF training
python code/scripts/6_rhlf_Finetuning/train_rlhf.py

# Data download
python code/scripts/1_data_download/unified_download.py

# Text generation
python code/scripts/7_generation/generate.py --prompt "Once upon a time"

# Check dependencies
python code/scripts/check_dependencies.py
```

## Project Structure

```
/root/Ava_AI/
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
│   │   │   │   ├── bucketing.py         # LengthBasedBucketing
│   │   │   │   ├── factory.py           # create_streaming_dataloaders
│   │   │   │   ├── pretokenized.py      # Memory-mapped Arrow loading
│   │   │   │   ├── multi_column.py      # Multi-modal data support
│   │   │   │   ├── conversation.py      # Turn-aware dialogue loading
│   │   │   │   ├── packing.py           # Sequence packing
│   │   │   │   ├── validation.py        # Data validation utilities
│   │   │   │   └── dataloader.py        # DataLoader utilities
│   │   │   │
│   │   │   ├── nn/                 # Neural network layers
│   │   │   │   ├── experts.py           # HighPerformanceExpert, ExpertParallelGroup
│   │   │   │   └── routing.py           # MixtralRouter, DeepSeekRouter, UnifiedMoERouter
│   │   │   │
│   │   │   ├── kernels/            # Triton kernels
│   │   │   │   ├── moe.py               # Fused gating/topk kernels
│   │   │   │   ├── activations.py       # Fused SwiGLU/GeGLU
│   │   │   │   └── fused_experts.py     # Fused expert computation
│   │   │   │
│   │   │   ├── models/             # Model architectures
│   │   │   │   ├── moe.py               # EnhancedMoEModel, EnhancedMoEConfig
│   │   │   │   └── moe_layer.py         # SparseMoELayer
│   │   │   │
│   │   │   ├── optim/              # Optimizers & LR scheduling
│   │   │   │   └── lr_managers.py       # AdaptiveLearningRateManager
│   │   │   │
│   │   │   ├── training/           # Training pipeline
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
│   │   │   ├── optimizations/      # Training optimizations
│   │   │   │   ├── checkpointing.py     # Gradient checkpointing
│   │   │   │   ├── overlapped_recomputation.py  # Parallel backward pass
│   │   │   │   ├── hybrid_cache.py      # KV + activation caching
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
│   │       ├── rlhf_trainer.py          # RLHF training loop
│   │       ├── ppo_trainer.py           # PPO implementation
│   │       └── reward_model.py          # Reward model
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
│   │   │   ├── prepare_prompts.py       # Prompt preparation
│   │   │   ├── test_rlhf_cpu.py         # RLHF CPU testing
│   │   │   └── test_training_cpu.py     # Training CPU testing
│   │   │
│   │   ├── 7_generation/           # Text generation
│   │   │   └── generate.py              # Generation interface
│   │   │
│   │   └── check_dependencies.py   # Dependency checker
│   │
│   ├── configs/                    # Configuration files
│   │   ├── moe/                    # MoE configurations
│   │   │   ├── large.yaml               # Production config (200M+ params)
│   │   │   ├── minimal_working.yaml     # Testing/development (62M params)
│   │   │   ├── large_Multy.yaml         # Multi-GPU configuration
│   │   │   ├── mw_8bit.yaml             # 8-bit quantized training
│   │   │   ├── max_randomness.yaml      # Maximum data randomization
│   │   │   └── deterministic.yaml       # Deterministic training
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
│   │   └── pretraining/            # Pre-training runs
│   │
│   ├── docs/                       # Documentation (28 guides)
│   │
│   └── tests/                      # Test suite
│       ├── test_coherence.py
│       ├── test_data_loader_fixes.py
│       ├── test_kernel_diagnostics.py
│       ├── test_triton_softmax_topk.py
│       └── test_triton_topk_fix.py
│
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
| `moe/large.yaml` | Production training (200M+ params) |
| `moe/minimal_working.yaml` | Development and testing (62M params) |
| `moe/large_Multy.yaml` | Multi-GPU distributed training |
| `moe/mw_8bit.yaml` | 8-bit quantized training |
| `moe/max_randomness.yaml` | Maximum data randomization |
| `moe/deterministic.yaml` | Reproducible deterministic training |
| `distributed/deepspeed_zero*.yaml` | DeepSpeed ZeRO configurations |
| `memory/memory_efficient_optimizers.yaml` | Memory-optimized optimizer settings |

## Key Features

### Model Architecture
- **MoE++ Architecture**: 8-32 experts with configurable top-k routing
- **Router Types**: Mixtral, DeepSeek (hybrid), Switch
- **Attention**: Flash Attention v2+, Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- **Embeddings**: Rotary Position Embeddings (RoPE)
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
- **Parallel Data Loading**: 8 workers for optimal throughput
- **Gradient Checkpointing**: Memory reduction for longer sequences
- **Mixed Precision**: FP16/BF16/FP8 support
- **Double Checkpointing**: O(sqrt(n)) memory for extended sequences
- **Hybrid Caching**: KV + activation caching
- **Overlapped Recomputation**: Parallel backward pass computation
- **Progressive Training**: Curriculum learning, sequence length scaling
- **Grouped GEMM**: Efficient expert computation

### Data Loading
- **Pre-tokenized Loading**: Memory-mapped Arrow files
- **Turn-Aware Conversation**: Preserves dialogue structure
- **Sequence Packing**: Efficient utilization by eliminating padding
- **Multi-Column Datasets**: Text, numeric, categorical, image, tensor support
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
| `code/src/ava/data/bucketing.py` | LengthBasedBucketing |
| `code/src/ava/data/pretokenized.py` | PreTokenizedDataset |
| `code/src/ava/optim/lr_managers.py` | AdaptiveLearningRateManager |
| `code/src/ava/training/loop.py` | TrainingLoopManager |
| `code/src/ava/training/pipeline.py` | TrainingPipeline |

## Common Tasks

### Training
```bash
# Single GPU
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml

# Multi-GPU
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
  --config code/configs/moe/large_Multy.yaml

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
python code/tests/test_coherence.py
python code/tests/test_kernel_diagnostics.py
```

## Outputs Location
- **Training Runs**: `code/outputs/pretraining/run_YYYYMMDD_HHMMSS/`
- **Checkpoints**: `code/outputs/pretraining/*/checkpoints/`
- **WandB Logs**: `code/scripts/5_training/wandb/`

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

### Data Loading
```yaml
data:
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
  use_pretokenized: true
```

### Data Randomization
```yaml
data:
  randomization:
    shuffle_buffer_size: 10000
    seed: null  # null for random seed each run
    deterministic: false
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
- `05_OPTIMIZATION_GUIDE.md` - Optimization strategies
- `07_CONFIGURATION_SYSTEM.md` - Config reference
- `TURN_AWARE_DATA_LOADING.md` - Conversation handling
- `AVA_MODEL_API.md` - Model API reference
