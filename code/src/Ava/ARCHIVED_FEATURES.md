# Archived Features Documentation

This document explains which features have been moved to the `_archived/` directory and why. These files are not used in the default training pipeline but are preserved for future reference, experimentation, or optional integration.

## Overview

**Total archived files:** 26 Python modules
**Archive location:** `code/src/Ava/_archived/`
**Purpose:** Clean up the codebase while preserving experimental/optional features

---

## Archived Files by Category

### 1. Generation/Inference (1 file)

**Location:** `_archived/generation/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `generator.py` | Text generation with multiple decoding strategies (greedy, beam search, top-k/nucleus sampling, repetition penalty) | Not used during training; only needed for inference/generation after training completes |

**Features:**
- Temperature-based sampling
- Top-k and nucleus (top-p) filtering
- Beam search decoding
- Repetition penalty

**To re-enable:** Import from `src.Ava._archived.generation.generator`

---

### 2. Data Preparation Tools (3 files)

**Location:** `_archived/data/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `data_profiler.py` | Data profiling and quality analysis for multi-column datasets | Used for pre-training data analysis, not during the training loop |
| `deduplication.py` | Global deduplication using bloom filters and xxhash | Used for data preprocessing, not during training |
| `optimized_dataloader.py` | Alternative dataloader with memory-mapped files, persistent workers, GPU prefetching | Training uses `data_streaming.py` instead |

**Features:**
- Statistical analysis of datasets
- Memory-efficient cross-batch deduplication
- Advanced prefetching strategies

**To re-enable:** Use for data preprocessing pipelines or switch dataloader implementation

---

### 3. Evaluation/Benchmarking (1 file)

**Location:** `_archived/evaluation/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `evaluator.py` | Basic model evaluation and perplexity calculation | Training uses `comprehensive_eval.py` instead |

**Note:** The training pipeline uses `ComprehensiveEvaluator` from `comprehensive_eval.py` which provides more detailed evaluation metrics.

**To re-enable:** Use for standalone benchmarking or simple evaluation tasks

---

### 4. Experimental Optimizations (8 files)

**Location:** `_archived/optimization/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `a100_optimizer.py` | NVIDIA A100 GPU-specific optimizations (TF32, CUDA graphs, memory pools) | Optional hardware-specific optimization not required for default training |
| `flash_attention_v3.py` | FlashAttention v3 with A100 optimizations and sequence parallelism | Optional attention optimization; training uses standard attention |
| `nvlink_optimizer.py` | NVLink topology optimization for multi-GPU A100 systems with hierarchical AllReduce | Optional distributed training optimization |
| `memory_optimizer.py` | Advanced memory management and gradient checkpointing for A100 | Alternative to `memory_monitor.py` used in training |
| `fused_optimizers.py` | Fused Adam/AdamW and 8-bit Adam with bitsandbytes | Training uses `advanced_optimizers.py` instead |
| `gradient_optimizations.py` | Advanced gradient techniques (compression, adaptive clipping, noise injection) | More advanced than what training currently uses |
| `compilation_optimizations.py` | torch.compile() integration, custom fused kernels, CUDA graphs | Optional compilation optimizations |
| `hardware_optimizations.py` | Automatic hardware optimization (TF32, cuDNN autotuner) | Optional hardware-specific tuning |

**Features:**
- A100-specific CUDA optimizations
- Advanced memory management
- Gradient compression and communication optimization
- torch.compile() integration
- Hardware auto-tuning

**To re-enable:** Import specific optimizers for hardware-specific or experimental training runs

---

### 5. Advanced Layer Architectures (5 files)

**Location:** `_archived/layers/`

These are experimental layer architectures controlled by feature flags in the training config but not enabled by default.

| File | Purpose | Feature Flag | Why Archived |
|------|---------|--------------|--------------|
| `attention.py` | Enhanced multi-head attention with rotary embeddings | N/A | Basic attention layer; MoE model has built-in attention |
| `advanced_attention.py` | Flash Attention v2/v3, sliding window, MQA, GQA | `use_flash_attention` | Advanced attention variants not currently enabled |
| `mixture_of_heads.py` | Dynamic attention head selection (MoH) | `use_moh` | Experimental feature not in default training |
| `mixture_of_activations.py` | Dynamic activation function selection (MoA) | `use_moa` | Experimental feature not in default training |
| `cross_attention.py` | Multi-modal cross-attention for text/vision/audio | `use_cross_attention` | Multi-modal feature not in default training |

**Features:**
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Dynamic head/activation selection
- Multi-modal cross-attention

**To re-enable:**
1. Set the corresponding feature flag in training config
2. Import from `src.Ava._archived.layers.*`
3. Integrate into model architecture

---

### 6. Profiling/Debugging Tools (1 file)

**Location:** `_archived/training/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `profiling_tools.py` | PyTorch Profiler integration, throughput tracking, memory profiling, bottleneck identification | Optional profiling tool for performance analysis |

**Features:**
- Samples/sec, tokens/sec metrics
- Model FLOPs Utilization (MFU) calculation
- Memory profiling
- Bottleneck detection

**To re-enable:** Import for detailed performance analysis and optimization

---

### 7. Alternative Training Strategies (4 files)

**Location:** `_archived/training/`

| File | Purpose | Why Archived |
|------|---------|--------------|
| `dynamic_batch_sampler.py` | Dynamic batch size adjustment based on GPU memory utilization | `progressive_training.py` has its own batch sizing logic |
| `progressive_batch_scheduler.py` | Progressive batch size scheduling (start small, grow large) | Different from batch sizing in `progressive_training.py` |
| `qlora_utils.py` | QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning | Specialized fine-tuning approach not in base training |
| `distributed_optimizations.py` | FSDP integration, gradient communication overlap, pipeline parallelism | Alternative to `distributed_manager.py` |

**Features:**
- Memory-adaptive batch sizing
- Progressive batch growth strategies
- QLoRA for parameter-efficient fine-tuning
- FSDP and pipeline parallelism

**To re-enable:** Use for specialized training scenarios (fine-tuning, extreme memory constraints, pipeline parallelism)

---

### 8. Conditionally Used Files (2 files)

**Location:** `_archived/models/` and `_archived/losses/`

| File | Purpose | When Used | Why Archived |
|------|---------|-----------|--------------|
| `deepspeed_wrapper.py` | DeepSpeed compatibility wrapper for EnhancedMoEModel | Only when DeepSpeed is enabled | Conditional usage; not needed for all training runs |
| `vocab_parallel_loss.py` | Vocabulary-parallel loss computation | Only for tensor parallel training | Conditional usage for distributed training |

**To re-enable:** These files may be automatically imported when specific features are enabled (DeepSpeed, tensor parallelism)

---

## Files Still in Use (Core Training Pipeline)

### Config
- `training_config.py` - Training configuration
- `feature_compatibility.py` - Feature compatibility checks

### Data
- `arrow_reader.py` - Apache Arrow dataset reading
- `encoding_detector.py` - Text encoding detection
- `data_streaming.py` - Streaming dataloaders (active)
- `multi_column_data.py` - Multi-column data handling (active)

### Evaluation
- `comprehensive_eval.py` - Comprehensive evaluation (used during training)

### Layers
- `experts.py` - Expert layers for MoE
- `routing.py` - Expert routing logic

### Losses
- `advanced_losses.py` - Advanced loss functions
- `deepseek_loss.py` - DeepSeek-inspired loss

### Memory
- `episodic_memory.py` - Episodic memory for continual learning

### Models
- `moe_model.py` - Enhanced MoE model architecture

### Optimization
- `quantization.py` - Model quantization
- `advanced_optimizers.py` - Lion, Sophia, AdaFactor optimizers
- `fp8_training.py` - FP8 training support

### Training
- `adaptive_lr.py` - Adaptive learning rate management
- `advanced_schedulers.py` - Advanced LR schedulers
- `advanced_warmup.py` - Advanced warmup strategies
- `distributed_health_checker.py` - Distributed training health monitoring
- `distributed_manager.py` - Distributed training coordination
- `enhanced_trainer.py` - Core trainer class
- `gradient_health.py` - Gradient health monitoring
- `gradient_surgery.py` - Gradient surgery techniques
- `lr_manager.py` - Learning rate management
- `memory_monitor.py` - Memory monitoring
- `metrics.py` - Training metrics collection
- `performance_modes.py` - Performance mode management
- `progressive_training.py` - Progressive training strategies
- `rank_aware_error_handler.py` - Rank-aware error handling
- `run_manager.py` - Training run management

### Utils
- `async_logging.py` - Asynchronous logging
- `checkpoint.py` - Checkpoint management
- `gpu_memory.py` - GPU memory utilities
- `logging.py` - Logging utilities

---

## How to Re-enable Archived Features

### Method 1: Import from Archive
```python
# Example: Using archived generator
from src.Ava._archived.generation.generator import TextGenerator

generator = TextGenerator(model, tokenizer)
```

### Method 2: Move Back to Active Directory
```bash
# Example: Moving flash_attention_v3 back
mv code/src/Ava/_archived/optimization/flash_attention_v3.py \
   code/src/Ava/optimization/flash_attention_v3.py

# Update __init__.py to import it
```

### Method 3: Symbolic Link
```bash
# Example: Linking profiling_tools for temporary use
ln -s code/src/Ava/_archived/training/profiling_tools.py \
      code/src/Ava/training/profiling_tools.py
```

---

## Maintenance Notes

### When to Archive
- Feature is experimental and not production-ready
- Alternative implementation is preferred in production
- Feature is hardware-specific (only for specific GPUs)
- Feature is only needed for specific use cases (fine-tuning, inference, etc.)
- Code is redundant with newer implementations

### When to Restore
- Feature becomes production-ready
- Specific use case requires the feature
- Performance testing shows benefit
- User explicitly requests the feature

### Archive Structure
```
_archived/
├── data/          # Data preparation and analysis
├── evaluation/    # Alternative evaluators
├── generation/    # Inference and generation
├── layers/        # Experimental layer architectures
├── losses/        # Alternative loss functions
├── models/        # Alternative model implementations
├── optimization/  # Experimental optimizations
└── training/      # Alternative training strategies
```

---

## Version History

**Date:** 2025-10-06
**Action:** Initial archival of 26 unused modules
**Reason:** Codebase cleanup to focus on production training pipeline

---

## Contact

For questions about archived features or to request restoration of specific modules, please refer to the project documentation or create an issue in the project repository.
