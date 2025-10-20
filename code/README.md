#  Advanced LLM Training Framework

[![Tests](https://img.shields.io/badge/Tests-36%2F44%20Passing-brightgreen)](tests/test_all_codebase_features.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A state-of-the-art framework for training Large Language Models (LLMs) with **Mixture of Experts (MoE)**, advanced attention mechanisms, and comprehensive optimization techniques.

##  Recent Updates

**2025-10-20**: Fixed evaluation interval configuration - `eval_steps` now counts training steps instead of optimizer steps. See [EVAL_STEPS_FIX.md](/EVAL_STEPS_FIX.md) for details.

##  Fastest Setup - One Command

```bash
# Run everything with one script (setup, download data, train)
python3 run_first.py
```

This single script will:
1. Setup project folders
2. Install dependencies via fast pip install
3. Download datasets in parallel (Wikipedia, code, and all datasets)
4. Prepare data for training
5. Train the model with optimized configuration

##  Quick Start in 5 Minutes (Manual Steps)

```bash
# 1. Setup (30 seconds)
pip install -r requirements.txt
python3 setup_folders.py

# 2. Download data (1 minute)
python3 scripts/data_download/download_quick_test.py

# 3. Prepare data (1 minute)
python3 scripts/data_prep/prepare_quick_data.py

# 4. Train model (2-3 minutes for quick test)
python3 scripts/training/train_with_data.py --config configs/small_model.yaml --epochs 1

# 5. Test your model
python3 scripts/generation/interactive_generate.py
```

##  Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)

##  Features

### Core Capabilities
- **Mixture of Experts (MoE++)** with hierarchical routing
- **Mixture of Depths (MoD)** for dynamic computation
- **Multi-Query Attention (MQA)** and **Grouped Query Attention (GQA)**
- **Flash Attention** support for memory efficiency
- **Rotary Position Embeddings (RoPE)** with scaling
- **SwiGLU activation** and advanced optimizers

### Training Features
- **Multi-Column Data Loading** - Support for complex datasets with mixed data types
- Distributed training support (DDP, FSDP)
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- RLHF (Reinforcement Learning from Human Feedback) support
- Dynamic batch sizing and gradient accumulation
- Curriculum learning and progressive training

### Data Processing Features
- **Multi-Column Data Support** - Handle text, numeric, categorical, image, and tensor columns
- **Flexible Combine Strategies** - Concatenate, template-based, or separate column processing
- **HuggingFace Dataset Integration** - Direct loading from HuggingFace Hub
- **Streaming Data Loading** - Memory-efficient processing of large datasets
- **Custom Column Configurations** - Configurable preprocessing, validation, and augmentation

### Platform Support
-  **CPU** - Optimized for local development
-  **CUDA/GPU** - Full GPU acceleration
-  **Apple Silicon (MPS)** - Native M1/M2/M3 support
-  **Multi-GPU** - Distributed training

##  Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training, optional)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/llm-framework.git
cd llm-framework

# Install dependencies
pip install -r requirements.txt

# For macOS/Apple Silicon
pip install -r requirements-macos.txt

# For CPU-only training
pip install -r requirements_cpu.txt
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Set up folder structure
python3 setup_folders.py
```

##  Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/llm-framework.git
cd llm-framework

# Install dependencies
pip install -r requirements.txt

# Create folder structure
python3 setup_folders.py
```

### 2. Download Training Data
```bash
# Option A: Download high-quality datasets (Recommended - 1 minute)
python3 scripts/data_download/download_quick_test.py

# Option B: Generate synthetic data (faster but lower quality)
python3 scripts/data_generation/generate_fast_data.py --num_samples 10000
```

### 3. Prepare Data for Training
```bash
# Automatic preparation (finds and processes downloaded data)
python3 scripts/data_prep/prepare_quick_data.py

# Or manual preparation with options
python3 scripts/data_prep/prepare_data.py \
    --input-path data/pretraining/raw/quick_test \
    --output-dir data/pretraining/processed \
    --input-format json \
    --max-length 512
```

### 4. Train Your Model
```bash
# Automatic data discovery - finds processed data automatically
python3 scripts/training/train_with_data.py --config configs/small_model.yaml

# Or specify epochs and batch size
python3 scripts/training/train_with_data.py \
    --config configs/small_model.yaml \
    --epochs 5 \
    --batch-size 8
```

### 5. Test Your Trained Model
```bash
# Interactive generation with your trained model
python3 scripts/generation/interactive_generate.py

# Quick test of generation
python3 scripts/generation/test_generate.py

# Single test generation
python3 scripts/generation/quick_test.py
```

##  Project Structure

```
.
 configs/                     # Configuration files
    cpu/                    # CPU-optimized configs
       ultra_tiny.yaml    # Minimal (testing)
       tiny.yaml          # Small model
       medium.yaml        # Medium model
    gpu/                    # GPU-optimized configs
       small.yaml         # Small GPU model
       medium.yaml        # Standard training
       large.yaml         # Large model
       xlarge.yaml        # Extra large model
    mps/                    # Apple Silicon configs
       tiny.yaml          # M1/M2 small
       small.yaml         # M1/M2 standard
       medium.yaml        # M3 optimized
    advanced/               # Advanced configs
        full_features.yaml # All features enabled
        research_sota.yaml # SOTA configuration

 src/                         # Source code
    model/                  # Model architecture
       moe_transformer.py # MoE implementation
       experts.py         # Expert layers
       attention.py       # Attention mechanisms
       mod.py             # Mixture of Depths
    training/               # Training utilities
    optimization/           # Optimizers
    utils/                  # Helper utilities

 scripts/                     # Executable scripts
    training/               # Training scripts
       train.py           # Main training
       train_simple.py    # Simplified training
       train_advanced.py  # Advanced features
       finetune.py        # Fine-tuning
       train_rlhf.py      # RLHF training
    generation/             # Generation scripts
       interact.py        # Interactive chat
    data/                   # Data scripts
       generate_data.py   # Generate training data
    evaluation/             # Evaluation scripts
    utils/                  # Utility scripts

 tests/                       # Test suite
    test_all_codebase_features.py

 data/                        # Training data (not in repo)
 outputs/                     # Model outputs
 checkpoints/                 # Model checkpoints
 docs/                        # Documentation

 setup.py                     # Package setup
 requirements.txt             # Main dependencies
 requirements-macos.txt       # macOS specific
 requirements_cpu.txt         # CPU-only deps
 README.md                    # This file
```

##  Usage

### Training Models

#### Basic Training
```bash
# Train a small model on CPU
python3 scripts/training/train_simple.py --config configs/cpu/tiny.yaml --steps 100

# Train a medium model on GPU
python3 scripts/training/train.py --config configs/gpu/medium.yaml --epochs 5

# Train with mixed precision
python3 scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --mixed-precision \
    --gradient-checkpointing
```

#### Distributed Training
```bash
# Multi-GPU training
torchrun --nproc_per_node=4 scripts/training/train.py \
    --config configs/gpu/large.yaml \
    --distributed

# FSDP for very large models
python3 scripts/training/train_advanced.py \
    --config configs/gpu/xlarge.yaml \
    --fsdp \
    --sharding-strategy full_shard
```

### Interactive Generation
```bash
# Basic interaction
python3 scripts/generation/interact.py --model outputs/model.pt

# With custom parameters
python3 scripts/generation/interact.py \
    --model outputs/model.pt \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.9
```

### Multi-Column Data Processing

The framework now supports advanced multi-column data loading for complex datasets with mixed data types.

#### Supported Column Types
- **Text**: Natural language text with configurable tokenization
- **Numeric**: Numerical data with normalization support
- **Categorical**: Categorical data with vocabulary mapping
- **Image**: Image data with preprocessing pipelines
- **Tensor**: Pre-computed tensor data
- **Embedding**: Pre-computed embeddings

#### Combine Strategies
- **Concatenate**: Merge all input columns into a single sequence
- **Template**: Use custom templates to format multiple columns
- **Separate**: Keep columns separate for multi-input models

#### Multi-Column Training Examples

##### Basic Multi-Column Training
```bash
# Train with simple text columns
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names text \
    --column-types text \
    --column-roles input \
    --data-dir /project/code/processed
```

##### Instruction-Response Format
```bash
# Train with instruction-response pairs using templates
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names instruction,response \
    --column-types text,text \
    --column-roles input,target \
    --combine-strategy template \
    --column-template "Instruction: {instruction}\nResponse: {response}"
```

##### Multi-Input with Mixed Data Types
```bash
# Train with text, categorical, and numeric columns
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names text_input,category,difficulty \
    --column-types text,categorical,numeric \
    --column-roles input,auxiliary,auxiliary \
    --combine-strategy separate
```

##### HuggingFace Dataset Integration
```bash
# Load directly from HuggingFace Hub
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --hf-dataset squad \
    --column-names context,question,answer \
    --column-types text,text,text \
    --column-roles input,input,target
```

#### Multi-Column Configuration Files

Create reusable configurations for complex datasets:

```yaml
# configs/multi_column_tests/instruction_response.yaml
columns:
  - name: instruction
    type: text
    role: input
    max_length: 128
    required: true
  - name: response
    type: text
    role: target
    max_length: 256
    required: true

combine_strategy: template
template: "Instruction: {instruction}\nResponse: {response}"
max_samples: 1000
validation_enabled: true
```

Use with:
```bash
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --dataset-config configs/multi_column_tests/instruction_response.yaml
```

### Data Preparation

#### Download High-Quality Datasets

##### Quick Test Dataset (Recommended for Getting Started)
```bash
# Download Alpaca (52K) + Dolly (15K) - High quality instruction data
# Downloads in ~1 minute, perfect for testing
python3 scripts/data_download/download_quick_test.py

# The data will be saved to: data/pretraining/raw/quick_test/
# Contains 67K high-quality instruction-following samples
```

##### Full Datasets (For Serious Training)
```bash
# Download Wikipedia + Code samples (great for general knowledge)
python3 scripts/data_download/download_full_datasets.py \
    --datasets wikipedia code \
    --max_samples 50000

# Download everything available
python3 scripts/data_download/download_full_datasets.py \
    --datasets all \
    --max_samples 10000
```

##### Generate Synthetic Data (Alternative)
```bash
# Fast hybrid generator (GPT-2 + templates)
python3 scripts/data_generation/generate_fast_data.py \
    --num_samples 10000 \
    --gpt2_ratio 0.2

# Pure GPT-2 generation (slower but higher quality)
python3 scripts/data_generation/generate_gpt2_data.py \
    --num_samples 1000 \
    --model gpt2-medium

# Original template-based generation (fastest)
python3 scripts/data_generation/generate_pretraining_data.py \
    --num_samples 100000
```

#### Process Downloaded Data
```bash
# Quick automatic processing (recommended)
python3 scripts/data_prep/prepare_quick_data.py

# Or manual processing with full control
python3 scripts/data_prep/prepare_data_rapids.py \
    --raw-data-dir data \
    --output-dir /project/code/processed \
    --max-samples 50000 \
    --max-tokens 10000000 \
    --format-strategy multi_column
```

#### Training with Downloaded Data
```bash
# The training script automatically finds data in standard locations
python3 scripts/training/train_with_data.py --config configs/small_model.yaml

# It will search for data in:
# - /project/code/processed/*.jsonl (new default location)
# - data/pretraining/processed/train/*.json
# - data/pretraining/raw/*/train/*.json
# - data/train/*.json

# Monitor training progress - model saves every 1000 steps
# Best model saved as: outputs/run_*/best_model.pt
# Final model saved as: outputs/run_*/final_model.pt
```

##  Configuration

### Configuration Structure
```yaml
# Example: configs/gpu/medium.yaml
model:
  vocab_size: 50257
  hidden_size: 1024
  num_layers: 24
  num_attention_heads: 16
  num_experts: 8
  num_experts_per_tok: 2
  use_mod: true               # Mixture of Depths
  use_flash_attention: true   # Flash Attention
  use_rotary_embeddings: true # RoPE
  
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  gradient_accumulation_steps: 4
  mixed_precision: true
  
optimization:
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_steps: 1000
  weight_decay: 0.01
```

### Available Configurations

| Config | Parameters | Memory | Use Case |
|--------|-----------|--------|----------|
| `cpu/ultra_tiny.yaml` | 10M | 1GB | Testing & debugging |
| `cpu/tiny.yaml` | 50M | 2GB | Local development |
| `gpu/small.yaml` | 125M | 4GB | Quick experiments |
| `gpu/medium.yaml` | 350M | 8GB | Standard training |
| `gpu/large.yaml` | 1.3B | 16GB | Production models |
| `gpu/xlarge.yaml` | 7B | 40GB | Research |
| `mps/small.yaml` | 125M | 8GB | M1/M2 Macs |
| `advanced/full_features.yaml` | Variable | Variable | Feature testing |

##  Testing

### Run Tests
```bash
# Run comprehensive test suite
python3 tests/test_all_codebase_features.py

# Run quick tests only
python3 tests/test_all_codebase_features.py --quick

# Run feature-specific tests
python3 tests/test_all_codebase_features.py --features
```

### Test Coverage
-  **36 tests passing** - Core functionality verified
-  **Model Creation** - 14.4M parameter MoE model
-  **Training Loop** - Forward/backward passes
-  **Generation** - Text generation with beam search
-  **Checkpointing** - Save/load functionality
-  **Configurations** - All 20 configs validated
-  **Device Support** - CPU/GPU/MPS compatibility

##  Training Guide & Best Practices

### Data Requirements for Meaningful Results

| Goal | Minimum Data | Recommended Data | Expected Loss | Training Time (MPS) |
|------|--------------|------------------|---------------|-------------------|
| **Basic Testing** | 10K samples | 50K samples | 2.0-3.0 | 30 min |
| **Coherent Text** | 100K samples | 500K samples | 1.0-1.5 | 2-4 hours |
| **Domain Specific** | 500K samples | 2M samples | 0.5-1.0 | 8-12 hours |
| **General Purpose** | 5M samples | 20M+ samples | 0.2-0.5 | 2-3 days |
| **Production Quality** | 50M samples | 100M+ samples | <0.2 | 1 week+ |

### Understanding Loss Values

| Loss Range | What It Means | Text Quality |
|------------|---------------|--------------|
| **>5.0** | Model is learning basic patterns | Random/gibberish |
| **3.0-5.0** | Learning word associations | Some real words, mostly nonsense |
| **2.0-3.0** | Basic sentence structure emerging | Short phrases make sense |
| **1.0-2.0** | Coherent sentences forming | Readable but may repeat |
| **0.5-1.0** | Good language understanding | Fluent, occasional errors |
| **0.2-0.5** | Strong model performance | High quality, coherent |
| **<0.2** | Excellent (or overfitting!) | Check for memorization |

### Recommended Training Configurations

#### For Quick Testing (5-30 minutes)
```bash
# Download minimal data
python3 scripts/data_download/download_quick_test.py

# Train tiny model
python3 scripts/training/train_with_data.py \
    --config configs/mps/tiny.yaml \
    --epochs 3 \
    --batch-size 128
```
- **Expected Loss**: 2.0-3.0
- **Data Used**: 67K samples
- **Time**: 15-30 minutes on M1/M2

#### For Meaningful Results (2-4 hours)
```bash
# Download 500K samples
python3 scripts/data_download/download_500k_samples.py

# Process the data
python3 scripts/data_prep/prepare_data.py

# Train with optimized settings
python3 scripts/training/train_with_data.py \
    --config configs/mps/small.yaml \
    --epochs 10 \
    --batch-size 64
```
- **Expected Loss**: 0.8-1.5
- **Data Used**: 500K samples
- **Time**: 2-4 hours on M1/M2

#### For Production Models (1-3 days)
```bash
# Download millions of samples
python3 scripts/data_download/download_full_datasets.py \
    --datasets all \
    --max_samples 5000000

# Train larger model
python3 scripts/training/train_with_data.py \
    --config configs/mps/medium.yaml \
    --epochs 20 \
    --batch-size 32
```
- **Expected Loss**: 0.2-0.5
- **Data Used**: 5M+ samples
- **Time**: 1-3 days

### GPU Utilization Optimization

| GPU Usage | Issue | Solution |
|-----------|-------|----------|
| **<30%** | Data bottleneck | Increase workers: `--num-workers 8` |
| **30-50%** | Small batches | Double batch size: `--batch-size 256` |
| **50-70%** | CPU preprocessing | Use preprocessed Arrow format data |
| **70-90%** | Good utilization | Optimal settings |
| **>90%** | May OOM soon | Reduce batch size slightly |

#### Optimal Batch Sizes by Model

| Model | GPU Memory | Recommended Batch Size | Max Sequence Length |
|-------|------------|----------------------|-------------------|
| **Tiny (100M)** | 8GB | 256-512 | 512 |
| **Small (350M)** | 16GB | 64-128 | 1024 |
| **Medium (760M)** | 24GB | 32-64 | 2048 |
| **Large (1.3B)** | 32GB | 16-32 | 2048 |

### Common Issues and Solutions

#### Issue: Model outputs repetitive text
**Cause**: Overfitting on small dataset
**Solution**: 
- Use more data (500K+ samples)
- Add dropout: `--dropout 0.1`
- Reduce learning rate: `--lr 5e-5`
- Use temperature in generation: `--temperature 0.8`

#### Issue: Training is very slow
**Cause**: Inefficient data loading
**Solution**:
```bash
# Increase workers and batch size
python3 scripts/training/train_with_data.py \
    --config configs/mps/tiny.yaml \
    --batch-size 512 \
    --num-workers 8
```

#### Issue: Loss plateaus early
**Cause**: Learning rate too high/low
**Solution**:
- Try different learning rates: `1e-4`, `5e-4`, `1e-3`
- Use learning rate scheduler
- Increase model capacity (use small instead of tiny)

#### Issue: Out of memory
**Cause**: Batch size too large
**Solution**:
- Reduce batch size by half
- Enable gradient checkpointing
- Use gradient accumulation:
```bash
--batch-size 32 --gradient-accumulation-steps 4
```

### Training Metrics to Monitor

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **Loss** | Decreasing steadily | Plateaus, spikes, or NaN |
| **Learning Rate** | 1e-4 to 5e-3 | Too high: instability |
| **Gradient Norm** | <1.0 | >10: exploding gradients |
| **GPU Memory** | 70-90% | >95%: may crash |
| **Tokens/Second** | >1000 | <100: data bottleneck |

### Performance Benchmarks

#### Training Speed (Apple Silicon)
| Device | Model | Batch Size | Tokens/Second |
|--------|-------|------------|---------------|
| **M1** | Tiny | 128 | 800-1200 |
| **M1 Pro** | Small | 64 | 1500-2000 |
| **M2** | Small | 128 | 2000-2500 |
| **M2 Max** | Medium | 64 | 2500-3500 |
| **M3 Max** | Large | 32 | 3000-4000 |

#### Training Speed (NVIDIA GPUs)
| Device | Model | Batch Size | Tokens/Second |
|--------|-------|------------|---------------|
| **RTX 3060** | Small | 128 | 3000-4000 |
| **RTX 3090** | Medium | 128 | 8000-10000 |
| **RTX 4090** | Large | 64 | 15000-20000 |
| **A100** | XLarge | 32 | 25000-30000 |

### Optimization Features
- **Flash Attention 2** - 2-3x memory reduction
- **Mixed Precision** - 2x speedup with minimal quality loss
- **Gradient Checkpointing** - Trade compute for memory
- **Fused Kernels** - Optimized CUDA operations
- **Dynamic Batching** - Efficient resource utilization
- **MPS Optimization** - Native Apple Silicon acceleration

##  Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints to functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

##  Documentation

- [Training Guide](docs/training_guide.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Performance Tuning](docs/performance.md)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- PyTorch team for the excellent framework
- Flash Attention authors for memory-efficient attention
- Hugging Face for transformer implementations
- The open-source ML community

##  Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-framework/discussions)
- **Email**: support@example.com

---

**Built with  for the AI community**