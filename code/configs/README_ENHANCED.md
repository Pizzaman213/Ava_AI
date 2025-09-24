# 🚀 Enhanced Ava MoE++ Configuration System

> **Comprehensive configurations for state-of-the-art training with DeepSpeed, RAG, quantization, and advanced AI features**

## 📖 Table of Contents

- [Directory Structure](#-directory-structure)
- [Enhanced Features](#-enhanced-features)
- [Configuration Categories](#-configuration-categories)
- [Quick Start Guide](#-quick-start-guide)
- [Usage Examples](#-usage-examples)
- [Configuration Structure](#-configuration-structure)
- [Feature Compatibility](#-feature-compatibility)
- [Command Line Integration](#-command-line-integration)

---

## 📁 Directory Structure

```
configs/
├── 📂 gpu/                    # Single-GPU configurations
│   ├── tiny.yaml              # 100M params, 6-8GB VRAM
│   ├── small.yaml             # 150M params, 8-10GB VRAM
│   ├── medium.yaml            # 300M params, 12-16GB VRAM
│   ├── large.yaml             # 1.5B params, 24GB+ VRAM
│   ├── xl.yaml                # 900M params, 22GB+ VRAM
│   └── 1b.yaml                # 1B params, 24GB+ VRAM
├── 📂 distributed/           # Multi-GPU DeepSpeed configurations
│   ├── deepspeed_zero1.yaml   # ZeRO-1: Optimizer sharding
│   ├── deepspeed_zero2.yaml   # ZeRO-2: Optimizer + gradient sharding
│   └── deepspeed_zero3.yaml   # ZeRO-3: Full parameter sharding
├── 📂 research/              # Research-focused configurations
│   ├── rag_enabled.yaml       # RAG system research
│   └── quantization_nvfp4.yaml # NVFP4 quantization research
├── 📂 hardware/              # Hardware-specific optimizations
│   ├── a100_80gb.yaml         # NVIDIA A100 80GB optimized
│   └── h100_80gb.yaml         # NVIDIA H100 80GB optimized
└── test_training.yaml        # Quick testing configuration
```

---

## 🔥 Enhanced Features

### 🧠 **Core AI Enhancements**
| Feature | Description | Benefits |
|---------|-------------|----------|
| **🚄 DeepSpeed Integration** | ZeRO stages 1-3 with CPU/NVMe offloading | Up to 64x memory savings |
| **📚 RAG System** | Retrieval Augmented Generation | Knowledge-grounded responses |
| **🎯 Advanced Losses** | Focal, contrastive, diversity losses | Better training dynamics |
| **⚔️ Gradient Surgery** | PCGrad, GradDrop, CAGrad | Multi-task learning optimization |
| **🧠 Episodic Memory** | Continual learning with replay | Prevents catastrophic forgetting |
| **🗜️ NVFP4 Quantization** | 4-bit quantization with transforms | 4x memory reduction |

### 🏗️ **Architecture Enhancements**
| Feature | Description | Use Case |
|---------|-------------|----------|
| **🎭 MoH (Mixture of Heads)** | Specialized attention heads | Dynamic attention patterns |
| **⚡ MoA (Mixture of Activations)** | Dynamic activation functions | Adaptive computation |
| **🔄 Cross-Attention** | Enhanced context modeling | Multi-modal learning |
| **📏 ALiBi** | Attention with Linear Biases | Length generalization |

### ⚡ **Performance Optimizations**
| Feature | Description | Speed Gain |
|---------|-------------|------------|
| **🚀 Performance Modes** | Ultra-fast, express, minimal progress | Up to 3x faster |
| **💾 Memory Management** | A100/H100 optimized pools | 90%+ utilization |
| **🌊 Streaming Data** | Large dataset handling | Unlimited dataset size |
| **📊 Multi-Column Data** | Complex dataset structures | Rich data formats |

## 📋 Configuration Categories

### GPU Configs (`/gpu/`)

| Config | Parameters | VRAM | Features | Use Case |
|--------|------------|------|----------|----------|
| `tiny.yaml` | ~100M | 6-8GB | Basic + Streaming | Entry-level development |
| `small.yaml` | ~150M | 8-10GB | MoH + Memory + Quantization | Development & research |
| `medium.yaml` | ~300M | 12-16GB | MoH + Cross-attention | Moderate research |
| `large.yaml` | ~1.5B | 24GB+ | All features + DeepSpeed | Production training |
| `xl.yaml` | ~900M | 22GB+ | Advanced features | Large-scale research |
| `1b.yaml` | ~1B | 24GB+ | Full feature set | State-of-the-art |

### Distributed Configs (`/distributed/`)

| Config | ZeRO Stage | Memory Savings | Min GPUs | Use Case |
|--------|------------|----------------|----------|----------|
| `deepspeed_zero1.yaml` | 1 | ~4x optimizer | 2-8 GPUs | Optimizer sharding |
| `deepspeed_zero2.yaml` | 2 | ~8x opt+grad | 4-16 GPUs | Large model training |
| `deepspeed_zero3.yaml` | 3 | ~64x parameters | 8+ GPUs | Ultra-large models |

### Research Configs (`/research/`)

| Config | Focus | Features | Use Case |
|--------|-------|----------|----------|
| `rag_enabled.yaml` | RAG Research | Retrieval + Cross-attention | Knowledge-grounded generation |
| `quantization_nvfp4.yaml` | Quantization | 4-bit NVFP4 | Memory-efficient deployment |

### Hardware Configs (`/hardware/`)

| Config | Hardware | Memory | Special Features |
|--------|----------|--------|------------------|
| `a100_80gb.yaml` | A100 80GB | 80GB HBM2e | BF16, Tensor Cores, 70GB pool |
| `h100_80gb.yaml` | H100 80GB | 80GB HBM3 | FP8, 4th Gen Tensor Cores |

## 🚀 Usage Examples

### Basic Development
```bash
python scripts/training/train.py --config configs/gpu/small.yaml
```

### DeepSpeed Multi-GPU
```bash
deepspeed --num_gpus=4 scripts/training/train.py --config configs/distributed/deepspeed_zero2.yaml --use-deepspeed
```

### RAG Research
```bash
python scripts/training/train.py --config configs/research/rag_enabled.yaml --use-rag
```

### Hardware Optimized (A100)
```bash
python scripts/training/train.py --config configs/hardware/a100_80gb.yaml --use-deepspeed --zero-stage 2
```

### Quantization Research
```bash
python scripts/training/train.py --config configs/research/quantization_nvfp4.yaml --use-nvfp4
```

## ⚙️ Configuration Structure

Each enhanced config contains these sections:

```yaml
# Core model architecture
model:
  hidden_size: 1024
  use_moh: true
  use_rag: true
  deepspeed_moe_param_groups: true

# DeepSpeed configuration
deepspeed:
  enabled: true
  zero_stage: 2
  cpu_offload: false
  precision: "bf16"

# Enhanced features
enhanced_features:
  architecture:
    use_moh: true
    use_moa: true
  rag:
    enabled: true
    knowledge_base_path: "data/kb"
  losses:
    focal_loss: true
    adaptive_loss_scaling: true
  gradient:
    gradient_surgery: true
  memory:
    use_episodic_memory: true
  quantization:
    use_nvfp4: true

# Data loading
data_loading:
  streaming: true
  distributed: true
  multi_column: true

# Performance modes
performance:
  express_mode: true
  fast_progress: true

# Memory optimization
memory:
  enable_memory_pool: true
  pool_size_gb: 20.0
```

## 🔧 Command Line Integration

All configs support command-line overrides:

```bash
# Override DeepSpeed settings
python train.py --config configs/gpu/large.yaml --use-deepspeed --zero-stage 3 --cpu-offload

# Enable additional features
python train.py --config configs/gpu/medium.yaml --use-rag --use-episodic-memory

# Performance modes
python train.py --config configs/gpu/small.yaml --ultra-fast-mode --express-mode

# Quantization
python train.py --config configs/gpu/large.yaml --use-nvfp4 --bit-width 4
```

## 📊 Feature Compatibility Matrix

| Model Size | MoH | MoA | RAG | Memory | DeepSpeed | NVFP4 | Recommended |
|------------|-----|-----|-----|--------|-----------|-------|-------------|
| Tiny (100M) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Development |
| Small (150M) | ✅ | ❌ | ❌ | ✅ | ZeRO-1 | ❌ | Research |
| Medium (300M) | ✅ | ❌ | ❌ | ✅ | ZeRO-1 | ✅ | Advanced research |
| Large (1.5B) | ✅ | ✅ | ✅ | ✅ | ZeRO-2 | ✅ | Production |
| XL/1B+ | ✅ | ✅ | ✅ | ✅ | ZeRO-3 | ✅ | State-of-the-art |

## 🎯 Quick Start Recommendations

1. **New Users**: Start with `configs/gpu/tiny.yaml`
2. **Development**: Use `configs/gpu/small.yaml`
3. **Research**: Try `configs/gpu/medium.yaml` or `configs/research/`
4. **Production**: Use `configs/gpu/large.yaml` with DeepSpeed
5. **Multi-GPU**: Start with `configs/distributed/deepspeed_zero1.yaml`
6. **A100 Users**: Use `configs/hardware/a100_80gb.yaml`

All configurations are validated and ready to use with the enhanced training system!