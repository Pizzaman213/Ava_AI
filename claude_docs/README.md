# Ava LLM Training Framework - Documentation

Welcome to the Ava LLM Training Framework documentation. This project provides a comprehensive system for training language models with advanced features including MoE (Mixture of Experts), RLHF (Reinforcement Learning from Human Feedback), and various optimization techniques.

## 📚 Documentation Structure

All documentation is organized into 5 comprehensive guides:

### 1. **ARCHITECTURE_AND_FEATURES.md** - Model & System Architecture
Deep dive into the system architecture, model components, and advanced features:
- Model architecture overview (transformer with MoE)
- MoE (Mixture of Experts) design and implementation
- RLHF system (PPO, reward models, training pipeline)
- Advanced features (RAG, adaptive MTP, etc.)
- Loss functions and training objectives

**Start here if**: You want to understand how the system works internally

### 2. **TRAINING_AND_CONFIGURATION.md** - Getting Started with Training
Complete guide to setting up and running training:
- Quick start (3 ways to start training)
- Configuration system (YAML hierarchy and parameters)
- Hardware requirements by model size
- Optimization strategies for different scenarios
- Learning rate tuning and LR Finder
- Data preparation and tokenization

**Start here if**: You want to train a model or configure the system

### 3. **FIXES_AND_TROUBLESHOOTING.md** - Fixes & Solutions
Comprehensive troubleshooting guide covering all known issues and solutions:
- Summary of all applied fixes
- Common training problems (loss spikes, mode collapse, etc.)
- Data pipeline issues and fixes
- Generation quality issues
- GPU optimization and memory fixes
- Configuration troubleshooting
- Emergency procedures

**Start here if**: Something isn't working or you need to debug an issue

### 4. **VALIDATION_AND_TESTING.md** - Testing & Monitoring
Guide to validating training, monitoring progress, and testing models:
- Validation strategy during training
- Key metrics to monitor
- Checkpoint testing procedures
- Generation quality testing
- Test results and benchmarks
- Development log with experiment history

**Start here if**: You want to test, validate, or monitor training

## 🎯 Quick Start Options

### Option 1: Train from Scratch (Recommended)
```bash
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

### Option 2: Fine-tune Existing Model
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --resume outputs/runs/latest/checkpoint
```

### Option 3: RLHF Fine-tuning
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml
```

## 🎯 Quick Navigation - "I Want To..."

| Goal | Go To |
|------|-------|
| **Get started immediately** | TRAINING_AND_CONFIGURATION.md → Quick Start |
| **Understand how it works** | ARCHITECTURE_AND_FEATURES.md → Overview |
| **Fix a problem** | FIXES_AND_TROUBLESHOOTING.md → Troubleshooting |
| **Train a model** | TRAINING_AND_CONFIGURATION.md → Training Guide |
| **Monitor training** | VALIDATION_AND_TESTING.md → Metrics |
| **Configure features** | TRAINING_AND_CONFIGURATION.md → Configuration |
| **Optimize performance** | TRAINING_AND_CONFIGURATION.md → Optimization |

## 📊 Model Sizes

| Size | Params | Memory | Speed | Use Case |
|------|--------|--------|-------|----------|
| tiny | 100M | 4-8GB | ⚡ | Testing |
| small | 233M | 8-12GB | ✓ | **Production** |
| base | 500M | 16-24GB | ✓ | Research |
| large | 1.3B | 24-40GB | Slow | Max Quality |

## ✨ Key Features

✅ Mixture of Experts (MoE) architecture
✅ RLHF fine-tuning with PPO
✅ Multi-GPU support (DeepSpeed)
✅ Advanced optimizations (Flash Attention, FusedAdam)
✅ Custom tokenizer support
✅ Comprehensive monitoring (W&B)
✅ Gradient checkpointing and mixed precision

## 🔧 Common Commands

```bash
# Train
python scripts/5_training/train.py --config configs/gpu/small.yaml

# Test generation
python scripts/evaluation/test_generation.py --checkpoint outputs/runs/latest

# Prepare data
python scripts/data_prep/prepare_data.py --input my_data.txt

# RLHF training
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/small.yaml
```

## 📁 Key Files

```
/project/code/
├── configs/gpu/small.yaml    ← Edit for training
├── scripts/5_training/train.py    ← Main training script
├── src/Ava/models/           ← Model definitions
├── data/processed/           ← Training data
└── outputs/runs/             ← Checkpoints
```

## 📞 Troubleshooting

| Issue | See |
|-------|-----|
| Out of Memory | FIXES_AND_TROUBLESHOOTING.md → GPU Issues |
| Bad generation | FIXES_AND_TROUBLESHOOTING.md → Generation Issues |
| Loss problems | TRAINING_AND_CONFIGURATION.md → LR Tuning |
| Slow training | FIXES_AND_TROUBLESHOOTING.md → Performance |
| Data errors | FIXES_AND_TROUBLESHOOTING.md → Data Issues |

## 📚 Documentation Files

1. **ARCHITECTURE_AND_FEATURES.md** (36 KB)
   - System design and architecture
   - Model components and features

2. **TRAINING_AND_CONFIGURATION.md** (58 KB)
   - Complete training guide
   - Configuration reference
   - Optimization strategies

3. **FIXES_AND_TROUBLESHOOTING.md** (57 KB)
   - All applied fixes
   - Common issues and solutions
   - Troubleshooting guide

4. **VALIDATION_AND_TESTING.md** (29 KB)
   - Validation procedures
   - Monitoring and metrics
   - Test results and benchmarks

## ✅ Pre-Training Checklist

- [ ] GPU available (`nvidia-smi`)
- [ ] Dependencies installed
- [ ] Data prepared
- [ ] Config file reviewed
- [ ] Sufficient disk space
- [ ] W&B account (optional)

## 🚀 Next Steps

1. **Choose**: Pick training option (scratch, fine-tune, or RLHF)
2. **Configure**: Edit configs/gpu/small.yaml
3. **Prepare**: Prepare data if needed
4. **Run**: Execute training command
5. **Monitor**: Watch progress in W&B
6. **Evaluate**: Test generation quality
7. **Iterate**: Improve based on results

---

**Version**: 2.3.0
**Status**: ✅ Production Ready
**Last Updated**: 2025-10-21

For detailed information, see the specific documentation files above.
