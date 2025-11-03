# Ava MoE++ Documentation

Welcome to the Ava MoE++ (Mixture of Experts Plus Plus) documentation. This advanced language model implementation features state-of-the-art routing mechanisms, dynamic expert selection, DeepSpeed integration, and comprehensive enhanced training utilities with RAG, quantization, and advanced AI features.

##  Documentation Index

### 🚀 Getting Started
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - ⭐ **START HERE!** Fast decision trees & cheat sheets
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Complete system architecture with diagrams
- [02_TRAINING_GUIDE.md](02_TRAINING_GUIDE.md) - Comprehensive training walkthrough
- [FLOWCHARTS_VISUAL.md](FLOWCHARTS_VISUAL.md) - Visual flowcharts for all major workflows

### 📚 Core Guides (Comprehensive with Flowcharts)
- **[04_LOSS_FUNCTIONS.md](04_LOSS_FUNCTIONS.md)** - Loss system with computation flowcharts
- **[05_OPTIMIZATION_GUIDE.md](05_OPTIMIZATION_GUIDE.md)** - Optimizers, LR schedules, gradient ops
- **[06_EVALUATION_GENERATION.md](06_EVALUATION_GENERATION.md)** - Evaluation, generation & RLHF
- **[07_CONFIGURATION_SYSTEM.md](07_CONFIGURATION_SYSTEM.md)** - Config hierarchy & best practices
- [03_MEMORY_OPTIMIZATION.md](03_MEMORY_OPTIMIZATION.md) - Memory management strategies

### 📖 Loss Functions (Detailed)
- [LOSSES_README.md](LOSSES_README.md) - Loss functions quick overview
- [LOSSES_USAGE_GUIDE.md](LOSSES_USAGE_GUIDE.md) - Detailed usage examples

### 🎯 Quick Access by Task

**I want to...**
- **Train a model** → [QUICK_REFERENCE.md#training-setup](QUICK_REFERENCE.md#training-setup)
- **Choose an optimizer** → [05_OPTIMIZATION_GUIDE.md#optimizer-selection](05_OPTIMIZATION_GUIDE.md#optimizer-selection)
- **Configure training** → [07_CONFIGURATION_SYSTEM.md](07_CONFIGURATION_SYSTEM.md)
- **Setup loss function** → [04_LOSS_FUNCTIONS.md#configuration-guide](04_LOSS_FUNCTIONS.md#configuration-guide)
- **Generate text** → [06_EVALUATION_GENERATION.md#generation-pipeline](06_EVALUATION_GENERATION.md#generation-pipeline)
- **Evaluate model** → [06_EVALUATION_GENERATION.md#evaluation-system](06_EVALUATION_GENERATION.md#evaluation-system)
- **Troubleshoot issues** → [QUICK_REFERENCE.md#troubleshooting](QUICK_REFERENCE.md#troubleshooting)

### 📊 Documentation by Topic

| Topic | Document | Description |
|-------|----------|-------------|
| **Architecture** | [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Model architecture, MoE, layers |
| **Training** | [02_TRAINING_GUIDE.md](02_TRAINING_GUIDE.md) | Complete training pipeline |
| **Memory** | [03_MEMORY_OPTIMIZATION.md](03_MEMORY_OPTIMIZATION.md) | OOM solutions, memory management |
| **Loss Functions** | [04_LOSS_FUNCTIONS.md](04_LOSS_FUNCTIONS.md) | UnifiedLoss, components, flowcharts |
| **Optimization** | [05_OPTIMIZATION_GUIDE.md](05_OPTIMIZATION_GUIDE.md) | Optimizers, LR, gradients |
| **Evaluation** | [06_EVALUATION_GENERATION.md](06_EVALUATION_GENERATION.md) | Metrics, generation, RLHF |
| **Configuration** | [07_CONFIGURATION_SYSTEM.md](07_CONFIGURATION_SYSTEM.md) | YAML configs, hierarchy |
| **Quick Reference** | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Decision trees, commands |
| **Visual Flows** | [FLOWCHARTS_VISUAL.md](FLOWCHARTS_VISUAL.md) | Emoji-annotated flowcharts |

##  Quick Example

```python
# Train a model
python scripts/training/train.py --config configs/cpu/small.yaml

# Generate text
python scripts/generation/generate.py \
    --model-path outputs/best_model.pt \
    --prompt "The future of AI is"

# Evaluate model
python scripts/evaluation/evaluate.py \
    --model-path outputs/best_model.pt \
    --metrics perplexity accuracy
```

##  Project Structure

```
/project/code/
 src/Ava/            # Core implementation
    models/         # Model architectures
    layers/         # Layer implementations
    data/           # Data utilities
    generation/     # Generation utilities
    evaluation/     # Evaluation tools
    utils/          # Helper functions
 scripts/            # Training and inference scripts
    training/       # Training scripts
    generation/     # Generation scripts
    evaluation/     # Evaluation scripts
 configs/            # Configuration files
    cpu/           # CPU configurations
    gpu/           # GPU configurations
    auto/          # Auto configurations
 data/              # Data directory
    pretraining/   # Pretraining datasets
        processed/ # Processed data
 outputs/           # Training outputs
 docs/             # Documentation

```

##  Key Features

### Enhanced Mixture of Experts (MoE++)
- **Dynamic Routing**: Confidence-based expert selection
- **Load Balancing**: Sinkhorn normalization for balanced expert usage
- **Sparse Computation**: Conditional computation for efficiency
- **Hierarchical Experts**: Multi-level expert organization

### Advanced Training
- **Curriculum Learning**: Progressive difficulty adjustment
- **Adaptive Regularization**: Dynamic regularization strategies
- **Smart Checkpointing**: Memory-efficient gradient checkpointing
- **Mixed Precision**: FP16/BF16 training support

### Comprehensive Utilities
- **Multiple Generation Strategies**: Greedy, beam search, nucleus sampling
- **Evaluation Suite**: Perplexity, accuracy, expert utilization metrics
- **Data Pipeline**: Efficient data loading and preprocessing
- **Logging System**: Detailed training and evaluation logs

##  Model Configurations

| Config | Parameters | Use Case | Hardware |
|--------|-----------|----------|----------|
| `cpu/small.yaml` | ~50M | Development/Testing | 8+ CPU cores, 16GB RAM |
| `cpu/medium.yaml` | ~200M | Small-scale training | 16+ CPU cores, 32GB RAM |
| `gpu/base.yaml` | ~500M | Standard training | 1x GPU (8GB+ VRAM) |
| `gpu/large.yaml` | ~1.5B | Large-scale training | 1x GPU (24GB+ VRAM) |

##  Performance

Training performance on standard hardware:

| Hardware | Config | Tokens/sec | Memory Usage |
|----------|--------|------------|--------------|
| CPU (8 cores) | small | ~100 | 8GB |
| CPU (16 cores) | medium | ~50 | 16GB |
| RTX 3090 | base | ~1,000 | 12GB |
| A100 40GB | large | ~5,000 | 32GB |

##  Development

### Contributing
See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

### Testing
```bash
# Run tests
python test_training.py

# Check model creation
python -c "from src.Ava.models import EnhancedMoEModel, EnhancedMoEConfig; \
           config = EnhancedMoEConfig(hidden_size=128, num_layers=2); \
           model = EnhancedMoEModel(config); \
           print(f'Model created: {sum(p.numel() for p in model.parameters())} params')"
```

### Debugging
- Enable debug logging: `--log-level DEBUG`
- Check outputs: `python scripts/show_outputs.py`
- Monitor training: `tail -f outputs/training_*.log`

##  Citation

If you use Ava MoE++ in your research, please cite:

```bibtex
@software{ava_moe_plus_plus,
  title = {Ava MoE++: Enhanced Mixture of Experts Implementation},
  year = {2024},
  url = {https://github.com/your-repo/ava-moe}
}
```

##  License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.

##  Acknowledgments

- Based on the Mixture of Experts architecture
- Implements techniques from recent MoE research papers
- Uses PyTorch and Hugging Face Transformers

##  Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@example.com

---

Last updated: September 2024
Version: 2.0.0 (Enhanced with DeepSpeed + Advanced AI Features)