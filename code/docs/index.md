# Ava MoE++ Documentation Index

## 📖 Documentation Structure

```
/project/code/docs/
├── README.md              # Main documentation overview
├── index.md              # This file - documentation index
├── quick_start.md        # 5-minute getting started guide
├── training_guide.md     # Complete training instructions
├── architecture.md       # Model architecture details
├── configuration.md      # Configuration guide
└── api/                  # API reference
    └── models.md        # Model API documentation
```

## 🚀 Getting Started

1. **[Quick Start Guide](quick_start.md)** - Get running in 5 minutes
2. **[Training Guide](training_guide.md)** - Detailed training instructions
3. **[Configuration Guide](configuration.md)** - Understanding YAML configs

## 📚 Core Documentation

### Architecture & Design
- **[Model Architecture](architecture.md)** - Detailed architecture overview
- **[MoE++ Features](architecture.md#advanced-features)** - Enhanced Mixture of Experts
- **[Routing Mechanisms](architecture.md#expert-routing)** - Expert selection strategies

### API Reference
- **[Models API](api/models.md)** - Model classes and methods
- **[Configuration API](api/models.md#class-enhancedmoeconfig)** - Config parameters

### Training & Usage
- **[Training Guide](training_guide.md)** - Complete training walkthrough
- **[Data Preparation](training_guide.md#data-preparation)** - Preparing datasets
- **[Monitoring Training](training_guide.md#monitoring)** - Tracking progress

### Configuration
- **[Configuration Guide](configuration.md)** - YAML configuration details
- **[Configuration Presets](configuration.md#configuration-presets)** - Pre-made configs
- **[Custom Configurations](configuration.md#creating-custom-configurations)** - Making your own

## 📁 Project Structure Reference

```
/project/code/
├── docs/                 # Documentation (you are here)
├── src/Ava/             # Core implementation
│   ├── models/          # Model architectures
│   ├── layers/          # Layer implementations
│   ├── data/            # Data loading utilities
│   ├── generation/      # Generation utilities
│   ├── evaluation/      # Evaluation tools
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
│   ├── training/        # Training scripts
│   ├── generation/      # Generation scripts
│   ├── evaluation/      # Evaluation scripts
│   └── show_outputs.py  # Output summary tool
├── configs/             # Configuration files
│   ├── cpu/            # CPU configurations
│   └── gpu/            # GPU configurations
├── data/               # Data directory
│   └── pretraining/
│       └── processed/  # Processed training data
└── outputs/            # Training outputs & logs
```

## 🔧 Key Scripts

### Training
```bash
python scripts/training/train.py --config configs/cpu/small.yaml
```
See: [Training Guide](training_guide.md)

### Generation
```bash
python scripts/generation/generate.py --model-path outputs/best_model.pt --prompt "text"
```
See: [Generation Guide](quick_start.md#4-generate-text)

### Evaluation
```bash
python scripts/evaluation/evaluate.py --model-path outputs/best_model.pt
```
See: [Evaluation Guide](quick_start.md#5-evaluate-model)

### Monitoring
```bash
python scripts/show_outputs.py
```
See: [Output Management](../README_OUTPUTS.md)

## 📊 Configuration Files

### CPU Configurations
- `configs/cpu/small.yaml` - 50M parameters, 8GB RAM
- `configs/cpu/medium.yaml` - 200M parameters, 16GB RAM

### GPU Configurations
- `configs/gpu/base.yaml` - 500M parameters, 8GB VRAM
- `configs/gpu/large.yaml` - 1.5B parameters, 24GB VRAM

See: [Configuration Guide](configuration.md)

## 🛠️ Development Resources

### Testing
```bash
python test_training.py
```

### API Usage
```python
from src.Ava.models import EnhancedMoEModel, EnhancedMoEConfig

config = EnhancedMoEConfig(hidden_size=768, num_layers=12)
model = EnhancedMoEModel(config)
```
See: [API Reference](api/models.md)

## 📈 Performance Guidelines

### Memory Usage
| Model Size | RAM Required | VRAM Required |
|------------|-------------|---------------|
| Small (50M) | 8GB | 2GB |
| Medium (200M) | 16GB | 4GB |
| Large (1.5B) | 32GB | 24GB |

### Training Speed
| Hardware | Tokens/sec (Small) | Tokens/sec (Large) |
|----------|-------------------|-------------------|
| CPU (8 cores) | ~100 | ~10 |
| RTX 3090 | ~2000 | ~300 |
| A100 40GB | ~5000 | ~1000 |

## 🔍 Quick Links

### Common Tasks
- [Start Training](training_guide.md#basic-training)
- [Resume Training](training_guide.md#resume-from-checkpoint)
- [Generate Text](quick_start.md#4-generate-text)
- [Evaluate Model](quick_start.md#5-evaluate-model)
- [Create Custom Config](configuration.md#creating-custom-configurations)
- [Monitor Training](training_guide.md#monitoring)

### Troubleshooting
- [Out of Memory](training_guide.md#out-of-memory-oom)
- [Slow Training](training_guide.md#slow-training)
- [Loss Not Decreasing](training_guide.md#loss-not-decreasing)
- [Configuration Issues](configuration.md#common-issues)

### Advanced Topics
- [Multi-Stage Training](training_guide.md#multi-stage-training)
- [Curriculum Learning](training_guide.md#curriculum-learning)
- [Memory Optimization](configuration.md#memory-optimized-configuration)
- [Expert Statistics](api/models.md#expert-statistics)

## 📝 Documentation Updates

- **Version**: 1.0.0
- **Last Updated**: September 2024
- **Model Version**: Ava MoE++ v1.0

## 💡 Tips

1. **Start with Quick Start**: Begin with [quick_start.md](quick_start.md)
2. **Use Small Configs**: Test with `configs/cpu/small.yaml` first
3. **Monitor Progress**: Use `scripts/show_outputs.py` regularly
4. **Read Logs**: Check `/project/code/outputs/training_*.log`
5. **Save Checkpoints**: Use `--save-every` for frequent saves

## 📞 Help & Support

- **GitHub Issues**: Report bugs and issues
- **Documentation**: This directory (`/project/code/docs/`)
- **Examples**: See scripts in `/project/code/scripts/`
- **Configs**: Pre-made in `/project/code/configs/`

---

Happy Learning with Ava MoE++! 🚀