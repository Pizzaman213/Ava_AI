# MoE++ Examples

This directory contains comprehensive examples for using the MoE++ language model with state-of-the-art optimizations.

## 📁 [Text Generation](generation/)
Examples for generating text with various strategies:
- Basic text generation with different sampling methods
- Batch generation for multiple prompts
- Interactive generation with streaming
- Optimized generation with continuous batching
- Simple test scripts for quick verification

[**View Generation Examples →**](generation/README.md)

## 📁 [Fine-tuning](fine_tuning/)
Complete examples for fine-tuning the model:
- Supervised Fine-tuning (SFT) with standard and optimized versions
- Parameter-Efficient Fine-tuning (LoRA/QLoRA)
- RLHF with DPO and PPO implementations
- Constitutional AI for harmlessness
- Memory optimization techniques

[**View Fine-tuning Examples →**](fine_tuning/README.md)

## 🚀 Quick Start

### Installation
```bash
# From the repository root
pip install -e .

# For macOS users
pip install -r requirements-macos.txt
```

### Basic Text Generation
```python
from src.model.moe_transformer import MoEForCausalLM
from src.generation.text_generator import TextGenerator
from transformers import AutoTokenizer

# Load model and tokenizer
model = MoEForCausalLM.from_pretrained("outputs/moe_model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = TextGenerator(model, tokenizer)

# Generate text
response = generator.generate(
    "Explain quantum computing",
    max_length=256,
    temperature=0.7
)
print(response)
```

### Optimized Fine-tuning with WandB
```bash
# Supervised fine-tuning with all optimizations
python fine_tuning/supervised_finetuning_optimized.py \
  --dataset databricks/databricks-dolly-15k \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --num-epochs 3 \
  --wandb \
  --wandb-project "moe-finetune"

# Fine-tuning with LoRA
python fine_tuning/supervised_finetuning_optimized.py \
  --model-path outputs/moe_model \
  --use-lora \
  --lora-r 32 \
  --lora-alpha 64 \
  --wandb
```

## 📊 Key Features

### Performance Optimizations
- **Continuous Batching**: Dynamic batch optimization for 2-3x throughput
- **PagedAttention**: Efficient memory management for long sequences
- **Fused Kernels**: Hardware-optimized operations
- **Hierarchical KV Cache**: Multi-level caching for inference

### Training Features
- **WandB Integration**: Comprehensive metrics and visualization
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision**: FP16/BF16 training support

### Monitoring with WandB
All training scripts support WandB logging with:
- Training/validation loss and perplexity
- Expert routing statistics (MoE specific)
- Memory usage and performance metrics
- LoRA-specific metrics when applicable
- Generation quality tracking

## 📂 Directory Structure
```
examples/
├── README.md                    # This file
├── generation/                  # Text generation examples
│   ├── README.md               # Generation guide
│   ├── simple_generation.py    # Basic generation
│   ├── batch_generation.py     # Batch processing
│   ├── interactive_generation.py # Interactive mode
│   └── ...                     # Other generation examples
└── fine_tuning/                # Fine-tuning examples
    ├── README.md               # Fine-tuning guide
    ├── supervised_finetuning.py         # Standard SFT
    ├── supervised_finetuning_optimized.py # Optimized SFT with WandB
    ├── dpo_training.py         # Direct Preference Optimization
    ├── ppo_training.py         # Proximal Policy Optimization
    └── constitutional_ai.py    # Constitutional AI training
```

## 🛠️ Advanced Usage

### Using Optimizations
```python
# Enable all optimizations for generation
python use_optimizations.py --enable-all

# Test optimization impact
python test_optimizations.py --benchmark
```

### RLHF Examples
```python
# Simple RLHF demonstration
python rlhf_simple.py

# Working RLHF example with reward model
python rlhf_working_example.py
```

## ⚠️ Common Issues

### Import Errors
If you encounter import errors:
1. Install the package: `pip install -e .` from root
2. Ensure correct Python path setup
3. Check all dependencies are installed

### Memory Issues
For large models or long sequences:
1. Enable gradient checkpointing
2. Use LoRA for parameter efficiency
3. Reduce batch size
4. Enable CPU offloading if needed

### Performance Tips
1. Use optimized scripts (*_optimized.py) for production
2. Enable mixed precision training (fp16/bf16)
3. Use continuous batching for inference
4. Monitor with WandB for bottleneck identification

## 📚 Additional Resources

- **Documentation**: See [docs/](../docs/) for detailed guides
- **Configs**: Check [configs/](../configs/) for model configurations
- **Scripts**: Production-ready scripts in [scripts/](../scripts/)

## 🤝 Contributing

To add new examples:
1. Place generation examples in `generation/`
2. Place training examples in `fine_tuning/`
3. Update relevant README files
4. Include WandB integration where applicable
5. Test thoroughly before submitting

For questions or issues, please refer to the main project documentation or open an issue on GitHub.