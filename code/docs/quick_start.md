# Quick Start Guide

Get Ava MoE++ running in 5 minutes!

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 16GB+ RAM (for CPU training)
- Optional: CUDA-capable GPU

## Installation

```bash
# Clone the repository (if using git)
cd /project/code

# Install dependencies
pip install torch transformers tqdm pandas pyarrow numpy matplotlib

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## 1. Test the Setup

Run the test script to verify everything works:

```bash
python test_training.py
```

Expected output:
```
1. Creating tokenizer...
2. Creating model...
   Model has 7,111,434 parameters
3. Creating dataloaders...
4. Testing one training step...
   ✓ Training step successful!
✅ All tests passed!
```

## 2. Prepare Data

The system expects data in `/project/code/data/pretraining/processed/`.

For testing, create dummy data:
```bash
python -c "
import pandas as pd
data = {'text': ['Sample text for training.'] * 100}
df = pd.DataFrame(data)
df.to_parquet('/project/code/data/pretraining/processed/train_dummy.parquet')
print('Created dummy training data')
"
```

## 3. Start Training

Train a small model on CPU:

```bash
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --epochs 3 \
    --batch-size 2
```

Monitor training:
```bash
# In another terminal
tail -f /project/code/outputs/training_*.log
```

## 4. Generate Text

After training, generate text:

```bash
python scripts/generation/generate.py \
    --model-path /project/code/outputs/best_model.pt \
    --prompt "Once upon a time" \
    --max-length 100 \
    --temperature 0.8
```

## 5. Evaluate Model

Evaluate the trained model:

```bash
python scripts/evaluation/evaluate.py \
    --model-path /project/code/outputs/best_model.pt \
    --config configs/cpu/small.yaml
```

## Interactive Generation

Start interactive mode:

```bash
python scripts/generation/generate.py \
    --model-path /project/code/outputs/best_model.pt \
    --interactive
```

Then type prompts and see generated text!

## Common Commands

### Training
```bash
# Small model (50M params) - good for testing
python scripts/training/train.py --config configs/cpu/small.yaml

# Medium model (200M params) - better quality
python scripts/training/train.py --config configs/cpu/medium.yaml

# Resume from checkpoint
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --resume /project/code/outputs/checkpoint_epoch_5.pt
```

### Generation
```bash
# Basic generation
python scripts/generation/generate.py \
    --model-path outputs/best_model.pt \
    --prompt "The meaning of life is"

# Advanced generation with beam search
python scripts/generation/generate.py \
    --model-path outputs/best_model.pt \
    --prompt "In the year 2050" \
    --num-beams 5 \
    --temperature 0.7 \
    --top-p 0.9
```

### Monitoring
```bash
# Show outputs summary
python scripts/show_outputs.py

# Watch training progress
watch -n 5 "tail -20 /project/code/outputs/training_*.log"

# Check GPU usage (if using GPU)
nvidia-smi -l 1
```

## Quick Tips

1. **Start Small**: Begin with `configs/cpu/small.yaml` for testing
2. **Monitor Memory**: Use `htop` or `top` to watch RAM usage
3. **Save Checkpoints**: Use `--save-every 100` for frequent checkpoints
4. **Batch Size**: Reduce batch size if running out of memory
5. **Learning Rate**: Start with default, adjust if loss plateaus

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 1

# Enable gradient checkpointing (already in config)
# Reduce model size (use small.yaml)
```

### Slow Training
```bash
# Reduce sequence length
--max-length 256

# Use fewer workers
--num-workers 2

# Disable logging
--log-level WARNING
```

### No Data Found
```bash
# Check data directory
ls -la /project/code/data/pretraining/processed/

# Create dummy data (see step 2 above)
```

## Next Steps

- Read the [Training Guide](training_guide.md) for detailed training instructions
- Explore [Configuration Guide](configuration.md) to customize model settings
- Check [Architecture Overview](architecture.md) to understand the model
- Learn about [Generation Strategies](generation.md) for better outputs

## Get Help

```bash
# Show help for any script
python scripts/training/train.py --help
python scripts/generation/generate.py --help
python scripts/evaluation/evaluate.py --help
```

Happy training! 🚀