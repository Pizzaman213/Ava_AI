# RLHF Examples Usage Guide

This directory contains examples demonstrating how to use the RLHF (Reinforcement Learning from Human Feedback) system with trained models.

## Prerequisites

Before running RLHF examples, you need a trained base model. You can obtain one by:

1. Running supervised fine-tuning:
```bash
python examples/fine_tuning/supervised_finetuning.py
```

2. Or using an existing checkpoint from:
   - `examples/fine_tuning/outputs/sft_model/final_model/pytorch_model.bin`
   - `examples/fine_tuning/outputs/sft_model/checkpoint-best/pytorch_model.bin`

## Available Examples

### 1. rlhf_working_example.py - Simple Working Example
A simplified example that demonstrates basic RLHF usage:

```bash
# Run with default settings (will auto-detect trained model)
python examples/rlhf_working_example.py

# Specify a model path
python examples/rlhf_working_example.py --model-path path/to/model.bin

# Test generation only (no training)
python examples/rlhf_working_example.py --test-only

# Run with custom settings
python examples/rlhf_working_example.py --num-epochs 2 --output-dir outputs/my_rlhf
```

Features:
- Auto-detects trained models
- Shows before/after generation comparisons
- Simplified configuration for quick testing
- Handles errors gracefully

### 2. rlhf_example.py - Full RLHF Pipeline
Comprehensive example with all RLHF components:

```bash
# Run full RLHF pipeline
python examples/rlhf_example.py

# Skip specific phases
python examples/rlhf_example.py --skip-phases self_supervised constitutional

# Use specific model and config
python examples/rlhf_example.py --model-path path/to/model.bin --config-path configs/gpu/medium.yaml
```

Training phases:
1. **self_supervised_pretrain** - Self-supervised learning for consistency
2. **constitutional_alignment** - Constitutional AI for safety
3. **ppo_optimization** - PPO with human preferences  
4. **reasoning_enhancement** - Improve logical reasoning
5. **final_tuning** - Combined objective optimization

### 3. rlhf_simple.py - Minimal Demo
Bare minimum example for quick testing:

```bash
python examples/rlhf_simple.py
```

## What RLHF Does

The RLHF system enhances your trained model with:

1. **Better Alignment**: Model outputs better match human preferences
2. **Improved Safety**: Constitutional AI ensures safer responses
3. **Enhanced Reasoning**: Better logical and mathematical reasoning
4. **Reduced Hallucination**: More factual and grounded responses
5. **Better Calibration**: Model is more confident when correct

## Configuration Options

Key configuration parameters in `TrainingConfig`:

```python
config = TrainingConfig(
    # Basic settings
    model_name="my_rlhf_model",
    num_epochs=3,
    output_dir="outputs/rlhf",
    
    # Choose training phases
    training_phases=["ppo_optimization", "reasoning_enhancement"],
    
    # PPO settings
    ppo=EnhancedPPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        ppo_epochs=4,
    ),
    
    # Reasoning settings
    reasoning=ReasoningConfig(
        use_chain_of_thought=True,
        reasoning_types=["mathematical", "logical"],
    ),
)
```

## Expected Output

After successful RLHF training, you should see:
- Improved response quality
- Better adherence to instructions
- More coherent reasoning chains
- Reduced harmful or incorrect outputs

## Troubleshooting

If training fails:
1. Ensure you have a trained base model
2. Check GPU/memory availability
3. Reduce batch size or model size
4. Try running with fewer phases
5. Check logs for specific error messages

## Next Steps

After RLHF training:
1. Evaluate model with `examples/generation/test_generation.py`
2. Deploy with `scripts/deploy_rlhf.py`
3. Fine-tune further with domain-specific data
4. Monitor performance in production

For more details, see the [RLHF documentation](../docs/rlhf_guide.md).