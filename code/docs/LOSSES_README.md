# Ava AI Losses Module

This directory contains all loss functions for the Ava AI training pipeline, **consolidated into a single file** for easier maintenance and imports.

## Files

### Core Loss Module
- **[losses.py](losses.py)** - **SINGLE UNIFIED FILE** containing all loss classes and functions
  - UnifiedLoss - Main interface combining all components
  - DeepSeek-style losses (temperature-scaled CE, MTP, MoE balancing)
  - Adaptive MTP loss (confidence-weighted multi-token prediction)
  - Repetition penalties (n-gram, immediate repetition)
  - Anti-repetition losses (with adaptive penalties)
  - Advanced losses (focal, contrastive, diversity, auxiliary, etc.)

### Module Interface
- **[\_\_init\_\_.py](__init__.py)** - Exports all loss classes from losses.py

### Documentation & Tests
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Comprehensive usage guide with examples
- **[/project/code/tests/test_unified_loss.py](/project/code/tests/test_unified_loss.py)** - Test suite

## Quick Start

> **Note:** All loss classes are now in a single file (`losses.py`), but you can still import them easily via the module interface.

### Recommended: Use UnifiedLoss

```python
from Ava.losses import UnifiedLoss

# Create a comprehensive loss function
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    primary_loss_type="deepseek",  # "standard", "deepseek", or "adaptive_mtp"
    # Enable desired features
    use_mtp=True,
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# During training
loss = loss_fn(logits, targets, attention_mask=attention_mask, hidden_states=hidden_states)
loss.backward()
```

## Features

### Primary Loss Types

1. **Standard Cross-Entropy** (`primary_loss_type="standard"`)
   - Basic cross-entropy with optional label smoothing
   - Fast and simple

2. **DeepSeek-Style Loss** (`primary_loss_type="deepseek"`) ⭐ RECOMMENDED
   - Temperature-scaled cross-entropy
   - Adaptive temperature adjustment
   - Label smoothing
   - EOS penalties

3. **Adaptive MTP** (`primary_loss_type="adaptive_mtp"`)
   - Confidence-weighted multi-token prediction
   - Adaptive loss scaling
   - Best for models with MTP heads

### Additional Components

#### Multi-Token Prediction (MTP)
Predicts multiple future tokens simultaneously for better context learning.

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    use_mtp=True,
    num_future_tokens=3,
    mtp_type="deepseek"  # or "adaptive"
)
```

#### Repetition Penalties
Prevents mode collapse and repetitive outputs.

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    # N-gram repetition (prevents "the the the")
    use_ngram_penalty=True,
    ngram_size=4,
    ngram_penalty_weight=0.1,
    # Immediate repetition (prevents "time time time")
    use_immediate_repetition_penalty=True,
    immediate_repetition_weight=0.5
)
```

#### EOS Penalties
Prevents premature sequence termination.

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    eos_token_id=50256,
    min_sequence_length=20,
    eos_penalty_weight=0.05
)
```

#### MoE Load Balancing
Auxiliary-free gradient-based expert balancing.

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    num_experts=8,
    use_moe_balancing=True,
    gradient_balance_weight=0.1
)
```

#### Advanced Losses
Focal, contrastive, diversity, and auxiliary losses.

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    use_focal_loss=True,
    use_diversity_loss=True,
    use_auxiliary_loss=True
)
```

## Usage Patterns

### For Standard LLM Training

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    primary_loss_type="deepseek",
    label_smoothing=0.1,
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True,
    eos_token_id=tokenizer.eos_token_id
)
```

### For MoE Models

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_experts=8,
    use_moe_balancing=True,
    use_diversity_loss=True,
    use_ngram_penalty=True
)

# Pass MoE outputs during training
loss = loss_fn(
    logits, targets,
    gate_logits=outputs.gate_logits,
    expert_indices=outputs.expert_indices,
    expert_outputs=outputs.expert_outputs
)
```

### For Models with Repetition Issues

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    primary_loss_type="deepseek",
    # Stronger penalties
    use_ngram_penalty=True,
    ngram_penalty_weight=0.2,
    use_immediate_repetition_penalty=True,
    immediate_repetition_weight=1.0,
    eos_penalty_weight=0.1,
    min_sequence_length=30
)
```

## Integration with Training Pipeline

The UnifiedLoss is designed to work seamlessly with the existing training pipeline:

```python
# In your training script
from Ava.losses import UnifiedLoss

# Create loss function
loss_fn = UnifiedLoss(
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    primary_loss_type="deepseek",
    use_mtp=config.use_mtp,
    use_ngram_penalty=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(**batch, output_hidden_states=True)

    loss = loss_fn(
        logits=outputs.logits,
        targets=batch['labels'],
        attention_mask=batch.get('attention_mask'),
        hidden_states=outputs.hidden_states[-1] if config.use_mtp else None
    )

    loss.backward()
    optimizer.step()
```

## Detailed Loss Breakdown

For monitoring and debugging, you can get a detailed breakdown:

```python
loss_dict = loss_fn(
    logits, targets,
    attention_mask=attention_mask,
    hidden_states=hidden_states,
    return_detailed=True
)

# Access individual components
print(f"Main Loss: {loss_dict['main_loss']}")
print(f"MTP Loss: {loss_dict.get('mtp_loss', 0)}")
print(f"N-gram Penalty: {loss_dict.get('ngram_penalty', 0)}")
print(f"Total Loss: {loss_dict['total_loss']}")

# Backward pass on total loss
loss_dict['total_loss'].backward()
```

## Migration from Old Losses

### From DeepSeekLoss
```python
# Old
from Ava.losses import DeepSeekLoss
loss_fn = DeepSeekLoss(vocab_size=50257, hidden_size=768)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(vocab_size=50257, hidden_size=768, primary_loss_type="deepseek")
```

### From AdaptiveMTPLoss
```python
# Old
from Ava.losses import AdaptiveMTPLoss
loss_fn = AdaptiveMTPLoss(vocab_size=50257)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(vocab_size=50257, hidden_size=768, primary_loss_type="adaptive_mtp")
```

### From AntiRepetitionLoss
```python
# Old
from Ava.losses import AntiRepetitionLoss
loss_fn = AntiRepetitionLoss(vocab_size, eos_token_id, pad_token_id)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id
)
```

## Architecture

```
UnifiedLoss
├── Primary Loss (standard/deepseek/adaptive_mtp)
├── Multi-Token Prediction (optional)
├── N-gram Repetition Penalty (optional)
├── Immediate Repetition Penalty (optional)
├── MoE Balancing (optional)
├── Focal Loss (optional)
├── Diversity Loss (optional)
└── Auxiliary Losses (optional)
```

All components are modular and can be enabled/disabled independently.

## Performance Characteristics

- **Memory**: Minimal overhead (~1-2% vs standard CE)
- **Speed**: ~5-10% slower than standard CE with all features enabled
- **Gradient Accumulation**: Fully compatible
- **Mixed Precision**: Works with AMP/bfloat16
- **Distributed Training**: Compatible with DDP, FSDP, DeepSpeed

## Best Practices

1. **Start Simple**: Begin with `primary_loss_type="deepseek"` and basic repetition penalties
2. **Monitor Closely**: Use `return_detailed=True` during initial training to understand component contributions
3. **Tune Weights**: Adjust penalty weights based on your specific problem
4. **Use MTP Carefully**: MTP adds overhead; only enable if you have the compute budget
5. **MoE Balancing**: Essential for MoE models; use auxiliary-free approach for best performance

## Testing

Run the test suite to verify installation:

```bash
cd /project/code
python tests/test_unified_loss.py
```

All tests should pass with "All tests passed! ✓"

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Comprehensive usage guide with examples
- **[unified_loss.py](unified_loss.py)** - Source code with detailed docstrings
- **Individual component files** - Each loss module has detailed documentation

## Support

For issues or questions:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Review test examples in [test_unified_loss.py](/project/code/tests/test_unified_loss.py)
3. Check individual module documentation

## License

Part of the Ava AI project.