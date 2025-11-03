# Unified Loss Usage Guide

This guide explains how to use the unified loss system in the Ava AI training pipeline.

> **Important:** All loss classes have been consolidated into a single file (`losses.py`) for easier maintenance. You can still import them via the module interface: `from Ava.losses import UnifiedLoss, DeepSeekLoss, ...`

## Overview

The `UnifiedLoss` module combines all available loss functions into a single, flexible interface:
- Temperature-scaled cross-entropy (DeepSeek-style)
- Multi-token prediction (DeepSeek and Adaptive MTP)
- N-gram repetition penalties
- Immediate repetition detection
- EOS token penalties
- MoE load balancing (auxiliary-free)
- Advanced losses (focal, contrastive, diversity)

## Quick Start

### Basic Usage (Standard Cross-Entropy)

```python
from Ava.losses import UnifiedLoss

# Create a basic loss with default settings
loss_fn = UnifiedLoss(
    vocab_size=50257,
    primary_loss_type="standard"
)

# During training
logits = model(input_ids, attention_mask=attention_mask)
loss = loss_fn(logits, targets, attention_mask=attention_mask)
loss.backward()
```

### DeepSeek-Style Loss (Recommended)

```python
from Ava.losses import UnifiedLoss

# Create DeepSeek-style loss with all features
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    primary_loss_type="deepseek",
    # Temperature scaling
    initial_temperature=1.0,
    adaptive_temperature=True,
    label_smoothing=0.1,
    # Multi-token prediction
    use_mtp=True,
    num_future_tokens=3,
    mtp_weight=0.1,
    # Repetition penalties
    use_ngram_penalty=True,
    ngram_size=4,
    ngram_penalty_weight=0.1,
    use_immediate_repetition_penalty=True,
    immediate_repetition_weight=0.5,
    # EOS penalties
    eos_token_id=50256,
    pad_token_id=50256,
    min_sequence_length=20,
    eos_penalty_weight=0.05
)

# During training
outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
logits = outputs.logits
hidden_states = outputs.hidden_states[-1]

loss = loss_fn(
    logits=logits,
    targets=targets,
    attention_mask=attention_mask,
    hidden_states=hidden_states
)
loss.backward()
```

### Adaptive MTP Loss

```python
from Ava.losses import UnifiedLoss

# Create Adaptive MTP as primary loss
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    primary_loss_type="adaptive_mtp",
    # Adaptive MTP specific
    use_confidence_weighting=True,
    confidence_reg_strength=0.01,
    label_smoothing=0.1,
    # Repetition penalties
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True
)

# During training (requires model with MTP heads)
outputs = model(
    input_ids,
    attention_mask=attention_mask,
    return_mtp_outputs=True
)

loss = loss_fn(
    logits=outputs.primary_logits,
    targets=targets,
    attention_mask=attention_mask,
    primary_logits=outputs.primary_logits,
    additional_logits=outputs.additional_logits,
    confidence_scores=outputs.confidence_scores,
    mtp_active=outputs.mtp_active
)
loss.backward()
```

### MoE Model with Load Balancing

```python
from Ava.losses import UnifiedLoss

# Create loss for MoE model
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    primary_loss_type="deepseek",
    # MoE balancing
    num_experts=8,
    use_moe_balancing=True,
    gradient_balance_weight=0.1,
    # Advanced losses
    use_diversity_loss=True,
    diversity_weight=0.01,
    use_auxiliary_loss=True,
    load_balancing_weight=0.0001,
    router_z_weight=0.001
)

# During training
outputs = model(
    input_ids,
    attention_mask=attention_mask,
    return_expert_outputs=True
)

loss = loss_fn(
    logits=outputs.logits,
    targets=targets,
    attention_mask=attention_mask,
    gate_logits=outputs.gate_logits,
    expert_indices=outputs.expert_indices,
    expert_outputs=outputs.expert_outputs
)
loss.backward()
```

## Configuration Options

### Primary Loss Type

- `"standard"`: Standard cross-entropy loss
- `"deepseek"`: Temperature-scaled cross-entropy with adaptive temperature
- `"adaptive_mtp"`: Adaptive MTP as primary loss

### Temperature Scaling

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    initial_temperature=1.0,      # Starting temperature
    adaptive_temperature=True,     # Adapt temperature during training
    label_smoothing=0.1           # Label smoothing factor
)
```

### Multi-Token Prediction

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    use_mtp=True,                 # Enable MTP
    num_future_tokens=3,          # Number of future tokens to predict
    mtp_weight=0.1,               # Weight for MTP loss
    mtp_type="deepseek"           # "deepseek" or "adaptive"
)
```

### Repetition Penalties

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    # N-gram repetition
    use_ngram_penalty=True,
    ngram_size=4,                 # Size of n-grams to track
    ngram_penalty_weight=0.1,
    # Immediate repetition
    use_immediate_repetition_penalty=True,
    immediate_repetition_weight=0.5,
    # EOS penalties
    eos_token_id=50256,
    min_sequence_length=20,
    eos_penalty_weight=0.05
)
```

### MoE Balancing

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    num_experts=8,
    use_moe_balancing=True,
    gradient_balance_weight=0.1   # Weight for gradient-based balancing
)
```

### Advanced Losses

```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    # Focal loss
    use_focal_loss=True,
    focal_alpha=1.0,
    focal_gamma=2.0,
    # Diversity loss
    use_diversity_loss=True,
    diversity_weight=0.01,
    # Auxiliary losses
    use_auxiliary_loss=True,
    load_balancing_weight=0.0001,
    router_z_weight=0.001
)
```

## Getting Detailed Loss Breakdown

```python
# Get detailed loss components
loss_dict = loss_fn(
    logits=logits,
    targets=targets,
    attention_mask=attention_mask,
    return_detailed=True
)

print(f"Total Loss: {loss_dict['total_loss']}")
print(f"Main Loss: {loss_dict['main_loss']}")
print(f"MTP Loss: {loss_dict.get('mtp_loss', 'N/A')}")
print(f"N-gram Penalty: {loss_dict.get('ngram_penalty', 'N/A')}")
print(f"Immediate Repetition: {loss_dict.get('immediate_repetition_penalty', 'N/A')}")

# For backward pass, use total_loss
loss_dict['total_loss'].backward()
```

## Using with Configuration Files

```python
from Ava.losses import create_unified_loss

# Load configuration from dict or config file
loss_config = {
    'vocab_size': 50257,
    'hidden_size': 768,
    'primary_loss_type': 'deepseek',
    'use_mtp': True,
    'num_future_tokens': 3,
    'use_ngram_penalty': True,
    'eos_token_id': 50256,
    'pad_token_id': 50256
}

loss_fn = create_unified_loss(loss_config)
```

## Best Practices

### For Standard Training

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    primary_loss_type="deepseek",
    initial_temperature=1.0,
    adaptive_temperature=True,
    label_smoothing=0.1,
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
```

### For MoE Models

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    hidden_size=model_config.hidden_size,
    primary_loss_type="deepseek",
    num_experts=8,
    use_moe_balancing=True,
    gradient_balance_weight=0.1,
    use_diversity_loss=True,
    use_ngram_penalty=True
)
```

### For Models with Repetition Issues

```python
loss_fn = UnifiedLoss(
    vocab_size=vocab_size,
    primary_loss_type="deepseek",
    # Stronger repetition penalties
    use_ngram_penalty=True,
    ngram_penalty_weight=0.2,  # Increased
    use_immediate_repetition_penalty=True,
    immediate_repetition_weight=1.0,  # Increased
    # EOS penalties
    eos_token_id=tokenizer.eos_token_id,
    min_sequence_length=30,
    eos_penalty_weight=0.1  # Increased
)
```

## Monitoring Loss Statistics

```python
# Get statistics from loss components
stats = loss_fn.get_loss_statistics()

print("Adaptive MTP Stats:", stats.get('adaptive_mtp'))
print("MoE Balancing Stats:", stats.get('moe_balancing'))

# Reset statistics periodically
loss_fn.reset_statistics()
```

## Integration with Existing Code

The unified loss is designed to be a drop-in replacement:

### Before (old code):
```python
loss_fn = DeepSeekLoss(vocab_size=50257, hidden_size=768)
loss = loss_fn(logits, targets, hidden_states, attention_mask)
```

### After (unified loss):
```python
loss_fn = UnifiedLoss(
    vocab_size=50257,
    hidden_size=768,
    primary_loss_type="deepseek"
)
loss = loss_fn(logits, targets, attention_mask=attention_mask, hidden_states=hidden_states)
```

## Troubleshooting

### Issue: "hidden_size required for multi-token prediction"
**Solution**: Provide `hidden_size` parameter when enabling MTP:
```python
loss_fn = UnifiedLoss(vocab_size=50257, hidden_size=768, use_mtp=True)
```

### Issue: MTP loss is 0 or not computed
**Solution**: Ensure `hidden_states` is passed to forward():
```python
loss = loss_fn(logits, targets, hidden_states=outputs.hidden_states[-1])
```

### Issue: MoE balancing not working
**Solution**: Ensure `gate_logits` and `expert_indices` are passed:
```python
loss = loss_fn(
    logits, targets,
    gate_logits=outputs.gate_logits,
    expert_indices=outputs.expert_indices
)
```

## Migration Guide

### From DeepSeekLoss:
```python
# Old
from Ava.losses import DeepSeekLoss
loss_fn = DeepSeekLoss(vocab_size, hidden_size)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(vocab_size, hidden_size, primary_loss_type="deepseek")
```

### From AdaptiveMTPLoss:
```python
# Old
from Ava.losses import AdaptiveMTPLoss
loss_fn = AdaptiveMTPLoss(vocab_size)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(vocab_size, hidden_size, primary_loss_type="adaptive_mtp")
```

### From AntiRepetitionLoss:
```python
# Old
from Ava.losses import AntiRepetitionLoss
loss_fn = AntiRepetitionLoss(vocab_size, eos_token_id, pad_token_id)

# New
from Ava.losses import UnifiedLoss
loss_fn = UnifiedLoss(
    vocab_size,
    primary_loss_type="standard",
    use_ngram_penalty=True,
    use_immediate_repetition_penalty=True,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id
)
```

## Performance Tips

1. **Disable unused components**: Only enable loss components you need
2. **Use return_detailed=False**: In production, avoid detailed breakdown for performance
3. **Batch size**: Repetition penalties work best with larger batch sizes
4. **Gradient accumulation**: Works seamlessly with gradient accumulation
5. **Mixed precision**: Compatible with AMP/bfloat16 training

## Support

For issues or questions:
- Check the USAGE_GUIDE.md (this file)
- Review individual loss module documentation
- Check training pipeline integration examples
