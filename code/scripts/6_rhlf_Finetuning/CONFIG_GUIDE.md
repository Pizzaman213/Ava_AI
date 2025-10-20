# RLHF Configuration Guide

All GPU configs now include RLHF settings! Each config is optimized for its model size.

## Available Configurations

### 1. Tiny Model (`configs/gpu/tiny.yaml`)
**Best for**: Fast experimentation, testing

```yaml
rlhf:
  rollout_batch_size: 24
  ppo:
    learning_rate: 1.0e-6
    batch_size: 24
    mini_batch_size: 6
    max_gen_length: 128
    gradient_checkpointing: false  # Not needed for tiny
```

**Usage**:
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/tiny.yaml
```

---

### 2. Small Model (`configs/gpu/small.yaml`)
**Best for**: Production on single RTX 3090 Ti

```yaml
rlhf:
  rollout_batch_size: 16
  ppo:
    learning_rate: 5.0e-7  # Very conservative
    batch_size: 16
    mini_batch_size: 4
    max_gen_length: 128
    gradient_checkpointing: true
```

**Usage**:
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/small.yaml
```

---

### 3. Base Model (`configs/gpu/base.yaml`)
**Best for**: Balanced performance

```yaml
rlhf:
  rollout_batch_size: 16
  max_prompt_length: 1024
  ppo:
    learning_rate: 8.0e-7
    batch_size: 16
    mini_batch_size: 4
    max_gen_length: 256
    gradient_checkpointing: true
```

**Usage**:
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/base.yaml
```

---

### 4. Large Model (`configs/gpu/large.yaml`)
**Best for**: Maximum quality (requires more memory)

```yaml
rlhf:
  rollout_batch_size: 8  # Smaller batches
  num_epochs: 2  # Fewer epochs
  ppo:
    learning_rate: 3.0e-7  # Very low LR
    batch_size: 8
    mini_batch_size: 2
    gradient_accumulation_steps: 8
    max_grad_norm: 0.3  # Strict clipping
    target_kl: 0.005  # Stricter KL
    gradient_checkpointing: true  # Essential
```

**Usage**:
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/large.yaml
```

---

## Quick Comparison

| Config | Rollout Batch | PPO Batch | Learning Rate | Gen Length | Grad Checkpoint |
|--------|--------------|-----------|---------------|------------|----------------|
| Tiny   | 24           | 24        | 1.0e-6        | 128        | No             |
| Small  | 16           | 16        | 5.0e-7        | 128        | Yes            |
| Base   | 16           | 16        | 8.0e-7        | 256        | Yes            |
| Large  | 8            | 8         | 3.0e-7        | 256        | Yes            |

## Common Settings (All Configs)

All configs share these defaults:

```yaml
rlhf:
  use_model_to_model_reward: true
  freeze_reward_model: true
  num_epochs: 2-3

  ppo:
    ppo_epochs: 4
    clip_range: 0.2
    target_kl: 0.01 (0.005 for large)
    entropy_coef: 0.01
    adaptive_kl: true
    whiten_rewards: true
```

## Customization Tips

### Adjust Memory Usage

If OOM (Out of Memory):
```yaml
rlhf:
  rollout_batch_size: 8  # Reduce
  ppo:
    batch_size: 8
    mini_batch_size: 2
    gradient_accumulation_steps: 16  # Increase
```

### Make Training Faster

```yaml
rlhf:
  num_epochs: 1  # Reduce epochs
  rollout_batch_size: 32  # Increase if memory allows
  eval_every: 500  # Evaluate less frequently
```

### More Stable Training

```yaml
rlhf:
  ppo:
    learning_rate: 1.0e-7  # Lower LR
    target_kl: 0.005  # Stricter KL
    init_kl_coef: 0.5  # Higher KL penalty
    max_grad_norm: 0.3  # Stricter clipping
```

### Longer Responses

```yaml
rlhf:
  ppo:
    max_gen_length: 512  # Increase
    temperature: 1.2  # More creative
```

## Model-Specific Notes

### Tiny
- Fast iterations for testing
- No gradient checkpointing needed
- Can use larger batches
- Good for prompt engineering experiments

### Small
- Production-ready for single GPU
- Conservative settings prevent collapse
- Best balance of speed/quality
- **Recommended for most users**

### Base
- Longer context (1024 tokens)
- Longer generations (256 tokens)
- Good for complex tasks

### Large
- Most conservative settings
- Requires careful tuning
- Save checkpoints frequently
- Monitor KL divergence closely

## Complete Example

```bash
# 1. Choose your config based on model size
CONFIG=configs/gpu/small.yaml

# 2. Prepare prompts
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --create-samples --num-samples 200 \
  --output /project/code/data/rlhf/prompts.json

# 3. Update model paths in config (optional)
# Edit the config file or use command-line:
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config $CONFIG \
  --policy-model /path/to/your/model \
  --judge-model /path/to/judge/model

# 4. Run training
python scripts/6_rhlf_Finetuning/train_rlhf.py --config $CONFIG

# 5. Monitor in W&B
# Check: train/reward, train/kl_div, eval/reward_mean
```

## Troubleshooting by Config

### Tiny: "Training too unstable"
```yaml
ppo:
  learning_rate: 5.0e-7  # Lower from 1e-6
```

### Small: "OOM errors"
```yaml
rlhf:
  rollout_batch_size: 8  # Reduce from 16
ppo:
  batch_size: 8
  mini_batch_size: 2
```

### Base: "Rewards not improving"
```yaml
ppo:
  learning_rate: 1.0e-6  # Increase from 8e-7
  max_gen_length: 128  # Reduce from 256
```

### Large: "Policy collapse"
```yaml
ppo:
  learning_rate: 1.0e-7  # Lower from 3e-7
  target_kl: 0.003  # Stricter from 0.005
  init_kl_coef: 0.5  # Higher from 0.3
```

## Files Modified

✓ `configs/gpu/tiny.yaml` - RLHF section added
✓ `configs/gpu/small.yaml` - RLHF section added
✓ `configs/gpu/base.yaml` - RLHF section added
✓ `configs/gpu/large.yaml` - RLHF section added

All configs are ready to use with the RLHF training script!
