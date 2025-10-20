# RLHF Fine-tuning - Usage Guide

## Overview

Complete RLHF (Reinforcement Learning from Human Feedback) implementation with:
- **Model-to-model rating**: One model judges another's outputs
- **PPO training**: Proximal Policy Optimization algorithm
- **Config integration**: Uses your existing training configuration system

## Quick Start (3 Steps)

### 1. Prepare Prompts

Option A - Use example prompts:
```bash
# Copy example prompts
cp /project/code/data/rlhf/example_prompts.json /project/code/data/rlhf/prompts.json
```

Option B - Create your own:
```bash
# From text file (one prompt per line)
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --input my_prompts.txt \
  --output /project/code/data/rlhf/prompts.json

# Create sample prompts for testing
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --create-samples \
  --num-samples 100 \
  --output /project/code/data/rlhf/prompts.json
```

### 2. Configure RLHF

Edit `configs/gpu/small.yaml` (already configured with RLHF settings):

```yaml
rlhf:
  policy_model_path: /project/code/outputs/runs/latest/model  # Your trained model
  judge_model_path: /project/code/outputs/runs/latest/model   # Can be same
  prompt_dataset_path: /project/code/data/rlhf/prompts.json

  ppo:
    learning_rate: 5.0e-7  # Very low for stability
    max_gen_length: 128
    target_kl: 0.01
```

### 3. Run Training

```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml
```

## How It Works

```
┌─────────────┐
│   Prompts   │
└──────┬──────┘
       │
       v
┌─────────────────┐
│  Policy Model   │  Generates responses
│  (to finetune)  │
└────────┬────────┘
         │
         v
    ┌────────┐
    │Response│
    └───┬────┘
        │
        v
┌────────────────┐
│  Judge Model   │  Rates quality (0-10)
│ (evaluates)    │
└────────┬───────┘
         │
         v
    ┌────────┐
    │ Reward │
    └───┬────┘
        │
        v
┌────────────────┐
│  PPO Update    │  Improves policy
└────────────────┘
```

## Command-Line Options

### Basic Usage
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py --config CONFIG_PATH
```

### With Custom Models
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --policy-model /path/to/policy/model \
  --judge-model /path/to/judge/model
```

### Resume Training
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --resume outputs/rlhf/checkpoint_step_1000.pt
```

### Disable W&B
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --no-wandb
```

## Configuration Reference

### RLHF Settings (`rlhf:`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `policy_model_path` | Model to fine-tune | Required |
| `judge_model_path` | Model that rates responses | Required |
| `use_model_to_model_reward` | Use judge-based rating | `true` |
| `prompt_dataset_path` | Path to prompts JSON | Required |
| `num_epochs` | Training epochs | `3` |
| `rollout_batch_size` | Prompts per batch | `16` |
| `save_dir` | Checkpoint directory | `outputs/rlhf/small` |

### PPO Settings (`rlhf.ppo:`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Learning rate | `5e-7` |
| `batch_size` | PPO batch size | `16` |
| `mini_batch_size` | Mini-batch size | `4` |
| `ppo_epochs` | PPO epochs per batch | `4` |
| `clip_range` | PPO clipping | `0.2` |
| `target_kl` | Target KL divergence | `0.01` |
| `entropy_coef` | Entropy bonus | `0.01` |
| `max_gen_length` | Max tokens to generate | `128` |
| `temperature` | Sampling temperature | `1.0` |
| `top_k` | Top-k sampling | `50` |
| `top_p` | Nucleus sampling | `0.95` |
| `whiten_rewards` | Normalize advantages | `true` |
| `adaptive_kl` | Adaptive KL coefficient | `true` |

## Monitoring

### W&B Metrics
- `train/reward`: Average reward per step
- `train/policy_loss`: PPO policy loss
- `train/kl_div`: KL divergence from reference
- `train/entropy`: Policy entropy (diversity)
- `train/kl_coef`: Adaptive KL coefficient
- `eval/reward_mean`: Evaluation reward mean
- `eval/reward_std`: Evaluation reward std

### Checkpoints
Saved to `save_dir`:
- `checkpoint_step_{N}.pt` - Full checkpoint with optimizer state
- `checkpoint_epoch_{N}.pt` - End-of-epoch checkpoint
- `model_step_{N}.pt` - Model weights only

## Tips & Best Practices

### 1. Start Small
- Use small `rollout_batch_size` (16)
- Low `learning_rate` (5e-7)
- Few epochs (1-3)

### 2. Monitor KL Divergence
- Keep `kl_div` < 0.01
- If too high: increase `kl_coef` or reduce `learning_rate`
- Enable `adaptive_kl: true`

### 3. Prompt Quality
- Diverse prompts = better generalization
- 100-1000 prompts recommended
- Use `prepare_prompts.py` to format

### 4. Judge Model
- Can use same model as policy
- Or use a different/larger model
- Judge quality affects training significantly

### 5. Generation Settings
- `temperature: 1.0` - balanced creativity
- `top_k: 50` - reasonable diversity
- `max_gen_length: 128` - memory-friendly

## Troubleshooting

### Policy Collapse (Reward Drops)
**Symptoms**: Model generates nonsense, rewards decrease

**Solutions**:
```yaml
ppo:
  learning_rate: 1.0e-7  # Lower LR
  target_kl: 0.005       # Stricter KL
  entropy_coef: 0.02     # More diversity
  init_kl_coef: 0.5      # Higher KL penalty
```

### Training Instability
**Symptoms**: Loss spikes, NaN values

**Solutions**:
```yaml
ppo:
  max_grad_norm: 0.3     # Stricter clipping
  clip_reward: 5.0       # Clip rewards
  whiten_rewards: true   # Normalize
```

### OOM (Out of Memory)
**Solutions**:
```yaml
rlhf:
  rollout_batch_size: 8  # Reduce batch

ppo:
  batch_size: 8
  mini_batch_size: 2
  gradient_accumulation_steps: 8
  max_gen_length: 64     # Shorter generations
```

### Poor Reward Scores
**Check**:
1. Is judge model working? Test separately
2. Are prompts good quality?
3. Is rating template appropriate?

**Solutions**:
- Improve prompt quality
- Use better judge model
- Adjust rating scale/template

## Examples

### Example 1: Basic RLHF
```bash
# Prepare prompts
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --create-samples --num-samples 100

# Train
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml
```

### Example 2: From Text File
```bash
# Convert text to JSON
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --input my_questions.txt \
  --output /project/code/data/rlhf/prompts.json

# Train
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml
```

### Example 3: Separate Judge Model
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --policy-model outputs/my_model \
  --judge-model outputs/large_judge_model
```

### Example 4: Resume Training
```bash
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --resume outputs/rlhf/checkpoint_step_1000.pt
```

## Files Created

### Source Code
- `src/Ava/rlhf/__init__.py` - Package init
- `src/Ava/rlhf/reward_model.py` - Reward models (500 lines)
- `src/Ava/rlhf/ppo_trainer.py` - PPO algorithm (600 lines)
- `src/Ava/rlhf/rlhf_trainer.py` - Main trainer (400 lines)

### Scripts
- `scripts/6_rhlf_Finetuning/train_rlhf.py` - Training entry point
- `scripts/6_rhlf_Finetuning/prepare_prompts.py` - Prompt preparation utility
- `scripts/6_rhlf_Finetuning/README.md` - Quick reference
- `scripts/6_rhlf_Finetuning/USAGE.md` - This file

### Configuration
- `configs/gpu/small.yaml` - Updated with RLHF settings

### Example Data
- `data/rlhf/example_prompts.json` - Sample prompts

## Architecture

```
src/Ava/rlhf/
├── __init__.py           # Package exports
├── reward_model.py       # Reward models
│   ├── RewardModel       # Base reward model
│   ├── ModelToModelReward # Judge-based rating
│   └── EnsembleRewardModel # Multiple reward models
├── ppo_trainer.py        # PPO algorithm
│   ├── PPOConfig         # PPO configuration
│   └── PPOTrainer        # PPO training loop
└── rlhf_trainer.py       # Main orchestration
    ├── RLHFConfig        # RLHF configuration
    └── RLHFTrainer       # Full training pipeline
```

## Next Steps

1. **Prepare your prompts** using `prepare_prompts.py`
2. **Update config** in `configs/gpu/small.yaml`
3. **Run training** with `train_rlhf.py`
4. **Monitor** progress in W&B
5. **Evaluate** the fine-tuned model

## Support

For issues:
1. Check troubleshooting section above
2. Review configuration in `configs/gpu/small.yaml`
3. Check logs in `logs/rlhf/`
4. Verify prompts format in `data/rlhf/prompts.json`

## References

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - Original RLHF paper
- [PPO Paper](https://arxiv.org/abs/1707.06347) - PPO algorithm
- [RLHF Survey](https://arxiv.org/abs/2304.05302) - Comprehensive RLHF overview
