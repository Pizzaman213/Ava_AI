# RLHF Fine-tuning

RLHF (Reinforcement Learning from Human Feedback) fine-tuning using PPO with model-to-model rating.

## Quick Start

```bash
# 1. Prepare prompts file
mkdir -p /project/code/data/rlhf
echo '{"prompts": ["Explain AI:", "What is ML?", "Define NLP:"]}' > /project/code/data/rlhf/prompts.json

# 2. Run RLHF training with your existing config
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml

# 3. With custom models
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --policy-model /path/to/model \
  --judge-model /path/to/judge
```

## Configuration

The RLHF settings are now integrated into your existing config file (`configs/gpu/small.yaml`).

Key settings in the `rlhf:` section:
- `policy_model_path`: Model to fine-tune
- `judge_model_path`: Model that rates responses (can be same as policy)
- `use_model_to_model_reward`: Use judge-based rating (default: true)
- `prompt_dataset_path`: Path to prompts JSON file

PPO settings in `rlhf.ppo:`:
- `learning_rate`: Very low LR (5e-7 for small model)
- `clip_range`: PPO clipping (0.2)
- `target_kl`: Target KL divergence (0.01)
- `max_gen_length`: Max tokens to generate (128)

## How It Works

1. **Policy Model** generates responses to prompts
2. **Judge Model** rates response quality (0-10 scale)
3. **PPO Algorithm** updates policy to maximize rewards
4. **KL Penalty** keeps model close to reference model

## Files Created

- `src/Ava/rlhf/reward_model.py` - Reward models and model-to-model rating
- `src/Ava/rlhf/ppo_trainer.py` - PPO training algorithm
- `src/Ava/rlhf/rlhf_trainer.py` - Main RLHF training orchestration
- `scripts/6_rhlf_Finetuning/train_rlhf.py` - Training entry point
- `configs/gpu/small.yaml` - Extended with RLHF settings

## Features

✅ Model-to-model rating (judge evaluates policy)
✅ PPO with adaptive KL penalty
✅ Compatible with existing training configs
✅ Memory-optimized for RTX 3090 Ti
✅ W&B logging support
✅ Checkpointing and resume

## Tips

- Start with low learning rate (5e-7)
- Monitor KL divergence (keep < 0.01)
- Use same config as your pre-trained model
- Prepare diverse prompts for best results
