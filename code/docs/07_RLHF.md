# RLHF Training Guide

Reinforcement Learning from Human Feedback (RLHF) training with Ava.

## Overview

RLHF aligns language models with human preferences through:

1. **Supervised Fine-Tuning (SFT)**: Train on curated examples
2. **Reward Model Training**: Learn human preferences
3. **PPO Training**: Optimize policy against reward model

```
SFT Model → Reward Model → PPO Training → Aligned Model
               ↑                ↓
         Human Preferences    Feedback Loop
```

## Quick Start

### 1. Prepare Prompts

```bash
python code/scripts/6_rhlf_Finetuning/prepare_prompts.py \
    --input data/raw_prompts.jsonl \
    --output data/rlhf_prompts.jsonl
```

### 2. Train RLHF

```bash
python code/scripts/6_rhlf_Finetuning/train_rlhf.py \
    --config code/configs/rlhf/rlhf_config.yaml
```

## RLHF Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      RLHF Pipeline                          │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  Policy Model │  │ Reward Model  │  │ Reference Model │  │
│  │  (Trainable)  │  │  (Frozen)     │  │   (Frozen)      │  │
│  └───────┬───────┘  └───────┬───────┘  └────────┬────────┘  │
│          │                  │                   │           │
│          ▼                  ▼                   ▼           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    PPO Trainer                         │  │
│  │  Generate → Score → Compute Advantages → Update       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Policy Model

The model being optimized:

```python
# code/src/rlhf/ppo_trainer.py
class PPOTrainer:
    def __init__(self, policy_model, reward_model, reference_model):
        self.policy = policy_model      # Updated by PPO
        self.reward = reward_model      # Frozen
        self.reference = reference_model # Frozen, for KL penalty
```

### Reward Model

Scores generations for quality:

```python
# code/src/rlhf/reward_model.py
class RewardModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Output: scalar reward per sequence
        hidden = self.backbone(input_ids, attention_mask)
        reward = self.reward_head(hidden[:, -1, :])
        return reward
```

## Configuration

### RLHF Config

```yaml
# code/configs/rlhf/rlhf_config.yaml
rlhf:
  # Model paths
  policy_model_path: "code/outputs/pretraining/best_model.pt"
  reward_model_path: "code/outputs/reward_model/best.pt"
  reference_model_path: null  # Uses policy model snapshot

  # PPO hyperparameters
  ppo_epochs: 4
  batch_size: 32
  mini_batch_size: 8
  learning_rate: 1e-5
  kl_coef: 0.1              # KL penalty coefficient
  clip_range: 0.2           # PPO clip range
  value_coef: 0.5           # Value loss coefficient
  entropy_coef: 0.01        # Entropy bonus

  # Generation settings
  max_new_tokens: 128
  temperature: 0.8
  top_p: 0.9

  # Training settings
  num_episodes: 10000
  save_frequency: 500
  eval_frequency: 100
```

### Minimal Config

```yaml
# code/configs/rlhf/rlhf_minimal.yaml
rlhf:
  policy_model_path: "path/to/model.pt"
  reward_model_path: "path/to/reward.pt"

  ppo_epochs: 2
  batch_size: 16
  learning_rate: 5e-6

  max_new_tokens: 64
  num_episodes: 1000
```

## Training Process

### PPO Training Loop

```python
for episode in range(num_episodes):
    # 1. Sample prompts from dataset
    prompts = dataloader.sample(batch_size)

    # 2. Generate responses with policy
    responses = policy.generate(prompts, max_new_tokens=128)

    # 3. Score with reward model
    rewards = reward_model(prompts + responses)

    # 4. Compute KL penalty (stay close to reference)
    kl_penalty = compute_kl(policy, reference, prompts + responses)
    adjusted_rewards = rewards - kl_coef * kl_penalty

    # 5. Compute advantages using GAE
    advantages = compute_gae(adjusted_rewards, values)

    # 6. PPO update
    for _ in range(ppo_epochs):
        policy_loss = ppo_loss(advantages, old_probs, new_probs)
        value_loss = mse_loss(values, returns)
        entropy_loss = -entropy(new_probs)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
        loss.backward()
        optimizer.step()
```

### Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `kl_coef` | 0.01-0.5 | Higher = more conservative updates |
| `clip_range` | 0.1-0.3 | PPO clipping range |
| `ppo_epochs` | 1-8 | Optimization passes per batch |
| `learning_rate` | 1e-6 to 1e-4 | Lower than SFT typically |

## Reward Model Training

### Data Format

```json
{
    "prompt": "Write a poem about nature",
    "chosen": "The gentle breeze whispers through trees...",
    "rejected": "Nature is nice. Trees are green..."
}
```

### Training

```bash
python code/scripts/6_rhlf_Finetuning/train_reward_model.py \
    --data data/preference_pairs.jsonl \
    --base-model code/outputs/pretraining/best_model.pt \
    --output code/outputs/reward_model
```

### Loss Function

```python
def reward_loss(chosen_rewards, rejected_rewards):
    # Bradley-Terry model
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

## Prompt Preparation

### Prompt Format

```json
{
    "prompt": "Explain quantum computing in simple terms:",
    "category": "education",
    "difficulty": "medium"
}
```

### Prompt Filtering

```bash
python code/scripts/6_rhlf_Finetuning/prepare_prompts.py \
    --input raw_prompts.jsonl \
    --output filtered_prompts.jsonl \
    --min-length 10 \
    --max-length 200 \
    --filter-duplicates
```

## Evaluation

### Automatic Metrics

- **Reward Score**: Average reward from reward model
- **KL Divergence**: Distance from reference model
- **Generation Length**: Response length statistics
- **Perplexity**: Language model perplexity

### Human Evaluation

```bash
python code/scripts/6_rhlf_Finetuning/eval_rlhf.py \
    --model code/outputs/rlhf/final_model.pt \
    --prompts data/eval_prompts.jsonl \
    --output eval_results.jsonl
```

## Best Practices

### Reward Hacking Prevention

1. **KL Penalty**: Keep model close to reference

```yaml
rlhf:
  kl_coef: 0.2  # Increase if model diverges too much
```

2. **Reward Clipping**: Prevent extreme rewards

```python
rewards = torch.clamp(rewards, -10, 10)
```

3. **Diverse Prompts**: Train on varied prompt distribution

### Stability Tips

1. **Lower Learning Rate**: 1e-6 to 1e-5 typical
2. **Warmup**: Gradual learning rate increase
3. **Gradient Clipping**: Max norm 0.5-1.0
4. **Small Batch Updates**: Multiple mini-batches per batch

### Memory Optimization

For large models:

```yaml
rlhf:
  # Use gradient checkpointing
  gradient_checkpointing: true

  # Offload reference model to CPU
  offload_reference: true

  # Reduce generation batch size
  generation_batch_size: 8
```

## Common Issues

### Reward Collapse

Symptoms: All generations receive same reward

Solutions:
1. Check reward model quality
2. Increase prompt diversity
3. Reduce learning rate

### KL Explosion

Symptoms: KL divergence grows unboundedly

Solutions:
1. Increase `kl_coef`
2. Use adaptive KL controller
3. Reduce PPO epochs

### Poor Generation Quality

Symptoms: Model outputs degrade

Solutions:
1. Verify reward model alignment
2. Add entropy bonus
3. Check for mode collapse

## Advanced Topics

### PPO-ptx (Mixed Training)

Combine RLHF with continued pretraining:

```yaml
rlhf:
  use_ptx: true
  ptx_coef: 0.1
  ptx_data: "data/pretraining_samples.jsonl"
```

### Constitutional AI

Self-improvement without human labels:

```python
# Generate critiques
critique = model.generate(f"Critique this response: {response}")

# Generate revision
revised = model.generate(f"Improve based on: {critique}")

# Train on (original, revised) pairs
```

### Direct Preference Optimization (DPO)

Simpler alternative to PPO:

```python
# Skip reward model, train directly on preferences
loss = -log_sigmoid(beta * (log_pi_chosen - log_pi_rejected))
```

## Testing

### CPU Testing

```bash
python code/scripts/6_rhlf_Finetuning/test_rlhf_cpu.py
```

### Training Test

```bash
python code/scripts/6_rhlf_Finetuning/test_training_cpu.py
```

## Next Steps

- [Training Guide](./03_TRAINING_GUIDE.md) - Pre-training basics
- [Configuration Reference](./04_CONFIGURATION.md) - All parameters
- [Troubleshooting](./09_TROUBLESHOOTING.md) - Common issues
