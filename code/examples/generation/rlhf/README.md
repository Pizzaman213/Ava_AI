# RLHF Generation Examples

This directory contains examples for generating text using RLHF (Reinforcement Learning from Human Feedback) trained models.

## 📋 Available Examples

### 1. **rlhf_simple.py**
Basic RLHF generation example showing how to use a model trained with RLHF.

```bash
python rlhf_simple.py
```

### 2. **rlhf_generation.py**
Advanced RLHF generation with reward model integration and preference-based sampling.

```bash
python rlhf_generation.py --model-path path/to/rlhf/model
```

### 3. **rlhf_working_example.py**
Complete working example showing the full RLHF generation pipeline with:
- Reward model loading
- Preference-based ranking
- Best-of-N sampling
- Safety filtering

```bash
python rlhf_working_example.py
```

### 4. **rlhf_example.py**
Comprehensive example demonstrating various RLHF generation strategies.

```bash
python rlhf_example.py
```

## 🚀 Quick Start

### Basic RLHF Generation
```python
from src.model.moe_transformer import MoEForCausalLM
from src.rlhf.reward_model import RewardModel
from transformers import AutoTokenizer

# Load RLHF-trained model
model = MoEForCausalLM.from_pretrained("outputs/rlhf_model")
reward_model = RewardModel.from_pretrained("outputs/reward_model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate with reward guidance
prompt = "Explain machine learning"
outputs = model.generate(
    tokenizer(prompt, return_tensors="pt").input_ids,
    max_length=200,
    num_return_sequences=4,
    temperature=0.8
)

# Rank by reward
rewards = []
for output in outputs:
    text = tokenizer.decode(output, skip_special_tokens=True)
    reward = reward_model.compute_reward(prompt, text)
    rewards.append((text, reward))

# Select best response
best_response = max(rewards, key=lambda x: x[1])[0]
print(best_response)
```

## 📊 RLHF Generation Features

### Preference-Based Sampling
- Generate multiple candidates
- Rank by reward model scores
- Select based on human preferences

### Safety Filtering
- Constitutional AI integration
- Harmful content filtering
- Bias reduction

### Advanced Techniques
- Best-of-N sampling
- Reward-weighted decoding
- Preference-conditioned generation

## 🛠️ Configuration Options

### Generation Parameters
```python
generation_config = {
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 4,
    "do_sample": True,
    "reward_threshold": 0.5,  # Minimum reward score
    "safety_filter": True,     # Enable safety checks
}
```

### Reward Model Settings
```python
reward_config = {
    "model_path": "outputs/reward_model",
    "batch_size": 8,
    "normalize_rewards": True,
    "temperature": 0.1,  # For reward distribution
}
```

## 📈 Performance Tips

1. **Batch Generation**: Generate multiple candidates in parallel
2. **Caching**: Cache reward computations for common prompts
3. **Early Stopping**: Stop generation if reward drops below threshold
4. **Ensemble Methods**: Use multiple reward models for robustness

## 🔍 Monitoring

When using RLHF generation, monitor:
- Average reward scores
- Generation diversity
- Safety filter triggers
- Response quality metrics

## 📚 Related Resources

- [RLHF Training Examples](../../fine_tuning/)
- [RLHF Usage Guide](rlhf_usage.md)
- [Main Generation README](../README.md)

## ⚠️ Important Notes

1. RLHF models require both the base model and reward model
2. Generation is slower due to reward computation
3. Memory usage is higher (two models loaded)
4. Results depend heavily on reward model quality

For more details on training RLHF models, see the [fine-tuning examples](../../fine_tuning/).