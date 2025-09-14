# RLHF (Reinforcement Learning from Human Feedback) Pipeline

## Table of Contents
- [Overview](#overview)
- [PPO Training](#ppo-training)
- [DPO Training](#dpo-training)
- [Constitutional AI](#constitutional-ai)
- [Reward Modeling](#reward-modeling)
- [Data Collection](#data-collection)
- [Training Strategies](#training-strategies)
- [Evaluation](#evaluation)

## Overview

Our RLHF pipeline implements three complementary approaches:

1. **PPO (Proximal Policy Optimization)**: Classic RLHF with reward models
2. **DPO (Direct Preference Optimization)**: Simplified RLHF without reward models
3. **Constitutional AI**: Self-supervised harmlessness training

### Pipeline Architecture

```
Human Preferences
       ↓
┌─────────────────────────────────────┐
│        Data Collection              │
│  • Comparisons                      │
│  • Constitutional Critiques         │
│  • Multi-objective Labels           │
└─────────────────────────────────────┘
       ↓                    ↓
┌──────────────┐    ┌────────────────┐
│Reward Model  │    │ DPO Dataset    │
│Training      │    │ Preparation    │
└──────────────┘    └────────────────┘
       ↓                    ↓
┌──────────────┐    ┌────────────────┐
│PPO Training  │    │ DPO Training   │
└──────────────┘    └────────────────┘
       ↓                    ↓
┌─────────────────────────────────────┐
│     Constitutional AI Refinement    │
└─────────────────────────────────────┘
       ↓
  Aligned Model
```

## PPO Training

### Basic PPO Setup

```python
from moe_llm.rlhf import PPOTrainer, PPOConfig

# Configure PPO
ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=128,
    mini_batch_size=4,
    ppo_epochs=4,
    gamma=1.0,
    lam=0.95,
    clip_ratio=0.2,
    value_loss_coef=0.1,
    entropy_coef=0.01,
    target_kl=6.0,
    init_kl_coef=0.2
)

# Initialize trainer
ppo_trainer = PPOTrainer(
    policy_model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    config=ppo_config
)

# Train
ppo_trainer.train(
    train_prompts=prompts,
    eval_prompts=eval_prompts,
    output_dir="outputs/ppo"
)
```

### Advanced PPO Features

```python
# 1. Multi-objective rewards
from moe_llm.rlhf import MultiObjectiveRewardModel

reward_model = MultiObjectiveRewardModel(
    objectives=["helpfulness", "harmlessness", "honesty"],
    weights=[0.5, 0.3, 0.2]
)

# 2. Adaptive KL control
ppo_config.kl_control = "adaptive"
ppo_config.target_kl = 6.0
ppo_config.kl_horizon = 10000

# 3. Reward shaping
def reward_shaping_fn(rewards, responses):
    # Penalize repetition
    for i, response in enumerate(responses):
        unique_ratio = len(set(response.split())) / len(response.split())
        rewards[i] += 0.1 * unique_ratio
    
    # Bonus for appropriate length
    for i, response in enumerate(responses):
        length_bonus = min(len(response.split()) / 100, 1.0)
        rewards[i] += 0.05 * length_bonus
    
    return rewards

ppo_trainer.set_reward_shaping(reward_shaping_fn)
```

### PPO Monitoring

```python
# Custom callbacks
class PPOCallback:
    def on_step(self, step, logs):
        # Monitor KL divergence
        if logs["kl_divergence"] > 2 * config.target_kl:
            print(f"Warning: High KL divergence: {logs['kl_divergence']}")
        
        # Track reward statistics
        wandb.log({
            "ppo/reward_mean": logs["reward_mean"],
            "ppo/reward_std": logs["reward_std"],
            "ppo/kl_divergence": logs["kl_divergence"],
            "ppo/kl_coef": logs["kl_coef"],
            "ppo/entropy": logs["entropy"],
            "ppo/approx_kl": logs["approx_kl"],
            "ppo/clipfrac": logs["clipfrac"],
            "ppo/explained_variance": logs["explained_variance"]
        }, step=step)

ppo_trainer.add_callback(PPOCallback())
```

## DPO Training

### Basic DPO Setup

```python
from moe_llm.rlhf import DPOTrainer, DPOConfig

# Configure DPO
dpo_config = DPOConfig(
    beta=0.1,  # KL regularization
    learning_rate=5e-7,
    batch_size=4,
    gradient_accumulation_steps=4,
    loss_type="sigmoid",  # or "hinge", "ipo"
    label_smoothing=0.0,
    reference_free=False,
    max_length=512,
    max_prompt_length=128
)

# Prepare preference dataset
preference_dataset = [
    {
        "prompt": "Explain machine learning",
        "chosen": "Machine learning is a subset of AI...",
        "rejected": "Machine learning is when machines learn..."
    },
    # More examples...
]

# Initialize trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    config=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer
)

# Train
dpo_trainer.train()
```

### Advanced DPO Techniques

```python
# 1. IPO (Identity Preference Optimization)
dpo_config.loss_type = "ipo"
dpo_config.ipo_tau = 0.05

# 2. Reference-free DPO
dpo_config.reference_free = True
dpo_config.ref_model = None

# 3. Filtered DPO (f-DPO)
from moe_llm.rlhf import FilteredDPOTrainer

fdpo_trainer = FilteredDPOTrainer(
    model=model,
    ref_model=ref_model,
    config=dpo_config,
    filter_threshold=0.6,  # Only use high-confidence pairs
    filter_model=reward_model
)

# 4. Online DPO
from moe_llm.rlhf import OnlineDPOTrainer

online_dpo = OnlineDPOTrainer(
    model=model,
    ref_model=ref_model,
    config=dpo_config,
    sampling_temperature=0.8,
    num_samples_per_prompt=4
)
```

### DPO Data Augmentation

```python
# Generate synthetic preferences
from moe_llm.rlhf import PreferenceGenerator

generator = PreferenceGenerator(
    model=model,
    reward_model=reward_model,
    temperature_range=(0.7, 1.2),
    num_samples=4
)

synthetic_preferences = generator.generate_preferences(
    prompts=unlabeled_prompts,
    min_score_gap=0.2
)

# Mix real and synthetic data
combined_dataset = real_preferences + synthetic_preferences
```

## Constitutional AI

### Basic Constitutional Setup

```python
from moe_llm.rlhf import ConstitutionalAITrainer, ConstitutionalPrinciple

# Define principles
principles = [
    ConstitutionalPrinciple(
        name="helpfulness",
        description="The AI should provide useful and relevant information",
        critique_prompt="Is this response helpful and addresses the query?",
        revision_prompt="Revise to be more helpful and directly address the query."
    ),
    ConstitutionalPrinciple(
        name="harmlessness",
        description="The AI should not provide harmful information",
        critique_prompt="Could this response cause harm?",
        revision_prompt="Revise to remove any potentially harmful content."
    ),
    ConstitutionalPrinciple(
        name="honesty",
        description="The AI should be honest about its limitations",
        critique_prompt="Is this response overconfident or misleading?",
        revision_prompt="Revise to be more honest about uncertainties."
    )
]

# Initialize trainer
constitutional_trainer = ConstitutionalAITrainer(
    model=model,
    tokenizer=tokenizer,
    principles=principles,
    base_trainer=dpo_trainer  # Use DPO as base
)

# Train with constitutional AI
constitutional_trainer.train(
    train_prompts=prompts,
    num_rounds=3,
    prompts_per_round=1000
)
```

### Advanced Constitutional AI

```python
# 1. Custom principles from file
principles = ConstitutionalPrinciple.from_json("principles.json")

# 2. Hierarchical principles
from moe_llm.rlhf import HierarchicalConstitution

constitution = HierarchicalConstitution([
    {
        "category": "safety",
        "weight": 2.0,
        "principles": [harmlessness, privacy, fairness]
    },
    {
        "category": "quality",
        "weight": 1.0,
        "principles": [helpfulness, accuracy, coherence]
    }
])

# 3. Dynamic constitutional training
constitutional_trainer.train_dynamic(
    train_prompts=prompts,
    adjust_principles_every=1000,
    principle_selection="adaptive"  # Focus on violated principles
)

# 4. Constitutional self-play
from moe_llm.rlhf import ConstitutionalSelfPlay

self_play = ConstitutionalSelfPlay(
    actor_model=model,
    critic_model=critic_model,
    principles=principles,
    num_iterations=5
)

improved_model = self_play.run()
```

### Constitutional Evaluation

```python
# Evaluate constitutional adherence
from moe_llm.rlhf import ConstitutionalEvaluator

evaluator = ConstitutionalEvaluator(
    model=model,
    principles=principles,
    num_samples=1000
)

results = evaluator.evaluate(test_prompts)

print("Constitutional Scores:")
for principle, score in results.items():
    print(f"{principle}: {score:.2%}")
```

## Reward Modeling

### Training Reward Models

```python
from moe_llm.rlhf import RewardModelTrainer, RewardModelConfig

# Configure reward model
reward_config = RewardModelConfig(
    base_model_name="bert-base-uncased",
    num_objectives=3,
    learning_rate=2e-5,
    batch_size=32,
    loss_type="ranking",
    label_smoothing=0.1
)

# Prepare training data
reward_training_data = [
    {
        "text_a": "Response A to prompt",
        "text_b": "Response B to prompt",
        "labels": {
            "helpfulness": 1.0,  # A is more helpful
            "harmlessness": 0.0,  # B is more harmless
            "honesty": 0.5  # Equally honest
        }
    },
    # More examples...
]

# Train reward model
reward_trainer = RewardModelTrainer(
    model=create_reward_model(reward_config),
    config=reward_config,
    train_dataset=reward_training_data
)

reward_trainer.train()
```

### Multi-Objective Rewards

```python
# 1. Weighted combination
from moe_llm.rlhf import WeightedRewardModel

weighted_reward = WeightedRewardModel(
    reward_models={
        "helpfulness": helpfulness_model,
        "harmlessness": harmlessness_model,
        "honesty": honesty_model
    },
    weights=[0.5, 0.3, 0.2]
)

# 2. Pareto-optimal rewards
from moe_llm.rlhf import ParetoRewardModel

pareto_reward = ParetoRewardModel(
    reward_models=[model1, model2, model3],
    pareto_front_size=50
)

# 3. Conditional rewards
from moe_llm.rlhf import ConditionalRewardModel

conditional_reward = ConditionalRewardModel(
    reward_models=reward_models,
    condition_fn=lambda prompt: "code" in prompt,
    conditional_weights={
        True: [0.3, 0.2, 0.5],  # Weights for code
        False: [0.5, 0.3, 0.2]  # Weights for non-code
    }
)
```

### Reward Model Ensemble

```python
from moe_llm.rlhf import RewardEnsemble

# Create ensemble
ensemble = RewardEnsemble(
    models=[reward1, reward2, reward3],
    aggregation="mean",  # or "median", "trimmed_mean"
    confidence_weighted=True
)

# Calibrate ensemble
ensemble.calibrate(calibration_data)

# Use for scoring
scores = ensemble.score(responses)
```

## Data Collection

### Human Preference Collection

```python
from moe_llm.rlhf import PreferenceCollector

# Initialize collector
collector = PreferenceCollector(
    interface="gradio",  # or "streamlit", "api"
    save_path="preferences.jsonl"
)

# Create annotation interface
interface = collector.create_interface(
    model_a=model_a,
    model_b=model_b,
    prompts=annotation_prompts,
    criteria=["helpfulness", "harmlessness", "honesty"]
)

# Launch interface
interface.launch(share=True)
```

### Active Learning

```python
from moe_llm.rlhf import ActivePreferenceLearning

active_learner = ActivePreferenceLearning(
    model=model,
    uncertainty_method="entropy",
    selection_strategy="diverse",
    budget=1000
)

# Select prompts for annotation
selected_prompts = active_learner.select_prompts(
    candidate_prompts,
    num_select=100
)

# Generate comparisons
comparisons = active_learner.generate_comparisons(
    selected_prompts,
    num_samples_per_prompt=4,
    temperature_range=(0.7, 1.2)
)
```

### Synthetic Preference Generation

```python
from moe_llm.rlhf import SyntheticPreferenceGenerator

# Using LLM as judge
llm_judge = SyntheticPreferenceGenerator(
    judge_model="gpt-4",
    criteria="helpful and harmless",
    num_workers=10
)

synthetic_prefs = llm_judge.generate(
    prompts=prompts,
    model=model,
    num_samples_per_prompt=4
)

# Using constitutional AI for preferences
constitutional_judge = SyntheticPreferenceGenerator(
    judge_model=model,  # Self-judge
    criteria=constitutional_principles,
    use_critiques=True
)
```

## Training Strategies

### Curriculum RLHF

```python
from moe_llm.rlhf import CurriculumRLHF

# Define curriculum stages
curriculum = CurriculumRLHF([
    {
        "name": "basic_alignment",
        "data": basic_preferences,
        "method": "dpo",
        "steps": 10000
    },
    {
        "name": "safety_focus",
        "data": safety_preferences,
        "method": "constitutional",
        "steps": 5000
    },
    {
        "name": "advanced_reasoning",
        "data": reasoning_preferences,
        "method": "ppo",
        "steps": 20000
    }
])

# Train with curriculum
curriculum.train(model, ref_model)
```

### Iterative RLHF

```python
from moe_llm.rlhf import IterativeRLHF

iterative_trainer = IterativeRLHF(
    model=model,
    ref_model=ref_model,
    num_iterations=5,
    samples_per_iteration=10000
)

for iteration in range(5):
    # Generate new samples
    samples = iterative_trainer.generate_samples()
    
    # Collect preferences (real or synthetic)
    preferences = collect_preferences(samples)
    
    # Update model
    iterative_trainer.update(preferences)
    
    # Evaluate
    metrics = iterative_trainer.evaluate()
    print(f"Iteration {iteration}: {metrics}")
```

### Mixed Training

```python
from moe_llm.rlhf import MixedRLHFTrainer

# Combine multiple RLHF methods
mixed_trainer = MixedRLHFTrainer(
    model=model,
    ref_model=ref_model,
    methods={
        "ppo": {"weight": 0.4, "config": ppo_config},
        "dpo": {"weight": 0.4, "config": dpo_config},
        "constitutional": {"weight": 0.2, "config": const_config}
    }
)

# Alternate between methods
mixed_trainer.train(
    strategy="round_robin",  # or "weighted_sampling", "adaptive"
    total_steps=100000
)
```

## Evaluation

### RLHF Metrics

```python
from moe_llm.rlhf import RLHFEvaluator

evaluator = RLHFEvaluator(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model
)

# Compute comprehensive metrics
metrics = evaluator.evaluate(
    test_prompts,
    metrics=[
        "reward_score",
        "kl_divergence",
        "win_rate",
        "diversity",
        "coherence",
        "constitutional_adherence"
    ]
)

print("RLHF Evaluation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

### A/B Testing

```python
from moe_llm.rlhf import ABTester

# Compare models
ab_tester = ABTester(
    model_a=baseline_model,
    model_b=rlhf_model,
    judge_model=judge_model
)

results = ab_tester.compare(
    test_prompts,
    criteria=["helpfulness", "harmlessness", "honesty"],
    num_samples=1000
)

print(f"Win rate: {results['win_rate']:.2%}")
print(f"Statistical significance: p={results['p_value']:.4f}")
```

### Human Evaluation

```python
from moe_llm.rlhf import HumanEvaluation

# Setup human evaluation
human_eval = HumanEvaluation(
    models={"baseline": baseline_model, "rlhf": rlhf_model},
    prompts=eval_prompts,
    criteria={
        "helpful": "The response is helpful and informative",
        "harmless": "The response avoids harmful content",
        "honest": "The response is honest about limitations"
    }
)

# Create evaluation interface
interface = human_eval.create_interface()
interface.launch()

# Analyze results
results = human_eval.analyze_results()
```

## Best Practices

### 1. Data Quality
- Ensure diverse, high-quality preference data
- Balance different types of preferences
- Validate human annotations
- Use multiple annotators when possible

### 2. Training Stability
- Start with small learning rates
- Monitor KL divergence carefully
- Use gradient clipping
- Implement early stopping

### 3. Multi-Objective Balance
- Carefully weight different objectives
- Monitor trade-offs between objectives
- Use Pareto-optimal solutions
- Consider conditional objectives

### 4. Evaluation
- Use multiple evaluation methods
- Include human evaluation
- Test on out-of-distribution prompts
- Monitor for reward hacking

For more details, see:
- [Training Guide](training.md)
- [Evaluation Guide](evaluation.md)
- [API Reference](api_reference.md)