# Fine-tuning Examples for MoE++ Model

This directory contains executable scripts for fine-tuning the MoE++ model using various techniques. All scripts are ready to run with appropriate command-line arguments.

## Executable Scripts

- **`supervised_finetuning.py`** - Standard supervised fine-tuning with LoRA support
- **`supervised_finetuning_optimized.py`** - Optimized SFT with WandB integration and all optimizations
- **`dpo_training.py`** - Direct Preference Optimization training  
- **`ppo_training.py`** - PPO reinforcement learning from human feedback
- **`constitutional_ai.py`** - Constitutional AI for self-supervised harmlessness

## Quick Start

```bash
# Navigate to the fine-tuning directory
cd examples/fine_tuning

# Run supervised fine-tuning
python supervised_finetuning.py --checkpoint ../../outputs/checkpoints/best --dataset tatsu-lab/alpaca

# Run optimized supervised fine-tuning with WandB
python supervised_finetuning_optimized.py \
  --dataset databricks/databricks-dolly-15k \
  --batch-size 8 \
  --wandb \
  --wandb-project "moe-finetune"

# Run DPO training
python dpo_training.py --model-path ../../outputs/checkpoints/best --dataset Anthropic/hh-rlhf

# Run PPO training with sentiment reward
python ppo_training.py --model-path ../../outputs/checkpoints/best --reward-model-path sentiment --prompts prompts.txt

# Run Constitutional AI training
python constitutional_ai.py --model-path ../../outputs/checkpoints/best --dataset helpful-base --use-default-principles
```

## Table of Contents
- [Executable Scripts](#executable-scripts)
- [Supervised Fine-tuning (SFT)](#supervised-fine-tuning-sft)
- [RLHF with DPO](#rlhf-with-dpo)
- [RLHF with PPO](#rlhf-with-ppo)
- [Constitutional AI](#constitutional-ai)
- [Parameter-Efficient Fine-tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
- [Memory Optimization](#memory-optimization)
- [Complete Fine-tuning Pipeline](#complete-fine-tuning-pipeline)

## Supervised Fine-tuning (SFT)

### Using the Executable Script

```bash
# Basic supervised fine-tuning
python supervised_finetuning.py \
  --checkpoint ../../outputs/checkpoints/best \
  --dataset tatsu-lab/alpaca \
  --output-dir ../../outputs/sft_model

# With LoRA for efficient training
python supervised_finetuning.py \
  --checkpoint ../../outputs/checkpoints/best \
  --dataset your_dataset \
  --use-lora \
  --learning-rate 3e-4

# Freeze base model and only train new layers
python supervised_finetuning.py \
  --checkpoint ../../outputs/checkpoints/best \
  --dataset your_dataset \
  --freeze-base
```

### Optimized Fine-tuning with WandB

```bash
# Use the optimized version with comprehensive logging
python supervised_finetuning_optimized.py \
  --dataset databricks/databricks-dolly-15k \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --num-epochs 3 \
  --enable-all-optimizations \
  --wandb \
  --wandb-project "moe-sft-optimized"

# With LoRA and quantization
python supervised_finetuning_optimized.py \
  --model-path outputs/moe_model \
  --use-lora \
  --lora-r 32 \
  --lora-alpha 64 \
  --quantization 4bit \
  --wandb
```

### Script Arguments

```bash
python supervised_finetuning.py --help

# Key arguments:
#   --checkpoint PATH         Path to pretrained checkpoint
#   --dataset NAME/PATH       Dataset name or path
#   --max-length INT         Maximum sequence length (default: 512)
#   --learning-rate FLOAT    Learning rate (default: 5e-5)
#   --batch-size INT         Training batch size (default: 4)
#   --num-epochs INT         Number of training epochs (default: 3)
#   --use-lora              Use LoRA for parameter-efficient fine-tuning
#   --freeze-base           Freeze base model parameters
#   --gradient-checkpointing Enable gradient checkpointing
```

### Instruction Tuning

```python
from src.model.moe_transformer import MoEForCausalLM
from src.training.trainer import MoETrainer, TrainingConfig
from transformers import AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare instruction dataset
def format_instruction(example):
    return {
        "text": f"### Instruction: {example['instruction']}\n\n### Response: {example['response']}"
    }

dataset = load_dataset("your_instruction_dataset")
dataset = dataset.map(format_instruction)

# Configure training
training_config = TrainingConfig(
    model_config=model.config,
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    output_dir="outputs/instruction_tuned"
)

# Train
trainer = MoETrainer(
    model=model,
    config=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)
trainer.train()
```

### Domain-Specific Fine-tuning

```python
# Fine-tune specific experts for domain adaptation
from src.model.experts import freeze_non_expert_params

# Freeze all parameters except specific experts
model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")
freeze_non_expert_params(model, expert_ids=[0, 1, 2, 3])  # Only train first 4 experts

# Configure for domain-specific data
domain_config = TrainingConfig(
    learning_rate=1e-5,  # Lower LR for fine-tuning
    num_epochs=5,
    batch_size=8,
    max_grad_norm=0.5,  # More conservative gradient clipping
)
```

## RLHF with DPO

### Using the Executable Script

```bash
# Basic DPO training
python dpo_training.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset Anthropic/hh-rlhf \
  --output-dir ../../outputs/dpo_model

# With IPO loss
python dpo_training.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset your_preference_dataset \
  --loss-type ipo \
  --ipo-tau 0.05

# Reference-free DPO
python dpo_training.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset your_preference_dataset \
  --reference-free
```

### Script Arguments

```bash
python dpo_training.py --help

# Key arguments:
#   --model-path PATH        Path to model checkpoint
#   --dataset NAME/PATH      Preference dataset name or path
#   --beta FLOAT            KL regularization coefficient (default: 0.1)
#   --loss-type TYPE        DPO loss type: sigmoid, hinge, ipo (default: sigmoid)
#   --reference-free        Use reference-free DPO
#   --use-peft             Use parameter-efficient fine-tuning
#   --learning-rate FLOAT   Learning rate (default: 5e-7)
```

### Basic DPO Training

```python
from src.training.dpo_trainer import DPOTrainer, DPOConfig
from datasets import load_dataset

# Load preference dataset
# Expected format: {"prompt": str, "chosen": str, "rejected": str}
preference_dataset = load_dataset("your_preference_dataset")

# Configure DPO training
dpo_config = DPOConfig(
    model_name_or_path="outputs/checkpoints/best",
    ref_model_name_or_path="outputs/checkpoints/best",  # Reference model
    beta=0.1,  # KL regularization strength
    learning_rate=5e-7,
    batch_size=4,
    gradient_accumulation_steps=8,
    num_epochs=1,
    max_length=512,
    max_prompt_length=256,
    warmup_steps=150,
    logging_steps=10,
    eval_steps=500,
    save_steps=1000,
    output_dir="outputs/dpo_model"
)

# Initialize trainer
trainer = DPOTrainer(
    config=dpo_config,
    tokenizer=tokenizer,
    train_dataset=preference_dataset["train"],
    eval_dataset=preference_dataset["validation"]
)

# Train
trainer.train()
```

### Advanced DPO with IPO Loss

```python
# Use IPO (Identity Preference Optimization) loss
dpo_config = DPOConfig(
    model_name_or_path="outputs/checkpoints/best",
    loss_type="ipo",  # Use IPO instead of standard DPO
    ipo_tau=0.05,  # IPO temperature
    beta=0.1,
    learning_rate=5e-7,
    label_smoothing=0.1,  # Add label smoothing
    gradient_checkpointing=True,
    mixed_precision="bf16"
)
```

### Reference-Free DPO

```python
# Train without reference model (more memory efficient)
dpo_config = DPOConfig(
    model_name_or_path="outputs/checkpoints/best",
    reference_free=True,  # No reference model needed
    beta=0.1,
    learning_rate=1e-6,  # Can use higher LR without reference
)
```

## RLHF with PPO

### Using the Executable Script

```bash
# PPO with sentiment-based rewards
python ppo_training.py \
  --model-path ../../outputs/checkpoints/best \
  --reward-model-path sentiment \
  --prompts ../../data/prompts.txt \
  --output-dir ../../outputs/ppo_model

# PPO with custom reward model
python ppo_training.py \
  --model-path ../../outputs/checkpoints/best \
  --reward-model-path path/to/reward_model \
  --prompts your_prompts_dataset \
  --max-steps 10000
```

### Script Arguments

```bash
python ppo_training.py --help

# Key arguments:
#   --model-path PATH        Path to model checkpoint
#   --reward-model-path PATH Path to reward model or "sentiment"
#   --prompts NAME/PATH      Path to prompts file or dataset name
#   --learning-rate FLOAT    Learning rate (default: 1.4e-5)
#   --batch-size INT         Total batch size (default: 128)
#   --ppo-epochs INT         Number of PPO epochs per batch (default: 4)
#   --target-kl FLOAT        Target KL divergence (default: 6.0)
```

### Complete PPO Pipeline

```python
from src.training.ppo_trainer import PPOTrainer, PPOConfig
from src.training.reward_modeling import RewardModel
from transformers import AutoModelForSequenceClassification

# Load models
model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")
ref_model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")
ref_model.eval()  # Reference model in eval mode

# Load or train reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "your_reward_model"
)

# Configure PPO
ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=128,
    mini_batch_size=4,
    gradient_accumulation_steps=32,
    ppo_epochs=4,
    gamma=1.0,
    lam=0.95,
    clip_ratio=0.2,
    value_clip=0.2,
    entropy_coef=0.01,
    value_loss_coef=0.1,
    init_kl_coef=0.2,
    target_kl=6.0,
    max_length=512,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)

# Initialize trainer
trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    config=ppo_config
)

# Prepare prompts
prompts = load_dataset("your_prompt_dataset")["train"]["prompt"]

# Train
trainer.train(prompts)
```

### PPO with Custom Reward

```python
# Define custom reward function
def custom_reward_fn(responses, prompts):
    rewards = []
    for response, prompt in zip(responses, prompts):
        # Custom reward logic
        reward = 0.0
        if "helpful" in response.lower():
            reward += 1.0
        if len(response.split()) > 50:  # Encourage longer responses
            reward += 0.5
        rewards.append(reward)
    return torch.tensor(rewards)

# Use in PPO training
trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_fn=custom_reward_fn,  # Use custom reward instead of reward model
    config=ppo_config
)
```

## Constitutional AI

### Using the Executable Script

```bash
# Constitutional AI with default principles
python constitutional_ai.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset helpful-base \
  --use-default-principles \
  --output-dir ../../outputs/constitutional_model

# With custom principles
python constitutional_ai.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset your_dataset \
  --principles "Be helpful" "Be harmless" "Be honest"

# Generate critiques only (no training)
python constitutional_ai.py \
  --model-path ../../outputs/checkpoints/best \
  --dataset your_dataset \
  --use-default-principles \
  --generate-only \
  --save-critiques
```

### Script Arguments

```bash
python constitutional_ai.py --help

# Key arguments:
#   --model-path PATH        Path to model checkpoint
#   --dataset NAME/PATH      Training dataset name or path
#   --principles-file PATH   Path to file containing principles
#   --principles PRIN1 PRIN2 List of constitutional principles
#   --use-default-principles Use default constitutional principles
#   --num-iterations INT     Number of critique-revision iterations (default: 3)
#   --generate-only         Only generate critiques without training
#   --save-critiques        Save generated critiques and revisions
```

### Basic Constitutional Training

```python
from src.training.constitutional_ai import ConstitutionalAITrainer, ConstitutionalConfig

# Define constitutional principles
principles = [
    "The assistant should be helpful and provide accurate information",
    "The assistant should avoid harmful or dangerous content",
    "The assistant should be honest and acknowledge uncertainty",
    "The assistant should respect user privacy and confidentiality"
]

# Configure constitutional training
const_config = ConstitutionalConfig(
    principles=principles,
    critique_model="outputs/checkpoints/best",  # Can use same model
    revision_model="outputs/checkpoints/best",
    num_iterations=3,
    batch_size=8,
    learning_rate=1e-5,
    temperature=0.7
)

# Train
trainer = ConstitutionalAITrainer(
    model=model,
    tokenizer=tokenizer,
    config=const_config
)

trainer.train_with_constitution(
    train_dataset=dataset,
    num_epochs=2
)
```

### Multi-Stage Constitutional AI

```python
# Stage 1: Critique generation
critique_dataset = trainer.generate_critiques(
    dataset,
    principles=principles[:2]  # Focus on helpfulness first
)

# Stage 2: Revision training
trainer.train_on_revisions(
    critique_dataset,
    num_epochs=1
)

# Stage 3: Final constitutional training
trainer.train_with_constitution(
    dataset,
    principles=principles,  # All principles
    num_epochs=2
)
```

## Parameter-Efficient Fine-tuning (PEFT)

### LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model, TaskType
from src.model.moe_transformer import MoEForCausalLM

# Load base model
base_model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling parameter
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "v_proj",  # Attention modules
        "w1", "w2", "w3",    # MoE expert modules
        "gate"               # Router modules
    ],
    modules_to_save=["embed_tokens", "lm_head"]  # Full fine-tuning for these
)

# Create PEFT model
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()  # Shows % of trainable params

# Train with standard trainer
training_config = TrainingConfig(
    learning_rate=3e-4,  # Can use higher LR with LoRA
    num_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4
)

trainer = MoETrainer(
    model=model,
    config=training_config,
    train_dataset=dataset["train"]
)
trainer.train()
```

### QLoRA (Quantized LoRA)

```python
import torch
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = MoEForCausalLM.from_pretrained(
    "outputs/checkpoints/best",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA to quantized model
model = get_peft_model(model, peft_config)
```

### Expert-Specific LoRA

```python
# Apply LoRA only to specific experts
expert_specific_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "layers.*.mlp.experts.0.w1",  # Only expert 0
        "layers.*.mlp.experts.0.w2",
        "layers.*.mlp.experts.0.w3",
    ],
    lora_dropout=0.05
)
```

## Memory Optimization

### Why Training Uses So Much RAM

1. **Model Parameters**: The MoE model has multiple experts, each requiring memory
2. **Optimizer States**: Adam optimizer stores 2 states per parameter (doubles memory)
3. **Gradients**: Each parameter needs gradient storage
4. **Activations**: Forward pass stores activations for backward pass
5. **Batch Size**: Larger batches = more memory for data and intermediate computations

### Memory Usage Breakdown (Approximate)

For a small MoE model with 768 hidden size:
- Model parameters: ~1-2 GB
- Optimizer states (Adam): ~2-4 GB
- Gradients: ~1-2 GB
- Activations (batch_size=16, seq_len=1024): ~4-8 GB
- Data loading buffers: ~1-2 GB

**Total: ~10-20 GB RAM**

### Solutions to Reduce Memory Usage

#### 1. Enable Gradient Checkpointing
```bash
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --gradient-checkpointing
```
**Saves: ~33% activation memory**

#### 2. Use LoRA (Parameter-Efficient Fine-tuning)
```bash
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --use-lora
```
**Saves: ~90% of trainable parameters**

#### 3. Reduce Batch Size
```bash
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --batch-size 2 \
    --gradient-accumulation-steps 16
```
**Saves: ~75% activation memory**

#### 4. Reduce Sequence Length
```bash
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --max-length 256
```
**Saves: ~75% memory (quadratic with length)**

#### 5. Use Memory-Optimized Config
```bash
# Copy the optimized config to checkpoint directory
cp memory_optimized_config.yaml ../../outputs/checkpoints/best/config.yaml

# Run training
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --gradient-checkpointing \
    --use-lora
```

#### 6. Monitor Memory Usage
```python
# Add to your training script
import psutil
import torch

def print_memory_usage():
    # RAM usage
    ram_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    print(f"RAM Usage: {ram_usage:.2f} GB")
    
    # GPU memory (if using CUDA)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        print(f"GPU Memory: {gpu_memory:.2f} GB")
```

#### 7. Best Practices for Limited Memory

1. **Start Small**: Begin with tiny batches and short sequences
2. **Use Mixed Precision** (on GPU): Saves ~50% memory
3. **Clear Cache**: Call `torch.cuda.empty_cache()` periodically
4. **Stream Data**: Use `streaming=True` in dataset config
5. **Offload to Disk**: Use `map_location='cpu'` when loading checkpoints

#### 8. Emergency Memory Fixes

If still running out of memory:
```bash
# Ultra-low memory configuration
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 128 \
    --gradient-checkpointing \
    --use-lora \
    --freeze-base
```

This uses:
- Batch size 1 (minimum possible)
- Very short sequences (128 tokens)
- LoRA (only ~1% parameters trainable)
- Frozen base model (only trains LoRA adapters)
- Gradient checkpointing (recomputes activations)

## Complete Fine-tuning Pipeline

### End-to-End Example

```python
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from src.model.moe_transformer import MoEForCausalLM
from src.training.trainer import MoETrainer, TrainingConfig
from src.training.dpo_trainer import DPOTrainer, DPOConfig

# Step 1: Initial supervised fine-tuning
print("Step 1: Supervised Fine-tuning")
model = MoEForCausalLM.from_pretrained("outputs/checkpoints/best")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

sft_dataset = load_dataset("your_sft_dataset")
sft_config = TrainingConfig(
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=8,
    output_dir="outputs/sft_model"
)

sft_trainer = MoETrainer(
    model=model,
    config=sft_config,
    train_dataset=sft_dataset["train"],
    eval_dataset=sft_dataset["validation"]
)
sft_trainer.train()

# Step 2: DPO alignment
print("Step 2: DPO Alignment")
preference_dataset = load_dataset("your_preference_dataset")
dpo_config = DPOConfig(
    model_name_or_path="outputs/sft_model/checkpoint-best",
    ref_model_name_or_path="outputs/sft_model/checkpoint-best",
    beta=0.1,
    learning_rate=5e-7,
    num_epochs=1,
    output_dir="outputs/dpo_model"
)

dpo_trainer = DPOTrainer(
    config=dpo_config,
    tokenizer=tokenizer,
    train_dataset=preference_dataset["train"]
)
dpo_trainer.train()

# Step 3: Constitutional AI refinement
print("Step 3: Constitutional AI")
from src.training.constitutional_ai import ConstitutionalAITrainer

model = MoEForCausalLM.from_pretrained("outputs/dpo_model/checkpoint-best")
const_trainer = ConstitutionalAITrainer(model, tokenizer)
const_trainer.train_with_constitution(
    principles=["helpful", "harmless", "honest"],
    num_epochs=1
)

print("Fine-tuning pipeline complete!")
```

### Evaluation and Testing

```python
from src.generation.text_generator import TextGenerator

# Load fine-tuned model
model = MoEForCausalLM.from_pretrained("outputs/final_model/checkpoint-best")
generator = TextGenerator(model, tokenizer)

# Test generation
test_prompts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to sort a list",
    "What are the benefits of exercise?"
]

for prompt in test_prompts:
    response = generator.generate(
        prompt,
        max_length=256,
        temperature=0.7,
        top_p=0.9
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

## Best Practices

1. **Start with SFT**: Begin with supervised fine-tuning on high-quality data
2. **Use PEFT for large models**: LoRA/QLoRA significantly reduces memory requirements
3. **Monitor for overfitting**: Use validation sets and early stopping
4. **Adjust learning rates**: Use lower LRs for fine-tuning (1e-5 to 5e-5)
5. **Gradient accumulation**: Increase effective batch size on limited hardware
6. **Mixed precision**: Use bf16 for stable training with memory savings

## Troubleshooting

### Out of Memory
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`
- Reduce batch size and increase gradient accumulation
- Use QLoRA for 4-bit fine-tuning
- Enable CPU offloading in training config

### Unstable Training
- Reduce learning rate
- Increase warmup steps
- Use gradient clipping: `max_grad_norm=0.5`
- Check for data quality issues

### Poor Performance
- Ensure sufficient training data (>10k examples for SFT)
- Verify data preprocessing and tokenization
- Monitor training metrics closely
- Consider multi-stage training approach

## WandB Monitoring

The optimized fine-tuning script includes comprehensive WandB logging:

### Metrics Tracked
- **Training**: loss, learning rate, gradient norm
- **Performance**: samples/second, tokens/second
- **Memory**: GPU allocation and usage
- **LoRA**: trainable parameters percentage
- **Evaluation**: validation loss and perplexity
- **Generation**: quality metrics and throughput

### Usage
```bash
python supervised_finetuning_optimized.py \
  --dataset your_dataset \
  --wandb \
  --wandb-project "your-project-name"
```

View your training progress at: https://wandb.ai/your-username/your-project-name

For more details, see:
- [Training Guide](../../docs/training.md)
- [RLHF Pipeline](../../docs/rlhf.md)
- [API Reference](../../docs/api_reference.md)