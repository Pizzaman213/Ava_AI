# Training Guide

This comprehensive guide covers training MoE++ models with advanced attention mechanisms and optimizations.

## Table of Contents
- [Quick Start](#quick-start)
- [Advanced Attention Training](#advanced-attention-training)
- [Data Preparation](#data-preparation)
- [Training Configuration](#training-configuration)
- [Fine-Tuning](#fine-tuning)
- [RLHF Training](#rlhf-training)
- [Distributed Training](#distributed-training)
- [Memory Optimization](#memory-optimization)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)

## Quick Start

### Using the Training Launcher (Recommended)

The easiest way to train models with different attention mechanisms:

```bash
# Standard training
python scripts/train_launcher.py standard --experiment-name my_model

# Long document processing (sparse attention)
python scripts/train_launcher.py long-context \
    --data-path /path/to/documents \
    --experiment-name long_doc_model

# Chat model (streaming attention)  
python scripts/train_launcher.py chat \
    --data-path /path/to/conversations \
    --experiment-name chat_assistant

# Extreme length (linear attention)
python scripts/train_launcher.py extreme-length \
    --data-path /path/to/long_sequences \
    --batch-size 1

# Fast inference optimized
python scripts/train_launcher.py fast-inference \
    --model-size small \
    --experiment-name speed_optimized
```

### Direct Training Script

```bash
# Single GPU/MPS training
python scripts/training/train.py --config configs/mps/small.yaml

# GPU training with custom attention
python scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --sparse-global-tokens 128 \
    --output-dir outputs/my_model \
    --num-epochs 5

# Resume from checkpoint
python scripts/training/train.py \
    --config configs/mps/small.yaml \
    --checkpoint outputs/checkpoints/step_1000
```

### Training Script Example

```python
from src.model.moe_transformer import MoEForCausalLM
from src.training.trainer import MoETrainer, TrainingConfig
from src.utils.config import load_config

# Load configuration
config = load_config("configs/mps/small.yaml")
model = MoEForCausalLM(config['model'])

# Configure training
training_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=3,
    warmup_steps=1000,
    gradient_accumulation_steps=4,
    mixed_precision="no",  # or "fp16", "bf16"
    output_dir="outputs/my_model"
)

# Create trainer
trainer = MoETrainer(
    model=model,
    config=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()
```

## Data Preparation

### Dataset Format

The training pipeline expects data in the following formats:

1. **JSON Lines Format**
```json
{"text": "This is a training example."}
{"text": "Another training example with longer content..."}
```

2. **Conversation Format** (for instruction tuning)
```json
{
  "conversations": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

3. **Preference Format** (for RLHF)
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum mechanics...",
  "rejected": "Quantum computing is too complex to explain..."
}
```

### Data Processing Pipeline

```bash
# Step 1: Download raw data
python scripts/download_data.py --dataset c4 --split train

# Step 2: Filter and score data
python scripts/filter_data.py \
    --input data/raw/c4 \
    --output data/filtered \
    --min_score 0.7 \
    --use_bert_scorer

# Step 3: Tokenize and create memory-mapped files
python scripts/prepare_data.py \
    --input data/filtered \
    --output data/prepared \
    --tokenizer configs/tokenizer \
    --max_length 2048 \
    --create_mmap
```

### Data Quality Control

```python
# Custom data filtering
from moe_llm.data import DataFilter, BertScorer

# Initialize quality scorer
scorer = BertScorer(model_name="bert-base-uncased")

# Create filter
filter = DataFilter(
    min_length=50,
    max_length=10000,
    min_quality_score=0.7,
    remove_duplicates=True,
    scorer=scorer
)

# Apply filtering
filtered_data = filter.filter_dataset(raw_data)
```

## Training Configuration

### Configuration Structure

```yaml
# configs/training.yaml
model:
  type: moe_transformer
  hidden_size: 2048
  num_layers: 24
  num_attention_heads: 16
  num_experts: 64
  experts_per_token: 4
  
training:
  batch_size: 512
  micro_batch_size: 8
  gradient_accumulation_steps: 64
  learning_rate: 1e-4
  min_learning_rate: 1e-5
  warmup_steps: 10000
  max_steps: 1000000
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-8
  grad_clip_norm: 1.0
  
optimization:
  use_mixed_precision: true
  precision: bf16
  gradient_checkpointing: true
  zero_optimization:
    stage: 3
    offload_optimizer: true
    offload_param: true
    overlap_comm: true
    reduce_scatter: true
  
data:
  train_data_path: data/prepared/train
  eval_data_path: data/prepared/eval
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
  
logging:
  log_interval: 10
  eval_interval: 1000
  save_interval: 5000
  tensorboard: true
  wandb:
    project: moe-llm
    entity: your-entity
```

### Learning Rate Scheduling

```python
# Cosine schedule with warmup
def get_scheduler(optimizer, config):
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        
        progress = (step - config.warmup_steps) / (
            config.max_steps - config.warmup_steps
        )
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * \
               0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)
```

### Expert-specific Configuration

```yaml
experts:
  num_experts: 64
  experts_per_token: 4
  expert_capacity_factor: 1.25
  load_balancing_loss_weight: 0.01
  router_z_loss_weight: 0.001
  expert_dropout: 0.1
  
  # Hierarchical routing
  use_hierarchical: true
  num_domains: 8
  experts_per_domain: 8
  
  # Expert initialization
  expert_init_scale: 0.1
  router_init_range: 0.02
```

## Fine-Tuning

### Supervised Fine-Tuning (SFT)

Fine-tune a pre-trained model on instruction datasets:

```bash
# Basic fine-tuning
python examples/fine_tuning/supervised_finetuning.py \
  --checkpoint outputs/moe_ultra_tiny_mps/best \
  --dataset tatsu-lab/alpaca \
  --output-dir outputs/sft_model \
  --num-epochs 3 \
  --batch-size 8

# With LoRA for memory efficiency
python examples/fine_tuning/supervised_finetuning.py \
  --checkpoint outputs/moe_ultra_tiny_mps/best \
  --dataset tatsu-lab/alpaca \
  --use-lora \
  --output-dir outputs/lora_model
```

### LoRA/QLoRA Configuration

```yaml
# In your config file
training:
  use_lora: true
  lora_config:
    r: 16  # Rank
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    use_qlora: false  # Set to true for 4-bit quantization
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
```

### Custom Dataset Fine-Tuning

```python
from datasets import Dataset

# Prepare custom dataset
data = [
    {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
    {"instruction": "Summarize", "input": "Long text...", "output": "Summary..."}
]

dataset = Dataset.from_list(data)

# Fine-tune
trainer.train(dataset=dataset)
```

## RLHF Training

### Quick RLHF Pipeline

```bash
# Run complete RLHF pipeline
python examples/rlhf_example.py \
  --model-path outputs/sft_model/best \
  --output-dir outputs/rlhf_model

# Test-only mode
python examples/rlhf_working_example.py \
  --model-path outputs/sft_model/best \
  --test-only
```

### RLHF Configuration

```python
from src.rlhf import UnifiedRLHFTrainer, TrainingConfig

# Configure RLHF training
config = TrainingConfig(
    model_name="rlhf_model",
    num_epochs=3,
    
    # Choose training phases
    training_phases=[
        "self_supervised_pretrain",
        "constitutional_alignment",
        "ppo_optimization",
        "reasoning_enhancement",
        "final_tuning"
    ],
    
    # PPO settings
    ppo=EnhancedPPOConfig(
        learning_rate=1e-5,
        batch_size=128,
        ppo_epochs=4,
        adaptive_kl=True
    ),
    
    # Multi-reward settings
    multi_reward=MultiRewardConfig(
        factual_weight=0.3,
        coherence_weight=0.25,
        helpfulness_weight=0.25,
        safety_weight=0.2
    )
)

# Initialize and train
trainer = UnifiedRLHFTrainer(model, tokenizer, config)
trainer.train()
```

### Constitutional AI Training

```yaml
constitutional:
  use_default_principles: true
  custom_principles:
    - "Be helpful and harmless"
    - "Provide accurate information"
    - "Respect user privacy"
  max_refinement_iterations: 3
  use_adversarial_augmentation: true
  difficulty_progression: true
```

### Monitoring RLHF Training

```bash
# Monitor training progress
tensorboard --logdir outputs/rlhf_model/logs

# Key metrics to track:
# - Average reward
# - KL divergence
# - Policy entropy
# - Individual reward components
```

## Distributed Training

### Multi-GPU Setup

```python
# torchrun launcher
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py --config configs/large.yaml
```

### DeepSpeed Configuration

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 64,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.95],
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 10000
    }
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

### Model Parallelism

```python
# Configure model and pipeline parallelism
from moe_llm.distributed import ModelParallelConfig

mp_config = ModelParallelConfig(
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    expert_parallel_size=8,
    sequence_parallel=True
)

# Initialize distributed model
model = create_distributed_model(
    config,
    mp_config,
    device_mesh=device_mesh
)
```

### Expert Parallelism

```python
# Distribute experts across GPUs
def setup_expert_parallel(model, world_size):
    num_experts = model.config.num_experts
    experts_per_gpu = num_experts // world_size
    
    # Assign experts to GPUs
    for layer in model.layers:
        layer.moe_block.set_expert_parallel(
            rank=dist.get_rank(),
            world_size=world_size,
            experts_per_rank=experts_per_gpu
        )
```

## Memory Optimization

### Gradient Checkpointing

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Custom checkpointing policy
def checkpoint_policy(module):
    # Checkpoint every other layer
    return isinstance(module, MoELayer) and module.layer_idx % 2 == 0

model.set_checkpoint_policy(checkpoint_policy)
```

### Activation Offloading

```python
# Configure activation offloading
from moe_llm.memory import ActivationOffloadConfig

offload_config = ActivationOffloadConfig(
    offload_device="cpu",
    offload_layers=[12, 13, 14, 15],  # Offload middle layers
    prefetch_ahead=2,
    pin_memory=True
)

model.enable_activation_offloading(offload_config)
```

### Memory-Efficient Data Loading

```python
# Use memory-mapped datasets
from moe_llm.data import MemoryMappedDataset

dataset = MemoryMappedDataset(
    data_path="data/prepared/train.mmap",
    index_path="data/prepared/train.idx",
    sequence_length=2048,
    seed=42
)

# Streaming dataloader
dataloader = StreamingDataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True
)
```

### ZeRO-Offload Optimization

```python
# CPU offloading configuration
zero_config = {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu",
        "nvme_path": "/nvme/offload",
        "buffer_count": 5,
        "fast_init": true
    },
    "offload_param": {
        "device": "nvme",
        "nvme_path": "/nvme/offload",
        "buffer_count": 5,
        "buffer_size": 1e9,
        "max_in_cpu": 1e10
    }
}
```

## Monitoring and Debugging

### Training Metrics

```python
# Custom metrics callback
class MetricsCallback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log to TensorBoard
        if state.global_step % args.logging_steps == 0:
            writer.add_scalar("loss/train", logs["loss"], state.global_step)
            writer.add_scalar("lr", logs["learning_rate"], state.global_step)
            writer.add_scalar("grad_norm", logs["grad_norm"], state.global_step)
            
            # Expert metrics
            writer.add_scalar("experts/load_balance", logs["load_balance_loss"], state.global_step)
            writer.add_scalar("experts/router_z_loss", logs["router_z_loss"], state.global_step)
            
            # MoD metrics
            writer.add_scalar("mod/skip_rate", logs["mod_skip_rate"], state.global_step)
```

### Gradient Analysis

```python
# Monitor gradient statistics
def analyze_gradients(model):
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "max": grad.max().item(),
                "min": grad.min().item(),
                "norm": grad.norm().item()
            }
    
    return grad_stats
```

### Expert Utilization Monitoring

```python
# Track expert usage
def monitor_expert_usage(model, dataloader):
    expert_counts = defaultdict(int)
    
    for batch in dataloader:
        with torch.no_grad():
            # Get routing decisions
            routing_probs = model.get_routing_probs(batch)
            
            # Count expert assignments
            for layer_probs in routing_probs:
                experts_selected = layer_probs.argmax(dim=-1)
                for expert_id in experts_selected.flatten():
                    expert_counts[expert_id.item()] += 1
    
    # Analyze distribution
    utilization = analyze_expert_distribution(expert_counts)
    return utilization
```

## Best Practices

### 1. Data Quality

- **Filter aggressively**: Use BERT scoring to maintain quality
- **Deduplicate**: Remove duplicate and near-duplicate content
- **Balance domains**: Ensure diverse training data
- **Monitor data**: Track token statistics and domain distribution

### 2. Training Stability

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Loss spike detection
if loss > 3 * running_avg_loss:
    logger.warning(f"Loss spike detected: {loss}")
    # Skip update or reduce learning rate
    
# NaN detection
if torch.isnan(loss):
    raise ValueError("NaN loss detected")
```

### 3. Learning Rate Tuning

```python
# Learning rate finder
def find_learning_rate(model, dataloader, init_lr=1e-8, end_lr=10):
    lrs = []
    losses = []
    
    lr = init_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for batch in dataloader:
        # Forward pass
        loss = model(**batch).loss
        
        # Record
        lrs.append(lr)
        losses.append(loss.item())
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Increase LR
        lr *= 1.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if lr > end_lr:
            break
    
    # Find optimal LR (steepest descent)
    optimal_lr = find_steepest_descent(lrs, losses)
    return optimal_lr
```

### 4. Checkpointing Strategy

```python
# Save checkpoints efficiently
def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "config": model.config
    }
    
    # Save with rotation
    checkpoint_path = f"checkpoint-{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Keep only recent checkpoints
    cleanup_old_checkpoints(keep_last=5)
```

### 5. Mixed Precision Training

```python
# Configure automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

## Common Issues

### 1. Out of Memory

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3
- Offload to CPU/NVMe
- Use mixed precision training

### 2. Expert Collapse

**Symptoms:** All tokens routed to same experts

**Solutions:**
```python
# Increase load balancing loss
config.load_balancing_loss_weight = 0.1

# Add noise during training
config.router_noise_epsilon = 1e-2

# Use auxiliary losses
config.router_z_loss_weight = 0.01
```

### 3. Training Instability

**Solutions:**
```python
# Reduce learning rate
config.learning_rate *= 0.5

# Increase warmup steps
config.warmup_steps *= 2

# Use gradient accumulation
config.gradient_accumulation_steps = 16

# Enable loss scaling
config.loss_scale = "dynamic"
```

### 4. Slow Training

**Solutions:**
- Enable Flash Attention 2
- Use compiled mode: `model = torch.compile(model)`
- Optimize data loading
- Use faster optimizers (e.g., LAMB)
- Enable sequence parallelism

### 5. Poor Convergence

**Solutions:**
- Check data quality
- Verify loss functions
- Monitor gradient flow
- Adjust hyperparameters
- Use learning rate scheduling

## Advanced Training Techniques

### 1. Curriculum Learning

```python
# Start with shorter sequences
def get_curriculum_length(step, max_length=2048, warmup_steps=10000):
    if step < warmup_steps:
        return min(512 + (max_length - 512) * step / warmup_steps, max_length)
    return max_length
```

### 2. Progressive Training

```python
# Gradually increase model capacity
def progressive_training(base_config):
    # Stage 1: Small model
    config_stage1 = base_config.copy()
    config_stage1.num_experts = 16
    train_stage(config_stage1, steps=100000)
    
    # Stage 2: Medium model
    config_stage2 = base_config.copy()
    config_stage2.num_experts = 32
    train_stage(config_stage2, steps=200000, init_from=stage1_checkpoint)
    
    # Stage 3: Full model
    train_stage(base_config, steps=500000, init_from=stage2_checkpoint)
```

### 3. Knowledge Distillation

```python
# Distill from larger model
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * temperature**2
```

For more information, see:
- [Memory Optimization Guide](memory_optimization.md)
- [RLHF Training](rlhf.md)
- [Performance Tuning](performance_tuning.md)