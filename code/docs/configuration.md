# Configuration Guide

Complete guide to configuring Ava MoE++ models through YAML configuration files.

## Configuration Structure

Configuration files are organized in three main sections:

```yaml
model:      # Model architecture parameters
  ...

training:   # Training hyperparameters
  ...

data:       # Data loading configuration
  ...
```

## Model Configuration

### Basic Model Parameters

```yaml
model:
  # Core Architecture
  hidden_size: 768              # Model dimension (embed_dim)
  num_layers: 12                # Number of transformer blocks
  num_attention_heads: 12       # Attention heads per layer
  intermediate_size: 3072       # FFN intermediate dimension
  vocab_size: 50257            # Vocabulary size (GPT-2 default)
  max_position_embeddings: 1024 # Maximum sequence length
```

### MoE (Mixture of Experts) Configuration

```yaml
model:
  # Expert System
  num_experts: 8                # Total number of experts
  num_experts_per_tok: 2        # Active experts per token (top-k)
  expert_capacity_factor: 1.25  # Capacity for load balancing
  expert_routing_type: switch   # Routing algorithm: switch, gshard, base

  # Expert Features
  use_hierarchical_moe: true    # Hierarchical expert organization
  use_continuous_experts: true  # Continuous (soft) routing
  use_sparse_experts: false     # Sparse expert computation
  use_expert_cache: true        # Cache expert selections
```

### Attention Configuration

```yaml
model:
  # Attention Mechanism
  attention_type: gqa           # mha (multi-head), mqa, gqa (grouped-query)
  num_key_value_heads: 2        # For GQA - reduces memory
  use_flash_attn: false        # Flash Attention (GPU only)
  attention_dropout: 0.1        # Dropout in attention
  use_rotary_embeddings: true  # RoPE position encoding
  rope_theta: 10000.0          # RoPE base frequency
```

### Regularization

```yaml
model:
  # Dropout
  hidden_dropout: 0.1           # General dropout rate
  attention_dropout: 0.1        # Attention-specific dropout
  expert_dropout: 0.1          # Expert layer dropout

  # Other Regularization
  label_smoothing: 0.1         # Label smoothing factor
  gradient_checkpointing: true # Trade compute for memory
```

## Training Configuration

### Basic Training Parameters

```yaml
training:
  # Batch Settings
  batch_size: 4                       # Per-device batch size
  gradient_accumulation_steps: 8      # Effective batch = 4 * 8 = 32

  # Training Duration
  num_epochs: 10                      # Number of epochs
  max_steps: -1                       # Max steps (-1 = use epochs)

  # Evaluation
  eval_steps: 500                     # Evaluate every N steps
  save_steps: 1000                   # Save checkpoint every N steps
  logging_steps: 50                  # Log metrics every N steps
```

### Optimizer Configuration

```yaml
training:
  # Optimizer
  optimizer: adamw                    # Optimizer type
  learning_rate: 5e-4                # Base learning rate
  weight_decay: 0.01                 # L2 regularization
  adam_beta1: 0.9                    # Adam beta1
  adam_beta2: 0.95                   # Adam beta2
  adam_epsilon: 1e-8                 # Adam epsilon

  # Gradient Management
  max_grad_norm: 1.0                 # Gradient clipping
  gradient_centralization: false     # Gradient centralization
  adaptive_gradient_clipping: true   # Dynamic clipping
```

### Learning Rate Schedule

```yaml
training:
  # LR Schedule
  lr_scheduler_type: cosine          # linear, cosine, cosine_with_restarts
  warmup_steps: 500                  # Warmup steps
  warmup_ratio: 0.1                  # Or use ratio of total steps
  num_restarts: 2                    # For cosine with restarts
  lr_decay_factor: 0.95              # Decay after each restart
```

### Advanced Training Features

```yaml
training:
  # Curriculum Learning
  curriculum_learning: true           # Enable curriculum
  initial_sequence_length: 256       # Start with shorter sequences
  curriculum_warmup_steps: 1000      # Steps to reach full length
  progressive_sequence_growth: true   # Continue growing
  sequence_growth_rate: 1.5          # Growth multiplier
  sequence_growth_interval: 5000     # Steps between growth

  # Anti-Overfitting
  stochastic_depth_prob: 0.1         # Layer dropout probability
  mixout_prob: 0.1                   # Mix with initialization
  use_ema: true                      # Exponential moving average
  ema_decay: 0.999                   # EMA decay rate
```

### Loss Functions

```yaml
training:
  # Focal Loss
  use_focal_loss: true               # Enable focal loss
  focal_gamma: 2.0                   # Focus on hard examples
  focal_alpha: 0.25                  # Class balance
  focal_blend: 0.3                   # Blend with standard loss

  # Auxiliary Losses
  entropy_regularization_weight: 0.01     # Output diversity
  token_frequency_penalty_weight: 0.1     # Reduce repetition
  expert_diversity_weight: 0.01           # Expert specialization
```

## Data Configuration

```yaml
data:
  # Dataset Paths
  train_paths:
    - ./data/pretraining/processed/train_*.parquet
  val_path: ./data/pretraining/processed/val_*.parquet

  # Tokenization
  tokenizer: gpt2                    # Tokenizer name
  max_length: 1024                  # Maximum sequence length
  stride: 512                        # For overlapping sequences

  # Data Loading
  max_train_examples: 100000        # Limit training samples
  max_val_examples: 5000            # Limit validation samples
  num_workers: 4                     # DataLoader workers
  prefetch_factor: 2                 # Prefetch batches
```

## Hardware-Specific Configurations

### CPU Configuration

```yaml
# configs/cpu/small.yaml
model:
  hidden_size: 512
  num_layers: 8
  num_experts: 4                    # Fewer experts for CPU
  use_flash_attn: false             # Not supported on CPU

training:
  batch_size: 4                     # Small batch for memory
  gradient_accumulation_steps: 8
  mixed_precision: false            # No mixed precision on CPU

cpu_optimization:
  use_mkl: true                     # Intel MKL optimization
  num_threads: 8                    # CPU threads
  enable_jemalloc: true            # Memory management
```

### GPU Configuration

```yaml
# configs/gpu/base.yaml
model:
  hidden_size: 1024
  num_layers: 16
  num_experts: 16
  use_flash_attn: true              # Enable Flash Attention

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  mixed_precision: true             # FP16/BF16 training

inference:
  use_flash_attn: true
  torch_compile: true               # PyTorch 2.0 compilation
  dtype: float16                    # or bfloat16
```

## Configuration Presets

### Development/Testing

```yaml
# configs/dev.yaml
model:
  hidden_size: 128
  num_layers: 2
  num_experts: 2

training:
  batch_size: 2
  num_epochs: 1
  eval_steps: 10
  save_steps: 50
```

### Small Model (50M)

```yaml
# configs/cpu/small.yaml
model:
  hidden_size: 512
  num_layers: 8
  num_attention_heads: 8
  num_experts: 4
  ffn_hidden_size: 2048
```

### Medium Model (200M)

```yaml
# configs/cpu/medium.yaml
model:
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12
  num_experts: 8
  ffn_hidden_size: 3072
```

### Large Model (1.5B)

```yaml
# configs/gpu/large.yaml
model:
  hidden_size: 1536
  num_layers: 24
  num_attention_heads: 24
  num_experts: 16
  ffn_hidden_size: 6144
```

## Creating Custom Configurations

### Step 1: Copy a Base Configuration

```bash
cp configs/cpu/small.yaml configs/custom.yaml
```

### Step 2: Modify Parameters

```yaml
# configs/custom.yaml
model:
  hidden_size: 640          # Custom dimension
  num_layers: 10           # Custom depth
  num_experts: 6           # Custom expert count

training:
  batch_size: 8
  learning_rate: 3e-4
  num_epochs: 20
```

### Step 3: Use Custom Configuration

```bash
python scripts/training/train.py --config configs/custom.yaml
```

## Configuration Override

Command-line arguments override configuration file values:

```bash
python scripts/training/train.py \
    --config configs/cpu/small.yaml \
    --batch-size 8 \              # Overrides config
    --learning-rate 1e-4 \        # Overrides config
    --epochs 20                   # Overrides config
```

## Advanced Configuration Examples

### Memory-Optimized Configuration

```yaml
model:
  gradient_checkpointing: true
  use_cache: false                # Disable during training
  expert_capacity_factor: 1.0    # Minimal capacity

training:
  batch_size: 1
  gradient_accumulation_steps: 32
  mixed_precision: true           # If GPU available
```

### Speed-Optimized Configuration

```yaml
model:
  num_experts: 4                  # Fewer experts
  num_experts_per_tok: 1         # Single expert routing
  use_cache: true

training:
  batch_size: 32
  gradient_accumulation_steps: 1
  num_workers: 8
```

### Quality-Optimized Configuration

```yaml
model:
  num_experts: 32                 # Many experts
  num_experts_per_tok: 8         # Use multiple experts
  sinkhorn_iterations: 10        # Better load balancing

training:
  learning_rate: 1e-4
  warmup_ratio: 0.2
  num_epochs: 50
  weight_decay: 0.05
```

## Validation

Validate configuration before training:

```python
# validate_config.py
import yaml
from src.Ava.models import EnhancedMoEConfig
from dataclasses import fields

# Load configuration
with open('configs/custom.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Check model configuration
model_config = config_dict.get('model', {})
valid_fields = {f.name for f in fields(EnhancedMoEConfig)}

invalid_fields = set(model_config.keys()) - valid_fields
if invalid_fields:
    print(f"Warning: Invalid fields: {invalid_fields}")

# Create configuration
try:
    config = EnhancedMoEConfig(**{k: v for k, v in model_config.items() if k in valid_fields})
    print("✓ Configuration is valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

## Best Practices

1. **Start Simple**: Begin with small configurations and scale up
2. **Version Control**: Track configuration files in git
3. **Document Changes**: Comment important modifications
4. **Test First**: Validate configurations with short training runs
5. **Profile Performance**: Monitor resource usage with different configs

## Common Issues

### Issue: Out of Memory
```yaml
# Reduce memory usage
model:
  hidden_size: 256               # Smaller model
  gradient_checkpointing: true   # Trade compute for memory

training:
  batch_size: 1                  # Minimal batch
  gradient_accumulation_steps: 16
```

### Issue: Slow Training
```yaml
# Increase speed
model:
  num_experts: 2                 # Fewer experts
  num_layers: 6                  # Shallower model

training:
  batch_size: 16                 # Larger batch
  mixed_precision: true          # If supported
```

### Issue: Poor Quality
```yaml
# Improve quality
model:
  hidden_size: 1024              # Larger model
  num_experts: 16               # More experts

training:
  learning_rate: 1e-4           # Lower LR
  num_epochs: 50                # More training
  warmup_ratio: 0.1             # Proper warmup
```

## Configuration Templates

Find example configurations in:
- `/project/code/configs/cpu/` - CPU configurations
- `/project/code/configs/gpu/` - GPU configurations
- `/project/code/configs/auto/` - Auto-scaled configurations

Each configuration is documented with hardware requirements and use cases.