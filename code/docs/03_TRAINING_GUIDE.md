# Training Guide

Comprehensive guide to training models with Ava, from basic runs to advanced configurations.

## Training Overview

The training pipeline follows this flow:

```
Config Loading → Model Building → Data Loading → Training Loop → Checkpointing
                                                      ↓
                              [Forward → Loss → Backward → Optimizer Step]
```

## Basic Training

### Single GPU Training

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large_Multy.yaml
```

### Resuming Training

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --resume code/outputs/pretraining/run_xyz/checkpoints/checkpoint_step_5000.pt
```

## Configuration Structure

Training is controlled by YAML configuration files:

```yaml
# code/configs/moe/large.yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_experts: 8
  num_experts_per_token: 2

training:
  batch_size: 128
  learning_rate: 0.0006
  warmup_steps: 1000
  num_epochs: 5
  gradient_accumulation_steps: 4
  max_gradient_norm: 1.0

data:
  data_dir: "code/data/processed"
  max_length: 512
  num_workers: 8

output:
  output_dir: "code/outputs/pretraining"
  save_every: 500
```

## Training Pipeline Components

The `TrainingPipeline` class orchestrates training through registered components:

```python
# code/src/ava/training/pipeline.py
pipeline = TrainingPipeline(context)
pipeline.register('model', ModelBuilder(context))
pipeline.register('optimizer', OptimizerManager(context))
pipeline.register('data', DataLoaderManager(context))
pipeline.register('training', TrainingLoopManager(context))
pipeline.register('validation', ValidationManager(context))
pipeline.register('metrics', MetricsManager(context))

pipeline.initialize_all()
# ... training loop ...
pipeline.cleanup_all()
```

### Component Lifecycle

Each component implements lifecycle hooks:

| Hook | When Called | Severity |
|------|-------------|----------|
| `initialize()` | Before training | FATAL |
| `cleanup()` | After training | WARNING |
| `on_epoch_start(epoch)` | Start of epoch | WARNING |
| `on_epoch_end(epoch)` | End of epoch | WARNING |
| `on_step_start(step)` | Start of step | WARNING |
| `on_step_end(step, loss)` | End of step | WARNING |
| `on_error(error)` | On training error | WARNING |

## Learning Rate Scheduling

### Warmup + Cosine Decay

```yaml
training:
  learning_rate: 0.0006
  warmup_steps: 1000
  max_steps: 100000

  adaptive_lr:
    scheduler: 'cosine'
    min_lr_ratio: 0.1
```

### Learning Rate Finder

Find optimal learning rate before training:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --run-lr-finder \
    --lr-finder-use-suggested
```

Configuration:
```yaml
lr_finder:
  run_lr_finder: true
  start_lr: 1e-8
  end_lr: 1.0
  num_iterations: 100
  suggestion_method: 'steepest'
```

## Gradient Management

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  batch_size: 32                    # Micro batch
  gradient_accumulation_steps: 4    # Effective: 32 × 4 = 128
```

### Gradient Clipping

Prevent exploding gradients:

```yaml
training:
  max_gradient_norm: 1.0
```

### Gradient Checkpointing

Trade compute for memory:

```yaml
model:
  gradient_checkpointing: true
```

## Mixed Precision Training

### BF16 (Recommended for Ampere+)

```yaml
hardware:
  mixed_precision: 'bf16'
```

### FP16 (Older GPUs)

```yaml
hardware:
  mixed_precision: 'fp16'
```

### FP8 (Hopper/Ada GPUs)

```yaml
fp8:
  enabled: true
  format: 'e4m3'
```

## Checkpointing

### Automatic Checkpointing

```yaml
output:
  save_every: 500           # Save every N steps
  output_dir: "code/outputs/pretraining"
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': current_epoch,
    'step': global_step,
    'best_loss': best_loss,
    'config': config,
}
```

### Async Checkpointing

Save checkpoints without blocking training:

```yaml
optimizations:
  checkpoint:
    async_saving: true
```

## Validation During Training

### Periodic Validation

```yaml
evaluation:
  eval_during_training: true
  eval_frequency: 500
  eval_metrics: 'loss,perplexity,coherence'
```

### Coherence Evaluation

Generate samples and measure quality:

```yaml
coherence:
  enabled: true
  eval_every_n_steps: 500
  num_samples: 10
  max_generation_length: 256
```

## Monitoring & Logging

### Weights & Biases

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --use-wandb \
    --wandb-project "Ava" \
    --wandb-tags "moe,training"
```

Configuration:
```yaml
wandb:
  use_wandb: true
  wandb_project: 'Ava'
  wandb_log_freq: 10
```

### Console Logging

```yaml
logging:
  verbosity: 'info'
  metrics_log_freq: 100
  health_summary_freq: 500
```

## Performance Modes

### Ultra Fast Mode

Disable all logging for maximum speed:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --ultra-fast-mode
```

### Express Mode

Optimized async logging:

```bash
python code/scripts/5_training/train_pipeline.py \
    --config code/configs/moe/large.yaml \
    --express-mode
```

## Fine-Tuning

### From Pretrained Checkpoint

```bash
python code/scripts/5_training/finetune.py \
    --checkpoint code/outputs/pretraining/run_xyz/checkpoints/best_model.pt \
    --data-dir code/data/fine-tuning
```

### Auto-Discovery

```bash
# Automatically finds latest checkpoint and data
python code/scripts/5_training/finetune.py
```

### LoRA Fine-Tuning

Parameter-efficient fine-tuning:

```yaml
model:
  use_lora_experts: true
  lora_rank: 8
  lora_alpha: 16
  freeze_lora_base: true
```

## Training Strategies

### Progressive Training

Start with shorter sequences, increase over training:

```yaml
training:
  progressive:
    enable_progressive_training: true
    enable_sequence_scaling: true
    initial_seq_length: 128
    final_seq_length: 2048
    length_schedule: 'linear'
```

### Curriculum Learning

Train on easier examples first:

```yaml
training:
  progressive:
    enable_curriculum: true
    curriculum_metric: 'loss'
```

## Common Training Patterns

### Memory-Efficient Training

```yaml
model:
  gradient_checkpointing: true
  use_flash_attention: true

moe_memory_optimization:
  use_expert_offloading: true
  max_active_experts_gpu: 4

hardware:
  mixed_precision: 'bf16'

training:
  gradient_accumulation_steps: 8
```

### Fast Iteration

```yaml
model:
  gradient_checkpointing: false
  use_triton_kernels: true

data:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true

performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
```

### Maximum Quality

```yaml
training:
  learning_rate: 0.0003       # Lower LR
  warmup_steps: 2000          # Longer warmup
  max_gradient_norm: 0.5      # Stricter clipping
  gradient_accumulation_steps: 8

coherence:
  enabled: true
  eval_every_n_steps: 250     # Frequent evaluation

model_selection:
  enabled: true
  val_loss_weight: 0.5
  coherence_score_weight: 0.3
```

## Troubleshooting Training

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Enable expert offloading
4. Use gradient accumulation

### Loss Not Decreasing

1. Check learning rate (try LR finder)
2. Verify data is loaded correctly
3. Check for NaN gradients
4. Increase warmup steps

### Training Too Slow

1. Enable Flash Attention
2. Increase num_workers
3. Enable persistent workers
4. Use TF32 on Ampere GPUs

See [Troubleshooting Guide](./09_TROUBLESHOOTING.md) for more solutions.

## Next Steps

- [Configuration Reference](./04_CONFIGURATION.md) - All parameters
- [Data Pipeline](./05_DATA_PIPELINE.md) - Data loading details
- [Distributed Training](./06_DISTRIBUTED.md) - Multi-GPU setup
- [Performance Tuning](./10_PERFORMANCE.md) - Optimization guide
