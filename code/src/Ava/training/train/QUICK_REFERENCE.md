# Quick Reference Card

## One-Minute Setup

```python
from Ava.training.train import SimplifiedEnhancedTrainer
from Ava.config.training_config import EnhancedTrainingConfig
import torch
import torch.nn as nn

# Create trainer
trainer = SimplifiedEnhancedTrainer(
    model=your_model,
    config=EnhancedTrainingConfig(config_file=None),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Initialize
trainer.initialize(optimizer)

# Train
for batch in data_loader:
    metrics = trainer.train_step(batch)

# Cleanup
trainer.cleanup()
```

## Manager Quick Reference

| Manager | Purpose | Key Method |
|---------|---------|-----------|
| **DistributedTrainingManager** | Multi-GPU/multi-node | `barrier()`, `broadcast()` |
| **CheckpointManager** | Save/load models | `save_checkpoint()`, `load_checkpoint()` |
| **LossComputationManager** | Loss & gradients | `compute_loss()`, `backward()` |
| **MonitoringManager** | Metrics & logging | `log_metrics()`, `log_training_step()` |

## Core Methods

```python
# Initialization
trainer.initialize(optimizer)

# Training
metrics = trainer.train_step(batch)           # Single step
epoch_metrics = trainer.train_epoch(loader)  # Full epoch

# Evaluation
eval_metrics = trainer.evaluate(loader)

# Checkpointing
trainer.save_checkpoint(save_best=True)
trainer.load_checkpoint(path)

# Status
status = trainer.get_status()

# Cleanup
trainer.cleanup()
```

## Batch Format

```python
batch = {
    "input_ids": torch.tensor(...),  # Model input
    "labels": torch.tensor(...)      # Labels for loss
    # Optional: other metadata
}
```

## Loss Function Registration

```python
# Define your loss
class MyLoss(nn.Module):
    def forward(self, outputs, targets):
        # outputs: model output
        # targets: ground truth
        return loss_value

# Register
trainer.loss_manager.register_loss_function(
    name="my_loss",
    loss_fn=MyLoss(),
    weight=1.0  # relative weight
)
```

## Common Patterns

### Basic Training Loop
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        metrics = trainer.train_step(batch)
        if step % 100 == 0:
            print(f"Loss: {metrics['loss']:.4f}")
```

### Training with Evaluation
```python
for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)

    if val_metrics['eval_loss'] < best_loss:
        trainer.save_checkpoint(save_best=True)
```

### Multiple Loss Functions
```python
trainer.loss_manager.register_loss_function("ce", ce_loss, weight=1.0)
trainer.loss_manager.register_loss_function("kld", kld_loss, weight=0.1)

# Training automatically uses both losses
metrics = trainer.train_step(batch)
print(metrics)  # Shows both losses
```

### Custom Manager
```python
from Ava.training.train.base import ManagerInterface

class MyManager(ManagerInterface):
    def initialize(self):
        super().initialize()
        # your setup

    def cleanup(self):
        # your cleanup
        pass

    def on_step_end(self, step, loss):
        # called after each step
        pass
```

## Status Output

```python
status = trainer.get_status()

# Available fields:
status['training_context']  # epoch, step, loss
status['distributed']       # rank, world_size, is_main_rank
status['checkpoint']        # best_loss, checkpoint_count
status['loss']             # loss functions, metrics
status['monitoring']       # metrics, log_frequency
```

## Debugging

```python
# Check individual managers
print(trainer.distributed_manager.get_status())
print(trainer.checkpoint_manager.get_status())
print(trainer.loss_manager.get_status())
print(trainer.monitoring_manager.get_status())

# Check context
print(f"Epoch: {trainer.context.epoch}")
print(f"Step: {trainer.context.step}")
print(f"Loss: {trainer.context.current_loss}")
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "No loss functions registered" | Call `trainer.loss_manager.register_loss_function()` |
| "Batch must contain labels" | Add `"labels"` or `"targets"` key to batch |
| "input_ids not in batch" | Add `"input_ids"` or `"inputs"` key to batch |
| "Trainer not initialized" | Call `trainer.initialize(optimizer)` |
| Model input error | Use standard batch format with "input_ids" |

## Files Location

```
/project/code/src/Ava/training/train/
├── trainer.py              ← SimplifiedEnhancedTrainer
├── distributed_manager.py  ← DistributedTrainingManager
├── checkpoint_manager.py   ← CheckpointManager
├── loss_manager.py         ← LossComputationManager
├── monitoring_manager.py   ← MonitoringManager
├── base.py                 ← Base classes
├── __init__.py             ← Module exports
└── example_training.py     ← Working examples
```

## Key Metrics from Status

```python
status = trainer.get_status()

# Loss manager metrics
status['loss']['current_loss']      # Current loss value
status['loss']['average_loss']      # Average over window
status['loss']['consecutive_nan']   # NaN loss count

# Monitoring metrics
status['monitoring']['avg_step_time_ms']  # Step duration
status['monitoring']['epoch']             # Current epoch
status['monitoring']['step']              # Current step

# Checkpoint metrics
status['checkpoint']['best_loss']         # Best loss seen
status['checkpoint']['checkpoint_count']  # Total checkpoints saved
```

## Configuration

```python
config = EnhancedTrainingConfig(config_file=None)

# Key settings
config.training.gradient_clip_value = 1.0
config.training.log_frequency = 100
config.training.checkpoint_frequency = 1000

# W&B logging
config.wandb.enabled = True
config.wandb.project = "my-project"

# Distributed
config.distributed.enabled = True
```

## Cheat Sheet

| Want to... | Do this |
|-----------|---------|
| Run one training step | `metrics = trainer.train_step(batch)` |
| Train full epoch | `trainer.train_epoch(loader)` |
| Evaluate | `trainer.evaluate(val_loader)` |
| Save checkpoint | `trainer.save_checkpoint()` |
| Load checkpoint | `trainer.load_checkpoint(path)` |
| Add loss function | `trainer.loss_manager.register_loss_function(...)` |
| Get status | `status = trainer.get_status()` |
| Stop training | `trainer.cleanup()` |
| Check rank | `trainer.distributed_manager.is_main_rank()` |
| Log custom metric | `trainer.monitoring_manager.log_metrics({...})` |

---

## See Also

- **Navigation**: [`START_HERE.md`](START_HERE.md)
- **Getting started**: [`GETTING_STARTED.md`](GETTING_STARTED.md)
- **Complete reference**: [`README.md`](README.md)
- **For AI developers**: [`claude.md`](claude.md)
- **Architecture details**: [`ARCHITECTURE.md`](ARCHITECTURE.md)
