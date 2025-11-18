# Getting Started with the Modular Training Framework

Welcome! This guide will get you up and running with the new modular training framework in just a few minutes.

## Quick Start (5 minutes)

### 1. Create a Simple Trainer

```python
from Ava.training.train import SimplifiedEnhancedTrainer
from Ava.config.training_config import EnhancedTrainingConfig
import torch
import torch.nn as nn

# Create your model
model = nn.Linear(100, 10)

# Create configuration
config = EnhancedTrainingConfig()

# Create trainer
trainer = SimplifiedEnhancedTrainer(
    model=model,
    config=config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Initialize with optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer.initialize(optimizer)

print("✓ Trainer ready to use!")
```

### 2. Train for One Epoch

```python
# Your data loader (replace with your actual data)
def create_data_loader():
    for _ in range(10):  # 10 batches
        yield {
            "input_ids": torch.randn(32, 100),
            "labels": torch.randint(0, 10, (32,))
        }

# Train
epoch_metrics = trainer.train_epoch(create_data_loader())
print(f"Epoch loss: {epoch_metrics['avg_loss']:.4f}")

# Cleanup
trainer.cleanup()
```

That's it! You now have a fully functional training loop.

## Next Steps

### Add Custom Loss Functions (10 minutes)

```python
from torch.nn import CrossEntropyLoss, MSELoss

# Register loss functions
trainer.loss_manager.register_loss_function(
    "classification",
    CrossEntropyLoss(),
    weight=1.0
)

trainer.loss_manager.register_loss_function(
    "auxiliary",
    MSELoss(),
    weight=0.1
)

# Training step automatically computes both losses
for batch in data_loader:
    metrics = trainer.train_step(batch)
    # metrics includes breakdown of each loss function
    print(f"CE Loss: {metrics['classification']:.4f}")
    print(f"MSE Loss: {metrics['auxiliary']:.4f}")
```

### Save and Load Checkpoints (5 minutes)

```python
# Save checkpoint
trainer.save_checkpoint(save_best=True)

# Later, load it
trainer.load_checkpoint("checkpoints/best_model.pt")
```

### Monitor Training with W&B (5 minutes)

```python
# Configure W&B in your config
config.wandb.enabled = True
config.wandb.project = "my-project"

trainer = SimplifiedEnhancedTrainer(model, config, device)
trainer.initialize(optimizer)

# Logging happens automatically!
for epoch in range(num_epochs):
    epoch_metrics = trainer.train_epoch(data_loader)
    # Metrics automatically logged to W&B
```

### Evaluate on Validation Data (5 minutes)

```python
# Create validation data loader
val_loader = create_data_loader(split='val')

# Evaluate (can be called during or after training)
eval_metrics = trainer.evaluate(val_loader)
print(f"Validation loss: {eval_metrics['eval_loss']:.4f}")

# Metrics automatically logged
```

## Understanding the Framework

### What Each Component Does

```
SimplifiedEnhancedTrainer (Your main interface)
├── DistributedTrainingManager
│   └─ Handles multi-GPU/multi-node training
├── CheckpointManager
│   └─ Saves and loads models
├── LossComputationManager
│   └─ Computes losses and manages gradients
└── MonitoringManager
    └─ Tracks metrics and logs to W&B
```

### Key Methods

| Method | What It Does | Example |
|--------|-------------|---------|
| `initialize(optimizer)` | Setup trainer | `trainer.initialize(optimizer)` |
| `train_step(batch)` | One training step | `metrics = trainer.train_step(batch)` |
| `train_epoch(loader)` | Full epoch | `epoch_metrics = trainer.train_epoch(loader)` |
| `evaluate(loader)` | Validation | `eval_metrics = trainer.evaluate(val_loader)` |
| `save_checkpoint()` | Save model | `trainer.save_checkpoint(save_best=True)` |
| `load_checkpoint(path)` | Load model | `trainer.load_checkpoint(path)` |
| `cleanup()` | Cleanup resources | `trainer.cleanup()` |
| `get_status()` | Debug info | `status = trainer.get_status()` |

## Common Patterns

### Pattern 1: Full Training Loop

```python
trainer = SimplifiedEnhancedTrainer(model, config, device)
trainer.initialize(optimizer)

try:
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Evaluate
        val_metrics = trainer.evaluate(val_loader)

        # Save best checkpoint
        trainer.save_checkpoint(save_best=True)

        print(f"Epoch {epoch}: "
              f"train_loss={train_metrics['avg_loss']:.4f}, "
              f"val_loss={val_metrics['eval_loss']:.4f}")
finally:
    trainer.cleanup()
```

### Pattern 2: Per-Step Control

```python
trainer = SimplifiedEnhancedTrainer(model, config, device)
trainer.initialize(optimizer)

for epoch in range(num_epochs):
    trainer.monitoring_manager.on_epoch_start(epoch)

    for batch in train_loader:
        metrics = trainer.train_step(batch)

        # Custom logging
        if trainer.context.step % 100 == 0:
            print(f"Step {trainer.context.step}: "
                  f"loss={metrics['loss']:.4f}")

    trainer.monitoring_manager.on_epoch_end(epoch)

trainer.cleanup()
```

### Pattern 3: Using Managers Directly

```python
from Ava.training.train import (
    TrainingContext,
    CheckpointManager,
    LossComputationManager
)

# Create context
context = TrainingContext(model, device, config)

# Use managers independently
loss_mgr = LossComputationManager(context)
loss_mgr.initialize()
loss_mgr.register_loss_function("ce", CrossEntropyLoss())

# Can now test loss computation without full trainer
loss, breakdown = loss_mgr.compute_loss(outputs, targets)

# Checkpoint manager
ckpt_mgr = CheckpointManager(context)
ckpt_mgr.initialize()
ckpt_path = ckpt_mgr.save_checkpoint(epoch=0, step=100)
```

## Configuration

The framework reads from your `EnhancedTrainingConfig`:

```python
config = EnhancedTrainingConfig()

# Distributed training
config.distributed.enabled = True

# Gradient clipping
config.training.gradient_clip_value = 1.0

# Loss tracking
config.training.max_consecutive_nan_losses = 5

# Logging
config.training.log_frequency = 100

# W&B
config.wandb.enabled = True
config.wandb.project = "my-project"

trainer = SimplifiedEnhancedTrainer(model, config, device)
# All settings automatically applied!
```

## Debugging Tips

### Getting Full Status

```python
status = trainer.get_status()

# Training state
print(f"Epoch: {status['training_context']['epoch']}")
print(f"Step: {status['training_context']['step']}")

# Manager status
print(f"Distributed: {status['distributed']}")
print(f"Checkpoint best loss: {status['checkpoint']['best_loss']}")
print(f"Loss breakdown: {status['loss']['loss_breakdown']}")
```

### Monitoring Specific Manager

```python
# Check distributed training
dist_status = trainer.distributed_manager.get_status()
print(f"Is main rank: {dist_status['is_main_rank']}")

# Check checkpoint manager
ckpt_status = trainer.checkpoint_manager.get_status()
print(f"Best loss: {ckpt_status['best_loss']}")

# Check monitoring
mon_status = trainer.monitoring_manager.get_status()
print(f"Step time: {mon_status['avg_step_time_ms']:.1f}ms")
```

### Handling Errors

```python
try:
    trainer.train_epoch(train_loader)
except RuntimeError as e:
    if "NaN loss" in str(e):
        print("Loss became NaN - check your data and loss functions")
        # Get loss manager status for debugging
        print(trainer.loss_manager.get_status())
    else:
        print(f"Training error: {e}")
    raise
finally:
    trainer.cleanup()
```

## Common Issues

### "Trainer not initialized" Error

```python
trainer = SimplifiedEnhancedTrainer(model, config)
metrics = trainer.train_step(batch)  # ❌ Error!

# Fix: Must initialize first
trainer = SimplifiedEnhancedTrainer(model, config)
trainer.initialize(optimizer)  # ✓ Now it works
metrics = trainer.train_step(batch)
```

### "No loss functions registered"

```python
trainer = SimplifiedEnhancedTrainer(model, config)
trainer.initialize(optimizer)

# ❌ This will fail - no loss functions
loss, _ = trainer.loss_manager.compute_loss(outputs, targets)

# ✓ Register loss functions first
trainer.loss_manager.register_loss_function("ce", CrossEntropyLoss())
loss, _ = trainer.loss_manager.compute_loss(outputs, targets)
```

### Data Format Issues

Batch dictionary must have "labels" or "targets":

```python
# ❌ This will fail
batch = {"input_ids": torch.randn(32, 128)}
trainer.train_step(batch)  # Error: no labels

# ✓ Include labels
batch = {
    "input_ids": torch.randn(32, 128),
    "labels": torch.randint(0, 10, (32,))
}
trainer.train_step(batch)
```

## Performance Tips

### 1. Use Async Checkpointing

```python
# Saves checkpoint in background thread
trainer.checkpoint_manager.save_checkpoint(
    epoch=epoch,
    step=step,
    optimizer=optimizer,
    async_save=True  # 20-30x faster!
)
```

### 2. Configure Log Frequency

```python
# Less frequent logging = faster training
config.training.log_frequency = 1000  # Log every 1000 steps instead of 100
```

### 3. Enable Gradient Checkpointing

```python
# Saves memory for large models
config.training.gradient_checkpointing = True
```

### 4. Use Distributed Training

```python
# Automatic if using distributed launcher
config.distributed.enabled = True
```

## Next: Read Full Documentation

After this quick start:

1. **README.md** - Complete component documentation
2. **ARCHITECTURE.md** - Why changes were made
3. **MIGRATION_GUIDE.md** - If migrating from old trainer
4. **example_training.py** - More examples
5. **IMPLEMENTATION_SUMMARY.md** - What was created

## Still Have Questions?

1. Check the component's docstring: `help(trainer.loss_manager)`
2. Look at examples: `example_training.py`
3. Check migration guide: `MIGRATION_GUIDE.md`
4. Review status output: `trainer.get_status()`

## Summary

You now know how to:
- ✅ Create and initialize a trainer
- ✅ Run a training loop
- ✅ Add custom loss functions
- ✅ Save and load checkpoints
- ✅ Evaluate on validation data
- ✅ Debug training issues
- ✅ Access manager components

Next: Run `example_training.py` to see it in action!
