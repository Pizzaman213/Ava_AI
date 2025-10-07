# Learning Rate Finder Guide

## Overview

The Learning Rate Finder implements Leslie N. Smith's LR Range Test method to automatically discover the optimal learning rate for your model and dataset. This helps you avoid manually tuning the learning rate and can significantly speed up training convergence.

## How It Works

The LR Finder:
1. **Incrementally increases the learning rate** from a very small value to a large value
2. **Records the training loss** at each learning rate
3. **Analyzes the loss curve** to find the "sweet spot" where learning is most efficient
4. **Suggests an optimal learning rate** based on the steepest descent in the loss curve

## Quick Start

### Basic Usage

Run LR Finder before training:

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder
```

This will:
- Run the LR range test for 100 iterations
- Generate a plot showing loss vs learning rate
- Suggest an optimal learning rate
- Continue with normal training using your configured LR

### Auto-Apply Suggested LR

To automatically use the suggested learning rate:

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-use-suggested
```

### Custom LR Range

Specify a custom range to search:

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-start 1e-7 \
  --lr-finder-end 0.1 \
  --lr-finder-iterations 200
```

## Configuration Options

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-lr-finder` | False | Enable LR Finder |
| `--lr-finder-start` | 1e-8 | Starting LR for search |
| `--lr-finder-end` | 1.0 | Ending LR for search |
| `--lr-finder-iterations` | 100 | Number of iterations to test |
| `--lr-finder-method` | steepest | Suggestion method (steepest/minimum/valley) |
| `--lr-finder-use-suggested` | False | Auto-apply suggested LR |
| `--lr-finder-plot-path` | auto | Path to save plot |

### Suggestion Methods

**steepest** (recommended):
- Finds the point with the steepest negative gradient
- Best for most use cases
- Example: If loss drops fastest at LR=0.003, suggest that value

**minimum**:
- Finds the LR with the minimum loss
- Conservative approach
- May suggest too high of a learning rate

**valley**:
- Finds LR at 1/10th of the way to minimum loss
- Very conservative
- Good for unstable training runs

## Interpreting Results

### The Loss Curve

A typical LR Finder plot shows:

```
Loss
 |
 |    \              /
 |     \            /
 |      \__________/
 |        ↑      ↑
 |      ideal   divergence
 +-----------------------> Learning Rate
```

**Regions:**
1. **Too Low** (left): Loss barely decreases → slow learning
2. **Sweet Spot** (middle): Loss decreases rapidly → optimal learning
3. **Too High** (right): Loss explodes → divergence

### Reading the Output

```
✅ LR Finder Complete!
   Suggested Learning Rate: 3.50e-03
   Best Loss: 4.234567 at LR: 5.20e-03
```

- **Suggested LR**: The recommended learning rate to use
- **Best Loss**: The lowest loss achieved during the test
- The suggested LR is typically slightly lower than the LR at minimum loss for stability

## Best Practices

### 1. Run on a Small Dataset First

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder \
  --max-samples 10000 \
  --lr-finder-iterations 50
```

### 2. Adjust Range Based on Model Size

**Large models** (1B+ params):
```bash
--lr-finder-start 1e-9 --lr-finder-end 0.01
```

**Small models** (< 100M params):
```bash
--lr-finder-start 1e-7 --lr-finder-end 1.0
```

### 3. Consider Your Batch Size

Larger effective batch sizes typically need higher learning rates:
```bash
# With gradient accumulation 4x
--lr-finder-end 0.1  # Can go higher

# With gradient accumulation 1x
--lr-finder-end 0.01  # More conservative
```

### 4. Use with Mixed Precision

The LR Finder respects your training configuration including mixed precision:

```yaml
# configs/gpu/small.yaml
training:
  mixed_precision: bf16  # ✓ Already default in small.yaml
```

## Integration with Training Pipeline

### Workflow

1. **Run LR Finder** (one-time setup):
   ```bash
   python scripts/training/train.py \
     --config configs/gpu/small.yaml \
     --run-lr-finder \
     --lr-finder-use-suggested
   ```

2. **Note the suggested LR** from output

3. **Update your config** for future runs:
   ```yaml
   training:
     learning_rate: 0.0035  # Use suggested value
   ```

4. **Train normally**:
   ```bash
   python scripts/training/train.py --config configs/gpu/small.yaml
   ```

### Programmatic Usage

You can also use the LR Finder directly in Python:

```python
from src.Ava.training.lr_finder import LRFinder, LRFinderConfig
import torch

# Setup
config = LRFinderConfig(
    start_lr=1e-7,
    end_lr=1.0,
    num_iter=100,
    suggestion_method='steepest'
)

finder = LRFinder(model, optimizer, criterion, device, config)

# Run test
results = finder.range_test(
    train_loader,
    accumulation_steps=4
)

# Get suggestion
suggested_lr = results['suggested_lr']
print(f"Suggested LR: {suggested_lr:.2e}")

# Apply to optimizer
for param_group in optimizer.param_groups:
    param_group['lr'] = suggested_lr
```

## Troubleshooting

### Issue: Loss Immediately Explodes

**Symptom**: Loss goes to NaN or Inf within first few iterations

**Solution**: Lower the starting LR
```bash
--lr-finder-start 1e-9  # Instead of 1e-8
```

### Issue: No Clear Minimum in Plot

**Symptom**: Loss keeps decreasing or stays flat throughout

**Solution**: Extend the LR range
```bash
--lr-finder-end 10.0  # Increase upper bound
--lr-finder-iterations 200  # More samples
```

### Issue: Suggested LR Seems Too High/Low

**Symptom**: Training with suggested LR diverges or is too slow

**Solution**:
1. Try different suggestion method:
   ```bash
   --lr-finder-method valley  # More conservative
   ```

2. Manually adjust the suggestion:
   ```bash
   # If suggested LR is 0.005, try 50% lower
   --learning-rate 0.0025
   ```

## Advanced Features

### Custom Loss Function

When using custom loss functions, the LR Finder automatically uses your model's loss:

```python
# In your model
class MyModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.transformer(input_ids)

        if labels is not None:
            loss = my_custom_loss(logits, labels)
            return {'loss': loss, 'logits': logits}

        return {'logits': logits}
```

The LR Finder will use `outputs['loss']` automatically.

### Saving Results

The plot is automatically saved to your run directory:
```
outputs/runs/run_YYYYMMDD_HHMMSS_<id>/lr_finder_results.png
```

You can specify a custom path:
```bash
--lr-finder-plot-path /path/to/my_lr_plot.png
```

## Performance Impact

**Time Cost**:
- 100 iterations ≈ 1-2 minutes on small datasets
- Runs on a small subset of data (restores model state after)
- Negligible overhead for the accuracy gain

**Memory**:
- Same as regular training
- Model state is saved and restored
- No additional memory required

## When to Re-run LR Finder

Re-run the LR Finder when you:
- Change model architecture significantly
- Switch to a different dataset
- Change batch size or gradient accumulation
- Change optimizer (Adam → AdamW, etc.)
- Notice training instability

## References

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) - Leslie N. Smith
- [A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820) - Leslie N. Smith

## Examples

### Complete Training Run with LR Finder

```bash
# 1. Find optimal learning rate
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-iterations 150 \
  --lr-finder-method steepest \
  2>&1 | tee lr_finder.log

# Output: Suggested Learning Rate: 4.20e-03

# 2. Train with suggested rate
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --learning-rate 0.0042 \
  --epochs 10 \
  --wandb-project my-project
```

### Quick One-Shot Run

```bash
python scripts/training/train.py \
  --config configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-use-suggested \
  --epochs 10
```

This will:
1. Find the optimal LR automatically
2. Apply it to training
3. Train for 10 epochs with the optimal rate
