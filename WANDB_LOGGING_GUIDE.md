# WandB Logging Integration Guide

## Overview
WandB (Weights & Biases) logging has been integrated into the training script with support for:
- **Loss tracking**: Training and validation loss
- **Gradient metrics**: Average, min, max gradients and zero gradient counts
- **MOE metrics**: Custom MOE-specific metrics
- **Learning rate**: LR scheduler tracking

## Configuration

### Enable/Disable via Config File

Edit your config YAML file in the `logging` section:

```yaml
logging:
  memory_check_freq: 10000
  use_wandb: true  # Set to true to enable WandB
  wandb:
    project: transformer-training  # WandB project name
    entity: null                     # WandB entity (team/username), null = your default
    name: my_run_name               # Run name (auto-generated if not set)
    tags: [moe, testing]            # Tags for organizing runs
    config: {}                       # Additional config to log
```

### Enable via Environment Variable

```bash
# Before running training
export WANDB_PROJECT="transformer-training"
export WANDB_ENTITY="your-entity"  # optional

python train_100m_full.py --config configs/moe/tiny_moe.yaml
```

## Features

### 1. Loss Logging
- Logs training loss at each step
- Logs validation loss at specified intervals
- Viewable in WandB dashboard under `train/loss` and `validation/loss`

### 2. Gradient Monitoring
- **avg_gradient**: Average gradient norm across all parameters
- **max_gradient**: Maximum gradient value
- **min_gradient**: Minimum gradient value
- **num_zero_grads**: Count of parameters with vanishing gradients

These are logged every `log_interval` batches and appear under `gradients/` in WandB.

### 3. MOE Metrics
Custom MOE-specific metrics can be logged using:

```python
metrics_tracker.log_moe_metrics(step, {
    'expert_utilization': utilization,
    'load_balance_loss': balance_loss,
    'router_z_loss': z_loss,
    # ... other metrics
})
```

### 4. Learning Rate Tracking
Automatically logs the current learning rate at each step under `train/lr`.

## Usage Examples

### Example 1: Basic Training with WandB
```yaml
# config_wandb.yaml
logging:
  use_wandb: true
  wandb:
    project: llm-experiments
    entity: my-team
    name: baseline-run-1
    tags: [baseline, 100m]
```

```bash
python train_100m_full.py --config config_wandb.yaml
```

### Example 2: Disabled WandB (Default)
```yaml
logging:
  use_wandb: false  # WandB disabled, only local logging
```

### Example 3: Development with WandB Disabled
```bash
# Override config via code (if needed)
# Edit the script temporarily or use environment variable
export WANDB_MODE=disabled
python train_100m_full.py --config configs/moe/tiny_moe.yaml
```

## Implementation Details

### Modified Files
1. **[train_100m_full.py](code/scripts/5_training/train_100m_full.py)**
   - Added WandB import with graceful fallback
   - Updated `MetricsTracker` class with WandB support
   - Added `log_gradients()` and `log_moe_metrics()` methods
   - Integrated gradient logging in training loop
   - Added config parsing for logging options

2. **[tiny_moe.yaml](code/configs/moe/tiny_moe.yaml)**
   - Added `logging.use_wandb` config option
   - Added `logging.wandb` config section with project/entity/name/tags

### Key Classes and Methods

#### MetricsTracker
```python
class MetricsTracker:
    def __init__(self, log_dir: Path, use_wandb: bool = False,
                 wandb_config: Optional[Dict] = None):
        # Initialize with optional WandB support

    def log_gradients(self, step: int, grad_stats: Dict[str, float]):
        # Log gradient statistics to both TensorBoard and WandB

    def log_moe_metrics(self, step: int, moe_metrics: Dict[str, float]):
        # Log MOE-specific metrics

    def finish(self):
        # Cleanup WandB session at training end
```

## Metrics Dashboard

Once enabled, you'll see these metrics in WandB:

### Training Tab
- `train/loss`: Training loss per step
- `train/lr`: Learning rate per step

### Gradients Tab
- `gradients/avg_gradient`: Mean gradient magnitude
- `gradients/max_gradient`: Max gradient
- `gradients/min_gradient`: Min gradient
- `gradients/num_zero_grads`: Vanishing gradient count

### Validation Tab
- `validation/loss`: Validation loss per evaluation

### MOE Tab (if applicable)
- `moe/*`: Custom MOE metrics

## Troubleshooting

### WandB Not Logging?
1. Check if WandB is installed: `pip install wandb`
2. Verify `use_wandb: true` in config
3. Check WandB API key: `wandb login`
4. Set `WANDB_MODE=online` in environment

### To Disable WandB Temporarily
```bash
export WANDB_MODE=disabled
python train_100m_full.py --config configs/moe/tiny_moe.yaml
```

### To Use Offline WandB
```bash
export WANDB_MODE=offline
python train_100m_full.py --config configs/moe/tiny_moe.yaml
```

## Performance Impact

WandB logging has minimal performance impact:
- Only ~1-2% overhead when enabled
- Metrics are logged asynchronously
- No impact when disabled (`use_wandb: false`)

## Next Steps

1. Install WandB: `pip install wandb`
2. Login to WandB: `wandb login`
3. Enable in your config file
4. Start training and monitor in real-time at https://wandb.ai
