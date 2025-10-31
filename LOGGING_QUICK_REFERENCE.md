# Logging Quick Reference Guide

## For train.py Users

### Automatic Logging Setup

Logging is automatically initialized when you run `train.py`. No additional setup required!

```bash
python code/scripts/5_training/train.py --config code/configs/gpu/small.yaml
```

### Where Are My Logs?

All logs are saved to your run directory:

```
outputs/runs/run_YYYYMMDD_HHMMSS_<id>/logs/
├── training_rank_0.log    # Full detailed log (DEBUG+)
└── errors_rank_0.log      # Errors and warnings only (WARNING+)
```

### Log Levels

| Level | When to Use | Console | File | Example |
|-------|-------------|---------|------|---------|
| **DEBUG** | Detailed diagnostic info | ❌ No | ✅ Yes | `logger.debug("Batch details: size=32, seq_len=512")` |
| **INFO** | General progress updates | ✅ Yes | ✅ Yes | `logger.info("Model loaded successfully")` |
| **WARNING** | Something unexpected but recoverable | ✅ Yes | ✅ Yes | `logger.warning("GPU memory usage high")` |
| **ERROR** | Something failed but training continues | ✅ Yes | ✅ Yes | `logger.error("Failed to save checkpoint")` |
| **CRITICAL** | Fatal error, training must stop | ✅ Yes | ✅ Yes | `logger.critical("OOM, cannot continue")` |

### Common Logging Patterns

#### Simple Messages
```python
logger.info("Starting training...")
logger.warning("Batch size reduced due to OOM")
logger.error("Checkpoint save failed")
```

#### Messages with Variables
```python
logger.info(f"Epoch {epoch}/{total_epochs} completed")
logger.info(f"Loss: {loss:.4f}, LR: {lr:.2e}")
logger.debug(f"Batch {batch_idx}: {num_tokens} tokens processed")
```

#### Using LogPhase (Auto-timing)
```python
with LogPhase(logger, "Data Loading"):
    train_loader = create_dataloader(...)
    # Automatically logs: "✅ Data Loading completed in 2.34s"
```

#### Exceptions with Stack Traces
```python
try:
    model.load_checkpoint(path)
except Exception as e:
    logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
    # exc_info=True adds full stack trace to log file
```

### Viewing Logs

#### Real-time Monitoring
```bash
# Follow training progress
tail -f outputs/runs/{run_id}/logs/training_rank_0.log

# Watch for errors only
tail -f outputs/runs/{run_id}/logs/errors_rank_0.log
```

#### Search and Filter
```bash
# Find all errors
grep "\[ERROR\]" outputs/runs/{run_id}/logs/training_rank_0.log

# Find specific function calls
grep "train_epoch" outputs/runs/{run_id}/logs/training_rank_0.log

# Find OOM events
grep -i "out of memory" outputs/runs/{run_id}/logs/training_rank_0.log

# Find all checkpoints saves
grep "checkpoint" outputs/runs/{run_id}/logs/training_rank_0.log
```

#### View Specific Time Range
```bash
# All logs from 14:30 onwards
grep "14:3[0-9]:" outputs/runs/{run_id}/logs/training_rank_0.log

# Logs from specific date
grep "2025-10-29" outputs/runs/{run_id}/logs/training_rank_0.log
```

### Multi-GPU / Distributed Training

Each GPU rank gets its own log file:
```
logs/
├── training_rank_0.log
├── training_rank_1.log
├── training_rank_2.log
├── training_rank_3.log
├── errors_rank_0.log
├── errors_rank_1.log
├── errors_rank_2.log
└── errors_rank_3.log
```

View all ranks simultaneously:
```bash
# View all training logs
tail -f outputs/runs/{run_id}/logs/training_rank_*.log

# Check for errors on any rank
grep "\[ERROR\]" outputs/runs/{run_id}/logs/errors_rank_*.log
```

### Console Output Colors

The console automatically shows colored output:

- 🔍 **DEBUG** (Cyan) - Detailed diagnostics
- ✓ **INFO** (Green) - Normal progress
- ⚠️ **WARNING** (Yellow) - Warnings
- ❌ **ERROR** (Red) - Errors
- 🛑 **CRITICAL** (Magenta) - Fatal errors

### Advanced: Changing Log Level

To see more/less detail, edit the logging setup:

```python
# In train.py, modify setup_training_logger():

# More verbose (show DEBUG on console)
console_handler.setLevel(logging.DEBUG)  # Was: logging.INFO

# Less verbose (only show warnings/errors on console)
console_handler.setLevel(logging.WARNING)  # Was: logging.INFO
```

### Troubleshooting

**Q: I don't see any log files**
- Check that the run directory was created: `ls outputs/runs/`
- Verify logger was initialized: Look for "🚀 Ava Training Pipeline - Starting..." in console

**Q: Console output has weird color codes**
- Your terminal might not support ANSI colors
- Check terminal with: `echo -e "\033[32mGreen\033[0m"`
- Use a modern terminal (Windows Terminal, iTerm2, etc.)

**Q: Log files are too large**
- Current implementation doesn't rotate logs
- You can manually clean old runs: `rm -rf outputs/runs/run_*` (careful!)
- Or implement log rotation (see LOGGING_IMPROVEMENTS.md)

**Q: Logs show wrong timestamps**
- Check system timezone: `date`
- Logs use local system time

**Q: How do I disable file logging?**
```python
# In train.py main(), pass log_dir=None:
logger = setup_training_logger(log_dir=None, rank=rank)
```

**Q: Can I log to custom locations?**
```python
# Yes! Change log_dir:
log_dir = Path("/my/custom/logs")
logger = setup_training_logger(log_dir=log_dir, rank=rank)
```

---

## For Developers

### Adding New Logging to Functions

If you add new functions to train.py, use the global logger:

```python
def my_new_function():
    """My new training function."""
    global logger  # Access global logger

    logger.info("Starting my new function")

    try:
        # ... your code ...
        logger.debug(f"Processed {count} items")
    except Exception as e:
        logger.error(f"Function failed: {e}", exc_info=True)
        raise

    logger.info("Function completed successfully")
```

### Using LogPhase in New Code

```python
def setup_advanced_features():
    """Setup advanced training features."""

    with LogPhase(logger, "Advanced Features Setup",
                  feature_count=5, mode="production"):
        # Setup code here
        feature_1()
        feature_2()
        # ...
    # Automatically logs completion with timing
```

### Custom Log Formatters

Want different formatting? Create a custom formatter:

```python
class MyCustomFormatter(logging.Formatter):
    def format(self, record):
        # Custom formatting logic
        return f"[{record.levelname}] {record.getMessage()}"

# Use it:
handler.setFormatter(MyCustomFormatter())
```

### Integration with External Tools

#### WandB
Logging works alongside WandB (if available):
```python
# Logs go to both logger AND wandb
logger.info(f"Epoch {epoch} loss: {loss}")
wandb.log({"train/loss": loss})
```

#### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir / "tensorboard")

# Log to both
logger.info(f"Loss: {loss}")
writer.add_scalar("Loss/train", loss, step)
```

---

**Need Help?** See [LOGGING_IMPROVEMENTS.md](LOGGING_IMPROVEMENTS.md) for full documentation.
