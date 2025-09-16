# Output Directory Structure

All training outputs, logs, and generated files are saved to `/project/code/outputs/`.

## Directory Structure

```
/project/code/outputs/
├── training_*.log           # Training logs with timestamps
├── evaluation_*.log         # Evaluation logs
├── metrics_*.json          # Evaluation metrics
├── best_model.pt           # Best model checkpoint
├── final_model.pt          # Final model checkpoint
├── checkpoint_epoch_*.pt   # Periodic checkpoints
├── generated_*.txt         # Generated text outputs
└── run_*/                  # Training run directories
    ├── training.log
    └── checkpoints/
```

## Usage

### Training Outputs
When you run training, outputs will automatically be saved to `/project/code/outputs/`:

```bash
python scripts/training/train.py --config configs/cpu/small.yaml
# Outputs saved to: /project/code/outputs/
```

### Custom Output Directory
You can override the default output directory:

```bash
python scripts/training/train.py --config configs/cpu/small.yaml --output-dir /custom/path
```

### View Output Summary
To see a summary of all outputs:

```bash
python scripts/show_outputs.py
```

## Log Files

Training logs include:
- Timestamp for each event
- Training loss per batch
- Validation metrics
- Checkpoint save events
- Any warnings or errors

Example log format:
```
2025-09-16 02:19:35,833 - INFO - Starting training...
2025-09-16 02:19:40,123 - INFO - Epoch 1/3: loss=10.2345
```

## Checkpoints

Model checkpoints are saved in PyTorch format and include:
- Model state dict
- Optimizer state dict
- Training epoch and step
- Loss metrics
- Model configuration

Load a checkpoint:
```python
import torch
checkpoint = torch.load('/project/code/outputs/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Metrics

Evaluation metrics are saved as JSON files:
```json
{
  "loss": 2.345,
  "perplexity": 10.43,
  "accuracy": 0.876,
  "timestamp": "2025-09-16T02:19:35"
}
```

## Generated Text

Generated text outputs are saved with timestamps:
- `generated_YYYYMMDD_HHMMSS.txt`
- Includes the prompt and generated text
- Multiple generations are separated by dividers

## Monitoring Training

To monitor training progress in real-time:

```bash
# Watch the latest log file
tail -f /project/code/outputs/training_*.log

# Watch for new checkpoints
watch -n 10 ls -la /project/code/outputs/*.pt
```

## Cleanup

To clean up old outputs (be careful!):

```bash
# Remove logs older than 7 days
find /project/code/outputs -name "*.log" -mtime +7 -delete

# Remove old run directories
rm -rf /project/code/outputs/run_*
```

## Best Practices

1. **Regular Backups**: Backup important checkpoints to external storage
2. **Log Rotation**: Clean up old logs periodically to save space
3. **Naming Convention**: Use descriptive names for custom outputs
4. **Version Control**: Track metrics.json files in git for experiment comparison
5. **Documentation**: Document each training run's purpose and parameters

## Troubleshooting

If outputs are not being saved:
1. Check directory permissions: `ls -la /project/code/outputs`
2. Ensure sufficient disk space: `df -h`
3. Check for errors in training logs
4. Verify output directory exists: `mkdir -p /project/code/outputs`