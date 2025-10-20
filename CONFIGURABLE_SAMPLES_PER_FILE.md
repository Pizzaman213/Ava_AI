# Configurable samples_per_file Parameter

## Overview

The `samples_per_file` parameter controls how many samples are read from each data file before rotating to the next file. This is now fully configurable!

## Changes Made

### 1. Added Parameter to All Dataset Classes

**Files Modified:**
- `code/src/Ava/data_streaming.py`
  - `StreamingDataset.__init__` (line 177)
  - `InfiniteStreamingDataset.__init__` (line 999)
  - `create_streaming_dataloaders` (line 819)

**New Parameter:**
```python
samples_per_file: int = 1  # Default: 1 for maximum diversity
```

### 2. Updated Implementation

**In `_stream_examples` method (line 546):**
```python
# Changed from hardcoded:
samples_per_file = 1

# To configurable:
samples_per_file = self.samples_per_file
```

### 3. Integration with Training Script

**In `train.py` (lines 657, 668, 686):**
```python
# Read from config
samples_per_file = getattr(training_config.data, 'samples_per_file', 1)

# Display in training config
print(f"   Samples per file rotation: {samples_per_file} (1=max diversity, higher=less I/O)")

# Pass to dataloader creation
train_loader, val_loader = create_streaming_dataloaders(
    ...
    samples_per_file=samples_per_file,
)
```

### 4. Updated Logging

**In `_stream_examples` (line 542):**
```python
print(f"  📚 Streaming from {unique_files} unique files (all workers access all files, rotating {self.samples_per_file} sample(s) per file)")
```

## How to Use

### Option 1: Via YAML Configuration

Create or modify your data configuration YAML file:

```yaml
data:
  data_dir: /project/code/data/processed
  max_length: 2048
  buffer_size: 50000
  samples_per_file: 1    # Add this line
```

### Option 2: Default Behavior

If not specified, defaults to `1` (maximum diversity).

## Configuration Values

| Value | Behavior | Use Case |
|-------|----------|----------|
| `1` | **Maximum diversity** (recommended) | Best data mixing, diverse batches, prevents dataset bias |
| `10` | Good balance | Moderate diversity with reduced I/O overhead |
| `50` | Reduced diversity | Better I/O performance, still reasonable mixing |
| `100` | Batch-like | Minimal I/O overhead, less diversity (closer to old 500 behavior) |
| `500` | Old behavior | Maximum I/O efficiency, minimal diversity (not recommended) |

## Recommendations

### For Training from Scratch
```yaml
samples_per_file: 1
```
- Maximum data diversity
- Best for preventing overfitting to single datasets
- Recommended for most training scenarios

### For Fine-tuning
```yaml
samples_per_file: 10
```
- Good balance between diversity and speed
- Acceptable for shorter training runs

### For Fast Iteration/Debugging
```yaml
samples_per_file: 100
```
- Reduced I/O overhead
- Use only for quick experiments

## Example Configurations

### Maximum Diversity (Recommended)
```yaml
data:
  data_dir: /project/code/data/processed
  max_length: 2048
  buffer_size: 50000
  samples_per_file: 1
  num_workers: 8
```

### Balanced Performance
```yaml
data:
  data_dir: /project/code/data/processed
  max_length: 2048
  buffer_size: 50000
  samples_per_file: 10
  num_workers: 8
```

### Fast I/O (Not Recommended for Training)
```yaml
data:
  data_dir: /project/code/data/processed
  max_length: 2048
  buffer_size: 50000
  samples_per_file: 100
  num_workers: 8
```

## How It Works

### With samples_per_file=1 (Default):
```
File A → sample 0
File B → sample 1
File C → sample 2
File D → sample 3
File E → sample 4
...
File A → sample 11  (rotates back)
File B → sample 12
...
```

### With samples_per_file=10:
```
File A → samples 0-9
File B → samples 10-19
File C → samples 20-29
...
File A → samples 110-119  (rotates back)
...
```

## Benefits

✓ **Flexible configuration** - Tune based on your needs
✓ **Maximum diversity default** - Best practice out of the box
✓ **I/O optimization option** - Can reduce overhead when needed
✓ **Backward compatible** - Defaults to best behavior
✓ **Clear documentation** - Shows value in training logs

## Training Output

When training starts, you'll see:

```
🎯 Training Configuration:
   Batch size: 4
   Gradient accumulation steps: 1
   Effective batch size: 4
   Training samples: 1,000,000
   Validation samples: 100,000 (10.0% of training)
   Expected training steps: 250,000
   Workers: 8
   Buffer size: 50,000
   Samples per file rotation: 1 (1=max diversity, higher=less I/O)
================================================================================

 Creating enhanced streaming dataloaders...
Creating streaming dataloaders...
  📚 Streaming from 11 unique files (all workers access all files, rotating 1 sample(s) per file)
```

## Testing

Run the test script to verify:
```bash
python /project/test_samples_config_simple.py
```

Expected output: `Tests passed: 9/9` ✅

## Summary

The `samples_per_file` parameter is now fully configurable across all components:

1. ✅ StreamingDataset
2. ✅ InfiniteStreamingDataset
3. ✅ create_streaming_dataloaders
4. ✅ train.py integration
5. ✅ Configuration display
6. ✅ Logging messages
7. ✅ Documentation
8. ✅ Tests

Default value of `1` provides maximum data diversity while allowing optimization when needed.
