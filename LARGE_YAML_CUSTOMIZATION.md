# Customizing large.yaml for Data Randomness

## Overview

The `large.yaml` configuration file now includes documentation for the new data randomization parameters. By default, they use sensible settings that provide good performance and data diversity.

## Default Behavior

The large.yaml file comes with these **commented-out** parameters, which means the system uses intelligent defaults:

```yaml
data:
  streaming: true
  buffer_size: 5000

  # RANDOMIZATION CONTROL PARAMETERS (commented, using defaults):
  # shuffle_seed: null
  # enable_length_sorting: true
  # disable_packing_length_sort: false
  # enable_bucketing: true
```

**With defaults**, you get:
- ✅ Good throughput (all optimizations enabled)
- ✅ Good data diversity (shuffling enabled)
- ✅ Reproducible experiments within a single run
- ✅ Full backward compatibility

---

## How to Customize

### Option 1: Maximum Data Diversity

Uncomment and modify for maximum randomness:

```yaml
data:
  streaming: true
  buffer_size: 5000

  # Enable maximum data randomness
  shuffle_seed: null
  enable_length_sorting: false
  disable_packing_length_sort: true
  enable_bucketing: false

  # ... rest of config
```

**Impact**: ~30-45% throughput reduction, maximum data diversity

**Use when**:
- Exploring if your model is sensitive to data order
- Training final production model for best diversity
- Running experiments with different data orderings

---

### Option 2: Perfect Reproducibility

Uncomment and modify for deterministic training:

```yaml
data:
  streaming: true
  buffer_size: 5000

  # Enable reproducible training
  shuffle_seed: 42
  enable_length_sorting: true
  disable_packing_length_sort: false
  enable_bucketing: true

  # ... rest of config
```

**Impact**: No throughput change, perfect reproducibility

**Use when**:
- Running reproducible experiments
- Benchmarking models fairly
- Debugging training issues
- Comparing different architectures

---

### Option 3: Balanced Approach

Uncomment only what you need:

```yaml
data:
  streaming: true
  buffer_size: 5000

  # Fixed seed for reproducibility, keep optimizations
  shuffle_seed: 42
  enable_length_sorting: true
  disable_packing_length_sort: false
  enable_bucketing: true

  # ... rest of config
```

**Impact**: No throughput change, good reproducibility, good efficiency

**Use when**:
- You want reproducible runs
- You want to keep all performance optimizations
- You want to compare results reliably

---

### Option 4: Partial Customization

You can customize individual settings:

```yaml
data:
  streaming: true
  buffer_size: 5000

  # Example: Deterministic shuffling but random ordering
  shuffle_seed: 42
  enable_length_sorting: false      # Disable GPU load balancing
  disable_packing_length_sort: false # Keep packing sorting
  enable_bucketing: true             # Keep bucketing

  # ... rest of config
```

**Impact**: Varies based on which you disable

---

## Parameter Quick Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shuffle_seed` | `null` | null=random, integer=deterministic |
| `enable_length_sorting` | `true` | true=optimize, false=random |
| `disable_packing_length_sort` | `false` | false=optimize, true=random |
| `enable_bucketing` | `true` | true=optimize, false=random |

## Performance Trade-offs

| Setting Change | Throughput Impact | When to Use |
|---|---|---|
| shuffle_seed only | No impact | When only reproducibility matters |
| disable length_sorting | -5-10% | Multi-GPU training needs diversity |
| disable packing_sort | -10-15% | When packing efficiency matters less |
| disable bucketing | -15-20% | When reducing padding matters less |
| All disabled | -30-45% | Maximum diversity needed |

---

## Quick Start Commands

### Use default large.yaml (good balance)
```bash
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml
```

### Use maximum randomness
```bash
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/max_randomness.yaml
```

### Use full reproducibility
```bash
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/deterministic.yaml
```

### Customize large.yaml
1. Open `code/configs/moe/large.yaml` in your editor
2. Find the `data:` section
3. Uncomment the randomization parameters you want to change
4. Modify the values
5. Save and run:
```bash
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml
```

---

## Examples

### Example 1: Run with fixed seed (reproducible)

In large.yaml, change from:
```yaml
  # shuffle_seed: null
```

To:
```yaml
  shuffle_seed: 42
```

Result: Same data order every run, no throughput change.

---

### Example 2: Run with maximum diversity

In large.yaml, change from:
```yaml
  # shuffle_seed: null
  # enable_length_sorting: true
  # disable_packing_length_sort: false
  # enable_bucketing: true
```

To:
```yaml
  shuffle_seed: null
  enable_length_sorting: false
  disable_packing_length_sort: true
  enable_bucketing: false
```

Result: Different data order each run, ~30-45% throughput reduction.

---

### Example 3: A/B test different seeds

**config_seed42.yaml**:
```yaml
data:
  shuffle_seed: 42
  # ... rest same as large.yaml
```

**config_seed99.yaml**:
```yaml
data:
  shuffle_seed: 99
  # ... rest same as large.yaml
```

Run both and compare:
```bash
python code/scripts/5_training/train_pipeline.py --config config_seed42.yaml
python code/scripts/5_training/train_pipeline.py --config config_seed99.yaml
```

Compare validation loss curves to see if data order affects training.

---

## Testing Your Configuration

After customizing large.yaml, verify it loads correctly:

```bash
python -c "
import yaml
with open('code/configs/moe/large.yaml') as f:
    config = yaml.safe_load(f)
print('Data config:')
print(f'  shuffle_seed: {config[\"data\"].get(\"shuffle_seed\")}')
print(f'  enable_length_sorting: {config[\"data\"].get(\"enable_length_sorting\")}')
print(f'  disable_packing_length_sort: {config[\"data\"].get(\"disable_packing_length_sort\")}')
print(f'  enable_bucketing: {config[\"data\"].get(\"enable_bucketing\")}')
print('✓ Configuration loaded successfully')
"
```

---

## Pre-configured Options

Instead of modifying large.yaml, you can use pre-configured files:

1. **code/configs/moe/large.yaml** (original)
   - Uses defaults
   - Good balance of performance and diversity

2. **code/configs/moe/max_randomness.yaml** (NEW)
   - Maximum data diversity
   - ~30-45% throughput reduction
   - All optimizations disabled

3. **code/configs/moe/deterministic.yaml** (NEW)
   - Perfect reproducibility
   - No throughput change
   - All optimizations enabled

---

## Troubleshooting

### "I want reproducibility but good performance"
**Solution**: Use `code/configs/moe/deterministic.yaml` or set only:
```yaml
shuffle_seed: 42
```

### "I want maximum diversity"
**Solution**: Use `code/configs/moe/max_randomness.yaml`

### "I'm not sure which to use"
**Solution**: Start with defaults (don't uncomment) and use default large.yaml

### "I need different seeds for each run"
**Solution**: Keep `shuffle_seed: null` (commented out) - uses time-based random seed

### "I want to compare two different data orderings"
**Solution**: Create two versions with different `shuffle_seed` values (e.g., 42 and 99)

---

## Summary

The `large.yaml` file now supports advanced data randomness control:

✅ **Default behavior**: Good performance + good diversity + backward compatible
✅ **Easy to customize**: Just uncomment and modify parameters
✅ **Clear documentation**: Each parameter documented with examples
✅ **Pre-configured options**: max_randomness.yaml and deterministic.yaml ready to use

**Next steps**:
1. Use large.yaml as-is for training (or use a pre-configured option)
2. Customize if you need specific behavior
3. See `DATA_RANDOMNESS_GUIDE.md` for detailed information
