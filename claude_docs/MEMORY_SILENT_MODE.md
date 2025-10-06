# Memory Silent Mode - Suppress Memory Warnings

**Created**: 2025-10-06  
**Purpose**: Suppress memory warning messages during training while maintaining monitoring

---

## Overview

Added `silent_mode` flag to hide memory warnings and cleanup messages during training. The memory monitor still tracks and manages memory, it just doesn't print warnings to the console.

## Problem

During training, memory warnings were appearing frequently:
```
Memory CRITICAL: 96.0% used, recommend batch_size=4
INFO:src.Ava.training.memory_monitor:Memory cleanup: freed 0.23GB GPU memory
```

These warnings are informative but can clutter the training output, especially when running in production or when memory usage is intentionally high.

## Solution

### 1. Added `silent_mode` Parameter

**File**: `src/Ava/training/memory_monitor.py`

```python
class MemoryMonitor:
    def __init__(
        self,
        # ... other params ...
        silent_mode: bool = False  # NEW: Suppress memory warnings
    ):
```

### 2. Conditional Logging

All memory warnings now respect `silent_mode`:

```python
# GPU info at init
if not self.silent_mode:
    logger.info(f"GPU {i}: {total_memory:.1f}GB total...")
else:
    logger.debug(f"GPU {i}: {total_memory:.1f}GB total...")

# Memory cleanup messages
if not self.silent_mode:
    logger.info(f"Memory cleanup: freed {freed_gb:.2f}GB GPU memory")
else:
    logger.debug(f"Memory cleanup: freed {freed_gb:.2f}GB GPU memory")
```

### 3. Enhanced Trainer Integration

**File**: `src/Ava/training/enhanced_trainer.py`

```python
# Check for silent_mode in config
memory_silent = False
if hasattr(config, 'memory') and config.memory:
    if hasattr(config.memory, 'silent_mode'):
        memory_silent = config.memory.silent_mode
    elif isinstance(config.memory, dict) and 'silent_mode' in config.memory:
        memory_silent = config.memory['silent_mode']

self.memory_monitor = MemoryMonitor(
    # ... other params ...
    silent_mode=memory_silent,
)
```

Console warnings also respect silent mode:
```python
if not self.memory_monitor.silent_mode:
    print(f"Memory {status.upper()}: {utilization:.1%} used...")
```

### 4. Configuration Option

**File**: `configs/gpu/small.yaml`

```yaml
memory:
  enable_memory_pool: true
  pool_size_gb: 8.0
  gradient_checkpointing: false
  memory_threshold_gb: 8.0
  enable_cpu_offload: false
  clear_cache_frequency: 2000
  silent_mode: true  # NEW: Suppress memory warnings
```

## Usage

### Enable Silent Mode (Default in small.yaml)

```yaml
memory:
  silent_mode: true
```

### Disable Silent Mode (Show All Warnings)

```yaml
memory:
  silent_mode: false
```

### Command Line Override

```bash
# In train.py, you can override via config
python train.py --config configs/gpu/small.yaml
# (uses silent_mode: true from config)
```

## Behavior

| silent_mode | Console Output | Log Level | Memory Monitoring |
|-------------|----------------|-----------|-------------------|
| `true` | ✅ Clean (no warnings) | DEBUG | ✅ Active |
| `false` | ⚠️ Shows warnings | INFO | ✅ Active |

### What's Still Shown (Even with silent_mode=true)

- Emergency memory situations (potential OOM)
- Critical errors requiring intervention
- Training step progress
- Loss metrics

### What's Hidden (with silent_mode=true)

- "Memory CRITICAL: X% used" warnings
- "Memory cleanup: freed XGB" messages
- GPU memory stats at initialization
- Non-critical memory recommendations

## Benefits

1. **Cleaner Output**: Focus on training metrics, not memory warnings
2. **Production Ready**: Reduce log noise in production environments
3. **Still Safe**: Memory monitoring and cleanup still active
4. **Configurable**: Easy to enable warnings when debugging

## Files Modified

1. `src/Ava/training/memory_monitor.py` (+15 lines)
   - Added `silent_mode` parameter
   - Conditional logging (INFO → DEBUG when silent)

2. `src/Ava/training/enhanced_trainer.py` (+10 lines)
   - Read `silent_mode` from config
   - Pass to MemoryMonitor
   - Suppress console warnings

3. `configs/gpu/small.yaml` (+1 line)
   - Added `silent_mode: true`

## Testing

```bash
# Test with silent mode enabled (default)
python scripts/training/train.py --config configs/gpu/small.yaml

# Expected: No memory warnings in console
# Memory monitoring still active in background

# Test with silent mode disabled
# Edit config: silent_mode: false
python scripts/training/train.py --config configs/gpu/small.yaml

# Expected: Memory warnings appear as before
```

## Related

- [Memory Monitor](../code/src/Ava/training/memory_monitor.py)
- [Enhanced Trainer](../code/src/Ava/training/enhanced_trainer.py)
- [Small Config](../code/configs/gpu/small.yaml)

---

**Status**: ✅ Implemented and tested  
**Version**: 2025-10-06
