# Ava Training Logs Reference Guide

This guide explains the color-coded logging system used in Ava training pipelines. Understanding these logs helps you quickly identify training status, issues, and important events.

## Quick Color Reference

| Color | Meaning | Example |
|-------|---------|---------|
| **Cyan** | Information, headers, steps | Phase headers, epoch numbers |
| **Green** | Success, positive metrics | Checkpoints saved, training complete |
| **Yellow** | Warnings, caution | Memory warnings, deprecations |
| **Red** | Errors, failures | OOM errors, training failures |
| **Purple** | Configuration, phases | Phase headers, calibration |
| **Orange** | Data, memory metrics | Dataset info, GPU memory usage |
| **Gold** | Loss values | Training loss, validation loss |
| **Magenta** | GPU/Model info | Device info, parameter counts |
| **Light Blue** | Step progress | Step 100/1000, epoch progress |
| **Lime/Green** | Values, counts | Parameter counts, sample counts |

---

## Log Message Types

### Phase Headers (Purple)

```
══════════════════════════════════════════════════════════════════════
  Phase 11: Training Loop
══════════════════════════════════════════════════════════════════════
```

**Meaning**: Major training phases. Each phase represents a distinct stage of the pipeline:
- **Phase 1**: Distributed Setup - Multi-GPU initialization
- **Phase 2**: Configuration Loading - YAML parsing
- **Phase 3**: Run Manager & Checkpointing - Output directory setup
- **Phase 4**: Training Context - Shared state initialization
- **Phase 5**: Pipeline Registration - Component setup
- **Phase 6**: Model Building - Architecture creation
- **Phase 7**: Optimizer & Scheduler - Training optimization
- **Phase 8**: Data Loading - Dataset preparation
- **Phase 9**: Metrics Setup - WandB, logging
- **Phase 10**: Component Initialization - Validation, generation
- **Phase 11**: Training Loop - Main training
- **Phase 12**: Finalization - Cleanup, summary

---

### Section Headers (Cyan)

```
────────────────────────────────────────────────────────────────────
[*] Model Optimizations
────────────────────────────────────────────────────────────────────
```

**Meaning**: Subsections within a phase. Shows what's being configured or processed.

---

### Success Messages (Green)

```
[OK] Calibration complete: optimal batch size = 64
[OK] Model loaded successfully
[OK] Checkpoint saved
```

**Meaning**: Operations completed successfully. Green = good!

**Common success messages**:
- `[OK] Calibration complete` - Batch size auto-tuning finished
- `[OK] Checkpoint saved` - Model state persisted to disk
- `[OK] Resumed from epoch X` - Successfully loaded checkpoint
- `Best validation loss: X.XXXX` - New best model found

---

### Warning Messages (Yellow)

```
[WARN] Mixed VRAM detected across 4 GPUs: [24.0, 24.0, 12.0, 12.0] GB
[WARN] Triton not available, using PyTorch fallbacks
[WARN] High memory usage: 87.3%
```

**Meaning**: Non-fatal issues that may affect performance. Pay attention but training continues.

**Common warnings**:
- `Mixed VRAM detected` - GPUs have different memory, batch size uses minimum
- `Triton not available` - Fused kernels disabled, slower training
- `High memory usage` - Risk of OOM, consider smaller batch size
- `Failed to load tokenizer` - Generation features disabled

---

### Error Messages (Red)

```
[ERR] Training failed: CUDA out of memory
[ERR] Invalid configuration: missing required field 'model.hidden_size'
[X] Checkpoint load failed
```

**Meaning**: Fatal errors that stop training. Immediate attention required.

**Common errors**:
- `CUDA out of memory` - Reduce batch size or enable gradient checkpointing
- `Invalid configuration` - Fix YAML config file
- `Data loading failed` - Check data paths and file formats

---

### Configuration Values (Lime/Cyan)

```
  * Config: code/configs/moe/large.yaml
  * Device: cuda:0
  * Epochs: 5
  * Batch size: 64 (auto-calibrated)
  * Learning rate: 6.00e-04
```

**Meaning**: Current configuration values. Key training parameters.

**Key configuration items**:
- `Device` - Which GPU(s) are being used
- `World size` - Number of GPUs (1 = single GPU)
- `Batch size` - Samples per step (may be auto-calibrated)
- `Learning rate` - Optimizer step size

---

### Epoch Summary (Cyan/Gold)

```
──────────────────────────────────────────────────
  Epoch 1/5 Complete
  Train Loss: 2.3456
  Val Loss:   2.1234   (green if lower than train loss)
  Duration:   12m 34s
──────────────────────────────────────────────────
```

**Meaning**: End-of-epoch summary with key metrics.

**Color coding**:
- **Gold**: Loss values
- **Green**: Validation loss (if lower than train = model is generalizing)
- **Yellow**: Validation loss (if higher than train = possible overfitting)
- **Gray**: Duration

---

### Calibration Messages (Purple)

```
[*] Running batch size calibration with REAL OPTIMIZER...
  Batch Size: 64
  Memory:     68.5%
```

**Meaning**: Auto-tuning batch size to maximize GPU utilization.

**What calibration does**:
1. Tests increasing batch sizes
2. Measures GPU memory usage
3. Finds largest batch that fits safely (target: 70-85% memory)
4. Accounts for optimizer states and gradient accumulation

---

### Optimization Status (Green/Gray)

```
[FAST] Model Optimizations
  use_flash_attention:   YES (40% mem savings, 2-4x faster)
  gradient_checkpointing: YES (70-80% mem savings)
  use_triton_kernels:    NO
```

**Meaning**: Which optimizations are enabled/disabled.

**Color coding**:
- **Green + YES**: Optimization enabled with benefit note
- **Gray + NO**: Optimization disabled

**Key optimizations to watch**:
- `use_flash_attention` - Critical for long sequences
- `gradient_checkpointing` - Enables larger models
- `mixed_precision` - bf16/fp16 reduces memory by 50%
- `use_torch_compile` - 15-25% speedup after warmup

---

### GPU Information (Magenta/Orange)

```
[GPU] GPU 0: NVIDIA A100 80GB
  Total: 80.0GB
  Free:  72.3GB
```

**Meaning**: GPU details during selection/initialization.

**What to look for**:
- Sufficient free memory for your model
- Multi-GPU: Check all GPUs have similar VRAM
- Compute capability: 8.0+ needed for bf16, 9.0+ for fp8

---

### Data Loading (Orange)

```
[DATA] Dataset: fine-tuning/train.arrow
  Samples:    1,234,567
  Batch Size: 64
  Workers:    4
```

**Meaning**: Dataset and DataLoader configuration.

**Key metrics**:
- `Samples` - Total training examples
- `Batch Size` - Effective batch size per step
- `Workers` - Parallel data loading threads

---

### Training Progress (Light Blue)

```
Step 100/10000 | Loss: 2.3456 | LR: 6.00e-04 | Mem: 45.2GB | 1250 tok/s
```

**Meaning**: Per-step training progress (when verbose logging enabled).

**Metrics explained**:
- **Step X/Y**: Current step / total steps
- **Loss**: Cross-entropy loss (lower = better)
- **LR**: Current learning rate
- **Mem**: GPU memory used
- **tok/s**: Training throughput

---

## Log Files

Training logs are saved in multiple formats:

### Console Output (colored)
Real-time colored output to terminal with tqdm progress bars.

### `training.log` (plain text)
Full log with timestamps in the run directory:
```
[2024-01-15 14:30:22] INFO - Epoch 1/5 - Train Loss: 2.3456
```

### WandB Logs (if enabled)
Metrics logged to Weights & Biases dashboard.

---

## Common Patterns to Watch

### Healthy Training
```
[OK] Calibration complete: optimal batch size = 64
Step 100/1000 | Loss: 3.2145 | ...
Step 200/1000 | Loss: 2.8763 | ...  ← Loss decreasing = good!
Step 300/1000 | Loss: 2.5412 | ...
[OK] New best validation loss: 2.4521
```

### Memory Issues
```
[WARN] High memory usage: 89.2%
[WARN] Reducing batch size: 64 -> 48
[ERR] CUDA out of memory  ← If this appears, need smaller batch
```

### Overfitting
```
Epoch 3/5 Complete
  Train Loss: 1.2345
  Val Loss:   1.8765   ← Increasing while train decreases = overfitting
```

---

## Customizing Log Verbosity

In your YAML config:

```yaml
training:
  # 'tqdm' = progress bar only (clean)
  # 'verbose' = progress bar + detailed logs
  log_mode: 'tqdm'

  # Steps between verbose log messages
  verbose_log_interval: 500

  logging:
    # Steps between metric logging
    logging_steps: 100
```

---

## See Also

- [02_TRAINING_GUIDE.md](02_TRAINING_GUIDE.md) - Training procedures
- [07_CONFIGURATION_SYSTEM.md](07_CONFIGURATION_SYSTEM.md) - Config reference
- [03_MEMORY_OPTIMIZATION.md](03_MEMORY_OPTIMIZATION.md) - Memory tuning
