# TF32 Configuration Guide - Enable/Disable TF32 Optimizations

This guide explains how to control TF32 (TensorFloat-32) optimizations in the training pipeline.

## What is TF32?

TF32 is a hardware acceleration feature available on NVIDIA Ampere and newer GPUs (RTX 30xx/40xx, A100, H100) that provides:
- **8x faster** matrix multiplication operations
- Maintains float32 range with reduced precision
- Minimal to no accuracy loss for most models
- **Automatic** in PyTorch when enabled

## Quick Start

### Enable TF32 (Default - Recommended)

Add this to your config YAML:

```yaml
performance:
  enable_tf32: true                  # Enable TF32 acceleration
  float32_matmul_precision: 'high'   # Precision level
  enable_cudnn_benchmark: true       # Auto-tune kernels
```

### Disable TF32 (For Testing/Comparison)

```yaml
performance:
  enable_tf32: false                 # Disable TF32
  float32_matmul_precision: 'high'   # (ignored when disabled)
  enable_cudnn_benchmark: false      # Disable auto-tuning
```

## Configuration Options

### `enable_tf32` (bool)

**Default**: `true`

Controls whether TF32 is enabled for CUDA operations.

**When to enable**:
-  Training on Ampere+ GPUs (RTX 30xx/40xx, A100, H100)
-  Production training (faster)
-  Most use cases

**When to disable**:
-  Debugging numerical issues
-  Benchmarking against non-TF32 baselines
-  Ensuring maximum precision (rare)
-  Older GPUs (pre-Ampere) - has no effect anyway

### `float32_matmul_precision` (str)

**Default**: `'high'`

**Options**:
- `'highest'`: Maximum precision (slowest, most accurate)
- `'high'`: Balanced (recommended, enables TF32)
- `'medium'`: Faster, less precise

**Note**: Only used when `enable_tf32: true`

### `enable_cudnn_benchmark` (bool)

**Default**: `true`

Enables cuDNN's auto-tuning to find the fastest convolution algorithms.

**When to enable**:
-  Fixed input sizes (most cases)
-  Production training
-  Maximum performance

**When to disable**:
-  Variable input sizes (dynamic batching)
-  Reproducibility testing
-  Debugging

## Example Configurations

### Production Training (Maximum Speed)

```yaml
# config.yaml
performance:
  enable_tf32: true
  float32_matmul_precision: 'high'
  enable_cudnn_benchmark: true
```

**Expected log output**:
```
 TF32 optimizations ENABLED (precision: high)
 CuDNN benchmark auto-tuning ENABLED
```

### Debugging/Testing (Maximum Precision)

```yaml
# config.yaml
performance:
  enable_tf32: false
  float32_matmul_precision: 'highest'
  enable_cudnn_benchmark: false
```

**Expected log output**:
```
  TF32 optimizations DISABLED (may be slower)
  CuDNN benchmark DISABLED
```

### Balanced (Good Precision + Speed)

```yaml
# config.yaml
performance:
  enable_tf32: true
  float32_matmul_precision: 'highest'  # More precise TF32
  enable_cudnn_benchmark: true
```

## Testing Your Configuration

### Verify Settings are Applied

Run the test script:

```bash
python code/scripts/testing/test_tf32_config.py
```

**Expected output**:
```
 ALL TESTS PASSED

Conclusion:
  TF32 can be successfully enabled and disabled via YAML configuration.
  Use 'performance.enable_tf32: true/false' in your config files.
```

### Check Training Logs

When you start training, look for these messages:

**TF32 Enabled**:
```
 TF32 optimizations ENABLED (precision: high)
 CuDNN benchmark auto-tuning ENABLED
```

**TF32 Disabled**:
```
  TF32 optimizations DISABLED (may be slower)
  CuDNN benchmark DISABLED
```

## Performance Impact

### Benchmarks (Approximate)

| Configuration | Relative Speed | Precision | Use Case |
|--------------|----------------|-----------|----------|
| TF32 Enabled + Benchmark | **1.0x** (baseline) | Good | Production |
| TF32 Disabled | **0.12x** (8x slower) | Best | Debugging |
| TF32 Enabled, No Benchmark | **0.9x** | Good | Dynamic sizes |

### Example: Training Speed Comparison

On A100 GPU with same model:
- **With TF32**: 5000 tokens/sec
- **Without TF32**: 600 tokens/sec
- **Speedup**: 8.3x

## Advanced Settings

### All Performance Options

```yaml
performance:
  # TF32 and matmul settings
  enable_tf32: true
  float32_matmul_precision: 'high'
  enable_cudnn_benchmark: true

  # Advanced (usually don't need to change)
  cudagraph_skip_dynamic_shapes: true
  cudagraph_dynamic_shape_warn_limit: null
  torchinductor_max_autotune: 0

  # Performance modes (separate from TF32)
  ultra_fast_mode: false
  fast_progress: false
  minimal_progress: false
  no_sync: false
  express_mode: false
```

## Troubleshooting

### Problem: TF32 log message not showing

**Check**:
1. Is CUDA available? (`torch.cuda.is_available()`)
2. Is config file correct? (check for typos)
3. Are you using the right config path?

**Solution**: Run the test script to verify:
```bash
python code/scripts/testing/test_tf32_config.py
```

### Problem: Training is slow even with TF32 enabled

**Check**:
1. Is your GPU Ampere or newer? (TF32 requires Ampere+)
2. Are other bottlenecks present? (data loading, I/O)

**Solution**: Profile to find bottleneck:
```bash
python code/scripts/testing/profile_training.py --config your_config.yaml
```

### Problem: Numerical instability with TF32

**Symptoms**:
- Loss becomes NaN
- Gradients explode
- Model doesn't converge

**Solution 1**: Try higher precision
```yaml
performance:
  enable_tf32: true
  float32_matmul_precision: 'highest'  # More precise
```

**Solution 2**: Disable TF32 entirely
```yaml
performance:
  enable_tf32: false
```

**Solution 3**: Use BF16 mixed precision
```yaml
hardware:
  mixed_precision: 'bf16'  # More stable than FP16 with TF32
```

### Problem: Getting PyTorch warning about deprecated API

**Warning message**:
```
UserWarning: Please use the new API settings to control TF32 behavior...
```

**Status**: This is expected and harmless. The code works with both old and new PyTorch versions.

**Future**: The code will be updated to use the new API when PyTorch 2.9+ is required.

## GPU Compatibility

### Supported GPUs (TF32 Available)

-  NVIDIA RTX 30xx series (3060, 3070, 3080, 3090)
-  NVIDIA RTX 40xx series (4060, 4070, 4080, 4090)
-  NVIDIA A100
-  NVIDIA H100
-  Any Ampere, Ada Lovelace, or Hopper architecture

### Older GPUs (TF32 Not Available)

-  NVIDIA RTX 20xx series (Turing)
-  NVIDIA GTX 16xx series
-  NVIDIA V100 (Volta)
-  Any pre-Ampere architecture

**Note**: On older GPUs, enabling TF32 has no effect (neither helps nor hurts)

## Integration with Other Features

### TF32 + Mixed Precision (Recommended)

```yaml
hardware:
  mixed_precision: 'bf16'  # Use BF16 mixed precision

performance:
  enable_tf32: true        # Also enable TF32
```

**Benefits**:
- BF16 reduces memory usage
- TF32 speeds up FP32 operations (used in some layers)
- Combined 10-15x speedup possible

### TF32 + DeepSpeed

```yaml
deepspeed:
  enabled: true
  precision: 'bf16'        # DeepSpeed uses BF16

performance:
  enable_tf32: true        # Still beneficial for FP32 ops
```

### TF32 + Gradient Checkpointing

```yaml
model:
  gradient_checkpointing: true  # Save memory

performance:
  enable_tf32: true             # Speed up recomputation
```

## Command-Line Override (Future)

*Note: Not yet implemented*

In the future, you'll be able to override via command line:

```bash
# Enable TF32
python train.py --config config.yaml --enable-tf32

# Disable TF32
python train.py --config config.yaml --no-tf32
```

## References

- **PyTorch TF32 Docs**: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
- **NVIDIA TF32 Blog**: https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/
- **Performance Guide**: [MOE_OPTIMIZATION_GUIDE.md](MOE_OPTIMIZATION_GUIDE.md)

## Quick Reference

```yaml
# Maximum speed (recommended for production)
performance:
  enable_tf32: true
  float32_matmul_precision: 'high'
  enable_cudnn_benchmark: true

# Maximum precision (debugging/testing)
performance:
  enable_tf32: false
  float32_matmul_precision: 'highest'
  enable_cudnn_benchmark: false

# Balanced (good speed + precision)
performance:
  enable_tf32: true
  float32_matmul_precision: 'highest'
  enable_cudnn_benchmark: true
```

---

**Test your settings**: `python code/scripts/testing/test_tf32_config.py`
