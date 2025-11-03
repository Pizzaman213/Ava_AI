# Ava Memory Management Module

Comprehensive memory monitoring and optimization system for efficient LLM training.

## Overview

The Ava Memory Management module provides real-time GPU/CPU memory monitoring, proactive OOM prevention, and detailed memory profiling for training large language models. It enables **2-3x memory reduction** through intelligent optimization tracking and management.

## Features

### Core Capabilities

- **Real-time Memory Monitoring**: Track GPU/CPU memory usage with NVML support
- **Proactive OOM Prevention**: Automatic batch size reduction and emergency cleanup
- **Memory Optimization Tracking**: Monitor enabled optimizations (gradient checkpointing, flash attention, etc.)
- **Detailed Profiling**: Memory breakdowns by component (parameters, gradients, activations, etc.)
- **Activation Memory Estimation**: Calculate transformer model memory requirements
- **Memory Leak Detection**: Identify and track memory leaks during training
- **Fragmentation Analysis**: Monitor memory fragmentation and wasted space

### Memory Optimizations Supported

| Optimization | Memory Savings | Tracked |
|--------------|----------------|---------|
| Gradient Checkpointing | 60-80% | ✅ |
| Flash Attention | 50-70% | ✅ |
| Mixed Precision (FP16/BF16) | 50% | ✅ |
| DeepSpeed ZeRO | 1.5x - 4x | ✅ |
| Activation Checkpointing | 30-50% | ✅ |

## Installation

The module is part of the Ava framework and requires:

```bash
pip install torch>=2.0.0
pip install pynvml  # For GPU monitoring
pip install psutil  # For CPU monitoring
pip install numpy
```

## Quick Start

### Basic Usage

```python
from Ava.memory_management import MemoryMonitor

# Initialize monitor
monitor = MemoryMonitor(
    target_utilization=0.90,
    warning_threshold=0.92,
    critical_threshold=0.95,
    emergency_threshold=0.97,
    silent_mode=False
)

# Check memory health during training
health = monitor.check_memory_health(batch_size=12)

print(f"Status: {health['status']}")  # normal, warning, critical, emergency
print(f"GPU Usage: {health['gpu_utilization']:.1%}")
print(f"Recommendations: {health['recommendations']}")

# Cleanup if needed
if health['status'] in ['critical', 'emergency']:
    cleanup_stats = monitor.cleanup_memory(aggressive=True)
    print(f"Freed: {cleanup_stats['freed_gb']:.2f} GB")
```

### Advanced Features

#### 1. Detailed Memory Breakdown

```python
# Get comprehensive memory statistics
breakdown = monitor.get_detailed_memory_breakdown()

print(f"Allocated: {breakdown['total']['allocated_gb']:.2f} GB")
print(f"Reserved: {breakdown['total']['reserved_gb']:.2f} GB")
print(f"Fragmentation: {breakdown['fragmentation']['ratio']:.1%}")
print(f"Wasted: {breakdown['fragmentation']['wasted_gb']:.2f} GB")
```

#### 2. Activation Memory Estimation

```python
# Estimate memory for your transformer model
estimates = monitor.estimate_activation_memory(
    batch_size=12,
    sequence_length=256,
    hidden_size=512,
    num_layers=14,
    num_attention_heads=8,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

print(f"Effective Memory: {estimates['effective_memory_gb']:.2f} GB")
print(f"Savings from Checkpointing: {estimates['savings_gb']:.2f} GB")
print(f"Attention Type: {estimates['attention_type']}")

# Get optimization recommendations
if 'potential_checkpointing_savings_gb' in estimates:
    print(f"Enable checkpointing to save {estimates['potential_checkpointing_savings_gb']:.2f} GB")
if 'potential_flash_attention_savings_gb' in estimates:
    print(f"Enable flash attention to save {estimates['potential_flash_attention_savings_gb']:.2f} GB")
```

#### 3. Track Enabled Optimizations

```python
# Check which optimizations are enabled
config = load_yaml_config('config.yaml')
optimizations = monitor.track_memory_optimizations(config)

for opt_name, opt_info in optimizations.items():
    status = "✅ Enabled" if opt_info['enabled'] else "❌ Disabled"
    savings = opt_info['memory_savings']
    print(f"{opt_name}: {status} - saves {savings}")

# Example output:
# gradient_checkpointing: ✅ Enabled - saves 60-80%
# flash_attention: ✅ Enabled - saves 50-70%
# mixed_precision: ✅ Enabled (fp16) - saves 50%
# deepspeed_zero: ✅ Enabled (Stage 1) - saves 1.5x
# estimated_combined_savings: 80%
```

## API Reference

### MemoryMonitor Class

#### Constructor

```python
MemoryMonitor(
    target_utilization: float = 0.85,
    warning_threshold: float = 0.99,
    critical_threshold: float = 0.99,
    emergency_threshold: float = 0.99,
    history_size: int = 100,
    memory_headroom_gb: float = 1.0,
    silent_mode: bool = False
)
```

**Parameters**:
- `target_utilization`: Target GPU memory utilization (0.85 = 85%)
- `warning_threshold`: Threshold to trigger warnings (0.92 = 92%)
- `critical_threshold`: Threshold for batch size reduction (0.95 = 95%)
- `emergency_threshold`: Threshold for aggressive cleanup (0.97 = 97%)
- `history_size`: Number of memory measurements to keep
- `memory_headroom_gb`: GB of memory to reserve for safety
- `silent_mode`: If True, suppress warning messages

#### Main Methods

##### check_memory_health()

```python
health = monitor.check_memory_health(
    current_batch_size: int,
    skip_sync: bool = False
) -> Dict[str, Any]
```

Returns comprehensive memory health status with recommendations.

**Returns**:
```python
{
    'status': 'normal' | 'warning' | 'critical' | 'emergency',
    'gpu_utilization': 0.85,  # 85%
    'cpu_utilization': 0.60,  # 60%
    'gpu_memory_mb': 20480,
    'available_memory_mb': 3616,
    'recommendations': ['Enable gradient checkpointing', ...],
    'should_reduce_batch': False,
    'suggested_batch_size': 12
}
```

##### cleanup_memory()

```python
cleanup_stats = monitor.cleanup_memory(
    aggressive: bool = False
) -> Dict[str, float]
```

Perform memory cleanup and return statistics.

**Returns**:
```python
{
    'freed_gb': 2.5,
    'before_gb': 22.0,
    'after_gb': 19.5
}
```

##### get_detailed_memory_breakdown()

```python
breakdown = monitor.get_detailed_memory_breakdown() -> Dict[str, Any]
```

Get comprehensive memory breakdown by component.

**Returns**:
```python
{
    'total': {
        'allocated_gb': 18.5,
        'reserved_gb': 20.0,
        'max_allocated_gb': 22.0,
        'free_reserved_gb': 1.5
    },
    'memory_pools': {
        'active_gb': 17.0,
        'inactive_gb': 1.5
    },
    'fragmentation': {
        'ratio': 0.075,  # 7.5% fragmentation
        'wasted_gb': 1.5
    },
    'allocations': {
        'num_alloc': 0,
        'num_ooms': 0
    }
}
```

##### estimate_activation_memory()

```python
estimates = monitor.estimate_activation_memory(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    use_gradient_checkpointing: bool = False,
    use_flash_attention: bool = False
) -> Dict[str, float]
```

Estimate activation memory for transformer models.

**Returns**:
```python
{
    'attention_type': 'flash' | 'standard',
    'attention_memory_gb': 2.5,
    'ffn_memory_gb': 3.0,
    'total_without_checkpointing_gb': 5.5,
    'effective_memory_gb': 1.1,  # With checkpointing
    'checkpointing_enabled': True,
    'savings_gb': 4.4,
    'potential_flash_attention_savings_gb': 2.0  # If not using flash
}
```

##### track_memory_optimizations()

```python
optimizations = monitor.track_memory_optimizations(
    config: Dict[str, Any]
) -> Dict[str, Any]
```

Track which memory optimizations are enabled in the config.

**Returns**:
```python
{
    'gradient_checkpointing': {
        'enabled': True,
        'memory_savings': '60-80%'
    },
    'flash_attention': {
        'enabled': True,
        'memory_savings': '50-70%'
    },
    'mixed_precision': {
        'enabled': True,
        'type': 'fp16',
        'memory_savings': '50%'
    },
    'deepspeed_zero': {
        'enabled': True,
        'stage': 1,
        'memory_savings': 'Stage 1: 1.5x'
    },
    'activation_checkpointing': {
        'enabled': True,
        'memory_savings': '30-50%'
    },
    'estimated_combined_savings': '80%'
}
```

## Integration with Training

### In Enhanced Trainer

The MemoryMonitor is automatically integrated into the EnhancedTrainer:

```python
from Ava.training.enhanced_trainer import EnhancedTrainer

trainer = EnhancedTrainer(config)
# Memory monitor is automatically initialized based on config

# Access during training
memory_health = trainer.memory_monitor.check_memory_health(batch_size)
```

### Configuration

Configure memory monitoring in your `config.yaml`:

```yaml
training:
  memory:
    enable_memory_pool: true
    pool_size_gb: 24
    memory_threshold_gb: 24
    enable_cpu_offload: false
    clear_cache_frequency: 50
    silent_mode: true
    target_utilization: 0.90
    warning_threshold: 0.92
    critical_threshold: 0.95
    emergency_threshold: 0.97
```

## Best Practices

### 1. Memory Threshold Configuration

- **Aggressive** (90%+ target): Use for stable, well-tested models
- **Balanced** (85-90% target): Default for most use cases
- **Conservative** (75-85% target): Use for experimental models or unstable training

### 2. Monitoring Frequency

- Check memory health every 10-50 steps (configurable)
- Use `skip_sync=True` for lightweight checks between full checks
- Enable `silent_mode` to reduce console spam during stable training

### 3. Emergency Handling

```python
# Set up emergency callback
def on_memory_emergency(health):
    print(f"⚠️ Memory emergency at {health['gpu_utilization']:.1%}")
    # Save checkpoint before potential OOM
    trainer.save_checkpoint('emergency')
    # Aggressive cleanup
    cleanup_stats = monitor.cleanup_memory(aggressive=True)
    print(f"Freed {cleanup_stats['freed_gb']:.2f} GB")

# Use in training loop
if memory_health['status'] == 'emergency':
    on_memory_emergency(memory_health)
```

### 4. Memory Profiling

Profile your model before training:

```python
# Estimate memory requirements
estimates = monitor.estimate_activation_memory(
    batch_size=12,
    sequence_length=256,
    hidden_size=512,
    num_layers=14,
    num_attention_heads=8,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

print(f"Estimated activation memory: {estimates['effective_memory_gb']:.2f} GB")

# Add model parameters (rough estimate)
num_params = 100_000_000  # 100M parameters
param_memory_gb = (num_params * 2) / (1024**3)  # FP16
gradient_memory_gb = param_memory_gb  # Same size
optimizer_memory_gb = param_memory_gb * 2  # AdamW (momentum + variance)

total_estimated = (
    estimates['effective_memory_gb'] +
    param_memory_gb +
    gradient_memory_gb +
    optimizer_memory_gb
)

print(f"Total estimated memory: {total_estimated:.2f} GB")
```

## Troubleshooting

### OOM Despite Memory Monitoring

If you still encounter OOM errors:

1. **Enable all optimizations**:
   - Gradient checkpointing ✅
   - Flash attention ✅
   - Mixed precision ✅
   - Activation checkpointing ✅

2. **Reduce batch size incrementally**: 12 → 8 → 6 → 4 → 2

3. **Increase gradient accumulation** to maintain effective batch size

4. **Enable DeepSpeed ZeRO Stage 2 or 3** for larger models

5. **Check for memory leaks**:
   ```python
   breakdown = monitor.get_detailed_memory_breakdown()
   print(f"Fragmentation: {breakdown['fragmentation']['ratio']:.1%}")
   ```

### High Memory Fragmentation

If fragmentation > 15%:

```python
# Force full cache clear
torch.cuda.empty_cache()
gc.collect()

# Or restart training from checkpoint
```

### Slow Training After Optimizations

Gradient checkpointing adds 10-20% overhead. Compensate by:
- Using larger batch sizes (enabled by freed memory)
- Enabling flash attention (2-4x speedup)
- Using mixed precision (2-3x speedup)

**Net result**: Usually 1.8-2.7x faster overall throughput!

## Examples

See the complete memory optimization guide at [`docs/03_MEMORY_OPTIMIZATION.md`](../../docs/03_MEMORY_OPTIMIZATION.md) for comprehensive examples and configuration templates.

## Version History

### v1.0.0 (2025-11-03)
- Initial release with comprehensive memory management
- Added detailed memory breakdown
- Added activation memory estimation
- Added optimization tracking
- Moved to dedicated Ava_Memory_Management module

## License

Part of the Ava MoE++ Training Framework.

## Support

For issues, questions, or contributions, please refer to the main Ava repository.
