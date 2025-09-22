# DeepSpeed Configuration Files

This directory contains DeepSpeed configuration files for optimized training with ZeRO (Zero Redundancy Optimizer).

## Current Configuration

All configurations are now set to use **ZeRO Stage 3** for maximum memory efficiency.

### Files

- **`ds_config.json`** - Default configuration with ZeRO-3 optimization
- **`ds_config_zero3.json`** - Alternative ZeRO-3 configuration (identical to default)

## ZeRO-3 Benefits

ZeRO-3 provides the most aggressive memory optimization by partitioning:
- ✅ **Parameters** - Model parameters are partitioned across GPUs/CPU
- ✅ **Gradients** - Gradients are partitioned and reduced efficiently
- ✅ **Optimizer States** - Adam momentum and variance are partitioned

This results in:
- **75-90% memory reduction** compared to standard training
- **8-10x larger models** on the same hardware
- **CPU offloading** for even more memory savings

## Key Settings

```json
"zero_optimization": {
    "stage": 3,                    // ZeRO-3 full partitioning
    "offload_optimizer": {
        "device": "cpu",           // Offload optimizer to CPU
        "pin_memory": true         // Pin CPU memory for faster transfers
    },
    "offload_param": {
        "device": "cpu",           // Offload parameters to CPU
        "pin_memory": true
    },
    "round_robin_gradients": true  // Better gradient distribution
}
```

## Usage

### With Launch Script
```bash
./train_deepspeed.sh <config_file> <deepspeed_config> <batch_size> <num_gpus>
```

### Direct DeepSpeed Command
```bash
deepspeed train.py --config <yaml_config> --use-deepspeed --deepspeed-config ds_config.json
```

## Memory Optimization Tips

1. **Batch Size**: Start with batch_size=1 and increase gradually
2. **Gradient Accumulation**: Use higher accumulation steps for effective larger batches
3. **CPU Offloading**: Enabled by default for maximum memory savings
4. **Activation Checkpointing**: Already enabled in most configs

## Troubleshooting

If you encounter OOM errors even with ZeRO-3:
1. Reduce batch size to 1
2. Increase gradient accumulation steps
3. Enable gradient checkpointing in your YAML config
4. Use the `ultra_tiny.yaml` configuration for testing