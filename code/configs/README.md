# MoE++ Configuration Files

This directory contains configuration files for training MoE++ models, organized by hardware type.

## Directory Structure

```
configs/
├── cpu/          # CPU-optimized configurations
│   ├── ultra_tiny.yaml # 3-layer, 128 hidden - rapid prototyping (~100k params)
│   ├── tiny.yaml       # 6-layer, 384 hidden - fast testing
│   ├── small.yaml      # 12-layer, 768 hidden - development
│   └── medium.yaml     # 16-layer, 1024 hidden - production
│
├── mps/          # Apple Silicon GPU (Metal Performance Shaders)
│   ├── ultra_tiny.yaml # 3-layer, 128 hidden - rapid prototyping (~100k params)
│   ├── tiny.yaml       # 6-layer, 384 hidden - M1/M2/M3
│   ├── small.yaml      # 12-layer, 768 hidden - M1/M2/M3
│   ├── medium.yaml     # 24-layer, 1024 hidden - M1/M2/M3 Pro/Max
│   └── large.yaml      # 36-layer, 1536 hidden - M1/M2/M3 Max/Ultra
│
└── gpu/          # NVIDIA/AMD GPU configurations
    ├── ultra_tiny.yaml # 3-layer, 128 hidden - rapid prototyping (~100k params)
    ├── small.yaml      # 12-layer, 768 hidden - single GPU
    ├── medium.yaml     # 24-layer, 1024 hidden - 1-4 GPUs
    ├── large.yaml      # 36-layer, 1536 hidden - 4-8 GPUs
    └── xlarge.yaml     # 48-layer, 2048 hidden - 8+ GPUs
```

## Usage Examples

### CPU Training (macOS)
```bash
# Rapid prototyping with ultra-tiny model (~100k params)
python3 scripts/train.py --config configs/cpu/ultra_tiny.yaml

# Fast testing with tiny model
python3 scripts/train.py --config configs/cpu/tiny.yaml

# Development with small model
python3 scripts/train.py --config configs/cpu/small.yaml

# macOS CPU-optimized training with threading
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export VECLIB_MAXIMUM_THREADS=$(sysctl -n hw.ncpu)
python3 scripts/train.py --config configs/cpu/small.yaml

# For Apple Silicon (M1/M2/M3) with MPS acceleration
python3 scripts/train.py --config configs/cpu/small.yaml --device mps
```

### GPU Training
```bash
# Single GPU training
python3 scripts/train.py --config configs/gpu/small.yaml

# Multi-GPU training
torchrun --nproc_per_node=4 scripts/train.py --config configs/gpu/medium.yaml --distributed

# Large model with DeepSpeed
deepspeed scripts/train.py --config configs/gpu/large.yaml --deepspeed configs/deepspeed/zero2_config.json
```

## Key Differences

### CPU Configurations
- **Reduced model complexity**: Fewer experts, layers, and attention heads
- **Disabled GPU features**: No Flash Attention, mixed precision, or MoD
- **Larger batch sizes**: Better CPU utilization (16-64 vs 1-8)
- **Multi-worker data loading**: 4-8 workers for parallel data processing
- **Shorter sequences**: 256-512 tokens vs 1024-8192

### GPU Configurations
- **Full model capacity**: All experts and features enabled
- **Advanced optimizations**: Flash Attention, mixed precision (fp16/bf16)
- **Gradient checkpointing**: Trade compute for memory
- **DeepSpeed integration**: ZeRO stages 1-3 for large models
- **Streaming datasets**: Handle large-scale training data

## Model Sizes

| Config      | Parameters | Memory (fp32) | Memory (fp16) | Recommended Hardware |
|-------------|------------|---------------|---------------|---------------------|
| ultra_tiny  | ~100K      | ~0.4MB        | ~0.2MB        | Any CPU/GPU         |
| tiny        | ~50M       | ~200MB        | ~100MB        | Any CPU/GPU         |
| small       | ~776M      | ~3.1GB        | ~1.6GB        | 16GB+ RAM/8GB+ GPU  |
| medium      | ~1.3B      | ~5.2GB        | ~2.6GB        | 32GB+ RAM/16GB+ GPU |
| large       | ~2.9B      | ~11.6GB       | ~5.8GB        | 64GB+ RAM/24GB+ GPU |
| xlarge      | ~5.6B      | ~22.4GB       | ~11.2GB       | 128GB+ RAM/40GB+ GPU|

## macOS Optimization Tips

### For Intel Macs

1. **Set environment variables**:
   ```bash
   export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
   export VECLIB_MAXIMUM_THREADS=$(sysctl -n hw.ncpu)
   ```

2. **Use Accelerate framework** (built into macOS):
   ```python
   # PyTorch automatically uses Accelerate on macOS
   torch.set_num_threads(os.cpu_count())
   ```

### For Apple Silicon (M1/M2/M3)

1. **Use Metal Performance Shaders (MPS)**:
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

2. **Install ARM64-optimized PyTorch**:
   ```bash
   # Ensure you have the ARM64 version
   pip install --upgrade torch torchvision torchaudio
   ```

3. **Enable MPS in configs**:
   ```yaml
   # In your config file
   device_map: mps  # Instead of cpu or cuda
   ```

4. **Memory optimization for Apple Silicon**:
   ```python
   # Unified memory allows larger models
   os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
   ```

### General macOS Tips

1. **Activity Monitor**: Use to check CPU/GPU usage
2. **powermetrics**: Command-line tool for detailed performance metrics
   ```bash
   sudo powermetrics --samplers cpu_power,gpu_power -i 1000
   ```

3. **Disable Spotlight during training**:
   ```bash
   sudo mdutil -a -i off  # Turn off
   sudo mdutil -a -i on   # Turn back on
   ```

## GPU Optimization Tips

1. **Use mixed precision training**:
   - `fp16` for V100 and older
   - `bf16` for A100 and newer

2. **Enable Flash Attention** (if available):
   ```bash
   pip install flash-attn
   ```

3. **Optimize batch size**:
   - Start with batch_size=1 and gradient_accumulation_steps=N
   - Increase batch_size until OOM, then use gradient accumulation

4. **Use DeepSpeed for large models**:
   ```bash
   pip install deepspeed
   ```

## Optimizer Configuration

All configs now support multiple optimizers with adaptive learning rates:

```yaml
training:
  optimizer: adamw  # Options: adamw, lion, sophia, adafactor, rmsprop
  lr_scheduler_type: cosine  # Adaptive learning rate schedule
  warmup_steps: 500  # Gradual LR increase at start
```

### Available Optimizers

#### AdamW (Default - Most widely used)
- Used by: GPT-3/4, LLaMA, PaLM, Chinchilla
- Best for: General purpose, proven performance
```yaml
optimizer: adamw
learning_rate: 6e-4      # Higher for small models, lower for large
weight_decay: 0.01       # Standard for transformers
adam_beta1: 0.9         
adam_beta2: 0.95         # 0.95 for large models, 0.999 for small
adam_epsilon: 1e-8
```

#### Lion (Memory Efficient - 50% less memory)
- Developed by: Google
- Best for: Memory-constrained training
```yaml
optimizer: lion
learning_rate: 3e-4      # Usually half of AdamW LR
weight_decay: 0.1        # Higher than AdamW
adam_beta1: 0.9
adam_beta2: 0.99         # Different from AdamW
```

#### Sophia (Fast Convergence - 2x faster)
- Best for: Experimental, potentially faster training
- Note: Still relatively new (2023)
```yaml
optimizer: sophia
learning_rate: 1e-4      # Lower than AdamW
weight_decay: 0.01
sophia_rho: 0.04         # Hessian diagonal estimate
```

#### Adafactor (Ultra Memory Efficient)
- Used by: T5 and other Google models
- Best for: Very large models with extreme memory constraints
```yaml
optimizer: adafactor
learning_rate: 1e-3      # Can use higher LR
lr_scheduler_type: constant  # Often works better with constant
```

#### RMSprop (Simple and Stable)
- Best for: Smaller models, specific architectures
```yaml
optimizer: rmsprop
learning_rate: 1e-3
rmsprop_momentum: 0.9
```

### Recommendations by Model Size
- **Small models (<1B params)**: AdamW with beta2=0.999
- **Medium models (1-10B)**: AdamW with beta2=0.95 or Lion
- **Large models (10B+)**: Adafactor or Lion for memory efficiency
- **Experimental/Research**: Sophia for potentially faster convergence

## NEW: Attention Mechanism Configuration

All configs now support advanced attention mechanisms:

```yaml
model:
  attention_variant: "standard"  # Options: standard, sliding_window, sparse, streaming, alibi, linear, cached
  
  # Sliding Window Attention
  sliding_window_size: 512
  
  # Sparse Attention (BigBird-style)
  sparse_global_tokens: 128
  sparse_random_blocks: 3
  sparse_local_window_size: 256
  
  # Streaming Attention
  num_sink_tokens: 4
  recent_window_size: 1024
  
  # Linear Attention
  linear_eps: 1e-6
  
  # Cached Attention
  cache_size: 128
  cache_refresh_interval: 100
  
  # Position Embeddings
  use_xpos: false  # Enable extrapolatable position embeddings
  xpos_decay_base: 512.0
```

### Choosing Attention Variants

| Use Case | Config | Key Settings |
|----------|---------|--------------|
| Standard Training | `attention_variant: standard` | Default, works with Flash Attention 2 |
| Long Documents (16k+) | `attention_variant: sparse` | `sparse_global_tokens: 128` |
| Extreme Length (100k+) | `attention_variant: linear` | O(n) complexity |
| Chat Models | `attention_variant: streaming` | `num_sink_tokens: 4` |
| Code Generation | `attention_variant: sliding_window` | `sliding_window_size: 512` |
| Length Extrapolation | `attention_variant: alibi` | No position embeddings |

## NEW: MoE Enhancements

```yaml
model:
  # Parallel expert processing (30-40% faster)
  parallel_expert_processing: true
  
  # Adaptive capacity prediction
  use_adaptive_capacity: false
  capacity_warmup_steps: 1000
  capacity_ema_decay: 0.99
  
  # Expert dropout for regularization
  expert_dropout: 0.0  # 0.1 = drop 10% of experts
  
  # Enhanced load balancing
  use_importance_weighting: true
  importance_temp: 0.5
  load_balancing_loss_fn: "smooth_l1"
  smooth_l1_beta: 1.0
  
  # Memory-efficient training
  expert_gradient_checkpointing: false
  checkpoint_every_n_experts: 2
```

## NEW: PyTorch 2.0 Compile

```yaml
training:
  use_torch_compile: true
  compile_mode: "default"  # Options: default, reduce-overhead, max-autotune
  compile_backend: "inductor"  # Options: inductor, cudagraphs
  compile_fullgraph: true
  compile_dynamic: false  # Set true for dynamic shapes
```

## Customizing Configurations

To create a custom configuration:

1. Copy an existing config that's closest to your needs
2. Adjust model parameters:
   - `hidden_size`: Model width
   - `num_layers`: Model depth
   - `num_experts`: Number of experts in MoE layers
   - `num_experts_per_tok`: Experts activated per token
   - `attention_variant`: Choose attention mechanism

3. Tune training parameters:
   - `batch_size`: Samples per GPU
   - `gradient_accumulation_steps`: Effective batch size multiplier
   - `learning_rate`: Typically 1e-4 to 6e-4
   - `num_epochs` or `max_steps`: Training duration
   - `optimizer`: Choose from adamw, lion, sophia, adafactor, rmsprop

4. Set hardware-specific options:
   - CPU: `device_map: cpu`, `mixed_precision: no`
   - GPU: `device_map: cuda`, `mixed_precision: fp16/bf16`
   - MPS: `device_map: mps`, `mixed_precision: no` (Apple Silicon)