# Ava MoE++ Model API Documentation

Advanced Mixture of Experts (MoE++) training framework with production-grade optimizations.

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Model Architecture](#model-architecture)
- [Expert Layers](#expert-layers)
- [Routing Strategies](#routing-strategies)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance Optimizations](#performance-optimizations)
- [API Reference](#api-reference)

---

## Overview

The `ava` package provides a comprehensive framework for training Mixture of Experts language models with state-of-the-art optimizations:

- **MoE++ Architecture**: Transformer with sparse expert layers
- **Multiple Routing Strategies**: Mixtral-style and DeepSeek-style routing
- **High Performance**: Grouped GEMM, Triton kernels, torch.compile support
- **Memory Efficient**: Gradient checkpointing, mixed precision, dynamic batching
- **Production Ready**: Auxiliary losses, capacity limits, expert utilization tracking

```python
from ava.models import EnhancedMoEModel, EnhancedMoEConfig
from ava.models import OptimizedMoETransformer, OptimizedMoEConfig

# Standard MoE for development
config = EnhancedMoEConfig(num_experts=8, num_experts_per_token=2)
model = EnhancedMoEModel(config)

# Optimized MoE for production
config = OptimizedMoEConfig(num_experts=32, use_grouped_gemm=True)
model = OptimizedMoETransformer(config)
```

---

## Package Structure

```
ava/
├── __init__.py              # Package root
├── models/                  # Model architectures
│   ├── moe.py              # EnhancedMoEModel, OptimizedMoETransformer
│   └── moe_layer.py        # SparseMoELayer
├── nn/                      # Neural network layers
│   ├── experts.py          # Expert implementations
│   └── routing.py          # Router implementations
├── kernels/                 # Triton GPU kernels
│   ├── moe.py              # Fused gating/topk kernels
│   └── activations.py      # Fused SwiGLU/GeGLU
├── config/                  # Configuration
│   ├── training_config.py  # Config dataclasses
│   ├── yaml_loader.py      # YAML parsing
│   ├── validator.py        # Config validation
│   └── constants.py        # Constants
├── core/                    # Core utilities
│   ├── activations.py      # Activation factory
│   ├── checkpoint.py       # Async checkpointing
│   ├── mixed_precision.py  # AMP utilities
│   ├── logging.py          # Colored logging
│   └── paths.py            # Path management
├── data/                    # Data loading
│   ├── streaming.py        # StreamingDataset
│   ├── bucketing.py        # Dynamic batching
│   ├── pretokenized.py     # 60x faster Arrow loading
│   └── packing.py          # Sequence packing
├── training/                # Training pipeline
│   ├── pipeline.py         # TrainingPipeline
│   ├── loop.py             # Training loop
│   └── context.py          # Training context
├── optimizations/           # Training speedups
│   ├── dynamic_batching.py # Memory-aware batching
│   ├── checkpointing.py    # Gradient checkpointing
│   └── hybrid_cache.py     # KV + activation caching
├── cuda/                    # CUDA utilities
│   ├── streams.py          # Stream management
│   ├── buffers.py          # Pinned buffers
│   └── metrics.py          # Async metrics
├── optim/                   # Optimizers
│   └── lr_managers.py      # LR scheduling
├── strategies/              # Training strategies
│   └── progressive.py      # Curriculum learning
└── eval/                    # Evaluation
    └── coherence.py        # Coherence metrics
```

---

## Model Architecture

### EnhancedMoEModel

Standard MoE transformer for development and testing.

```python
from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig

config = EnhancedMoEConfig(
    # Architecture
    vocab_size=50680,
    hidden_size=1024,
    num_layers=16,
    num_attention_heads=16,
    intermediate_size=4096,
    max_position_embeddings=2048,

    # MoE settings
    num_experts=8,
    num_experts_per_token=2,
    router_type='switch',  # or 'deepseek'
    expert_capacity_factor=1.25,
    router_aux_loss_coef=0.01,

    # Optimization
    use_flash_attention=True,
    gradient_checkpointing=True,
    hidden_act='gelu',
)

model = EnhancedMoEModel(config)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    return_dict=True
)
loss = outputs['loss']
logits = outputs['logits']
```

**Key Features:**
- RoPE (Rotary Position Embeddings) with caching
- Flash Attention 3/xformers/SDPA fallback chain
- KV cache quantization (75% memory savings)
- Gradient checkpointing (70-80% memory savings)
- Causal attention mask caching

### OptimizedMoETransformer

Production-grade MoE with all performance optimizations.

```python
from ava.models.moe import OptimizedMoETransformer, OptimizedMoEConfig

config = OptimizedMoEConfig(
    # Architecture
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    intermediate_size=14336,  # 3.5x hidden (Mixtral-style)

    # MoE settings
    num_experts=32,
    num_experts_per_token=2,
    router_type='mixtral',  # or 'deepseek'
    capacity_factor=1.25,
    activation='swiglu',

    # Performance
    use_grouped_gemm=True,      # 5-10x speedup
    use_triton_kernels=True,    # 20-30% speedup
    use_torch_compile=True,     # 15-25% speedup

    # Auxiliary losses
    router_z_loss_coef=0.001,
    load_balance_loss_coef=0.01,
    diversity_loss_coef=0.001,

    # DeepSeek shared expert
    use_shared_expert=False,
    shared_expert_weight=0.5,
)

model = OptimizedMoETransformer(config)
```

---

## Expert Layers

### HighPerformanceExpert

Single expert FFN with gated activations (SwiGLU/GeGLU).

```python
from ava.nn.experts import HighPerformanceExpert

expert = HighPerformanceExpert(
    hidden_size=4096,
    intermediate_size=14336,
    activation='swiglu',    # or 'geglu', 'gelu'
    dropout=0.0,
    use_bias=False,
    dtype=torch.bfloat16,
)

# x: [batch_size, hidden_size] or [tokens, hidden_size]
output = expert(x)  # Same shape as input
```

**Architecture:**
```
SwiGLU: x -> [gate_proj, up_proj] -> SiLU(gate) * up -> down_proj -> output
GeGLU:  x -> [gate_proj, up_proj] -> GELU(gate) * up -> down_proj -> output
```

### ExpertParallelGroup

Grouped GEMM for parallel expert computation (5-10x faster).

```python
from ava.nn.experts import ExpertParallelGroup

experts = ExpertParallelGroup(
    num_experts=32,
    hidden_size=4096,
    intermediate_size=14336,
    activation='swiglu',
    dropout=0.0,
    dtype=torch.bfloat16,
)

# Process tokens through selected experts
x = torch.randn(1024, 4096)           # [num_tokens, hidden]
expert_indices = torch.randint(0, 32, (1024, 2))  # [num_tokens, k]
expert_weights = torch.softmax(torch.randn(1024, 2), dim=-1)

output = experts(
    hidden_states=x,
    expert_indices=expert_indices,
    expert_weights=expert_weights,
    use_loop_experts=True,        # Default: D2D-optimized loop
    use_grouped_gemm=False,       # Alternative: batched computation
    use_sparse_dispatch=False,    # Alternative: sparse gather
)
# output: [num_tokens, k, hidden_size]
```

**Dispatch Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `use_loop_experts=True` | Loop over experts, minimal D2D copies | Default (10x less memory bandwidth) |
| `use_grouped_gemm=True` | Batched matmul with index_select | torch.compile friendly |
| `use_sparse_dispatch=True` | Sparse gather, 16x less memory | Inference |

### SharedExpertLayer

Always-active shared expert (DeepSeek-style).

```python
from ava.nn.experts import SharedExpertLayer

shared = SharedExpertLayer(
    hidden_size=4096,
    intermediate_size=14336,
    activation='swiglu',
)

# Processes ALL tokens (no routing)
output = shared(x)  # [batch, seq, hidden]
```

### SparseExpert

Expert with optional sparsity patterns.

```python
from ava.nn.experts import SparseExpert

expert = SparseExpert(
    hidden_size=4096,
    intermediate_size=14336,
    sparsity_ratio=0.5,  # Keep 50% of weights active
)

stats = expert.get_sparsity_stats()
```

---

## Routing Strategies

### MixtralRouter

Top-K routing with learned gating (production-proven).

```python
from ava.nn.routing import MixtralRouter

router = MixtralRouter(
    hidden_size=4096,
    num_experts=32,
    num_selected_experts=2,       # K in top-K
    capacity_factor=1.25,
    router_z_loss_coef=0.001,     # Prevents unbounded logits
    load_balance_loss_coef=0.01,  # Uniform expert utilization
    router_jitter_noise=0.0,      # Exploration noise
    use_triton_kernels=True,      # 15-25% speedup
)

x = torch.randn(1024, 4096)  # [num_tokens, hidden]
indices, weights, aux_loss, metrics = router(x, training=True)

# indices: [1024, 2] - selected expert IDs
# weights: [1024, 2] - normalized routing weights
# aux_loss: scalar - z_loss + load_balance_loss
# metrics: dict - utilization, entropy, balance_score
```

### DeepSeekRouter

Hybrid shared + routed experts.

```python
from ava.nn.routing import DeepSeekRouter

router = DeepSeekRouter(
    hidden_size=4096,
    num_experts=32,              # Routed experts
    num_selected_experts=2,
    num_shared_experts=1,        # Always-active experts
    shared_expert_weight=0.5,    # 50% shared, 50% routed
)

indices, weights, aux_loss, metrics = router(x, training=True)
# indices: [num_tokens, num_shared + k]
# weights: [num_tokens, num_shared + k]
```

### Auxiliary Losses

| Loss | Purpose | Coefficient |
|------|---------|-------------|
| **Z-Loss** | Prevents unbounded logits, improves stability | 0.001 |
| **Load Balance** | Encourages uniform expert utilization | 0.01 |
| **Diversity** | Prevents all tokens using same experts | 0.001 |
| **Expert Dropout** | Regularizes routing confidence | 0.001 |

---

## Quick Start

### Training Script

```python
import torch
from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig
from ava.data import create_streaming_dataloaders
from ava.config import load_config

# Load configuration
config = load_config('configs/moe/large.yaml')

# Create model
model_config = EnhancedMoEConfig(**config['model'])
model = EnhancedMoEModel(model_config).cuda()

# Create dataloaders
train_loader, val_loader = create_streaming_dataloaders(
    data_path='data/fine-tuning/',
    batch_size=config['training']['batch_size'],
    max_seq_length=config['training']['max_seq_length'],
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

for batch in train_loader:
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs['loss']

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Text Generation

```python
# Load trained model
model.eval()

# Generate text
prompt_ids = tokenizer.encode("Once upon a time", return_tensors='pt').cuda()

generated = model.generate(
    input_ids=prompt_ids,
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    do_sample=True,
    use_cache=True,  # KV caching for 20-50x speedup
)

text = tokenizer.decode(generated[0])
```

---

## Configuration

### YAML Configuration

```yaml
model:
  vocab_size: 50680
  hidden_size: 1024
  num_layers: 16
  num_attention_heads: 16
  intermediate_size: 4096
  num_experts: 8
  num_experts_per_token: 2
  router_type: 'mixtral'
  activation: 'swiglu'
  use_flash_attention: true
  gradient_checkpointing: true

training:
  batch_size: 128
  gradient_accumulation_steps: 4
  learning_rate: 0.0006
  warmup_steps: 1000
  num_epochs: 5
  mixed_precision: 'bf16'

dynamic_batching:
  enabled: true
  min_batch_size: 32
  max_batch_size: 512
  target_memory_threshold: 0.7
```

---

## Performance Optimizations

### Summary Table

| Optimization | Speedup | Memory | How to Enable |
|--------------|---------|--------|---------------|
| **Loop Experts** | 10x less D2D | -90% bandwidth | `use_loop_experts=True` (default) |
| **Grouped GEMM** | 5-10x | - | `use_grouped_gemm=True` |
| **Triton Kernels** | 20-30% | - | `use_triton_kernels=True` |
| **torch.compile** | 15-25% | - | `use_torch_compile=True` |
| **Flash Attention** | 2-4x | -40% | `use_flash_attention=True` |
| **Gradient Checkpointing** | - | -70-80% | `gradient_checkpointing=True` |
| **Mixed Precision** | 20-30% | -50% | `mixed_precision='bf16'` |
| **Dynamic Batching** | 15-25% | - | `dynamic_batching.enabled=True` |
| **Pre-tokenized Data** | 60x | - | Use Arrow format |
| **KV Cache** | 20-50x | - | `use_cache=True` in generate |

### Loop-over-Experts (D2D Optimized)

```python
# Default dispatch strategy - minimizes Device-to-Device memory copies
# Instead of gathering all expert weights at once (massive D2D):
#   index_select creates [N*k, H, I*2] tensor = ~60GB for typical batch

# Loop approach processes one expert at a time:
#   Only copies token hidden states for that expert (~1MB vs 60GB)
#   More kernel launches but ~10x less memory bandwidth

experts = ExpertParallelGroup(num_experts=32, ...)
output = experts(tokens, expert_indices, expert_weights, use_loop_experts=True)
```

### Triton Kernels

```python
from ava.kernels.moe import fused_softmax_topk
from ava.kernels.activations import fused_swiglu

# Fused softmax + topk (automatic optimization)
# - Uses Triton for large batches (>=4K tokens, <=128 experts): 1.2-1.3x speedup
# - Uses PyTorch for smaller batches (lower kernel launch overhead)
weights, indices = fused_softmax_topk(logits, top_k=2)

# Fused gated activation (10-15% speedup)
hidden = fused_swiglu(gate_up_output)
```

---

## API Reference

### Models

#### `EnhancedMoEModel`
```python
class EnhancedMoEModel(nn.Module):
    """Standard MoE transformer."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[tuple]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns:
            loss: Cross-entropy + aux losses
            logits: [batch, seq, vocab]
            hidden_states: [batch, seq, hidden]
            aux_info: Per-layer routing info
            past_key_values: KV cache for generation
        """

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor: ...

    def clear_caches(self) -> None:
        """Clear RoPE and attention mask caches to free VRAM."""
```

#### `OptimizedMoETransformer`
```python
class OptimizedMoETransformer(nn.Module):
    """Production MoE with all optimizations."""

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert utilization across all layers."""

    def reset_expert_counts(self) -> None:
        """Reset utilization counters."""
```

### Expert Layers

#### `ExpertParallelGroup`
```python
class ExpertParallelGroup(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,      # [num_tokens, hidden]
        expert_indices: torch.Tensor,      # [num_tokens, k]
        expert_weights: Optional[torch.Tensor] = None,
        use_loop_experts: bool = True,     # D2D-optimized (default)
        use_grouped_gemm: bool = False,
        use_sparse_dispatch: bool = False,
    ) -> torch.Tensor:
        """Returns: [num_tokens, k, hidden_size]"""
```

### Routers

#### `MixtralRouter`
```python
class MixtralRouter(UnifiedMoERouter):
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[
        torch.Tensor,      # expert_indices [num_tokens, k]
        torch.Tensor,      # expert_weights [num_tokens, k]
        torch.Tensor,      # aux_loss (scalar)
        Dict[str, Any],    # metrics
    ]: ...
```

### Sparse MoE Layer

#### `SparseMoELayer`
```python
class SparseMoELayer(nn.Module):
    """Drop-in FFN replacement with MoE."""

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            output: [batch, seq, hidden]
            aux_loss: Total auxiliary loss
            metrics: Routing metrics
        """
```

### Activations

```python
from ava.core.activations import (
    get_activation,       # Factory function
    is_gated_activation,  # Check if needs 2x intermediate
    ACTIVATION_REGISTRY,  # Name -> nn.Module mapping
    GATED_ACTIVATIONS,    # Set of gated activations
)

# Usage
act = get_activation('swiglu')  # Returns nn.SiLU()
if is_gated_activation('swiglu'):
    intermediate_size *= 2  # Gated activations need 2x
```

---

## Best Practices

### Memory Management

```python
# 1. Enable gradient checkpointing for large models
config.gradient_checkpointing = True

# 2. Use mixed precision
config.mixed_precision = 'bf16'

# 3. Clear caches periodically during training
if step % 500 == 0:
    model.clear_caches()
    torch.cuda.empty_cache()

# 4. Use dynamic batching for variable-length sequences
config.dynamic_batching.enabled = True
```

### Training Stability

```python
# 1. Use auxiliary losses for balanced training
config.router_aux_loss_coef = 0.01
config.router_z_loss_coef = 0.001

# 2. Add jitter noise for exploration
config.router_jitter_noise = 0.01

# 3. Set appropriate capacity factor
config.expert_capacity_factor = 1.25  # 25% buffer
```

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project overview
- [02_TRAINING_GUIDE.md](02_TRAINING_GUIDE.md) - Training procedures
- [03_MEMORY_OPTIMIZATION.md](03_MEMORY_OPTIMIZATION.md) - Memory techniques
