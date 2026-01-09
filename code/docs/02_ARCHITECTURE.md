# Ava Architecture Guide

This document provides a deep dive into the Ava MoE++ architecture, covering model components, routing strategies, and design decisions.

## Architecture Overview

Ava implements a **Mixture of Experts Plus Plus (MoE++)** architecture that combines:

- Sparse expert routing for compute efficiency
- Flash Attention for memory-efficient attention
- Rotary Position Embeddings (RoPE) for position encoding
- SwiGLU/GeGLU gated activations

```
Input Tokens
     ↓
Token Embeddings + RoPE
     ↓
┌─────────────────────────────────────┐
│         Transformer Block x N        │
│  ┌─────────────────────────────────┐ │
│  │   Multi-Head Attention (MHA)    │ │
│  │   with Flash Attention          │ │
│  └─────────────────────────────────┘ │
│              ↓                       │
│  ┌─────────────────────────────────┐ │
│  │      Sparse MoE Layer           │ │
│  │  Router → Top-K Experts → Merge │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
     ↓
Output Logits
```

## Core Components

### 1. Token Embeddings

Located in `code/src/ava/models/moe.py`:

```python
class EnhancedMoEModel:
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = RotaryEmbedding(config.hidden_size // config.num_attention_heads)
```

Configuration:
```yaml
model:
  vocab_size: 50680       # Vocabulary size
  hidden_size: 1024       # Embedding dimension
  max_position_embeddings: 512
```

### 2. Attention Mechanism

Supports multiple attention variants:

| Type | Description | Config |
|------|-------------|--------|
| MHA | Multi-Head Attention | Default |
| MQA | Multi-Query Attention | `num_kv_heads: 1` |
| GQA | Grouped-Query Attention | `num_kv_heads: 4` |

Flash Attention integration:

```yaml
model:
  use_flash_attention: true  # Requires flash-attn package
```

Implementation in `code/src/ava/models/moe.py:EnhancedMoEModel`:

```python
# Flash Attention path
if self.use_flash_attention and flash_attn_available:
    attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout)
else:
    # Standard attention fallback
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

### 3. Sparse MoE Layer

The core innovation: each token is routed to only `k` experts.

Located in `code/src/ava/models/moe_layer.py`:

```python
class SparseMoELayer:
    def forward(self, hidden_states):
        # 1. Compute routing probabilities
        router_logits = self.router(hidden_states)

        # 2. Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token
        )

        # 3. Process through selected experts
        expert_outputs = self.dispatch_to_experts(
            hidden_states, selected_experts, routing_weights
        )

        return expert_outputs
```

### 4. Router Architectures

Three routing strategies available:

#### Mixtral Router
Standard learned gating with load balancing:

```python
class MixtralRouter(nn.Module):
    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        weights = F.softmax(logits, dim=-1)
        return weights
```

#### DeepSeek Router
Hybrid routing with shared experts:

```python
class DeepSeekRouter(nn.Module):
    def forward(self, hidden_states):
        # Some experts always active (shared)
        # Others selected via routing
        shared_output = self.shared_experts(hidden_states)
        routed_output = self.route_to_experts(hidden_states)
        return shared_output + routed_output
```

#### Switch Router
Simplified single-expert routing:

```python
class SwitchRouter(nn.Module):
    def forward(self, hidden_states):
        # Route each token to exactly one expert
        logits = self.gate(hidden_states)
        expert_idx = logits.argmax(dim=-1)
        return expert_idx
```

Configuration:

```yaml
model:
  router_type: 'mixtral'  # 'mixtral', 'deepseek', or 'switch'
  num_experts: 8
  num_experts_per_token: 2
  capacity_factor: 1.25   # Expert capacity buffer
```

### 5. Expert Networks

High-performance FFN experts with gated activations:

```python
class HighPerformanceExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation='swiglu'):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = get_activation(activation)

    def forward(self, x):
        # SwiGLU: gate * up, then project down
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### 6. Auxiliary Losses

MoE training requires auxiliary losses for load balancing:

```yaml
model:
  load_balance_loss_coef: 0.01    # Expert load balancing
  router_z_loss_coef: 0.0001      # Router logit regularization
  diversity_loss_coef: 0.0001     # Encourage diverse routing
```

**Load Balance Loss**: Penalizes uneven expert utilization
```python
# Encourages uniform expert selection across batch
balance_loss = num_experts * (expert_fraction * router_prob_fraction).sum()
```

**Router Z-Loss**: Prevents router logits from growing too large
```python
z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
```

## Model Configuration

### EnhancedMoEConfig

Located in `code/src/ava/models/moe.py`:

```python
@dataclass
class EnhancedMoEConfig:
    # Architecture
    vocab_size: int = 50680
    hidden_size: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    router_type: str = 'mixtral'
    capacity_factor: float = 1.25

    # Optimizations
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    activation: str = 'swiglu'
```

### Parameter Counts

| Component | Formula | Example (1024 hidden) |
|-----------|---------|----------------------|
| Embeddings | vocab × hidden | 50M |
| Attention | 4 × hidden² × layers | 67M |
| Experts | 3 × hidden × intermediate × experts | 100M+ |
| Router | hidden × experts × layers | 0.5M |

## Memory Optimizations

### Gradient Checkpointing

Recomputes activations during backward pass:

```yaml
model:
  gradient_checkpointing: true
```

Memory reduction: ~60% at cost of ~30% slower training.

### KV Cache Quantization

Reduce attention memory for long sequences:

```yaml
model:
  quantize_kv_cache: true
```

## Triton Kernels

Custom CUDA kernels for performance:

- `code/src/ava/kernels/moe.py`: Fused gating and top-k
- `code/src/ava/kernels/activations.py`: Fused SwiGLU/GeGLU
- `code/src/ava/kernels/fused_experts.py`: Batched expert computation

Enable via:

```yaml
model:
  use_triton_kernels: true
  use_fused_activations: true
```

## Design Decisions

### Why MoE?

1. **Compute efficiency**: Only k/N experts active per token
2. **Scaling**: Add experts without proportional compute increase
3. **Specialization**: Experts learn different features

### Why Flash Attention?

1. **Memory**: O(N) vs O(N²) for sequence length N
2. **Speed**: Fused CUDA kernels, no materialized attention matrix
3. **Long contexts**: Enables 8K+ sequence lengths

### Why SwiGLU?

1. **Performance**: Better than ReLU/GELU on LLM benchmarks
2. **Gating**: Learned gating improves expressivity
3. **Stability**: Smoother gradients than ReLU

## Next Steps

- [Training Guide](./03_TRAINING_GUIDE.md) - Train your model
- [Configuration Reference](./04_CONFIGURATION.md) - All parameters
- [API Reference](./08_API_REFERENCE.md) - Code documentation
