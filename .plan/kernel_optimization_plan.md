# Kernel Optimization Plan for Ava MoE Framework

## Executive Summary

This plan targets **50-80% throughput improvement** through comprehensive kernel optimizations across router gating, expert computation, and fused activations. Implementation spans 3 phases over approximately 6-8 weeks.

---

## Current State Analysis

### Identified Bottlenecks

| Component | File | Issue | GPU Idle % |
|-----------|------|-------|------------|
| Router Gating | `routing.py:347-351` | 3 separate kernels (linear→softmax→topk) | 18-22% |
| Top-K Selection | `moe_kernels.py:119-131` | O(k) iterative, k≤8 limit, BLOCK_SIZE=1 | 10-15% |
| Expert Dispatch | `experts.py:313-334` | Dense one-hot [N*k, E], 94% waste | 25-30% |
| Gated Activation | `experts.py:332-334` | 3 ops: chunk→silu→multiply | 8-12% |
| Capacity Limiting | `moe_layer.py:433-455` | Sequential loop over num_experts | 15-20% |

### Current Performance Baseline
- Router: ~3 kernel launches per forward pass
- Expert computation: Dense tensor dispatch with einsum
- Activation: Separate operations, no fusion
- **Total estimated idle time: 40-50%**

---

## Phase 1: Router & Gating Kernel Optimization

**Timeline**: 1-2 weeks
**Expected Gain**: 15-25%
**Risk**: Low

### 1.1 Optimized Top-K Algorithm

**Current** (`moe_kernels.py:119-131`):
```python
# O(k) sequential max-finding per token
for k_idx in range(top_k):
    max_prob = tl.max(probs, axis=1)
    max_idx = tl.argmax(probs, axis=1)
    # ... mask and repeat
```

**Optimized**: Implement bitonic sort for k≤8, heap-based for k>8

**File**: `code/src/Ava/kernels/moe_kernels.py`

```python
# New function: _bitonic_topk_kernel
# - Parallel sorting network for k≤8
# - O(log²k) depth instead of O(k)
# - Better GPU utilization through parallelism

# New function: _heap_topk_kernel
# - For k>8 (removes current limitation)
# - O(E log k) complexity
# - Maintains heap in shared memory
```

### 1.2 Parallel Batch Processing

**Current**: `BLOCK_SIZE_TOKEN = 1` (poor GPU occupancy)

**Optimized**: `BLOCK_SIZE_TOKEN = 4-8` with proper tiling

**Changes**:
```python
# moe_kernels.py line 180
BLOCK_SIZE_TOKEN = 1  # Current

# Optimized:
BLOCK_SIZE_TOKEN = min(8, triton.cdiv(num_tokens, 128))  # Adaptive
```

### 1.3 Fused Softmax + Top-K

**Current**: Separate softmax then top-k selection

**Optimized**: Single-pass online softmax with simultaneous top-k tracking

**New kernel**: `_fused_softmax_topk_kernel`
- Compute running max during softmax
- Track top-k candidates in registers
- Single memory write for final results

### 1.4 Files to Modify

| File | Changes |
|------|---------|
| `code/src/Ava/kernels/moe_kernels.py` | Add bitonic/heap top-k, parallel batching |
| `code/src/Ava/kernels/__init__.py` | Export new kernels |
| `code/src/Ava/layers/routing.py` | Remove k≤8 restriction, use optimized kernels |
| `code/src/Ava/config/training_config.py` | Add `router_kernel_mode` config option |

---

## Phase 2: Expert Computation Optimization

**Timeline**: 2-3 weeks
**Expected Gain**: 20-35%
**Risk**: Medium

### 2.1 Sparse Expert Dispatch

**Current** (`experts.py:313-334`):
```python
# Dense one-hot tensor - wastes 94% memory when k=2, E=32
dispatch_tensor = F.one_hot(flat_indices, num_classes=num_experts).to(dtype)
# Shape: [N*k, num_experts] = [256*2, 32] = 16K elements for 512 useful indices
```

**Optimized**: Direct sparse gather with index tensor

**New implementation**:
```python
def _forward_sparse_gather(self, hidden_states, expert_indices, expert_weights):
    """
    Sparse gather: Only load weights for selected experts.

    Memory: O(N*k) instead of O(N*k*E)
    Bandwidth: 16x reduction for E=32, k=2
    """
    num_tokens, k = expert_indices.shape

    # Gather only needed expert weights using advanced indexing
    # This is torch.compile compatible with proper handling
    flat_indices = expert_indices.reshape(-1)

    # Use index_select for contiguous memory access
    selected_gate_up = torch.index_select(
        self.gate_up_weights, 0, flat_indices
    )  # [N*k, H, I*2]

    # Batched matmul
    hidden_expanded = hidden_states.repeat_interleave(k, dim=0)
    gate_up = torch.bmm(
        hidden_expanded.unsqueeze(1),
        selected_gate_up
    ).squeeze(1)  # [N*k, I*2]

    # ... rest of computation
```

### 2.2 Fused Gated Activation Kernel

**Current** (`experts.py:332-334`):
```python
gate, up = gate_up.chunk(2, dim=-1)  # Kernel 1: slice
hidden = self.activation(gate) * up   # Kernel 2: silu, Kernel 3: multiply
```

**Optimized**: Single Triton kernel for SwiGLU/GeGLU

**New file**: `code/src/Ava/kernels/activation_kernels.py`

```python
@triton.jit
def _fused_swiglu_kernel(
    input_ptr, output_ptr,
    intermediate_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SwiGLU: output = silu(gate) * up

    - Single kernel instead of 3
    - No intermediate tensor allocation
    - 10-15% speedup for expert computation
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load gate and up in single read
    gate = tl.load(input_ptr + offset)
    up = tl.load(input_ptr + offset + intermediate_size)

    # Fused SiLU: x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)

    # Output
    result = silu_gate * up
    tl.store(output_ptr + offset, result)
```

### 2.3 Selective Expert Loading (for large E)

For models with many experts (32+), load only active expert weights:

```python
class SelectiveExpertLoader:
    """
    Load expert weights on-demand for memory efficiency.

    Benefits:
    - Reduces memory footprint by (E-k)/E (93.75% for E=32, k=2)
    - Enables larger models on same hardware
    - Async prefetching hides latency
    """

    def __init__(self, expert_weights, num_experts, device):
        self.weights_cpu = expert_weights  # Keep on CPU
        self.cache = {}  # GPU cache for hot experts
        self.stream = torch.cuda.Stream()

    def load_experts(self, expert_indices):
        unique_experts = expert_indices.unique()
        # Async load missing experts
        with torch.cuda.stream(self.stream):
            for e in unique_experts:
                if e not in self.cache:
                    self.cache[e] = self.weights_cpu[e].to(device, non_blocking=True)
        return self.cache
```

### 2.4 Files to Modify

| File | Changes |
|------|---------|
| `code/src/Ava/kernels/activation_kernels.py` | **NEW**: Fused SwiGLU/GeGLU kernels |
| `code/src/Ava/layers/experts.py` | Sparse gather, integrate fused activations |
| `code/src/Ava/utils/selective_loader.py` | **NEW**: On-demand expert loading |
| `code/src/Ava/config/training_config.py` | Add expert optimization configs |

---

## Phase 3: Advanced Kernel Fusion

**Timeline**: 3-4 weeks
**Expected Gain**: 30-50%
**Risk**: Higher (more complex)

### 3.1 Mega-Kernel: Fused Routing + Expert Dispatch

Combine routing decision and expert dispatch into single kernel:

**New file**: `code/src/Ava/kernels/fused_moe_kernel.py`

```python
@triton.jit
def _fused_moe_forward_kernel(
    # Inputs
    hidden_ptr,           # [N, H]
    router_weight_ptr,    # [H, E]
    expert_weights_ptr,   # [E, H, I*2]
    expert_down_ptr,      # [E, I, H]
    # Outputs
    output_ptr,           # [N, H]
    aux_loss_ptr,         # Scalar
    # Dimensions
    num_tokens, hidden_size, num_experts, intermediate_size, top_k,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """
    MEGA KERNEL: Full MoE forward in single kernel launch.

    Fuses:
    1. Router linear projection
    2. Softmax + Top-K selection
    3. Expert weight gathering
    4. Expert forward (gate_up + activation + down)
    5. Weighted combination

    Benefits:
    - Single kernel launch vs 8-10 separate
    - No intermediate tensor allocation
    - Optimal register/shared memory usage
    - 40-60% speedup for full MoE layer
    """
    # ... implementation details
```

### 3.2 Vectorized Capacity Limiting

**Current** (`moe_layer.py:433-455`): Sequential loop

**Optimized**: Parallel prefix sum + scatter

```python
def _apply_capacity_limits_vectorized(self, expert_indices, expert_weights, num_tokens):
    """
    Vectorized capacity limiting using parallel algorithms.

    Algorithm:
    1. Sort tokens by (expert_id, -weight) using radix sort
    2. Compute prefix sum per expert
    3. Mask tokens exceeding capacity
    4. Scatter back to original positions

    Complexity: O(N log N) parallel vs O(N * E) sequential
    """
    batch_size, k = expert_indices.shape
    device = expert_indices.device

    # Flatten
    flat_indices = expert_indices.reshape(-1)
    flat_weights = expert_weights.reshape(-1)

    # Sort by expert (stable sort preserves weight ordering)
    sorted_experts, sort_perm = torch.sort(flat_indices, stable=True)
    sorted_weights = flat_weights[sort_perm]

    # Compute per-expert token counts via segment boundaries
    expert_boundaries = torch.searchsorted(
        sorted_experts,
        torch.arange(self.num_experts + 1, device=device)
    )

    # Prefix sum within each expert segment
    positions_in_expert = torch.zeros_like(flat_indices)
    for i in range(self.num_experts):
        start, end = expert_boundaries[i], expert_boundaries[i+1]
        if end > start:
            positions_in_expert[sort_perm[start:end]] = torch.arange(end - start, device=device)

    # Capacity mask
    capacity = int((num_tokens * k / self.num_experts) * self.capacity_factor)
    keep_mask = positions_in_expert < capacity

    # Apply mask and renormalize
    expert_weights = expert_weights * keep_mask.reshape(batch_size, k).float()
    weight_sum = expert_weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
    expert_weights = expert_weights / weight_sum

    return expert_indices, expert_weights
```

### 3.3 Quantized Attention Kernels

Extend existing Flash Attention with INT8 KV cache support:

**New file**: `code/src/Ava/kernels/quantized_attention.py`

```python
@triton.jit
def _flash_attention_int8_kv_kernel(
    q_ptr, k_ptr, v_ptr, k_scale_ptr, v_scale_ptr,
    output_ptr,
    # ...
):
    """
    Flash Attention with INT8 quantized KV cache.

    - 4x memory reduction for KV cache
    - Fused dequantization during attention
    - <1% accuracy loss with proper scaling
    """
    # Load INT8 K/V and dequantize in registers
    k_int8 = tl.load(k_ptr + k_offset).to(tl.int8)
    k_scale = tl.load(k_scale_ptr + scale_offset)
    k = k_int8.to(tl.float16) * k_scale
    # ... rest of attention
```

### 3.4 Files to Modify/Create

| File | Changes |
|------|---------|
| `code/src/Ava/kernels/fused_moe_kernel.py` | **NEW**: Mega MoE kernel |
| `code/src/Ava/kernels/quantized_attention.py` | **NEW**: INT8 attention |
| `code/src/Ava/models/moe_layer.py` | Integrate mega kernel, vectorized capacity |
| `code/src/Ava/models/moe_model.py` | Integrate quantized attention |

---

## Implementation Order & Dependencies

```
Phase 1 (Week 1-2)
├── 1.1 Bitonic top-k kernel
├── 1.2 Parallel batch processing
├── 1.3 Fused softmax+topk
└── 1.4 Router integration

Phase 2 (Week 3-5)
├── 2.1 Sparse expert dispatch (depends on 1.x)
├── 2.2 Fused SwiGLU kernel
├── 2.3 Selective expert loading
└── 2.4 Expert integration

Phase 3 (Week 6-8)
├── 3.1 Mega MoE kernel (depends on 1.x, 2.x)
├── 3.2 Vectorized capacity limiting
├── 3.3 Quantized attention
└── 3.4 Full integration & testing
```

---

## New Files to Create

1. **`code/src/Ava/kernels/activation_kernels.py`**
   - `_fused_swiglu_kernel`
   - `_fused_geglu_kernel`
   - `fused_gated_activation()` wrapper

2. **`code/src/Ava/kernels/fused_moe_kernel.py`**
   - `_fused_moe_forward_kernel`
   - `_fused_moe_backward_kernel`
   - `FusedMoEFunction` (autograd)

3. **`code/src/Ava/kernels/quantized_attention.py`**
   - `_flash_attention_int8_kv_kernel`
   - `quantized_attention()` wrapper

4. **`code/src/Ava/utils/selective_loader.py`**
   - `SelectiveExpertLoader`
   - `ExpertCacheManager`

---

## Files to Modify

| File | Modifications |
|------|---------------|
| `code/src/Ava/kernels/moe_kernels.py` | Bitonic top-k, heap top-k, parallel batching |
| `code/src/Ava/kernels/__init__.py` | Export all new kernels |
| `code/src/Ava/layers/routing.py` | Remove k≤8 limit, use optimized kernels |
| `code/src/Ava/layers/experts.py` | Sparse gather, fused activations |
| `code/src/Ava/models/moe_layer.py` | Vectorized capacity, mega kernel option |
| `code/src/Ava/models/moe_model.py` | Quantized attention integration |
| `code/src/Ava/config/training_config.py` | New optimization config options |

---

## Configuration Additions

Add to `training_config.py`:

```python
@dataclass
class KernelOptimizationConfig:
    """Kernel-level optimization settings."""

    # Router optimizations
    router_kernel_mode: str = 'auto'  # 'auto', 'triton', 'pytorch'
    use_bitonic_topk: bool = True     # Parallel top-k for k≤8
    use_heap_topk: bool = True        # Enable k>8 support
    router_block_size: int = 4        # Tokens per thread block

    # Expert optimizations
    use_sparse_expert_dispatch: bool = True   # Sparse vs dense dispatch
    use_fused_activations: bool = True        # Fused SwiGLU/GeGLU
    use_selective_expert_loading: bool = False # On-demand loading (large E)
    expert_cache_size: int = 8                # Experts to keep on GPU

    # Advanced optimizations
    use_fused_moe_kernel: bool = False  # Mega kernel (experimental)
    use_vectorized_capacity: bool = True
    use_quantized_attention: bool = False
    kv_cache_dtype: str = 'float16'  # 'float16', 'int8', 'fp8'
```

---

## Testing Strategy

### Unit Tests
```python
# test_kernels.py
def test_bitonic_topk_correctness():
    """Verify bitonic top-k matches torch.topk"""

def test_fused_swiglu_correctness():
    """Verify fused SwiGLU matches sequential"""

def test_sparse_dispatch_correctness():
    """Verify sparse dispatch matches dense"""
```

### Performance Benchmarks
```python
# benchmark_kernels.py
def benchmark_router_kernels():
    """Compare: PyTorch vs Triton vs Optimized Triton"""

def benchmark_expert_computation():
    """Compare: Dense vs Sparse dispatch"""

def benchmark_full_moe_layer():
    """End-to-end MoE layer throughput"""
```

### Integration Tests
```python
# test_integration.py
def test_training_convergence():
    """Verify training loss matches baseline with optimizations"""

def test_generation_quality():
    """Verify generation output unchanged"""
```

---

## Expected Performance Gains

| Phase | Component | Gain | Cumulative |
|-------|-----------|------|------------|
| 1.1 | Bitonic top-k | 8-12% | 8-12% |
| 1.2 | Parallel batching | 5-8% | 13-20% |
| 1.3 | Fused softmax+topk | 5-8% | 18-28% |
| 2.1 | Sparse dispatch | 15-20% | 33-48% |
| 2.2 | Fused activations | 8-12% | 41-60% |
| 3.1 | Mega kernel | 15-25% | 56-85% |
| 3.2 | Vectorized capacity | 3-5% | 59-90% |

**Target: 50-80% overall throughput improvement**

---

## Risk Mitigation

1. **Fallback Paths**: Every optimization has PyTorch fallback
2. **Config Flags**: All optimizations toggleable via config
3. **Correctness Tests**: Extensive numerical validation
4. **Gradual Rollout**: Enable one optimization at a time
5. **Monitoring**: Add kernel timing metrics to training loop

---

## Hardware Considerations

| GPU | Recommended Settings |
|-----|---------------------|
| **Ampere (A100)** | All Phase 1-2, selective Phase 3 |
| **Ada (L40/4090)** | All phases, enable FP8 where available |
| **Hopper (H100)** | All phases, full FP8, TMA for mega kernel |
| **Older (V100)** | Phase 1 only, conservative settings |

---

## Success Metrics

1. **Throughput**: tokens/sec improvement ≥50%
2. **Memory**: No regression, ideally 10-20% reduction
3. **Accuracy**: Loss curves within 1% of baseline
4. **Stability**: No new training failures or NaNs
