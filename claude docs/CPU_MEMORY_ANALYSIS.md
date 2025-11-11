# CPU Memory Management Analysis - Ava MoE++ Training Pipeline

## Executive Summary

The training pipeline implements a **sophisticated multi-layered approach** to CPU memory management with focus on:
- Efficient data loading and batching
- Gradient accumulation with proper memory handling
- CPU offloading mechanisms for experts
- Memory pinning for GPU transfers
- Proactive OOM prevention

This analysis identifies **current patterns**, **optimization opportunities**, and **potential bottlenecks**.

---

## 1. CURRENT CPU MEMORY MANAGEMENT

### 1.1 Memory Monitoring & Proactive Management

**File**: `/project/code/src/Ava/distributed/memory_monitor.py` (Lines 67-150)

The `MemoryMonitor` class tracks real-time memory usage:

```python
class MemoryMonitor:
    def __init__(
        self,
        target_utilization: float = 0.85,
        warning_threshold: float = 0.99,    # 99% - allows max GPU utilization
        critical_threshold: float = 0.99,   # 99% - batch size reduction trigger
        emergency_threshold: float = 0.99,  # 99% - emergency cleanup trigger
        memory_headroom_gb: float = 1.0,    # Reserve 1GB for safety
        silent_mode: bool = False
    )
```

**Key Features**:
- Maintains history of GPU and CPU memory (deque, max 100 samples)
- OOM prediction with false alarm tracking
- Configurable thresholds with headroom reservation
- Both torch.cuda and psutil integration

**Findings**:
- Thresholds set very aggressively (99%) to allow maximum GPU utilization
- Headroom of only 1GB may be insufficient for large models
- Silent mode available but logs are still generated at DEBUG level

---

### 1.2 GPU Memory Manager

**File**: `/project/code/src/Ava/utils/gpu_memory.py` (Lines 25-150)

Implements comprehensive cleanup:

```python
def cleanup_gpu_memory(self, aggressive: bool = False, enabled: bool = True) -> Dict[str, float]:
```

**Cleanup Operations**:
1. Record initial state: `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
2. Basic cleanup: `torch.cuda.empty_cache()` + `gc.collect()`
3. Aggressive cleanup (when enabled):
   - `torch.cuda.synchronize()`
   - `torch.cuda.ipc_collect()` (IPC memory)
   - **2-round garbage collection** (optimized from 5 rounds)
   - Force memory pool reset

**Findings**:
- Aggressive cleanup trades off 1.5s of time per cleanup to free memory
- Called during OOM recovery (`train.py`, line 3833)
- Defragmentation available every 1000 steps (line 51)

---

## 2. DATA LOADING & BATCHING STRATEGIES

### 2.1 Dynamic Token-Based Batching

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 40-91)

Instead of fixed batch sizes, targets fixed **token counts per batch**:

```python
class DynamicTokenBatcher:
    def __init__(self, max_tokens: int = 8192, max_batch_size: int = 64):
        # Reduces padding overhead by 15-20%
        # Groups samples by token count, not sample count
```

**Memory Impact**:
- **Reduces padding waste**: 15-20% less wasted tokens
- **Dynamic batch sizes**: Sequences varying 64-4096 tokens → same batch
- **Pre-allocated output tensors** (Line 883): Eliminates redundant allocations (10-15% faster)

### 2.2 Length-Based Bucketing

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 93-201)

Groups sequences by length to minimize padding:

```python
class LengthBasedBucketing:
    def __init__(
        self,
        bucket_boundaries: Optional[List[int]] = None,
        max_bucket_size: int = 200,  # Increased from 100
        min_bucket_size: int = 8,
        enable_bucketing: bool = True,
        use_dynamic_batching: bool = False,
        max_tokens_per_batch: int = 8192
    ):
```

**Bucketing Strategy**:
- Default boundaries: `[64, 128, 256, 512, 1024, 2048, 4096]`
- Samples collected into buckets until `max_bucket_size` reached
- Zero-copy implementation: old list returned, new list created

**CPU Memory Impact**:
- Buckets stored in CPU RAM as `Dict[int, List[Dict]]` (Line 126)
- Bucket flush triggered every `buffer_size * 5` samples (Line 998)
- Statistics tracked per bucket for monitoring

### 2.3 Async File Prefetching

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 203-223)

```python
class AsyncFilePrefetcher:
    def __init__(self, max_workers: int = 4, prefetch_size: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=4)
        # OPTIMIZATION: 20-40% faster data loading via background threads
```

**CPU Thread Pool**:
- **4 worker threads** reading files in background
- Fetches next files while current batch is processing
- Non-blocking H2D transfers

### 2.4 Batch Tokenization Optimization

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 741-781)

Batch tokenization vs. individual:

```python
# OPTIMIZED: Batch tokenization with padding to longest in batch
# This is 5-10x faster due to vectorized operations
encoded = self.tokenizer(
    texts,
    max_length=max_length,
    truncation=True,
    padding='longest',  # Pad to longest in this batch only
    return_tensors='pt'
)
```

**Memory Efficiency**:
- Processes minimum 32 samples at a time (Line 960)
- Removes padding before bucketing: `actual_length = attention_mask.sum().item()`
- Reduces intermediate tensor sizes

### 2.5 Dynamic Prefetch Factor Adjustment

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 1177-1189)

```python
# MEMORY OPTIMIZATION: Dynamic prefetch factor based on sequence length
# Formula: prefetch_factor = max(1, min(4, 2048 / max_length))
# - max_length=512  -> prefetch=4 (full prefetch, short sequences)
# - max_length=1024 -> prefetch=2 (default, medium sequences)
# - max_length=2048 -> prefetch=1 (minimal, long sequences)
# - max_length=4096 -> prefetch=1 (minimal, very long sequences)

if prefetch_factor == 2:  # Only auto-adjust if using default
    prefetch_factor = max(1, min(4, int(2048 / max_length)))
    memory_saved_gb = num_workers * (original_prefetch - prefetch_factor) * batch_size * max_length * 2 / (1024**3)
```

**Memory Savings Calculation**:
- Saves ~(original_prefetch - new_prefetch) × num_workers × batch_size × max_length × 2 bytes
- Example: 4 workers, batch=32, seq_len=4096: saves ~4.1GB RAM

### 2.6 Buffer and Worker Configuration

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 1103-1299)

```python
def create_streaming_dataloaders(
    ...
    num_workers: int = 6,        # Optimized for CPU cache
    buffer_size: int = 15000,    # Increased from default for throughput
    prefetch_factor: int = 2,    # Reduced from 4 to 2 for lower memory
    persistent_workers: bool = True,  # Keep workers alive
    samples_per_file: int = 32,  # Read 32 samples per file rotation
    ...
):
```

**CPU Memory per Worker**:
- Each worker maintains: buffer (15000 samples) + bucketing state + file handles
- Persistent workers avoid fork/spawn overhead but consume more CPU RAM
- `prefetch_factor=2`: Each worker prefetches 2 batches = buffer memory

**Calculation**:
- Buffer: 15000 × avg_sequence_tokens × 8 bytes (float32)
- With 6 workers and max_len=2048: ~1.5GB for buffers alone

---

## 3. GRADIENT ACCUMULATION & MEMORY USAGE

### 3.1 Gradient Accumulation Setup

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 2652-2668)

```python
# CRITICAL FIX: Define gradient_accumulation_steps early
gradient_accumulation_steps: int = getattr(
    self.config.training, "gradient_accumulation_steps", 1
)
is_accumulation_complete: bool = ((current_micro_step + 1) % gradient_accumulation_steps) == 0

# DeepSpeed training - handles gradient accumulation internally
self.deepspeed_engine.backward(total_loss)
self.deepspeed_engine.step()
```

**Memory Pattern**:
- **Micro-steps**: Every forward/backward (stored `micro_step_count`)
- **Optimizer steps**: Only when accumulation complete
- DeepSpeed handles gradient accumulation **internally** via config parameter

### 3.2 DeepSpeed Configuration for Memory

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 831-880)

ZeRO stage-specific memory optimizations:

```python
if ds_config.zero_stage == 3:
    config["zero_optimization"].update({
        "stage3_prefetch_bucket_size": ds_config.zero_stage3_prefetch_bucket_size,
        "stage3_param_persistence_threshold": ds_config.zero_stage3_param_persistence_threshold,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    })

# CPU Offloading
if ds_config.cpu_offload:
    config["zero_optimization"]["offload_optimizer"] = {
        "device": "cpu",
        "pin_memory": True,  # Critical for fast H2C transfers
    }
    config["zero_optimization"]["offload_param"] = {
        "device": "cpu",
        "pin_memory": True,
    }
```

**Memory Impact**:
- **ZeRO-3 with offload**: 30-50% GPU memory savings (optimizer state on CPU)
- **Pinned memory**: Enables faster transfer rates (~10GB/s vs ~5GB/s)
- **Stage 3 prefetch**: 50MB bucket size (reduced from 500MB)

### 3.3 Gradient Checkpointing

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 244-260)

```python
# Enable gradient checkpointing for memory savings if configured
if config.model.gradient_checkpointing:
    self.model.gradient_checkpointing_enabled = True
    print(f"   Expected memory savings: ~25%, slowdown: ~8%")
```

**Trade-off**:
- **Saves**: ~25% activation memory (not recomputed during backward)
- **Cost**: 8% slowdown (recompute activations during backward pass)
- Automatically enabled for models >500M parameters

### 3.4 Backward Pass Memory Management

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 2748-3020)

```python
# NOTE: Expert cache clearing moved to AFTER optimizer.step()
# The experts need to stay on GPU through entire backward + optimizer step cycle

# CRITICAL FIX: Ensure loss is scalar before backward
if torch.isnan(total_loss) or torch.isinf(total_loss):
    return {..., "skipped": True, "skip_reason": "invalid_loss"}

# Standard training
scaled_loss.backward()  # or self.scaler.scale(scaled_loss).backward()

# OPTIMIZATION FIX: Clear MoE expert cache AFTER optimizer step
self._clear_moe_expert_cache()
```

**Key Points**:
- Experts kept on GPU during entire backward cycle (prevents device mismatch)
- Cache cleared **after** optimizer.step() completes
- Invalid loss detection prevents crash (line 2628-2647)

---

## 4. CPU OFFLOADING MECHANISMS

### 4.1 Expert CPU Offloading

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 171-348)

```python
class CPUOffloadedExpertGroup(nn.Module):
    """
    Expert group with CPU offloading for inactive experts.
    Memory savings:
    - 8 experts, 2 active: 75% reduction
    - 32 experts, 4 active: 87.5% reduction
    - 64 experts, 8 active: 87.5% reduction
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        max_active_experts: int = 4,
        pin_memory: bool = True,
        async_transfers: bool = True,
        ...
    ):
        # All experts start on CPU (pinned memory)
        for expert in self.experts:
            expert.cpu()
            if pin_memory and torch.cuda.is_available():
                for param in expert.parameters():
                    if param.device.type == 'cpu' and not param.is_pinned():
                        param.data = param.data.pin_memory()  # Pin each parameter
```

**CPU Memory Usage**:
- All expert parameters held in pinned CPU RAM
- For 32 experts × 100M params = 3.2GB CPU RAM pinned
- Pinned memory enables PCIe transfers at ~10-20GB/s

### 4.2 LRU Cache for Active Experts

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 249-258)

```python
# Training mode cache: Keep experts on GPU during training
self._training_cache: Dict[int, nn.Module] = {}
self._cache_enabled = True
self._max_cache_size = max_active_experts if max_active_experts > 0 else num_experts
self._cache_access_order: List[int] = []  # LRU tracking
self._experts_on_gpu: set = set()  # Track ALL experts currently on GPU
```

**Cache Eviction Logic** (Lines 538-556):

```python
if self.training and self._cache_enabled:
    # During training: Keep expert on GPU until after backward pass
    if len(self._training_cache) >= self._max_cache_size and expert_id not in self._training_cache:
        # Evict least recently used expert
        if self._cache_access_order:
            lru_expert_id = self._cache_access_order.pop(0)
            if lru_expert_id in self._training_cache:
                del self._training_cache[lru_expert_id]

    # Store in cache and update access order
    self._training_cache[expert_id] = expert
    if expert_id in self._cache_access_order:
        self._cache_access_order.remove(expert_id)
    self._cache_access_order.append(expert_id)
```

**Memory Guarantee**:
- Cache size capped at `max_active_experts` (typically 4)
- Prevents OOM from unbounded cache growth
- LRU policy prioritizes recently used experts

### 4.3 Multi-Stage Async Prefetch Pipeline

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 259-276)

```python
# PHASE 2 OPTIMIZATION: Dynamic multi-stage prefetch pipeline
self.prefetch_lookahead = prefetch_lookahead
self._adaptive_prefetch = True
self._prefetch_depth_min = 1
self._prefetch_depth_max = min(prefetch_lookahead, 5)  # Increased from 3 to 5
self._current_prefetch_depth = min(prefetch_lookahead, 3)  # Start with default
self._prefetch_miss_count = 0
self._prefetch_hit_count = 0
self._prefetch_adjustment_interval = 100  # Adjust every 100 accesses

# Create max number of CUDA streams (one per prefetch stage)
if torch.cuda.is_available():
    self._prefetch_streams = [torch.cuda.Stream() for _ in range(self._prefetch_depth_max)]
```

**Prefetch Operation** (Lines 501-522):

```python
if prefetch_stream is not None and self.prefetch_lookahead > 0:
    for lookahead_idx in range(1, min(self.prefetch_lookahead + 1, len(unique_experts) - idx)):
        next_expert_id = unique_experts[idx + lookahead_idx]
        next_expert = self.experts[next_expert_id]
        
        # Use different stream for pipeline parallelism
        stream_idx = (lookahead_idx - 1) % len(self._prefetch_streams)
        prefetch_stream_stage = self._prefetch_streams[stream_idx]
        
        with torch.cuda.stream(prefetch_stream_stage):
            # Non-blocking async H2D transfer
            for param in next_expert.parameters():
                if param.device.type == 'cpu':
                    param.data = param.data.to(device, non_blocking=True)
```

**Performance Impact**:
- **25-35% speedup** for CPU offloading via multi-stage prefetch
- Adaptive depth adjustment based on hit rate:
  - If hit_rate < 70%: increase depth (more misses = need deeper prefetch)
  - If hit_rate > 90%: decrease depth (high hits = reduce overhead)

### 4.4 Predictive Caching

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 277-413)

```python
# OPTIMIZATION: Predictive caching based on access patterns
self._access_patterns: Dict[int, List[int]] = defaultdict(list)
self._pattern_history_size = 1000  # Keep last 1000 transitions

def _predict_next_experts(self, current_experts: List[int], k: int = 3) -> List[int]:
    """Predict next likely experts based on access patterns."""
    from collections import Counter
    
    predictions = []
    for expert_id in current_experts:
        if expert_id in self._access_patterns and self._access_patterns[expert_id]:
            # Get most common followers
            counter = Counter(self._access_patterns[expert_id])
            predictions.extend([e for e, _ in counter.most_common(k)])
```

**Optimization**:
- Tracks which experts typically follow which others
- Prefetches predicted experts before they're accessed
- Reduces prefetch misses on predictable patterns

### 4.5 Expert Quantization (Optional)

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 289-333)

```python
if use_lora and use_quantization:
    # All three optimizations: LoRA + Quantization + Offloading
    expert = QuantizedLoRAExpert(...)
elif use_quantization:
    # Quantization + Offloading (no LoRA)
    expert = QuantizedExpert(...)
```

**Memory Reduction**:
- **INT8**: 75% reduction (8-bit vs 32-bit)
- **INT4**: 87% reduction (4-bit vs 32-bit)
- Trade-off: 10-15% slowdown for INT8, 20-30% for INT4

---

## 5. MEMORY PINNING & CPU-GPU TRANSFERS

### 5.1 Pinned Memory Setup

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 343-347)

```python
# Only pin memory if CUDA is available
if pin_memory and torch.cuda.is_available():
    for param in expert.parameters():
        if param.device.type == 'cpu' and not param.is_pinned():
            param.data = param.data.pin_memory()
```

**Impact**:
- Enables ~10-20GB/s transfer speeds (vs ~5GB/s without pinning)
- **Cost**: CPU RAM becomes non-pageable (takes ~10% overhead)
- Only pinned if CUDA available (no wasted CPU RAM on CPU-only systems)

### 5.2 DataLoader Pin Memory

**File**: `/project/code/src/Ava/data/dataloader.py` (Lines 1268-1268)

```python
dataloader_kwargs = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'pin_memory': torch.cuda.is_available(),  # PyTorch will automatically use current accelerator
    'drop_last': True,
    'prefetch_factor': prefetch_factor if num_workers > 0 else None,
    'persistent_workers': persistent_workers if num_workers > 0 else False,
    'collate_fn': None
}
```

**Mechanism**:
- DataLoader's `pin_memory=True` automatically pins batches in collate_fn
- Faster H2D transfer of minibatch data
- Transparent to user - handled by PyTorch internally

### 5.3 Non-Blocking H2D Transfer

**File**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 517-522)

```python
with torch.cuda.stream(prefetch_stream_stage):
    # Non-blocking async H2D transfer
    for param in next_expert.parameters():
        if param.device.type == 'cpu':
            param.data = param.data.to(device, non_blocking=True)
```

**Benefit**:
- Transfer happens asynchronously on separate CUDA stream
- Computation can overlap with data transfer
- Requires stream synchronization before expert use (Line 565-569)

---

## 6. EXISTING MEMORY OPTIMIZATION TECHNIQUES

### 6.1 Gradient Clipping with Memory Awareness

**File**: `/project/code/src/Ava/training/core/trainer.py` (Line 805)

```python
"gradient_clipping": ds_config.gradient_clipping or 1.0,
```

**Memory Impact**:
- Prevents gradient explosion (which would require larger loss scale)
- Stable gradients = smaller gradient buffers needed

### 6.2 Memory Cleanup Strategy

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 2276-2282)

```python
# MEMORY OPTIMIZATION: Cleanup GPU memory before validation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Timing**:
- Called before validation (Line 2280)
- Called after evaluation (Line 2395)
- Prevents memory fragmentation during validation

### 6.3 Conditional Cache Clearing

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 2341-2349)

```python
# OPTIMIZED: Clear cache only when memory usage is high (>95%)
if batch_idx % cache_clear_freq == 0 and torch.cuda.is_available():
    # Only clear if memory usage is high
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved / total_memory > 0.95:
        torch.cuda.empty_cache()
```

**Trade-off**:
- Only clears cache when reserved memory >95%
- Avoids performance loss from excessive cache clearing
- Cache clear frequency: 100 batches (default)

### 6.4 Loss Scaling for Mixed Precision

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 2801-2804)

```python
# Scale loss and backward for mixed precision
self.scaler.scale(scaled_loss).backward()
```

**Memory Impact**:
- FP16 activations use 50% memory vs FP32
- Scaler manages gradient scaling to prevent underflow
- Works with both DeepSpeed and standard training

### 6.5 Batch Size Reduction on OOM

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 1771-1782)

```python
if "out of memory" in str(e).lower():
    logger.warning("Out of memory detected, reducing batch size...")
    cleanup_enabled = getattr(trainer, 'enable_gpu_memory_cleanup', True)
    trainer.gpu_manager.cleanup_gpu_memory(aggressive=True, enabled=cleanup_enabled)
    
    # Reduce batch size for next iteration
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Limitation**:
- Reactive (after OOM), not proactive
- Training must restart from last checkpoint

### 6.6 Contiguous Memory Optimization

**File**: `/project/code/src/Ava/training/core/trainer.py` (Lines 1340)

```python
"contiguous_memory_optimization": True,
```

**Benefit**:
- DeepSpeed's contiguous memory allocation strategy
- Reduces memory fragmentation
- Better cache locality

---

## 7. POTENTIAL BOTTLENECKS & INEFFICIENCIES

### 7.1 CPU Buffer Memory Usage

**Issue**: Buffers accumulate in CPU RAM before tokenization

**Location**: `/project/code/src/Ava/data/dataloader.py` (Lines 931-932)

```python
buffer = []  # Accumulates text until buffer_size reached
if len(buffer) >= self.buffer_size:  # Default 10000
```

**Problem**:
- 10000 text samples × avg 2KB per text = 20MB+ per worker
- With 6 workers = 120MB just for raw text buffers
- Plus bucketing state: `Dict[int, List[...]]` in CPU RAM

**Recommendation**:
- Profile actual buffer memory usage
- Consider streaming tokenization instead of batch tokenization
- Reduce buffer_size for memory-constrained systems

### 7.2 DataLoader Prefetch Memory

**Issue**: `prefetch_factor=2` with 6 workers means 12 prefetched batches

**Location**: `/project/code/src/Ava/data/dataloader.py` (Line 1120)

```python
prefetch_factor: int = 2  # OPTIMIZED: Reduced from 4 to 2
```

**Calculation**:
- 6 workers × 2 prefetch × 32 batch_size × 2048 seq_len × 8 bytes = ~3GB+ RAM
- Plus 6 × 10000 buffer_size = ~1.2GB raw text

**Current Mitigation**:
- Dynamic prefetch factor based on sequence length (Lines 1177-1189)
- Reduces to prefetch=1 for long sequences

**Potential Issue**: 
- Static 6 workers may be overkill
- Auto-detect formula: `num_workers = min(cpu_count, batch_size / 4)`

### 7.3 Expert Offloading Transfer Overhead

**Issue**: CPU-GPU transfers can become bottleneck

**Location**: `/project/code/src/Ava/layers/offloaded_experts.py` (Lines 501-522)

**Transfer Rate**: ~10-20GB/s (pinned memory)

**Latency per Expert Transfer**:
- 100M parameter expert = ~400MB (FP32)
- Transfer time: 400MB / 15GB/s ≈ 27ms
- With 4 experts per forward: 108ms

**Problem**:
- Prefetch helps but can't eliminate transfer cost
- Quantization (INT8) reduces to 27ms → 6.75ms

**Current Mitigation**:
- Adaptive prefetch depth (Lines 670-686)
- Predictive caching (Lines 363-412)
- Non-blocking async transfers (Line 521)

### 7.4 Gradient Accumulation Memory

**Issue**: Gradient buffers grow with accumulation steps

**Location**: `/project/code/src/Ava/training/core/trainer.py` (Line 2654)

```python
gradient_accumulation_steps: int = getattr(self.config.training, "gradient_accumulation_steps", 1)
```

**Memory Impact**:
- With grad_accum=4: all 4 micro-step gradients kept in memory
- Activation memory also scaled (checkpointing helps but not complete)

**Current Mitigation**:
- Gradient checkpointing: 25% activation memory savings
- DeepSpeed ZeRO: distributes gradients across GPUs

### 7.5 Synchronous Cache Clearing Overhead

**Issue**: `torch.cuda.empty_cache()` blocks training

**Location**: `/project/code/src/Ava/training/core/trainer.py` (Line 2281)

```python
torch.cuda.empty_cache()  # Blocks until complete
```

**Impact**:
- Can take 100-500ms on systems with fragmented memory
- Called before validation, causing ~500ms pause

**Alternative**: 
- Async cleanup (would require separate thread)
- More selective clearing based on fragmentation metrics

---

## 8. MEMORY CONFIGURATION & DEFAULTS

### 8.1 Small MoE Configuration

**File**: `/project/code/configs/moe/small_moe.yaml`

```yaml
# Model memory: 4 experts × 100M params ≈ 400M expert parameters
# With batch_size=16, gradient_accum=4: effective batch=64

model:
  num_experts: 4
  num_experts_per_token: 2
  use_lora_experts: true
  lora_rank: 8
  use_expert_offloading: false  # Not needed for 400M experts

training:
  batch_size: 16
  gradient_accumulation_steps: 4

moe_memory_optimization:
  use_lora_experts: true      # 40-60% memory savings
  use_expert_offloading: false # Not needed with LoRA
  use_expert_quantization: false
```

**Memory Budget (24GB GPU)**:
- Model weights: ~2GB (4 experts, LoRA reduces this)
- Batch: 16 × 2048 tokens × 8 bytes × 2 (forward+backward) ≈ 1GB
- Optimizer state (ZeRO offload): GPU only needs ~1GB
- Activations: ~8GB (with checkpointing: ~6GB)
- Headroom: ~5GB
- **Total Available**: ~24GB ✓

---

## 9. SUMMARY OF FINDINGS

### Memory Management Strengths:
1. ✓ Multi-layered approach: data loading, experts, gradients, optimizer
2. ✓ Proactive monitoring with configurable thresholds
3. ✓ Async prefetching with predictive caching
4. ✓ Dynamic adaptation based on sequence length
5. ✓ Both pinned memory and quantization support

### Key Bottlenecks:
1. ✗ CPU buffer accumulation (20-120MB per worker)
2. ✗ DataLoader prefetch memory (3GB+ with defaults)
3. ✗ Expert offloading transfer latency (27-108ms)
4. ✗ Synchronous cache clearing pauses (100-500ms)
5. ✗ Reactive OOM handling (requires restart)

### Optimization Opportunities:
1. Profile actual memory usage of data pipeline
2. Implement async garbage collection
3. Add streaming tokenization option
4. Proactive batch size reduction (not reactive)
5. Expert quantization by default for >1B models

---

## 10. RECOMMENDATIONS

### Immediate (Low Effort):
1. **Reduce buffer_size** dynamically based on available RAM
2. **Increase worker count threshold** before cache memory exceeds budget
3. **Enable expert quantization** for models >1B parameters
4. **Use async cleanup** in separate thread

### Medium Term:
1. Implement **streaming tokenization** (avoid buffer accumulation)
2. Add **memory profiling** utilities for data pipeline
3. Implement **proactive batch reduction** before OOM
4. Multi-stage prefetch tuning per model size

### Long Term:
1. Develop **memory-aware scheduler** for optimal batch/sequence/accumulation
2. Implement **CPU-GPU pipelining** for expert processing
3. Add **memory tracing** for end-to-end pipeline analysis
4. Auto-tune all memory parameters based on hardware + model

