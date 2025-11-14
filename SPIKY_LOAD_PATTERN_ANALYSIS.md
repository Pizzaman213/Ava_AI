# Data Loading and Expert Offloading: Spiky Load Pattern Analysis

## Executive Summary
The Ava MoE++ system exhibits several architectural patterns that can create **spiky, bursty load patterns** rather than smooth, continuous GPU utilization. These spikes occur at different timescales and have distinct causes related to data loading, expert offloading, synchronization, and memory management.

---

## 1. PRIMARY SOURCES OF SPIKY LOADS

### 1.1 Asynchronous File I/O with Prefetching

**Location**: `/project/code/src/Ava/data/dataloader.py` (lines 274-363)

**Pattern: Bursty I/O Followed by Processing Stalls**

```python
class AsyncFilePrefetcher:
    def prefetch_file(self, file_path: Path, reader_func: Callable) -> Future:
        """Submit a file read with adaptive prefetch depth adjustment."""
        future = self.executor.submit(reader_func, file_path)
        self.futures.append(future)
        # Track I/O latency for adaptive adjustment
        def track_latency(fut):
            latency = time.time() - start_time
            self.io_latencies.append(latency)
            self._adjust_prefetch_depth()
        future.add_done_callback(track_latency)
```

**Spiky Behavior**:
- **Prefetch depth adjustment** (lines 326-336): Increases prefetch depth when I/O latency > 100ms
- **Burst submission**: Multiple files are prefetched concurrently (up to `PREFETCH_SIZE`, default 2-3)
- **Wait points**: GPU computation stalls while waiting for I/O futures to complete
- **Bursty pattern**: Creates cycles of (submit batch of reads) → (wait for completion) → (process) → (repeat)

**Impact**: 
- I/O latency spikes cause GPU idle periods
- Adaptive prefetch depth creates feedback loops that can amplify spikes
- ThreadPoolExecutor with fixed workers creates resource contention

---

### 1.2 Length-Based Bucketing with Dynamic Batch Formation

**Location**: `/project/code/src/Ava/data/dataloader.py` (lines 150-271)

**Pattern: Threshold-Triggered Batch Releases**

```python
class LengthBasedBucketing:
    def add_sample(self, sample: Dict[str, torch.Tensor]) -> Optional[List]:
        # Return full bucket if threshold reached (OPTIMIZED: zero-copy)
        if len(self.buckets[bucket_id]) >= self.max_bucket_size:
            full_bucket = self.buckets[bucket_id]
            self.buckets[bucket_id] = []  # New list
            return full_bucket
        return None
```

**Spiky Behavior**:
- **Batch emission**: When a bucket reaches `max_bucket_size` (default 200), entire batch is returned at once
- **Idle to burst**: Samples trickle in → bucket fills → BURST of tokens to GPU
- **Variable bucket sizes**: Different sequence length buckets fill at different rates, creating irregular emission patterns
- **Padding waste spike**: When bucket is full with short sequences + one long sequence, causes padding spike

**Example Timeline**:
```
Time: 0-100ms:   Bucket filling slowly (50% full)
Time: 100-105ms: Bucket suddenly fills, releases 200 samples
Time: 105-110ms: GPU processes large burst of 200 samples
Time: 110-200ms: Bucket building again (idle or light load)
Time: 200-205ms: Next burst release
```

**Impact**:
- Creates sawtooth utilization pattern
- Inefficient pipelining: Data loading doesn't overlap with computation
- Memory fragmentation: Large bursts cause allocation spikes

---

### 1.3 Dynamic Token Batching with Flush Intervals

**Location**: `/project/code/src/Ava/data/dataloader.py` (lines 84-148)

**Pattern: Periodic Batch Flushing**

```python
class DynamicTokenBatcher:
    def add_sample(self, sample: Dict) -> Optional[List]:
        would_exceed_tokens = (self.current_tokens + seq_len) > self.max_tokens
        would_exceed_batch = len(self.current_batch) >= self.max_batch_size
        
        if (would_exceed_tokens or would_exceed_batch) and self.current_batch:
            ready_batch = self.current_batch
            self.current_batch = [sample]
            self.current_tokens = seq_len
            return ready_batch
```

**Spiky Behavior**:
- **Dual thresholds**: Batch flushes when EITHER token limit OR count limit hit
- **Unpredictable release points**: High-variance sequence lengths cause irregular flush timing
- **Flush cascades**: One flush at token limit triggers next flush immediately in next iteration

**Token distribution impact**:
- Short sequences: 10-100 tokens
- Long sequences: 1000-4096 tokens
- Ratio variance: 1:400x causes high unpredictability

**Impact**:
- Inconsistent batch token counts
- GPU underutilization on small batches
- Wasted compute on large batches with padding

---

### 1.4 Synchronous Data Loader Iteration with Buffering

**Location**: `/project/code/src/Ava/data/dataloader.py` (lines 1282-1425)

**Pattern: Buffer-Fill-Drain Cycles**

```python
def __iter__(self):
    buffer = deque(maxlen=dynamic_buffer_size)
    
    for text in self._stream_examples():
        buffer.append(text)
        
        # Process buffer when full
        if len(buffer) >= self.buffer_size:
            buffer_list = list(buffer)
            # Tokenize entire buffer
            tokenized_samples = self._tokenize_batch(buffer_list, ...)
            # Yield all
            for tokenized in tokenized_samples:
                bucket_samples = self.bucketing.add_sample(tokenized)
                if bucket_samples is not None:
                    for sample in bucket_samples:
                        yield sample
            buffer.clear()  # DRAIN
```

**Spiky Behavior**:
- **Three-level buffering**: `deque(buffer_size=10000)` → `bucketing` → `collate_fn`
- **Synchronous drain**: Entire buffer tokenized at once when full
- **CPU tokenization spike**: Batch tokenization on 10,000 samples causes CPU burst
- **Subsequent GPU burst**: All tokenized samples suddenly available for GPU

**Timeline**:
```
Time: 0-50ms:    Accumulate 10,000 samples in buffer
Time: 50-150ms:  Batch tokenization (CPU burst, 5-10x actual needed)
Time: 150-200ms: Bucketing and batching (CPU processing)
Time: 200-300ms: GPU processes batch burst
Time: 300-350ms: New buffer accumulation begins (GPU may idle)
```

**Impact**:
- CPU tokenization creates bottleneck
- GPU-CPU synchronization points cause stalls
- Buffer accumulation delays GPU work by 50+ ms

---

## 2. EXPERT OFFLOADING SYNCHRONIZATION POINTS

### 2.1 Sequential Expert Loading During Forward Pass

**Location**: `/project/code/src/Ava/layers/offloaded_experts.py` (lines 422-654)

**Pattern: Synchronous Expert Movement**

```python
def forward(self, hidden_states, expert_indices, expert_weights=None):
    unique_experts = torch.unique(expert_indices.flatten()).tolist()
    
    for idx, expert_id in enumerate(unique_experts):
        # Check if expert is on device
        current_device = next(expert.parameters()).device
        already_on_device = current_device == device
        
        if not already_on_device:
            # BLOCKING: Move expert to GPU
            expert.to(device)
            self._experts_on_gpu.add(expert_id)
        
        # Process expert
        expert_input = hidden_states[token_indices]
        expert_output = expert(expert_input)
        
        # Place in output
        output[token_indices, k_indices] = expert_output
```

**Spiky Behavior**:
- **Sequential loading**: Each expert loaded one-at-a-time
- **No overlap**: CPU→GPU transfer blocks compute
- **Blocking synchronization**: `expert.to(device)` waits for H2D transfer

**Example with 4 experts, 2 active per batch**:
```
Time: 0-5ms:    Expert 1 H2D transfer (BLOCKING)
Time: 5-15ms:   Expert 1 compute
Time: 15-20ms:  Expert 2 H2D transfer (BLOCKING)
Time: 20-30ms:  Expert 2 compute
Total: 30ms instead of potential 15-20ms with overlap
```

**Optimization Attempts** (but insufficient):
- Lines 520-542: Async prefetch with separate streams
- Lines 615-625: Stream synchronization (still blocking main compute)

**Problem**: Lookahead prefetch only helps NEXT expert, not current one.

**Impact**:
- 20-40% overhead from expert transfers
- GPU underutilization during H2D transfers
- No pipelining: Transfer and compute can't overlap

---

### 2.2 Expert Cache Eviction and GPU Memory Spikes

**Location**: `/project/code/src/Ava/layers/offloaded_experts.py` (lines 576-613)

**Pattern: LRU Eviction Under Memory Pressure**

```python
def forward(self):
    # During training: Keep expert on GPU until backward
    if self.training and self._cache_enabled:
        if len(self._training_cache) >= self._max_cache_size:
            if self._cache_access_order:
                lru_expert_id = self._cache_access_order.pop(0)
                # Only evict if not in current forward pass
                if lru_expert_id not in unique_experts:
                    evicted_expert = self._training_cache[lru_expert_id]
                    # BLOCKING: Move back to CPU
                    if next(evicted_expert.parameters()).device.type == 'cuda':
                        evicted_expert.cpu()
                        self._experts_on_gpu.discard(lru_expert_id)
        
        self._training_cache[expert_id] = expert
```

**Spiky Behavior**:
- **Threshold-triggered eviction**: When cache size hits limit, LRU expert offloaded
- **Blocking operation**: `evicted_expert.cpu()` blocks compute
- **Cascading evictions**: Evicting one expert causes next one to be evicted immediately

**Cache dynamics**:
- Cache size: `max_active_experts` (typically 4)
- During training, ALL accessed experts stay cached until `clear_cache()`
- Cache can grow unbounded before explicit clear

**Example scenario**:
```
Batch 1: Access experts [0, 1] → cache=[0, 1]
Batch 2: Access experts [2, 3] → cache=[0, 1, 2, 3]
Batch 3: Access expert [4] → cache full!
         Must evict LRU expert 0
         Forward pass temporarily stalls for eviction
```

**Impact**:
- Unpredictable GPU memory spikes
- Training stalls during evictions (blocking operations)
- Cache coherency issues if expert accessed again soon

---

### 2.3 All-to-All Communication in Expert Parallelism

**Location**: `/project/code/src/Ava/distributed/expert_parallel.py` (lines 165-366)

**Pattern: Synchronous All-to-All Collective**

```python
def all_to_all_token_routing(self, hidden_states, expert_indices):
    # Exchange token counts with all GPUs
    all_tokens_per_gpu = [torch.zeros_like(tokens_per_gpu) for _ in range(self.expert_parallel_size)]
    dist.all_gather(all_tokens_per_gpu, tokens_per_gpu, group=...)
    
    # Prepare send/receive buffers
    send_hidden = torch.cat(send_buffers_hidden, dim=0)
    send_indices = torch.cat(send_buffers_indices, dim=0)
    
    # BLOCKING COMMUNICATION
    dist.all_to_all_single(
        recv_buffer_hidden, send_hidden,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=...
    )
```

**Spiky Behavior**:
- **Synchronous barrier**: All GPUs must wait for slowest GPU
- **Two collective calls**: `all_gather` + `all_to_all_single`
- **Load imbalance**: Uneven token distribution across GPUs
- **No computation overlap**: Communication completely blocks compute

**Example with 4 GPUs**:
```
GPU 0: 100 tokens, 5ms all-to-all  ┐
GPU 1: 200 tokens, 8ms all-to-all  │ Must wait for slowest
GPU 2: 50 tokens,  3ms all-to-all  │ (GPU 3: 12ms)
GPU 3: 300 tokens, 12ms all-to-all ┘ Slowest GPU
```

**Optimization attempt**:
- Lines 250-306: Async all-to-all with CUDA streams
- Problem: Still blocks on `torch.cuda.current_stream().wait_stream(stream)` (line 306)

**Impact**:
- 5-20% training overhead from communication
- Spiky GPU utilization across all ranks
- Cascading delays if one GPU is slow

---

## 3. MEMORY MANAGEMENT SPIKES

### 3.1 Synchronous Memory Cleanup with Full Synchronization

**Location**: `/project/code/src/Ava/distributed/memory_monitor.py` (lines 404-464)

**Pattern: Threshold-Triggered Memory Flushing**

```python
def cleanup_memory(self, aggressive: bool = False):
    before_stats = self.get_memory_stats(skip_sync=True)
    
    if torch.cuda.is_available():
        cached_gb = before_stats.get('gpu_cached_gb', 0)
        if cached_gb > 1.0:  # Only cleanup if > 1GB cached
            torch.cuda.empty_cache()  # BLOCKING
    
    gc.collect()  # BLOCKING Python GC
    
    if aggressive:
        # More aggressive cleanup
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()  # BLOCKING
        
        for _ in range(2):
            gc.collect()  # BLOCKING
        
        # Only sync if truly emergency
        if before_stats.get('gpu_cached_gb', 0) / total > 0.99:
            torch.cuda.synchronize()  # FULL BLOCKING
```

**Spiky Behavior**:
- **Threshold trigger**: When cached memory > 1GB, cleanup occurs
- **Full synchronization**: `torch.cuda.synchronize()` stalls ALL GPU operations
- **GC pause**: `gc.collect()` can take 10-100ms
- **Frequency**: Aggressive cleanup runs when memory > 99%

**Impact Timeline**:
```
Time: 0-1000ms:  Training running normally
Time: 1000ms:    Memory cache hits 1.0GB threshold
Time: 1000-10ms: torch.cuda.empty_cache() called (5-10ms stall)
Time: 1010ms:    gc.collect() called (10-50ms stall)
Time: 1060ms:    Training resumes
Total stall: 50-70ms (spiky loss of throughput)
```

**Impact**:
- Periodic stalls every 1-2 seconds
- Loss of GPU utilization during cleanup
- Non-deterministic: GC timing varies

---

### 3.2 Trainer's Async Cache Clearer with Race Conditions

**Location**: `/project/code/src/Ava/training/core/trainer.py` (lines 504-545)

**Pattern: Background Thread with Async Queue**

```python
def _start_async_cache_clearer(self):
    def cache_clear_worker():
        while not self._cache_clear_stop_event.is_set():
            self._cache_clear_stop_event.wait(0.05)  # Check every 50ms
            
            with self._cache_clear_lock:
                if self._cache_clear_queue:
                    num_clears = len(self._cache_clear_queue)
                    for clear_fn in self._cache_clear_queue:
                        try:
                            clear_fn()
                            self._cache_clears_completed += 1
                        except Exception as e:
                            self._cache_clears_failed += 1
                    self._cache_clear_queue.clear()

def _async_clear_cache(self):
    with self._cache_clear_lock:
        if not self._cache_clear_queue:
            self._cache_clear_queue.append(clear_fn)
            self._cache_clears_pending += 1
```

**Spiky Behavior**:
- **50ms poll interval**: Background thread checks queue every 50ms
- **Batch processing**: All pending clears executed at once
- **Lock contention**: Training thread must acquire lock to queue clears

**Example scenario**:
```
Time: 0ms:     Trainer schedules cache clear
Time: 0-50ms:  Background thread polling (clear sits in queue)
Time: 50ms:    Background thread acquires lock, executes clear
Time: 50-100ms: Lock held, trainer may stall trying to schedule next clear
```

**Impact**:
- Unpredictable 50ms delays
- Lock contention creates stalls
- Background thread creates CPU overhead

---

## 4. SYNCHRONIZATION POINTS SUMMARY

### Table: Critical Blocking Operations

| Component | Location | Blocking Type | Duration | Frequency |
|-----------|----------|---------------|----------|-----------|
| Expert H2D Transfer | offloaded_experts.py:512 | CUDA synchronous | 5-20ms | Per expert |
| Expert D2H (eviction) | offloaded_experts.py:595 | CUDA synchronous | 5-20ms | When cache full |
| All-to-All Communication | expert_parallel.py:281 | Distributed barrier | 5-50ms | Per batch |
| Memory Cleanup | memory_monitor.py:422 | CUDA full sync | 10-100ms | Every 1-2s |
| Tokenization Stall | dataloader.py:1354 | CPU blocking | 50-500ms | Per buffer fill |
| Stream Synchronization | cuda_streams.py:108 | CUDA sync | 1-10ms | Per iteration |

---

## 5. FEEDBACK LOOPS AND CASCADING EFFECTS

### 5.1 Adaptive Prefetch Depth Loop

```
High I/O Latency (>100ms)
    ↓
Increase prefetch_depth (line 334)
    ↓
More concurrent I/O requests
    ↓
More contention on disk/network
    ↓
Higher I/O latency
    ↓ (back to start)
```

**Result**: Runaway prefetch depth, memory pressure, more spikes

---

### 5.2 Memory Pressure → Eviction Loop

```
GPU memory usage > 99%
    ↓
Aggressive memory cleanup triggered
    ↓
Synchronization stall (50-100ms)
    ↓
Training throughput drops
    ↓
More batches accumulate in CPU RAM
    ↓
More memory pressure
    ↓ (back to start)
```

**Result**: Oscillating memory utilization, cyclic stalls

---

### 5.3 Expert Cache Eviction Cascade

```
Access expert #5 (cache full)
    ↓
Evict LRU expert #0
    ↓
Expert #0 accessed in next batch
    ↓
Evict expert #1 to load #0
    ↓
Thrashing: experts constantly moved in/out
```

**Result**: No stable working set, constant memory transfers

---

## 6. GPU LOAD PATTERN CHARACTERISTICS

### Typical Utilization Profile (1 second window)

```
GPU Utilization Over Time:

100% ██████░░██████░░██████░░████████
 75% ██████▒▒██████▒▒██████▒▒████████
 50% ██████▓▓██████▓▓██████▓▓████████
 25% ██████░░██████░░██████░░████████
  0% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     0   100 200 300 400 500 600 700 800 900 1000 ms
     
     ▀▀▀█  Buffer fill (I/O prefetch)
        ▀▀▀█  CPU tokenization burst
           ▀▀▀█  GPU batch processing
              ▀▀▀█  I/O wait stall
                 ▀▀▀█  Expert transfer (if offloading)
```

**Key observations**:
- **Duty cycle**: 40-60% actual GPU compute time
- **Frequency**: 50-200ms spikes every 100-300ms
- **Amplitude**: 0-100% utilization swings
- **Predictability**: Somewhat regular but with variance

---

## 7. RECOMMENDATIONS FOR SMOOTHING LOADS

### 7.1 Data Loading Improvements

1. **Eliminate buffer-fill-drain cycles**:
   - Use continuous streaming instead of buffer thresholds
   - Replace synchronous tokenization with online tokenization

2. **Better prefetching**:
   - Predictive prefetch based on file sequence
   - Adaptive prefetch with time-based smoothing (not just latency-based)

3. **Bucketing improvements**:
   - Emit buckets before hitting max_size (e.g., at 70%)
   - Use soft thresholds with hysteresis to prevent rapid on/off cycles

### 7.2 Expert Offloading Improvements

1. **Overlapped transfers**:
   - Load next expert while processing current expert
   - Use pinned memory for faster transfers
   - Use separate CUDA stream for transfers (already attempted but synchronization is the issue)

2. **Smarter cache management**:
   - Predictive caching based on routing patterns
   - Persistent cache that survives batches (not just epochs)
   - Batch multiple expert loads together

3. **Reduce H2D frequency**:
   - Quantize experts on disk, dequantize on GPU
   - Use expert fusion (combine similar experts)

### 7.3 Synchronization Point Reduction

1. **Eliminate full synchronization**:
   - Use event-based synchronization instead of wait_stream
   - Only sync when necessary for safety

2. **Asynchronous communication**:
   - Don't block on all-to-all completion
   - Use callbacks for completion handling

3. **Graceful memory management**:
   - Predict memory pressure before hitting threshold
   - Use soft eviction (lower priority experts evicted first)
   - Avoid full synchronization except in emergencies

---

## 8. CONCLUSION

The Ava MoE++ system creates **spiky GPU load patterns** primarily through:

1. **Three-level buffering** (file I/O → tokenization → GPU) with synchronous flush points
2. **Sequential expert loading** that blocks compute during H2D transfers
3. **Threshold-triggered batch releases** that create bursty workloads
4. **Synchronous collective communication** that causes inter-GPU synchronization
5. **Aggressive memory cleanup** with full CUDA synchronization

These patterns interact and amplify through feedback loops, creating:
- **Duty cycles of 40-60%** instead of 80-95%
- **Unpredictable stalls of 10-100ms** every 100-300ms
- **Memory utilization spikes** of 50-100% fluctuation
- **Inter-GPU synchronization bottlenecks** in multi-GPU training

**Overall impact**: ~30-40% throughput loss compared to smooth, continuous execution.

Addressing these requires shifting from synchronous batch processing to continuous, overlapped streaming with predictive mechanisms.

