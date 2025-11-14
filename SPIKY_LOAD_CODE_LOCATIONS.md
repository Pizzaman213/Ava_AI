# Spiky Load Pattern - Code Locations and Fix Targets

## File Structure Overview

```
code/src/Ava/
├── data/
│   ├── dataloader.py          ← PRIMARY: Buffer, bucketing, tokenization
│   ├── pretokenized_loader.py
│   └── multi_column_data.py
├── layers/
│   ├── offloaded_experts.py   ← SECONDARY: Expert H2D transfers, cache eviction
│   ├── experts.py
│   └── routing.py
├── distributed/
│   ├── expert_parallel.py     ← TERTIARY: All-to-all communication
│   ├── gpu_load_balancer.py
│   └── memory_monitor.py      ← QUATERNARY: Memory cleanup stalls
├── training/
│   └── core/trainer.py        ← QUINARY: Async cache clearer
└── utils/
    └── cuda_streams.py        ← Stream management
```

---

## Issue #1: Three-Level Buffering with Synchronous Drain

### Location
`/project/code/src/Ava/data/dataloader.py` (lines 1282-1425)

### Key Lines

**Line 1299**: Buffer created with 10,000 sample capacity
```python
buffer = deque(maxlen=dynamic_buffer_size)
```

**Lines 1343-1352**: Synchronous drain when buffer full
```python
if len(buffer) >= self.buffer_size:
    buffer_list = list(buffer)
    buffer_seed = 42 + epoch_number + (samples_processed // self.buffer_size)
    rng = random.Random(buffer_seed)
    rng.shuffle(buffer_list)
    # ... tokenization follows
```

**Lines 1354-1381**: Batch tokenization (CPU burst)
```python
tokenized_samples = []
if text_batch:
    current_max_length = self._get_current_max_length()
    min_tokenize_batch = DATA_CONSTANTS.MIN_TOKENIZE_BATCH
    
    if len(text_batch) < min_tokenize_batch:
        tokenized_samples = self._tokenize_batch(text_batch, current_max_length)
    else:
        for i in range(0, len(text_batch), min_tokenize_batch):
            chunk = text_batch[i:i + min_tokenize_batch]
            tokenized_samples.extend(self._tokenize_batch(chunk, current_max_length))
```

### Root Cause
- Buffer size: 10,000 samples (lines 552-584 in StreamingDataset)
- Entire buffer tokenized at once when full
- No streaming or pipelining

### Fix Strategy
1. **Reduce buffer size**: 10,000 → 1,000-2,000
2. **Enable streaming tokenization**: Line 568 has `use_streaming_tokenization` parameter
3. **Implement rolling tokenization**: Tokenize as samples arrive, not in bursts

### Implementation Priority
**HIGH (Biggest impact)**
- Effort: Medium (requires tokenization refactoring)
- Payoff: +15-20% throughput
- Risk: Low (backward compatible)

---

## Issue #2: Length-Based Bucketing with Threshold Release

### Location
`/project/code/src/Ava/data/dataloader.py` (lines 150-271)

### Key Lines

**Lines 220-228**: Bucketing logic with hard threshold
```python
self.buckets[bucket_id].append(sample)
self.bucket_stats[bucket_id] += 1
self._stats_dirty = True

# Return full bucket if threshold reached (OPTIMIZED: zero-copy)
if len(self.buckets[bucket_id]) >= self.max_bucket_size:
    full_bucket = self.buckets[bucket_id]
    self.buckets[bucket_id] = []
    return full_bucket
```

**Lines 168-169**: Bucket size default
```python
self.max_bucket_size = max_bucket_size or DATA_CONSTANTS.MAX_BUCKET_SIZE
self.min_bucket_size = min_bucket_size or DATA_CONSTANTS.MIN_BUCKET_SIZE
```

### Root Cause
- Hard threshold at 200 samples (max_bucket_size)
- Releases everything at once when threshold hit
- No soft thresholds or hysteresis

### Fix Strategy
1. **Soft threshold**: Emit at 70% of max_bucket_size
2. **Hysteresis**: Once below 50%, start accumulating again
3. **Time-based flushing**: Flush every 100ms regardless of count

### Implementation Priority
**HIGH-MEDIUM (Quick win)**
- Effort: Low (simple threshold change)
- Payoff: +5-10% throughput
- Risk: Very low
- Location: Lines 224-228

### Code Change Example
```python
# OLD: Hard threshold
if len(self.buckets[bucket_id]) >= self.max_bucket_size:
    return full_bucket

# NEW: Soft threshold with hysteresis
soft_threshold = int(self.max_bucket_size * 0.7)
if len(self.buckets[bucket_id]) >= soft_threshold:
    if not hasattr(self, '_bucket_emission_cooldown'):
        self._bucket_emission_cooldown = {}
    
    current_time = time.time()
    last_emission = self._bucket_emission_cooldown.get(bucket_id, 0)
    
    # Emit if 70% full OR if 100ms since last emission
    if (len(self.buckets[bucket_id]) >= soft_threshold and 
        current_time - last_emission > 0.1):
        self._bucket_emission_cooldown[bucket_id] = current_time
        return full_bucket
```

---

## Issue #3: Sequential Expert Loading (Expert Offloading)

### Location
`/project/code/src/Ava/layers/offloaded_experts.py` (lines 422-654)

### Key Lines

**Line 467**: Sequential iteration over experts
```python
unique_experts = torch.unique(expert_indices.flatten()).tolist()
```

**Lines 475-519**: Sequential loading and compute
```python
for idx, expert_id in enumerate(unique_experts):
    # ... find tokens using this expert
    
    if not already_on_device:
        expert.to(device)  # LINE 512: BLOCKING H2D TRANSFER
        self._experts_on_gpu.add(expert_id)
    
    # ... compute with expert
```

**Lines 520-542**: Async prefetch attempt (insufficient)
```python
if prefetch_stream is not None and self.prefetch_lookahead > 0:
    for lookahead_idx in range(1, min(self.prefetch_lookahead + 1, ...)):
        next_expert_id = unique_experts[idx + lookahead_idx]
        # ... async prefetch to GPU
```

**Lines 615-625**: Stream sync (still blocking)
```python
if prefetch_stream is not None:
    torch.cuda.current_stream().wait_stream(prefetch_stream)

if self.prefetch_lookahead > 0 and len(unique_experts) > 1:
    num_streams_used = min(self._current_prefetch_depth, len(self._prefetch_streams))
    for i in range(num_streams_used):
        torch.cuda.current_stream().wait_stream(self._prefetch_streams[i])
```

### Root Cause
- **Line 512**: Synchronous `expert.to(device)` blocks all compute
- **Lines 615-625**: Wait for prefetch still blocks main compute stream
- No true pipelining: prefetch helps only NEXT expert, not CURRENT

### Fix Strategy
1. **Move expert to default stream BEFORE forward pass**
2. **Load next expert while computing current (true pipelining)**
3. **Use callback for H2D completion instead of wait_stream**

### Implementation Priority
**HIGH (If using expert offloading)**
- Effort: High (requires refactoring forward pass logic)
- Payoff: +10-15% throughput (if offloading enabled)
- Risk: Medium (complex synchronization changes)
- Prerequisite: Check if `use_lora` or CPU offloading enabled

### Key Changes Needed

**Change 1: Pre-load first expert** (before loop)
```python
# Pre-load first expert before entering main loop
if len(unique_experts) > 0:
    first_expert = self.experts[unique_experts[0]]
    if device.type == 'cuda' and torch.cuda.is_available():
        first_expert.to(device)
        self._experts_on_gpu.add(unique_experts[0])
```

**Change 2: Load next expert while computing current**
```python
for idx, expert_id in enumerate(unique_experts):
    # ... existing compute code ...
    
    # AFTER compute, prefetch NEXT expert (if exists)
    if idx + 1 < len(unique_experts):
        next_expert_id = unique_experts[idx + 1]
        next_expert = self.experts[next_expert_id]
        
        # Load next expert asynchronously
        with torch.cuda.stream(self._prefetch_streams[0]):
            next_expert.to(device, non_blocking=True)
            self._experts_on_gpu.add(next_expert_id)
        
        # NO WAIT HERE - let it overlap with loop iterations
```

---

## Issue #4: Synchronous All-to-All Communication

### Location
`/project/code/src/Ava/distributed/expert_parallel.py` (lines 165-366)

### Key Lines

**Lines 229-235**: All-gather for token counts
```python
all_tokens_per_gpu = [
    torch.zeros_like(tokens_per_gpu) for _ in range(self.expert_parallel_size)
]
dist.all_gather(all_tokens_per_gpu, tokens_per_gpu, group=self.expert_parallel_group)
```

**Lines 281-294**: Blocking all-to-all communication
```python
dist.all_to_all_single(
    recv_buffer_hidden, send_hidden,
    output_split_sizes=recv_sizes,
    input_split_sizes=send_sizes,
    group=self.expert_parallel_group
)

dist.all_to_all_single(
    recv_buffer_indices, send_indices,
    output_split_sizes=recv_sizes,
    input_split_sizes=send_sizes,
    group=self.expert_parallel_group
)
```

**Lines 300-306**: Blocking wait
```python
if len(self._comm_events) > 0:
    self._comm_events[0].wait()
else:
    torch.cuda.current_stream().wait_stream(stream)
```

### Root Cause
- **All-gather barrier**: All GPUs wait for slowest
- **All-to-all barrier**: All GPUs wait for slowest
- **Blocking wait**: Main compute stream blocked until complete
- **Load imbalance**: Different GPUs have different token counts

### Fix Strategy
1. **Use non-blocking all-to-all** (if available in PyTorch)
2. **Overlap communication with computation**
3. **Load balance token distribution** before communication

### Implementation Priority
**MEDIUM-HIGH (Multi-GPU only)**
- Effort: High (distributed training complexity)
- Payoff: +10-20% throughput (multi-GPU scenarios)
- Risk: High (distributed synchronization is tricky)
- Only relevant if `expert_parallel_size > 1`

### Code Sketch
```python
# Use work objects for non-blocking collectives
work_hidden = dist.all_to_all_single(
    recv_buffer_hidden, send_hidden,
    output_split_sizes=recv_sizes,
    input_split_sizes=send_sizes,
    group=self.expert_parallel_group,
    async_op=True  # NEW: Non-blocking
)

# Do other work while communication happens
# ... prepare next batch ...

# Only wait when needed
work_hidden.wait()  # Wait for all-to-all to complete
```

---

## Issue #5: Aggressive Memory Cleanup with Full Sync

### Location
`/project/code/src/Ava/distributed/memory_monitor.py` (lines 404-464)

### Key Lines

**Lines 419-422**: Cache cleanup threshold
```python
if torch.cuda.is_available():
    cached_gb = before_stats.get('gpu_cached_gb', 0)
    if cached_gb > 1.0:  # Only cleanup if > 1GB cached
        torch.cuda.empty_cache()  # BLOCKING
```

**Lines 434-444**: Aggressive cleanup with full sync
```python
for _ in range(2):
    gc.collect()  # BLOCKING GC

if torch.cuda.is_available():
    # Only sync if we're in true emergency (>99% memory usage)
    if before_stats.get('gpu_cached_gb', 0) / total > 0.99:
        torch.cuda.synchronize()  # FULL BLOCKING SYNC
    torch.cuda.empty_cache()
```

### Root Cause
- **Line 422**: `torch.cuda.empty_cache()` is a full synchronization
- **Line 435-436**: Multiple `gc.collect()` calls block execution
- **Line 443**: Full `torch.cuda.synchronize()` stalls all GPU operations
- **Frequency**: Every 1-2 seconds when memory > 1GB

### Fix Strategy
1. **Remove full synchronization**: Use async GC only
2. **Increase threshold**: 1.0GB → 2.0GB or higher
3. **Use soft eviction**: Don't clear cache, just mark for lazy eviction

### Implementation Priority
**MEDIUM (Quick win)**
- Effort: Low (just remove sync calls)
- Payoff: +3-5% throughput reduction in stall overhead
- Risk: Very low (memory is already managed elsewhere)

### Code Change Example
```python
def cleanup_memory(self, aggressive: bool = False):
    before_stats = self.get_memory_stats(skip_sync=True)
    
    # REMOVE: Synchronous empty_cache call
    # if cached_gb > 1.0:
    #     torch.cuda.empty_cache()
    
    # REPLACE WITH: Async garbage collection only
    gc.collect()  # Single call, not blocking GPU
    
    if aggressive:
        # REMOVE: Full synchronization
        # if before_stats.get('gpu_cached_gb', 0) / total > 0.99:
        #     torch.cuda.synchronize()
        
        # REPLACE WITH: Async-only cleanup
        for _ in range(2):
            gc.collect()
        
        # Empty cache without sync (PyTorch handles this async now)
        if torch.cuda.is_available():
            # This is now mostly async in modern PyTorch
            torch.cuda.empty_cache()
```

---

## Issue #6: Trainer's Async Cache Clearer (Secondary)

### Location
`/project/code/src/Ava/training/core/trainer.py` (lines 504-545)

### Key Lines

**Line 510**: 50ms poll interval (too frequent)
```python
self._cache_clear_stop_event.wait(0.05)  # 50ms
```

**Lines 514-524**: Batch processing with lock
```python
with self._cache_clear_lock:
    if self._cache_clear_queue:
        num_clears = len(self._cache_clear_queue)
        for clear_fn in self._cache_clear_queue:
            # ... execute clears ...
```

### Root Cause
- **Line 510**: 50ms poll creates unpredictable 0-50ms delays
- **Lock contention**: Training thread must wait for clearer thread
- **Batch execution**: All pending clears execute together, causing stalls

### Fix Strategy
1. **Increase poll interval**: 50ms → 200-500ms
2. **Use event signaling**: Don't poll, use threading event
3. **Execute clears asynchronously**: Don't block main thread

### Implementation Priority
**LOW-MEDIUM (Nice to have)**
- Effort: Low (simple parameter changes)
- Payoff: +1-2% reduction in random stalls
- Risk: Very low
- Impact: Mainly affects variance, not mean

### Code Change Example
```python
def _start_async_cache_clearer(self):
    def cache_clear_worker():
        while not self._cache_clear_stop_event.is_set():
            # OLD: Poll every 50ms
            # self._cache_clear_stop_event.wait(0.05)
            
            # NEW: Wait for signal or timeout
            self._cache_clear_stop_event.wait(0.5)  # 500ms timeout
            
            with self._cache_clear_lock:
                if self._cache_clear_queue:
                    # ... execute clears ...
```

---

## Summary Table

| Issue | File | Lines | Priority | Effort | Payoff | Risk |
|-------|------|-------|----------|--------|--------|------|
| Buffer/drain | dataloader.py | 1282-1425 | HIGH | Medium | +15-20% | Low |
| Bucketing | dataloader.py | 150-271 | HIGH | Low | +5-10% | Very Low |
| Expert loading | offloaded_experts.py | 422-654 | HIGH* | High | +10-15% | Medium |
| All-to-all | expert_parallel.py | 165-366 | MEDIUM* | High | +10-20% | High |
| Memory cleanup | memory_monitor.py | 404-464 | MEDIUM | Low | +3-5% | Very Low |
| Async clearer | trainer.py | 504-545 | LOW | Low | +1-2% | Very Low |

*Only if feature is enabled (expert offloading, multi-GPU)

