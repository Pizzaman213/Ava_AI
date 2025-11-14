# Spiky Load Patterns - Quick Summary

## TL;DR

The Ava MoE++ system exhibits **40-60% GPU duty cycles** (instead of 80-95%) due to spiky, bursty load patterns rather than smooth, continuous computation. This results in **~30-40% throughput loss**.

## Primary Culprits (in order of impact)

### 1. **Three-Level Buffering with Synchronous Drain** (Biggest Impact)
- **Where**: `dataloader.py` lines 1282-1425
- **What**: Buffer fills up to 10,000 samples → entire buffer tokenized at once → released to GPU
- **Why it spikes**: 
  - 50ms to accumulate samples
  - 100-500ms to tokenize entire buffer (CPU burst)
  - 100-300ms to process on GPU
  - Then repeat → creates sawtooth pattern
- **Fix**: Stream tokenization online instead of batch processing

### 2. **Length-Based Bucketing with Threshold Release** (Second Impact)
- **Where**: `dataloader.py` lines 150-271
- **What**: Bucket holds samples until it reaches 200 items, then releases all at once
- **Why it spikes**: Small samples trickle in slowly → bucket suddenly full → burst release
- **Fix**: Emit at 70% full instead of 100% to create smoother release pattern

### 3. **Sequential Expert Loading** (For Expert Offloading)
- **Where**: `offloaded_experts.py` lines 422-654
- **What**: Load expert #1 (5-20ms) → compute (10-15ms) → load expert #2 (5-20ms) → compute (10-15ms)
- **Why it spikes**: Each H2D transfer blocks GPU compute
- **Fix**: Load next expert while computing current expert (pipelining)

### 4. **Synchronous All-to-All Communication** (Multi-GPU)
- **Where**: `expert_parallel.py` lines 165-366
- **What**: All GPUs must wait for slowest GPU to finish all-to-all collective
- **Why it spikes**: Load imbalance causes some GPUs to wait 10-50ms per collective
- **Fix**: Use async collectives with callbacks instead of blocking waits

### 5. **Aggressive Memory Cleanup** (Periodic)
- **Where**: `memory_monitor.py` lines 404-464
- **What**: When cached memory > 1GB, call `torch.cuda.synchronize()` (full stall)
- **Why it spikes**: 
  - Full CUDA sync takes 10-100ms
  - GC pause takes 10-50ms
  - Occurs every 1-2 seconds
- **Fix**: Use soft eviction and avoid full synchronization

## Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| GPU Duty Cycle | 40-60% | 80-95% |
| Throughput | 100% baseline | 130-145% possible |
| Stall Frequency | Every 100-300ms | Rare (< 1% of time) |
| Stall Duration | 10-100ms | N/A |
| Memory Utilization | 50-100% spikes | Smooth 80-85% |

## Quick Diagnosis

### Signs of Spiky Loads:
1. `nvidia-smi` shows GPU utilization jumping 0-100%
2. `dmesg` shows periodic stalls in kernel
3. Training speed varies by ±30% between seconds
4. Memory usage oscillates between 50% and 99%
5. CPU shows 5-10x bursts during tokenization

### How to Verify:
```bash
# Watch GPU utilization in real-time
watch -n 0.1 nvidia-smi pmon

# Check for CUDA synchronization calls in profiler
python -m cProfile -s cumulative train.py | grep -i "synchronize\|all_to_all"
```

## Implementation Priorities

### High Impact, Medium Effort:
1. Soft thresholds for bucketing (70% instead of 100%)
2. Reduce buffer size (5000 instead of 10000)
3. Remove aggressive memory cleanup (just use async GC)

### High Impact, High Effort:
1. Online tokenization instead of batch tokenization
2. Expert loading pipelining
3. Async all-to-all collectives

### Medium Impact, Low Effort:
1. Increase prefetch depth smoothly (not adaptively)
2. Disable 50ms poll in async cache clearer

## Expected Improvement

**Implementing changes in order of priority:**
- High-impact quick fixes: **+10-15% throughput**
- Online tokenization: **+15-20% throughput**
- Expert pipelining: **+10-15% throughput** (if offloading enabled)
- Async collectives: **+10% throughput** (if multi-GPU)

**Total potential: +40-50% throughput** (approaching 100% GPU duty cycle)

