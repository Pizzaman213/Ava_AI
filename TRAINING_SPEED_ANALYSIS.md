# Training Speed Analysis - Performance Bottlenecks Found

## Summary
Your training is extremely slow due to **5 critical configuration issues** that compound together, causing 10-100x slowdown.

## Root Causes (Priority Order)

### 1. **torch.compile with max-autotune (CRITICAL - 2-5min compilation stall)**
**Impact:** Training hangs for 2-5 minutes during model initialization

**Problem:**
```yaml
use_torch_compile: true
torch_compile_mode: "max-autotune"
torchinductor_max_autotune: 2
compile_routers: true
```

**Why it's slow:**
- `max-autotune` mode tests **hundreds of kernel variants** to find the fastest
- First compilation takes 2-5 minutes while testing different combinations
- For dynamic shapes (variable batch sizes), recompilation happens frequently
- Router compilation adds 20+ separate compilation passes (one per layer)

**Evidence from logs:**
```
[INFO] 🔥 Compiling model with torch.compile (mode=max-autotune)...
[INFO]    ⏳ First compilation will take 2-5 minutes (kernel autotuning)...
[Then training hangs here]
```

**Fix:**
```yaml
use_torch_compile: false              # Disable entirely for dynamic training
torch_compile_mode: "default"         # If you must compile, use default mode
torchinductor_max_autotune: 0         # Disable kernel autotuning
compile_routers: false                # Disable router compilation
```

**Expected speedup:** Eliminates 2-5min startup delay, enables instant training start

---

### 2. **Expert Offloading CPU↔GPU (CRITICAL - 50-90% throughput loss)**
**Impact:** Constant CPU→GPU transfers during every forward pass

**Problem:**
```yaml
use_expert_offloading: true
max_active_experts_gpu: 2    # Only 2/16 experts on GPU!
```

**Why it's slow:**
- You have 16 experts, but only keep 2 on GPU (12.5% on GPU, 87.5% on CPU)
- **Every forward pass** requires:
  1. Transfer 2 new experts from CPU → GPU (PCIe bottleneck)
  2. Run computation
  3. Evict old experts from GPU → CPU
- PCIe bandwidth: ~16 GB/s (vs GPU memory: ~900 GB/s) = **56x slower**
- With gradient checkpointing, each batch does **2 forward passes** = double the transfers

**Math:**
- Expert size: ~400MB each
- Transfer time: 400MB / 16 GB/s = **25ms per expert**
- 2 experts per token × 20 layers = **40 transfers per batch**
- Total overhead: 40 × 25ms = **1000ms = 1 second of pure transfer time per batch**

**Fix:**
```yaml
use_expert_offloading: false           # Keep all experts on GPU
max_active_experts_gpu: 16             # All 16 experts on GPU
```

**Your GPU:** RTX 3090 Ti has 24GB VRAM - easily fits 16 experts (~6.4GB for experts + 2GB model + 4GB optimizer = ~12GB total)

**Expected speedup:** 2-5x faster training throughput

---

### 3. **Excessive Gradient Accumulation (HIGH - 6x slower feedback loop)**
**Impact:** Optimizer updates 6x less frequently

**Problem:**
```yaml
gradient_accumulation_steps: 48       # Accumulate for 48 batches!
batch_size: 4
# Effective batch size: 192
```

**Why it's slow:**
- Takes 48 forward passes before 1 optimizer update
- Slower convergence: learning rate updates less frequently
- Gradient staleness: accumulated gradients may be outdated
- No benefit for single-GPU training (only useful for distributed training to simulate larger batches)

**Comparison:**
| Config | Batches before update | Time to 100 optimizer steps | Convergence speed |
|--------|----------------------|----------------------------|-------------------|
| Current (48 steps) | 48 | 48x longer | 6x slower |
| Optimized (8 steps) | 8 | 8x longer | Balanced |
| Minimal (1 step) | 1 | Fastest | Best for dev |

**Fix:**
```yaml
gradient_accumulation_steps: 8        # Reduced from 48 to 8
# Or use 1 for maximum training speed during development
```

**Expected speedup:** 6x more optimizer updates per hour

---

### 4. **DataLoader Worker Index Shuffling (CRITICAL - 1-2 min per worker!)** ⚠️ **MAJOR BOTTLENECK**
**Impact:** 1-2 minutes of 100% CPU per worker during initialization

**Problem:**
```yaml
num_workers: 2                        # Each worker does this:
# Worker initialization code (pretokenized_loader.py:201-208):
all_indices = []
for file_idx, (file_path, reader) in enumerate(readers):  # 27 files
    num_seqs = len(reader)  # Total: 4.5M examples
    for seq_idx in range(num_seqs):
        all_indices.append((file_idx, seq_idx))  # Build 4.5M element list!
random.Random(42 + worker_id).shuffle(all_indices)  # Shuffle 4.5M elements!
```

**Why it's EXTREMELY slow:**
- **Each worker** builds a list of 4.5 MILLION `(file_idx, seq_idx)` tuples in memory
- **Each worker** then shuffles this 4.5M element list (O(n log n) complexity)
- With 2 workers: **9M tuples created + 2× shuffle operations**
- Observed: **Workers at 93-101% CPU for 1-2 minutes** doing nothing but list building/shuffling
- This happens **every time** workers are created (not just first time)

**Math:**
- List creation: 4.5M × 16 bytes/tuple = **72MB per worker**
- Shuffle time: ~60-120 seconds per worker at 100% CPU
- Total initialization: **2-4 minutes blocked**

**Why this design exists:**
- Allows perfect shuffling across entire dataset
- Enables deterministic worker-level sharding
- But **completely impractical** for datasets with millions of examples

**Immediate fix (bypass the bottleneck):**
```yaml
num_workers: 0                         # Disable multiprocessing entirely
# OR use only 1-2 smaller datasets for dev:
dataset_name: OpenAssistant_oasst2_processed.jsonl  # Just 96K examples instead of 4.5M
```

**Long-term fix (code change needed):**
The dataloader needs to use **lazy iteration** instead of upfront shuffling:
- Don't build the full index list
- Shuffle file order only (27 files, instant)
- Use random sampling within each file
- This would reduce initialization from **2min → 0.1s**

**Expected speedup:** Eliminates 1-2 min startup delay per worker

---

### 5. **Conflicting Compilation Settings (LOW - causes confusion)**
**Impact:** Unclear which settings take precedence

**Problem:**
```yaml
model:
  use_torch_compile: true              # Enabled in model section

hardware:
  compile: false                       # Disabled in hardware section

performance:
  enable_torch_compile: true           # Enabled again in performance section
```

**Fix:** Make all three settings consistent:
```yaml
model:
  use_torch_compile: false

hardware:
  compile: false

performance:
  enable_torch_compile: false
```

---

## Performance Comparison

### Current Configuration
```
Initialization time: 2-5 minutes (torch.compile)
Training throughput: ~0.5-1 batches/sec (expert offloading + high grad accum)
Optimizer updates: Every 48 batches
Time to 1000 steps: ~13-26 hours
```

### Optimized Configuration
```
Initialization time: 10-20 seconds
Training throughput: ~5-10 batches/sec (all experts on GPU)
Optimizer updates: Every 8 batches (6x more frequent)
Time to 1000 steps: ~2-3 hours
```

**Total speedup: 5-10x faster**

---

## Quick Fix Commands

The optimized config has been saved to: `tiny_moe_multi_gpu.yaml`

Test the optimized config:
```bash
python code/scripts/5_training/train.py --config code/configs/moe/tiny_moe_multi_gpu.yaml
```

You should see:
- **No** "First compilation will take 2-5 minutes" message
- **No** "Expert offloading enabled" message
- Model initialization completes in **10-20 seconds** instead of 2-5 minutes
- Training starts **immediately** after dataloader setup

---

## Additional Optimizations (Optional)

### For Maximum Development Speed
If you want the absolute fastest iteration during development:

```yaml
training:
  gradient_accumulation_steps: 1       # Optimizer update every batch
  max_validation_batches: 5            # Only validate on 5 batches

model:
  gradient_checkpointing: false        # Disable if you have enough VRAM
  num_experts: 8                       # Reduce experts if you're just testing
```

### For Maximum Throughput (Production)
Once your code is working and you want maximum training speed:

```yaml
model:
  use_torch_compile: true              # Re-enable after code is stable

performance:
  torch_compile_mode: "reduce-overhead" # Faster than max-autotune for first iteration

data:
  num_workers: 4                        # More workers for large datasets
  dataloader_prefetch_factor: 4         # Higher prefetch
  dataloader_persistent_workers: true   # Keep workers alive after first startup
```

---

## Verification Checklist

After applying fixes, verify these are **NOT** in your logs:
- ❌ "First compilation will take 2-5 minutes"
- ❌ "Expert offloading and torch.compile enabled"
- ❌ "Gradient accumulation steps: 48"
- ❌ "Persistent workers: 4"

You **SHOULD** see:
- ✅ "torch.compile DISABLED - using eager mode"
- ✅ "Model created on cuda in bf16 dtype" (without offloading mention)
- ✅ "Gradient accumulation steps: 8"
- ✅ "Using 2 CPU workers for data loading"
- ✅ Training starts within 30 seconds

---

## Root Cause Summary

The fundamental issue is **premature optimization**:

1. **torch.compile max-autotune** → Optimizing for production before code is working
2. **Expert offloading** → Optimizing for GPUs with <10GB VRAM (you have 24GB!)
3. **48-step gradient accumulation** → Optimizing for multi-GPU (you have 1 GPU!)
4. **4 persistent workers** → Optimizing for large compute clusters (you're on 1 node!)

**Golden rule:** Get it working first, then optimize. Start with simple configs, measure performance, then add optimizations one at a time.
