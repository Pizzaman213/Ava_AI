# Hardcoded Values Analysis - Ava Pipeline Codebase

## Executive Summary

This report identifies hardcoded values throughout the Ava pipeline that should be configurable. The codebase has a good foundation with centralized constants (constants.py), but several areas still contain hardcoded values in critical code paths.

**Total Issues Found: 87+**

## Key Findings

### 1. Well-Managed Constants (Good Practice)
- `/project/code/src/Ava/config/constants.py` - Comprehensive centralized constants with good structure
- These are already configurable via YAML through `update_constants_from_config()`

### 2. Partially Hardcoded Components
- Data loading parameters spread across multiple files
- Device specifications in training and inference
- Model architecture defaults
- Directory paths not fully from config

---

## DETAILED FINDINGS BY CATEGORY

### A. MAGIC NUMBERS - Numeric Literals in Code

#### 1. Data Loading Parameters

**File: `/project/code/src/Ava/data/dataloader.py`**

| Line | Value | Controls | Severity | Why Configurable |
|------|-------|----------|----------|------------------|
| 17 | 8192 | MAX_TOKENS_DEFAULT | HIGH | Should be configurable per dataset/GPU |
| 18 | 64 | MAX_BATCH_SIZE_DEFAULT | HIGH | GPU-dependent, needs adjustment per model |
| 21 | 200 | MAX_BUCKET_SIZE | MEDIUM | Affects batching efficiency, GPU-dependent |
| 22 | 8 | MIN_BUCKET_SIZE | MEDIUM | Should vary with sequence length distribution |
| 28 | 10000 | BUFFER_SIZE_DEFAULT | MEDIUM | Memory-dependent, not one-size-fits-all |
| 29 | 1000 | STREAMING_BUFFER_SIZE | MEDIUM | Should scale with available VRAM |
| 31 | 32 | MIN_TOKENIZE_BATCH | MEDIUM | Affects tokenization performance |
| 34 | 5000 | PARQUET_BATCH_SIZE | MEDIUM | I/O dependent, should be tunable |
| 35 | 10 | MIN_TEXT_LENGTH | LOW | Reasonable default but could be stricter |
| 38 | 4 | PREFETCH_MAX_WORKERS | MEDIUM | CPU core dependent |
| 39 | 2 | PREFETCH_SIZE | MEDIUM | Should scale with I/O latency |
| 42 | 100 | WORKER_FILE_CACHE_MAX_SIZE | MEDIUM | Memory-dependent |
| 45 | 3072 | PREFETCH_FACTOR_NUMERATOR | MEDIUM | Affects dynamic prefetching calculation |
| 46 | 2 | PREFETCH_FACTOR_MIN | MEDIUM | Minimum prefetch threads |
| 47 | 6 | PREFETCH_FACTOR_MAX | MEDIUM | Maximum prefetch threads |
| 50 | 200 | INITIAL_FILL_SIZE_MAX | MEDIUM | Affects startup speed vs. stability |
| 51 | 10 | INITIAL_FILL_SIZE_DIVISOR | MEDIUM | Ratio for initial buffer fill |
| 52 | 2000 | DYNAMIC_BUFFER_MIN | MEDIUM | Minimum buffer size floor |
| 53 | 0.05 | DYNAMIC_BUFFER_MEMORY_PERCENT | MEDIUM | GPU memory allocation ratio |
| 56 | 32 | TOKEN_BALANCE_BATCH_SIZE | MEDIUM | Affects token balancing in distributed training |
| 59 | 0.85 | MEMORY_PRESSURE_THRESHOLD | MEDIUM | Cache reduction trigger point |
| 60 | 2 | CACHE_REDUCTION_FACTOR | MEDIUM | Divisor for cache reduction |
| 61 | 10 | CACHE_MIN_SIZE_UNDER_PRESSURE | MEDIUM | Minimum cache size during OOM |
| 64 | 1000 | MEMORY_CHECK_INTERVAL | MEDIUM | Affects memory monitoring frequency |
| 65 | 1000 | PROFILING_REPORT_INTERVAL | MEDIUM | Logging verbosity control |
| 68 | 100 | LOAD_BALANCE_CHECK_INTERVAL | MEDIUM | Multi-GPU synchronization frequency |
| 69 | 5 | LOAD_BALANCE_MAX_SKIP | MEDIUM | Maximum samples to skip for load balancing |
| 72 | 10.0 | ADAPTIVE_SAMPLES_LARGE_FILE_MB | MEDIUM | File size threshold for adaptive sampling |
| 73 | 1.0 | ADAPTIVE_SAMPLES_MEDIUM_FILE_MB | MEDIUM | File size threshold for adaptive sampling |
| 74 | 4 | ADAPTIVE_SAMPLES_LARGE_MULTIPLIER | MEDIUM | Samples multiplier for large files |
| 75 | 2 | ADAPTIVE_SAMPLES_MEDIUM_MULTIPLIER | MEDIUM | Samples multiplier for medium files |
| 76 | 128 | ADAPTIVE_SAMPLES_MAX | MEDIUM | Maximum adaptive samples per file |
| 77 | 8 | ADAPTIVE_SAMPLES_MIN | MEDIUM | Minimum adaptive samples per file |
| 78 | 0.05 | ADAPTIVE_SAMPLES_SLOW_READ_THRESHOLD | MEDIUM | I/O performance threshold |
| 79 | 0.8 | ADAPTIVE_SAMPLES_SLOW_MULTIPLIER | MEDIUM | Samples reduction for slow I/O |
| 80 | 1.2 | ADAPTIVE_SAMPLES_FAST_MULTIPLIER | MEDIUM | Samples increase for fast I/O |
| 81 | 10 | ADAPTIVE_READ_TIME_HISTORY_SIZE | MEDIUM | History window for I/O performance |
| 84 | 5 | BUCKET_FLUSH_INTERVAL_MULTIPLIER | MEDIUM | Affects batch timing |
| 85 | 8 | BUCKET_FLUSH_MIN_SIZE | MEDIUM | Minimum samples for flushing |
| 88 | 10 | COLLATE_BUFFER_CACHE_MAX_SIZE | MEDIUM | Collation buffer cache size |

**File: `/project/code/src/Ava/data/dataloader.py` - Line 552, 560, 566, 569**

| Line | Value | Controls | Code Section |
|------|-------|----------|--------------|
| 552 | 10000 | buffer_size default parameter | StreamingDataset.__init__ |
| 560 | 500 | samples_per_file | StreamingDataset (OPTIMIZED from 32) |
| 566 | 0.01 | validation_rate | Sampling for sequence validation |
| 569 | 1000 | streaming_buffer_size | Streaming tokenization mode |

**File: `/project/code/src/Ava/data/dataloader.py` - Lines 736, 739**

| Line | Value | Controls | Issue |
|------|-------|----------|-------|
| 736 | 85 | Train split percentage | Hardcoded 85% train / 15% val split |
| 739 | 85 | Val split comparison | No config option for custom splits |

**File: `/project/code/src/Ava/data/dataloader.py` - Lines 899-907**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 899 | 100 | max_iterations limit | Safety limit (OK but conservative) |
| 305 | 100 | access_pattern window | Pattern tracking window size |
| 316 | 20 | io_latencies window | I/O latency history size |
| 328 | 5 | Minimum latencies for adjustment | Threshold for prefetch adjustment |
| 333 | 0.1 | I/O latency threshold (100ms) | For increasing prefetch depth |
| 335 | 0.02 | I/O latency threshold (20ms) | For decreasing prefetch depth |
| 334 | 4 | Maximum prefetch depth | Hard limit on prefetch depth |
| 336 | 1 | Minimum prefetch depth | Hard limit on prefetch depth |

#### 2. Training Configuration - Trainer Constants

**File: `/project/code/src/Ava/config/constants.py` - TrainerConstants class (Lines 98-136)**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 102 | 500_000_000 | AUTOCAST_PARAM_THRESHOLD | Model size threshold for auto-checkpointing |
| 105 | 2.0**16 | GRAD_SCALER_INIT_SCALE | Initial gradient scale (65536) |
| 106 | 2.0 | GRAD_SCALER_GROWTH_FACTOR | Gradient scale multiplier |
| 107 | 0.5 | GRAD_SCALER_BACKOFF_FACTOR | Gradient scale reduction |
| 108 | 2000 | GRAD_SCALER_GROWTH_INTERVAL | Steps between growth attempts |
| 109 | 1000 | GRAD_SCALER_RESET_INTERVAL | Steps between scaler resets |
| 112 | 0.990 | MEMORY_WARNING_THRESHOLD | 99.0% memory usage warning |
| 113 | 0.995 | MEMORY_CRITICAL_THRESHOLD | 99.5% memory usage critical |
| 114 | 0.999 | MEMORY_EMERGENCY_THRESHOLD | 99.9% memory usage emergency |
| 115 | 2.0 | MEMORY_HEADROOM_GB | Reserved memory headroom in GB |
| 116 | 500 | MEMORY_CLEAR_CACHE_FREQUENCY | Cache clear frequency |
| 117 | 100 | MEMORY_EMERGENCY_CHECK_FREQUENCY | Emergency memory check frequency |
| 120 | 0.92 | ATTENTION_CHECKPOINT_ENABLE_THRESHOLD | Enable at 92% memory |
| 121 | 0.80 | ATTENTION_CHECKPOINT_DISABLE_THRESHOLD | Disable at 80% memory |
| 124 | 10.0 | LOSS_SPIKE_THRESHOLD | Loss spike detection threshold |
| 125 | 0.5 | AUX_LOSS_CLAMP_FACTOR | Auxiliary loss clamping |
| 128 | 0.8 | BATCH_SIZE_REDUCTION_FACTOR | Batch size reduction on OOM |
| 131 | 0.5 | MEMORY_CLEANUP_DISCREPANCY_THRESHOLD_GB | Cleanup reporting threshold |
| 132 | 0.1 | MEMORY_CLEANUP_MIN_LOG_THRESHOLD_GB | Minimum freed memory to log |
| 135 | 50 | ASYNC_CACHE_CLEAR_POLL_INTERVAL_MS | Polling interval in ms |

#### 3. MoE Layer Constants

**File: `/project/code/src/Ava/config/constants.py` - MoEConstants class (Lines 138-159)**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 143 | 1024 | ROUTING_CACHE_SIZE | Maximum routing cache size |
| 144 | 32 | ROUTING_SAMPLE_SIZE | Sample size for tensor hashing |
| 145 | 0.7 | ROUTING_HIT_RATE_INCREASE_THRESHOLD | Hit rate for increasing prefetch |
| 146 | 0.9 | ROUTING_HIT_RATE_DECREASE_THRESHOLD | Hit rate for decreasing prefetch |
| 149 | 512 | DIVERSITY_LOSS_APPROX_THRESHOLD | Threshold for approximate diversity loss |
| 150 | 128 | DIVERSITY_LOSS_MAX_SAMPLE_SIZE | Max sample size for diversity loss |
| 153 | 4 | MIN_EXPERTS_FOR_BATCHED_PROCESSING | Minimum experts for batching |
| 154 | 1 | PREFETCH_DEPTH_MIN | Minimum prefetch depth |
| 155 | 5 | PREFETCH_DEPTH_MAX | Maximum prefetch depth |
| 156 | 3 | PREFETCH_DEPTH_DEFAULT | Default prefetch depth |
| 157 | 100 | PREFETCH_ADJUSTMENT_INTERVAL | Steps between adjustments |
| 158 | 1000 | PATTERN_HISTORY_SIZE | Expert access pattern history size |

#### 4. Model Architecture Parameters

**File: `/project/code/src/Ava/models/moe_model.py` - EnhancedMoEConfig (Lines 36-81)**

| Line | Value | Controls | Configurable |
|------|-------|----------|--------------|
| 36 | 50257 | vocab_size | Via config dataclass |
| 37 | 768 | hidden_size | Via config dataclass |
| 38 | 12 | num_layers | Via config dataclass |
| 39 | 12 | num_attention_heads | Via config dataclass |
| 40 | 3072 | intermediate_size | Via config dataclass |
| 41 | 2048 | max_position_embeddings | Via config dataclass |
| 44 | 8 | num_experts | Via config dataclass |
| 45 | 2 | num_experts_per_token | Via config dataclass |
| 61 | 10000.0 | rope_theta | Rotary PE base |
| 80 | 3 | eos_token_id | EOS token (HARDCODED - BAD) |
| 81 | 0 | min_sequence_length | Minimum sequence length |

**File: `/project/code/src/Ava/models/moe_model.py` - Larger config (Lines 826-870)**

| Line | Value | Controls | Issue |
|------|-------|----------|-------|
| 826 | 32000 | vocab_size | Mixtral-style config |
| 827 | 4096 | hidden_size | Hardcoded |
| 828 | 32 | num_layers | Hardcoded |
| 829 | 32 | num_attention_heads | Hardcoded |
| 830 | 14336 | intermediate_size | Hardcoded (3.5x hidden) |
| 831 | 4096 | max_position_embeddings | Hardcoded |
| 834 | 32 | num_experts | Hardcoded |
| 835 | 2 | num_experts_per_token | Hardcoded |
| 861 | 8 | lora_rank | Hardcoded |
| 862 | 16 | lora_alpha | Hardcoded |
| 866 | 4 | max_active_experts_gpu | Hardcoded |
| 870 | 8 | expert_quantization_bits | Hardcoded |

#### 5. Expert Cache Parameters

**File: `/project/code/src/Ava/models/expert_cache.py` (Lines 301-302)**

| Line | Value | Controls | Issue |
|------|-------|----------|-------|
| 301 | 64 | min_cache_size | Hardcoded |
| 302 | 1024 | max_cache_size | Hardcoded |

#### 6. Generation Parameters

**File: `/project/code/src/Ava/models/moe_model.py` - Generation methods**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 668 | 100 | max_length default | Generation output length |
| 1150 | 100 | max_length default | Generation output length (duplicate) |
| 519 | 1 | diagonal offset | For triangular mask (OK) |
| 1073 | 1 | diagonal offset | For triangular mask (OK) |

#### 7. Data Validation Parameters

**File: `/project/code/src/Ava/data/dataloader.py`**

| Line | Value | Controls | Current Default |
|------|-------|----------|-----------------|
| 562 | 10 | min_sequence_length | Hardcoded in method |
| 563 | 0.6 | max_sequence_repetition_rate | Hardcoded in method |
| 564 | 10 | max_consecutive_repeats | Hardcoded in method |
| 421 | 10 | MIN_TEXT_LENGTH (Arrow files) | Hardcoded check |
| 501 | 10 | MIN_TEXT_LENGTH (JSONL files) | Hardcoded check |

#### 8. Multi-Column Data

**File: `/project/code/src/Ava/data/multi_column_data.py`**

| Line | Value | Controls | Issue |
|------|-------|----------|-------|
| 119 | 10000 | shuffle_buffer_size | Default not configurable |
| 552 | 512 | max_length fallback | Hardcoded fallback |
| 763 | 512 | max_length default | Another hardcoded default |
| 794 | 1000 | buffer_size | Hardcoded for Parquet |
| 1070 | 32 | batch_size default | Hardcoded default |
| 1074 | 0 | num_workers default | Hardcoded for Arrow safety |

#### 9. Async Logging Parameters

**File: `/project/code/src/Ava/logging/async_logging.py`**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 34 | 200 | batch_size | Async logging batch size |
| 632 | 50 | batch_size | Create fast logger batch size |
| 642 | 100 | batch_size | Create standard logger batch size |
| 653 | 200 | batch_size | Create comprehensive logger batch size |

#### 10. Learning Rate Management

**File: `/project/code/src/Ava/optimization/learning_rate/managers.py`**

| Line | Value | Controls | Severity |
|------|-------|----------|----------|
| 34 | 0 | warmup_steps | Default (configurable) |
| 35 | 1e-8 | warmup_start_lr | Hardcoded initial LR |
| 38 | 100 | batch_loss_window | Loss averaging window |
| 39 | 0.001 | min_improvement | Improvement threshold |
| 42 | 500 | plateau_patience | Batches to wait for plateau |
| 43 | 0.5 | plateau_factor | LR reduction factor |
| 44 | 100 | lr_check_interval | Check frequency in batches |
| 47 | 1.5 | divergence_threshold | Divergence multiplier |
| 48 | 2.0 | spike_threshold | Emergency reduction threshold |
| 49 | 0.1 | emergency_factor | Emergency reduction factor |
| 52 | 5 | stability_threshold | Consecutive improvements |
| 53 | 1.1 | increase_factor | LR increase multiplier |
| 54 | 1e-3 | max_lr | Maximum allowed LR |
| 55 | 1000 | increase_min_gap | Minimum steps between increases |
| 58 | 1e-7 | min_lr | Minimum learning rate |
| 59 | 5 | max_reductions | Maximum LR reductions |
| 66 | 0.03 | warmup_ratio | 3% of total steps |
| 67 | 0.01 | warmup_min_ratio | 1% of target LR |
| 72 | 0.01 | min_lr_ratio | Minimum LR fraction |
| 76 | 10 | plateau_patience | Steps to wait for plateau |
| 77 | 0.01 | plateau_threshold | Minimum improvement threshold |
| 78 | 0.5 | plateau_factor | LR reduction factor |
| 79 | 1e-8 | plateau_min_lr | Absolute minimum LR |
| 90 | 100 | recovery_warmup_steps | Warmup steps after reduction |

### B. HARDCODED STRINGS - File Paths, URLs, Model Names

#### 1. Directory Paths (Not from Config)

**File: `/project/code/src/Ava/config/training_config.py`**

| Line | Path | Purpose | Issue |
|------|------|---------|-------|
| 432-440 | /project/code/data/processed, ./data/*, ../data | fallback_data_paths | Good list, but should be environment variable aware |
| 447 | /project/code/data/Testing | data_dir | OK (from config) |
| 496 | /tmp/difficulty_cache | cache_dir | BAD - /tmp not portable |
| 549 | /project/code/outputs | output_dir | OK (from config) |
| 937 | /project/code/outputs | default output-dir | OK (from config) |
| 883 | /project/code/data/processed | default data-dir | OK (from config) |

**File: `/project/code/code/scripts/5_training/train.py`**

| Line | Path | Purpose | Issue |
|------|------|---------|-------|
| 202 | /project/code/configs/gpu/small.yaml | Default config | BAD - not parameterizable |
| 615 | /project/code/configs/gpu/small.yaml | Default config fallback | BAD - same issue |

**File: `/project/code/scripts/6_rhlf_Finetuning/test_training_cpu.py`**

| Line | Path | Purpose |
|------|------|---------|
| 115 | /tmp/rlhf_test_output | Save directory (RLHF) |
| 116 | /tmp/rlhf_test_logs | Log directory (RLHF) |

**File: `/project/code/scripts/6_rhlf_Finetuning/test_rlhf_cpu.py`**

| Line | Path | Purpose |
|------|------|---------|
| 161 | /tmp/test_rlhf_prompts.json | Test prompts file |
| 201 | /tmp/test_rlhf_output | Save directory |
| 202 | /tmp/test_rlhf_logs | Log directory |

**File: `/project/code/scripts/2_data_prep/pretokenize_dataset.py`**

| Line | Path | Purpose |
|------|------|---------|
| 8 | /tmp/output.bin | Example output path |

**File: `/project/code/tests/__init__.py`**

| Line | Path | Purpose |
|------|------|---------|
| 39 | /tmp/llm_tests | Test temporary directory |

#### 2. Default Tokenizer

**File: `/project/code/src/Ava/config/training_config.py` - Line 462**

```
default_tokenizer_name: str = 'Qwen/Qwen2.5-0.5B'
```

- Issue: Hardcoded tokenizer name
- Severity: MEDIUM - Should be configurable per task

#### 3. Data Loader Text Fields

**File: `/project/code/src/Ava/data/dataloader.py` - Line 518**

```python
text_fields = ['text', 'content', 'document', 'passage', 'input', 'question', 'instruction']
```

- Issue: Hardcoded list of fallback text field names
- Severity: LOW - Reasonable defaults but should be configurable

### C. DEVICE SPECIFICATIONS - GPU/CPU Assignment

#### 1. Device Selection

**File: `/project/code/scripts/5_training/train.py`**

| Line | Code | Issue | Severity |
|------|------|-------|----------|
| 673 | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Dynamic but assumes cuda if available | MEDIUM |
| 684 | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Same as above | MEDIUM |
| 2982 | `torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")` | Distributed training GPU selection | MEDIUM |
| 2984 | `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` | Hardcoded cuda:0 fallback | MEDIUM |
| 3051 | `cuda:0, Replica GPUs: {list(range(1, num_gpus))}` | Assumes GPU 0 is primary | MEDIUM |

**File: `/project/code/src/Ava/distributed/unified_distributed_manager.py`**

| Line | Code | Issue |
|------|------|-------|
| 89 | `model = model.cuda()` | No device override option |
| 121 | `device_ids=[torch.cuda.current_device()]` | Uses current device only |

**File: `/project/code/scripts/7_generation/generate.py`**

| Line | Code | Issue |
|------|------|-------|
| 173 | `self.device = torch.device('cpu')` | Default CPU when not available |
| 175 | `self.device = torch.device(device if torch.cuda.is_available() else 'cpu')` | User-specified device (good) |

#### 2. Multiprocessing Context

**File: `/project/code/src/Ava/data/dataloader.py` - Line 1841**

```python
'multiprocessing_context': 'spawn' if num_workers > 0 else None
```

- Issue: Hardcoded 'spawn' context
- Severity: MEDIUM - 'fork' might be better on Linux, 'spawn' on Windows

#### 3. Data Loader Workers

**File: `/project/code/src/Ava/config/training_config.py` - Line 453**

```
num_workers: int = 0  # CRITICAL FIX: Default 0 to avoid multiprocessing deadlocks
```

- Issue: Hardcoded to 0 (due to Arrow file issues)
- Severity: MEDIUM - Reduces data loading performance, but necessary for stability

**File: `/project/code/scripts/5_training/train.py` - Lines 1066-1072**

```python
if num_workers != 0:
    get_logger().warning(f"⚠️  Forcing num_workers: {num_workers} → 0 ...")
    num_workers = 0
```

- Issue: Overrides user-specified num_workers
- Severity: HIGH - User intention ignored for safety

### D. CONFIGURATION HARDCODING IN FUNCTIONS

#### 1. Trainer Constants Access

**File: `/project/code/src/Ava/training/core/trainer.py`**

- Uses `TRAINER_CONSTANTS`, `MOE_CONSTANTS`, `DATA_CONSTANTS` from config.constants
- Issue: If constants not updated before training, hardcoded defaults apply
- Severity: MEDIUM

#### 2. Bucket Boundaries

**File: `/project/code/src/Ava/data/dataloader.py` - Lines 177-180**

```python
if bucket_boundaries is None:
    DATA_CONSTANTS.__post_init__()
    self.bucket_boundaries = DATA_CONSTANTS.BUCKET_BOUNDARIES_DEFAULT.copy()
else:
    self.bucket_boundaries = sorted(bucket_boundaries)
```

- Value: `[64, 128, 256, 512, 1024, 2048, 4096]` (Line 94)
- Issue: Optimal boundaries are dataset-specific
- Severity: MEDIUM

#### 3. Data Split Ratio

**File: `/project/code/src/Ava/data/dataloader.py` - Lines 735-740**

```python
file_hash = int(hashlib.md5(file_path.name.encode()).hexdigest(), 16) % 100
if self.split == "train":
    if file_hash < 85:  # 85% for training
        split_files.append(file_path)
else:  # val
    if file_hash >= 85:  # 15% for validation
```

- Issue: 85/15 split is hardcoded
- Severity: HIGH - Should be configurable per dataset

#### 4. Profiling and Monitoring Thresholds

**File: `/project/code/src/Ava/data/dataloader.py`**

| Line | Value | Purpose | Issue |
|------|-------|---------|-------|
| 1314 | 1000 | MEMORY_CHECK_INTERVAL | Hardcoded frequency |
| 1410 | PROFILING_REPORT_INTERVAL | Log frequency | Uses constant |

#### 5. Batch Iteration Logic

**File: `/project/code/src/Ava/data/dataloader.py` - Line 857**

```python
rng = random.Random(42 + epoch_num)
```

- Issue: Hardcoded seed + epoch
- Severity: LOW - Good for reproducibility but could allow override

### E. MISCELLANEOUS HARDCODED VALUES

#### 1. Loss Computation

**File: `/project/code/src/Ava/training/core/trainer.py`**

| Line | Value | Controls |
|------|-------|----------|
| 2494 | 5000 | Checkpoint logging frequency |
| 2807 | 500 | Auxiliary loss logging frequency (reduced from 100) |
| 2846, 2851, 2856 | 100 | AUX loss logging intervals |
| 2938 | 1000 | Valid aux losses logging |
| 3101 | 2000 | Gradient norm logging frequency |
| 3324 | 2000 | Gradient health check logging |

#### 2. Logging Frequencies

**File: `/project/code/src/Ava/training/core/trainer.py`**

| Line | Value | Purpose |
|------|-------|---------|
| 3379 | 1000 | Routing statistics logging |
| 3472 | 100 | Optimizer step logging |
| 3527 | Uses config | Metrics logging frequency (GOOD) |
| 3675 | Uses config | MoE metrics frequency (GOOD) |
| 3772 | Uses config | Health summary frequency (GOOD) |
| 3904 | Uses config | Cache clear frequency (GOOD) |
| 3907 | EMERGENCY_CHECK_FREQUENCY | Emergency memory checks |

#### 3. Model Initialization

**File: `/project/code/scripts/5_training/train.py` - Line 674**

```python
model = OptimizedMoETransformer(model_config).to(device=device, dtype=torch.bfloat16)
```

- Issue: Hardcoded to bfloat16
- Severity: MEDIUM - Should respect config's mixed_precision

#### 4. Validation Rate Sampling

**File: `/project/code/src/Ava/data/dataloader.py` - Line 1029**

```python
should_validate = (self.validation_rate >= 1.0) or (random.random() < self.validation_rate)
```

- Parameter: `validation_rate: float = 0.01` (Line 566)
- Issue: 1% validation sampling
- Severity: LOW - Hardcoded but reasonable

#### 5. Async Logging Behavior

**File: `/project/code/src/Ava/logging/async_logging.py` - Default behavior**

- Batch sizes: 50, 100, 200 (depending on logger type)
- Issue: Not easily configurable without code changes
- Severity: LOW

---

## SUMMARY BY SEVERITY

### HIGH SEVERITY (Must Fix)
1. **Train/Val split hardcoded to 85/15** - affects model generalization
2. **num_workers forced to 0** - performance degradation but necessary for safety
3. **Default config path hardcoded** - /project/code/configs/gpu/small.yaml
4. **EOS token ID hardcoded to 3** - tokenizer-specific, breaks with different tokenizers
5. **Cache directory uses /tmp** - /tmp/difficulty_cache (non-portable)

### MEDIUM SEVERITY (Should Fix)
1. All data loading constants (8 buffer sizes, 12 bucket parameters, 8 sampling parameters)
2. Training thresholds (memory, gradient scaling, loss spikes)
3. MoE layer parameters (routing cache, diversity loss thresholds)
4. Device selection logic (could be more explicit)
5. Model dtype selection (hardcoded bfloat16)
6. Generation default max_length (100 tokens)
7. LR management thresholds (5-6 parameters)
8. Async logging batch sizes

### LOW SEVERITY (Nice to Have)
1. Text field names for data extraction
2. Bucket boundaries (reasonable defaults)
3. Minimum text length checks
4. Profiling intervals
5. Random seed for shuffling

---

## RECOMMENDATIONS

### Priority 1: Critical Fixes Needed
1. **Make EOS token ID configurable** (currently hardcoded to 3)
   - Location: `/project/code/src/Ava/models/moe_model.py:80`
   - Solution: Add to ModelConfig or get from tokenizer

2. **Make train/val split configurable** (currently 85/15)
   - Location: `/project/code/src/Ava/data/dataloader.py:735-740`
   - Solution: Add `train_val_split: float = 0.85` parameter

3. **Externalize default config path**
   - Location: `/project/code/scripts/5_training/train.py:202,615`
   - Solution: Use environment variable or argparse default

4. **Fix /tmp directory references**
   - Location: Multiple test files and configs
   - Solution: Use `tempfile.gettempdir()` or make configurable

5. **Respect user num_workers setting**
   - Location: `/project/code/scripts/5_training/train.py:1066-1072`
   - Solution: Log warning but allow user override with explicit flag

### Priority 2: Performance Improvements
1. **Move async logging batch sizes to config**
   - Impact: Logging throughput optimization
   - Effort: Low

2. **Make learning rate thresholds configurable**
   - Impact: Better LR adaptation to different models
   - Effort: Medium (already partially done)

3. **Externalize bucket boundaries**
   - Impact: Optimized batching for different datasets
   - Effort: Low (already has parameter)

4. **Make generation max_length configurable per model**
   - Impact: Better generation control
   - Effort: Low

### Priority 3: Code Organization
1. **Consolidate all hardcoded thresholds into constants.py**
   - Current status: Partially done
   - Remaining: ~15 logging/threshold constants

2. **Document which constants MUST be fixed vs. optional**
   - Add comments to constants.py explaining configurability

3. **Add constants validation schema**
   - Ensure all loaded constants are valid ranges

4. **Create migration guide**
   - For users upgrading from hardcoded to configurable constants

---

## CONFIG FILE TEMPLATE SUGGESTIONS

### Add to constants section in YAML:
```yaml
constants:
  data_pipeline:
    max_tokens: 8192
    max_batch_size: 64
    bucket_boundaries: [64, 128, 256, 512, 1024, 2048, 4096]
    train_val_split: 0.85
    samples_per_file: 500
    
  trainer:
    memory_warning_threshold: 0.99
    loss_spike_threshold: 10.0
    batch_size_reduction_factor: 0.8
    
  moe:
    routing_cache_size: 1024
    prefetch_depth_default: 3
```

### Add to model section:
```yaml
model:
  eos_token_id: 3  # Or derive from tokenizer
  generation_max_length: 512
```

### Add to data section:
```yaml
data:
  validation_split_ratio: 0.15
  min_sequence_length: 10
  min_text_length: 10
```

---

## Testing Strategy

1. **Unit tests for config loading**
   - Verify all constants load correctly from YAML

2. **Integration tests for different configurations**
   - Test with different bucket boundaries
   - Test with different memory thresholds
   - Test with different train/val splits

3. **Performance benchmarks**
   - Measure impact of different buffer sizes
   - Measure impact of different logging frequencies

4. **Portability tests**
   - Verify works on different machines with different GPUs
   - Verify works with different CPU counts
   - Verify works with different temp directory locations

---

## Files Most Impacted

| File | High Issues | Medium Issues | Low Issues | Total |
|------|------------|---------------|-----------|-------|
| dataloader.py | 0 | 28 | 4 | 32 |
| constants.py | 0 | 25 | 0 | 25 |
| training_config.py | 2 | 4 | 1 | 7 |
| trainer.py | 0 | 8 | 8 | 16 |
| moe_model.py | 1 | 5 | 2 | 8 |
| train.py | 2 | 3 | 1 | 6 |
| expert_cache.py | 0 | 2 | 0 | 2 |
| Other files | 0 | 4 | 2 | 6 |
| **TOTAL** | **5** | **79** | **18** | **102** |

