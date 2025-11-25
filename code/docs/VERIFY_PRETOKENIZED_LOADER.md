# Verify Pre-Tokenized Loader is Active - Debugging Guide

## The Problem

Your logs show the loading is still slow (~18,000 examples/sec) even though we set `use_pretokenized: true`. This indicates the **streaming loader is still being used** instead of the ultra-fast pre-tokenized loader.

### Current Symptoms
```
 No conversation JSONL files found in /project/code/data/Ava_Ai/data, using standard loading
 Data directory: /project/code/data/Ava_Ai/data
 Datasets available: True              # ← This line = STREAMING loader active!
 Starting data loading...
  Loaded 10/1374 parquet files...       # ← Very slow - iterating ALL files
  Loaded 20/1374 parquet files...
Generating train split: 25000 examples [00:01, 18176.07 examples/s]  # ← 18K/sec = streaming
```

**Should see instead:**
```
 Using pretokenized Arrow data loader (60x faster)   # ← PRE-TOKENIZED loader!
...no "Loaded X/1374" messages...                       # ← Should NOT iterate files
Generating train split: 25000 examples [00:01, 60000.00 examples/s]  # ← 60K/sec = pre-tokenized
```

---

## Why This Happens

The data_loader_manager has logic:
```python
if use_pretokenized:
    create_ultra_fast_dataloaders()  # ← Should use this
else:
    create_streaming_dataloaders()   # ← Is using this instead
```

If it's using streaming, it means `use_pretokenized` is either:
1. Not in the config
2. Set to `false`
3. Not being read from the config
4. Being overridden somewhere

---

## Debug Steps

### Step 1: Verify Config File

```bash
# Check if use_pretokenized is in your config
grep "use_pretokenized" /project/code/configs/moe/minimal_working.yaml
```

Should output:
```
use_pretokenized: true
```

 If you see this, proceed to Step 2.
 If not, add it to the config.

### Step 2: Verify Config is Being Used

Check what command you're using to start training. It should be:
```bash
python code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/minimal_working.yaml
```

The `--config` flag must point to the right file.

### Step 3: Add Debug Logging

Edit `/project/code/src/Ava/training/train/data_loader_manager.py` and add debug output:

Find this section (around line 177):
```python
# Check if using pretokenized data
use_pretokenized = getattr(training_config.data, "use_pretokenized", False)
```

Add debugging right after:
```python
# Check if using pretokenized data
use_pretokenized = getattr(training_config.data, "use_pretokenized", False)

# DEBUG: Print what we found
print(f"\n DEBUG: use_pretokenized = {use_pretokenized}")
print(f" DEBUG: training_config.data attributes: {dir(training_config.data)}")
get_logger().warning(f" DEBUG: use_pretokenized={use_pretokenized}")
```

Then re-run training. You'll see the debug output showing what value was actually loaded.

### Step 4: Check YAML Parsing

The YAML might not be parsed correctly. Test it:

```bash
python << 'EOF'
import yaml

with open('/project/code/configs/moe/minimal_working.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("use_pretokenized value:", config.get('data', {}).get('use_pretokenized'))
print("Type:", type(config.get('data', {}).get('use_pretokenized')))
print("Expected: True (bool)")

# Check if it's a string instead of bool
val = config.get('data', {}).get('use_pretokenized')
if isinstance(val, str):
    print(f"  WARNING: use_pretokenized is a STRING '{val}', not a bool!")
    print("This will be treated as TRUE (non-empty string) in Python")
EOF
```

### Step 5: Force Pre-Tokenized Directly

If config loading is the issue, you can modify the data loader manager directly to **always** use pre-tokenized:

Find this line in `/project/code/src/Ava/training/train/data_loader_manager.py` (line ~177):
```python
use_pretokenized = getattr(training_config.data, "use_pretokenized", False)
```

Change to:
```python
# FORCE pre-tokenized for debugging
use_pretokenized = True  # ← TEMPORARY: Force it
# use_pretokenized = getattr(training_config.data, "use_pretokenized", False)
```

Then re-run training. You should see:
```
 Using pretokenized Arrow data loader (60x faster)
```

If you see this message, the config loading is the issue (not the code).

---

## Common Issues & Fixes

### Issue 1: "use_pretokenized: true" but still using streaming

**Cause**: Config value is a string "true" instead of boolean `true`

**Check**:
```bash
python << 'EOF'
import yaml
with open('/project/code/configs/moe/minimal_working.yaml', 'r') as f:
    config = yaml.safe_load(f)
val = config['data']['use_pretokenized']
print(f"Type: {type(val)}, Value: {val}")
EOF
```

**Fix**: In YAML, use bare `true` not quoted `"true"`:
```yaml
use_pretokenized: true   #  Correct (boolean)
use_pretokenized: "true" #  Wrong (string)
use_pretokenized: yes    #  Also works (boolean)
```

### Issue 2: Config file not in git (if you recreated it)

**Cause**: Maybe the config wasn't saved properly.

**Fix**:
```bash
# Verify file was saved
cat /project/code/configs/moe/minimal_working.yaml | grep -A 30 "^data:"
```

Should show all your data config.

### Issue 3: Training script ignores the config

**Cause**: Training script might have hardcoded `use_pretokenized=False` somewhere.

**Find it**:
```bash
grep -r "use_pretokenized" /project/code/scripts/5_training/
```

If you see `use_pretokenized = False` hardcoded, that's the problem.

---

## The Nuclear Option: Direct Code Fix

If config loading is broken, directly fix the data_loader_manager.py:

Edit `/project/code/src/Ava/training/train/data_loader_manager.py`, find line ~177:

**BEFORE**:
```python
use_pretokenized = getattr(training_config.data, "use_pretokenized", False)
```

**AFTER**:
```python
# Use pre-tokenized if available in config, but default to True for pre-tokenized Parquet files
use_pretokenized = getattr(training_config.data, "use_pretokenized", True)  # ← Changed default from False to True
```

This makes pre-tokenized the DEFAULT (more sensible for your pre-tokenized Parquet data).

---

## Verification Checklist

After making changes, verify with this checklist:

- [ ] Config file has `use_pretokenized: true` (not string, not commented out)
- [ ] Training command uses `--config code/configs/moe/minimal_working.yaml`
- [ ] No hardcoded `use_pretokenized = False` in training scripts
- [ ] Logs show "Using pretokenized Arrow data loader"
- [ ] Throughput is 30,000+ examples/sec (not 18,000)
- [ ] No "Loaded X/1374 parquet files..." messages during initialization
- [ ] Data loading takes seconds, not minutes

---

## Expected Log Output (With Pre-Tokenized Active)

```
 Creating dataloaders...
 Using pretokenized Arrow data loader (60x faster)   # ← This should appear
 Pre-tokenized Arrow loader initialized
 - Cache size: 200 tables
 - 16 workers
 - 8192 tokens max per batch
Generating train split: 25000 examples [00:01, 60000.00 examples/s]  # ← Should be 60K/sec
Generating train split: 25000 examples [00:01, 60000.00 examples/s]
Generating train split: 25000 examples [00:01, 60000.00 examples/s]
 Data loading complete in 2.3 seconds
```

---

## Current Config (Ultra-Optimized for Pre-Tokenized)

```yaml
data:
  streaming: true
  buffer_size: 2000               # Small (pre-tokenized loads fast)
  num_workers: 16                 # Maximum parallelism
  dataloader_prefetch_factor: 16  # Maximum prefetch
  dataloader_samples_per_file: 8000  # 6-8 files before rotating
  use_dynamic_batching: true      # Token-based batching
  use_pretokenized: true          # ← THE KEY SETTING
  multiprocessing_context: spawn  # Proper Arrow handling
  cache_size: 200                 # Large cache for hot files
```

**Expected result**: 30,000-60,000 samples/sec (40x faster than current 18,000)

---

## Next Step: Run This Test

Run training with the ultra-optimized config and watch the logs:

```bash
python code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/minimal_working.yaml 2>&1 | tee training.log
```

Then check the log:
```bash
grep -E "Using pretokenized|Loaded [0-9]+/1374|Generating train split" training.log | head -20
```

If you see:
-  "Using pretokenized Arrow data loader" = SUCCESS
-  "Loaded X/1374" messages = Still using streaming (debug further)

---

## Direct Test (No Training)

Test the pre-tokenized loader directly:

```bash
python << 'EOF'
from src.Ava.data.pretokenized_loader import create_ultra_fast_dataloaders

train_loader, val_loader = create_ultra_fast_dataloaders(
    batch_size=128,
    max_length=128,
    data_dir='/project/code/data/Ava_Ai/data',
    num_workers=16,
    buffer_size=2000,
    prefetch_factor=16,
    persistent_workers=True,
    samples_per_file=8000,
    cache_size=200,
    pad_token_id=0,
    eos_token_id=3,
)

print(f" Train loader created: {train_loader}")
print(f" Val loader created: {val_loader}")

# Try one batch
for batch in train_loader:
    print(f" Batch keys: {batch.keys()}")
    print(f" Batch shape: input_ids={batch['input_ids'].shape}")
    break

print(" Pre-tokenized loader works!")
EOF
```

If this works, the pre-tokenized loader code is fine. The issue is config loading.

---

## Summary

**If you see 60,000 examples/sec in logs**:
 Pre-tokenized loader is active - you're done!

**If you see 18,000 examples/sec in logs**:
 Streaming loader is still active - use debug steps above

The bottleneck is clear - we need to ensure `use_pretokenized=true` is being read from your config and used by the training pipeline.
