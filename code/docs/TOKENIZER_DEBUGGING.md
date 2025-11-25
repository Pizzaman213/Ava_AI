# Tokenizer Loading and Text Decoding Debugging Guide

**Date**: November 19, 2025
**Issue**: Text generation showing token IDs instead of decoded English text
**Status**:  Investigating with enhanced logging

---

## Problem

During generation testing, the output shows:
```
Generated sequence (token IDs): [35207, 413, 0, 313, 489, 868, ...]
```

Instead of:
```
Generated text:
Once upon a time, in a land far away, there lived a wise old merchant...
```

This indicates the tokenizer is not being loaded or not being passed to the generation function.

---

## Root Cause Analysis

### Confirmed Working:
 Tokenizer files exist at `/project/code/models/tokenizer/enhanced-50680/`
 Tokenizer can be loaded manually with `AutoTokenizer.from_pretrained()`
 Tokenizer.encode() and .decode() work correctly
 Generation function properly handles tokenizer when provided
 Tokenizer is passed to generate_sample() in train_epoch()

### Suspected Issues:
1. Path doubling: `/code/code/models/tokenizer/...` instead of `/code/models/tokenizer/...`
2. Silent exception during tokenizer loading
3. Tokenizer `None` at generation time

---

## Fixes Applied

### Fix #1: Path Deduplication ( Applied)
Added code to detect and fix double-prefixed paths:
```python
tokenizer_path = str(tokenizer_name)
if '/code/code' in tokenizer_path:
    tokenizer_path = tokenizer_path.replace('/code/code/', '/code/')
```

### Fix #2: trust_remote_code Flag ( Applied)
Added `trust_remote_code=True` to allow local tokenizer loading:
```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
```

### Fix #3: Detailed Debug Logging ( Applied)
Added comprehensive logging to trace tokenizer initialization:
```python
logger.info(f"  Tokenizer name from config: {tokenizer_name}")
logger.info(f"  Loading tokenizer from: {tokenizer_path}")
logger.info(f" Loaded tokenizer from {tokenizer_path} (vocab_size: {len(tokenizer)})")
# On error:
logger.error(f" Failed to load tokenizer from {tokenizer_path}: {e}")
logger.error(f"  Traceback: {traceback.format_exc()}")
```

---

## How to Debug

### Step 1: Check Initial Logs
When training starts, look for these log messages:
```
Tokenizer name from config: <path>
Loading tokenizer from: <path>
 Loaded tokenizer from <path> (vocab_size: 50680)
```

**If you see this**: Tokenizer loaded successfully 
**If you see an error**: Check the error message and traceback

### Step 2: Manual Test
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    '/project/code/models/tokenizer/enhanced-50680',
    trust_remote_code=True
)
print(f'Vocab size: {len(tokenizer)}')
print(tokenizer.decode([4794, 2259, 251, 594]))
"
```

Should output:
```
Vocab size: 50680
Once upon a time
```

### Step 3: Check Generation Output
During training, at step 500/1000/1500 etc., check generation logs:

**Good output**:
```
Prompt: Once upon a time, in a land far away,
Generated text:
Once upon a time, in a land far away, there lived...
```

**Bad output**:
```
Prompt: Once upon a time, in a land far away,
Generated sequence (token IDs): [35207, 413, 0, ...]
```

---

## Common Issues & Solutions

### Issue: "No tokenizer available - generation using token IDs only"
**Cause**: `tokenizer is None` - loading failed silently
**Solution**:
1. Check logs for ` Failed to load tokenizer` message
2. Verify path exists: `ls /project/code/models/tokenizer/enhanced-50680/`
3. Check file permissions
4. Check for path doubling: Does it say `/code/code/...`?

### Issue: Path shows as `/code/code/...` (doubled)
**Cause**: YAML path resolution is creating double paths
**Solution**: Already fixed with deduplication code
- Old path: `/code/code/models/tokenizer/enhanced-50680`
- Fixed to: `/code/models/tokenizer/enhanced-50680`

### Issue: "Tokenizer decode failed: ..."
**Cause**: Tokenizer loaded but decode threw exception
**Solution**:
1. Check error message in logs
2. Might be invalid token IDs from generation
3. Might be tokenizer version mismatch
4. Check config for `generation_skip_special_tokens: true`

---

## File Changes

### `/project/code/scripts/5_training/train_100m_full.py`

**Lines 1137-1162**: Enhanced tokenizer loading with:
- Path fixing for `/code/code/` → `/code/`
- Detailed logging at each step
- Full traceback on errors
- `trust_remote_code=True` flag

**Lines 1249-1259**: Turn-aware loader tokenizer loading with same fixes

**Lines 822-860**: Enhanced generation output with per-sample logging

---

## Test Plan

### During Next Training Run:

1. **Check startup logs** for tokenizer loading messages
2. **Look for generation output** at steps: 500, 1000, 1500, 2000, etc.
3. **Verify text is decoded** (not showing token IDs)
4. **Check diversity** - multiple samples should show different text

### Expected Log Output:
```
 Setting up optimizer and scheduler...

 Creating dataloaders...
  Tokenizer name from config: /project/code/models/tokenizer/enhanced-50680
  Loading tokenizer from: /project/code/models/tokenizer/enhanced-50680
 Loaded tokenizer from /project/code/models/tokenizer/enhanced-50680 (vocab_size: 50680)
```

### Expected Generation Output (Step 500+):
```
 Testing generation at step 500...
  Sample 1/5: 256 tokens → 1245 chars
Prompt: Once upon a time, in a land far away,
Generated text:
Once upon a time, in a land far away, there lived a wise old merchant...
---
  Sample 2/5: 256 tokens → 1189 chars
Prompt: Once upon a time, in a land far away,
Generated text:
Once upon a time, in a land far away, a young girl stood at the edge...
 Generation test complete
```

---

## Technical Details

### Why Tokenizer is Critical

The tokenizer:
1. **Encodes text → token IDs** (used during training)
2. **Decodes token IDs → text** (used during generation testing)

Without it, generation output is just raw numbers.

### Config Settings

From `code/configs/moe/minimal_working.yaml`:
```yaml
data:
  tokenizer_name: code/models/tokenizer/enhanced-50680
```

This gets resolved to absolute path by the YAML loader:
```
code/models/tokenizer/enhanced-50680
→ /project/code/models/tokenizer/enhanced-50680
```

### Tokenizer Files Required

The tokenizer needs these files in the directory:
-  `tokenizer.json` - Main tokenizer state (3.6 MB)
-  `tokenizer_config.json` - Configuration
-  `special_tokens_map.json` - Special tokens mapping

All three are present in your tokenizer directory.

---

## Next Steps

1. **Run training with new logging** to see exact tokenizer status
2. **Check logs for error messages** if tokenizer doesn't load
3. **Verify generation output** shows decoded text at step 500+
4. **Post logs here** if still not working (the debug logging will pinpoint the issue)

---

## Debugging Commands

Test tokenizer directly:
```bash
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/project/code/models/tokenizer/enhanced-50680', trust_remote_code=True)
print(f' Loaded (vocab_size={len(t)})')
print(t.decode([4794, 2259, 251, 594]))
"
```

Check path exists:
```bash
test -d /project/code/models/tokenizer/enhanced-50680 && echo ' Directory exists' || echo ' Not found'
ls -la /project/code/models/tokenizer/enhanced-50680/
```

Check for path doubling:
```bash
grep -n "code/code" code/scripts/5_training/train_100m_full.py
```

---

## Summary

The tokenizer is working, but might not be loading in the training script due to:
1. **Path issues** ( Fixed with deduplication)
2. **Exception handling** ( Fixed with detailed logging)
3. **Missing flags** ( Fixed with `trust_remote_code=True`)

Next training run will show exactly where the issue is through enhanced logging.

**Status**: Awaiting training run with new logging to diagnose exact issue.

---

*Last Updated: November 19, 2025*
*Debug Logging Added: Lines 1137-1162 in train_100m_full.py*
