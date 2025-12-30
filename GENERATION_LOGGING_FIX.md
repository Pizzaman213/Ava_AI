# Generation Logging Issue - Root Cause and Fix

## Problem
Generation samples were not being logged to the WandB table during training.

## Root Cause
The generation logging failure was caused by **silent WandB initialization failures**. When WandB initialization throws an exception, the code sets `_use_wandb = False` but doesn't provide clear error messages about why it failed.

### Common Failure Points:
1. **WandB not installed** - Most common issue
   - Fix: `pip install wandb`

2. **WandB not authenticated**
   - Fix: `wandb login` and provide API key

3. **Network connectivity issues**
   - Fix: Check network connectivity, firewall rules

4. **Missing wandb_config in training configuration**
   - Fix: Ensure `wandb` section exists in your config file

5. **Invalid wandb configuration**
   - Fix: Check project name, entity, and other WandB settings

## How the Bug Manifested
The code flow:
1. `TrainingLoopManager._maybe_generate()` checks if `self._metrics_manager` exists
2. If it does, calls `metrics_manager.log_generation()`
3. `log_generation()` checks `if not self._use_wandb: return`
4. If WandB init failed silently, `_use_wandb` was False and nothing was logged
5. **No error message** was shown to the user about why generation logging failed

## Fixes Applied

### 1. Enhanced Debug Logging in `TrainingLoopManager._maybe_generate()` (loop.py)
Added messages showing:
- When generation is being logged to metrics manager
- Whether WandB is enabled in the metrics manager
- Warning if metrics manager is not available

### 2. Enhanced Debug Logging in `MetricsManager.log_generation()` (metrics.py)
Added messages showing:
- When log_generation is called but WandB is disabled
- Whether WandB is available and enabled

### 3. Improved Error Messages in `MetricsManager.setup()` (metrics.py)
- Added check if WandB was requested but not installed
- Captures full traceback if WandB initialization fails
- Provides clear error message with common fixes

## How to Verify the Fix
1. Check training logs for messages like:
   - `"⚠️  GENERATION LOGGING ISSUE: WandB was requested but is NOT installed!"`
   - `"⚠️  GENERATION LOGGING ISSUE: Failed to initialize WandB!"`
   - `"[Gen] Logging to metrics manager (use_wandb=True)"`

2. Look for `[Gen DEBUG]` messages showing:
   - When generation logging is attempted
   - Whether WandB is enabled or disabled

## To Enable Generation Logging
1. Install WandB:
   ```bash
   pip install wandb
   ```

2. Authenticate:
   ```bash
   wandb login
   ```

3. Ensure your training config has wandb section:
   ```yaml
   wandb:
     enabled: true
     project: "your-project"
     entity: "your-entity"  # optional
   ```

4. Run training and check for the debug messages

## Example Debug Output (When Fixed)
```
[Epoch 1] Step 500/1000 | BS: 32 | Loss: 2.5432 | LR: 5.0e-04
  [Gen] Step 500: launching async CPU generation...
  [Gen] Async thread started for step 500
  [Gen] Async generation completed for step 500
  [Gen] Found 1 completed generation(s)
============================================================
[Generation at Step 500]
Prompt: Once upon a time
Generated: There was a young girl who lived in a small village...
============================================================

  [Gen] Logging to metrics manager (use_wandb=True)
  [Gen DEBUG] log_generation called with _use_wandb=True, WANDB_AVAILABLE=True
  [Gen] Logged 1 generation(s) to WandB table at step 500
```

## Files Modified
- `code/src/ava/training/loop.py` - Added debug logging in `_maybe_generate()`
- `code/src/ava/training/metrics.py` - Added debug and error logging in `log_generation()` and `setup()`
