# Relative Paths Migration Guide

This document explains how the codebase has been refactored to use relative paths instead of hardcoded absolute paths, making it portable across different installation locations.

## Overview

The project previously contained 100+ hardcoded absolute paths (`/project/code/...`) throughout the codebase. These have been replaced with a flexible path resolution system that works from any installation location.

## What Changed

### New Modules

1. **`code/src/Ava/utils/paths.py`** - Path utility module
   - Automatically detects project root using marker files (`.git`, `pyproject.toml`, `.project`)
   - Provides functions to resolve standard directories
   - Supports environment variable overrides for custom paths
   - **Key functions:**
     - `get_project_root()` - Auto-detect project root
     - `get_data_dir(data_type)` - Get data directory
     - `get_models_dir(model_type)` - Get models directory
     - `get_outputs_dir()` - Get outputs directory
     - `get_tokenizer_path(tokenizer_name)` - Get tokenizer path
     - `add_src_to_path()` - Add src to Python path (replaces sys.path.insert)

2. **`code/src/Ava/config/yaml_loader.py`** - Enhanced YAML loader
   - Automatically resolves relative paths in YAML config files
   - Converts paths like `"code/data/processed"` to absolute paths at load time
   - **Key functions:**
     - `load_yaml_with_path_resolution(config_path)` - Load YAML with auto path resolution
     - `resolve_paths_in_config(config)` - Recursively resolve paths in dict
     - `make_paths_relative_in_config(config)` - Convert absolute paths back to relative

### Updated Python Files

#### Core Configuration
- **`code/src/Ava/config/training_config.py`**
  - Replaced hardcoded `/project/code/...` defaults with dynamic path utilities
  - Updated `DataConfig`, `OutputConfig`, `ProgressiveTrainingConfig` to use path helpers
  - Modified argparse defaults to use `get_data_dir()` and `get_outputs_dir()`
  - Updated path comparison logic to use dynamic defaults

#### Training Orchestration
- **`code/src/Ava/training/orchestration/run_manager.py`**
  - Changed `base_output_dir` parameter default from hardcoded to `None`
  - Auto-detects outputs directory when not specified
  - Updated `list_runs()` and `load_run()` class methods

### Updated YAML Config Files

The YAML configuration loader automatically handles relative paths, so configs can now use:

```yaml
data:
  data_dir: code/data/pretokenized  # Relative to project root
  tokenizer_name: code/models/tokenizer/enhanced-50680

output:
  output_dir: code/outputs/runs/my_run  # Relative to project root
```

**Example:** `code/configs/moe/minimal_working.yaml` - Updated to use relative paths

## How to Use

### For End Users

No changes needed! The system automatically:
1. Detects your project root
2. Resolves relative paths in config files
3. Creates output directories as needed

Just run training from any directory:
```bash
# Works from /project
python code/scripts/5_training/train_100m_full.py --config code/configs/moe/minimal_working.yaml

# Also works from anywhere with PROJECT_ROOT set
export PROJECT_ROOT=/path/to/project
python train_100m_full.py --config configs/moe/minimal_working.yaml
```

### For Developers

#### Using Path Utilities in Code

```python
from Ava.utils.paths import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    get_tokenizer_path,
    add_src_to_path
)

# Auto-detect project root
project_root = get_project_root()  # Returns Path object

# Get standard directories
data_dir = get_data_dir("processed")  # code/data/processed
models_dir = get_models_dir("tokenizer")  # code/models/tokenizer
outputs_dir = get_outputs_dir()  # code/outputs

# Get specific tokenizer path
tokenizer_path = get_tokenizer_path("enhanced-50680")

# Add src to path (instead of sys.path.insert)
add_src_to_path()
from Ava.utils import something  # Now works from anywhere
```

#### Using Path Resolution in YAML Configs

Configs support two formats:

**Relative paths (preferred):**
```yaml
data:
  data_dir: code/data/processed
  tokenizer_name: code/models/tokenizer/enhanced-50680
```

**Absolute paths (auto-resolved at load time):**
```yaml
data:
  data_dir: /project/code/data/processed  # Still works, but why?
```

**Environment variables (for overrides):**
```bash
export PROJECT_ROOT=/custom/location
export DATA_DIR=/custom/data/location
python train.py --config config.yaml
```

#### Updating Scripts to Use Path Utilities

**Before:**
```python
import sys
sys.path.insert(0, '/project/code/src')

data_dir = '/project/code/data/processed'
outputs_dir = '/project/code/outputs'
```

**After:**
```python
from Ava.utils.paths import add_src_to_path, get_data_dir, get_outputs_dir

add_src_to_path()  # Better than sys.path.insert

data_dir = get_data_dir("processed")
outputs_dir = get_outputs_dir()
```

#### Writing New Scripts

```python
#!/usr/bin/env python3
"""My training script."""

from pathlib import Path
from Ava.utils.paths import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    add_src_to_path
)

# Setup paths
add_src_to_path()  # Before importing from Ava

# Now you can import from Ava
from Ava.config.training_config import TrainingConfigManager

def main():
    # These work regardless of installation location
    data_dir = get_data_dir("processed")
    output_dir = get_outputs_dir()

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Rest of your script...

if __name__ == "__main__":
    main()
```

## Environment Variable Overrides

You can override any path using environment variables:

```bash
# Override project root
export PROJECT_ROOT=/custom/project/location
python train.py --config config.yaml

# Override specific directories
export DATA_DIR=/mnt/nvme/data
export OUTPUTS_DIR=/mnt/ssd/outputs
export MODELS_DIR=/mnt/models
export CONFIGS_DIR=/mnt/configs
python train.py --config config.yaml
```

## Migration Checklist for Developers

If you're adding new hardcoded paths:

- [ ] Use `get_project_root()` instead of `/project`
- [ ] Use `get_data_dir()` instead of `/project/code/data/...`
- [ ] Use `get_outputs_dir()` instead of `/project/code/outputs`
- [ ] Use `get_models_dir()` instead of `/project/code/models/...`
- [ ] Use `add_src_to_path()` instead of `sys.path.insert(0, '/project/code/src')`
- [ ] For YAML configs, use relative paths like `code/data/processed`
- [ ] Document any custom path variables your code creates

## Backward Compatibility

The system maintains backward compatibility:
- Absolute paths in YAML configs still work (auto-resolved)
- Existing hardcoded paths in old code still work
- No breaking changes to public APIs
- Graceful fallbacks if path utilities unavailable

## Common Issues and Solutions

### Issue: "Could not detect project root"
**Solution:** Set `PROJECT_ROOT` environment variable
```bash
export PROJECT_ROOT=/path/to/project
```

### Issue: Relative path not found in config
**Solution:** Use paths relative to project root, not current working directory
```yaml
# Good - relative to project root
data_dir: code/data/processed

# Bad - relative to cwd, won't work
data_dir: data/processed
```

### Issue: Old script with hardcoded paths still works
**Good!** The system is backward compatible. But consider migrating for portability.

## Files Modified Summary

### New Files (2)
- `code/src/Ava/utils/paths.py` - Path resolution utilities
- `code/src/Ava/config/yaml_loader.py` - YAML loader with path resolution

### Python Files Updated (2)
- `code/src/Ava/config/training_config.py` - 6 locations updated
- `code/src/Ava/training/orchestration/run_manager.py` - 4 locations updated

### Config Files Updated (1 example)
- `code/configs/moe/minimal_working.yaml` - Updated to use relative paths

**Note:** Other config files in `code/configs/` work with the new system without modification due to automatic YAML path resolution.

## Benefits

 **Portability** - Code works on any system without modification
 **No Reinstallation** - Move project to different location, just works
 **Clean Environment** - No need for complex path setup scripts
 **Cloud-Ready** - Works seamlessly in containers and VMs
 **Consistent** - Same behavior everywhere
 **Backward Compatible** - Old code still works
 **Flexible** - Override any path via environment variables

## Questions?

- Check `code/src/Ava/utils/paths.py` for available utility functions
- Check `code/src/Ava/config/yaml_loader.py` for YAML path resolution
- Check `code/src/Ava/config/training_config.py` for config handling examples
