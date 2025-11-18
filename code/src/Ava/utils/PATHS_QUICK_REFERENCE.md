# Path Resolution Quick Reference

## Quick Start

### In Python Scripts
```python
from Ava.utils.paths import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    get_tokenizer_path,
    add_src_to_path
)

# Initialize paths early in your script
add_src_to_path()

# Use the functions
project_root = get_project_root()
data = get_data_dir("processed")
outputs = get_outputs_dir()
tokenizer = get_tokenizer_path("enhanced-50680")
```

### In YAML Configs
```yaml
data:
  # Use relative paths from project root
  data_dir: code/data/processed
  tokenizer_name: code/models/tokenizer/enhanced-50680

output:
  output_dir: code/outputs/runs/my_experiment
```

## Available Functions

| Function | Returns | Example |
|----------|---------|---------|
| `get_project_root()` | Path to project root | `/home/user/ava_project` |
| `get_code_dir()` | `code/` subdirectory | `/home/user/ava_project/code` |
| `get_data_dir(type)` | Data directory | `/home/user/ava_project/code/data/processed` |
| `get_models_dir(type)` | Models directory | `/home/user/ava_project/code/models/tokenizer` |
| `get_tokenizer_path(name)` | Full tokenizer path | `/home/user/ava_project/code/models/tokenizer/enhanced-50680` |
| `get_outputs_dir()` | Outputs directory | `/home/user/ava_project/code/outputs` |
| `get_configs_dir()` | Configs directory | `/home/user/ava_project/code/configs` |
| `resolve_path(rel_path, base)` | Resolve relative to base | `/home/user/ava_project/code/data/raw` |
| `add_src_to_path()` | Add src/ to Python path | Enables `from Ava import ...` |

## Environment Variables

Override any path using environment variables:

```bash
# Override project root
export PROJECT_ROOT=/custom/location

# Override specific directories
export DATA_DIR=/mnt/data
export MODELS_DIR=/mnt/models
export OUTPUTS_DIR=/mnt/outputs
export CONFIGS_DIR=/mnt/configs
```

## Common Patterns

### Script Initialization
```python
#!/usr/bin/env python3
from pathlib import Path
from Ava.utils.paths import add_src_to_path, get_project_root, get_data_dir

# Always do this first
add_src_to_path()

# Now you can import Ava modules
from Ava.config.training_config import TrainingConfigManager

def main():
    project_root = get_project_root()
    data_dir = get_data_dir("processed")
    print(f"Working in: {project_root}")
    print(f"Data at: {data_dir}")
```

### Data Loading
```python
from Ava.utils.paths import get_data_dir

def load_training_data():
    data_dir = get_data_dir("pretokenized")
    arrow_files = list(data_dir.glob("*.arrow"))
    return arrow_files
```

### Checkpoint Management
```python
from Ava.utils.paths import get_outputs_dir

def save_checkpoint(model, step):
    output_dir = get_outputs_dir()
    checkpoint_dir = output_dir / "runs" / "current_run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step}.pt"
    torch.save(model.state_dict(), checkpoint_path)
```

### Multi-Stage Script
```python
from pathlib import Path
from Ava.utils.paths import (
    add_src_to_path,
    get_project_root,
    get_data_dir,
    get_outputs_dir
)

# Stage 1: Setup
add_src_to_path()
from Ava.training.train import train

# Stage 2: Resolve paths
root = get_project_root()
data = get_data_dir("processed")
output = get_outputs_dir() / "runs" / "experiment_1"

# Stage 3: Execute
train(data_dir=data, output_dir=output)
```

## YAML Config Examples

### Minimal Config
```yaml
data:
  data_dir: code/data/processed
  tokenizer_name: code/models/tokenizer/enhanced-50680

output:
  output_dir: code/outputs/runs/test

training:
  batch_size: 32
  learning_rate: 0.001
```

### With Environment Overrides
```yaml
# This works even if DATA_DIR environment variable is set
data:
  data_dir: code/data/processed  # Used if env var not set
  tokenizer_name: code/models/tokenizer/enhanced-50680

# Can also use absolute paths if needed
output:
  output_dir: /custom/outputs/runs/test
```

## Troubleshooting

### "Could not detect project root"
```python
# Set manually if auto-detection fails
import os
os.environ['PROJECT_ROOT'] = '/path/to/project'

from Ava.utils.paths import get_project_root
root = get_project_root()  # Now works
```

### "ModuleNotFoundError: No module named 'Ava'"
```python
# Forgot to call add_src_to_path()!
from Ava.utils.paths import add_src_to_path
add_src_to_path()  # Call this first

from Ava.config import something  # Now works
```

### Config file not found
```python
# Use full relative path from project root
config_path = "code/configs/gpu/small.yaml"  # Good

# Or absolute path
config_path = "/home/user/ava_project/code/configs/gpu/small.yaml"  # Also good

# Not recommended (relative to cwd)
config_path = "configs/gpu/small.yaml"  # May not work
```

## Project Root Detection

The system looks for these markers in this order:
1. `.git/` directory (git repository)
2. `pyproject.toml` (Python project)
3. `.project/` directory (custom marker)
4. Falls back to parent directories

If auto-detection fails, set:
```bash
export PROJECT_ROOT=/actual/project/path
```

## Tips

- Always use these utilities instead of hardcoding paths
- Call `add_src_to_path()` before importing Ava modules
- Use relative paths in YAML configs (better for version control)
- Environment variables are useful for CI/CD and container deployments
- Path functions return `Path` objects (from `pathlib`) for maximum flexibility
