# Dynamic Configuration System

The config loader now supports **dynamic loading** of all YAML fields without requiring predefined dataclasses.

## Overview

The new `DynamicConfig` class allows you to:
- ✅ Load **any YAML structure** without code changes
- ✅ Access config values using **dot notation** (`config.training.batch_size`)
- ✅ Access config values using **dictionary style** (`config['training']['batch_size']`)
- ✅ Add **new fields to YAML** without modifying Python code
- ✅ Maintain **backward compatibility** with existing code

## Quick Start

### Basic Usage

```python
from src.Ava.config.training_config import TrainingConfigManager

# Create manager
manager = TrainingConfigManager()

# Load YAML config dynamically (new way)
config = manager.load_yaml_config('configs/gpu/small.yaml')

# Access any field with dot notation
batch_size = config.training.batch_size
learning_rate = config.training.learning_rate
hidden_size = config.model.hidden_size
data_dir = config.data.data_dir

# Access with safe fallback
max_samples = config.data.get('max_samples', None)
custom_field = config.get('custom_section', {})
```

### With Command-Line Arguments

```python
import argparse
from src.Ava.config.training_config import TrainingConfigManager

# Parse command-line arguments
manager = TrainingConfigManager()
parser = manager.create_argument_parser()
args = parser.parse_args()

# Create unified config (YAML + CLI overrides)
config = manager.create_unified_config(args)

# Command-line args override YAML values
batch_size = config.training.batch_size  # Uses CLI arg if provided, otherwise YAML
```

## Examples

### Example 1: Accessing Nested Values

```python
config = manager.load_yaml_config('configs/gpu/small.yaml')

# All these work:
print(config.model.hidden_size)                    # 96
print(config.training.batch_size)                  # 86
print(config.enhanced_features.losses.diversity_loss)  # True
print(config.wandb.enabled)                        # True
```

### Example 2: Adding New YAML Fields

You can add **any new field** to your YAML without modifying Python code:

```yaml
# In your YAML file:
custom_section:
  my_new_feature: true
  my_setting: 42
  nested:
    deep_value: "works!"
```

```python
# Python code - no changes needed!
config = manager.load_yaml_config('configs/gpu/small.yaml')

# Access your new fields immediately
print(config.custom_section.my_new_feature)  # True
print(config.custom_section.my_setting)      # 42
print(config.custom_section.nested.deep_value)  # "works!"
```

### Example 3: Safe Access with Defaults

```python
config = manager.load_yaml_config('configs/gpu/small.yaml')

# Get with default fallback
max_samples = config.data.get('max_samples', 10000)
custom_value = config.get('nonexistent_section', {})

# Check if attribute exists
if hasattr(config, 'custom_section'):
    print("Custom section found!")
```

### Example 4: Converting to Dictionary

```python
config = manager.load_yaml_config('configs/gpu/small.yaml')

# Convert back to dict for legacy code
config_dict = config.to_dict()

# Now use like a regular dict
batch_size = config_dict['training']['batch_size']
```

### Example 5: Validation

```python
config = manager.load_yaml_config('configs/gpu/small.yaml')

# Validate configuration
messages = manager.validate_dynamic_config(config)

if messages:
    for msg in messages:
        print(f"Warning: {msg}")
else:
    print("Configuration is valid!")
```

## Migration Guide

### Old Way (Manual Field Mapping)

```python
# train.py - OLD approach (100+ lines of manual merging)
config_dict = yaml.safe_load(f)

if "training" in config_dict:
    training_yaml = config_dict["training"]
    if "batch_size" in training_yaml:
        training_config.training.batch_size = training_yaml["batch_size"]
    if "learning_rate" in training_yaml:
        training_config.training.learning_rate = training_yaml["learning_rate"]
    # ... repeat for every field ...
```

### New Way (Dynamic Loading)

```python
# train.py - NEW approach (2 lines!)
manager = TrainingConfigManager()
config = manager.load_yaml_config(args.config)

# All fields automatically accessible!
batch_size = config.training.batch_size
learning_rate = config.training.learning_rate
# ... any field in YAML works immediately ...
```

## Benefits

| Feature | Old System | New System |
|---------|-----------|------------|
| Add new YAML field | ❌ Modify Python code | ✅ Just edit YAML |
| Access nested values | ✅ `config_dict['a']['b']` | ✅ `config.a.b` |
| Type safety | ✅ Dataclasses | ⚠️ Dynamic (runtime) |
| Lines of code | ❌ 1000+ lines | ✅ ~100 lines |
| Backward compatible | N/A | ✅ Yes |

## API Reference

### DynamicConfig Class

```python
class DynamicConfig:
    """Dynamic configuration with dot notation access"""

    def __init__(self, data: Dict[str, Any])
        """Create from dictionary"""

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with fallback"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access: config['key']"""

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment"""
```

### TrainingConfigManager Methods

```python
def load_yaml_config(self, config_path: str) -> DynamicConfig:
    """Load YAML into dynamic structure"""

def create_unified_config(self, args: Namespace) -> DynamicConfig:
    """Load YAML + merge CLI args"""

def validate_dynamic_config(self, config: DynamicConfig) -> List[str]:
    """Validate dynamic config"""
```

## Notes

- The old `parse_args_to_config()` method still works for backward compatibility
- Existing code using `EnhancedTrainingConfig` dataclasses continues to work
- You can gradually migrate to the dynamic system
- Type hints are preserved where critical for validation

## See Also

- [training_config.py](training_config.py) - Implementation
- [small.yaml](../../configs/gpu/small.yaml) - Example YAML config
- [train.py](../../scripts/5_training/train.py) - Usage in training script
