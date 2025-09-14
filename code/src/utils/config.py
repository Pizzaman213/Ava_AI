"""
Configuration utilities for MoE++ models
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import os


class DotDict(dict):
    """Dictionary with dot notation access"""
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


def load_config(config_path: Union[str, Path]) -> DotDict:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration as DotDict for easy access
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try YAML first, then JSON
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse config file: {config_path}")
    
    return DotDict(config)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert DotDict to regular dict
    if isinstance(config, DotDict):
        config = dict(config)
    
    # Save based on extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configurations
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert configuration object to dictionary
    
    Args:
        config: Configuration object (DotDict, dataclass, etc.)
        
    Returns:
        Configuration as dictionary
    """
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    elif hasattr(config, '__dict__'):
        return {k: config_to_dict(v) for k, v in config.__dict__.items() if not k.startswith('_')}
    elif isinstance(config, (list, tuple)):
        return [config_to_dict(item) for item in config]
    else:
        return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation path
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "model.hidden_size")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set configuration value using dot notation path
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "model.hidden_size")
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set value
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that configuration contains required keys
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths
        
    Returns:
        True if all required keys present
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        if get_config_value(config, key_path) is None:
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


# Common configuration keys for validation
MODEL_REQUIRED_KEYS = [
    "model.vocab_size",
    "model.hidden_size",
    "model.num_layers",
    "model.num_attention_heads",
]

TRAINING_REQUIRED_KEYS = [
    "training.batch_size",
    "training.learning_rate",
    "training.num_epochs",
]

DATA_REQUIRED_KEYS = [
    "data.train_path",
    "data.tokenizer",
    "data.max_length",
]

DATA_OPTIONAL_KEYS = [
    "data.max_train_examples",  # int, null, or "auto" - limits training examples
    "data.max_val_examples",    # int, null, or "auto" - limits validation examples
]