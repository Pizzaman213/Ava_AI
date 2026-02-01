"""
Enhanced YAML loader that resolves relative paths automatically.

This module provides a YAML loader that converts relative paths in config files
to absolute paths based on the project root directory, allowing YAML configs
to be portable across different installation locations.

Usage:
    from ava.config.yaml_loader import load_config_with_path_resolution

    config = load_config_with_path_resolution("code/configs/gpu/small.yaml")
    # Paths in YAML like "code/data/processed" are automatically resolved
"""

import logging
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Issue #12 fix: Make PATH_KEYS configurable and extensible
# Default path keys that should be resolved relative to project root
DEFAULT_PATH_KEYS = {
    'data_dir', 'val_data_dir', 'output_dir', 'tokenizer_name', 'tokenizer_path',
    'model_path', 'checkpoint_path', 'resume', 'cache_dir',
    'knowledge_base_path', 'config_file', 'deepspeed_config',
    'policy_model_path', 'judge_model_path', 'prompt_dataset_path',
    'eval_prompt_dataset_path', 'plot_path', 'log_dir', 'wandb_dir',
    'profile_dir', 'calibration_cache_dir', 'input_dir', 'custom_data_path'
}

# Module-level setting for additional path keys (can be extended at runtime)
_additional_path_keys: set = set()


def add_path_key(key: str) -> None:
    """Add a custom key to be treated as a path for resolution.

    Args:
        key: The config key name to treat as a path
    """
    _additional_path_keys.add(key.lower())


def get_path_keys() -> set:
    """Get current set of path keys (default + additional).

    Returns:
        Set of all path keys that will be resolved
    """
    return DEFAULT_PATH_KEYS | _additional_path_keys


# Import from centralized errors module
try:
    from ava.core.errors import ConfigurationError
except ImportError:
    # Fallback for when errors module is not available
    class ConfigurationError(Exception):
        """Raised when configuration loading or parsing fails."""
        pass

# Always import from centralized paths module
from ava.core.paths import get_project_root, resolve_path


def resolve_paths_in_config(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Recursively resolve relative paths in configuration dictionary.

    Converts paths like "code/data/processed" to absolute paths based on
    the project root. Handles nested dictionaries and lists.

    Args:
        config: Configuration dictionary (typically from YAML)
        project_root: Project root directory (auto-detected if not provided)

    Returns:
        Configuration dictionary with resolved paths
    """
    if project_root is None:
        project_root = get_project_root()

    # Issue #12 fix: Use dynamic PATH_KEYS that can be extended at runtime
    PATH_KEYS = get_path_keys()

    def resolve_value(value: Any, key: str = '') -> Any:
        """Recursively resolve paths in values."""
        if isinstance(value, dict):
            return {k: resolve_value(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, key) for item in value]
        elif isinstance(value, str) and key.lower() in PATH_KEYS:
            # Check if it looks like a relative path (doesn't start with /, ~ or have drive letter)
            if value and not value.startswith(('/','~')) and not (len(value) > 1 and value[1] == ':'):
                # Try to resolve as project-relative path
                resolved = resolve_path(value, project_root)
                return str(resolved)
        return value

    return resolve_value(config)


def load_yaml_with_path_resolution(
    config_path: str,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load YAML config file and resolve relative paths.

    Automatically converts relative paths in the YAML to absolute paths
    based on the project root.

    Args:
        config_path: Path to YAML configuration file (can be relative or absolute)
        project_root: Project root directory (auto-detected if not provided)

    Returns:
        Loaded configuration dictionary with resolved paths

    Raises:
        FileNotFoundError: If config file cannot be found
        yaml.YAMLError: If YAML parsing fails
    """
    if project_root is None:
        project_root = get_project_root()

    # Resolve config file path
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # Try as relative to current directory first
        if (Path.cwd() / config_file).exists():
            config_file = Path.cwd() / config_file
        # Otherwise try relative to project root
        elif (project_root / config_file).exists():
            config_file = project_root / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML with explicit error handling
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Re-raise with more context about which file failed
        error_msg = f"Failed to parse YAML config file '{config_file}': {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e
    except IOError as e:
        error_msg = f"Failed to read config file '{config_file}': {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e

    # Handle empty config files
    if config is None:
        logger.warning(f"Config file '{config_file}' is empty, using empty dict")
        config = {}

    # Resolve paths in config
    config = resolve_paths_in_config(config, project_root)

    return config


def make_paths_relative_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert absolute paths in config to relative paths (for saving to YAML).

    This is useful when saving a config after loading and potentially modifying paths.

    Args:
        config: Configuration dictionary with absolute paths

    Returns:
        Configuration dictionary with relative paths
    """
    project_root = get_project_root()

    # List of config keys that typically contain paths
    PATH_KEYS = {
        'data_dir', 'output_dir', 'tokenizer_name', 'tokenizer_path',
        'model_path', 'checkpoint_path', 'resume', 'cache_dir',
        'knowledge_base_path', 'config_file', 'deepspeed_config',
        'policy_model_path', 'judge_model_path', 'prompt_dataset_path',
        'eval_prompt_dataset_path', 'plot_path'
    }

    def make_relative_value(value: Any, key: str = '') -> Any:
        """Recursively convert paths to relative."""
        if isinstance(value, dict):
            return {k: make_relative_value(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [make_relative_value(item, key) for item in value]
        elif isinstance(value, str) and key.lower() in PATH_KEYS:
            try:
                path = Path(value).resolve()
                relative = path.relative_to(project_root)
                return str(relative)
            except (ValueError, OSError):
                # Path not under project root, keep as is
                pass
        return value

    return make_relative_value(config)


__all__ = [
    'load_yaml_with_path_resolution',
    'resolve_paths_in_config',
    'make_paths_relative_in_config',
    'get_project_root',
    'resolve_path'
]
