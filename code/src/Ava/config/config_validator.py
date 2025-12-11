"""
Centralized Configuration Validator for Ava Training Framework.

Provides:
- Single source of truth for config access
- Nested key access with dot notation
- Default value handling
- Schema validation
- Deprecation warnings

Usage:
    from Ava.config.config_validator import ConfigValidator

    validator = ConfigValidator(config_dict)
    batch_size = validator.get('training.batch_size', default=8)
    validator.validate()  # Raises if missing required fields
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """
    Centralized configuration access and validation.

    Provides:
    - Unified access pattern via dot notation: get('training.batch_size')
    - Default values for optional fields
    - Validation of required fields
    - Deprecation warnings for old config paths
    - Type checking (optional)

    Example:
        >>> config = {'training': {'batch_size': 32, 'lr': 0.001}}
        >>> validator = ConfigValidator(config)
        >>> validator.get('training.batch_size')
        32
        >>> validator.get('training.epochs', default=10)
        10
    """

    # Deprecated config paths -> new paths
    DEPRECATED_PATHS: Dict[str, str] = {
        'data.tokenizer_name': 'data.tokenizer_path',
        'training.lr': 'training.learning_rate',
        'model.n_layers': 'model.num_layers',
        'model.n_heads': 'model.num_attention_heads',
    }

    # Required fields for training
    REQUIRED_FIELDS: List[str] = [
        'model.vocab_size',
        'model.hidden_size',
    ]

    # Fields with type requirements
    FIELD_TYPES: Dict[str, type] = {
        'model.vocab_size': int,
        'model.hidden_size': int,
        'model.num_layers': int,
        'model.num_attention_heads': int,
        'training.batch_size': int,
        'training.num_epochs': int,
        'training.learning_rate': float,
        'training.warmup_steps': int,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        strict: bool = False,
        warn_deprecated: bool = True,
    ):
        """
        Initialize the config validator.

        Args:
            config: Raw configuration dictionary
            strict: If True, raise errors on validation failures
            warn_deprecated: If True, warn about deprecated config paths
        """
        self._config = config
        self._strict = strict
        self._warn_deprecated = warn_deprecated
        self._accessed_paths: Set[str] = set()
        self._warnings: List[str] = []

    def get(
        self,
        path: str,
        default: T = None,
        required: bool = False,
    ) -> Union[Any, T]:
        """
        Get a config value using dot notation.

        Args:
            path: Dot-separated path, e.g., 'training.batch_size'
            default: Default value if path not found
            required: If True, raise error if path not found

        Returns:
            Config value or default

        Raises:
            ConfigValidationError: If required and path not found
        """
        self._accessed_paths.add(path)

        # Check for deprecated path
        if self._warn_deprecated and path in self.DEPRECATED_PATHS:
            new_path = self.DEPRECATED_PATHS[path]
            warning = f"Config path '{path}' is deprecated, use '{new_path}' instead"
            self._warnings.append(warning)
            warnings.warn(warning, DeprecationWarning, stacklevel=2)

        # Navigate nested dict
        keys = path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                if required:
                    raise ConfigValidationError(
                        f"Required config field '{path}' not found"
                    )
                return default

        return value

    def get_nested(
        self,
        path: str,
        default: T = None,
    ) -> Union[Any, T]:
        """
        Alias for get() - for backwards compatibility.

        Args:
            path: Dot-separated path
            default: Default value if not found

        Returns:
            Config value or default
        """
        return self.get(path, default=default)

    def set(self, path: str, value: Any) -> None:
        """
        Set a config value using dot notation.

        Args:
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        config = self._config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

    def validate(self, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate the configuration.

        Checks:
        - Required fields are present
        - Field types are correct (if specified)
        - No unknown deprecated paths used

        Args:
            raise_on_error: If True, raise ConfigValidationError on failure

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            try:
                value = self.get(field, required=True)
                if value is None:
                    errors.append(f"Required field '{field}' is None")
            except ConfigValidationError as e:
                errors.append(str(e))

        # Check field types
        for field, expected_type in self.FIELD_TYPES.items():
            value = self.get(field)
            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"Field '{field}' should be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        is_valid = len(errors) == 0

        if not is_valid and raise_on_error and self._strict:
            raise ConfigValidationError(
                f"Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return is_valid, errors

    def get_warnings(self) -> List[str]:
        """Get all deprecation warnings encountered."""
        return self._warnings.copy()

    def get_accessed_paths(self) -> Set[str]:
        """Get all config paths that have been accessed."""
        return self._accessed_paths.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Get the underlying config dict."""
        return self._config.copy()

    def __contains__(self, path: str) -> bool:
        """Check if a config path exists."""
        try:
            value = self.get(path)
            return value is not None
        except ConfigValidationError:
            return False

    def __getitem__(self, path: str) -> Any:
        """Get config value using bracket notation."""
        return self.get(path, required=True)

    def merge(self, other: Dict[str, Any], override: bool = True) -> None:
        """
        Merge another config dict into this one.

        Args:
            other: Config dict to merge
            override: If True, other values override existing
        """
        self._merge_dicts(self._config, other, override)

    def _merge_dicts(
        self,
        base: Dict[str, Any],
        other: Dict[str, Any],
        override: bool,
    ) -> None:
        """Recursively merge dicts."""
        for key, value in other.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value, override)
            elif override or key not in base:
                base[key] = value


def validate_training_config(config: Dict[str, Any]) -> ConfigValidator:
    """
    Convenience function to validate a training config.

    Args:
        config: Configuration dictionary

    Returns:
        Validated ConfigValidator instance

    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator(config, strict=True)
    validator.validate()
    return validator
