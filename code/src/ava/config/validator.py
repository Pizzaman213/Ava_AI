"""
Centralized Configuration Validator for Ava Training Framework.

Provides:
- Single source of truth for config access
- Nested key access with dot notation
- Default value handling
- Schema validation
- Deprecation warnings
- Backward compatibility via path resolution

Usage:
    from ava.config.validator import ConfigValidator

    validator = ConfigValidator(config_dict)
    batch_size = validator.get('training.batch_size', default=8)
    validator.validate()  # Raises if missing required fields

Path Resolution:
    The config system has been reorganized from 38 sections into 8 categories.
    Old paths are automatically resolved to new canonical paths with deprecation
    warnings. See ava.config.path_mapping for the full mapping.
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Environment variable to suppress deprecation warnings
SUPPRESS_DEPRECATION_WARNINGS = os.environ.get('SUPPRESS_CONFIG_DEPRECATION', '').lower() in ('1', 'true', 'yes')


def _get_path_mappings() -> Dict[str, str]:
    """
    Get the full path mapping dictionary, combining legacy and new mappings.

    Returns:
        Dictionary mapping old paths to new canonical paths
    """
    # Start with legacy mappings
    mappings = {
        'data.tokenizer_name': 'data.tokenizer_path',
        'training.lr': 'training.learning_rate',
        'model.n_layers': 'model.num_layers',
        'model.n_heads': 'model.num_attention_heads',
    }

    # Try to import the comprehensive path mappings
    try:
        from ava.config.path_mapping import CONFIG_PATH_MAPPINGS
        mappings.update(CONFIG_PATH_MAPPINGS)
    except ImportError:
        # path_mapping module not available yet
        pass

    return mappings


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
    - Backward compatibility via path resolution

    Example:
        >>> config = {'training': {'batch_size': 32, 'lr': 0.001}}
        >>> validator = ConfigValidator(config)
        >>> validator.get('training.batch_size')
        32
        >>> validator.get('training.epochs', default=10)
        10

    Path Resolution:
        Old paths (e.g., 'hardware.device') are automatically resolved to
        new canonical paths (e.g., 'compute.device.type') with deprecation warnings.
    """

    # Deprecated config paths -> new paths (loaded dynamically)
    DEPRECATED_PATHS: Dict[str, str] = _get_path_mappings()

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
        auto_resolve_paths: bool = True,
    ):
        """
        Initialize the config validator.

        Args:
            config: Raw configuration dictionary
            strict: If True, raise errors on validation failures
            warn_deprecated: If True, warn about deprecated config paths
            auto_resolve_paths: If True, automatically resolve deprecated paths
        """
        self._config = config
        self._strict = strict
        self._warn_deprecated = warn_deprecated
        self._auto_resolve_paths = auto_resolve_paths
        self._accessed_paths: Set[str] = set()
        self._warnings: List[str] = []
        self._warned_deprecated: Set[str] = set()  # Track warned paths to avoid spam

    def resolve_path(self, path: str) -> str:
        """
        Resolve a deprecated config path to its canonical form.

        Args:
            path: Configuration path (may be deprecated)

        Returns:
            Canonical path (may be same as input if not deprecated)
        """
        return self.DEPRECATED_PATHS.get(path, path)

    def is_deprecated_path(self, path: str) -> bool:
        """
        Check if a config path is deprecated.

        Args:
            path: Configuration path to check

        Returns:
            True if path is deprecated
        """
        return path in self.DEPRECATED_PATHS

    def _warn_deprecated(self, path: str, new_path: str) -> None:
        """Emit deprecation warning (once per path)."""
        if SUPPRESS_DEPRECATION_WARNINGS:
            return

        if path in self._warned_deprecated:
            return

        self._warned_deprecated.add(path)
        warning = (
            f"Config path '{path}' is deprecated, use '{new_path}' instead. "
            f"Set SUPPRESS_CONFIG_DEPRECATION=1 to silence this warning."
        )
        self._warnings.append(warning)
        warnings.warn(warning, DeprecationWarning, stacklevel=3)

    def get(
        self,
        path: str,
        default: T = None,
        required: bool = False,
    ) -> Union[Any, T]:
        """
        Get a config value using dot notation.

        Deprecated paths are automatically resolved to new canonical paths
        with a deprecation warning.

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

        # Check for deprecated path and emit warning
        if path in self.DEPRECATED_PATHS:
            new_path = self.DEPRECATED_PATHS[path]
            self._warn_deprecated(path, new_path)
            # If auto-resolve is enabled and we can find the value at new path,
            # use that instead
            if self._auto_resolve_paths:
                new_value = self._get_value(new_path)
                if new_value is not None:
                    return new_value

        # Try to get value at the original path
        value = self._get_value(path)
        if value is not None:
            return value

        # Path not found
        if required:
            raise ConfigValidationError(
                f"Required config field '{path}' not found"
            )
        return default

    def _get_value(self, path: str) -> Optional[Any]:
        """
        Get value at a path without deprecation handling.

        Args:
            path: Dot-separated path

        Returns:
            Value at path, or None if not found
        """
        keys = path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

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
        - Numeric ranges are valid
        - Cross-field dependencies are satisfied
        - Paths exist

        Args:
            raise_on_error: If True, raise ConfigValidationError on failure

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []
        warnings: List[str] = []

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

        # NEW: Comprehensive validation
        self._validate_ranges(errors, warnings)
        self._validate_cross_fields(errors, warnings)
        self._validate_paths(errors, warnings)

        # Log warnings
        if warnings:
            warning_msg = "\n".join(f"  - {w}" for w in warnings)
            logger.warning(f"Configuration warnings:\n{warning_msg}")
            self._warnings.extend(warnings)

        is_valid = len(errors) == 0

        if not is_valid and raise_on_error and self._strict:
            raise ConfigValidationError(
                f"Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return is_valid, errors

    def _validate_ranges(self, errors: List[str], warnings: List[str]) -> None:
        """Validate numeric field ranges."""
        # Batch size
        bs = self.get('training.batch_size')
        if bs is not None:
            if bs <= 0:
                errors.append(f"training.batch_size must be > 0, got {bs}")
            elif bs > 10000:
                warnings.append(f"training.batch_size ({bs}) is very large, may cause OOM")

        # Gradient accumulation steps
        gas = self.get('training.gradient_accumulation_steps')
        if gas is not None:
            if gas <= 0:
                errors.append(f"training.gradient_accumulation_steps must be > 0, got {gas}")
            elif gas > 1000:
                warnings.append(f"training.gradient_accumulation_steps ({gas}) is very large")

        # Learning rate
        lr = self.get('training.learning_rate')
        if lr is not None:
            if lr <= 0:
                errors.append(f"training.learning_rate must be > 0, got {lr}")
            elif lr > 1.0:
                warnings.append(f"training.learning_rate ({lr}) is very high, may cause instability")

        # Max grad norm
        mgn = self.get('training.max_grad_norm')
        if mgn is not None and mgn <= 0:
            errors.append(f"training.max_grad_norm must be > 0, got {mgn}")

        # Hidden size
        hs = self.get('model.hidden_size')
        if hs is not None:
            if hs <= 0:
                errors.append(f"model.hidden_size must be > 0, got {hs}")
            elif hs % 64 != 0:
                warnings.append(
                    f"model.hidden_size ({hs}) not divisible by 64, "
                    "may be inefficient for GPU"
                )

        # Num layers
        nl = self.get('model.num_layers')
        if nl is not None and nl <= 0:
            errors.append(f"model.num_layers must be > 0, got {nl}")

        # Attention heads
        nah = self.get('model.num_attention_heads')
        if nah is not None and nah <= 0:
            errors.append(f"model.num_attention_heads must be > 0, got {nah}")

        # Batch size calibration validation
        calib = self.get('training.batch_size_calibration')
        if calib is not None and isinstance(calib, dict):
            if calib.get('enabled', False):
                min_bs = calib.get('min_batch_size', 1)
                max_bs = calib.get('max_batch_size', 256)
                if min_bs >= max_bs:
                    errors.append(
                        f"batch_size_calibration.min_batch_size ({min_bs}) must be < "
                        f"max_batch_size ({max_bs})"
                    )
                target_mem = calib.get('target_memory', 0.75)
                if not 0.0 < target_mem <= 1.0:
                    errors.append(
                        f"batch_size_calibration.target_memory must be in (0.0, 1.0], "
                        f"got {target_mem}"
                    )
                calib_headroom = calib.get('calibration_headroom', 0.90)
                if not 0.0 < calib_headroom < 1.0:
                    errors.append(
                        f"batch_size_calibration.calibration_headroom must be in (0.0, 1.0), "
                        f"got {calib_headroom}"
                    )

    def _validate_cross_fields(self, errors: List[str], warnings: List[str]) -> None:
        """Validate dependencies between fields."""
        # Hidden size must be divisible by attention heads
        hs = self.get('model.hidden_size')
        nah = self.get('model.num_attention_heads')
        if hs is not None and nah is not None:
            if hs % nah != 0:
                errors.append(
                    f"model.hidden_size ({hs}) must be divisible by "
                    f"model.num_attention_heads ({nah})"
                )

        # MoE validation
        num_experts = self.get('model.num_experts')
        if num_experts is not None and num_experts > 0:
            nept = self.get('model.num_experts_per_token')
            if nept is not None:
                if nept <= 0:
                    errors.append(f"model.num_experts_per_token must be > 0, got {nept}")
                elif nept > num_experts:
                    errors.append(
                        f"model.num_experts_per_token ({nept}) cannot exceed "
                        f"model.num_experts ({num_experts})"
                    )

        # Effective batch size warning
        bs = self.get('training.batch_size')
        gas = self.get('training.gradient_accumulation_steps')
        if bs is not None and gas is not None:
            effective_bs = bs * gas
            if effective_bs > 10000:
                warnings.append(
                    f"Effective batch size ({effective_bs} = {bs} * {gas}) is very large"
                )

        # Warn about potentially problematic torch.compile + gradient_checkpointing + DeepSpeed
        # This combination can cause CUDA "illegal memory access" errors due to nested
        # gradient checkpoints conflicting with torch.compile's CUDA graphs.
        use_torch_compile = (
            self.get('model.use_torch_compile') or
            self.get('performance.enable_torch_compile')
        )
        gradient_checkpointing = self.get('model.gradient_checkpointing')
        deepspeed_enabled = self.get('deepspeed.enabled')

        if use_torch_compile and gradient_checkpointing and deepspeed_enabled:
            warnings.append(
                "torch.compile + gradient_checkpointing + DeepSpeed enabled together. "
                "Inner expert checkpointing has been automatically disabled to prevent "
                "CUDA illegal memory access errors. If errors persist, try setting "
                "model.use_torch_compile: false OR model.gradient_checkpointing: false"
            )

    def _validate_paths(self, errors: List[str], warnings: List[str]) -> None:
        """Validate file/directory paths."""
        from pathlib import Path

        # Data directory
        data_dir = self.get('data.data_dir')
        if data_dir is not None:
            data_path = Path(data_dir)
            if not data_path.exists():
                errors.append(f"data.data_dir does not exist: {data_dir}")
            elif not data_path.is_dir():
                errors.append(f"data.data_dir is not a directory: {data_dir}")

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

    def validate_against_dataclass(
        self,
        dataclass_type: type,
        config_prefix: str = "",
    ) -> Tuple[bool, List[str]]:
        """
        Validate config against a dataclass schema for type-safe validation.

        This method uses dataclass field annotations to validate that config
        values have the correct types and required fields are present.

        Args:
            dataclass_type: A dataclass type to validate against
            config_prefix: Optional prefix for nested config access (e.g., 'model')

        Returns:
            Tuple of (is_valid, list of error messages)

        Example:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class ModelConfig:
            ...     hidden_size: int
            ...     num_layers: int = 12
            >>> validator = ConfigValidator({'model': {'hidden_size': 768}})
            >>> is_valid, errors = validator.validate_against_dataclass(ModelConfig, 'model')
        """
        from dataclasses import fields, is_dataclass, MISSING
        import typing

        errors: List[str] = []

        if not is_dataclass(dataclass_type):
            errors.append(f"{dataclass_type.__name__} is not a dataclass")
            return False, errors

        for field in fields(dataclass_type):
            # Build full path
            path = f"{config_prefix}.{field.name}" if config_prefix else field.name
            value = self.get(path)

            # Check if required (no default value)
            has_default = field.default is not MISSING or field.default_factory is not MISSING

            if value is None:
                if not has_default:
                    errors.append(f"Required field '{path}' is missing")
                continue

            # Type check
            expected_type = field.type

            # Handle Optional types
            origin = typing.get_origin(expected_type)
            if origin is typing.Union:
                args = typing.get_args(expected_type)
                # Check if it's Optional (Union with NoneType)
                if type(None) in args:
                    if value is None:
                        continue
                    # Get the non-None type
                    expected_type = next(t for t in args if t is not type(None))
                    origin = typing.get_origin(expected_type)

            # Handle generic types (List, Dict, etc.)
            if origin is not None:
                expected_type = origin

            # Perform type check
            if expected_type in (int, float, str, bool, list, dict):
                # Handle int/float compatibility
                if expected_type == float and isinstance(value, int):
                    continue  # int is acceptable for float
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{path}': expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        return len(errors) == 0, errors


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
