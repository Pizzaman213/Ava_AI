"""
Configuration Provenance Tracking

This module tracks where each configuration value comes from to help debug
configuration issues when multiple layers of overrides are applied
(YAML → CLI → environment → code defaults).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum
import os


class ConfigSource(Enum):
    """Source of configuration value."""
    DEFAULT = "default"  # Code default value
    YAML = "yaml"  # From YAML config file
    CLI = "cli"  # From command-line argument
    ENV = "env"  # From environment variable
    RUNTIME = "runtime"  # Set at runtime
    INHERITED = "inherited"  # Inherited from parent config


@dataclass
class ConfigValue:
    """
    Configuration value with provenance information.

    Tracks the actual value, where it came from, and when it was set.
    """
    value: Any
    source: ConfigSource
    precedence: int  # Higher number = higher precedence
    yaml_file: Optional[str] = None  # Path to YAML file if from YAML
    cli_arg: Optional[str] = None  # CLI argument name if from CLI
    env_var: Optional[str] = None  # Environment variable name if from ENV
    overridden: bool = False  # Whether this value was overridden
    previous_value: Optional[Any] = None  # Previous value if overridden

    def __repr__(self) -> str:
        """String representation."""
        source_detail = ""
        if self.source == ConfigSource.YAML and self.yaml_file:
            source_detail = f" ({self.yaml_file})"
        elif self.source == ConfigSource.CLI and self.cli_arg:
            source_detail = f" (--{self.cli_arg})"
        elif self.source == ConfigSource.ENV and self.env_var:
            source_detail = f" (${self.env_var})"

        override_info = ""
        if self.overridden:
            override_info = f" [overrode: {self.previous_value}]"

        return f"{self.value} (from: {self.source.value}{source_detail}){override_info}"


class ConfigProvenanceTracker:
    """
    Tracks provenance of all configuration values.

    Provides debugging information about where each config value comes from
    and which values were overridden.
    """

    # Precedence order (higher number wins)
    PRECEDENCE = {
        ConfigSource.DEFAULT: 0,
        ConfigSource.INHERITED: 1,
        ConfigSource.YAML: 2,
        ConfigSource.ENV: 3,
        ConfigSource.CLI: 4,
        ConfigSource.RUNTIME: 5,
    }

    def __init__(self):
        """Initialize provenance tracker."""
        self.values: Dict[str, ConfigValue] = {}
        self.override_history: List[Dict[str, Any]] = []

    def set_value(
        self,
        key: str,
        value: Any,
        source: ConfigSource,
        yaml_file: Optional[str] = None,
        cli_arg: Optional[str] = None,
        env_var: Optional[str] = None,
    ) -> None:
        """
        Set a configuration value with provenance tracking.

        Args:
            key: Configuration key (e.g., "training.batch_size")
            value: Configuration value
            source: Source of this value
            yaml_file: Path to YAML file (if from YAML)
            cli_arg: CLI argument name (if from CLI)
            env_var: Environment variable name (if from ENV)
        """
        precedence = self.PRECEDENCE[source]

        # Check if this key already exists
        if key in self.values:
            existing = self.values[key]

            # Only override if new source has higher precedence
            if precedence > existing.precedence:
                # Record override
                self.override_history.append({
                    'key': key,
                    'old_value': existing.value,
                    'new_value': value,
                    'old_source': existing.source.value,
                    'new_source': source.value,
                })

                # Create new ConfigValue with override info
                new_config_value = ConfigValue(
                    value=value,
                    source=source,
                    precedence=precedence,
                    yaml_file=yaml_file,
                    cli_arg=cli_arg,
                    env_var=env_var,
                    overridden=True,
                    previous_value=existing.value,
                )
                self.values[key] = new_config_value
            # else: Keep existing value (higher precedence)
        else:
            # New key, just set it
            self.values[key] = ConfigValue(
                value=value,
                source=source,
                precedence=precedence,
                yaml_file=yaml_file,
                cli_arg=cli_arg,
                env_var=env_var,
            )

    def get_value(self, key: str) -> Optional[ConfigValue]:
        """Get configuration value with provenance."""
        return self.values.get(key)

    def get_raw_value(self, key: str, default: Any = None) -> Any:
        """Get just the raw value without provenance."""
        config_value = self.values.get(key)
        return config_value.value if config_value else default

    def set_from_dict(
        self,
        config_dict: Dict[str, Any],
        source: ConfigSource,
        prefix: str = "",
        yaml_file: Optional[str] = None,
    ) -> None:
        """
        Recursively set configuration values from a dictionary.

        Args:
            config_dict: Configuration dictionary
            source: Source of these values
            prefix: Key prefix for nested dicts
            yaml_file: YAML file path (if applicable)
        """
        for key, value in config_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dicts
                self.set_from_dict(value, source, prefix=full_key, yaml_file=yaml_file)
            else:
                self.set_value(full_key, value, source, yaml_file=yaml_file)

    def set_from_cli_args(self, args_dict: Dict[str, Any]) -> None:
        """
        Set configuration values from CLI arguments.

        Args:
            args_dict: Dictionary of CLI arguments
        """
        for key, value in args_dict.items():
            if value is not None:  # Only set if explicitly provided
                self.set_value(key, value, ConfigSource.CLI, cli_arg=key)

    def set_from_env(self, env_prefix: str = "AVA_") -> None:
        """
        Set configuration values from environment variables.

        Args:
            env_prefix: Prefix for environment variables (e.g., "AVA_")
        """
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # Convert env key to config key
                # AVA_TRAINING_BATCH_SIZE -> training.batch_size
                config_key = env_key[len(env_prefix):].lower().replace('_', '.')

                # Try to parse value
                parsed_value = self._parse_env_value(env_value)
                self.set_value(config_key, parsed_value, ConfigSource.ENV, env_var=env_key)

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value (int, float, bool, or str)
        """
        # Try bool
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get_overrides(self) -> List[Dict[str, Any]]:
        """Get list of all configuration overrides."""
        return self.override_history.copy()

    def get_sources_summary(self) -> Dict[str, int]:
        """Get summary of configuration sources."""
        summary = {source.value: 0 for source in ConfigSource}
        for config_value in self.values.values():
            summary[config_value.source.value] += 1
        return summary

    def print_config(
        self,
        filter_source: Optional[ConfigSource] = None,
        show_overrides_only: bool = False,
    ) -> None:
        """
        Print configuration with provenance information.

        Args:
            filter_source: Only show values from this source
            show_overrides_only: Only show values that were overridden
        """
        print("\n=== Configuration Provenance ===")

        if show_overrides_only and not self.override_history:
            print("No configuration overrides")
            return

        # Group by source
        by_source: Dict[str, List[tuple]] = {source.value: [] for source in ConfigSource}

        for key, config_value in sorted(self.values.items()):
            if filter_source and config_value.source != filter_source:
                continue
            if show_overrides_only and not config_value.overridden:
                continue

            by_source[config_value.source.value].append((key, config_value))

        # Print by source
        for source in ConfigSource:
            items = by_source[source.value]
            if not items:
                continue

            print(f"\n[{source.value.upper()}] ({len(items)} values)")
            for key, config_value in items:
                print(f"  {key} = {config_value}")

        # Print override summary
        if self.override_history:
            print(f"\n[OVERRIDES] ({len(self.override_history)} total)")
            for override in self.override_history[-10:]:  # Show last 10
                print(
                    f"  {override['key']}: {override['old_value']} "
                    f"({override['old_source']}) → {override['new_value']} "
                    f"({override['new_source']})"
                )
            if len(self.override_history) > 10:
                print(f"  ... and {len(self.override_history) - 10} more")

        # Print summary
        summary = self.get_sources_summary()
        print("\n[SUMMARY]")
        for source, count in summary.items():
            if count > 0:
                print(f"  {source}: {count} values")
        print()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tracked configuration to a dictionary (values only).

        Returns:
            Dictionary with all configuration values
        """
        result = {}
        for key, config_value in self.values.items():
            # Build nested dict structure
            keys = key.split('.')
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = config_value.value
        return result

    def to_provenance_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to dictionary with full provenance information.

        Returns:
            Dictionary mapping keys to provenance info
        """
        return {
            key: {
                'value': cv.value,
                'source': cv.source.value,
                'precedence': cv.precedence,
                'yaml_file': cv.yaml_file,
                'cli_arg': cv.cli_arg,
                'env_var': cv.env_var,
                'overridden': cv.overridden,
                'previous_value': cv.previous_value,
            }
            for key, cv in self.values.items()
        }


# Global provenance tracker instance
_global_tracker: Optional[ConfigProvenanceTracker] = None


def get_global_tracker() -> ConfigProvenanceTracker:
    """Get or create the global provenance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ConfigProvenanceTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global provenance tracker."""
    global _global_tracker
    _global_tracker = ConfigProvenanceTracker()
