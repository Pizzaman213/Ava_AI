#!/usr/bin/env python3
"""
Configuration Migration Tool for Ava AI Training Framework.

Migrates old configuration files to the new v2.0 structure with 8 main categories:
  1. model, 2. training, 3. data, 4. compute, 5. distributed,
  6. logging, 7. checkpoints, 8. experimental

Usage:
    # Validate a config (check for deprecated paths)
    python scripts/migrate_config.py old_config.yaml --validate

    # Show what would be migrated (dry run)
    python scripts/migrate_config.py old_config.yaml --dry-run

    # Migrate to new file
    python scripts/migrate_config.py old_config.yaml --output new_config.yaml

    # Migrate in place (backup created automatically)
    python scripts/migrate_config.py old_config.yaml --in-place

    # List all deprecated paths found
    python scripts/migrate_config.py old_config.yaml --list-deprecated
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set

import yaml

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from ava.config.path_mapping import CONFIG_PATH_MAPPINGS, is_deprecated_path, get_new_path
except ImportError:
    print("Error: Could not import path_mapping module.")
    print("Make sure you're running from the project root.")
    sys.exit(1)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_yaml(config: Dict[str, Any], path: str) -> None:
    """Save a configuration to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def find_deprecated_paths(config: Dict[str, Any], prefix: str = '') -> List[Tuple[str, str]]:
    """
    Recursively find all deprecated paths in a config.

    Returns:
        List of (old_path, new_path) tuples
    """
    deprecated = []

    for key, value in config.items():
        current_path = f"{prefix}.{key}" if prefix else key

        # Check if this path is deprecated
        if is_deprecated_path(current_path):
            new_path = get_new_path(current_path)
            deprecated.append((current_path, new_path))

        # Check if just the key (top-level section) is deprecated
        if not prefix and is_deprecated_path(key):
            new_path = get_new_path(key)
            deprecated.append((key, new_path))

        # Recurse into nested dicts
        if isinstance(value, dict):
            deprecated.extend(find_deprecated_paths(value, current_path))

    return deprecated


def get_value_at_path(config: Dict[str, Any], path: str) -> Any:
    """Get value at a dot-separated path."""
    keys = path.split('.')
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def set_value_at_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set value at a dot-separated path, creating intermediate dicts as needed."""
    keys = path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def migrate_config(old_config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, str, Any]]]:
    """
    Migrate old config to new structure.

    Returns:
        Tuple of (new_config, list of migrations performed)
    """
    new_config = {}
    migrations = []

    # First, copy over everything that's not deprecated
    def copy_non_deprecated(src: Dict[str, Any], dst: Dict[str, Any], prefix: str = ''):
        for key, value in src.items():
            current_path = f"{prefix}.{key}" if prefix else key

            # Skip deprecated top-level sections (they'll be migrated)
            if not prefix and is_deprecated_path(key):
                continue

            if isinstance(value, dict):
                if key not in dst:
                    dst[key] = {}
                copy_non_deprecated(value, dst[key], current_path)
            else:
                dst[key] = value

    copy_non_deprecated(old_config, new_config)

    # Now migrate deprecated paths
    deprecated = find_deprecated_paths(old_config)

    for old_path, new_path in deprecated:
        value = get_value_at_path(old_config, old_path)
        if value is not None:
            set_value_at_path(new_config, new_path, value)
            migrations.append((old_path, new_path, value))

    return new_config, migrations


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a config file.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Check for deprecated paths
    deprecated = find_deprecated_paths(config)
    for old_path, new_path in deprecated:
        warnings.append(f"Deprecated path '{old_path}' -> use '{new_path}' instead")

    # Check required fields
    required_fields = ['model.vocab_size', 'model.hidden_size']
    for field in required_fields:
        if get_value_at_path(config, field) is None:
            errors.append(f"Required field '{field}' is missing")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description='Migrate Ava AI configuration files to the new v2.0 structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'config_file',
        help='Path to the configuration file to migrate'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output path for migrated config (default: stdout)'
    )

    parser.add_argument(
        '--in-place', '-i',
        action='store_true',
        help='Migrate the file in place (creates backup)'
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Only validate the config, do not migrate'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be migrated without making changes'
    )

    parser.add_argument(
        '--list-deprecated', '-l',
        action='store_true',
        help='List all deprecated paths found in the config'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress informational output'
    )

    args = parser.parse_args()

    # Load the config
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml(config_path)

    # Handle --validate
    if args.validate:
        is_valid, errors, warnings = validate_config(config)

        if warnings and not args.quiet:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                print(f"  - {w}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            if not args.quiet:
                print(f"\nConfig is valid! ({len(warnings)} warnings)")
            sys.exit(0)

    # Handle --list-deprecated
    if args.list_deprecated:
        deprecated = find_deprecated_paths(config)

        if deprecated:
            print(f"\nDeprecated paths found ({len(deprecated)}):\n")
            for old_path, new_path in deprecated:
                value = get_value_at_path(config, old_path)
                print(f"  {old_path}")
                print(f"    -> {new_path}")
                if isinstance(value, dict):
                    print(f"       (section with {len(value)} keys)")
                else:
                    print(f"       = {value}")
                print()
        else:
            print("\nNo deprecated paths found. Config is already using new structure!")

        sys.exit(0)

    # Perform migration
    new_config, migrations = migrate_config(config)

    # Handle --dry-run
    if args.dry_run:
        if migrations:
            print(f"\nMigrations that would be performed ({len(migrations)}):\n")
            for old_path, new_path, value in migrations:
                print(f"  {old_path} -> {new_path}")
                if not isinstance(value, dict):
                    print(f"    value: {value}")
        else:
            print("\nNo migrations needed. Config is already using new structure!")

        sys.exit(0)

    # Handle --in-place
    if args.in_place:
        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = config_path.with_suffix(f'.backup_{timestamp}.yaml')
        shutil.copy(config_path, backup_path)

        if not args.quiet:
            print(f"Backup created: {backup_path}")

        save_yaml(new_config, config_path)

        if not args.quiet:
            print(f"Config migrated in place: {config_path}")
            print(f"Migrations performed: {len(migrations)}")

        sys.exit(0)

    # Handle --output
    if args.output:
        save_yaml(new_config, args.output)

        if not args.quiet:
            print(f"Migrated config saved to: {args.output}")
            print(f"Migrations performed: {len(migrations)}")

        sys.exit(0)

    # Default: print to stdout
    print(yaml.dump(new_config, default_flow_style=False, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    main()
