"""
Path utilities for resolving project-relative and config paths.

This module provides a centralized way to resolve paths relative to the project
root, making the codebase portable across different installation locations.

Usage:
    from Ava.utils.paths import get_project_root, get_data_dir, get_models_dir

    # Get project root (automatically detected)
    root = get_project_root()

    # Get standard directories
    data_dir = get_data_dir()
    models_dir = get_models_dir()
    outputs_dir = get_outputs_dir()
    configs_dir = get_configs_dir()

    # Resolve paths relative to project root
    my_file = resolve_path("code/data/processed/file.arrow")
"""

import os
import sys
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Detect and return the project root directory.

    Searches for the project root by looking for marker files/directories:
    - .git directory (git repository)
    - pyproject.toml (Python project marker)
    - .project directory (project config directory)

    Falls back to the parent of the code directory if found.

    Returns:
        Path: The project root directory

    Raises:
        RuntimeError: If project root cannot be detected
    """
    # Check if PROJECT_ROOT environment variable is set
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()

    # Start from the directory containing this file
    current = Path(__file__).resolve()

    # Walk up the directory tree looking for marker files
    # Check .git first (most reliable indicator of project root)
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent

    # If no .git found, check for .project (custom marker)
    for parent in current.parents:
        if (parent / ".project").exists():
            return parent

    # As last resort, check for pyproject.toml
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # If we're inside code/src/Ava, go up to find project root
    # code/src/Ava/utils/paths.py -> up 4 levels -> /project
    if current.parts[-4:-1] == ("code", "src", "Ava"):
        project_root = current.parents[4]
        if (project_root / "code").exists():
            return project_root

    raise RuntimeError(
        f"Could not detect project root from {current}. "
        "Please set PROJECT_ROOT environment variable."
    )


def get_code_dir() -> Path:
    """Get the code directory (project_root/code)."""
    return get_project_root() / "code"


def get_data_dir(data_type: str = "pretokenized") -> Path:
    """
    Get the data directory.

    Args:
        data_type: Type of data directory. Options: 'pretokenized', 'processed',
                   'rlhf', or custom path. Defaults to 'pretokenized'.

    Returns:
        Path: The data directory

    Examples:
        >>> get_data_dir()  # Returns code/data/pretokenized
        >>> get_data_dir("processed")  # Returns code/data/processed
        >>> get_data_dir("rlhf")  # Returns code/data/rlhf
    """
    # Check for DATA_DIR environment variable
    if "DATA_DIR" in os.environ:
        return Path(os.environ["DATA_DIR"]).resolve()

    data_base = get_code_dir() / "data"

    if data_type == "pretokenized":
        return data_base / "pretokenized"
    elif data_type == "processed":
        return data_base / "processed"
    elif data_type == "rlhf":
        return data_base / "rlhf"
    else:
        # Treat as relative path
        return data_base / data_type


def get_models_dir(model_type: str = "tokenizer") -> Path:
    """
    Get the models directory.

    Args:
        model_type: Type of model directory. Options: 'tokenizer' or custom path.
                    Defaults to 'tokenizer'.

    Returns:
        Path: The models directory

    Examples:
        >>> get_models_dir()  # Returns code/models/tokenizer
        >>> get_models_dir("custom")  # Returns code/models/custom
    """
    # Check for MODELS_DIR environment variable
    if "MODELS_DIR" in os.environ:
        return Path(os.environ["MODELS_DIR"]).resolve()

    models_base = get_code_dir() / "models"

    if model_type == "tokenizer":
        return models_base / "tokenizer"
    else:
        return models_base / model_type


def get_tokenizer_path(tokenizer_name: str = "enhanced-50680") -> Path:
    """
    Get the full path to a tokenizer.

    Args:
        tokenizer_name: Name of the tokenizer directory.
                       Defaults to 'enhanced-50680'.

    Returns:
        Path: Full path to the tokenizer

    Examples:
        >>> get_tokenizer_path()  # Returns code/models/tokenizer/enhanced-50680
        >>> get_tokenizer_path("enhanced-65536")  # Returns code/models/tokenizer/enhanced-65536
    """
    return get_models_dir() / tokenizer_name


def get_outputs_dir() -> Path:
    """
    Get the outputs directory.

    Returns:
        Path: The outputs directory (code/outputs)
    """
    # Check for OUTPUTS_DIR environment variable
    if "OUTPUTS_DIR" in os.environ:
        return Path(os.environ["OUTPUTS_DIR"]).resolve()

    return get_code_dir() / "outputs"


def get_configs_dir() -> Path:
    """
    Get the configs directory.

    Returns:
        Path: The configs directory (code/configs)
    """
    # Check for CONFIGS_DIR environment variable
    if "CONFIGS_DIR" in os.environ:
        return Path(os.environ["CONFIGS_DIR"]).resolve()

    return get_code_dir() / "configs"


def resolve_path(relative_path: str, base: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the project root or a specified base.

    Args:
        relative_path: Path relative to base (or project root if base is None)
        base: Base directory to resolve from. Defaults to project root.

    Returns:
        Path: Resolved absolute path

    Examples:
        >>> resolve_path("code/data/processed")
        Path('/project/code/data/processed')

        >>> resolve_path("file.txt", get_outputs_dir())
        Path('/project/code/outputs/file.txt')
    """
    if base is None:
        base = get_project_root()

    resolved = (base / relative_path).resolve()
    return resolved


def add_src_to_path():
    """
    Add the src directory to Python path.

    This is a convenience function for scripts that need to import from
    the Ava package. Use this instead of sys.path.insert() with hardcoded paths.

    Examples:
        >>> add_src_to_path()
        >>> from Ava.utils import paths  # Now this works from anywhere
    """
    src_dir = str(get_code_dir() / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def create_data_dirs() -> None:
    """
    Create standard data directories if they don't exist.

    Useful for initialization scripts and setup routines.
    """
    for data_type in ["pretokenized", "processed", "rlhf"]:
        data_dir = get_data_dir(data_type)
        data_dir.mkdir(parents=True, exist_ok=True)


def create_output_dirs() -> None:
    """
    Create standard output directories if they don't exist.

    Useful for training scripts that need output directories.
    """
    outputs_dir = get_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Create common subdirectories
    (outputs_dir / "runs").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "finetune_runs").mkdir(parents=True, exist_ok=True)


__all__ = [
    "get_project_root",
    "get_code_dir",
    "get_data_dir",
    "get_models_dir",
    "get_tokenizer_path",
    "get_outputs_dir",
    "get_configs_dir",
    "resolve_path",
    "add_src_to_path",
    "create_data_dirs",
    "create_output_dirs",
]
