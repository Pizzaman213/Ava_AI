"""Path utilities for deployment-ready configurations"""

import os
from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """Get the project root directory"""
    # This will work when the package is installed properly
    try:
        # First try to find the LLM package root
        import LLM
        return Path(LLM.__file__).parent
    except (ImportError, AttributeError):
        # Fallback to finding the directory containing this file
        current_file = Path(__file__).resolve()
        # Go up from src/utils/path_utils.py to get to LLM/
        return current_file.parent.parent.parent


def get_data_dir(relative_path: str = "") -> Path:
    """Get data directory path"""
    base_path = os.environ.get("LLM_DATA_DIR", get_project_root() / "data")
    return Path(base_path) / relative_path


def get_checkpoints_dir(relative_path: str = "") -> Path:
    """Get checkpoints directory path"""
    base_path = os.environ.get("LLM_CHECKPOINTS_DIR", get_project_root() / "checkpoints")
    return Path(base_path) / relative_path


def get_outputs_dir(relative_path: str = "") -> Path:
    """Get outputs directory path"""
    base_path = os.environ.get("LLM_OUTPUTS_DIR", get_project_root() / "outputs")
    return Path(base_path) / relative_path


def get_configs_dir(relative_path: str = "") -> Path:
    """Get configs directory path"""
    base_path = os.environ.get("LLM_CONFIGS_DIR", get_project_root() / "configs")
    return Path(base_path) / relative_path


def get_scripts_dir(relative_path: str = "") -> Path:
    """Get scripts directory path"""
    base_path = os.environ.get("LLM_SCRIPTS_DIR", get_project_root() / "scripts")
    return Path(base_path) / relative_path


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir(subdir: str = "") -> Path:
    """Get cache directory with XDG compliance"""
    # Follow XDG Base Directory Specification
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        cache_base = Path(xdg_cache) / "llm"
    else:
        cache_base = Path.home() / ".cache" / "llm"
    
    # Allow override
    cache_base = Path(os.environ.get("LLM_CACHE_DIR", cache_base))
    
    if subdir:
        return cache_base / subdir
    return cache_base