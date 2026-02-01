"""
Script Utilities - Common functionality for training scripts.

This module eliminates code duplication across training scripts by providing:
- auto_install_requirements(): Auto-install Python dependencies
- setup_training_logging(): Configure logging for training scripts
- get_project_root(): Find project root directory (re-exported from paths.py)

Usage:
    from ava.core.script_utils import auto_install_requirements, setup_training_logging

    # At script start
    auto_install_requirements()

    # After other imports
    logger = setup_training_logging(log_dir, rank=0)
"""

import functools
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Set, Tuple, Type, Callable

# Import get_project_root from centralized paths module
from .paths import get_project_root

# Track configured loggers to avoid redundant setup
_configured_loggers: Set[str] = set()


def auto_install_requirements(
    project_root: Optional[Path] = None,
    requirements_file: str = "requirements.txt",
    quiet: bool = True,
) -> bool:
    """
    Auto-install requirements if imports fail.

    This function should be called early in script execution,
    before other imports that might fail.

    Args:
        project_root: Project root directory (auto-detected if None)
        requirements_file: Name of requirements file
        quiet: Suppress pip output if True

    Returns:
        True if installation succeeded, False otherwise

    Example:
        # At top of training script, before other imports
        from ava.core.script_utils import auto_install_requirements
        auto_install_requirements()
    """
    if project_root is None:
        project_root = get_project_root()

    req_path = project_root / requirements_file

    if not req_path.exists():
        print(f"Requirements file not found: {req_path}")
        return False

    print("Installing Python requirements...")
    try:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(req_path), "--upgrade"
        ]
        if quiet:
            cmd.append("-q")

        subprocess.check_call(cmd)
        print("Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"Failed to install requirements: {e}")
        return False


@functools.lru_cache(maxsize=1)
def _get_colored_formatter() -> Tuple[Optional[Type], Callable[[], bool]]:
    """
    Lazy-load ColoredFormatter and supports_color with caching.

    Returns:
        Tuple of (ColoredFormatter class or None, supports_color function)
    """
    try:
        from ava.logging.console.colored import ColoredFormatter, supports_color
        return ColoredFormatter, supports_color
    except ImportError:
        return None, lambda: False


def setup_training_logging(
    log_dir: Optional[Path] = None,
    rank: int = 0,
    logger_name: str = "training",
    level: int = logging.INFO,
    log_to_file: bool = True,
    use_colors: bool = True,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Setup logging for training scripts.

    Creates a logger with console and optionally file handlers.
    Supports colored output if available.
    Caches configuration to avoid redundant setup on repeated calls.

    Args:
        log_dir: Directory for log files (creates if needed)
        rank: Process rank for distributed training (only rank 0 logs to console)
        logger_name: Name for the logger
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        use_colors: Whether to use colored console output (if available)
        force_reconfigure: If True, reconfigure even if already configured

    Returns:
        Configured logger instance

    Example:
        logger = setup_training_logging(Path("logs"), rank=0)
        logger.info("Starting training...")
    """
    logger = logging.getLogger(logger_name)

    # Check if already configured (skip redundant setup)
    cache_key = f"{logger_name}_{rank}_{log_dir}"
    if cache_key in _configured_loggers and not force_reconfigure:
        return logger

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Only log to console on rank 0 in distributed training
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Try to use colored formatting if available and requested
        formatter = None
        if use_colors:
            ColoredFormatter, supports_color = _get_colored_formatter()
            if ColoredFormatter and supports_color():
                formatter = ColoredFormatter(show_level=False)

        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (all ranks)
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / f"training_rank{rank}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    # Mark as configured
    _configured_loggers.add(cache_key)

    return logger


def clear_logger_cache() -> None:
    """Clear the configured loggers cache. Useful for testing."""
    _configured_loggers.clear()


def suppress_noisy_loggers(loggers: Optional[list] = None) -> None:
    """
    Suppress verbose logging from common libraries.

    Args:
        loggers: List of logger names to suppress (uses defaults if None)
    """
    if loggers is None:
        loggers = [
            'transformers',
            'datasets',
            'tokenizers',
            'torch.distributed',
            'urllib3',
            'filelock',
        ]

    for name in loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger(name).propagate = False


def configure_stdout_buffering(unbuffered: bool = True) -> None:
    """
    Configure stdout buffering for real-time output.

    Args:
        unbuffered: If True, disable buffering for immediate output
    """
    if unbuffered:
        # Force unbuffered stdout
        sys.stdout.reconfigure(line_buffering=True)


__all__ = [
    'get_project_root',
    'auto_install_requirements',
    'setup_training_logging',
    'suppress_noisy_loggers',
    'configure_stdout_buffering',
    'clear_logger_cache',
]
