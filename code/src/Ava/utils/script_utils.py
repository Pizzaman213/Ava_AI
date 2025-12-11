"""
Script Utilities - Common functionality for training scripts.

This module eliminates code duplication across training scripts by providing:
- auto_install_requirements(): Auto-install Python dependencies
- setup_training_logging(): Configure logging for training scripts
- get_project_root(): Find project root directory

Usage:
    from src.Ava.utils.script_utils import auto_install_requirements, setup_training_logging

    # At script start
    auto_install_requirements()

    # After other imports
    logger = setup_training_logging(log_dir, rank=0)
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory.

    Looks for common project markers (requirements.txt, .git, setup.py).

    Args:
        start_path: Starting path for search (defaults to current file's directory)

    Returns:
        Path to project root directory
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    # Walk up directory tree looking for project markers
    current = start_path
    for _ in range(10):  # Limit search depth
        if (current / "requirements.txt").exists():
            return current
        if (current / ".git").exists():
            return current
        if (current / "setup.py").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # Fallback: assume we're in src/Ava/utils
    return Path(__file__).resolve().parents[4]


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
        from src.Ava.utils.script_utils import auto_install_requirements
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


def setup_training_logging(
    log_dir: Optional[Path] = None,
    rank: int = 0,
    logger_name: str = "training",
    level: int = logging.INFO,
    log_to_file: bool = True,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup logging for training scripts.

    Creates a logger with console and optionally file handlers.
    Supports colored output if available.

    Args:
        log_dir: Directory for log files (creates if needed)
        rank: Process rank for distributed training (only rank 0 logs to console)
        logger_name: Name for the logger
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        use_colors: Whether to use colored console output (if available)

    Returns:
        Configured logger instance

    Example:
        logger = setup_training_logging(Path("logs"), rank=0)
        logger.info("Starting training...")
    """
    logger = logging.getLogger(logger_name)

    # Clear existing handlers to avoid duplicates
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
            try:
                from .colored_logging import ColoredFormatter, supports_color
                if supports_color():
                    formatter = ColoredFormatter(show_level=False)
            except ImportError:
                pass

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

    return logger


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
