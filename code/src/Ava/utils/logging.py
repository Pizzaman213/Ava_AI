"""
Logging utilities for training and evaluation.

This module provides consistent logging configuration across all components
of the Qwen MoE++ implementation, ensuring proper tracking of training progress
and debugging information.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logging configuration for training and evaluation.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        log_dir (str): Directory for log files

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(log_level="INFO", log_file="training.log")
        >>> logger.info("Starting training...")
    """
    # Create logger
    logger = logging.getLogger("qwen_moe")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp to log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{timestamp}_{log_file}"

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    return logger


def get_logger(name: str = "qwen_moe") -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)