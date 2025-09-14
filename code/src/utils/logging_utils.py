"""
Logging utilities for MoE++ model
"""
import logging
import sys
from typing import Optional
from pathlib import Path
import datetime
import torch

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: Optional[str] = None
):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format: Optional log format string
    """
    # Default format
    if format is None:
        format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set levels for specific loggers to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized at level {level}")

class RankFilter(logging.Filter):
    """Filter to only log from specific rank in distributed training"""
    def __init__(self, rank: int = 0):
        self.rank = rank
    
    def filter(self, record):
        try:
            from .parallel_utils import get_rank
            return get_rank() == self.rank
        except:
            return True

def get_logger(name: str) -> logging.Logger:
    """Get logger with given name"""
    return logging.getLogger(name)

def log_metrics(metrics: dict, step: int, prefix: str = ""):
    """Log metrics in a formatted way"""
    logger = logging.getLogger("metrics")
    
    metric_str = f"Step {step}"
    if prefix:
        metric_str = f"{prefix} - {metric_str}"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_str += f" | {key}: {value:.4f}"
        else:
            metric_str += f" | {key}: {value}"
    
    logger.info(metric_str)

def log_model_info(model: torch.nn.Module):
    """Log model information"""
    logger = logging.getLogger("model")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    logger.info(f"  Model size: {size_mb:.2f} MB")