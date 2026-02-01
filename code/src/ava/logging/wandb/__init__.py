"""
WandB Logging Module.

Provides WandB and TensorBoard integration for experiment tracking.

Features:
- WandB logging with batching and async support
- TensorBoard logging
- Metric accumulation and storage
- Generation table logging
- Unified MetricsManager for training pipelines

Usage:
    from ava.logging.wandb import WandBLogger, MetricsManager

    # Standalone WandB logging
    logger = WandBLogger()
    logger.init(project="my-project", config={"lr": 0.001})
    logger.log({"loss": 0.5}, step=100)
    logger.finish()
"""

from ava.logging.wandb.wandb import (
    WandBLogger,
    MetricsManager,
    WANDB_AVAILABLE,
    TENSORBOARD_AVAILABLE,
    get_wandb_logger,
    init_wandb,
    log_wandb,
    finish_wandb,
)

__all__ = [
    'WandBLogger',
    'MetricsManager',
    'WANDB_AVAILABLE',
    'TENSORBOARD_AVAILABLE',
    'get_wandb_logger',
    'init_wandb',
    'log_wandb',
    'finish_wandb',
]
