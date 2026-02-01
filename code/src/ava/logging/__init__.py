"""
Ava Logging Module.

Centralized logging infrastructure for the Ava training framework.

Submodules:
- console: Colored console logging, async file handlers
- experiment: WandB integration, MetricsManager
- metrics: Async non-blocking metrics logging
- diagnostics: Training diagnostics (gradients, routing, memory, timing)

Usage:
    # Console logging
    from ava.logging.console.colored import setup_ava_logging, print_header

    # WandB/experiment tracking
    from ava.logging.wandb.wandb import WandBLogger, MetricsManager

    # Async metrics
    from ava.logging.metrics.async_logger import AsyncMetricsLogger

    # Diagnostics
    from ava.logging.diagnostics.training import DiagnosticsManager
"""

# Console logging exports
from ava.logging.console.colored import (
    Colors,
    Icons,
    ColoredFormatter,
    CleanFormatter,
    TqdmLoggingHandler,
    AsyncLoggingHandler,
    ParallelLoggerPool,
    PipelineLogger,
    setup_pipeline_logging,
    setup_ava_logging,
    setup_colored_logging,
    configure_root_logger,
    get_logger,
    supports_color,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_step,
    print_config,
    print_metric,
    print_progress,
    print_table_row,
    print_box,
    print_phase,
    print_training_step,
    print_epoch_summary,
    print_checkpoint,
    print_model_info,
    print_data_info,
    print_gpu_info,
    print_calibration,
)

# Experiment/WandB exports
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

# Async metrics exports
from ava.logging.metrics.async_logger import (
    AsyncMetricsLogger,
    MetricEntry,
    get_async_logger,
    shutdown_async_logger,
)

# Diagnostics exports
from ava.logging.diagnostics.training import (
    DiagnosticsManager,
    LayerGradientStats,
    ExpertRoutingStats,
    MemoryBreakdown,
    TimingProfile,
)

__all__ = [
    # Console
    'Colors',
    'Icons',
    'ColoredFormatter',
    'CleanFormatter',
    'TqdmLoggingHandler',
    'AsyncLoggingHandler',
    'ParallelLoggerPool',
    'PipelineLogger',
    'setup_pipeline_logging',
    'setup_ava_logging',
    'setup_colored_logging',
    'configure_root_logger',
    'get_logger',
    'supports_color',
    'print_header',
    'print_subheader',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_step',
    'print_config',
    'print_metric',
    'print_progress',
    'print_table_row',
    'print_box',
    'print_phase',
    'print_training_step',
    'print_epoch_summary',
    'print_checkpoint',
    'print_model_info',
    'print_data_info',
    'print_gpu_info',
    'print_calibration',
    # Experiment/WandB
    'WandBLogger',
    'MetricsManager',
    'WANDB_AVAILABLE',
    'TENSORBOARD_AVAILABLE',
    'get_wandb_logger',
    'init_wandb',
    'log_wandb',
    'finish_wandb',
    # Async Metrics
    'AsyncMetricsLogger',
    'MetricEntry',
    'get_async_logger',
    'shutdown_async_logger',
    # Diagnostics
    'DiagnosticsManager',
    'LayerGradientStats',
    'ExpertRoutingStats',
    'MemoryBreakdown',
    'TimingProfile',
]
