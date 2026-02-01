"""
Console Logging Module.

Provides colorful, async-capable logging for training pipelines.

Features:
- ANSI color support with auto-detection
- Level-based coloring with icons
- Async/parallel file logging with worker threads (non-blocking I/O)
- Pipeline-aware directory structure (integrates with RunManager)
- Metric highlighting (loss, LR, memory, etc.)
- Beautiful section headers and status indicators
- Distributed training support (rank-aware logging)

Usage:
    from ava.logging.console import setup_ava_logging, print_header

    logger = setup_ava_logging(level=logging.INFO)
    print_header("Training Configuration")
"""

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

__all__ = [
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
]
