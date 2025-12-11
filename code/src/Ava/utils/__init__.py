"""
Utility Module

General utility functions for the Ava training framework.
"""

from .paths import get_project_root, resolve_path

# Checkpoint management (unified async checkpoint manager)
from .checkpoint_manager import CheckpointManager

# Colored logging utilities
from .colored_logging import (
    ColoredFormatter,
    CleanFormatter,
    Colors,
    setup_colored_logging,
    configure_root_logger,
    get_logger,
    supports_color,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
)

# Shared utilities (consolidated from duplicate code)
from .shared import (
    # Mixed precision
    MixedPrecisionConfig,
    setup_mixed_precision,
    # Checkpoints
    save_checkpoint,
    load_checkpoint,
    # Data utilities
    find_data_files,
    collate_batch,
    BaseCollator,
    # Activation factory
    get_activation,
    is_gated_activation,
    # Config getters
    get_config_value,
    get_num_workers,
    get_prefetch_factor,
    get_persistent_workers,
    get_samples_per_file,
    get_enable_bucketing,
    get_val_split_ratio,
    extract_dynamic_batching_config,
    # Device utilities
    move_to_device,
    # Gradient utilities
    clip_gradients_and_step,
)

# Script utilities (for training scripts)
from .script_utils import (
    auto_install_requirements,
    setup_training_logging,
    suppress_noisy_loggers,
    configure_stdout_buffering,
    get_project_root as get_project_root_from_script,
)

# Nsight profiling utilities
from .nsight_profiler import (
    NsightProfiler,
    ProfilerConfig,
    NVTXAnnotator,
    create_profiler_from_config,
    enable_nsight_profiling,
)

# CUDA stream optimization utilities
from .cuda_streams import (
    StreamPool,
    CUDATimer,
    NonBlockingTransfer,
    BatchSynchronizer,
    cuda_timed_region,
    optimal_sync_context,
    get_stream_pool,
    get_transfer_utils,
    efficient_sync,
    sync_if_needed,
    wait_stream,
    record_and_wait,
)

__all__ = [
    "get_project_root",
    "resolve_path",
    "get_logger",
    # Checkpoint management
    "CheckpointManager",
    # Colored logging
    "ColoredFormatter",
    "CleanFormatter",
    "Colors",
    "setup_colored_logging",
    "configure_root_logger",
    "supports_color",
    "print_header",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    # Shared utilities
    "MixedPrecisionConfig",
    "setup_mixed_precision",
    "save_checkpoint",
    "load_checkpoint",
    "find_data_files",
    "collate_batch",
    "BaseCollator",
    "get_activation",
    "is_gated_activation",
    "get_config_value",
    "get_num_workers",
    "get_prefetch_factor",
    "get_persistent_workers",
    "get_samples_per_file",
    "get_enable_bucketing",
    "get_val_split_ratio",
    "extract_dynamic_batching_config",
    "move_to_device",
    "clip_gradients_and_step",
    # Script utilities
    "auto_install_requirements",
    "setup_training_logging",
    "suppress_noisy_loggers",
    "configure_stdout_buffering",
    # Nsight profiling
    "NsightProfiler",
    "ProfilerConfig",
    "NVTXAnnotator",
    "create_profiler_from_config",
    "enable_nsight_profiling",
    # CUDA stream optimization
    "StreamPool",
    "CUDATimer",
    "NonBlockingTransfer",
    "BatchSynchronizer",
    "cuda_timed_region",
    "optimal_sync_context",
    "get_stream_pool",
    "get_transfer_utils",
    "efficient_sync",
    "sync_if_needed",
    "wait_stream",
    "record_and_wait",
]
