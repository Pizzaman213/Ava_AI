"""
Core utilities for the Ava framework.

This module provides fundamental utilities used throughout the codebase:
- paths: Project path resolution
- logging: Colored console logging
- mixed_precision: AMP training utilities
- activations: Activation function factory
- data_utils: Data loading utilities
- script_utils: Training script helpers
"""

from .paths import (
    get_project_root,
    get_code_dir,
    get_data_dir,
    get_models_dir,
    get_tokenizer_path,
    get_outputs_dir,
    get_configs_dir,
    resolve_path,
    add_src_to_path,
    create_data_dirs,
    create_output_dirs,
)

from .logging import (
    Colors,
    supports_color,
    ColoredFormatter,
    CleanFormatter,
    setup_colored_logging,
    configure_root_logger,
    get_logger,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
)

from .mixed_precision import (
    MixedPrecisionConfig,
    setup_mixed_precision,
    save_checkpoint,
    load_checkpoint,
    clip_gradients_and_step,
)

from .activations import (
    ACTIVATION_REGISTRY,
    GATED_ACTIVATIONS,
    get_activation,
    is_gated_activation,
)

from .data_utils import (
    find_data_files,
    collate_batch,
    BaseCollator,
    move_to_device,
    get_config_value,
    get_num_workers,
    get_prefetch_factor,
    get_persistent_workers,
    get_samples_per_file,
    get_enable_bucketing,
    get_val_split_ratio,
)

from .script_utils import (
    auto_install_requirements,
    setup_training_logging,
    suppress_noisy_loggers,
    configure_stdout_buffering,
)

__all__ = [
    # paths
    'get_project_root',
    'get_code_dir',
    'get_data_dir',
    'get_models_dir',
    'get_tokenizer_path',
    'get_outputs_dir',
    'get_configs_dir',
    'resolve_path',
    'add_src_to_path',
    'create_data_dirs',
    'create_output_dirs',
    # logging
    'Colors',
    'supports_color',
    'ColoredFormatter',
    'CleanFormatter',
    'setup_colored_logging',
    'configure_root_logger',
    'get_logger',
    'print_header',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    # mixed_precision
    'MixedPrecisionConfig',
    'setup_mixed_precision',
    'save_checkpoint',
    'load_checkpoint',
    'clip_gradients_and_step',
    # activations
    'ACTIVATION_REGISTRY',
    'GATED_ACTIVATIONS',
    'get_activation',
    'is_gated_activation',
    # data_utils
    'find_data_files',
    'collate_batch',
    'BaseCollator',
    'move_to_device',
    'get_config_value',
    'get_num_workers',
    'get_prefetch_factor',
    'get_persistent_workers',
    'get_samples_per_file',
    'get_enable_bucketing',
    'get_val_split_ratio',
    # script_utils
    'auto_install_requirements',
    'setup_training_logging',
    'suppress_noisy_loggers',
    'configure_stdout_buffering',
]
