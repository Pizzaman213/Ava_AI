"""
Centralized constants and magic numbers for Ava MoE++ training.

This module extracts all hardcoded values into named constants for better
maintainability and configurability.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DataPipelineConstants:
    """Constants for data loading and processing pipeline."""

    # Dynamic token batching
    MAX_TOKENS_DEFAULT: int = 8192  # Maximum tokens per batch
    MAX_BATCH_SIZE_DEFAULT: int = 64  # Maximum batch size

    # Bucketing
    MAX_BUCKET_SIZE: int = 200  # Increased from 100 for 5-10% speedup
    MIN_BUCKET_SIZE: int = 8
    MAX_TOKENS_PER_BATCH: int = 8192
    BUCKET_BOUNDARIES_DEFAULT: List[int] = None  # Will be set in __post_init__

    # File handling
    BUFFER_SIZE_DEFAULT: int = 10000  # Default shuffle buffer size
    SAMPLES_PER_FILE: int = 32  # File sampling rate
    STREAMING_BUFFER_SIZE: int = 1000  # Streaming mode buffer (saves 500MB-1GB RAM)
    MIN_FILE_SIZE_BYTES: int = 10 * 1024  # 10KB minimum file size
    MIN_TOKENIZE_BATCH: int = 32  # Minimum batch size for vectorized tokenization

    # Parquet reading
    PARQUET_BATCH_SIZE: int = 5000  # OPTIMIZED: Reduced from 10000 for better memory efficiency
    MIN_TEXT_LENGTH: int = 10  # Minimum text length after stripping whitespace

    # Async prefetching
    PREFETCH_MAX_WORKERS: int = 4  # Thread pool size for async prefetching
    PREFETCH_SIZE: int = 2  # Number of files to prefetch ahead

    # Worker file cache
    WORKER_FILE_CACHE_MAX_SIZE: int = 100  # LRU cache size for file generators

    def __post_init__(self):
        """Initialize computed constants."""
        if self.BUCKET_BOUNDARIES_DEFAULT is None:
            # Optimized default boundaries based on common sequence lengths
            self.BUCKET_BOUNDARIES_DEFAULT = [64, 128, 256, 512, 1024, 2048, 4096]


@dataclass
class TrainerConstants:
    """Constants for training loop and optimization."""

    # Model size thresholds
    AUTOCAST_PARAM_THRESHOLD: int = 500_000_000  # 500M parameter threshold for auto-checkpointing

    # Gradient scaling
    GRAD_SCALER_INIT_SCALE: float = 2.0**16  # Initial gradient scale
    GRAD_SCALER_GROWTH_FACTOR: float = 2.0  # Growth factor for gradient scale
    GRAD_SCALER_BACKOFF_FACTOR: float = 0.5  # Backoff factor for gradient scale
    GRAD_SCALER_GROWTH_INTERVAL: int = 2000  # Steps between growth attempts
    GRAD_SCALER_RESET_INTERVAL: int = 1000  # Steps between scaler resets

    # Memory thresholds
    MEMORY_WARNING_THRESHOLD: float = 0.990  # 99.0% memory usage warning
    MEMORY_CRITICAL_THRESHOLD: float = 0.995  # 99.5% memory usage critical
    MEMORY_EMERGENCY_THRESHOLD: float = 0.999  # 99.9% memory usage emergency
    MEMORY_HEADROOM_GB: float = 2.0  # Reserved memory headroom in GB
    MEMORY_CLEAR_CACHE_FREQUENCY: int = 500  # Cache clear frequency (reduced from 10000)
    MEMORY_EMERGENCY_CHECK_FREQUENCY: int = 100  # Emergency memory check frequency

    # Attention checkpointing thresholds
    ATTENTION_CHECKPOINT_ENABLE_THRESHOLD: float = 0.92  # Enable at 92% memory
    ATTENTION_CHECKPOINT_DISABLE_THRESHOLD: float = 0.80  # Disable at 80% memory

    # Loss thresholds
    LOSS_SPIKE_THRESHOLD: float = 10.0  # Loss spike detection threshold (raised from 5.0)
    AUX_LOSS_CLAMP_FACTOR: float = 0.5  # Auxiliary loss clamping factor

    # Optimization factors
    BATCH_SIZE_REDUCTION_FACTOR: float = 0.8  # Batch size reduction on OOM (was 0.5)

    # Memory cleanup verification
    MEMORY_CLEANUP_DISCREPANCY_THRESHOLD_GB: float = 0.5  # Threshold for reporting discrepancies
    MEMORY_CLEANUP_MIN_LOG_THRESHOLD_GB: float = 0.1  # Minimum freed memory to log

    # Async cache clearing
    ASYNC_CACHE_CLEAR_POLL_INTERVAL_MS: int = 50  # Polling interval for cache clear thread


@dataclass
class MoEConstants:
    """Constants for Mixture of Experts layers."""

    # Routing
    ROUTING_CACHE_SIZE: int = 1024  # Maximum size of routing cache
    ROUTING_SAMPLE_SIZE: int = 32  # Sample size for tensor hashing
    ROUTING_HIT_RATE_INCREASE_THRESHOLD: float = 0.7  # Hit rate threshold for increasing prefetch
    ROUTING_HIT_RATE_DECREASE_THRESHOLD: float = 0.9  # Hit rate threshold for decreasing prefetch

    # Diversity loss approximation
    DIVERSITY_LOSS_APPROX_THRESHOLD: int = 512  # Use approximate diversity loss above this threshold
    DIVERSITY_LOSS_MAX_SAMPLE_SIZE: int = 128  # Max sample size for diversity loss (reduced from 512)

    # Expert offloading
    MIN_EXPERTS_FOR_BATCHED_PROCESSING: int = 4  # Minimum experts to use batched processing
    PREFETCH_DEPTH_MIN: int = 1  # Minimum prefetch depth
    PREFETCH_DEPTH_MAX: int = 5  # Maximum prefetch depth (increased from 3)
    PREFETCH_DEPTH_DEFAULT: int = 3  # Default prefetch depth
    PREFETCH_ADJUSTMENT_INTERVAL: int = 100  # Steps between prefetch depth adjustments
    PATTERN_HISTORY_SIZE: int = 1000  # Size of expert access pattern history


# Global constants instances (can be overridden by config)
DATA_CONSTANTS = DataPipelineConstants()
TRAINER_CONSTANTS = TrainerConstants()
MOE_CONSTANTS = MoEConstants()


def update_constants_from_config(config):
    """
    Update global constants from a configuration object.

    Args:
        config: Configuration object with optional 'constants' section
    """
    if not hasattr(config, 'constants'):
        return

    constants_config = config.constants

    # Update data pipeline constants
    if hasattr(constants_config, 'data_pipeline'):
        for key, value in constants_config.data_pipeline.items():
            if hasattr(DATA_CONSTANTS, key.upper()):
                setattr(DATA_CONSTANTS, key.upper(), value)

    # Update trainer constants
    if hasattr(constants_config, 'trainer'):
        for key, value in constants_config.trainer.items():
            if hasattr(TRAINER_CONSTANTS, key.upper()):
                setattr(TRAINER_CONSTANTS, key.upper(), value)

    # Update MoE constants
    if hasattr(constants_config, 'moe'):
        for key, value in constants_config.moe.items():
            if hasattr(MOE_CONSTANTS, key.upper()):
                setattr(MOE_CONSTANTS, key.upper(), value)
