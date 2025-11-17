"""
Centralized constants and magic numbers for Ava MoE++ training.

This module extracts all hardcoded values into named constants for better
maintainability and configurability.
"""

from dataclasses import dataclass
from typing import List, Optional


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
    BUCKET_BOUNDARIES_DEFAULT: Optional[List[int]] = None  # Will be set in __post_init__

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
    PREFETCH_MAX_WORKERS: int = 16  # SPEED OPTIMIZATION: Increased for better concurrency (8→16)
    PREFETCH_SIZE: int = 8  # SPEED OPTIMIZATION: Deeper prefetch queue for better pipelining (4→8)

    # Worker file cache
    WORKER_FILE_CACHE_MAX_SIZE: int = 50  # RAM FIX: Reduced from 200 to save ~600MB (200→50)

    # Prefetch factor calculation
    PREFETCH_FACTOR_NUMERATOR: int = 4096  # ULTRA-FAST: Numerator for dynamic prefetch calculation (3072→4096)
    PREFETCH_FACTOR_MIN: int = 2  # Minimum prefetch factor
    PREFETCH_FACTOR_MAX: int = 16  # ULTRA-FAST: Maximum prefetch factor (6→16)

    # Buffer sizing
    INITIAL_FILL_SIZE_MAX: int = 50  # GPU UTIL FIX: Reduced from 200 to 50 for 10-30s faster startup
    INITIAL_FILL_SIZE_DIVISOR: int = 10  # Divisor for calculating initial fill size: min(buffer_size // DIVISOR, MAX)
    DYNAMIC_BUFFER_MIN: int = 2000  # Minimum dynamic buffer size
    DYNAMIC_BUFFER_MEMORY_PERCENT: float = 0.05  # Percentage of available GPU memory for buffer (5%)

    # Token balancing
    TOKEN_BALANCE_BATCH_SIZE: int = 32  # Buffer size for token-balanced distribution

    # Memory management
    MEMORY_PRESSURE_THRESHOLD: float = 0.85  # Memory usage threshold for cache reduction (85%)
    CACHE_REDUCTION_FACTOR: int = 2  # Factor to reduce cache size under memory pressure
    CACHE_MIN_SIZE_UNDER_PRESSURE: int = 10  # Minimum cache size under memory pressure

    # Profiling and monitoring
    MEMORY_CHECK_INTERVAL: int = 5000  # ULTRA-FAST: Check memory less often (1000→5000, saves sync overhead)
    PROFILING_REPORT_INTERVAL: int = 5000  # ULTRA-FAST: Report profiling stats less often (1000→5000)

    # Load balancing
    LOAD_BALANCE_CHECK_INTERVAL: int = 500  # ULTRA-FAST: Check load balance less often (100→500)
    LOAD_BALANCE_MAX_SKIP: int = 2  # ULTRA-FAST: Minimal skipping for better balance (5→2)

    # Adaptive file sampling
    ADAPTIVE_SAMPLES_LARGE_FILE_MB: float = 10.0  # File size threshold for "large" file (MB)
    ADAPTIVE_SAMPLES_MEDIUM_FILE_MB: float = 1.0  # File size threshold for "medium" file (MB)
    ADAPTIVE_SAMPLES_LARGE_MULTIPLIER: int = 4  # Multiplier for large files
    ADAPTIVE_SAMPLES_MEDIUM_MULTIPLIER: int = 2  # Multiplier for medium files
    ADAPTIVE_SAMPLES_MAX: int = 128  # Maximum adaptive samples per file
    ADAPTIVE_SAMPLES_MIN: int = 8  # Minimum adaptive samples per file
    ADAPTIVE_SAMPLES_SLOW_READ_THRESHOLD: float = 0.05  # Read time threshold for slow reads (seconds)
    ADAPTIVE_SAMPLES_SLOW_MULTIPLIER: float = 0.8  # Multiplier for slow reads
    ADAPTIVE_SAMPLES_FAST_MULTIPLIER: float = 1.2  # Multiplier for fast reads
    ADAPTIVE_READ_TIME_HISTORY_SIZE: int = 10  # Number of read times to track for averaging

    # Bucket flushing
    BUCKET_FLUSH_INTERVAL_MULTIPLIER: int = 2  # OPTIMIZED: Faster throughput (was 5, flushes more frequently)
    BUCKET_FLUSH_MIN_SIZE: int = 8  # Minimum bucket size for flushing at end

    # Collate buffer cache
    COLLATE_BUFFER_CACHE_MAX_SIZE: int = 50  # OPTIMIZED: Increased for better cache hit rate with variable batches (was 10)

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
    MEMORY_WARNING_THRESHOLD: float = 0.950  # ULTRA-FAST: Less aggressive warning (0.990→0.950)
    MEMORY_CRITICAL_THRESHOLD: float = 0.975  # ULTRA-FAST: Less aggressive critical (0.995→0.975)
    MEMORY_EMERGENCY_THRESHOLD: float = 0.990  # ULTRA-FAST: Less aggressive emergency (0.999→0.990)
    MEMORY_HEADROOM_GB: float = 0.5  # ULTRA-FAST: Minimal headroom (2.0→0.5)
    MEMORY_CLEAR_CACHE_FREQUENCY: int = 2000  # ULTRA-FAST: Cache clear less often (500→2000)
    MEMORY_EMERGENCY_CHECK_FREQUENCY: int = 500  # ULTRA-FAST: Emergency check less often (100→500)

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
    ROUTING_CACHE_SIZE: int = 512  # RAM FIX: Reduced from 4096 to save ~100MB (4096→512)
    ROUTING_SAMPLE_SIZE: int = 64  # ULTRA-FAST: Larger sample for better hashing (32→64)
    ROUTING_HIT_RATE_INCREASE_THRESHOLD: float = 0.6  # ULTRA-FAST: Earlier prefetch increase (0.7→0.6)
    ROUTING_HIT_RATE_DECREASE_THRESHOLD: float = 0.95  # ULTRA-FAST: Later prefetch decrease (0.9→0.95)

    # Diversity loss approximation
    DIVERSITY_LOSS_APPROX_THRESHOLD: int = 128  # SPEED OPTIMIZATION: Earlier approximation (256→128)
    DIVERSITY_LOSS_MAX_SAMPLE_SIZE: int = 64  # SPEED OPTIMIZATION: Smaller sample for speed (128→64)
    DIVERSITY_LOSS_COMPUTATION_FREQ: int = 10  # SPEED OPTIMIZATION: Compute every N steps (not every step)

    # Expert offloading
    MIN_EXPERTS_FOR_BATCHED_PROCESSING: int = 2  # ULTRA-FAST: Batch even with 2 experts (4→2)
    PREFETCH_DEPTH_MIN: int = 2  # ULTRA-FAST: Higher minimum prefetch (1→2)
    PREFETCH_DEPTH_MAX: int = 8  # ULTRA-FAST: Higher maximum prefetch (5→8)
    PREFETCH_DEPTH_DEFAULT: int = 5  # ULTRA-FAST: Higher default prefetch (3→5)
    PREFETCH_ADJUSTMENT_INTERVAL: int = 500  # ULTRA-FAST: Adjust less often (100→500)
    PATTERN_HISTORY_SIZE: int = 2000  # ULTRA-FAST: Larger pattern history (1000→2000)


# Global constants instances (can be overridden by config)
DATA_CONSTANTS = DataPipelineConstants()
TRAINER_CONSTANTS = TrainerConstants()
MOE_CONSTANTS = MoEConstants()


def update_constants_from_config(config):
    """
    Update global constants from a configuration object.

    Args:
        config: Configuration object with optional 'constants' section
            Can be either a dict or a DynamicConfig object
    """
    # Handle both dict and DynamicConfig objects
    constants_config = None
    if hasattr(config, 'constants') and config.constants is not None:
        constants_config = config.constants
    elif isinstance(config, dict) and 'constants' in config:
        constants_config = config['constants']

    if constants_config is None:
        return

    # Update data pipeline constants
    data_pipeline = None
    if hasattr(constants_config, 'data_pipeline') and constants_config.data_pipeline is not None:
        data_pipeline = constants_config.data_pipeline
    elif isinstance(constants_config, dict) and 'data_pipeline' in constants_config:
        data_pipeline = constants_config['data_pipeline']

    if data_pipeline:
        # Convert to dict if it's a DynamicConfig
        if hasattr(data_pipeline, 'to_dict'):
            data_pipeline = data_pipeline.to_dict()
        elif hasattr(data_pipeline, '__dict__'):
            data_pipeline = {k: v for k, v in data_pipeline.__dict__.items() if not k.startswith('_')}

        for key, value in data_pipeline.items():
            const_key = key.upper()
            if hasattr(DATA_CONSTANTS, const_key):
                setattr(DATA_CONSTANTS, const_key, value)
                print(f"✓ Updated DATA_CONSTANTS.{const_key} = {value}")

    # Update trainer constants
    trainer = None
    if hasattr(constants_config, 'trainer') and constants_config.trainer is not None:  # type: ignore[attr-defined]
        trainer = constants_config.trainer  # type: ignore[attr-defined]
    elif isinstance(constants_config, dict) and 'trainer' in constants_config:
        trainer = constants_config['trainer']

    if trainer:
        # Convert to dict if it's a DynamicConfig
        if hasattr(trainer, 'to_dict'):
            trainer = trainer.to_dict()
        elif hasattr(trainer, '__dict__'):
            trainer = {k: v for k, v in trainer.__dict__.items() if not k.startswith('_')}

        for key, value in trainer.items():
            const_key = key.upper()
            if hasattr(TRAINER_CONSTANTS, const_key):
                setattr(TRAINER_CONSTANTS, const_key, value)
                print(f"✓ Updated TRAINER_CONSTANTS.{const_key} = {value}")

    # Update MoE constants
    moe = None
    if hasattr(constants_config, 'moe') and constants_config.moe is not None:  # type: ignore[attr-defined]
        moe = constants_config.moe  # type: ignore[attr-defined]
    elif isinstance(constants_config, dict) and 'moe' in constants_config:
        moe = constants_config['moe']

    if moe:
        # Convert to dict if it's a DynamicConfig
        if hasattr(moe, 'to_dict'):
            moe = moe.to_dict()
        elif hasattr(moe, '__dict__'):
            moe = {k: v for k, v in moe.__dict__.items() if not k.startswith('_')}

        for key, value in moe.items():
            const_key = key.upper()
            if hasattr(MOE_CONSTANTS, const_key):
                setattr(MOE_CONSTANTS, const_key, value)
                print(f"✓ Updated MOE_CONSTANTS.{const_key} = {value}")
