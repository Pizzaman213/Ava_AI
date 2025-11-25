"""
Advanced training optimizations for Ava

This package contains cutting-edge optimization techniques based on recent research:

Phase 2 High Priority:
- Dynamic batching with memory awareness (arXiv:2412.21124, arXiv:2503.05248)
- Sequence packing (already in codebase)

Phase 2 Medium Priority:
- Overlapped activation recomputation (arXiv:2406.08756)
- Double checkpointing (arXiv:2412.11810)
- KV-activation hybrid caching (arXiv:2501.01792)

All optimizations are independently configurable and can be enabled/disabled
without affecting other features.
"""

# Dynamic batching (Phase 2 High Priority)
from .dynamic_batching import (
    DynamicBatchScheduler,
    DynamicBatchConfig,
    create_dynamic_batch_scheduler,
)

# Overlapped recomputation (Phase 2 Medium Priority)
from .overlapped_recomputation import (
    overlapped_checkpoint,
    OverlappedCheckpointWrapper,
    overlapped_checkpoint_context,
    apply_overlapped_checkpointing,
    checkpoint_stats,
)

# Double checkpointing (Phase 2 Medium Priority)
from .double_checkpointing import (
    double_checkpoint,
    DoubleCheckpointConfig,
    DoubleCheckpointSequential,
    apply_double_checkpointing,
    calculate_memory_savings,
)

# Hybrid caching (Phase 2 Medium Priority)
from .hybrid_cache import (
    HybridCache,
    HybridCacheConfig,
    CachedAttention,
    apply_hybrid_caching,
)

__all__ = [
    # Dynamic batching
    'DynamicBatchScheduler',
    'DynamicBatchConfig',
    'create_dynamic_batch_scheduler',
    
    # Overlapped recomputation
    'overlapped_checkpoint',
    'OverlappedCheckpointWrapper',
    'overlapped_checkpoint_context',
    'apply_overlapped_checkpointing',
    'checkpoint_stats',
    
    # Double checkpointing
    'double_checkpoint',
    'DoubleCheckpointConfig',
    'DoubleCheckpointSequential',
    'apply_double_checkpointing',
    'calculate_memory_savings',
    
    # Hybrid caching
    'HybridCache',
    'HybridCacheConfig',
    'CachedAttention',
    'apply_hybrid_caching',
]

__version__ = '1.0.0'
