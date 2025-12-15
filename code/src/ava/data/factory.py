"""
Data loader factory functions.

This module provides factory functions for creating optimized dataloaders:
- create_streaming_dataloaders: Create train/val dataloaders with all optimizations

Usage:
    from ava.data.factory import create_streaming_dataloaders

    train_loader, val_loader = create_streaming_dataloaders(
        tokenizer=tokenizer,
        batch_size=32,
        max_length=2048,
        data_dir='/path/to/data',
    )
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Import centralized constants
from ..config.constants import DATA_CONSTANTS
from ..core.logging import (
    Icons,
    print_header, print_subheader, print_info, print_warning, print_error,
    print_config, print_step,
)
from .distributed import DistributedStreamingDataset
from .streaming import (
    InfiniteStreamingDataset,
    StreamingDataset,
    _worker_init_fn,
)

# Distributed training imports
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


def create_streaming_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_dir: str,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    buffer_size: int = 15000,
    distributed: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    dynamic_length_fn: Optional[Callable[[], int]] = None,
    enable_bucketing: bool = True,
    bucket_boundaries: Optional[List[int]] = None,
    max_bucket_size: int = 200,
    val_max_samples: Optional[int] = None,
    val_split_ratio: float = 0.1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    use_weighted_mixing: bool = True,
    mixing_temperature: float = 1.0,
    data_mixer: Optional[Any] = None,
    samples_per_file: int = 500,
    use_streaming_tokenization: bool = False,
    streaming_buffer_size: int = 1000,
    use_dynamic_batching: bool = False,
    max_tokens_per_batch: Optional[int] = None,
    dataset_name: Optional[str] = None,
    dev_log_config: Optional[Any] = None,
    dynamic_batching_config: Optional[Dict[str, Any]] = None,
    shuffle_seed: Optional[int] = None,
    enable_length_sorting: bool = True,
) -> Tuple[Any, Any]:
    """
    Create optimized streaming train and validation dataloaders.

    Enhanced with:
    - Weighted data mixing (DoReMi-style quality-based sampling)
    - Efficient prefetching and worker management
    - Distributed training support
    - Progressive training compatibility

    Args:
        tokenizer: Tokenizer for text encoding
        batch_size: Batch size per device
        max_length: Maximum sequence length
        data_dir: Directory containing data files
        num_workers: Number of data loading workers
        max_samples: Maximum training samples (None = unlimited)
        buffer_size: Shuffle buffer size
        distributed: Enable distributed mode
        world_size: Number of distributed processes
        rank: Process rank in distributed training
        dynamic_length_fn: Function returning current sequence length
        enable_bucketing: Enable length-based bucketing
        bucket_boundaries: Custom bucket boundaries
        max_bucket_size: Maximum samples per bucket
        val_max_samples: Maximum validation samples
        val_split_ratio: Validation split ratio
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        use_weighted_mixing: Enable quality-based weighted sampling
        mixing_temperature: Sampling temperature
        data_mixer: Custom WeightedDataMixer instance
        samples_per_file: Samples per file before rotation
        use_streaming_tokenization: Enable streaming tokenization mode
        streaming_buffer_size: Buffer size for streaming mode
        use_dynamic_batching: Enable dynamic token-based batching
        max_tokens_per_batch: Max tokens per batch for dynamic batching
        dataset_name: Optional dataset name to filter to a single file
        dev_log_config: Config for development logging
        dynamic_batching_config: Dynamic batching configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Safety checks
    if batch_size is None:
        batch_size = 8
        print_warning(f"batch_size was None, defaulting to {batch_size}")

    # Auto-detect CPU cores
    if num_workers == -1:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        print_info(f"Auto-detected {num_workers} CPU cores")
    elif num_workers > 0:
        print_info(f"Using {num_workers} CPU workers for data loading")
    else:
        print_info(f"Using 0 workers (main process only) for data loading")

    # RAM-SAFE: Cap workers based on available system memory
    if num_workers > 0:
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            reserved_ram_gb = 25
            ram_per_worker_gb = 3.0
            max_workers_by_ram = max(1, int((available_ram_gb - reserved_ram_gb) / ram_per_worker_gb))

            if num_workers > max_workers_by_ram:
                print_warning(f"[RAM-SAFE] Reducing workers {num_workers} → {max_workers_by_ram}")
                num_workers = max_workers_by_ram
        except ImportError:
            if num_workers > 4:
                print_warning(f"[RAM-SAFE] Capping workers at 4")
                num_workers = 4

    # Dynamic prefetch factor based on sequence length
    if prefetch_factor == DATA_CONSTANTS.PREFETCH_FACTOR_MIN:
        original_prefetch = prefetch_factor
        calculated_prefetch = max(
            DATA_CONSTANTS.PREFETCH_FACTOR_MIN,
            min(DATA_CONSTANTS.PREFETCH_FACTOR_MAX, int(DATA_CONSTANTS.PREFETCH_FACTOR_NUMERATOR / max_length))
        )

        max_prefetch_memory_gb = 4.0
        bytes_per_sample = max_length * 2
        samples_in_prefetch = num_workers * calculated_prefetch * batch_size
        estimated_memory_gb = samples_in_prefetch * bytes_per_sample / (1024**3)

        if estimated_memory_gb > max_prefetch_memory_gb and calculated_prefetch > DATA_CONSTANTS.PREFETCH_FACTOR_MIN:
            safe_prefetch = max(
                DATA_CONSTANTS.PREFETCH_FACTOR_MIN,
                int(max_prefetch_memory_gb * (1024**3) / (num_workers * batch_size * bytes_per_sample))
            )
            calculated_prefetch = min(calculated_prefetch, safe_prefetch)

        prefetch_factor = calculated_prefetch

    # Auto-detect distributed training
    if distributed is None:
        distributed = DISTRIBUTED_AVAILABLE and (
            'WORLD_SIZE' in os.environ or
            (world_size is not None and world_size > 1)
        )

    if distributed and DISTRIBUTED_AVAILABLE:
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        print_info(f"Distributed training: rank {rank}/{world_size}")
    else:
        print_step("Creating streaming dataloaders...")

    # Create datasets
    dataset_kwargs = {
        'data_dir': data_dir,
        'tokenizer': tokenizer,
        'max_length': max_length,
        'buffer_size': buffer_size,
        'dynamic_length_fn': dynamic_length_fn,
        'enable_bucketing': enable_bucketing,
        'bucket_boundaries': bucket_boundaries,
        'max_bucket_size': max_bucket_size,
        'use_weighted_mixing': use_weighted_mixing,
        'mixing_temperature': mixing_temperature,
        'data_mixer': data_mixer,
        'samples_per_file': samples_per_file,
        'use_streaming_tokenization': use_streaming_tokenization,
        'streaming_buffer_size': streaming_buffer_size,
        'use_dynamic_batching': use_dynamic_batching,
        'max_tokens_per_batch': max_tokens_per_batch,
        'dataset_name': dataset_name,
        'dev_log_config': dev_log_config,
        'validation_rate': 0.0,
        'shuffle_seed': shuffle_seed,
    }

    # Training dataset
    if max_samples is None:
        train_dataset = InfiniteStreamingDataset(
            split='train',
            max_samples=None,
            **dataset_kwargs
        )
    else:
        train_dataset = StreamingDataset(
            split='train',
            max_samples=max_samples,
            **dataset_kwargs
        )

    # Validation dataset
    if val_max_samples is not None:
        computed_val_samples = val_max_samples
    elif max_samples is not None:
        computed_val_samples = int(max_samples * val_split_ratio)
    else:
        computed_val_samples = None

    val_buffer_size = buffer_size // 10 if buffer_size >= 10 else buffer_size

    val_dataset = StreamingDataset(
        split='val',
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=computed_val_samples,
        buffer_size=val_buffer_size,
        dynamic_length_fn=dynamic_length_fn,
        enable_bucketing=enable_bucketing,
        bucket_boundaries=bucket_boundaries,
        max_bucket_size=max_bucket_size,
        use_weighted_mixing=False,
        mixing_temperature=1.0,
        data_mixer=None,
        samples_per_file=samples_per_file,
        shuffle_seed=shuffle_seed,
    )

    # Use spawn method for multiprocessing with Arrow files
    if num_workers > 0:
        print_info(f"Using {num_workers} workers with 'spawn' multiprocessing context")

    # Dataloader configuration
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'multiprocessing_context': 'spawn' if num_workers > 0 else None,
        'timeout': 120 if num_workers > 0 else 0,
        'worker_init_fn': _worker_init_fn if num_workers > 0 else None,
        'collate_fn': None
    }

    if num_workers > 0:
        print_subheader("Data Pipeline Optimizations")
        print_config("Parallel workers", str(num_workers))
        print_config("Sample buffer", f"{buffer_size:,}")
        print_config("Prefetch factor", f"{prefetch_factor} batches/worker")
        print_config("Persistent workers", str(persistent_workers))

    # Apply distributed wrapping if needed
    if distributed and DISTRIBUTED_AVAILABLE:
        train_dataset = DistributedStreamingDataset(train_dataset, world_size or 1, rank or 0, enable_length_sorting=enable_length_sorting)
        val_dataset = DistributedStreamingDataset(val_dataset, world_size or 1, rank or 0, enable_length_sorting=enable_length_sorting)

    # Get base datasets for collate_fn
    base_train_dataset = train_dataset.base_dataset if isinstance(train_dataset, (InfiniteStreamingDataset, DistributedStreamingDataset)) else train_dataset
    base_val_dataset = val_dataset.base_dataset if isinstance(val_dataset, (InfiniteStreamingDataset, DistributedStreamingDataset)) else val_dataset

    train_collate_fn = getattr(base_train_dataset, 'collate_fn', None)
    val_collate_fn = getattr(base_val_dataset, 'collate_fn', None)

    # Check if dynamic batching is enabled
    use_dynamic_batch_iterator = (
        dynamic_batching_config is not None
        and dynamic_batching_config.get('enabled', False)
    )

    if use_dynamic_batch_iterator:
        from .dynamic_batch_iterator import DynamicBatchIterator
        from ..optimizations.dynamic_batching import create_dynamic_batch_scheduler

        db_config = dynamic_batching_config
        min_batch_size = db_config.get('min_batch_size', 64)
        max_batch_size = db_config.get('max_batch_size', 256)

        print_header("DYNAMIC BATCHING ENABLED", icon=Icons.LIGHTNING)
        print_config("Min batch size", str(min_batch_size))
        print_config("Max batch size", str(max_batch_size))

        dataloader_kwargs['batch_size'] = min_batch_size

    # Create DataLoaders
    try:
        train_loader_base = DataLoader(
            train_dataset,
            collate_fn=train_collate_fn,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'collate_fn'}
        )
        val_loader_base = DataLoader(
            val_dataset,
            collate_fn=val_collate_fn,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'collate_fn'}
        )
    except (BrokenPipeError, OSError) as e:
        print_error(f"DataLoader creation failed: {e}")
        print_warning("Retrying with num_workers=0...")

        dataloader_kwargs['num_workers'] = 0
        dataloader_kwargs['prefetch_factor'] = None
        dataloader_kwargs['persistent_workers'] = False
        dataloader_kwargs['multiprocessing_context'] = None
        dataloader_kwargs['worker_init_fn'] = None

        train_loader_base = DataLoader(
            train_dataset,
            collate_fn=train_collate_fn,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'collate_fn'}
        )
        val_loader_base = DataLoader(
            val_dataset,
            collate_fn=val_collate_fn,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'collate_fn'}
        )

    # Wrap with DynamicBatchIterator if enabled
    if use_dynamic_batch_iterator:
        scheduler_config = {
            'dynamic_batching': db_config,
            'training': {'batch_size': min_batch_size}
        }

        train_scheduler = create_dynamic_batch_scheduler(scheduler_config)
        val_scheduler = create_dynamic_batch_scheduler(scheduler_config)

        train_loader = DynamicBatchIterator(
            dataloader=train_loader_base,
            scheduler=train_scheduler,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
        val_loader = DynamicBatchIterator(
            dataloader=val_loader_base,
            scheduler=val_scheduler,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
    else:
        train_loader = train_loader_base
        val_loader = val_loader_base

    return train_loader, val_loader


__all__ = [
    'create_streaming_dataloaders',
]
