"""
Data Loading Manager for Ava Training Pipeline

Handles all data loading functionality including:
- Format detection (Arrow, Parquet, multi-column)
- Pretokenized Arrow/Parquet dataloaders (ultra-fast)
- Indexed Arrow dataloaders (map-style with true random shuffling)
- Validation dataset creation
- Data statistics logging

Worker Cleanup Strategy:
    DataLoader workers can block on distributed barriers (dist.barrier()) during
    training. If the distributed process group is destroyed before workers are
    terminated, this causes a deadlock.

    Cleanup order to prevent deadlocks:
    1. DataLoaderManager.cleanup() - Terminate worker processes FIRST
    2. (other component cleanup)
    3. cleanup_distributed() - Destroy process group LAST

    Two termination modes:
    - Normal (distributed_cleanup_started=False):
      Graceful shutdown via _shutdown_workers(), workers exit cleanly
    - Aggressive (distributed_cleanup_started=True):
      SIGTERM → wait 1s → SIGKILL, used when workers may be blocked on barriers

Format Detection:
    Uses cached detection to avoid repeated expensive file scans.
    Supports: Arrow (.arrow), JSONL (.jsonl), Parquet (.parquet), multi-column
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

# Module-level logger
_logger = logging.getLogger(__name__)
_logger.propagate = False  # Prevent duplicate logs

from ..data.multi_column import create_multi_column_dataloader, DatasetConfig
from ..data.pretokenized import create_ultra_fast_dataloaders
from ..core.data_utils import (
    get_num_workers,
    get_prefetch_factor,
    get_persistent_workers,
    get_samples_per_file,
    get_val_split_ratio,
)
from ..core.paths import get_code_dir

from .context import TrainingComponent, TrainingContext


# Format detection cache to avoid repeated expensive file scans
_format_detection_cache: Dict[str, Dict[str, Any]] = {}
_format_cache_lock = threading.Lock()  # Thread-safe cache access


class DataLoaderManager(TrainingComponent):
    """Manages all data loading operations for training."""

    def __init__(self, context: TrainingContext):
        """Initialize data loader manager.

        Args:
            context: Training context with configuration
        """
        super().__init__(context)
        self.train_loader = None
        self.val_loader = None

    def initialize(self) -> None:
        """Initialize component. Called once at startup."""
        self._initialized = True

    def cleanup(self, distributed_cleanup_started: bool = False) -> None:
        """Cleanup resources - properly terminate DataLoader workers.

        This fixes the semaphore leak issue by properly shutting down
        DataLoader worker processes before distributed cleanup.

        If distributed cleanup has already started, workers may be blocked
        on dist.barrier() calls, so we use aggressive termination.

        Args:
            distributed_cleanup_started: If True, use aggressive worker termination
                to avoid deadlocks when distributed group is already destroyed
        """
        import signal
        import os

        def _force_terminate_worker(w, timeout: float = 2.0):
            """Terminate a worker with SIGTERM, then SIGKILL if needed."""
            if not w.is_alive():
                return
            try:
                w.terminate()  # Send SIGTERM
                w.join(timeout=timeout)
                if w.is_alive():
                    # Still alive? Send SIGKILL
                    try:
                        os.kill(w.pid, signal.SIGKILL)
                        w.join(timeout=1.0)  # Brief wait after SIGKILL
                        _logger.debug(f"Force killed worker {w.pid}")
                    except ProcessLookupError:
                        pass  # Already dead
            except Exception as e:
                _logger.debug(f"Error terminating worker: {e}")

        for loader in [self.train_loader, self.val_loader]:
            if loader is not None:
                try:
                    # Shutdown worker processes to release semaphores
                    # The _iterator holds references to worker processes
                    if hasattr(loader, '_iterator') and loader._iterator is not None:
                        iterator = loader._iterator

                        if distributed_cleanup_started:
                            # Aggressive cleanup: workers may be blocked on dist.barrier()
                            # Force terminate them to prevent deadlocks
                            if hasattr(iterator, '_workers') and iterator._workers:
                                _logger.debug(f"Aggressively terminating {len(iterator._workers)} dataloader workers")
                                for w in iterator._workers:
                                    _force_terminate_worker(w, timeout=1.0)
                        else:
                            # Normal graceful shutdown
                            # Issue #4 fix: Check method existence before calling (PyTorch version compat)
                            if hasattr(iterator, '_shutdown_workers'):
                                try:
                                    iterator._shutdown_workers()
                                except Exception as e:
                                    _logger.warning(f"Error during graceful worker shutdown: {e}")
                                    # Fallback to aggressive termination if graceful fails
                                    if hasattr(iterator, '_workers') and iterator._workers:
                                        for w in iterator._workers:
                                            _force_terminate_worker(w, timeout=2.0)
                            else:
                                # Fallback: delete iterator to trigger __del__ cleanup
                                try:
                                    del loader._iterator
                                except Exception:
                                    pass
                except Exception as e:
                    _logger.debug(f"DataLoader cleanup: {e}")
        # Clear references
        self.train_loader = None
        self.val_loader = None

    def create_dataloaders(
        self,
        training_config: Any,
        tokenizer: Any,
        config_dict: dict,
        batch_size: Optional[int] = None,
    ) -> Tuple:
        """Create training and validation dataloaders.

        Supports multiple data formats:
        - Multi-column datasets
        - Streaming JSONL with on-the-fly tokenization
        - Pretokenized Arrow format (ultra-fast)

        Args:
            training_config: Enhanced training configuration
            tokenizer: Tokenizer for data processing
            config_dict: Raw configuration dictionary
            batch_size: Optional override for batch size

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Determine batch size
        if batch_size is None:
            # Try to get from training_config.training.batch_size
            if hasattr(training_config, 'training') and hasattr(training_config.training, 'batch_size'):
                batch_size = training_config.training.batch_size
            # Fallback to config_dict - check both training.batching.batch_size and training.batch_size
            if batch_size is None:
                training_dict = config_dict.get("training", {})
                batch_size = (
                    training_dict.get("batching", {}).get("batch_size") or
                    training_dict.get("batch_size", 8)
                )

        batch_size = int(batch_size) if batch_size is not None else 8
        if batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size: {batch_size}. Batch size must be positive. "
                f"Check your configuration under 'training.batch_size'"
            )

        # Route to appropriate loader type
        # Check if multi_column_data exists and is enabled
        use_multi_column = False
        if hasattr(training_config, 'multi_column_data'):
            use_multi_column = getattr(training_config.multi_column_data, 'use_multi_column', False)

        if use_multi_column:
            train_loader, val_loader = self._create_multi_column_loaders(
                training_config, tokenizer, batch_size
            )
        elif hasattr(training_config, 'data') and getattr(training_config.data, 'streaming', True):
            train_loader, val_loader = self._create_streaming_loaders(
                training_config, tokenizer, batch_size, config_dict
            )
        else:
            # Fallback: use streaming loader with defaults when no specific config found
            # This handles cases where streaming is explicitly set to False or missing
            _logger.info("No streaming config found, using streaming loader with defaults")
            train_loader, val_loader = self._create_streaming_loaders(
                training_config, tokenizer, batch_size, config_dict
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader

    def _create_multi_column_loaders(
        self, training_config: Any, tokenizer: Any, batch_size: int
    ) -> Tuple:
        """Create multi-column dataloaders.

        Args:
            training_config: Training configuration
            tokenizer: Tokenizer instance
            batch_size: Batch size

        Returns:
            Tuple of (train_loader, val_loader)
        """
        _logger.info("Using multi-column data loader")

        # Load dataset config if it's a file path
        dataset_config = training_config.multi_column_data.dataset_config
        if isinstance(dataset_config, str) and dataset_config.strip():
            import yaml

            with open(dataset_config, "r") as f:
                dataset_config = yaml.safe_load(f)
        elif not dataset_config or dataset_config == "":
            # Default config if none provided
            dataset_config = {
                "columns": [{"name": "text", "type": "text"}],
                "combine_strategy": "concatenate",
            }

        train_loader = create_multi_column_dataloader(
            config=dataset_config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            split="train",
        )

        val_loader = create_multi_column_dataloader(
            config=dataset_config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            split="validation",
        )

        return train_loader, val_loader

    def _create_streaming_loaders(
        self,
        training_config: Any,
        tokenizer: Any,
        batch_size: int,
        config_dict: dict,
    ) -> Tuple:
        """Create streaming dataloaders with format detection.

        Args:
            training_config: Training configuration
            tokenizer: Tokenizer instance
            batch_size: Batch size
            config_dict: Raw config dictionary

        Returns:
            Tuple of (train_loader, val_loader)
        """
        _logger.info("Using streaming data loader")

        # Find data directory with intelligent fallback
        data_dir = self._find_data_directory(training_config)

        # Log dataset information (SKIP if skip_sequence_count or fast_startup enabled)
        fast_startup = getattr(training_config.data, 'fast_startup', False)
        skip_stats = fast_startup or getattr(training_config.data, 'skip_sequence_count', False)
        if not skip_stats:
            self._log_dataset_stats(data_dir, batch_size, training_config)
        else:
            _logger.info(f" Data directory: {data_dir} (stats skipped for fast startup)")

        # Get configuration parameters
        num_workers = self._get_num_workers(training_config)
        prefetch_factor = self._get_prefetch_factor(training_config)
        persistent_workers = self._get_persistent_workers(training_config)
        samples_per_file = self._get_samples_per_file(training_config)
        # Get validation config
        val_split_ratio = self._get_val_split_ratio(training_config)

        # Always use pretokenized data (non-pretokenized streaming removed)

        # Get sequence packing config
        use_sequence_packing = getattr(training_config.data, "use_sequence_packing", False)
        packing_strategy = getattr(training_config.data, "packing_strategy", "greedy")

        # Check for indexed loader (map-style with true random shuffling)
        use_indexed_loader = getattr(training_config.data, 'use_indexed_loader', False)

        # Get randomization control from config (used by all loader types)
        shuffle_seed = getattr(training_config.data, 'shuffle_seed', None)
        enable_length_sorting = getattr(training_config.data, 'enable_length_sorting', True)

        if use_indexed_loader:
            _logger.info("=" * 60)
            _logger.info("INDEXED ARROW DATASET (Map-Style)")
            _logger.info("=" * 60)
            _logger.info("  True random shuffling at epoch start")
            _logger.info("  Length-binned sampling for reduced padding")
            _logger.info("  Sample-level train/val split")
            _logger.info("  Parallel indexing for fast startup")
            _logger.info("=" * 60)

            from ..data.indexed import create_indexed_dataloaders

            # Get indexed loader config
            indexed_num_bins = getattr(training_config.data, 'indexed_num_bins', 8)
            indexed_cache_size = getattr(training_config.data, 'indexed_cache_size', 50)
            indexed_index_workers = getattr(training_config.data, 'indexed_index_workers', None)
            max_files = getattr(training_config.data, 'max_files_to_load', None)

            # Handle tokenizer for pad_token_id
            # Issue #5 fix: Defensive tokenizer validation
            if tokenizer is not None:
                if not hasattr(tokenizer, 'pad_token_id'):
                    _logger.warning("Tokenizer missing pad_token_id attribute, using default 0")
                    pad_token_id = 0
                else:
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            else:
                pad_token_id = 0

            train_loader, val_loader = create_indexed_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                max_length=training_config.data.max_length,
                val_split_ratio=val_split_ratio,
                num_workers=num_workers,
                cache_size=indexed_cache_size,
                pad_token_id=pad_token_id,
                num_bins=indexed_num_bins,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                seed=shuffle_seed,
                max_files=max_files,
                index_workers=indexed_index_workers,
            )

            return train_loader, val_loader

        # Log GPU I/O optimizations
        self._log_io_optimizations(
            True,  # Always pretokenized
            num_workers,
            persistent_workers,
            prefetch_factor,
            use_sequence_packing,
            packing_strategy,
        )

        # Create pretokenized loaders (always use pretokenized Arrow data)
        _logger.info(" Using pretokenized Arrow data loader")

        # Get cache size from config or use optimized default
        cache_size = getattr(training_config.data, "cache_size", 200)

        # Get max_files_to_load from config to limit memory usage
        max_files_to_load = getattr(training_config.data, 'max_files_to_load', None)

        # Get lazy_file_discovery from config for memory-efficient large datasets
        lazy_file_discovery = getattr(training_config.data, 'lazy_file_discovery', False)

        # Get warm_start_files for fast startup (load N files initially, more progressively)
        warm_start_files = getattr(training_config.data, 'warm_start_files', 3)

        # Get packing-specific sorting control
        disable_packing_length_sort = getattr(training_config.data, 'disable_packing_length_sort', False)

        # Handle tokenizer being None (pretokenized data doesn't need tokenizer)
        # First try to get token IDs from model config (most reliable source)
        model_config = getattr(training_config, 'model', None)
        pad_token_id = getattr(model_config, 'pad_token_id', None) if model_config else None
        bos_token_id = getattr(model_config, 'bos_token_id', None) if model_config else None
        eos_token_id = getattr(model_config, 'eos_token_id', None) if model_config else None

        # Fall back to tokenizer if config doesn't have them
        if tokenizer is not None:
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
            if bos_token_id is None:
                bos_token_id = getattr(tokenizer, 'bos_token_id', 2)
            if eos_token_id is None:
                eos_token_id = getattr(tokenizer, 'eos_token_id', 1)

        # Final fallback to defaults matching the tokenizer vocab
        # Default special tokens: <|pad|>=0, <|eos|>=1, <|bos|>=2
        if pad_token_id is None:
            pad_token_id = 0
        if bos_token_id is None:
            bos_token_id = 2
        if eos_token_id is None:
            eos_token_id = 1

        # Get add_special_tokens from data config (default True for coherent generation)
        add_special_tokens = getattr(training_config.data, 'add_special_tokens', True)

        _logger.info(f"Special tokens: pad={pad_token_id}, bos={bos_token_id}, eos={eos_token_id}, add_special_tokens={add_special_tokens}")

        train_loader, val_loader = create_ultra_fast_dataloaders(
            batch_size=batch_size,
            max_length=training_config.data.max_length,
            data_dir=data_dir,
            num_workers=num_workers,
            buffer_size=training_config.data.buffer_size,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            samples_per_file=samples_per_file,
            cache_size=cache_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            add_special_tokens=add_special_tokens,
            max_samples=getattr(training_config.data, 'max_samples', None),
            val_split_ratio=val_split_ratio,
            use_sequence_packing=use_sequence_packing,
            packing_strategy=packing_strategy,
            max_files_to_load=max_files_to_load,
            lazy_file_discovery=lazy_file_discovery,
            warm_start_files=warm_start_files,
            shuffle_seed=shuffle_seed,
            enable_length_sorting=enable_length_sorting,
            disable_packing_length_sort=disable_packing_length_sort,
        )

        # Validate dataloaders (skip if configured or fast_startup - useful for pretokenized data)
        skip_validation = fast_startup or getattr(training_config.data, "skip_dataloader_validation", False)
        if not skip_validation:
            self._validate_dataloaders(train_loader, batch_size)
        else:
            _logger.info(" Skipping dataloader validation (fast_startup or skip_dataloader_validation=True)")

        return train_loader, val_loader

    def _find_data_directory(self, training_config: Any) -> str:
        """Find valid data directory with intelligent fallback.

        Args:
            training_config: Training configuration

        Returns:
            Path to valid data directory

        Raises:
            RuntimeError: If no valid data directory found
        """
        data_dir = None

        # First, try the configured data directory
        if hasattr(training_config.data, "data_dir") and training_config.data.data_dir:
            config_data_dir = Path(training_config.data.data_dir)
            if config_data_dir.exists():
                data_dir = str(config_data_dir)
                _logger.info(f"Using configured data_dir: {data_dir}")
                return data_dir
            else:
                _logger.warning(f"Configured data_dir does not exist: {config_data_dir}")

        # Get fallback paths from config
        # Use dynamic path resolution instead of hardcoded absolute paths
        code_dir = get_code_dir()
        default_fallback_paths = [
            str(code_dir / "data" / "processed"),
            str(code_dir / "data" / "combined"),
            str(code_dir / "data"),
            "./data/processed",
            "./data/combined",
            "./data",
            "../data/processed",
            "../data",
            "../../data",
        ]

        if hasattr(training_config, "data_loading"):
            fallback_paths = getattr(
                training_config.data_loading,
                "fallback_data_paths",
                default_fallback_paths,
            )
        else:
            fallback_paths = default_fallback_paths

        # Try each fallback path
        for fallback_path in fallback_paths:
            fallback_dir = Path(fallback_path)
            if fallback_dir.exists():
                format_info = self.enhanced_format_detection(
                    fallback_dir, training_config=training_config
                )

                if format_info["confidence"] > 0.0:
                    data_dir = str(fallback_dir)
                    _logger.info(f"Using fallback data_dir: {data_dir}")
                    _logger.info(
                        f"   Format detection: {format_info['detected_format']} "
                        f"(confidence: {format_info['confidence']:.2f})"
                    )
                    _logger.info(
                        f"   Files checked: {format_info['files_checked']}, "
                        f"Distribution: {format_info.get('format_distribution', {})}"
                    )
                    return data_dir
                else:
                    _logger.info(
                        f"   Checked {fallback_path}: exists but no valid data files found"
                    )
            else:
                _logger.info(f"   Checked {fallback_path}: does not exist")

        # No valid directory found
        raise RuntimeError(
            "No valid data directory found. Please ensure data is available in one of:\n"
            f"  - Configured path: {getattr(training_config.data, 'data_dir', 'Not set')}\n"
            f"  - {code_dir / 'data' / 'processed'}\n"
            f"  - {code_dir / 'data' / 'combined'}\n"
            f"  - {code_dir / 'data'}\n"
            "  - ./data/processed\n"
            "  - ./data/combined\n"
            "  - ./data\n"
            "Or set training_config.data.data_dir to a valid path"
        )

    def _log_dataset_stats(
        self, data_dir: str, batch_size: int, training_config: Any
    ) -> None:
        """Log dataset statistics and training configuration.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            training_config: Training configuration
        """
        _logger.info("=" * 70)
        _logger.info("DATASET INFORMATION")
        _logger.info("=" * 70)

        data_path = Path(data_dir)
        total_examples = 0
        file_count = 0

        _logger.info(f" Data directory: {data_dir}")

        # Count JSONL files (check both root and subdirectories)
        for jsonl_file in data_path.glob("**/*_processed.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    file_lines = sum(1 for _ in f)
                    total_examples += file_lines
                    file_count += 1
                    rel_path = jsonl_file.relative_to(data_path)
                    _logger.info(
                        f"    {rel_path}: {file_lines:,} examples"
                    )
            except Exception as e:
                _logger.warning(f"     Could not read {jsonl_file.name}: {e}")

        # Count Arrow files (check both root and subdirectories, supports IPC File and Stream formats)
        for arrow_file in data_path.glob("**/*.arrow"):
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc

                # Try IPC File format first, then fall back to IPC Stream format
                try:
                    with pa.memory_map(str(arrow_file), "r") as source:
                        table = ipc.open_file(source).read_all()
                except pa.ArrowInvalid:
                    # IPC Stream format (HuggingFace datasets)
                    with open(str(arrow_file), 'rb') as f:
                        table = ipc.open_stream(f).read_all()

                file_rows = len(table)
                total_examples += file_rows
                file_count += 1
                rel_path = arrow_file.relative_to(data_path)
                _logger.info(
                    f"    {rel_path}: {file_rows:,} examples (pre-tokenized)"
                )
            except Exception as e:
                _logger.warning(f"     Could not read {arrow_file.name}: {e}")

        # Count Parquet files (check both root and subdirectories like train/)
        for parquet_pattern in ["*.parquet", "**/*.parquet"]:
            for parquet_file in data_path.glob(parquet_pattern):
                try:
                    import pyarrow.parquet as pq

                    pq_file = pq.ParquetFile(parquet_file)
                    file_rows = pq_file.metadata.num_rows
                    total_examples += file_rows
                    file_count += 1
                    rel_path = parquet_file.relative_to(data_path)
                    _logger.info(
                        f"    {rel_path}: {file_rows:,} examples (parquet)"
                    )
                except Exception as e:
                    _logger.warning(f"     Could not read {parquet_file.name}: {e}")

        _logger.info(f"\n Total examples found: {total_examples:,}")
        _logger.info(f" Total files: {file_count}")

        # Validate minimum dataset size
        min_samples_required = batch_size * 2
        if total_examples < min_samples_required:
            error_msg = (
                f" CRITICAL ERROR: Dataset too small for training!\n"
                f"   Found: {total_examples} examples\n"
                f"   Required minimum: {min_samples_required} examples (batch_size * 2)\n"
                f"   Batch size: {batch_size}\n"
                f"   Files checked: {file_count}\n"
                f"   Directory: {data_dir}\n"
                f"   \n"
                f"   Solutions:\n"
                f"     1. Add more training data to {data_dir}\n"
                f"     2. Reduce batch_size (current: {batch_size})\n"
                f"     3. Check data files are in correct format (*_processed.jsonl, *.arrow, or *.parquet)"
            )
            _logger.error(error_msg)
            raise RuntimeError(error_msg)

        if file_count == 0:
            error_msg = (
                f" CRITICAL ERROR: No data files found!\n"
                f"   Directory checked: {data_dir}\n"
                f"   Expected patterns: *_processed.jsonl, *.arrow, or *.parquet\n"
                f"   \n"
                f"   Please ensure your data files follow the naming convention:\n"
                f"     - <dataset_name>_processed.jsonl (for raw data)\n"
                f"     - <dataset_name>_processed.arrow (for pre-tokenized data)\n"
                f"     - *.parquet (for parquet data, can be in train/ subdirectory)"
            )
            _logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Log training configuration
        # Check batching subsection first, then flat training config
        if hasattr(training_config.training, "batching"):
            gradient_acc_steps = getattr(
                training_config.training.batching,
                "gradient_accumulation_steps",
                None
            )
            # If None or not found, try fallbacks
            if gradient_acc_steps is None:
                gradient_acc_steps = getattr(training_config.training, "gradient_accumulation_steps", None)
            if gradient_acc_steps is None:
                gradient_acc_steps = getattr(training_config.training, "gradient_accumulation", 4)
        else:
            gradient_acc_steps = getattr(
                training_config.training,
                "gradient_accumulation_steps",
                None
            )
            if gradient_acc_steps is None:
                gradient_acc_steps = getattr(training_config.training, "gradient_accumulation", 4)

        # Final safety check
        if gradient_acc_steps is None:
            gradient_acc_steps = 4

        effective_batch_size = batch_size * gradient_acc_steps

        val_split_ratio = self._get_val_split_ratio(training_config)

        max_samples = getattr(training_config.data, 'max_samples', None)
        if max_samples:
            train_samples = max_samples
        else:
            train_samples = total_examples

        if max_samples:
            val_samples = int(max_samples * val_split_ratio)
        else:
            val_samples = int(total_examples * val_split_ratio)

        expected_steps = train_samples // effective_batch_size if train_samples > 0 else 0

        samples_per_file = self._get_samples_per_file(training_config)

        _logger.info(f"\n Training Configuration:")
        _logger.info(f"   Batch size: {batch_size}")
        _logger.info(f"   Gradient accumulation steps: {gradient_acc_steps}")
        _logger.info(f"   Effective batch size: {effective_batch_size}")
        _logger.info(f"   Training samples: {train_samples:,}")
        _logger.info(
            f"   Validation samples: {val_samples:,} ({val_split_ratio:.1%} of training)"
        )
        _logger.info(f"   Expected training steps: {expected_steps:,}")
        _logger.info(f"   Workers: {self._get_num_workers(training_config)}")
        _logger.info(f"   Buffer size: {training_config.data.buffer_size:,}")
        _logger.info(
            f"   Samples per file rotation: {samples_per_file} "
            f"(1=max diversity, higher=less I/O)"
        )
        _logger.info("="*80 + "\n")

    def _log_io_optimizations(
        self,
        use_pretokenized: bool,
        num_workers: int,
        persistent_workers: bool,
        prefetch_factor: int,
        use_sequence_packing: bool,
        packing_strategy: str,
    ) -> None:
        """Log GPU I/O optimization status.

        Args:
            use_pretokenized: Whether using pretokenized data
            num_workers: Number of data loading workers
            persistent_workers: Whether workers persist
            prefetch_factor: Prefetch factor
            use_sequence_packing: Whether sequence packing is enabled
            packing_strategy: Packing strategy name
        """
        _logger.info("=" * 70)
        _logger.info("GPU I/O OPTIMIZATIONS ACTIVE")
        _logger.info("=" * 70)
        if use_pretokenized:
            _logger.info("Ultra-fast pretokenized Arrow loader")
        else:
            _logger.info("Streaming JSONL loader with on-the-fly tokenization")
        _logger.info(f"Multi-worker data loading: {num_workers} workers")
        _logger.info(f"Persistent workers: {persistent_workers}")
        _logger.info(f"Pin memory: {torch.cuda.is_available()}")
        _logger.info(f"Prefetch factor: {prefetch_factor}")
        _logger.info("Non-blocking GPU transfers: enabled")
        stream_status = (
            "enabled" if torch.cuda.is_available() else "not available (CPU mode)"
        )
        _logger.info(f"CUDA streams for async transfers: {stream_status}")
        if use_sequence_packing:
            _logger.info(
                f"Sequence packing: ENABLED ({packing_strategy} strategy)"
            )
        else:
            _logger.info(
                "Sequence packing: DISABLED"
            )
        _logger.info("=" * 70)

    def _validate_dataloaders(self, train_loader: Any, batch_size: int) -> None:
        """Validate that dataloaders work and have sufficient samples.

        Args:
            train_loader: Training dataloader
            batch_size: Batch size

        Raises:
            RuntimeError: If validation fails
        """
        min_samples_required = 5  # At least 5 samples for meaningful training
        try:
            train_iter = iter(train_loader)
            sample_count = 0
            for _ in range(min_samples_required):
                try:
                    next(train_iter)
                    sample_count += 1
                except StopIteration:
                    break

            if sample_count < min_samples_required:
                raise RuntimeError(
                    f"Insufficient training data: found {sample_count} samples, "
                    f"minimum {min_samples_required} required for stable training"
                )

            _logger.info(
                f" Training data validation passed: {sample_count}+ samples available"
            )

        except Exception as e:
            raise RuntimeError(f"Training dataloader validation failed: {e}")

    @staticmethod
    def enhanced_format_detection(
        data_dir: Path,
        max_samples: Optional[int] = None,
        training_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Detect data format with caching.

        Args:
            data_dir: Path to data directory
            max_samples: Number of files to sample for detection
            training_config: Training configuration

        Returns:
            Format detection results
        """
        # PERF: Double-checked locking - fast path without lock acquisition
        cache_key = str(data_dir.absolute())
        if cache_key in _format_detection_cache:
            cached_result = _format_detection_cache[cache_key]
            _logger.debug(f"Using cached format detection: {cached_result['detected_format']}")
            return cached_result

        # Slow path: acquire lock and check again
        with _format_cache_lock:
            if cache_key in _format_detection_cache:
                cached_result = _format_detection_cache[cache_key]
                _logger.debug(f"Using cached format detection: {cached_result['detected_format']}")
                return cached_result

        # Get max_samples from config with fallback
        if max_samples is None:
            if training_config and hasattr(training_config, "data_loading"):
                max_samples = getattr(
                    training_config.data_loading,
                    "format_detection_samples",
                    10,
                )
            else:
                max_samples = 10

        format_scores = {}
        total_files_checked = 0

        # Sample files from different locations
        sample_files = []
        for pattern in ["**/*.arrow", "**/*.parquet", "**/*.jsonl"]:
            files = list(data_dir.glob(pattern))
            if files:
                sampled = files[:max_samples] if len(files) >= max_samples else files
                sample_files.extend(sampled)

        if not sample_files:
            return {"detected_format": "unknown", "confidence": 0.0, "files_checked": 0}

        # Check each sampled file
        for file_path in sample_files[:max_samples]:
            total_files_checked += 1
            format_type = file_path.suffix.lower()

            try:
                if format_type == ".arrow":
                    import pyarrow as pa
                    import pyarrow.ipc as ipc

                    # Try IPC File format first, then fall back to IPC Stream format
                    try:
                        with pa.ipc.open_file(file_path) as reader:
                            if reader.num_record_batches > 0:
                                format_scores[format_type] = (
                                    format_scores.get(format_type, 0) + 1
                                )
                    except pa.ArrowInvalid:
                        # IPC Stream format (HuggingFace datasets)
                        with open(str(file_path), 'rb') as f:
                            table = ipc.open_stream(f).read_all()
                            if len(table) > 0:
                                format_scores[format_type] = (
                                    format_scores.get(format_type, 0) + 1
                                )
                elif format_type == ".parquet":
                    import pyarrow.parquet as pq

                    pq_file = pq.ParquetFile(file_path)
                    if pq_file.num_row_groups > 0:
                        format_scores[format_type] = format_scores.get(format_type, 0) + 1
                elif format_type == ".jsonl":
                    with open(file_path, "r") as f:
                        first_line = f.readline().strip()
                        if first_line and first_line.startswith("{"):
                            format_scores[format_type] = (
                                format_scores.get(format_type, 0) + 1
                            )
            except Exception as e:
                _logger.debug(f"Failed to detect format for {file_path}: {e}")
                continue

        # Calculate confidence
        if not format_scores:
            result = {
                "detected_format": "unknown",
                "confidence": 0.0,
                "files_checked": total_files_checked,
            }
            with _format_cache_lock:
                _format_detection_cache[cache_key] = result
            return result

        best_format = max(format_scores.keys(), key=lambda k: format_scores[k])
        confidence = format_scores[best_format] / total_files_checked

        result = {
            "detected_format": best_format,
            "confidence": confidence,
            "files_checked": total_files_checked,
            "format_distribution": format_scores,
        }

        with _format_cache_lock:
            _format_detection_cache[cache_key] = result
        return result

    # Config getter methods - delegating to shared utilities
    @staticmethod
    def _get_num_workers(training_config: Any) -> int:
        """Get number of data loading workers from config (using shared utility)."""
        return get_num_workers(training_config)

    @staticmethod
    def _get_prefetch_factor(training_config: Any) -> int:
        """Get prefetch factor from config (using shared utility)."""
        return get_prefetch_factor(training_config)

    @staticmethod
    def _get_persistent_workers(training_config: Any) -> bool:
        """Get persistent workers setting from config (using shared utility)."""
        return get_persistent_workers(training_config)

    @staticmethod
    def _get_samples_per_file(training_config: Any) -> int:
        """Get samples per file from config (using shared utility)."""
        return get_samples_per_file(training_config)

    @staticmethod
    def _get_val_split_ratio(training_config: Any) -> float:
        """Get validation split ratio from config (using shared utility)."""
        return get_val_split_ratio(training_config)
