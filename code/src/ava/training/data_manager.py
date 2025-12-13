"""
Data Loading Manager for Ava Training Pipeline

Handles all data loading functionality including:
- Format detection (Arrow, JSONL, multi-column)
- Streaming and pretokenized dataloaders
- Validation dataset creation
- Data statistics logging
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..data.factory import create_streaming_dataloaders

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
    get_enable_bucketing,
    get_val_split_ratio,
    extract_dynamic_batching_config,
)
from ..core.paths import get_code_dir

from .context import TrainingComponent, TrainingContext


# Format detection cache to avoid repeated expensive file scans
_format_detection_cache: Dict[str, Dict[str, Any]] = {}


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

    def cleanup(self) -> None:
        """Cleanup resources. Called on shutdown or error."""
        if self.train_loader is not None:
            if hasattr(self.train_loader, 'close'):
                self.train_loader.close()
        if self.val_loader is not None:
            if hasattr(self.val_loader, 'close'):
                self.val_loader.close()

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
            # Fallback to config_dict
            if batch_size is None:
                batch_size = config_dict.get("training", {}).get("batch_size", 8)

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

        # Log dataset information
        self._log_dataset_stats(data_dir, batch_size, training_config)

        # Get configuration parameters
        num_workers = self._get_num_workers(training_config)
        prefetch_factor = self._get_prefetch_factor(training_config)
        persistent_workers = self._get_persistent_workers(training_config)
        samples_per_file = self._get_samples_per_file(training_config)
        enable_bucketing = self._get_enable_bucketing(training_config)

        # Get validation config
        val_split_ratio = self._get_val_split_ratio(training_config)

        # Check if using pretokenized data
        # Default to True since most modern datasets are pre-tokenized
        use_pretokenized = getattr(training_config.data, "use_pretokenized", True)

        # Get sequence packing config
        use_sequence_packing = getattr(training_config.data, "use_sequence_packing", False)
        packing_strategy = getattr(training_config.data, "packing_strategy", "greedy")

        # Get dynamic batching config (token-based for less padding)
        use_dynamic_batching = getattr(training_config.data, "use_dynamic_batching", False)
        max_tokens_per_batch = getattr(training_config.data, "max_tokens_per_batch", None)

        # Get memory-aware dynamic batching config using shared utility
        # Checks: config.dynamic_batching, config.training.dynamic_batching, config.training.batching.dynamic_batching
        dynamic_batching_config = extract_dynamic_batching_config(training_config)

        if dynamic_batching_config:
            _logger.info(" Memory-aware dynamic batching enabled")
            token_budget_config = dynamic_batching_config.get('token_budget', {})
            predictive_config = dynamic_batching_config.get('predictive', {})
            if token_budget_config.get('enabled'):
                _logger.info(f"   Token budget: target={token_budget_config['target_tokens_per_batch']}, "
                            f"max={token_budget_config['max_tokens_per_batch']}")
            if predictive_config.get('enabled'):
                backward_margin = predictive_config.get('backward_safety_margin')
                if backward_margin is not None:
                    _logger.info(f"   Predictive: backward_margin={backward_margin:.2f}")
                else:
                    _logger.info("   Predictive: enabled (using defaults)")

        # Log GPU I/O optimizations
        self._log_io_optimizations(
            use_pretokenized,
            num_workers,
            persistent_workers,
            prefetch_factor,
            use_sequence_packing,
            packing_strategy,
        )

        # Create appropriate loaders
        if use_pretokenized:
            _logger.info(" Using pretokenized Arrow data loader (60x faster)")
            # Get cache size from config or use optimized default
            cache_size = getattr(training_config.data, "cache_size", 200)

            # Get max_files_to_load from config to limit memory usage
            max_files_to_load = getattr(training_config.data, 'max_files_to_load', None)

            # Get lazy_file_discovery from config for memory-efficient large datasets
            lazy_file_discovery = getattr(training_config.data, 'lazy_file_discovery', False)

            # Handle tokenizer being None (pretokenized data doesn't need tokenizer)
            if tokenizer is not None:
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None else 2
            else:
                _logger.info("Tokenizer not provided, using default pad_token_id=0, eos_token_id=2")
                pad_token_id = 0
                eos_token_id = 2

            # Get batch_controller from context if available
            # This connects the scheduler to the central batch size authority
            batch_controller = getattr(self.context, 'batch_controller', None)

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
                eos_token_id=eos_token_id,
                max_samples=getattr(training_config.data, 'max_samples', None),
                val_split_ratio=val_split_ratio,
                use_sequence_packing=use_sequence_packing,
                packing_strategy=packing_strategy,
                dynamic_batching_config=dynamic_batching_config,
                max_files_to_load=max_files_to_load,
                lazy_file_discovery=lazy_file_discovery,
                batch_controller=batch_controller,
            )
        else:
            _logger.info(
                " Using streaming JSONL data loader with on-the-fly tokenization"
            )
            train_loader, val_loader = create_streaming_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=training_config.data.max_length,
                data_dir=data_dir,
                num_workers=num_workers,
                buffer_size=training_config.data.buffer_size,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                samples_per_file=samples_per_file,
                max_samples=getattr(training_config.data, 'max_samples', None),
                val_split_ratio=val_split_ratio,
                enable_bucketing=enable_bucketing,
                use_dynamic_batching=use_dynamic_batching,
                max_tokens_per_batch=max_tokens_per_batch,
                use_streaming_tokenization=getattr(
                    training_config.data, "use_streaming_tokenization", False
                ),
                streaming_buffer_size=getattr(
                    training_config.data, "streaming_buffer_size", 1000
                ),
                dataset_name=getattr(training_config.data, "dataset_name", None),
                dev_log_config=getattr(training_config, "dev_log", None),
                dynamic_batching_config=dynamic_batching_config,
            )

        # Validate dataloaders (skip if configured - useful for pretokenized data with spawn workers)
        skip_validation = getattr(training_config.data, "skip_dataloader_validation", False)
        if not skip_validation:
            self._validate_dataloaders(train_loader, batch_size)
        else:
            _logger.info(" Skipping dataloader validation (skip_dataloader_validation=True)")

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

        # Count JSONL files
        for jsonl_file in data_path.glob("*_processed.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    file_lines = sum(1 for _ in f)
                    total_examples += file_lines
                    file_count += 1
                    _logger.info(
                        f"    {jsonl_file.name}: {file_lines:,} examples"
                    )
            except Exception as e:
                _logger.warning(f"     Could not read {jsonl_file.name}: {e}")

        # Count Arrow files
        for arrow_file in data_path.glob("*.arrow"):
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc

                with pa.memory_map(str(arrow_file), "r") as source:
                    table = ipc.open_file(source).read_all()
                    file_rows = len(table)
                    total_examples += file_rows
                    file_count += 1
                    _logger.info(
                        f"    {arrow_file.name}: {file_rows:,} examples (pre-tokenized)"
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
                getattr(training_config.training, "gradient_accumulation_steps",
                        getattr(training_config.training, "gradient_accumulation", 4)),
            )
        else:
            gradient_acc_steps = getattr(
                training_config.training,
                "gradient_accumulation_steps",
                getattr(training_config.training, "gradient_accumulation", 4),
            )
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
            _logger.info("Ultra-fast pretokenized Arrow loader (60x speedup)")
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
                f"Sequence packing: ENABLED ({packing_strategy} strategy, 20-35% speedup)"
            )
        else:
            _logger.info(
                "Sequence packing: DISABLED (enable for 20-35% speedup)"
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
        # Check cache first
        cache_key = str(data_dir.absolute())
        if cache_key in _format_detection_cache:
            cached_result = _format_detection_cache[cache_key]
            _logger.info(
                f"Using cached format detection: {cached_result['detected_format']}"
            )
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

                    with pa.ipc.open_file(file_path) as reader:
                        if reader.num_record_batches > 0:
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
    def _get_enable_bucketing(training_config: Any) -> bool:
        """Get bucketing enabled setting from config (using shared utility)."""
        return get_enable_bucketing(training_config)

    @staticmethod
    def _get_val_split_ratio(training_config: Any) -> float:
        """Get validation split ratio from config (using shared utility)."""
        return get_val_split_ratio(training_config)
