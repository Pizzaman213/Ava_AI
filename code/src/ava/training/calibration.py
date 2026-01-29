"""
Batch Size Calibration Module.

Provides automatic batch size calibration to maximize GPU utilization
while staying within memory limits.

Features:
    - Real DataLoader or synthetic batches
    - Binary search between min and max batch size
    - Multi-GPU synchronization (uses MIN across GPUs)
    - Configurable target memory percentage
    - Calibration result caching for faster resume
"""

import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ============================================================================
# Calibration Cache
# ============================================================================

class CalibrationCache:
    """
    Cache calibration results to disk for faster resume.

    The cache key is based on:
    - GPU name and memory
    - Model configuration (hidden_size, num_layers, etc.)
    - Sequence length
    - Target memory percentage

    Cache is invalidated if any of these change.
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ava" / "calibration"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the calibration cache.

        Args:
            cache_dir: Directory to store cache files (default: ~/.cache/ava/calibration)
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> str:
        """
        Compute a unique cache key based on configuration and hardware.

        Args:
            config: Full training config
            device: Target device

        Returns:
            Hex string cache key
        """
        # Collect cache-relevant values
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        calib_config = config.get('batch_size_calibration', {})

        cache_inputs = {
            # Hardware
            'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu',
            'gpu_memory_gb': round(
                torch.cuda.get_device_properties(device).total_memory / 1e9, 1
            ) if torch.cuda.is_available() else 0,
            # Model architecture
            'vocab_size': model_config.get('vocab_size', 0),
            'hidden_size': model_config.get('hidden_size', 0),
            'num_layers': model_config.get('num_layers', 0),
            'num_attention_heads': model_config.get('num_attention_heads', 0),
            'intermediate_size': model_config.get('intermediate_size', 0),
            'num_experts': model_config.get('num_experts', 0),
            'gradient_checkpointing': model_config.get('gradient_checkpointing', False),
            # Data
            'max_length': data_config.get('max_length', 512),
            # Calibration settings
            'target_memory': calib_config.get('target_memory', 0.70),
            'min_batch_size': calib_config.get('min_batch_size', 1),
            'max_batch_size': calib_config.get('max_batch_size', 128),
        }

        # Create deterministic hash
        cache_str = json.dumps(cache_inputs, sort_keys=True)
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()[:16]

        return cache_hash

    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file."""
        return self.cache_dir / f"calibration_{cache_key}.json"

    def load(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> Optional[int]:
        """
        Load cached calibration result if available.

        Args:
            config: Full training config
            device: Target device

        Returns:
            Cached batch size or None if not found
        """
        cache_key = self._compute_cache_key(config, device)
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            # Validate cache version
            if cached.get('version', 0) != 1:
                logger.debug(f"Cache version mismatch, invalidating: {cache_path}")
                cache_path.unlink()
                return None

            batch_size = cached.get('optimal_batch_size')
            if batch_size is not None and isinstance(batch_size, int) and batch_size > 0:
                logger.info(f"Loaded cached calibration: batch_size={batch_size} (key={cache_key})")
                return batch_size

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to load calibration cache: {e}")
            try:
                cache_path.unlink()
            except OSError:
                pass

        return None

    def save(
        self,
        config: Dict[str, Any],
        device: torch.device,
        optimal_batch_size: int,
    ) -> None:
        """
        Save calibration result to cache.

        Args:
            config: Full training config
            device: Target device
            optimal_batch_size: Calibrated batch size
        """
        cache_key = self._compute_cache_key(config, device)
        cache_path = self.get_cache_path(cache_key)

        cache_data = {
            'version': 1,
            'optimal_batch_size': optimal_batch_size,
            'cache_key': cache_key,
            'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu',
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved calibration cache: {cache_path}")
        except OSError as e:
            logger.debug(f"Failed to save calibration cache: {e}")

    def invalidate(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """
        Invalidate cached calibration for given config.

        Args:
            config: Full training config
            device: Target device
        """
        cache_key = self._compute_cache_key(config, device)
        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Invalidated calibration cache: {cache_path}")
            except OSError:
                pass

    def clear_all(self) -> int:
        """
        Clear all cached calibration results.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("calibration_*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass
        logger.info(f"Cleared {count} calibration cache files")
        return count


# Global cache instance
_calibration_cache: Optional[CalibrationCache] = None


def get_calibration_cache() -> CalibrationCache:
    """Get or create the global calibration cache."""
    global _calibration_cache
    if _calibration_cache is None:
        _calibration_cache = CalibrationCache()
    return _calibration_cache


class BatchSizeCalibrator:
    """
    Automatic batch size calibration for optimal GPU utilization.

    This class handles batch size calibration by:
    1. Creating a sample batch function (real or synthetic)
    2. Running binary search to find optimal batch size
    3. Synchronizing across GPUs (multi-GPU training)
    4. Managing memory cleanup after calibration

    Attributes:
        config: Full training config dict
        model: Neural network model
        model_builder: ModelBuilder for creating fresh models
        device: Target torch device
        rank: Process rank
        world_size: Total processes
        logger: Logger instance
        controller: BatchSizeController for memory-aware calibration
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        model_builder: Any,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the calibrator.

        Args:
            config: Full training config dict
            model: Built model for calibration
            model_builder: ModelBuilder for fresh model creation
            device: Target device
            rank: Process rank (0 for single GPU)
            world_size: Total processes (1 for single GPU)
            logger_instance: Logger (uses module logger if None)
        """
        self.config = config
        self.model = model
        self.model_builder = model_builder
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.log = logger_instance or logger

        # Controller will be created during calibration
        self.controller = None

        # Calibration config
        self.calib_config = config.get('batch_size_calibration', {})

    def calibrate(self) -> Optional[int]:
        """
        Run batch size calibration with caching support.

        The calibration result is cached to disk based on hardware and model
        configuration. On subsequent runs with the same config, the cached
        value is used to skip the calibration process.

        Cache can be disabled via batch_size_calibration.use_cache: false

        Returns:
            Optimal batch size, or None if calibration failed
        """
        from ava.optimizations.batch_controller import BatchSizeController
        from ava.training import OptimizerManager
        from ava.training.distributed import DistributedStateManager

        # Check cache first (unless disabled)
        use_cache = self.calib_config.get('use_cache', True)
        if use_cache and self.rank == 0:
            cache = get_calibration_cache()
            cached_batch_size = cache.load(self.config, self.device)
            if cached_batch_size is not None:
                self.log.info(f"Using cached calibration result: batch_size={cached_batch_size}")
                # Still need to create controller for OOM recovery support
                min_bs, max_bs, target_mem = self._get_calibration_bounds()
                calib_headroom = self.calib_config.get('calibration_headroom', 0.90)
                self.controller = BatchSizeController(
                    min_batch_size=min_bs,
                    max_batch_size=max_bs,
                    target_memory=target_mem,
                    calibration_headroom=calib_headroom,
                )
                return self._synchronize_batch_size(cached_batch_size)

        if self.rank == 0:
            self.log.info("Running batch size calibration with REAL OPTIMIZER...")

        # Get calibration settings
        min_bs, max_bs, target_mem = self._get_calibration_bounds()

        if self.rank == 0:
            self.log.info(
                f"  Calibration settings: min_bs={min_bs}, max_bs={max_bs}, "
                f"target_memory={target_mem}"
            )

        # Create controller
        calib_headroom = self.calib_config.get('calibration_headroom', 0.90)
        self.controller = BatchSizeController(
            min_batch_size=min_bs,
            max_batch_size=max_bs,
            target_memory=target_mem,
            calibration_headroom=calib_headroom,
        )

        # Create batch sampling function
        sample_batch_fn = self._create_sample_batch_fn()

        # Create optimizer factory
        optimizer_factory = self._create_optimizer_factory()

        # Create model factory if using fresh models
        model_factory = self._create_model_factory()

        # Run calibration
        optimal_batch_size = self.controller.startup_calibration(
            model=self.model,
            sample_batch_fn=sample_batch_fn,
            optimizer_factory=optimizer_factory,
            perform_optimizer_step=self.calib_config.get('perform_optimizer_step', True),
            max_time_seconds=self.calib_config.get('calibration_timeout_sec', 90.0),
            rank=self.rank,
            world_size=self.world_size,
            model_factory=model_factory,
        )

        if self.rank == 0:
            self.log.info(f"Calibration complete: optimal batch size = {optimal_batch_size}")

            # Save to cache for future runs
            if use_cache and optimal_batch_size is not None:
                cache = get_calibration_cache()
                cache.save(self.config, self.device, optimal_batch_size)

        # Synchronize across GPUs
        synced_batch_size = self._synchronize_batch_size(optimal_batch_size)

        return synced_batch_size

    def _get_calibration_bounds(self) -> Tuple[int, int, float]:
        """Get calibration min/max batch size and target memory."""
        calib_config = self.calib_config
        batching_cfg = self.config.get('training', {}).get('batching', {})

        # min_batch_size
        min_bs = calib_config.get('min_batch_size', 1)

        # max_batch_size
        max_batch_multiplier = calib_config.get('max_batch_multiplier', 4)
        default_max = batching_cfg.get('batch_size', 32) * max_batch_multiplier
        max_bs = calib_config.get('max_batch_size', default_max)

        # target_memory
        target_mem = calib_config.get('target_memory', 0.70)

        # Validate
        if min_bs >= max_bs:
            raise ValueError(
                f"Calibration min_batch_size ({min_bs}) must be < max_batch_size ({max_bs})"
            )
        if not 0.0 < target_mem <= 1.0:
            raise ValueError(
                f"Calibration target_memory must be in (0.0, 1.0], got {target_mem}"
            )

        return min_bs, max_bs, target_mem

    def _create_sample_batch_fn(self) -> Callable[[int], Dict[str, torch.Tensor]]:
        """
        Create batch sampling function.

        Returns real DataLoader function or synthetic batch function.
        """
        use_real_dataloader = self.calib_config.get('use_real_dataloader', False)

        if use_real_dataloader:
            fn = self._try_create_real_dataloader_fn()
            if fn is not None:
                return fn

        # Fallback: synthetic batches
        if self.rank == 0:
            self.log.info("Using synthetic batches for calibration")

        return self._create_synthetic_batch_fn()

    def _try_create_real_dataloader_fn(self) -> Optional[Callable]:
        """Try to create real DataLoader for calibration."""
        if self.rank == 0:
            self.log.info("Attempting to create REAL DataLoader for calibration...")

        try:
            # Load tokenizer
            tokenizer = self._load_tokenizer()
            if tokenizer is None:
                return None

            # Create calibration DataLoader
            loader = self._create_calibration_dataloader(tokenizer)
            if loader is None:
                return None

            if self.rank == 0:
                self.log.info("Using REAL DataLoader for calibration (includes I/O overhead)")

            # Create sampling function with iterator
            return self._create_real_batch_fn(loader)

        except Exception as e:
            if self.rank == 0:
                self.log.warning(f"Failed to create real DataLoader: {e}")
                self.log.warning("Falling back to synthetic batches")
            return None

    def _load_tokenizer(self):
        """Load tokenizer for calibration."""
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            from ava.core.paths import get_tokenizer_path

            data_config = self.config.get('data', {})
            tokenizer_path = (
                data_config.get('tokenizer_path') or
                data_config.get('tokenizer_name') or
                str(get_tokenizer_path())
            )

            tokenizer_path_obj = Path(tokenizer_path)

            if tokenizer_path_obj.exists() and tokenizer_path_obj.suffix == '.json':
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

            elif tokenizer_path_obj.is_dir() and (tokenizer_path_obj / 'tokenizer.json').exists():
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_path_obj / 'tokenizer.json')
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            if self.rank == 0:
                self.log.info(f"Loaded tokenizer for calibration: {tokenizer_path}")

            return tokenizer

        except Exception as e:
            if self.rank == 0:
                self.log.warning(f"Failed to load tokenizer: {e}")
            return None

    def _create_calibration_dataloader(self, tokenizer):
        """Create minimal DataLoader for calibration."""
        try:
            from ava.data.pretokenized import create_ultra_fast_dataloaders

            data_config = self.config.get('data', {})
            model_config = self.config.get('model', {})

            data_dir = data_config.get('data_dir')
            if not data_dir:
                if self.rank == 0:
                    self.log.warning("No data_dir specified, cannot create calibration DataLoader")
                return None

            max_length = data_config.get('max_length', 512)

            train_loader, _ = create_ultra_fast_dataloaders(
                batch_size=self.calib_config.get('min_batch_size', 8),
                max_length=max_length,
                data_dir=data_dir,
                num_workers=2,  # Minimal workers for calibration
                prefetch_factor=2,
                persistent_workers=False,
                lazy_file_discovery=data_config.get('lazy_file_discovery', True),
                cache_size=data_config.get('cache_size', 20),
                max_files_to_load=5,  # Only load a few files
                verbose=False,
                pad_token_id=model_config.get('pad_token_id', 0),
                eos_token_id=model_config.get('eos_token_id', 2),
            )

            return train_loader

        except Exception as e:
            if self.rank == 0:
                self.log.warning(f"Failed to create calibration DataLoader: {e}")
            return None

    def _create_real_batch_fn(self, loader) -> Callable[[int], Dict[str, torch.Tensor]]:
        """Create batch function using real DataLoader."""
        calib_iter = iter(loader)
        retry_count = [0]
        max_retries = 3

        def sample_batch_fn(bs: int) -> Dict[str, torch.Tensor]:
            nonlocal calib_iter

            try:
                batch = next(calib_iter)
                retry_count[0] = 0

                # Move to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Handle batch size mismatch
                actual_bs = batch['input_ids'].shape[0] if isinstance(batch, dict) else batch.shape[0]

                if actual_bs < bs:
                    # Collect more batches
                    batches = [batch]
                    batches_needed = (bs + actual_bs - 1) // actual_bs

                    for _ in range(batches_needed - 1):
                        try:
                            next_batch = next(calib_iter)
                        except StopIteration:
                            calib_iter = iter(loader)
                            next_batch = next(calib_iter)

                        if isinstance(next_batch, dict):
                            next_batch = {
                                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in next_batch.items()
                            }
                        batches.append(next_batch)

                    # Concatenate
                    if isinstance(batch, dict):
                        batch = {
                            k: torch.cat([b[k] for b in batches], dim=0)[:bs]
                            if isinstance(batches[0][k], torch.Tensor) else batches[0][k]
                            for k in batch.keys()
                        }
                    else:
                        batch = torch.cat(batches, dim=0)[:bs]

                    del batches

                elif actual_bs > bs:
                    # Truncate
                    if isinstance(batch, dict):
                        batch = {
                            k: v[:bs] if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                    else:
                        batch = batch[:bs]

                return batch

            except StopIteration:
                retry_count[0] += 1
                if retry_count[0] > max_retries:
                    raise RuntimeError(
                        f"Calibration DataLoader exhausted after {max_retries} retries"
                    )
                calib_iter = iter(loader)
                return sample_batch_fn(bs)

        return sample_batch_fn

    def _create_synthetic_batch_fn(self) -> Callable[[int], Dict[str, torch.Tensor]]:
        """Create synthetic batch function."""
        data_max_length = self.config.get('data', {}).get('max_length', 512)
        seq_len = self.calib_config.get('base_sequence_length', data_max_length)
        vocab_size = self.config.get('model', {}).get('vocab_size', 50680)
        device = self.device

        def sample_batch_fn(bs: int) -> Dict[str, torch.Tensor]:
            """Create worst-case synthetic batch."""
            # Full-length sequences (worst case)
            attention_mask = torch.ones(bs, seq_len, device=device)

            # Realistic token distribution
            uniform_rand = torch.rand(bs, seq_len, device=device)
            skewed_rand = torch.pow(uniform_rand, 2.0)
            input_ids = (skewed_rand * vocab_size).long().clamp(0, vocab_size - 1)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }

        return sample_batch_fn

    def _create_optimizer_factory(self) -> Callable:
        """Create optimizer factory for calibration."""
        from ava.training import OptimizerManager, TrainingContext

        # Create temporary context for optimizer manager
        temp_context = TrainingContext(
            model=self.model,
            device=self.device,
            config=self.config,
            run_manager=None,
            rank=self.rank,
            world_size=self.world_size,
            is_main_process=(self.rank == 0),
        )

        temp_opt_mgr = OptimizerManager(temp_context)
        temp_opt_mgr.initialize()

        training_config = self.config.get('training', {})
        optimizer_config = training_config.get('optimizer', {})

        learning_rate = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)

        use_fresh_models = self.calib_config.get('use_fresh_models', True)

        return temp_opt_mgr.create_optimizer_factory(
            model=self.model,
            config=self.config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            support_model_arg=use_fresh_models,
        )

    def _create_model_factory(self) -> Optional[Callable]:
        """Create model factory for fresh models per calibration test."""
        if not self.calib_config.get('use_fresh_models', True):
            return None

        if self.rank == 0:
            self.log.info("Using FRESH MODEL per calibration test (accurate memory measurement)")

        model_builder = self.model_builder
        config = self.config
        device = self.device

        def model_factory():
            """Create fresh model instance."""
            fresh_model = model_builder.build_model(config, device)

            # Apply quantization if enabled
            quant_config = config.get('quantization', {})
            if quant_config.get('enabled', False):
                fresh_model = model_builder.apply_quantization(fresh_model, quant_config)

            # Apply optimizations (but NOT torch.compile - too slow)
            optim_config = config.copy()
            if 'performance' in optim_config:
                optim_config['performance'] = optim_config['performance'].copy()
                optim_config['performance']['enable_torch_compile'] = False

            fresh_model = model_builder.apply_pre_device_optimizations(fresh_model, optim_config)
            fresh_model = model_builder.move_to_device(fresh_model, device)
            fresh_model = model_builder.apply_post_device_optimizations(fresh_model, optim_config)

            return fresh_model

        return model_factory

    def _synchronize_batch_size(self, optimal_batch_size: int) -> int:
        """Synchronize batch size across GPUs."""
        from ava.training.distributed import DistributedStateManager

        if self.world_size <= 1:
            # Single GPU - just validate
            if optimal_batch_size <= 0:
                raise RuntimeError(f"Invalid batch size: {optimal_batch_size}")
            return optimal_batch_size

        # Synchronize CUDA before memory measurement
        torch.cuda.synchronize()

        # Create distributed state manager
        dist_manager = DistributedStateManager(
            rank=self.rank,
            world_size=self.world_size,
            timeout_seconds=1800
        )

        # Gather VRAM info to detect mismatches
        total_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        vram_sizes = dist_manager.all_gather_values(total_memory_gb, "vram_gb")

        if self.rank == 0:
            vram_sizes_rounded = [round(v, 1) for v in vram_sizes]
            unique_vram = set(vram_sizes_rounded)
            if len(unique_vram) > 1:
                self.log.warning(
                    f"Mixed VRAM detected across {self.world_size} GPUs: {vram_sizes_rounded} GB\n"
                    f"   Using MINIMUM batch size to prevent OOM on smaller GPU(s)"
                )
            else:
                self.log.info(
                    f"Synchronizing batch size across {self.world_size} identical GPUs "
                    f"({vram_sizes_rounded[0]} GB each)..."
                )

        # Synchronize using MIN (prevents OOM on smallest GPU)
        synced_batch_size = dist_manager.synchronized_update(
            value=optimal_batch_size,
            reduction_op=dist.ReduceOp.MIN,
            validate_fn=lambda x: x > 0,
            value_name="optimal_batch_size"
        )

        return synced_batch_size


__all__ = ['BatchSizeCalibrator', 'CalibrationCache', 'get_calibration_cache']
