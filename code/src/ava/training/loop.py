"""
Training loop manager for the Ava pipeline.

Handles the core training loop with:
- Async batch prefetching
- Gradient accumulation
- Mixed precision training
- Generation testing
- Coherence measurement
- Proper error tracking
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .context import ManagerInterface, TrainingContext
from ..optimizations.prefetch import AsyncBatchPrefetcher
from ..optimizations.gradients import check_gradients


# ============================================================================
# Protocol Interfaces for Decoupled Dependencies
# ============================================================================

@runtime_checkable
class MetricsLoggerProtocol(Protocol):
    """Protocol for metrics logging during training."""

    def log_training_step(
        self, step: int, loss: float, lr: float, batch_size: int, epoch: int
    ) -> None:
        """Log metrics for a training step."""
        ...

    def log_gradients(self, step: int, grad_stats: Dict[str, Any]) -> None:
        """Log gradient statistics."""
        ...

    def log_generation(self, step: int, gen_data: Dict[str, Any]) -> None:
        """Log generated text samples."""
        ...

    def log_coherence(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log coherence metrics."""
        ...


@runtime_checkable
class GenerationProviderProtocol(Protocol):
    """Protocol for text generation during training."""

    def is_generation_pending(self) -> bool:
        """Check if a generation is currently running."""
        ...

    def generate_async(
        self,
        model: nn.Module,
        step: int,
        vocab_size: int,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        """Start async generation."""
        ...

    def process_completed_generations(self) -> list:
        """Get completed generation results."""
        ...

    def measure_coherence(
        self, model: nn.Module, step: int, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Measure model coherence."""
        ...


@runtime_checkable
class CheckpointSaverProtocol(Protocol):
    """Protocol for checkpoint saving during training."""

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> Any:
        """Save a checkpoint."""
        ...

# Optional Nsight profiler import
try:
    from ava.cuda.profiler import NsightProfiler
    NSIGHT_PROFILER_AVAILABLE = True
except ImportError:
    NSIGHT_PROFILER_AVAILABLE = False
    NsightProfiler = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingLoopConfig:
    """Configuration for the training loop."""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    # GPU SYNC FIX: Increased default from 10 to 100 to reduce .item() sync frequency
    # Each log step requires GPU->CPU sync for loss value. Logging every 10 steps
    # causes ~10% GPU idle time from synchronization overhead.
    log_interval: int = 100
    generate_every_n_steps: int = 500
    save_steps: int = 0
    max_consecutive_failures: int = 10
    max_steps: Optional[int] = None  # Stop after this many steps (None = no limit)
    # Profiling options (disabled by default - use --enable-profiling to enable)
    enable_profiling: bool = False
    profile_start_step: int = 0  # Profile from start
    profile_end_step: int = 999999  # Profile all steps
    profile_dir: str = "./profiles"  # Will be overridden to run folder when enabled
    # GPU SYNC FIX: Disable memory monitoring by default (causes sync overhead)
    enable_memory_monitoring: bool = False
    # Logging mode: 'tqdm' (clean progress bar), 'verbose' (both tqdm + INFO logs)
    # 'tqdm' suppresses step-by-step INFO logs, only shows events like BS changes
    log_mode: str = 'tqdm'
    # Show detailed INFO logs every N steps (0 = never, only events shown)
    verbose_log_interval: int = 500
    # CUDA Graphs: Capture training step as a graph for 10-30% speedup
    # Only works with fixed batch sizes, disabled when dynamic batching is active
    use_cuda_graphs: bool = False
    cuda_graph_warmup_steps: int = 10  # Steps before capturing graph


class TrainingLoopManager(ManagerInterface):
    """
    Manages the core training loop.

    Handles:
        - Async batch prefetching for GPU utilization
        - Gradient accumulation
        - Mixed precision (FP16/BF16)
        - Gradient clipping and monitoring
        - Generation testing during training
        - Coherence measurement
        - Step-based checkpointing

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> loop_mgr = TrainingLoopManager(context)
        >>> loop_mgr.initialize()
        >>> loop_mgr.set_components(metrics_mgr, gen_mgr, checkpoint_mgr)
        >>> train_loss = loop_mgr.train_epoch(model, train_loader, optimizer, scheduler, epoch, config)
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the training loop manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self.prefetcher: Optional[AsyncBatchPrefetcher] = None
        self.scaler: Optional[torch.amp.GradScaler] = None

        # Component references (set via set_components) - typed as protocols
        self._metrics_manager: Optional[MetricsLoggerProtocol] = None
        self._generation_manager: Optional[GenerationProviderProtocol] = None
        self._checkpoint_manager: Optional[CheckpointSaverProtocol] = None
        self._profiler: Optional['NsightProfiler'] = None

        # State
        self._consecutive_failures = 0
        self._global_step = 0

        # Async loss accumulation (avoids cudaStreamSynchronize per batch)
        self._loss_accumulator: Optional[torch.Tensor] = None
        self._loss_count: int = 0

        # CUDA Graph support for 10-30% training speedup
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cuda_graph_captured: bool = False
        self._graph_static_input: Optional[Dict[str, torch.Tensor]] = None
        self._graph_static_loss: Optional[torch.Tensor] = None
        self._graph_batch_size: Optional[int] = None
        self._graph_seq_len: Optional[int] = None

    def initialize(self) -> None:
        """Initialize the training loop manager."""
        self._initialized = True
        self.logger.debug("TrainingLoopManager initialized")

    def cleanup(self) -> None:
        """Stop prefetcher and cleanup resources."""
        if self.prefetcher is not None:
            self.prefetcher.stop()
            self.prefetcher = None
        # Clean up CUDA graph resources
        self._cuda_graph = None
        self._cuda_graph_captured = False
        self._graph_static_input = None
        self._graph_static_loss = None

    def _sync_batch_iterator(self, dataloader, device):
        """
        Simple synchronous batch iterator (no async prefetching).

        This is slower than AsyncBatchPrefetcher but avoids CUDA stream
        race conditions that can cause illegal memory access errors.

        Args:
            dataloader: PyTorch DataLoader
            device: Target device for batch transfer

        Yields:
            (batch_idx, gpu_batch) tuples
        """
        for batch_idx, batch in enumerate(dataloader):
            # Simple synchronous transfer to GPU
            gpu_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    gpu_batch[key] = value.to(device, non_blocking=False)
                else:
                    gpu_batch[key] = value
            yield batch_idx, gpu_batch

    def _capture_cuda_graph(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        sample_batch: Dict[str, torch.Tensor],
        config: TrainingLoopConfig,
        use_scaler: bool,
    ) -> bool:
        """
        Capture training step as a CUDA graph for 10-30% speedup.

        CUDA graphs eliminate kernel launch overhead by capturing and replaying
        a sequence of GPU operations as a single command.

        Requirements:
        - Fixed input shapes (batch_size, seq_len)
        - No dynamic control flow
        - No CPU-GPU synchronization in the captured region

        Args:
            model: Model to capture
            optimizer: Optimizer
            sample_batch: Sample batch for shape detection
            config: Training config
            use_scaler: Whether gradient scaler is used

        Returns:
            True if capture succeeded, False otherwise
        """
        try:
            self.logger.info("Capturing CUDA graph for training step...")

            # Allocate static input buffers with the same shape as sample_batch
            self._graph_batch_size = sample_batch['input_ids'].size(0)
            self._graph_seq_len = sample_batch['input_ids'].size(1)

            self._graph_static_input = {
                'input_ids': torch.empty_like(sample_batch['input_ids']),
                'attention_mask': torch.empty_like(sample_batch['attention_mask']),
                'labels': torch.empty_like(sample_batch['labels']),
            }

            # Copy sample data into static buffers
            for key in self._graph_static_input:
                self._graph_static_input[key].copy_(sample_batch[key])

            # Static loss tensor for output
            self._graph_static_loss = torch.zeros(1, device=self.device)

            # Warmup run (required for CUDA graph capture)
            for _ in range(3):
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                    outputs = model(
                        self._graph_static_input['input_ids'],
                        self._graph_static_input['attention_mask'],
                        self._graph_static_input['labels'],
                    )
                    loss = outputs['loss'] / config.gradient_accumulation_steps
                if use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if use_scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Synchronize before capture
            torch.cuda.synchronize()

            # Capture the graph
            self._cuda_graph = torch.cuda.CUDAGraph()

            optimizer.zero_grad()
            with torch.cuda.graph(self._cuda_graph):
                with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                    outputs = model(
                        self._graph_static_input['input_ids'],
                        self._graph_static_input['attention_mask'],
                        self._graph_static_input['labels'],
                    )
                    loss = outputs['loss'] / config.gradient_accumulation_steps

                if use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Store loss reference (will be updated on replay)
                self._graph_static_loss = loss.detach()

            self._cuda_graph_captured = True
            self.logger.info(
                f"CUDA graph captured successfully for shape "
                f"[{self._graph_batch_size}, {self._graph_seq_len}]"
            )
            return True

        except Exception as e:
            self.logger.warning(f"CUDA graph capture failed: {e}. Falling back to eager mode.")
            self._cuda_graph = None
            self._cuda_graph_captured = False
            return False

    def _replay_cuda_graph(
        self,
        gpu_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Replay the captured CUDA graph with new input data.

        Args:
            gpu_batch: New batch data (must match captured shapes)

        Returns:
            Loss tensor from the graph replay

        Raises:
            RuntimeError: If batch shapes don't match captured graph shapes
        """
        # Validate shapes before copy to prevent silent CUDA errors
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in gpu_batch and key in self._graph_static_input:
                expected_shape = self._graph_static_input[key].shape
                actual_shape = gpu_batch[key].shape
                if expected_shape != actual_shape:
                    raise RuntimeError(
                        f"CUDA graph shape mismatch for '{key}': "
                        f"expected {expected_shape}, got {actual_shape}. "
                        f"CUDA graphs require fixed shapes. Either disable CUDA graphs "
                        f"or ensure consistent batch sizes."
                    )

        # Copy new data into static buffers
        self._graph_static_input['input_ids'].copy_(gpu_batch['input_ids'])
        self._graph_static_input['attention_mask'].copy_(gpu_batch['attention_mask'])
        self._graph_static_input['labels'].copy_(gpu_batch['labels'])

        # Replay the graph
        self._cuda_graph.replay()

        return self._graph_static_loss

    def set_components(
        self,
        metrics_manager: Optional[MetricsLoggerProtocol] = None,
        generation_manager: Optional[GenerationProviderProtocol] = None,
        checkpoint_manager: Optional[CheckpointSaverProtocol] = None,
        profiler: Optional['NsightProfiler'] = None,
    ) -> None:
        """
        Set references to other managers for integration.

        Components are typed as protocols, allowing any implementation that
        satisfies the interface. Pass None to disable a feature.

        Args:
            metrics_manager: Any object implementing MetricsLoggerProtocol
            generation_manager: Any object implementing GenerationProviderProtocol
            checkpoint_manager: Any object implementing CheckpointSaverProtocol
            profiler: NsightProfiler for GPU profiling
        """
        self._metrics_manager = metrics_manager
        self._generation_manager = generation_manager
        self._checkpoint_manager = checkpoint_manager
        self._profiler = profiler

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        config: TrainingLoopConfig,
        vocab_size: int = 50680,
        tokenizer: Any = None,
        generation_config: Optional[Dict[str, Any]] = None,
        coherence_config: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            config: Training loop configuration
            vocab_size: Vocabulary size
            tokenizer: Optional tokenizer for generation
            generation_config: Optional generation configuration
            coherence_config: Optional coherence measurement config

        Returns:
            Average epoch loss

        Raises:
            RuntimeError: If too many consecutive batch failures
        """
        self.assert_initialized()

        model.train()
        device = self.device
        total_loss = 0.0
        num_batches = 0
        num_log_steps = 0  # Track log intervals for correct loss averaging
        epoch_step = 0  # Track steps within this epoch for progress bar
        self._consecutive_failures = 0

        # Reset async loss accumulator for this epoch
        self._reset_loss_accumulator()
        last_synced_loss = 0.0  # For display in progress bar
        last_batch_size = None  # Track batch size changes for dynamic batching

        # Calculate global step
        # Handle IterableDatasets that don't have __len__
        if hasattr(train_loader, 'get_dynamic_total'):
            loader_len = train_loader.get_dynamic_total()
        else:
            try:
                loader_len = len(train_loader)
            except TypeError:
                # Infinite/streaming datasets - use config value or default
                loader_len = self.context.config.get('training', {}).get('steps_per_epoch', 1000)
        # Only initialize global_step on first epoch to preserve resume state
        if self._global_step == 0:
            self._global_step = epoch * loader_len

        # GradScaler only needed for fp16, not bf16
        use_scaler = config.use_amp and config.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler('cuda') if use_scaler else None

        # Check if async prefetching is disabled (safer but slower)
        use_async_prefetch = self.context.config.get('cuda_streams', {}).get('enabled', True)

        if use_async_prefetch:
            # Create async batch prefetcher (faster, uses dedicated CUDA stream)
            logger.debug("Using async batch prefetching (cuda_streams.enabled=true)")
            self.prefetcher = AsyncBatchPrefetcher(train_loader, device, prefetch_count=5)
            batch_iterator = self.prefetcher
            initial_total = self.prefetcher.get_dynamic_total()
        else:
            # Simple synchronous iteration (safer, no CUDA stream issues)
            logger.info("Using synchronous batch iteration (cuda_streams.enabled=false)")
            self.prefetcher = None
            batch_iterator = self._sync_batch_iterator(train_loader, device)
            initial_total = loader_len

        # Progress bar
        is_main = self.context.metadata.get('is_main_process', True)
        pbar = tqdm(total=initial_total, desc=f"Epoch {epoch + 1}", disable=not is_main)

        # Helper for profiler step context
        def get_step_context(step):
            if self._profiler is not None and hasattr(self._profiler, 'step'):
                return self._profiler.step(step)
            from contextlib import nullcontext
            return nullcontext()

        try:
            for batch_idx, gpu_batch in batch_iterator:
                try:
                    # Determine if this is a logging step (only sync loss here)
                    is_accum_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
                    # GPU SYNC FIX: Use global_step for log interval, not batch_idx
                    # This ensures we sync exactly at log_interval steps (e.g., every 50 steps)
                    next_global_step = self._global_step + 1 if is_accum_step else self._global_step
                    is_log_step = is_accum_step and (next_global_step % config.log_interval == 0)

                    # Wrap step in profiler context for NVTX annotations
                    with get_step_context(self._global_step):
                        # Process batch - NO SYNC unless is_log_step
                        _ = self._process_batch(
                            batch_idx, gpu_batch, model, optimizer, scheduler,
                            config, use_scaler, should_sync=is_log_step
                        )

                    num_batches += 1
                    self._consecutive_failures = 0  # Reset on success

                    # Only do logging/generation after full accumulation step
                    if not is_accum_step:
                        # GPU SYNC FIX: Don't update progress bar every batch - causes display overhead
                        # Progress bar will be updated at accumulation steps only
                        continue

                    self._global_step += 1
                    epoch_step += 1

                    # Check max_steps limit
                    if config.max_steps is not None and self._global_step >= config.max_steps:
                        self.logger.info(f"Reached max_steps limit ({config.max_steps}), stopping training")
                        pbar.update(1)
                        raise StopIteration("max_steps reached")

                    # Sync loss only at log intervals (this is the ONLY cudaStreamSync for loss)
                    if is_log_step:
                        last_synced_loss = self._get_accumulated_loss(reset=True)
                        total_loss += last_synced_loss
                        num_log_steps += 1

                        # GPU SYNC FIX: Refresh memory stats at log intervals
                        # This triggers the DynamicBatchScheduler to query GPU memory
                        # Memory queries cause cudaStreamSynchronize, so we batch them here
                        if hasattr(self.prefetcher, 'dataloader'):
                            dl = self.prefetcher.dataloader
                            if hasattr(dl, 'refresh_memory_stats'):
                                dl.refresh_memory_stats()

                        # Log metrics with synced loss
                        self._log_step_metrics(
                            batch_idx, gpu_batch, last_synced_loss, optimizer, config, epoch, pbar
                        )

                    # Generation testing
                    if generation_config and config.generate_every_n_steps > 0:
                        self._maybe_generate(
                            model, vocab_size, tokenizer, generation_config, config
                        )

                    # Coherence measurement
                    if coherence_config:
                        self._maybe_measure_coherence(model, coherence_config)

                    # Step-based checkpointing (use last synced loss)
                    self._maybe_checkpoint(model, optimizer, epoch, last_synced_loss, config)

                    # VRAM optimization
                    self._maybe_clear_cache(model, config)

                    # GPU SYNC FIX: Only update progress bar at log intervals to reduce overhead
                    if is_log_step:
                        new_total = self.prefetcher.get_dynamic_total() if self.prefetcher else pbar.total
                        if pbar.total != new_total:
                            pbar.total = new_total

                        current_bs = gpu_batch['input_ids'].shape[0]

                        # Notify on significant batch size changes (dynamic batching)
                        if last_batch_size is not None and current_bs != last_batch_size:
                            bs_change = current_bs - last_batch_size
                            direction = "↑" if bs_change > 0 else "↓"
                            tqdm.write(
                                f"  [BS {direction}] {last_batch_size} → {current_bs} "
                                f"(step {self._global_step})"
                            )
                        last_batch_size = current_bs

                        # Get current LR for display
                        current_lr = optimizer.param_groups[0]['lr']
                        pbar.set_postfix({
                            'loss': f'{last_synced_loss:.4f}',
                            'BS': current_bs,
                            'LR': f'{current_lr:.1e}'
                        })
                        pbar.n = epoch_step  # Use epoch-relative step for correct progress
                        pbar.refresh()

                except StopIteration:
                    # max_steps reached - exit cleanly
                    break

                except torch.cuda.OutOfMemoryError as oom_error:
                    # OOM-specific handling with BatchSizeController integration
                    torch.cuda.empty_cache()

                    # Get batch controller from context
                    batch_controller = getattr(self.context, 'batch_controller', None)
                    if batch_controller is not None:
                        current_bs = gpu_batch['input_ids'].size(0) if 'input_ids' in gpu_batch else 0
                        new_size = batch_controller.record_failure(current_bs)
                        # Use tqdm.write for clean output during progress bar
                        tqdm.write(
                            f"  [OOM] BS {current_bs} → {new_size} "
                            f"(step {self._global_step}, clearing cache...)"
                        )
                        last_batch_size = new_size  # Update tracking
                        # Update iterator's target batch size if possible
                        if hasattr(self.prefetcher, 'dataloader'):
                            dl = self.prefetcher.dataloader
                            if hasattr(dl, 'set_batch_size'):
                                dl.set_batch_size(new_size)
                            # Connect controller to iterator if not already connected
                            if hasattr(dl, 'set_batch_controller'):
                                dl.set_batch_controller(batch_controller)
                            # Clear iterator's internal buffer to discard accumulated samples
                            if hasattr(dl, 'clear_buffer'):
                                dl.clear_buffer()

                        # CRITICAL: Clear stale prefetched batches that were generated
                        # with the old (too large) batch size. Without this, the retry
                        # will just get another oversized batch from the queue.
                        if hasattr(self.prefetcher, 'clear_queue'):
                            self.prefetcher.clear_queue()

                        # Reset consecutive failures since we handled it
                        self._consecutive_failures = 0
                    else:
                        # No controller - use legacy handling
                        self._consecutive_failures += 1
                        tqdm.write(
                            f"  [OOM ERROR] Batch {batch_idx}: {oom_error}. "
                            f"No batch controller. Failures: {self._consecutive_failures}"
                        )

                    if self._consecutive_failures > config.max_consecutive_failures:
                        raise RuntimeError(
                            f"Too many consecutive OOM failures ({self._consecutive_failures}). "
                            f"Consider reducing max_batch_size in config."
                        )
                    continue

                except Exception as e:
                    self._consecutive_failures += 1
                    self.logger.error(
                        f"Error in batch {batch_idx}: {e}. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )

                    if self._consecutive_failures > config.max_consecutive_failures:
                        raise RuntimeError(
                            f"Too many consecutive batch failures ({self._consecutive_failures}). "
                            f"Last error: {e}"
                        )
                    continue

        finally:
            # Final sync to get any remaining accumulated loss
            remaining_loss = self._get_accumulated_loss(reset=True)
            if remaining_loss > 0:
                total_loss += remaining_loss
                num_log_steps += 1

            if self.prefetcher is not None:
                self.prefetcher.stop()
            pbar.close()

        # Return average loss across all log intervals (not all batches)
        return total_loss / max(num_log_steps, 1)

    def reached_max_steps(self, config: TrainingLoopConfig) -> bool:
        """Check if max_steps limit has been reached."""
        if config.max_steps is None:
            return False
        return self._global_step >= config.max_steps

    def _process_batch(
        self,
        batch_idx: int,
        gpu_batch: Dict[str, torch.Tensor],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: TrainingLoopConfig,
        use_scaler: bool,
        should_sync: bool = False,
    ) -> torch.Tensor:
        """
        Process a single batch with mixed precision and optional profiling.

        Uses async loss accumulation to avoid cudaStreamSynchronize on every batch.
        Loss stays on GPU until should_sync=True (at log intervals).

        Args:
            batch_idx: Current batch index
            gpu_batch: Batch data already on GPU
            model: Model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training loop configuration
            use_scaler: Whether to use gradient scaler (FP16 only)
            should_sync: If True, return synced CPU value; if False, return GPU tensor

        Returns:
            Loss tensor on GPU (no sync) or float if should_sync=True
        """
        input_ids = gpu_batch['input_ids']
        labels = gpu_batch['labels']
        attention_mask = gpu_batch['attention_mask']

        # Helper for profiler range context
        def get_range_context(name):
            if self._profiler is not None and hasattr(self._profiler, 'range'):
                return self._profiler.range(name)
            from contextlib import nullcontext
            return nullcontext()

        # Forward pass with mixed precision and detailed profiling
        # NOTE: Loss scaling for gradient accumulation is INDEPENDENT from AMP scaler.
        # - Dividing by gradient_accumulation_steps normalizes accumulated gradients
        # - AMP scaler.scale() scales for fp16 numerical stability (prevents underflow)
        # These are NOT double-scaling because AMP scaling is reversed in scaler.step()
        with get_range_context("forward"):
            if config.use_amp:
                with get_range_context("forward/autocast_setup"):
                    autocast_ctx = torch.autocast(device_type='cuda', dtype=config.amp_dtype)
                with autocast_ctx:
                    with get_range_context("forward/model_forward"):
                        outputs = model(input_ids, attention_mask, labels)
                    with get_range_context("forward/loss_scale"):
                        # Divide by accumulation steps to average gradients correctly
                        loss = outputs['loss'] / config.gradient_accumulation_steps
            else:
                with get_range_context("forward/model_forward"):
                    outputs = model(input_ids, attention_mask, labels)
                with get_range_context("forward/loss_scale"):
                    loss = outputs['loss'] / config.gradient_accumulation_steps

        # Backward pass with detailed profiling
        with get_range_context("backward"):
            if config.use_amp and use_scaler:
                with get_range_context("backward/scale_loss"):
                    # AMP scaler scales loss for fp16 stability (reversed in scaler.step)
                    scaled_loss = self.scaler.scale(loss)
                with get_range_context("backward/gradient_compute"):
                    scaled_loss.backward()
            else:
                with get_range_context("backward/gradient_compute"):
                    loss.backward()

        # Accumulate loss on GPU (no sync!) - detach to avoid graph retention
        with get_range_context("loss_accumulate"):
            loss_detached = loss.detach() * config.gradient_accumulation_steps
            if self._loss_accumulator is None:
                self._loss_accumulator = loss_detached
            else:
                self._loss_accumulator = self._loss_accumulator + loss_detached
            self._loss_count += 1

        # Gradient accumulation step with detailed profiling
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            with get_range_context("optimizer_step"):
                if use_scaler:
                    with get_range_context("optimizer_step/unscale_grads"):
                        self.scaler.unscale_(optimizer)

                with get_range_context("optimizer_step/grad_clip"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )

                # Log gradients - only sync grad_norm at log intervals
                if self._metrics_manager and should_sync:
                    with get_range_context("optimizer_step/grad_stats"):
                        grad_stats = check_gradients(model)
                        grad_stats['grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                        self._metrics_manager.log_gradients(self._global_step + 1, grad_stats)

                if use_scaler:
                    with get_range_context("optimizer_step/scaler_step"):
                        self.scaler.step(optimizer)
                    with get_range_context("optimizer_step/scaler_update"):
                        self.scaler.update()
                else:
                    with get_range_context("optimizer_step/optimizer_update"):
                        optimizer.step()

                with get_range_context("optimizer_step/zero_grad"):
                    optimizer.zero_grad()
                with get_range_context("optimizer_step/scheduler_step"):
                    scheduler.step()

        # Return detached loss (stays on GPU, no sync)
        return loss_detached

    def _get_accumulated_loss(self, reset: bool = True) -> float:
        """
        Get the accumulated loss value (syncs to CPU).

        This is the ONLY place where cudaStreamSynchronize happens for loss.
        Call this at log intervals, not every batch.

        Args:
            reset: If True, reset accumulator after reading

        Returns:
            Average accumulated loss as a Python float
        """
        if self._loss_accumulator is None or self._loss_count == 0:
            return 0.0

        # This is where sync happens - only at log intervals
        avg_loss = (self._loss_accumulator / self._loss_count).item()

        if reset:
            self._loss_accumulator = None
            self._loss_count = 0

        return avg_loss

    def _reset_loss_accumulator(self) -> None:
        """Reset the loss accumulator without syncing."""
        self._loss_accumulator = None
        self._loss_count = 0

    def _log_step_metrics(
        self,
        batch_idx: int,
        gpu_batch: Dict[str, torch.Tensor],
        loss_value: float,
        optimizer: torch.optim.Optimizer,
        config: TrainingLoopConfig,
        epoch: int,
        pbar: Optional[tqdm] = None,
    ) -> None:
        """Log training metrics for this step."""
        if self._metrics_manager is None:
            return

        current_bs = gpu_batch['input_ids'].shape[0]
        lr = optimizer.param_groups[0]['lr']

        self._metrics_manager.log_training_step(
            step=self._global_step,
            loss=loss_value,
            lr=lr,
            batch_size=current_bs,
            epoch=epoch
        )

        # Console logging - respect log_mode setting
        is_main = self.context.metadata.get('is_main_process', True)
        if not is_main:
            return

        total = self.prefetcher.get_dynamic_total() if self.prefetcher else 0

        # GPU SYNC FIX: Only check memory if explicitly enabled
        mem_info = ""
        mem_pct = 0
        if config.enable_memory_monitoring and torch.cuda.is_available():
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            mem_pct = mem_reserved / mem_total * 100
            mem_info = f" | VRAM: {mem_pct:.0f}%"

        # In tqdm mode, INFO logs only at verbose_log_interval (or never if 0)
        show_info_log = (
            config.log_mode == 'verbose' or
            (config.verbose_log_interval > 0 and self._global_step % config.verbose_log_interval == 0)
        )

        if show_info_log:
            # Use tqdm.write() for clean output that doesn't conflict with progress bar
            msg = (
                f"Step {self._global_step}/{total} | "
                f"BS: {current_bs} | Loss: {loss_value:.4f} | "
                f"LR: {lr:.2e}{mem_info}"
            )
            if pbar is not None:
                tqdm.write(f"[Epoch {epoch + 1}] {msg}")
            else:
                self.logger.info(f"Epoch {epoch + 1} | {msg}")

    def _maybe_generate(
        self,
        model: nn.Module,
        vocab_size: int,
        tokenizer: Any,
        generation_config: Dict[str, Any],
        config: TrainingLoopConfig,
    ) -> None:
        """Generate samples if conditions are met."""
        if self._generation_manager is None:
            return

        if self._global_step <= 0:
            return

        if self._global_step % config.generate_every_n_steps != 0:
            return

        # Skip if previous generation still running
        if self._generation_manager.is_generation_pending():
            self.logger.debug(f"Skipping generation at step {self._global_step} - previous still running")
            return

        self.logger.info(f"Launching async generation at step {self._global_step}")

        self._generation_manager.generate_async(
            model,
            self._global_step,
            vocab_size,
            tokenizer=tokenizer,
            max_length=generation_config.get('max_length', 128),
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            top_k=generation_config.get('top_k', 50),
            repetition_penalty=generation_config.get('repetition_penalty', 1.0),
            prompt=generation_config.get('prompt', None),
        )

        # Process completed generations for WandB logging
        completed = self._generation_manager.process_completed_generations()
        if completed and self._metrics_manager:
            for gen_data in completed:
                self._metrics_manager.log_generation(gen_data['step'], gen_data)

    def _maybe_measure_coherence(
        self,
        model: nn.Module,
        coherence_config: Dict[str, Any],
    ) -> None:
        """Measure coherence if conditions are met."""
        if self._generation_manager is None:
            return

        if not coherence_config.get('enabled', False):
            return

        eval_every = coherence_config.get('eval_every_n_steps', 500)
        if self._global_step <= 0 or self._global_step % eval_every != 0:
            return

        try:
            model.eval()
            with torch.no_grad():
                metrics = self._generation_manager.measure_coherence(
                    model, self._global_step, coherence_config
                )

            if metrics:
                if coherence_config.get('log_to_console', True):
                    self.logger.info(
                        f"[Coherence] Step {self._global_step} | "
                        f"Score: {metrics.get('coherence_score', 0):.4f}"
                    )

                if self._metrics_manager:
                    self._metrics_manager.log_coherence(self._global_step, metrics)

            model.train()

        except Exception as e:
            self.logger.warning(f"Coherence measurement failed: {e}")
            model.train()

    def _maybe_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss_value: float,
        config: TrainingLoopConfig,
    ) -> None:
        """Save checkpoint if conditions are met."""
        if self._checkpoint_manager is None:
            return

        if config.save_steps <= 0:
            return

        if self._global_step % config.save_steps != 0:
            return

        self.logger.info(f"Queuing async checkpoint at step {self._global_step}")
        self._checkpoint_manager.save(
            model, optimizer, epoch, self._global_step,
            {'step_loss': loss_value}
        )

    def _maybe_clear_cache(self, model: nn.Module, config: TrainingLoopConfig) -> None:
        """Clear caches periodically to prevent VRAM fragmentation."""
        if not torch.cuda.is_available():
            return

        # GPU SYNC FIX: Only check memory utilization if monitoring is enabled
        # memory_reserved() causes implicit GPU sync which stalls the pipeline
        if config.enable_memory_monitoring:
            mem_util = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory

            if mem_util > 0.92:
                torch.cuda.empty_cache()
                if hasattr(model, 'clear_caches'):
                    model.clear_caches()
                elif hasattr(model, 'module') and hasattr(model.module, 'clear_caches'):
                    model.module.clear_caches()

        # Periodic cleanup every 1000 steps (does not require memory check)
        if self._global_step % 1000 == 0 and self._global_step > 0:
            if hasattr(model, 'clear_caches'):
                model.clear_caches()
            elif hasattr(model, 'module') and hasattr(model.module, 'clear_caches'):
                model.module.clear_caches()

            if self.prefetcher and hasattr(self.prefetcher, 'dataloader'):
                dataset = getattr(self.prefetcher.dataloader, 'dataset', None)
                if dataset and hasattr(dataset, 'clear_file_cache'):
                    dataset.clear_file_cache()

            # BOTTLENECK FIX: Run empty_cache() in background thread to avoid blocking
            # torch.cuda.empty_cache() is a blocking operation that can stall training
            if not hasattr(self, '_cache_clear_thread') or not self._cache_clear_thread.is_alive():
                self._cache_clear_thread = threading.Thread(
                    target=torch.cuda.empty_cache,
                    daemon=True,
                    name="cuda_cache_clearer"
                )
                self._cache_clear_thread.start()

    def get_global_step(self) -> int:
        """Get current global step."""
        return self._global_step

    def get_status(self) -> Dict[str, Any]:
        """Return current training loop status."""
        return {
            'global_step': self._global_step,
            'consecutive_failures': self._consecutive_failures,
            'prefetcher_active': self.prefetcher is not None,
        }

    def on_error(self, error: Exception) -> None:
        """Handle training errors."""
        self.logger.error(f"Training loop error: {error}", exc_info=True)
        if self.prefetcher:
            self.prefetcher.stop()

    def setup_profiler(self, config: TrainingLoopConfig) -> None:
        """
        Setup Nsight profiler based on config.

        Args:
            config: Training loop config with profiling settings
        """
        if not config.enable_profiling:
            return

        if not NSIGHT_PROFILER_AVAILABLE:
            self.logger.warning("Nsight profiler not available. Install with: pip install torch")
            return

        from ava.cuda.profiler import NsightProfiler, enable_nsight_profiling

        self._profiler = NsightProfiler(
            enabled=True,
            output_dir=config.profile_dir,
            profile_steps=(config.profile_start_step, config.profile_end_step),
            use_cuda_nvtx=True,
            use_torch_profiler=True,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
        )

        enable_nsight_profiling()

        self.logger.info(f"Nsight profiler enabled:")
        self.logger.info(f"  Output dir: {config.profile_dir}")
        self.logger.info(f"  Profile steps: {config.profile_start_step} - {config.profile_end_step}")
        self.logger.info(f"  Run with: nsys profile -o my_profile python train_pipeline.py ...")

    def stop_profiler(self) -> Optional[Path]:
        """
        Stop the profiler and return the output directory.

        Returns:
            Path to profile output directory, or None if profiler wasn't active
        """
        if self._profiler is None:
            return None

        self._profiler.stop()
        output_dir = getattr(self._profiler, 'run_dir', None)
        self.logger.info(f"Profiler stopped. Data saved to: {output_dir}")
        return output_dir
