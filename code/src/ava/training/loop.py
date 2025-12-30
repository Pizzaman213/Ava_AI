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

import copy
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

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
        self, step: int, loss: float, lr: float, batch_size: int, **extra_metrics
    ) -> None:
        """Log metrics for a training step (includes avg_loss and smoothed_loss)."""
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

# Optional diagnostics manager import
try:
    from .diagnostics import DiagnosticsManager
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    DiagnosticsManager = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingLoopConfig:
    """Configuration for the training loop."""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    # Increased default from 10 to 100 to reduce .item() sync frequency.
    # Each log step requires GPU->CPU sync for loss value.
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
    # Disable memory monitoring by default to avoid sync overhead
    enable_memory_monitoring: bool = False
    # Logging mode: 'tqdm' (clean progress bar), 'verbose' (both tqdm + INFO logs)
    # 'tqdm' suppresses step-by-step INFO logs, only shows events like BS changes
    log_mode: str = 'tqdm'
    # Show detailed INFO logs every N steps (0 = never, only events shown)
    verbose_log_interval: int = 500
    # CUDA Graphs: Capture training step as a graph for improved throughput.
    # Only works with fixed batch sizes, disabled when dynamic batching is active.
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
        self._diagnostics_manager: Optional['DiagnosticsManager'] = None

        # State
        self._consecutive_failures = 0
        self._global_step = 0

        # Async loss accumulation (avoids cudaStreamSynchronize per batch)
        self._loss_accumulator: Optional[torch.Tensor] = None
        self._loss_count: int = 0

        # FIX: Track per-step losses for accurate epoch-level averaging
        # These are synced at accumulation boundaries (not log intervals)
        self._step_losses: List[float] = []

        # Store last aux_info for loss component logging
        self._last_aux_info: Optional[List[Dict[str, Any]]] = None

        # CUDA Graph support for training optimization
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
        # Properly clean up CUDA graph resources to prevent memory leaks
        self._cleanup_cuda_graph()

    def _cleanup_cuda_graph(self) -> None:
        """Properly free CUDA graph resources from GPU memory."""
        # Issue #7 fix: Synchronize before cleanup to ensure all operations complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # May fail if CUDA in bad state

        # Free static input tensors - explicitly pop to ensure GPU release
        if self._graph_static_input is not None:
            for key in list(self._graph_static_input.keys()):
                tensor = self._graph_static_input.pop(key)
                if tensor is not None:
                    del tensor
            self._graph_static_input = None

        # Free static loss tensor
        if self._graph_static_loss is not None:
            del self._graph_static_loss
            self._graph_static_loss = None

        # Reset the CUDA graph (releases GPU memory)
        if self._cuda_graph is not None:
            try:
                self._cuda_graph.reset()
            except Exception as e:
                self.logger.debug(f"CUDA graph reset warning: {e}")
            self._cuda_graph = None

        self._cuda_graph_captured = False
        self._graph_batch_size = None
        self._graph_seq_len = None

        # Issue #7 fix: Force garbage collection before CUDA memory cleanup
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        Capture training step as a CUDA graph for improved throughput.

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
            # CUDA graphs are not compatible with DeepSpeed (due to dynamic control flow)
            from ..training.deepspeed_utils import is_deepspeed_engine
            if is_deepspeed_engine(model):
                self.logger.info("CUDA graphs disabled: incompatible with DeepSpeed")
                return False

            self.logger.info("Capturing CUDA graph for training step...")

            # Allocate static input buffers with the same shape as sample_batch
            self._graph_batch_size = sample_batch['input_ids'].size(0)
            self._graph_seq_len = sample_batch['input_ids'].size(1)

            self._graph_static_input = {
                'input_ids': torch.empty_like(sample_batch['input_ids']),
                'attention_mask': torch.empty_like(sample_batch['attention_mask']),
                'labels': torch.empty_like(sample_batch['labels']),
            }

            # COHERENCE FIX: Include position_ids in CUDA graph for sequence packing
            # Without this, document-relative positions are lost, causing cross-document attention
            if 'position_ids' in sample_batch:
                self._graph_static_input['position_ids'] = torch.empty_like(sample_batch['position_ids'])

            # Copy sample data into static buffers
            for key in self._graph_static_input:
                self._graph_static_input[key].copy_(sample_batch[key])

            # Static loss tensor for output
            self._graph_static_loss = torch.zeros(1, device=self.device)

            # Warmup run (required for CUDA graph capture)
            # FIX: Save optimizer state before warmup, restore after to prevent
            # warmup runs from polluting the momentum buffers
            # Issue #8 fix: Use deep copy for optimizer state to handle nested structures
            # The shallow clone approach may miss nested tensors in some optimizers (e.g., Lion)
            if optimizer.state:
                try:
                    # Attempt deep copy (handles all nested structures)
                    optimizer_state_before_warmup = copy.deepcopy(optimizer.state)
                except Exception:
                    # Fallback to manual clone if deepcopy fails (some CUDA tensors)
                    optimizer_state_before_warmup = {}
                    for k, v in optimizer.state.items():
                        if isinstance(v, dict):
                            optimizer_state_before_warmup[k] = {
                                sk: sv.clone().detach() if isinstance(sv, torch.Tensor) else copy.copy(sv)
                                for sk, sv in v.items()
                            }
                        else:
                            optimizer_state_before_warmup[k] = v.clone() if isinstance(v, torch.Tensor) else copy.copy(v)
            else:
                optimizer_state_before_warmup = {}

            for _ in range(3):
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                    # COHERENCE FIX: Use keyword args to ensure position_ids and 2D attention_mask
                    # are properly forwarded for sequence packing with document boundaries
                    outputs = model(
                        input_ids=self._graph_static_input['input_ids'],
                        attention_mask=self._graph_static_input['attention_mask'],
                        labels=self._graph_static_input['labels'],
                        position_ids=self._graph_static_input.get('position_ids'),
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

            # FIX: Restore optimizer state after warmup to avoid polluted momentum
            if optimizer_state_before_warmup:
                for param_id, state_dict in optimizer_state_before_warmup.items():
                    if param_id in optimizer.state:
                        for key, value in state_dict.items():
                            if isinstance(value, torch.Tensor):
                                optimizer.state[param_id][key].copy_(value)
                            else:
                                optimizer.state[param_id][key] = value

            # Synchronize before capture
            torch.cuda.synchronize()

            # Capture the graph
            self._cuda_graph = torch.cuda.CUDAGraph()

            optimizer.zero_grad()
            with torch.cuda.graph(self._cuda_graph):
                with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                    # COHERENCE FIX: Use keyword args to ensure position_ids and 2D attention_mask
                    # are properly forwarded for sequence packing with document boundaries
                    outputs = model(
                        input_ids=self._graph_static_input['input_ids'],
                        attention_mask=self._graph_static_input['attention_mask'],
                        labels=self._graph_static_input['labels'],
                        position_ids=self._graph_static_input.get('position_ids'),
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
            self._cleanup_cuda_graph()  # Properly free any partially allocated resources
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
        # COHERENCE FIX: Include position_ids in validation for sequence packing
        keys_to_validate = ['input_ids', 'attention_mask', 'labels']
        if 'position_ids' in self._graph_static_input:
            keys_to_validate.append('position_ids')

        for key in keys_to_validate:
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
        # COHERENCE FIX: Copy position_ids for proper document-relative positions
        self._graph_static_input['input_ids'].copy_(gpu_batch['input_ids'])
        self._graph_static_input['attention_mask'].copy_(gpu_batch['attention_mask'])
        self._graph_static_input['labels'].copy_(gpu_batch['labels'])
        if 'position_ids' in self._graph_static_input and 'position_ids' in gpu_batch:
            self._graph_static_input['position_ids'].copy_(gpu_batch['position_ids'])

        # Replay the graph
        self._cuda_graph.replay()

        return self._graph_static_loss

    def set_components(
        self,
        metrics_manager: Optional[MetricsLoggerProtocol] = None,
        generation_manager: Optional[GenerationProviderProtocol] = None,
        checkpoint_manager: Optional[CheckpointSaverProtocol] = None,
        profiler: Optional['NsightProfiler'] = None,
        diagnostics_manager: Optional['DiagnosticsManager'] = None,
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
            diagnostics_manager: DiagnosticsManager for detailed training diagnostics
        """
        self._metrics_manager = metrics_manager
        self._generation_manager = generation_manager
        self._checkpoint_manager = checkpoint_manager
        self._profiler = profiler
        self._diagnostics_manager = diagnostics_manager

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
        num_synced_batches = 0  # FIX: Track actual number of batches averaged (for accurate epoch loss)
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

        # FIX: Properly handle global_step for both fresh start and resume
        # Check if we're resuming from a checkpoint (step would be in context)
        resume_step = self.context.metadata.get('resume_step', 0)
        if resume_step > 0 and self._global_step == 0:
            # Resuming from checkpoint - use the resumed step
            self._global_step = resume_step
            self.logger.info(f"Resumed training from step {resume_step}")
        elif self._global_step == 0 and epoch == 0:
            # Fresh start at epoch 0
            self._global_step = 0
        elif self._global_step == 0:
            # Later epoch without checkpoint step - log warning and estimate
            # This can happen with streaming datasets or interrupted training
            self.logger.warning(
                f"Starting epoch {epoch} without resume_step in checkpoint. "
                f"Step count may be inaccurate for metrics/logging."
            )
            self._global_step = epoch * loader_len  # Best effort estimate

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
                    # CRITICAL: Truncate batch ONLY during OOM recovery
                    # This handles the case where we need to reduce batch size after an OOM
                    # We do NOT truncate in 'stable' mode (after calibration) - that defeats the purpose
                    batch_controller = getattr(self.context, 'batch_controller', None)
                    if batch_controller is not None and batch_controller.mode == 'recovering':
                        # Use get_batch_size() - the actual safe fallback after OOM
                        # NOT safe_ceiling which is just batch_size - 1
                        max_safe = batch_controller.get_batch_size()
                        current_bs = gpu_batch['input_ids'].size(0) if 'input_ids' in gpu_batch else 0
                        if current_bs > max_safe and max_safe > 0:
                            # Truncate batch to safe size (only during recovery)
                            gpu_batch = {
                                k: v[:max_safe] if isinstance(v, torch.Tensor) else v
                                for k, v in gpu_batch.items()
                            }
                            tqdm.write(f"  [TRUNCATE] Batch {current_bs} → {max_safe} (OOM recovery)")

                    # Determine if this is a logging step (only sync loss here)
                    is_accum_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
                    # Use global_step for log interval, not batch_idx.
                    # This ensures we sync exactly at log_interval steps.
                    next_global_step = self._global_step + 1 if is_accum_step else self._global_step
                    is_log_step = is_accum_step and (next_global_step % config.log_interval == 0)

                    # Wrap step in profiler context for NVTX annotations
                    with get_step_context(self._global_step):
                        # CUDA Graphs: Capture after warmup, replay on matching shapes
                        use_graph = False
                        if config.use_cuda_graphs and batch_idx >= config.cuda_graph_warmup_steps:
                            # Check if we need to capture the graph
                            if not self._cuda_graph_captured:
                                # Try to capture CUDA graph
                                tqdm.write(f"  [CUDA Graph] Capturing at batch {batch_idx}...")
                                success = self._capture_cuda_graph(
                                    model, optimizer, gpu_batch, config, use_scaler
                                )
                                if success:
                                    tqdm.write(f"  [CUDA Graph] Captured! Replay enabled for 15-25% speedup")
                                else:
                                    tqdm.write(f"  [CUDA Graph] Capture failed, using eager mode")

                            # Try to replay if captured and shapes match
                            if self._cuda_graph_captured:
                                current_bs = gpu_batch['input_ids'].size(0)
                                current_seq = gpu_batch['input_ids'].size(1)
                                if current_bs == self._graph_batch_size and current_seq == self._graph_seq_len:
                                    use_graph = True

                        if use_graph:
                            # Replay CUDA graph (15-25% faster)
                            loss = self._replay_cuda_graph(gpu_batch)
                            # Accumulate loss (same as _process_batch)
                            # FIX: Graph loss is pre-divided by gradient_accumulation_steps,
                            # restore original scale to match _process_batch behavior
                            original_loss = loss.detach() * config.gradient_accumulation_steps
                            if self._loss_accumulator is None:
                                self._loss_accumulator = original_loss
                            else:
                                self._loss_accumulator = self._loss_accumulator + original_loss
                            self._loss_count += 1
                            # Handle optimizer step for graph replay
                            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                                if use_scaler:
                                    self.scaler.step(optimizer)
                                    self.scaler.update()
                                else:
                                    optimizer.step()
                                optimizer.zero_grad()
                                scheduler.step()
                        else:
                            # Process batch normally - NO SYNC unless is_log_step
                            _ = self._process_batch(
                                batch_idx, gpu_batch, model, optimizer, scheduler,
                                config, use_scaler, should_sync=is_log_step
                            )

                    num_batches += 1
                    self._consecutive_failures = 0  # Reset on success

                    # Only do logging/generation after full accumulation step
                    if not is_accum_step:
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
                        # FIX: Capture batch count BEFORE reset to avoid using stale value
                        batch_count_for_this_sync = self._loss_count if self._loss_count > 0 else 1
                        last_synced_loss = self._get_accumulated_loss(reset=True)
                        # FIX: Track actual number of batches for accurate weighted average
                        # Each synced loss is an average over batch_count_for_this_sync batches
                        total_loss += last_synced_loss * batch_count_for_this_sync
                        num_synced_batches += batch_count_for_this_sync
                        num_log_steps += 1

                        # Refresh memory stats at log intervals.
                        # Memory queries cause synchronization, so batch them here.
                        if hasattr(self.prefetcher, 'dataloader'):
                            dl = self.prefetcher.dataloader
                            if hasattr(dl, 'refresh_memory_stats'):
                                dl.refresh_memory_stats()

                        # Log metrics with synced loss
                        self._log_step_metrics(
                            batch_idx, gpu_batch, last_synced_loss, optimizer, config, epoch, pbar, model
                        )

                    # Generation testing
                    if generation_config and generation_config.get('enabled', True) and config.generate_every_n_steps > 0:
                        self._maybe_generate(
                            model, vocab_size, tokenizer, generation_config, config
                        )

                    # Coherence measurement
                    if coherence_config:
                        self._maybe_measure_coherence(model, coherence_config)

                    # Flush all accumulated wandb metrics for this step
                    if is_log_step and self._metrics_manager is not None:
                        self._metrics_manager.flush_wandb_metrics()

                    # Step-based checkpointing (use last synced loss)
                    self._maybe_checkpoint(model, optimizer, epoch, last_synced_loss, config)

                    # VRAM optimization
                    self._maybe_clear_cache(model, config)

                    # Update progress bar at log intervals
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
                    # OOM-specific handling with comprehensive state reset
                    # Initialize OOMRecoveryManager if not already present
                    if not hasattr(self, '_oom_recovery_manager'):
                        from ..optimizations.oom_recovery import OOMRecoveryManager
                        self._oom_recovery_manager = OOMRecoveryManager(
                            context=self.context,
                            min_batch_size=1,
                            reduction_factor=0.75,
                            max_recovery_attempts=3
                        )

                    # Get current batch size
                    current_bs = gpu_batch['input_ids'].size(0) if 'input_ids' in gpu_batch else 0

                    # Attempt comprehensive OOM recovery
                    new_batch_size = self._oom_recovery_manager.recover_from_oom(current_bs)

                    if new_batch_size is None:
                        # Recovery failed - cannot continue
                        raise RuntimeError(
                            f"OOM recovery failed after {self._oom_recovery_manager.recovery_count} attempts. "
                            f"Cannot reduce batch size below {self._oom_recovery_manager.min_batch_size}."
                        )

                    # Update batch controller if present
                    batch_controller = getattr(self.context, 'batch_controller', None)
                    if batch_controller is not None:
                        # Inform controller of new size
                        batch_controller.record_failure(current_bs)
                        # Update iterator's target batch size if possible
                        if hasattr(self.prefetcher, 'dataloader'):
                            dl = self.prefetcher.dataloader
                            if hasattr(dl, 'set_batch_size'):
                                dl.set_batch_size(new_batch_size)
                            # Connect controller to iterator if not already connected
                            if hasattr(dl, 'set_batch_controller'):
                                dl.set_batch_controller(batch_controller)
                            # Clear iterator's internal buffer to discard accumulated samples
                            if hasattr(dl, 'clear_buffer'):
                                dl.clear_buffer()

                    # Synchronize new batch size across ranks if distributed
                    if self.context.world_size > 1:
                        try:
                            from ..training.distributed_sync import DistributedStateManager
                            import torch.distributed as dist

                            dist_manager = DistributedStateManager(
                                rank=self.context.rank,
                                world_size=self.context.world_size
                            )
                            # Use MIN to ensure all ranks use same (smallest) batch size
                            synced_batch_size = dist_manager.synchronized_update(
                                value=new_batch_size,
                                reduction_op=dist.ReduceOp.MIN,
                                validate_fn=lambda x: x > 0,
                                value_name="oom_recovery_batch_size"
                            )
                            new_batch_size = synced_batch_size
                        except Exception as e:
                            self.logger.warning(f"Failed to synchronize batch size across ranks: {e}")

                    # Update tracking
                    last_batch_size = new_batch_size

                    # Use tqdm.write for clean output during progress bar
                    tqdm.write(
                        f"  [OOM RECOVERY] BS {current_bs} → {new_batch_size} "
                        f"(step {self._global_step}, attempt {self._oom_recovery_manager.recovery_count})"
                    )

                    # Reset consecutive failures since we handled it
                    self._consecutive_failures = 0
                    continue

                except Exception as e:
                    self._consecutive_failures += 1
                    # Add full traceback for debugging
                    import traceback
                    tb_str = ''.join(traceback.format_tb(e.__traceback__))
                    self.logger.error(
                        f"Error in batch {batch_idx}: {e}\n"
                        f"Traceback:\n{tb_str}\n"
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
            # FIX: Check _loss_count instead of remaining_loss > 0 to handle edge cases
            # where accumulated loss might be exactly 0 but batches were processed
            remaining_count = self._loss_count
            remaining_loss = self._get_accumulated_loss(reset=True)
            if remaining_count > 0:
                # FIX: Weight by actual batch count for accurate averaging
                total_loss += remaining_loss * remaining_count
                num_synced_batches += remaining_count
                num_log_steps += 1

            if self.prefetcher is not None:
                self.prefetcher.stop()
            pbar.close()

        # FIX: Return weighted average over all batches (not log intervals)
        # This correctly handles variable batch counts at log interval boundaries
        return total_loss / max(num_synced_batches, 1)

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
        # Get position_ids if provided (from sequence packing with per-document positions)
        position_ids = gpu_batch.get('position_ids', None)

        # Detect DeepSpeed engine
        from ..training.deepspeed_utils import is_deepspeed_engine
        is_deepspeed = is_deepspeed_engine(model)

        # Helper for profiler range context
        def get_range_context(name):
            if self._profiler is not None and hasattr(self._profiler, 'range'):
                return self._profiler.range(name)
            from contextlib import nullcontext
            return nullcontext()

        # Diagnostics timing context
        def get_timing_context(phase):
            if self._diagnostics_manager is not None:
                return self._diagnostics_manager.time_phase(phase)
            from contextlib import nullcontext
            return nullcontext()

        # Forward pass with mixed precision and detailed profiling
        # NOTE: Loss scaling for gradient accumulation is INDEPENDENT from AMP scaler.
        # - Dividing by gradient_accumulation_steps normalizes accumulated gradients
        # - AMP scaler.scale() scales for fp16 numerical stability (prevents underflow)
        # These are NOT double-scaling because AMP scaling is reversed in scaler.step()
        with get_range_context("forward"), get_timing_context("forward"):
            # Only use torch.autocast if NOT using DeepSpeed (DeepSpeed manages mixed precision internally)
            if config.use_amp and not is_deepspeed:
                with get_range_context("forward/autocast_setup"):
                    autocast_ctx = torch.autocast(device_type='cuda', dtype=config.amp_dtype)
                with autocast_ctx:
                    with get_range_context("forward/model_forward"):
                        # Use keyword args to support position_ids from sequence packing
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            position_ids=position_ids,
                        )
                    with get_range_context("forward/loss_scale"):
                        # CRITICAL DEBUG: Check what model returns
                        raw_loss = outputs['loss']
                        if raw_loss.numel() != 1:
                            self.logger.error(
                                f"FORWARD: Model returned non-scalar loss! "
                                f"Shape: {raw_loss.shape}, numel: {raw_loss.numel()}"
                            )
                            raw_loss = raw_loss.mean()
                            outputs['loss'] = raw_loss
                        # Divide by accumulation steps to average gradients correctly
                        loss = outputs['loss'] / config.gradient_accumulation_steps
            else:
                # No autocast: either AMP disabled OR DeepSpeed manages mixed precision
                with get_range_context("forward/model_forward"):
                    # Use keyword args to support position_ids from sequence packing
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        position_ids=position_ids,
                    )
                with get_range_context("forward/loss_scale"):
                    # CRITICAL DEBUG: Check what model returns
                    raw_loss = outputs['loss']
                    if raw_loss.numel() != 1:
                        self.logger.error(
                            f"FORWARD: Model returned non-scalar loss! "
                            f"Shape: {raw_loss.shape}, numel: {raw_loss.numel()}"
                        )
                        raw_loss = raw_loss.mean()
                        outputs['loss'] = raw_loss
                    loss = outputs['loss'] / config.gradient_accumulation_steps

        # Backward pass with detailed profiling
        with get_range_context("backward"), get_timing_context("backward"):
            if is_deepspeed:
                # DeepSpeed path - engine handles backward + scaling automatically
                with get_range_context("backward/deepspeed_backward"):
                    model.backward(loss)
            elif config.use_amp and use_scaler:
                with get_range_context("backward/scale_loss"):
                    # AMP scaler scales loss for fp16 stability (reversed in scaler.step)
                    scaled_loss = self.scaler.scale(loss)
                with get_range_context("backward/gradient_compute"):
                    scaled_loss.backward()
            else:
                with get_range_context("backward/gradient_compute"):
                    loss.backward()

        # Accumulate loss on GPU (no sync!) - detach to avoid graph retention
        # FIX: Store the original (unscaled) loss for metrics, not the scaled loss
        # The loss was divided by gradient_accumulation_steps for correct gradient averaging,
        # but we want to display/track the actual loss value
        with get_range_context("loss_accumulate"):
            # Get original loss value (before division by accumulation steps)
            original_loss = outputs['loss'].detach()

            # CRITICAL DEBUG: Check loss shape before accumulation
            if original_loss.numel() != 1:
                self.logger.error(
                    f"Model returned non-scalar loss! Shape: {original_loss.shape}, "
                    f"numel: {original_loss.numel()}, values: {original_loss}"
                )
                # Force to scalar to prevent crash
                original_loss = original_loss.mean()

            if self._loss_accumulator is None:
                self._loss_accumulator = original_loss
            else:
                self._loss_accumulator = self._loss_accumulator + original_loss
            self._loss_count += 1

            # Store aux_info for logging loss components
            if 'aux_info' in outputs:
                self._last_aux_info = outputs['aux_info']

        # Gradient accumulation step with detailed profiling
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            with get_range_context("optimizer_step"), get_timing_context("optimizer_step"):
                if is_deepspeed:
                    # DeepSpeed path - engine.step() handles clipping, optimizer, and scheduler automatically
                    with get_range_context("optimizer_step/deepspeed_step"):
                        model.step()

                    # Log gradients if needed (DeepSpeed handles clipping internally)
                    if self._metrics_manager and should_sync:
                        with get_range_context("optimizer_step/grad_stats"):
                            # For DeepSpeed, access the underlying module
                            grad_stats = check_gradients(model.module if hasattr(model, 'module') else model)
                            self._metrics_manager.log_gradients(self._global_step + 1, grad_stats)
                else:
                    # DDP/single-GPU path - manual step
                    if use_scaler:
                        with get_range_context("optimizer_step/unscale_grads"):
                            self.scaler.unscale_(optimizer)

                    with get_range_context("optimizer_step/grad_clip"):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )

                    # TRAINING HEALTH: Detect gradient issues early
                    grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                    if grad_norm_val < 1e-7:
                        logger.warning(f"Step {self._global_step + 1}: VANISHING gradients (norm={grad_norm_val:.2e}) - model may not be learning")
                    elif grad_norm_val > 1e4:
                        logger.warning(f"Step {self._global_step + 1}: EXPLODING gradients (norm={grad_norm_val:.2e}) - consider reducing LR")

                    # Log gradients - only sync grad_norm at log intervals
                    if self._metrics_manager and should_sync:
                        with get_range_context("optimizer_step/grad_stats"):
                            grad_stats = check_gradients(model)
                            grad_stats['grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                            self._metrics_manager.log_gradients(self._global_step + 1, grad_stats)

                    # OPTIMIZATION: Only sync for multi-GPU training
                    # Single GPU doesn't need explicit synchronization before optimizer step
                    # DDP handles gradient sync via hooks during backward(), but with async prefetching
                    # we need to ensure all CUDA operations are complete before stepping
                    if self.context.world_size > 1:
                        torch.cuda.synchronize()

                    if use_scaler:
                        with get_range_context("optimizer_step/scaler_step"):
                            self.scaler.step(optimizer)
                        with get_range_context("optimizer_step/scaler_update"):
                            self.scaler.update()
                        # TRAINING HEALTH: Detect when scaler skips optimizer step due to NaN/Inf
                        if hasattr(self.scaler, '_found_inf_per_device'):
                            found_inf = any(v.item() for v in self.scaler._found_inf_per_device.values())
                            if found_inf:
                                if not hasattr(self, '_skipped_steps'):
                                    self._skipped_steps = 0
                                self._skipped_steps += 1
                                logger.warning(f"Step {self._global_step + 1}: Optimizer step SKIPPED (gradient overflow, total skipped: {self._skipped_steps})")
                    else:
                        with get_range_context("optimizer_step/optimizer_update"):
                            optimizer.step()

                    with get_range_context("optimizer_step/zero_grad"):
                        optimizer.zero_grad()
                    with get_range_context("optimizer_step/scheduler_step"):
                        scheduler.step()

        # Return detached loss (stays on GPU, no sync)
        return loss.detach()

    def _get_accumulated_loss(self, reset: bool = True, sync_distributed: bool = True) -> float:
        """
        Get the accumulated loss value with proper distributed synchronization.

        This is the ONLY place where cudaStreamSynchronize happens for loss.
        Call this at log intervals, not every batch.

        In distributed training, this method synchronizes loss values across all ranks
        using all_reduce(AVG) to ensure all ranks see the same loss value.

        Args:
            reset: If True, reset accumulator after reading
            sync_distributed: If True, synchronize across ranks in distributed training
                              (default: True, should always be True for correctness)

        Returns:
            Average accumulated loss as a Python float, synchronized across all ranks
        """
        if self._loss_accumulator is None or self._loss_count == 0:
            return 0.0

        # Compute average on GPU (stays on GPU for now)
        avg_loss_tensor = self._loss_accumulator / self._loss_count

        # CRITICAL: Synchronize across ranks BEFORE .item()
        # In distributed training, different ranks may have slightly different loss values
        # due to different batch samples. We need to average them for correctness.
        if sync_distributed and self.context.world_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    # Average loss across all ranks
                    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            except Exception as e:
                # If distributed sync fails, log warning but continue with local value
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to synchronize loss across ranks: {e}")

        # Now safe to move to CPU - all ranks have the same value
        avg_loss = avg_loss_tensor.item()

        if reset:
            self._loss_accumulator = None
            self._loss_count = 0

        return avg_loss

    def _reset_loss_accumulator(self) -> None:
        """Reset the loss accumulator without syncing."""
        self._loss_accumulator = None
        self._loss_count = 0
        self._step_losses.clear()  # FIX: Also clear step losses list

    def _clear_cuda_memory(self, tensors_to_clear: Optional[List[Any]] = None) -> None:
        """
        Aggressively clear CUDA memory to prevent fragmentation.

        Args:
            tensors_to_clear: Optional list of tensors/dicts to clear

        Example:
            >>> self._clear_cuda_memory([loss, outputs])
        """
        if tensors_to_clear:
            for tensor in tensors_to_clear:
                if tensor is None:
                    continue
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor.detach_()
                    try:
                        tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
                    except Exception:
                        pass
                elif isinstance(tensor, dict):
                    for val in tensor.values():
                        if isinstance(val, torch.Tensor) and val.is_cuda:
                            val.detach_()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_step_metrics(
        self,
        batch_idx: int,
        gpu_batch: Dict[str, torch.Tensor],
        loss_value: float,
        optimizer: torch.optim.Optimizer,
        config: TrainingLoopConfig,
        epoch: int,
        pbar: Optional[tqdm] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Log training metrics for this step."""
        if self._metrics_manager is None:
            return

        current_bs = gpu_batch['input_ids'].shape[0]
        lr = optimizer.param_groups[0]['lr']

        # Extract loss components from aux_info if available
        extra_metrics = {}
        if self._last_aux_info is not None and len(self._last_aux_info) > 0:
            if 'loss_components' in self._last_aux_info[0]:
                components = self._last_aux_info[0]['loss_components']
                extra_metrics.update({
                    'train/cross_entropy_loss': components['cross_entropy_loss'],
                    'train/aux_loss': components['aux_loss'],
                    'train/aux_loss_ratio': components['aux_loss_ratio']
                })

            # Extract routing metrics from all layers and aggregate
            router_types_seen = set()
            total_expert_utilization = {}
            num_layers_with_routing = 0

            for layer_aux in self._last_aux_info:
                # Track router type(s) being used
                if 'router_type' in layer_aux:
                    router_types_seen.add(layer_aux['router_type'])

                # Aggregate expert utilization across layers
                for key, value in layer_aux.items():
                    if key.startswith('expert_') and key.endswith('_utilization'):
                        if key not in total_expert_utilization:
                            total_expert_utilization[key] = 0.0
                        # Sum utilization across layers (will average later)
                        if isinstance(value, (int, float)):
                            total_expert_utilization[key] += value
                        elif hasattr(value, 'item'):
                            # Handle multi-element tensors by taking the mean
                            if hasattr(value, 'numel') and value.numel() > 1:
                                total_expert_utilization[key] += value.mean().item()
                            else:
                                total_expert_utilization[key] += value.item()

                # Track MoE-specific metrics from routing
                if 'routing_entropy' in layer_aux and layer_aux['routing_entropy'] is not None:
                    entropy_val = layer_aux['routing_entropy']
                    if hasattr(entropy_val, 'item'):
                        # Handle multi-element tensors by taking the mean
                        if hasattr(entropy_val, 'numel') and entropy_val.numel() > 1:
                            entropy_val = entropy_val.mean().item()
                        else:
                            entropy_val = entropy_val.item()
                    extra_metrics[f'routing/entropy_layer_{num_layers_with_routing}'] = entropy_val

                if 'balance_score' in layer_aux and layer_aux['balance_score'] is not None:
                    balance_val = layer_aux['balance_score']
                    if hasattr(balance_val, 'item'):
                        # Handle multi-element tensors by taking the mean
                        if hasattr(balance_val, 'numel') and balance_val.numel() > 1:
                            balance_val = balance_val.mean().item()
                        else:
                            balance_val = balance_val.item()
                    extra_metrics[f'routing/balance_layer_{num_layers_with_routing}'] = balance_val

                if 'router_confidence' in layer_aux and layer_aux['router_confidence'] is not None:
                    conf_val = layer_aux['router_confidence']
                    if hasattr(conf_val, 'item'):
                        # Handle multi-element tensors by taking the mean
                        if hasattr(conf_val, 'numel') and conf_val.numel() > 1:
                            conf_val = conf_val.mean().item()
                        else:
                            conf_val = conf_val.item()
                    extra_metrics[f'routing/confidence_layer_{num_layers_with_routing}'] = conf_val

                num_layers_with_routing += 1

            # Log router type(s) being used
            if router_types_seen:
                # Convert set to sorted string for consistent logging
                router_type_str = ','.join(sorted(router_types_seen))
                # WandB doesn't support string metrics directly in log(), so we log it as a tag
                # Instead, we'll just log the count of router types
                extra_metrics['routing/num_router_types'] = len(router_types_seen)
                # Store router type for later reference (could be logged to WandB config)
                if not hasattr(self, '_router_types_logged'):
                    self._router_types_logged = True
                    self.logger.info(f"Using router type(s): {router_type_str}")

            # Log averaged per-expert utilization across layers
            if total_expert_utilization and num_layers_with_routing > 0:
                for expert_key, total_util in total_expert_utilization.items():
                    avg_util = total_util / num_layers_with_routing
                    extra_metrics[f'routing/{expert_key}'] = avg_util

        self._metrics_manager.log_training_step(
            step=self._global_step,
            loss=loss_value,
            lr=lr,
            batch_size=current_bs,
            epoch=epoch,
            **extra_metrics
        )

        # Collect and log diagnostics if enabled
        if self._diagnostics_manager is not None and model is not None:
            # Per-layer gradients
            layer_stats = self._diagnostics_manager.collect_per_layer_gradients(
                model, self._global_step
            )
            if layer_stats:
                self._metrics_manager.log_per_layer_gradients(self._global_step, layer_stats)

            # Expert routing diagnostics
            routing_stats = self._diagnostics_manager.collect_routing_stats(
                model, self._global_step
            )
            if routing_stats:
                self._metrics_manager.log_routing_diagnostics(self._global_step, routing_stats)

            # Memory breakdown
            memory_breakdown = self._diagnostics_manager.collect_memory_breakdown(
                model, optimizer, self._global_step
            )
            if memory_breakdown:
                self._metrics_manager.log_memory_breakdown(self._global_step, memory_breakdown)

            # Timing profile
            timing_profile = self._diagnostics_manager.get_timing_profile(self._global_step)
            if timing_profile:
                self._metrics_manager.log_timing_profile(self._global_step, timing_profile)

        # Console logging - respect log_mode setting
        is_main = self.context.metadata.get('is_main_process', True)
        if not is_main:
            return

        total = self.prefetcher.get_dynamic_total() if self.prefetcher else 0

        # Only check memory if explicitly enabled
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
            # Add training health info
            health_info = ""
            if hasattr(self, '_skipped_steps') and self._skipped_steps > 0:
                health_info = f" | Skipped: {self._skipped_steps}"

            msg = (
                f"Step {self._global_step}/{total} | "
                f"BS: {current_bs} | Loss: {loss_value:.4f} | "
                f"LR: {lr:.2e}{mem_info}{health_info}"
            )
            if pbar is not None:
                tqdm.write(f"[Epoch {epoch + 1}] {msg}")
            else:
                self.logger.info(f"Epoch {epoch + 1} | {msg}")

            # Log expert utilization summary (helps diagnose expert collapse)
            if self._last_aux_info is not None and len(self._last_aux_info) > 0:
                # Get expert utilization from first layer (representative)
                first_layer = self._last_aux_info[0]
                expert_utils = []
                for key, value in first_layer.items():
                    if key.startswith('expert_') and key.endswith('_utilization'):
                        if isinstance(value, (int, float)):
                            expert_utils.append(value)
                        elif hasattr(value, 'item'):
                            expert_utils.append(value.item() if value.numel() == 1 else value.mean().item())
                if expert_utils:
                    # Check for expert collapse (one expert getting > 50% of tokens)
                    max_util = max(expert_utils)
                    total_tokens = sum(expert_utils)
                    if total_tokens > 0 and max_util / total_tokens > 0.5:
                        tqdm.write(f"  [WARN] Expert collapse detected: max expert has {max_util/total_tokens*100:.0f}% of tokens")

    def _maybe_generate(
        self,
        model: nn.Module,
        vocab_size: int,
        tokenizer: Any,
        generation_config: Dict[str, Any],
        config: TrainingLoopConfig,
    ) -> None:
        """Generate samples using async CPU generation."""
        if self._generation_manager is None:
            return

        # Always check for and display completed generations first
        completed = self._generation_manager.process_completed_generations()
        if completed:
            tqdm.write(f"  [Gen] Found {len(completed)} completed generation(s)")
        for gen_data in completed:
            generated_text = gen_data.get('generated_text', '')
            prompt = gen_data.get('prompt', '')
            step = gen_data.get('step', 0)

            # Print to console with tqdm.write to not break progress bar
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"[Generation at Step {step}]")
            if prompt:
                tqdm.write(f"Prompt: {prompt}")
            display_text = generated_text[:500] + ('...' if len(generated_text) > 500 else '')
            tqdm.write(f"Generated: {display_text}")
            tqdm.write(f"{'='*60}\n")

            # Log to WandB using current step (not generation start step)
            # to avoid "step less than current step" warning
            if self._metrics_manager:
                tqdm.write(f"  [Gen] Logging to metrics manager (use_wandb={getattr(self._metrics_manager, '_use_wandb', '?')})")
                self._metrics_manager.log_generation(self._global_step, gen_data)
            else:
                tqdm.write(f"  [Gen] WARNING: No metrics manager available! Generation will not be logged to WandB")

        # Check if we should start a new generation
        if self._global_step <= 0:
            return

        # Skip generation if before minimum step threshold
        skip_until_step = generation_config.get('skip_until_step', 0)
        if self._global_step < skip_until_step:
            return

        if self._global_step % config.generate_every_n_steps != 0:
            return

        # Skip if previous generation still running
        if self._generation_manager.is_generation_pending():
            tqdm.write(f"  [Gen] Step {self._global_step}: skipped (previous still running)")
            return

        # Launch async CPU generation
        # Get special token IDs from model config for consistency
        base_model = model.module if hasattr(model, 'module') else model
        model_config = getattr(base_model, 'config', None)

        model_eos_token_id = getattr(model_config, 'eos_token_id', None) if model_config else None
        model_bos_token_id = getattr(model_config, 'bos_token_id', None) if model_config else None
        model_pad_token_id = getattr(model_config, 'pad_token_id', None) if model_config else None

        tqdm.write(f"  [Gen] Step {self._global_step}: launching async CPU generation...")
        self._generation_manager.generate_async(
            model,
            self._global_step,
            vocab_size,
            tokenizer=tokenizer,
            eos_token_id=model_eos_token_id or (getattr(tokenizer, 'eos_token_id', None) if tokenizer else None),
            bos_token_id=model_bos_token_id or (getattr(tokenizer, 'bos_token_id', None) if tokenizer else None),
            pad_token_id=model_pad_token_id or (getattr(tokenizer, 'pad_token_id', None) if tokenizer else None),
            max_length=generation_config.get('max_length', 128),
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            top_k=generation_config.get('top_k', 50),
            # COHERENCE FIX: Better defaults to prevent repetitive generation
            # Old defaults (1.0, 0) allowed degenerate repetition
            repetition_penalty=generation_config.get('repetition_penalty', 1.2),
            prompt=generation_config.get('prompt', None),
            no_repeat_ngram_size=generation_config.get('no_repeat_ngram_size', 3),
            skip_special_tokens=generation_config.get('skip_special_tokens', True),
        )

    def _maybe_measure_coherence(
        self,
        model: nn.Module,
        coherence_config: Dict[str, Any],
    ) -> None:
        """Measure coherence if conditions are met.

        Uses safe_inference_mode to guarantee model state restoration
        even if coherence measurement fails.
        """
        if self._generation_manager is None:
            return

        if not coherence_config.get('enabled', False):
            return

        eval_every = coherence_config.get('eval_every_n_steps', 500)
        if self._global_step <= 0 or self._global_step % eval_every != 0:
            return

        try:
            # Use safe_inference_mode for guaranteed state restoration
            from .state_guard import safe_inference_mode

            with safe_inference_mode(model, preserve_training_mode=True, preserve_grad_state=True):
                # Model is now safely in eval mode with no gradients
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

            # Model state automatically restored here, even on exception

        except Exception as e:
            # State is still safely restored thanks to context manager
            self.logger.warning(f"Coherence measurement failed: {e}")
            # No need for manual model.train() - context manager handles it

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

        # Only check memory utilization if monitoring is enabled.
        # memory_reserved() causes implicit sync which stalls the pipeline.
        if config.enable_memory_monitoring:
            mem_util = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory

            if mem_util > 0.92:
                torch.cuda.empty_cache()
                if hasattr(model, 'clear_caches'):
                    model.clear_caches()
                elif hasattr(model, 'module') and hasattr(model.module, 'clear_caches'):
                    model.module.clear_caches()

        # Periodic memory defragmentation every 100 steps
        # This prevents progressive OOM from memory fragmentation
        if self._global_step % 100 == 0 and self._global_step > 0:
            self._clear_cuda_memory()

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

            # Run empty_cache() in background thread to avoid blocking.
            # torch.cuda.empty_cache() is a blocking operation that can stall training.
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
