"""
Training loop manager for the Ava pipeline.

This module contains the core training loop implementation with advanced
optimizations for throughput and memory efficiency.

Key Features:
- Async batch prefetching: Uses dedicated CUDA stream to overlap I/O with compute
- Gradient accumulation: Simulates larger batch sizes across multiple micro-batches
- Mixed precision training: FP16/BF16 with automatic loss scaling
- CUDA graphs: Captures training step to eliminate kernel launch overhead (15-25% speedup)
- Generation testing: Async sample generation to monitor training quality
- Coherence measurement: Evaluates model output coherence during training
- OOM recovery: Automatic batch size reduction on out-of-memory errors

Training Flow (per epoch):
    ┌─────────────────────────────────────────────────────────────────────┐
    │ for batch in dataloader:                                            │
    │   ┌───────────────────────────────────────────────────────────────┐│
    │   │ FORWARD PASS (with autocast for AMP)                          ││
    │   │   outputs = model(input_ids, attention_mask, labels)          ││
    │   │   loss = outputs['loss'] / gradient_accumulation_steps        ││
    │   └───────────────────────────────────────────────────────────────┘│
    │   ┌───────────────────────────────────────────────────────────────┐│
    │   │ BACKWARD PASS                                                  ││
    │   │   scaler.scale(loss).backward()  # or loss.backward()         ││
    │   │   # Gradients accumulate across micro-batches                 ││
    │   └───────────────────────────────────────────────────────────────┘│
    │   if (batch_idx + 1) % gradient_accumulation_steps == 0:           │
    │     ┌─────────────────────────────────────────────────────────────┐│
    │     │ OPTIMIZER STEP (only at accumulation boundaries)            ││
    │     │   grad_norm = clip_grad_norm_(parameters, max_grad_norm)    ││
    │     │   scaler.step(optimizer)  # or optimizer.step()             ││
    │     │   scheduler.step()                                          ││
    │     │   optimizer.zero_grad()                                     ││
    │     └─────────────────────────────────────────────────────────────┘│
    │     global_step += 1                                               │
    │     if global_step % log_interval == 0: log_metrics()             │
    │     if global_step % generate_every_n_steps == 0: generate_async()│
    └─────────────────────────────────────────────────────────────────────┘

CUDA Graph Capture:
    After warmup_steps, captures the forward/backward pass as a CUDA graph.
    IMPORTANT: Optimizer state is saved before capture and restored after
    to prevent warmup runs from polluting momentum/variance buffers.

Loss Accumulator:
    Uses async accumulation to avoid cudaStreamSynchronize on every batch.
    _loss_accumulator: Running sum of losses (not synced to CPU until log step)
    _step_losses: List of synced losses for logging (populated at log intervals)
"""

import copy
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .context import ManagerInterface, TrainingContext
from .distributed import OverlappedGradientSync
from .overlapped_accumulation import OverlappedGradientAccumulator, create_overlapped_accumulator
from .pipeline_executor import PipelinedTrainingStep, create_pipeline_executor
from ..optimizations.prefetch import AsyncBatchPrefetcher
from ..optimizations.gradients import check_gradients, check_gradients_deferred
from contextlib import nullcontext

# Module-level cached nullcontext to avoid per-step object allocation (nanoGPT optimization)
_CACHED_NULLCONTEXT = nullcontext()


# ============================================================================
# Kahan Summation for Numerically Stable Loss Accumulation
# ============================================================================

class KahanAccumulator:
    """
    Kahan summation algorithm for numerically stable loss accumulation.

    Standard floating-point addition accumulates error over many operations.
    For long training runs with many micro-batches, this can result in
    noticeable drift in reported loss values.

    Kahan summation tracks the accumulated error and compensates for it,
    achieving near-full precision regardless of the number of additions.

    Example:
        >>> accumulator = KahanAccumulator()
        >>> for loss in losses:
        ...     accumulator.add(loss)
        >>> mean_loss = accumulator.get_mean()
    """

    def __init__(self):
        """Initialize empty accumulator."""
        self.sum: Optional[torch.Tensor] = None
        self.compensation: Optional[torch.Tensor] = None
        self.count: int = 0

    def add(self, value: torch.Tensor) -> None:
        """
        Add a value to the accumulator using Kahan summation.

        Args:
            value: Tensor value to add (should be scalar or will be summed)
        """
        if value.numel() != 1:
            value = value.mean()

        if self.sum is None:
            self.sum = value.clone().detach()
            self.compensation = torch.zeros_like(value)
        else:
            # Kahan summation algorithm
            y = value - self.compensation
            t = self.sum + y
            self.compensation = (t - self.sum) - y
            self.sum = t
        self.count += 1

    def get_sum(self) -> Optional[torch.Tensor]:
        """Get the accumulated sum."""
        return self.sum

    def get_mean(self) -> Optional[torch.Tensor]:
        """Get the mean of accumulated values."""
        if self.sum is None or self.count == 0:
            return None
        return self.sum / self.count

    def get_count(self) -> int:
        """Get the number of accumulated values."""
        return self.count

    def reset(self) -> None:
        """Reset the accumulator to empty state."""
        self.sum = None
        self.compensation = None
        self.count = 0


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


@runtime_checkable
class EpisodicMemoryProtocol(Protocol):
    """Protocol for episodic memory integration during training.

    Episodic memory enables experience replay for continual learning.
    Implementations should manage a buffer of past experiences and
    provide methods to augment training batches with replay samples.
    """

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        global_step: int,
        current_phase: str = "training",
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """
        Augment current batch with replay samples from memory.

        Args:
            batch: Current training batch
            batch_idx: Current batch index
            global_step: Current global training step
            current_phase: 'training' or 'accumulating'

        Returns:
            Tuple of (augmented_batch, replay_indices)
        """
        ...

    def get_sample_weights(
        self,
        batch_idx: int,
        outputs: Dict[str, Any],
        aux_info: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Get per-sample weights for importance-weighted loss."""
        ...

    def store_batch(
        self,
        batch: Dict[str, torch.Tensor],
        losses: torch.Tensor,
        aux_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store current batch samples in memory buffer."""
        ...

    def update_replay_priorities(
        self, indices: Any, losses: torch.Tensor
    ) -> None:
        """Update priorities for replayed samples."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get episodic memory metrics for logging."""
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
    # Canonical shape padding: Pad batches to canonical sizes for CUDA graph reuse.
    # Instead of capturing a new graph for each unique (batch_size, seq_len) pair,
    # pad to the nearest canonical shape and reuse graphs. Reduces capture overhead.
    use_canonical_shapes: bool = True  # ENABLED by default for CUDA graphs
    canonical_batch_sizes: tuple = (8, 16, 32, 64, 128)  # Batch size buckets
    canonical_seq_lengths: tuple = (128, 256, 512, 1024, 2048)  # Sequence length buckets

    # =========================================================================
    # Logging controls (from LoggingConfig for faster training)
    # =========================================================================
    logging_disabled: bool = False            # Master disable ALL console/tqdm output
    progress_bar_enabled: bool = True         # Enable tqdm progress bar
    step_logging_enabled: bool = True         # Enable per-step console output
    epoch_logging_enabled: bool = True        # Enable epoch summary logging
    generation_logging_enabled: bool = True   # Enable generation sample output

    # =========================================================================
    # Distributed barrier optimization (nanoGPT-style zero-barrier training)
    # =========================================================================
    # When True, skip loss all_reduce sync - each rank logs local loss (nanoGPT approach)
    # This saves 20-50ms per log step. Gradients are still synced by DDP automatically.
    minimize_distributed_barriers: bool = True
    # When False, skip all_reduce for loss values (default: False for speed)
    # Set True only if you need exact loss averaging across ranks for analysis
    sync_loss_across_ranks: bool = False

    # =========================================================================
    # Overlapped gradient accumulation (10-20% speedup for accum_steps > 1)
    # =========================================================================
    # When True, uses separate CUDA streams to overlap backward(N) with forward(N+1)
    # during gradient accumulation. Provides 10-20% speedup with minimal overhead.
    # Only effective when gradient_accumulation_steps > 1.
    use_overlapped_accumulation: bool = True

    # =========================================================================
    # Pipeline micro-batching (10-25% speedup for models with 16+ layers)
    # =========================================================================
    # When True, uses layer-level pipelining where different layers process
    # different micro-batches concurrently using separate CUDA streams.
    # Most effective for models with 16+ layers and gradient_accumulation_steps > 1.
    use_pipeline_microbatching: bool = False  # Experimental - disabled by default
    pipeline_overlap_factor: int = 2  # Number of micro-batches to overlap


class MetricsBatcher:
    """
    Batches GPU tensor→scalar conversions to minimize cudaStreamSynchronize overhead.

    Instead of calling .item() on each metric tensor separately (each causing a sync),
    this class collects all scalar tensors and extracts them in a single .tolist() call.

    Usage:
        batcher = MetricsBatcher()
        batcher.add('loss', loss_tensor)
        batcher.add('grad_norm', grad_norm_tensor)
        batcher.add('entropy', entropy_tensor)

        # ONE sync for all metrics
        values = batcher.flush()
        # values = {'loss': 0.5, 'grad_norm': 1.2, 'entropy': 0.8}
    """

    def __init__(self):
        self._pending: List[tuple] = []  # List of (key, tensor) pairs

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """
        Queue a scalar tensor for batched extraction.

        Args:
            key: Identifier for this metric
            tensor: A scalar tensor (0-dim or 1-element) on GPU
        """
        if tensor is None:
            return
        # Detach to avoid holding onto computation graph
        self._pending.append((key, tensor.detach()))

    def add_multi(self, prefix: str, tensors: Dict[str, torch.Tensor]) -> None:
        """
        Queue multiple tensors with a common prefix.

        Args:
            prefix: Prefix for all keys (e.g., 'grad' -> 'grad/norm', 'grad/max')
            tensors: Dict of name -> tensor pairs
        """
        for name, tensor in tensors.items():
            if tensor is not None:
                self.add(f"{prefix}/{name}", tensor.detach())

    def flush(self) -> Dict[str, float]:
        """
        Extract all queued tensors in a single GPU→CPU sync.

        Returns:
            Dictionary mapping keys to their scalar float values.
            Clears the internal queue after extraction.
        """
        if not self._pending:
            return {}

        keys = [k for k, _ in self._pending]
        # Ensure all tensors are scalar (0-dim) or squeeze to scalar
        tensors = []
        for _, t in self._pending:
            if t.dim() == 0:
                tensors.append(t.view(1))
            elif t.numel() == 1:
                tensors.append(t.view(1))
            else:
                # Multi-element tensor - take mean
                tensors.append(t.mean().view(1))

        # ONE sync point: concatenate and transfer to CPU
        try:
            stacked = torch.cat(tensors)
            values = stacked.tolist()
        except Exception:
            # Fallback: individual extraction if cat fails (mixed devices)
            values = [t[0].item() if t.numel() == 1 else t.mean().item()
                      for _, t in self._pending]

        self._pending.clear()
        return dict(zip(keys, values))

    def pending_count(self) -> int:
        """Return number of pending tensors."""
        return len(self._pending)

    def clear(self) -> None:
        """Clear pending tensors without extracting."""
        self._pending.clear()


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

        # Overlapped gradient sync for multi-GPU DDP (10-30% speedup)
        self._gradient_sync: Optional[OverlappedGradientSync] = None

        # Component references (set via set_components) - typed as protocols
        self._metrics_manager: Optional[MetricsLoggerProtocol] = None
        self._generation_manager: Optional[GenerationProviderProtocol] = None
        self._checkpoint_manager: Optional[CheckpointSaverProtocol] = None
        self._profiler: Optional['NsightProfiler'] = None
        self._diagnostics_manager: Optional['DiagnosticsManager'] = None
        self._episodic_memory_manager: Optional['EpisodicMemoryProtocol'] = None

        # Episodic memory state tracking
        self._last_replay_indices: Optional[Any] = None
        self._original_batch_size: int = 0

        # Current loop config for logging checks (set in train_epoch)
        self._loop_config: Optional[TrainingLoopConfig] = None

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
        # Multi-shape cache: (batch_size, seq_len) -> {graph, static_input, static_loss}
        self._cuda_graph_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # MEMORY FIX: Reduced from 4 to 2 to save ~1.3GB GPU memory
        # Each cached graph stores static input buffers (~650MB each)
        self._max_cached_graphs: int = 2
        # Legacy single-graph attributes (for backward compatibility in cleanup)
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cuda_graph_captured: bool = False
        self._graph_static_input: Optional[Dict[str, torch.Tensor]] = None
        self._graph_static_loss: Optional[torch.Tensor] = None
        self._graph_batch_size: Optional[int] = None
        self._graph_seq_len: Optional[int] = None

        # OPTIMIZATION: Batch all GPU→CPU metric extractions to one sync
        self._metrics_batcher = MetricsBatcher()

        # Deferred gradient stats (GPU tensors, extracted in batched flush)
        self._deferred_grad_tensors: Dict[str, torch.Tensor] = {}

        # OPTIMIZATION: Overlapped gradient accumulation (10-20% speedup)
        # Uses separate CUDA streams to overlap backward of batch N with forward of batch N+1
        self._overlapped_accumulator: Optional[OverlappedGradientAccumulator] = None

        # OPTIMIZATION: Pipeline micro-batching (10-25% speedup for models with 16+ layers)
        # Uses layer-level pipelining where different layers process different micro-batches
        self._pipeline_executor: Optional[PipelinedTrainingStep] = None

    def initialize(self) -> None:
        """Initialize the training loop manager."""
        # Enable TF32 and CuDNN optimizations (nanoGPT best practices)
        # TF32 gives ~3x faster matmul on Ampere+ with minimal precision loss
        if torch.cuda.is_available():
            # New PyTorch 2.9+ API for TF32 precision settings
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cudnn.benchmark = True
            self.logger.info("Enabled TF32 matmul, CuDNN TF32, and CuDNN benchmark")

        self._initialized = True
        self.logger.debug("TrainingLoopManager initialized")

    def _log_output(self, message: str, category: str = 'step') -> None:
        """Log output to tqdm.write if logging is enabled.

        Args:
            message: Message to output
            category: Logging category
                     - 'step': Per-step training output (config.step_logging_enabled)
                     - 'epoch': Epoch summary output (config.epoch_logging_enabled)
                     - 'generation': Generation samples (config.generation_logging_enabled)
                     - 'debug': Verbose debug output (only in verbose mode)
                     - 'event': Important events always shown (OOM, errors)
        """
        if self._loop_config is None:
            return  # No config yet, suppress output

        if self._loop_config.logging_disabled:
            return

        # Check category-specific flags
        if category == 'step' and not self._loop_config.step_logging_enabled:
            return
        if category == 'epoch' and not self._loop_config.epoch_logging_enabled:
            return
        if category == 'generation' and not self._loop_config.generation_logging_enabled:
            return
        if category == 'debug' and self._loop_config.log_mode != 'verbose':
            return

        # 'event' category always outputs
        tqdm.write(message)

    def cleanup(self) -> None:
        """
        Stop prefetcher and cleanup resources in coordinated order.

        Cleanup order for distributed training:
        1. Stop prefetcher first (prevents new batches)
        2. Clean up CUDA graphs (free GPU memory)
        3. Clean up gradient sync hooks (remove backward hooks)
        4. Final barrier before exit (single sync point - eliminates 20-200ms overhead)
        """
        from datetime import timedelta

        # 1. Stop prefetcher first
        if self.prefetcher is not None:
            self.prefetcher.stop()
            self.prefetcher = None

        # 2. Clean up CUDA graph resources (no pre-barrier needed - final barrier handles sync)
        self._cleanup_cuda_graph()

        # 3. Clean up gradient sync hooks
        if self._gradient_sync is not None:
            self._gradient_sync.unregister_hooks()
            self._gradient_sync = None

        # 4. Final barrier before exit (single barrier eliminates 20-200ms overhead)
        if self.context.world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                try:
                    self.logger.debug("Final synchronization after cleanup")
                    dist.barrier(timeout=timedelta(seconds=10))
                except Exception:
                    pass  # Best effort on exit

    def _cleanup_cuda_graph(self) -> None:
        """Properly free CUDA graph resources from GPU memory."""
        # Issue #7 fix: Synchronize before cleanup to ensure all operations complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # May fail if CUDA in bad state

        # Clean up all cached graphs
        for shape_key in list(self._cuda_graph_cache.keys()):
            self._cleanup_single_graph(shape_key)
        self._cuda_graph_cache.clear()

        # Free legacy static input tensors - explicitly pop to ensure GPU release
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

    def _cleanup_single_graph(self, shape_key: Tuple[int, int]) -> None:
        """Clean up a single cached graph by shape key."""
        if shape_key not in self._cuda_graph_cache:
            return
        graph_data = self._cuda_graph_cache.pop(shape_key)
        # Free static inputs
        if 'static_input' in graph_data and graph_data['static_input']:
            for key in list(graph_data['static_input'].keys()):
                tensor = graph_data['static_input'].pop(key)
                if tensor is not None:
                    del tensor
        # Free static loss
        if 'static_loss' in graph_data and graph_data['static_loss'] is not None:
            del graph_data['static_loss']
        # Reset graph
        if 'graph' in graph_data and graph_data['graph'] is not None:
            try:
                graph_data['graph'].reset()
            except Exception:
                pass

    def _get_canonical_shape(
        self,
        batch_size: int,
        seq_len: int,
        config: TrainingLoopConfig,
    ) -> Tuple[int, int]:
        """
        Get canonical shape for CUDA graph reuse.

        Rounds up batch_size and seq_len to the nearest canonical bucket.
        This allows CUDA graphs to be reused across similar-sized batches,
        reducing graph capture overhead (20-50ms per capture).

        Args:
            batch_size: Current batch size
            seq_len: Current sequence length
            config: Training config with canonical shape buckets

        Returns:
            Tuple of (canonical_batch_size, canonical_seq_len)
        """
        # Find smallest canonical batch size >= current
        canonical_bs = batch_size
        for bs in config.canonical_batch_sizes:
            if bs >= batch_size:
                canonical_bs = bs
                break
        else:
            # If larger than all canonical sizes, use largest + padding
            canonical_bs = config.canonical_batch_sizes[-1]

        # Find smallest canonical seq length >= current
        canonical_seq = seq_len
        for sl in config.canonical_seq_lengths:
            if sl >= seq_len:
                canonical_seq = sl
                break
        else:
            # If larger than all canonical sizes, use largest
            canonical_seq = config.canonical_seq_lengths[-1]

        return (canonical_bs, canonical_seq)

    def _pad_batch_to_canonical(
        self,
        batch: Dict[str, torch.Tensor],
        target_bs: int,
        target_seq: int,
        pad_token_id: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        Pad batch tensors to canonical shape for CUDA graph reuse.

        Args:
            batch: Batch dict with 'input_ids', 'attention_mask', 'labels'
            target_bs: Target batch size
            target_seq: Target sequence length
            pad_token_id: Token ID for padding

        Returns:
            Tuple of (padded_batch, original_bs, original_seq)
        """
        input_ids = batch['input_ids']
        current_bs, current_seq = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype

        # No padding needed if already at canonical size
        if current_bs == target_bs and current_seq == target_seq:
            return batch, current_bs, current_seq

        # Create padded tensors
        padded_batch = {}

        # Pad input_ids
        padded_input_ids = torch.full(
            (target_bs, target_seq), pad_token_id, dtype=dtype, device=device
        )
        padded_input_ids[:current_bs, :current_seq] = input_ids
        padded_batch['input_ids'] = padded_input_ids

        # Pad attention_mask (0 for padded positions)
        if 'attention_mask' in batch:
            padded_mask = torch.zeros(target_bs, target_seq, dtype=batch['attention_mask'].dtype, device=device)
            padded_mask[:current_bs, :current_seq] = batch['attention_mask']
            padded_batch['attention_mask'] = padded_mask

        # Pad labels (-100 for ignored positions)
        if 'labels' in batch:
            padded_labels = torch.full(
                (target_bs, target_seq), -100, dtype=batch['labels'].dtype, device=device
            )
            padded_labels[:current_bs, :current_seq] = batch['labels']
            padded_batch['labels'] = padded_labels

        # Copy any other tensors unchanged (e.g., position_ids)
        for key in batch:
            if key not in padded_batch:
                padded_batch[key] = batch[key]

        return padded_batch, current_bs, current_seq

    def setup_gradient_sync(self, model: nn.Module, is_deepspeed: bool = False) -> None:
        """
        Set up overlapped gradient synchronization for multi-GPU DDP training.

        This enables 10-30% speedup by overlapping gradient all-reduce with
        backward computation. Must be called after model is built but before
        training starts.

        Args:
            model: The model (before or after DDP wrapping)
            is_deepspeed: Whether DeepSpeed is being used (skip if True)
        """
        # Skip for single GPU or DeepSpeed (DeepSpeed has built-in overlap)
        if self.context.world_size <= 1 or is_deepspeed:
            return

        # Get the underlying model for DDP-wrapped models
        base_model = model.module if hasattr(model, 'module') else model

        self._gradient_sync = OverlappedGradientSync(
            base_model,
            bucket_size_mb=25.0,
            enabled=True
        )
        self._gradient_sync.register_hooks()
        self.logger.info(
            f"OverlappedGradientSync enabled for {self.context.world_size}-GPU training"
        )

    def verify_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        expected_step: int,
        optimizer_name: str = "optimizer"
    ) -> bool:
        """
        Verify optimizer state was properly restored after checkpoint resume.

        Checks that optimizer momentum/variance buffers exist and step counts
        are consistent. This helps detect corrupted or incomplete checkpoint
        restoration.

        Args:
            optimizer: Optimizer to verify
            expected_step: Expected training step count
            optimizer_name: Name for logging purposes

        Returns:
            True if optimizer state appears valid, False otherwise
        """
        if not optimizer.state:
            self.logger.warning(f"{optimizer_name} has no state - may not be properly initialized")
            return False

        issues_found = False

        for param_idx, (param, state) in enumerate(optimizer.state.items()):
            if not isinstance(state, dict):
                continue

            # Check Adam/AdamW buffers
            if 'exp_avg' not in state and 'momentum_buffer' not in state:
                # Not all parameters have optimizer state (frozen layers)
                continue

            # Check for NaN values in optimizer state
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        self.logger.error(
                            f"{optimizer_name} has NaN in state[{key}] for param {param_idx}"
                        )
                        issues_found = True

            # Verify step count if available
            if 'step' in state:
                step_val = state['step']
                if isinstance(step_val, torch.Tensor):
                    step_val = step_val.item()
                # Allow some discrepancy due to gradient accumulation
                if abs(step_val - expected_step) > 100:
                    self.logger.warning(
                        f"{optimizer_name} step mismatch: expected ~{expected_step}, "
                        f"got {step_val} for param {param_idx}"
                    )

        if issues_found:
            self.logger.error(f"{optimizer_name} state verification FAILED - training may diverge")
            return False

        self.logger.debug(f"{optimizer_name} state verified successfully")
        return True

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
                    gpu_batch[key] = value.to(device, non_blocking=True)
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
            from .deepspeed import is_deepspeed_engine
            if is_deepspeed_engine(model):
                self.logger.info("CUDA graphs disabled: incompatible with DeepSpeed")
                return False

            self.logger.info("Capturing CUDA graph for training step...")

            # Allocate static input buffers with the same shape as sample_batch
            self._graph_batch_size = sample_batch['input_ids'].size(0)
            self._graph_seq_len = sample_batch['input_ids'].size(1)

            self._graph_static_input = {
                'input_ids': torch.empty_like(sample_batch['input_ids']),
                'labels': torch.empty_like(sample_batch.get('labels', sample_batch['input_ids'])),
            }
            # attention_mask is optional - some datasets don't provide it
            if 'attention_mask' in sample_batch:
                self._graph_static_input['attention_mask'] = torch.empty_like(sample_batch['attention_mask'])

            # COHERENCE FIX: Include position_ids in CUDA graph for sequence packing
            # Without this, document-relative positions are lost, causing cross-document attention
            if 'position_ids' in sample_batch:
                self._graph_static_input['position_ids'] = torch.empty_like(sample_batch['position_ids'])

            # Copy sample data into static buffers
            for key in self._graph_static_input:
                if key == 'labels':
                    # labels defaults to input_ids for causal LM
                    self._graph_static_input[key].copy_(sample_batch.get('labels', sample_batch['input_ids']))
                else:
                    self._graph_static_input[key].copy_(sample_batch[key])

            # Static loss tensor for output
            self._graph_static_loss = torch.zeros(1, device=self.device)

            # ═══════════════════════════════════════════════════════════════
            # CUDA GRAPH WARMUP - OPTIMIZED (P1.2, P1.3)
            # ═══════════════════════════════════════════════════════════════
            # CUDA graph capture requires warmup to "prime" GPU memory allocations.
            #
            # OPTIMIZATION: Reduced from 3 iterations to 1, and removed
            # optimizer.step() from warmup. This eliminates:
            # - 10-50ms deepcopy of optimizer state (no longer needed)
            # - 2 extra forward-backward passes (only 1 needed for priming)
            #
            # We still need forward+backward to prime both execution paths,
            # but skip optimizer.step() since we're only warming up memory.
            # ═══════════════════════════════════════════════════════════════
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                outputs = model(
                    input_ids=self._graph_static_input['input_ids'],
                    attention_mask=self._graph_static_input.get('attention_mask'),
                    labels=self._graph_static_input['labels'],
                    position_ids=self._graph_static_input.get('position_ids'),
                )
                loss = outputs['loss'] / config.gradient_accumulation_steps
            if use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Skip optimizer.step() - only priming memory allocations
            optimizer.zero_grad(set_to_none=True)

            # Synchronize before capture
            torch.cuda.synchronize()

            # Capture the graph
            self._cuda_graph = torch.cuda.CUDAGraph()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self._cuda_graph):
                with torch.autocast(device_type='cuda', dtype=config.amp_dtype, enabled=config.use_amp):
                    # COHERENCE FIX: Use keyword args to ensure position_ids and 2D attention_mask
                    # are properly forwarded for sequence packing with document boundaries
                    outputs = model(
                        input_ids=self._graph_static_input['input_ids'],
                        attention_mask=self._graph_static_input.get('attention_mask'),
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
        keys_to_validate = ['input_ids', 'labels']
        # Only validate attention_mask if both batch and static input have it
        if 'attention_mask' in gpu_batch and 'attention_mask' in self._graph_static_input:
            keys_to_validate.append('attention_mask')
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
        if 'attention_mask' in self._graph_static_input and 'attention_mask' in gpu_batch:
            self._graph_static_input['attention_mask'].copy_(gpu_batch['attention_mask'])
        self._graph_static_input['labels'].copy_(gpu_batch.get('labels', gpu_batch['input_ids']))
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
        episodic_memory_manager: Optional['EpisodicMemoryProtocol'] = None,
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
            episodic_memory_manager: Any object implementing EpisodicMemoryProtocol
        """
        self._metrics_manager = metrics_manager
        self._generation_manager = generation_manager
        self._checkpoint_manager = checkpoint_manager
        self._profiler = profiler
        self._diagnostics_manager = diagnostics_manager
        self._episodic_memory_manager = episodic_memory_manager

        # Startup diagnostic logged to debug only
        if self._metrics_manager and self._generation_manager:
            use_wandb = getattr(self._metrics_manager, '_use_wandb', False)
            if not use_wandb:
                self.logger.debug("WandB disabled - generation samples won't appear in WandB")

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

        # Store config for logging checks
        self._loop_config = config

        model.train()

        # nanoGPT-style: Skip epoch barrier when minimize_distributed_barriers=True (default)
        # DDP handles gradient sync automatically - no explicit barriers needed in hot path
        # Check nested path: training.distributed.minimize_distributed_barriers
        distributed_cfg = getattr(config, 'distributed', None)
        if distributed_cfg is not None:
            minimize_barriers = getattr(distributed_cfg, 'minimize_distributed_barriers', True)
        else:
            # Fallback to flat path for backward compatibility
            minimize_barriers = getattr(config, 'minimize_distributed_barriers', True)
        if not minimize_barriers and self.context.world_size > 1 and epoch == 0:
            import torch.distributed as dist
            if dist.is_initialized():
                self.logger.debug("Synchronizing ranks before first epoch (minimize_barriers=False)")
                dist.barrier()

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
            self.prefetcher = AsyncBatchPrefetcher(train_loader, device, prefetch_count=10)
            batch_iterator = self.prefetcher
            # FIX: For IterableDataset, get_dynamic_total() returns sample count, not batches
            # Convert to training steps: samples / (batch_size * gradient_accumulation)
            samples_total = self.prefetcher.get_dynamic_total()
            batch_size = train_loader.batch_size
            grad_accum = config.gradient_accumulation_steps
            initial_total = samples_total // (batch_size * grad_accum) if batch_size and grad_accum else samples_total
        else:
            # Simple synchronous iteration (safer, no CUDA stream issues)
            logger.info("Using synchronous batch iteration (cuda_streams.enabled=false)")
            self.prefetcher = None
            batch_iterator = self._sync_batch_iterator(train_loader, device)
            # FIX: For IterableDataset, loader_len = len(dataset) (samples, not batches)
            # Convert to training steps: samples / (batch_size * gradient_accumulation)
            batch_size = train_loader.batch_size
            grad_accum = config.gradient_accumulation_steps
            initial_total = loader_len // (batch_size * grad_accum) if batch_size and grad_accum else loader_len

        # Progress bar (check config.progress_bar_enabled and config.logging_disabled)
        is_main = self.context.metadata.get('is_main_process', True)
        show_progress = is_main and config.progress_bar_enabled and not config.logging_disabled
        pbar = tqdm(total=initial_total, desc=f"Epoch {epoch + 1}", disable=not show_progress)

        # P1.4 OPTIMIZATION: Cache batch_controller reference before loop
        # Avoids per-batch getattr() call (saves 0.5-2ms per batch)
        _cached_batch_controller = getattr(self.context, 'batch_controller', None)

        # P2.0 OPTIMIZATION: Cache model attribute checks before loop
        # Avoids per-batch hasattr() calls (~100-200ns each, adds up over millions of batches)
        from .deepspeed import is_deepspeed_engine
        _is_deepspeed = is_deepspeed_engine(model)
        _base_model = model.module if hasattr(model, 'module') else model
        _has_hybrid_cache = hasattr(_base_model, '_hybrid_cache_manager') and _base_model._hybrid_cache_manager is not None
        _hybrid_cache_mgr = _base_model._hybrid_cache_manager if _has_hybrid_cache else None
        _has_ddp_grad_sync = hasattr(model, 'require_backward_grad_sync')

        # P2.1 OPTIMIZATION: Pre-compute profiler/diagnostics flags
        # Avoids per-batch hasattr() and None checks inside helper functions
        _profiler_has_step = self._profiler is not None and hasattr(self._profiler, 'step')
        _profiler_has_range = self._profiler is not None and hasattr(self._profiler, 'range')
        _has_diagnostics = self._diagnostics_manager is not None

        # P2.2 OPTIMIZATION: Initialize overlapped gradient accumulation (10-20% speedup)
        # Only beneficial when gradient_accumulation_steps > 1 and NOT using CUDA graphs
        # (CUDA graphs and overlapped accumulation are mutually exclusive)
        _use_overlapped = False
        if (getattr(config, 'use_overlapped_accumulation', True) and
            config.gradient_accumulation_steps > 1 and
            not config.use_cuda_graphs and
            not _is_deepspeed):
            self._overlapped_accumulator = create_overlapped_accumulator(
                config=config,
                is_deepspeed=_is_deepspeed,
            )
            if self._overlapped_accumulator is not None:
                _use_overlapped = True
                self.logger.info(
                    f"Overlapped accumulation enabled (accum_steps={config.gradient_accumulation_steps})"
                )
        else:
            self._overlapped_accumulator = None

        # Micro-batch collection for overlapped accumulation
        _micro_batches: List[Dict[str, torch.Tensor]] = [] if _use_overlapped else None

        # P3 OPTIMIZATION: Initialize pipeline micro-batching (10-25% speedup for 16+ layer models)
        # Only beneficial when gradient_accumulation_steps > 1 and NOT using overlapped accumulation
        # (Pipeline executor and overlapped accumulator are mutually exclusive)
        _use_pipeline = False
        if (getattr(config, 'use_pipeline_microbatching', False) and
            config.gradient_accumulation_steps > 1 and
            not _use_overlapped and
            not config.use_cuda_graphs and
            not _is_deepspeed):
            self._pipeline_executor = create_pipeline_executor(
                model=model,
                config=config,
            )
            if self._pipeline_executor is not None:
                _use_pipeline = True
                self.logger.info(
                    f"Pipeline micro-batching enabled (overlap_factor={config.pipeline_overlap_factor})"
                )
        else:
            self._pipeline_executor = None

        # Helper for profiler step context - uses pre-computed flag for fast path
        def get_step_context(step):
            return self._profiler.step(step) if _profiler_has_step else _CACHED_NULLCONTEXT

        try:
            for batch_idx, gpu_batch in batch_iterator:
                try:
                    # CRITICAL: Truncate batch ONLY during OOM recovery
                    # This handles the case where we need to reduce batch size after an OOM
                    # We do NOT truncate in 'stable' mode (after calibration) - that defeats the purpose
                    batch_controller = _cached_batch_controller
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
                            self.logger.debug(f"Batch truncated {current_bs} → {max_safe}")

                    # Determine if this is a logging step (only sync loss here)
                    is_accum_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
                    # Use global_step for log interval, not batch_idx.
                    # This ensures we sync exactly at log_interval steps.
                    next_global_step = self._global_step + 1 if is_accum_step else self._global_step
                    is_log_step = is_accum_step and (next_global_step % config.log_interval == 0)

                    # Wrap step in profiler context for NVTX annotations
                    with get_step_context(self._global_step):
                        # CUDA Graphs: Multi-shape cache - capture after warmup, replay from cache
                        # OPTIMIZATION: Use canonical shapes for better graph reuse (10-15% speedup)
                        use_graph = False
                        graph_batch = gpu_batch  # May be padded below
                        if config.use_cuda_graphs and batch_idx >= config.cuda_graph_warmup_steps:
                            current_bs = gpu_batch['input_ids'].size(0)
                            current_seq = gpu_batch['input_ids'].size(1)

                            # Use canonical shapes if enabled (reduces graph captures)
                            if config.use_canonical_shapes:
                                canonical_bs, canonical_seq = self._get_canonical_shape(
                                    current_bs, current_seq, config
                                )
                                shape_key = (canonical_bs, canonical_seq)

                                # Pad batch to canonical shape if needed
                                if current_bs != canonical_bs or current_seq != canonical_seq:
                                    graph_batch, _, _ = self._pad_batch_to_canonical(
                                        gpu_batch, canonical_bs, canonical_seq,
                                        pad_token_id=self.context.config.get('model', {}).get('pad_token_id', 0)
                                    )
                            else:
                                shape_key = (current_bs, current_seq)

                            # Check cache for this shape
                            if shape_key in self._cuda_graph_cache:
                                # Cache hit - use cached graph
                                graph_data = self._cuda_graph_cache[shape_key]
                                self._cuda_graph = graph_data['graph']
                                self._graph_static_input = graph_data['static_input']
                                self._graph_static_loss = graph_data['static_loss']
                                self._graph_batch_size = shape_key[0]
                                self._graph_seq_len = shape_key[1]
                                self._cuda_graph_captured = True
                                use_graph = True
                            else:
                                # Cache miss - try to capture new graph
                                # Evict oldest if cache full
                                if len(self._cuda_graph_cache) >= self._max_cached_graphs:
                                    oldest_key = next(iter(self._cuda_graph_cache))
                                    self._cleanup_single_graph(oldest_key)

                                # Try to capture with (potentially padded) batch
                                success = self._capture_cuda_graph(
                                    model, optimizer, graph_batch, config, use_scaler
                                )
                                if success:
                                    # Store in cache
                                    self._cuda_graph_cache[shape_key] = {
                                        'graph': self._cuda_graph,
                                        'static_input': self._graph_static_input,
                                        'static_loss': self._graph_static_loss,
                                    }
                                    self.logger.info(f"CUDA Graph cached for canonical shape [{shape_key[0]}, {shape_key[1]}]")
                                    use_graph = True

                        if use_graph:
                            # Replay CUDA graph (15-25% faster)
                            # Use graph_batch (may be padded to canonical shape)
                            loss = self._replay_cuda_graph(graph_batch)
                            # Accumulate loss (same as _process_batch)
                            # FIX: Graph loss is pre-divided by gradient_accumulation_steps,
                            # restore original scale to match _process_batch behavior
                            original_loss = loss.detach() * config.gradient_accumulation_steps
                            if self._loss_accumulator is None:
                                self._loss_accumulator = original_loss.clone()
                            else:
                                self._loss_accumulator.add_(original_loss)
                            self._loss_count += 1
                            # Handle optimizer step for graph replay
                            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                                if use_scaler:
                                    self.scaler.step(optimizer)
                                    self.scaler.update()
                                else:
                                    optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                                scheduler.step()
                        elif _use_overlapped:
                            # P2.3 OPTIMIZATION: Overlapped gradient accumulation (10-20% speedup)
                            # Collect micro-batches and process them with overlapped backward/forward
                            _micro_batches.append(gpu_batch)
                            if len(_micro_batches) >= config.gradient_accumulation_steps:
                                # Process all micro-batches with overlapped backward/forward
                                total_loss, num_processed = self._overlapped_accumulator.accumulate(
                                    model=model,
                                    batch_iterator=iter(_micro_batches),
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    scaler=self.scaler,
                                )
                                # Accumulate loss for logging (total_loss is already the sum)
                                if self._loss_accumulator is None:
                                    self._loss_accumulator = torch.tensor(
                                        total_loss, device=self.device, dtype=torch.float32
                                    )
                                else:
                                    self._loss_accumulator.add_(total_loss)
                                self._loss_count += num_processed
                                _micro_batches.clear()
                        else:
                            # Process batch normally - NO SYNC unless is_log_step
                            _ = self._process_batch(
                                batch_idx, gpu_batch, model, optimizer, scheduler,
                                config, use_scaler, should_sync=is_log_step,
                                # P2.0: Pass cached values to avoid per-batch checks
                                is_deepspeed=_is_deepspeed,
                                base_model=_base_model,
                                has_hybrid_cache=_has_hybrid_cache,
                                hybrid_cache_mgr=_hybrid_cache_mgr,
                                has_ddp_grad_sync=_has_ddp_grad_sync,
                                profiler_has_range=_profiler_has_range,
                                has_diagnostics=_has_diagnostics,
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

                        # OPTIMIZATION: Queue loss tensor for batched extraction (no sync yet)
                        loss_tensor = self._get_accumulated_loss_tensor(reset=True)
                        if loss_tensor is not None:
                            self._metrics_batcher.add('loss', loss_tensor)

                        # Queue all deferred gradient tensors for batched extraction
                        for key, tensor in self._deferred_grad_tensors.items():
                            self._metrics_batcher.add(key, tensor)
                        self._deferred_grad_tensors.clear()

                        # Queue routing metrics for batched extraction (if MoE model)
                        # OPTIMIZATION: Only log routing metrics every 500 steps to reduce sync overhead
                        # Routing metrics queue 50+ tensors per log step - significant overhead
                        routing_tensor_keys = []  # Track keys for post-processing
                        should_log_routing = next_global_step % 500 == 0
                        if should_log_routing and self._last_aux_info is not None and len(self._last_aux_info) > 0:
                            layer_idx = 0
                            for layer_aux in self._last_aux_info:
                                # Queue expert utilization tensors
                                for key, value in layer_aux.items():
                                    if key.startswith('expert_') and key.endswith('_utilization'):
                                        if hasattr(value, 'item'):
                                            tensor_val = value.mean() if value.numel() > 1 else value
                                            batch_key = f'routing/util_{key}_l{layer_idx}'
                                            self._metrics_batcher.add(batch_key, tensor_val)
                                            routing_tensor_keys.append(('util', key, layer_idx))
                                # Queue routing metrics (entropy, balance, confidence)
                                for metric_name in ['routing_entropy', 'balance_score', 'router_confidence']:
                                    if metric_name in layer_aux and layer_aux[metric_name] is not None:
                                        val = layer_aux[metric_name]
                                        if hasattr(val, 'item'):
                                            tensor_val = val.mean() if val.numel() > 1 else val
                                            batch_key = f'routing/{metric_name}_l{layer_idx}'
                                            self._metrics_batcher.add(batch_key, tensor_val)
                                            routing_tensor_keys.append((metric_name, layer_idx))
                                layer_idx += 1
                            # Store for post-processing in _log_step_metrics
                            self._routing_tensor_keys = routing_tensor_keys
                            self._num_routing_layers = layer_idx

                        # SINGLE SYNC: Extract all metrics at once
                        batched_values = self._metrics_batcher.flush()
                        # Store for _log_step_metrics to access
                        self._batched_values = batched_values
                        last_synced_loss = batched_values.get('loss', 0.0)

                        # Process gradient health warnings (now on CPU, no additional sync)
                        if batched_values.get('vanishing', 0.0) > 0.5:
                            grad_norm_val = batched_values.get('grad_norm', 0.0)
                            logger.warning(f"Step {self._global_step}: VANISHING gradients (norm={grad_norm_val:.2e}) - model may not be learning")
                        elif batched_values.get('exploding', 0.0) > 0.5:
                            grad_norm_val = batched_values.get('grad_norm', 0.0)
                            logger.warning(f"Step {self._global_step}: EXPLODING gradients (norm={grad_norm_val:.2e}) - consider reducing LR")

                        # Process inf detection warning
                        if batched_values.get('found_inf', 0.0) > 0.5:
                            if not hasattr(self, '_skipped_steps'):
                                self._skipped_steps = 0
                            self._skipped_steps += 1
                            logger.warning(f"Step {self._global_step}: Optimizer step SKIPPED (gradient overflow, total skipped: {self._skipped_steps})")

                        # Log gradient stats if available
                        if self._metrics_manager and 'grad_norm' in batched_values:
                            grad_stats = {
                                'grad_norm': batched_values.get('grad_norm', 0.0),
                                'total_norm': batched_values.get('grad_total_norm', 0.0),
                                'max_grad': batched_values.get('grad_max', 0.0),
                                'min_grad': batched_values.get('grad_min', 0.0),
                                'num_params': int(batched_values.get('grad_num_params', 0)),
                                'num_grads': int(batched_values.get('grad_num_grads', 0)),
                                'num_zero_grads': int(batched_values.get('grad_num_zero', 0)),
                            }
                            self._metrics_manager.log_gradients(self._global_step, grad_stats)

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
                        # FIX: Convert sample count to training steps for IterableDataset
                        if self.prefetcher:
                            samples = self.prefetcher.get_dynamic_total()
                            bs = train_loader.batch_size
                            ga = config.gradient_accumulation_steps
                            new_total = samples // (bs * ga) if bs and ga else samples
                        else:
                            new_total = pbar.total
                        if pbar.total != new_total:
                            pbar.total = new_total

                        current_bs = gpu_batch['input_ids'].shape[0]
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
                            from .distributed import DistributedStateManager
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

                    # OOM recovery is important - log as event
                    self._log_output(
                        f"[OOM] Batch {current_bs} → {new_batch_size} (attempt {self._oom_recovery_manager.recovery_count})",
                        category='event'
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
        # P2.0 OPTIMIZATION: Pre-computed values from train_epoch to avoid per-batch checks
        is_deepspeed: bool = False,
        base_model: Optional[nn.Module] = None,
        has_hybrid_cache: bool = False,
        hybrid_cache_mgr: Optional[Any] = None,
        has_ddp_grad_sync: bool = False,
        profiler_has_range: bool = False,
        has_diagnostics: bool = False,
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
            is_deepspeed: Whether model is a DeepSpeed engine (cached)
            base_model: Unwrapped model (cached, model.module or model)
            has_hybrid_cache: Whether base_model has hybrid cache manager (cached)
            hybrid_cache_mgr: The hybrid cache manager if present (cached)
            has_ddp_grad_sync: Whether model has require_backward_grad_sync attr (cached)
            profiler_has_range: Whether profiler has range method (cached)
            has_diagnostics: Whether diagnostics manager is available (cached)

        Returns:
            Loss tensor on GPU (no sync) or float if should_sync=True
        """
        # Store original batch for episodic memory (before augmentation)
        original_gpu_batch = gpu_batch
        self._original_batch_size = gpu_batch['input_ids'].size(0)
        self._last_replay_indices = None

        # Episodic memory: augment batch with replay samples
        if self._episodic_memory_manager is not None:
            gpu_batch, self._last_replay_indices = self._episodic_memory_manager.augment_batch(
                batch=gpu_batch,
                batch_idx=batch_idx,
                global_step=self._global_step,
                current_phase="training",
            )

        input_ids = gpu_batch['input_ids']
        labels = gpu_batch.get('labels', input_ids)  # Default to input_ids for causal LM
        # attention_mask is optional - some datasets/collators don't provide it
        attention_mask = gpu_batch.get('attention_mask', None)
        # Get position_ids if provided (from sequence packing with per-document positions)
        position_ids = gpu_batch.get('position_ids', None)

        # Notify activation cache of batch start (for hybrid caching)
        # P2.0 OPTIMIZATION: Use cached values instead of per-batch hasattr() checks
        if has_hybrid_cache and hybrid_cache_mgr and hybrid_cache_mgr.activation_cache:
            hybrid_cache_mgr.activation_cache.on_batch_start(gpu_batch)

        # P2.0 OPTIMIZATION: DeepSpeed detection moved to train_epoch (is_deepspeed param)
        # P2.1 OPTIMIZATION: Helper functions use pre-computed flags instead of per-call checks
        def get_range_context(name):
            return self._profiler.range(name) if profiler_has_range else _CACHED_NULLCONTEXT

        def get_timing_context(phase):
            return self._diagnostics_manager.time_phase(phase) if has_diagnostics else _CACHED_NULLCONTEXT

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
        # nanoGPT-style DDP optimization: Skip gradient sync on non-final micro-steps
        # This saves 50-200ms per micro-step in multi-GPU training by deferring
        # the all-reduce until the final micro-step of each accumulation cycle.
        # The official way is model.no_sync() context manager, but directly setting
        # require_backward_grad_sync is cleaner (same effect, no code duplication).
        is_final_microstep = (batch_idx + 1) % config.gradient_accumulation_steps == 0
        # P2.0 OPTIMIZATION: Use cached has_ddp_grad_sync instead of per-batch hasattr()
        if has_ddp_grad_sync:
            # DDP model: only sync gradients on final micro-step
            model.require_backward_grad_sync = is_final_microstep

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
                # Clone to avoid holding computation graph reference
                self._loss_accumulator = original_loss.clone()
            else:
                # OPTIMIZATION: In-place addition avoids tensor allocation overhead
                self._loss_accumulator.add_(original_loss)
            self._loss_count += 1

            # Store aux_info for logging loss components
            if 'aux_info' in outputs:
                self._last_aux_info = outputs['aux_info']

            # Episodic memory: store batch and update priorities
            if self._episodic_memory_manager is not None:
                # Store original batch samples (not replay samples)
                self._episodic_memory_manager.store_batch(
                    batch=original_gpu_batch,
                    losses=original_loss,
                    aux_info=outputs.get('aux_info'),
                )

                # Update priorities for replayed samples based on new loss
                if self._last_replay_indices is not None and len(self._last_replay_indices) > 0:
                    # Extract replay portion of loss (after original batch)
                    replay_start = self._original_batch_size
                    total_size = original_loss.numel() if original_loss.numel() > 1 else gpu_batch['input_ids'].size(0)
                    if replay_start < total_size:
                        # For scalar loss, expand and slice
                        if original_loss.numel() == 1:
                            replay_loss = original_loss.expand(len(self._last_replay_indices))
                        else:
                            replay_loss = original_loss[replay_start:]
                        self._episodic_memory_manager.update_replay_priorities(
                            self._last_replay_indices, replay_loss
                        )

        # Gradient accumulation step with detailed profiling
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            with get_range_context("optimizer_step"), get_timing_context("optimizer_step"):
                if is_deepspeed:
                    # DeepSpeed path - engine.step() handles clipping, optimizer, and scheduler automatically
                    with get_range_context("optimizer_step/deepspeed_step"):
                        model.step()

                    # Queue gradient stats for batched extraction (DeepSpeed handles clipping internally)
                    if self._metrics_manager and should_sync:
                        with get_range_context("optimizer_step/grad_stats"):
                            # P2.0: Use cached base_model instead of per-batch hasattr() check
                            grad_tensors = check_gradients_deferred(base_model)
                            self._deferred_grad_tensors.update(grad_tensors)
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
                    # OPTIMIZATION: Use tensor comparisons on GPU to avoid .item() sync every step
                    # Queue gradient norm for batched extraction at log intervals
                    if hasattr(grad_norm, 'item'):
                        # GPU tensor comparisons (no sync) - flag issues for later warning
                        self._deferred_grad_tensors['grad_norm'] = grad_norm.detach()
                        if grad_norm < 1e-7:
                            self._deferred_grad_tensors['vanishing'] = torch.tensor(1.0, device=grad_norm.device)
                        elif grad_norm > 1e4:
                            self._deferred_grad_tensors['exploding'] = torch.tensor(1.0, device=grad_norm.device)

                    # Queue gradient stats for batched extraction at log intervals
                    if self._metrics_manager and should_sync:
                        with get_range_context("optimizer_step/grad_stats"):
                            # Get GPU tensors without extracting to CPU yet
                            grad_tensors = check_gradients_deferred(model)
                            self._deferred_grad_tensors.update(grad_tensors)

                    # OPTIMIZATION: For multi-GPU training, use overlapped gradient sync if available
                    # This overlaps all-reduce with backward computation for 10-30% speedup
                    if self._gradient_sync is not None:
                        with get_range_context("optimizer_step/gradient_sync_wait"):
                            self._gradient_sync.wait_for_all_reduce()
                    # NOTE: Removed fallback torch.cuda.synchronize() - DDP automatically
                    # synchronizes gradients during backward pass via hooks. The explicit sync
                    # was unnecessary and added 5-20ms latency per accumulation step.

                    if use_scaler:
                        with get_range_context("optimizer_step/scaler_step"):
                            self.scaler.step(optimizer)
                        with get_range_context("optimizer_step/scaler_update"):
                            self.scaler.update()
                        # TRAINING HEALTH: Detect when scaler skips optimizer step due to NaN/Inf
                        # OPTIMIZATION: Queue for batched extraction at log intervals
                        if should_sync and hasattr(self.scaler, '_found_inf_per_device'):
                            inf_tensors = list(self.scaler._found_inf_per_device.values())
                            if inf_tensors:
                                # Queue inf check tensor for batched extraction (no sync here)
                                self._deferred_grad_tensors['found_inf'] = torch.stack(inf_tensors).any().float()
                    else:
                        with get_range_context("optimizer_step/optimizer_update"):
                            optimizer.step()

                    with get_range_context("optimizer_step/zero_grad"):
                        optimizer.zero_grad(set_to_none=True)
                    with get_range_context("optimizer_step/scheduler_step"):
                        scheduler.step()

                    # Clear gradient sync handles after optimizer step
                    if self._gradient_sync is not None:
                        self._gradient_sync.clear_handles()

        # Notify activation cache of batch end (clear per-batch cache)
        # P2.0 OPTIMIZATION: Use cached values instead of per-batch hasattr() checks
        if has_hybrid_cache and hybrid_cache_mgr and hybrid_cache_mgr.activation_cache:
            hybrid_cache_mgr.activation_cache.on_batch_end()

        # Return detached loss (stays on GPU, no sync)
        return loss.detach()

    def _get_accumulated_loss(self, reset: bool = True, sync_distributed: Optional[bool] = None) -> float:
        """
        Get the accumulated loss value with optional distributed synchronization.

        This is the ONLY place where cudaStreamSynchronize happens for loss.
        Call this at log intervals, not every batch.

        OPTIMIZATION: By default (sync_distributed=None), uses config.sync_loss_across_ranks
        which defaults to False (nanoGPT-style). This saves 20-50ms per log step by skipping
        the all_reduce. Gradients are still synced by DDP automatically - only loss display
        shows local values which are similar enough for monitoring.

        Args:
            reset: If True, reset accumulator after reading
            sync_distributed: If True, synchronize across ranks in distributed training.
                              If None (default), uses config.sync_loss_across_ranks setting.
                              Default config is False for nanoGPT-style zero-barrier training.

        Returns:
            Average accumulated loss as a Python float (local or synchronized based on config)
        """
        if self._loss_accumulator is None or self._loss_count == 0:
            return 0.0

        # Compute average on GPU (stays on GPU for now)
        avg_loss_tensor = self._loss_accumulator / self._loss_count

        # Determine whether to sync based on config or explicit parameter
        # Default: False (nanoGPT-style) - skip barrier for 20-50ms savings per log step
        should_sync = sync_distributed
        if should_sync is None:
            # Check nested path: training.distributed.sync_loss_across_ranks
            distributed_cfg = getattr(self._loop_config, 'distributed', None)
            if distributed_cfg is not None:
                should_sync = getattr(distributed_cfg, 'sync_loss_across_ranks', False)
            else:
                # Fallback to flat path for backward compatibility
                should_sync = getattr(self._loop_config, 'sync_loss_across_ranks', False)

        # Only sync if explicitly requested - nanoGPT approach logs local loss per rank
        if should_sync and self.context.world_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    # Average loss across all ranks
                    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            except Exception as e:
                # Improved error handling for distributed sync failures
                import traceback
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to synchronize loss across ranks: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                # Re-raise NCCL errors - they indicate serious distributed issues
                if "NCCL" in str(e).upper():
                    raise RuntimeError(
                        f"NCCL error during loss synchronization. "
                        f"This indicates a distributed training issue. Original: {e}"
                    ) from e
                # For other errors, log warning but continue (backward compatibility)
                logger.warning("Continuing with unsynchronized loss value - metrics may be inaccurate")

        # Now safe to move to CPU - all ranks have the same value
        avg_loss = avg_loss_tensor.item()

        if reset:
            self._loss_accumulator = None
            self._loss_count = 0

        return avg_loss

    def _get_accumulated_loss_tensor(
        self, reset: bool = True, sync_distributed: Optional[bool] = None
    ) -> Optional[torch.Tensor]:
        """
        Get the accumulated loss as a GPU tensor (no CPU sync).

        This method is designed for batched metric extraction. It returns the
        loss tensor on GPU so it can be stacked with other metrics and extracted
        in a single .tolist() call.

        OPTIMIZATION: By default, uses config.sync_loss_across_ranks which defaults
        to False (nanoGPT-style zero-barrier training).

        Args:
            reset: If True, reset accumulator after reading
            sync_distributed: If True, synchronize across ranks. If None, uses config.

        Returns:
            Average accumulated loss as a GPU tensor, or None if no losses accumulated.
        """
        if self._loss_accumulator is None or self._loss_count == 0:
            return None

        # Compute average on GPU (stays on GPU)
        avg_loss_tensor = self._loss_accumulator / self._loss_count

        # Determine whether to sync based on config or explicit parameter
        should_sync = sync_distributed
        if should_sync is None:
            # Check nested path: training.distributed.sync_loss_across_ranks
            distributed_cfg = getattr(self._loop_config, 'distributed', None)
            if distributed_cfg is not None:
                should_sync = getattr(distributed_cfg, 'sync_loss_across_ranks', False)
            else:
                # Fallback to flat path for backward compatibility
                should_sync = getattr(self._loop_config, 'sync_loss_across_ranks', False)

        # Only sync if explicitly requested - nanoGPT approach logs local loss
        if should_sync and self.context.world_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            except Exception as e:
                # Improved error handling for distributed sync failures
                import traceback
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to synchronize loss tensor across ranks: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                # Re-raise NCCL errors - they indicate serious distributed issues
                if "NCCL" in str(e).upper():
                    raise RuntimeError(
                        f"NCCL error during loss synchronization. "
                        f"This indicates a distributed training issue. Original: {e}"
                    ) from e
                logger.warning("Continuing with unsynchronized loss tensor")

        if reset:
            self._loss_accumulator = None
            self._loss_count = 0

        return avg_loss_tensor

    def _reset_loss_accumulator(self) -> None:
        """Reset the loss accumulator without syncing.

        OPTIMIZATION: Reuses existing tensor via zero_() instead of allocating new one.
        This avoids per-epoch tensor allocation overhead.
        """
        if self._loss_accumulator is not None:
            self._loss_accumulator.zero_()
        # Will be initialized on first use if None (lazy allocation)
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
        # GPU SYNC FIX: loss_components now contains tensors, extract at log time
        extra_metrics = {}
        if self._last_aux_info is not None and len(self._last_aux_info) > 0:
            if 'loss_components' in self._last_aux_info[0]:
                components = self._last_aux_info[0]['loss_components']
                ce_loss = components['cross_entropy_loss']
                aux_loss = components['aux_loss']
                # Convert tensors to floats at log time (single sync batched with other metrics)
                ce_val = ce_loss.item() if hasattr(ce_loss, 'item') else float(ce_loss)
                aux_val = aux_loss.item() if hasattr(aux_loss, 'item') else float(aux_loss)
                extra_metrics.update({
                    'train/cross_entropy_loss': ce_val,
                    'train/aux_loss': aux_val,
                    'train/aux_loss_ratio': aux_val / max(ce_val, 1e-8)
                })

            # OPTIMIZATION: Use pre-extracted batched values (no GPU sync here)
            # Routing metrics were queued and extracted in the main loop via _metrics_batcher
            batched_values = getattr(self, '_batched_values', {})
            routing_keys = getattr(self, '_routing_tensor_keys', [])
            num_layers_with_routing = getattr(self, '_num_routing_layers', 0)

            router_types_seen = set()
            total_expert_utilization = {}

            # Process routing metrics from pre-extracted batched values
            for layer_aux in self._last_aux_info:
                if 'router_type' in layer_aux:
                    router_types_seen.add(layer_aux['router_type'])

            # Extract utilization and routing metrics from batched values
            for key_info in routing_keys:
                if key_info[0] == 'util':
                    _, util_key, layer_idx = key_info
                    batch_key = f'routing/util_{util_key}_l{layer_idx}'
                    if batch_key in batched_values:
                        if util_key not in total_expert_utilization:
                            total_expert_utilization[util_key] = 0.0
                        total_expert_utilization[util_key] += batched_values[batch_key]
                elif len(key_info) == 2:
                    metric_name, layer_idx = key_info
                    batch_key = f'routing/{metric_name}_l{layer_idx}'
                    if batch_key in batched_values:
                        display_key = f'routing/{metric_name.replace("routing_", "")}_layer_{layer_idx}'
                        extra_metrics[display_key] = batched_values[batch_key]

            # Also check for direct scalar values in aux_info
            layer_idx = 0
            for layer_aux in self._last_aux_info:
                for metric_name in ['routing_entropy', 'balance_score', 'router_confidence']:
                    if metric_name in layer_aux and layer_aux[metric_name] is not None:
                        val = layer_aux[metric_name]
                        if isinstance(val, (int, float)):
                            metric_key = f'routing/{metric_name.replace("routing_", "")}_layer_{layer_idx}'
                            extra_metrics[metric_key] = val
                layer_idx += 1

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

        # Episodic memory metrics
        if self._episodic_memory_manager is not None:
            episodic_metrics = self._episodic_memory_manager.get_metrics()
            if episodic_metrics:
                extra_metrics.update(episodic_metrics)

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
            # Verbose step logging (controlled by log_mode)
            health_info = ""
            if hasattr(self, '_skipped_steps') and self._skipped_steps > 0:
                health_info = f" | Skipped: {self._skipped_steps}"

            msg = (
                f"Step {self._global_step}/{total} | "
                f"BS: {current_bs} | Loss: {loss_value:.4f} | "
                f"LR: {lr:.2e}{mem_info}{health_info}"
            )
            self._log_output(f"[Epoch {epoch + 1}] {msg}", category='step')

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

        # Check for and display completed generations
        completed = self._generation_manager.process_completed_generations()
        for gen_data in completed:
            generated_text = gen_data.get('generated_text', '')
            prompt = gen_data.get('prompt', '')
            step = gen_data.get('step', 0)

            # Compact generation output
            display_text = generated_text[:200] + ('...' if len(generated_text) > 200 else '')
            self._log_output(f"[Gen Step {step}] {display_text}", category='generation')

            # Log to WandB silently
            if self._metrics_manager:
                self._metrics_manager.log_generation(gen_data.get('step', self._global_step), gen_data)

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
            return  # Silent skip - don't spam logs

        # Launch async CPU generation
        base_model = model.module if hasattr(model, 'module') else model
        model_config = getattr(base_model, 'config', None)

        model_eos_token_id = getattr(model_config, 'eos_token_id', None) if model_config else None
        model_bos_token_id = getattr(model_config, 'bos_token_id', None) if model_config else None
        model_pad_token_id = getattr(model_config, 'pad_token_id', None) if model_config else None
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

        # Only check memory utilization at intervals to avoid sync overhead.
        # memory_reserved() causes implicit cudaStreamSynchronize which stalls the pipeline.
        # Check every 5000 steps to minimize overhead (was 2000, reduced for performance).
        if config.enable_memory_monitoring and self._global_step % 5000 == 0:
            mem_util = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(self.device).total_memory

            if mem_util > 0.92:
                torch.cuda.empty_cache()
                if hasattr(model, 'clear_caches'):
                    model.clear_caches()
                elif hasattr(model, 'module') and hasattr(model.module, 'clear_caches'):
                    model.module.clear_caches()

        # Periodic memory defragmentation every 2000 steps (was 1000, reduced for performance)
        # Only clear if significant fragmentation detected (>30% reserved but unused)
        # empty_cache() costs 20-50ms - avoid calling unnecessarily
        if self._global_step % 2000 == 0 and self._global_step > 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                fragmentation = (1.0 - allocated / reserved) if reserved > 0 else 0.0
                if fragmentation > 0.30:
                    self._clear_cuda_memory()

        # Periodic cleanup every 5000 steps (was 1000, reduced for performance)
        if self._global_step % 5000 == 0 and self._global_step > 0:
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
