"""
Batch Size Controller - Central Authority for Dynamic Batch Sizing

This module provides a unified controller for batch size decisions during training.
It replaces the fragmented approach where DynamicBatchScheduler and DynamicBatchIterator
competed for control, leading to oscillating batch sizes and OOM errors.

Key Features:
- Startup calibration via binary search to find optimal batch size
- Explicit tracking of safe/unsafe batch sizes
- Immediate OOM response with known-safe fallback
- Convergence to stable batch size (no oscillation)
- Fresh model per test: Optional model_factory creates completely fresh models for each
  batch size test, ensuring no accumulated state affects memory measurements

Usage:
    controller = BatchSizeController(
        min_batch_size=16,
        max_batch_size=256,
        target_memory=0.75,
    )

    # Optional: Run calibration at startup (basic mode - reuses model)
    optimal = controller.startup_calibration(model, sample_batch_fn, optimizer_factory)

    # Advanced: Fresh model per test (recommended for accurate calibration)
    # When using model_factory, optimizer_factory must accept a model parameter
    def model_factory():
        return build_model(config).to(device)

    def optimizer_factory(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimal = controller.startup_calibration(
        model=model,
        sample_batch_fn=sample_batch_fn,
        optimizer_factory=optimizer_factory,
        model_factory=model_factory,  # Each test uses fresh model with new states
    )

    # During training, record outcomes
    controller.record_success(batch_size=64, memory_util=0.72)

    # On OOM, get safe fallback immediately
    safe_size = controller.record_failure(batch_size=128)
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default timeout for distributed barriers (30 minutes for calibration)
_BARRIER_TIMEOUT = timedelta(minutes=30)


@dataclass
class BatchSizeState:
    """Immutable snapshot of current batch size state for logging/debugging."""
    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    safe_ceiling: int
    mode: str
    known_safe_count: int
    known_unsafe_count: int
    consecutive_successes: int


class MemoryMonitor:
    """
    Simple GPU memory monitor with caching.

    Provides memory utilization readings without the complexity of the old
    DynamicBatchScheduler. All batch size decisions are delegated to
    BatchSizeController.
    """

    def __init__(self, cache_interval_sec: float = 0.5):
        """
        Initialize memory monitor.

        Args:
            cache_interval_sec: How often to refresh memory stats (seconds)
        """
        self._cache: Dict[str, float] = {}
        self._cache_time: float = 0.0
        self._cache_interval: float = cache_interval_sec
        self._device: Optional[int] = None
        self._total_memory: int = 0

        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            self._total_memory = torch.cuda.get_device_properties(self._device).total_memory

    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        return time.monotonic() - self._cache_time > self._cache_interval

    def _refresh_cache(self) -> None:
        """Query GPU memory (causes sync)."""
        if not torch.cuda.is_available() or self._total_memory == 0:
            self._cache = {'utilization': 0.0, 'reserved_gb': 0.0, 'allocated_gb': 0.0}
            return

        reserved = torch.cuda.memory_reserved(self._device)
        allocated = torch.cuda.memory_allocated(self._device)
        # CRITICAL: Use max_memory_allocated for PEAK memory during operations
        # This captures the true peak during forward/backward, not just current state
        peak_allocated = torch.cuda.max_memory_allocated(self._device)

        # Use PEAK ALLOCATED as primary metric for calibration accuracy
        # Current allocated misses activation memory that's freed after backward
        self._cache = {
            'utilization': peak_allocated / self._total_memory,  # Peak memory usage
            'utilization_current': allocated / self._total_memory,  # Current (for debugging)
            'utilization_reserved': reserved / self._total_memory,  # Reserved (for debugging)
            'reserved_gb': reserved / 1e9,
            'allocated_gb': allocated / 1e9,
            'peak_allocated_gb': peak_allocated / 1e9,
            'total_gb': self._total_memory / 1e9,
        }
        self._cache_time = time.monotonic()

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking before a measurement."""
        if torch.cuda.is_available() and self._device is not None:
            torch.cuda.reset_peak_memory_stats(self._device)

    def get_utilization(self, force_refresh: bool = False) -> float:
        """
        Get current memory utilization (0.0 to 1.0).

        Args:
            force_refresh: If True, query GPU even if cache is fresh

        Returns:
            Memory utilization as fraction (0.0 to 1.0)
        """
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.get('utilization', 0.0)

    def get_stats(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get full memory statistics.

        Args:
            force_refresh: If True, query GPU even if cache is fresh

        Returns:
            Dict with utilization, reserved_gb, allocated_gb, total_gb
        """
        if force_refresh or self._is_cache_stale():
            self._refresh_cache()
        return self._cache.copy()


class BatchSizeController:
    """
    Central authority for batch size decisions.

    Design Principles:
    1. Single source of truth - only this component changes batch size
    2. Explicit safe/unsafe tracking - remembers what worked and what didn't
    3. Convergence - finds stable batch size through calibration, then locks
    4. Immediate OOM response - returns known-safe size instantly on failure

    Modes:
    - 'calibrating': Finding optimal batch size at startup
    - 'stable': Locked onto optimal size, no changes unless OOM
    - 'recovering': Recently had OOM, using conservative size
    """

    def __init__(
        self,
        min_batch_size: int = 16,
        max_batch_size: int = 256,
        target_memory: float = 0.75,
        oom_recovery_factor: float = 0.5,
        stability_threshold: int = 10,
        calibration_headroom: float = 0.90,
        calibration_step_size: int = 8,
    ):
        """
        Initialize batch size controller.

        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory: Target GPU memory utilization (0.0 to 1.0)
            oom_recovery_factor: Factor to reduce batch size on OOM (0.5 = halve)
            stability_threshold: Consecutive successes before considering stable
            calibration_headroom: Safety factor applied during calibration (0.90 = 10% headroom).
                                  Calibration targets target_memory * calibration_headroom to leave
                                  room for DataLoader buffers, optimizer state growth, etc.
            calibration_step_size: Minimum step size during calibration (default 8).
                                   Controls granularity of binary search and fine-tuning.
                                   Larger values = faster calibration, smaller = more precise.
        """
        self._min_batch_size = max(1, min_batch_size)
        self._max_batch_size = max_batch_size
        self._target_memory = target_memory
        self._oom_recovery_factor = oom_recovery_factor
        self._stability_threshold = stability_threshold
        self._calibration_headroom = calibration_headroom
        self._calibration_step_size = max(1, calibration_step_size)

        # Current state
        self._current_batch_size = min_batch_size
        self._mode = 'calibrating'

        # Safety tracking
        self._known_safe: Dict[int, float] = {}  # batch_size -> max_memory_util seen
        self._known_unsafe: Set[int] = set()     # batch_sizes that caused OOM
        self._safe_ceiling = min_batch_size       # Highest confirmed safe size

        # Stability tracking
        self._consecutive_successes = 0
        self._stable_batch_size: Optional[int] = None

        # Memory monitor
        self._memory_monitor = MemoryMonitor()

        logger.info(
            f"BatchSizeController initialized: "
            f"range=[{min_batch_size}, {max_batch_size}], "
            f"target_memory={target_memory:.0%}"
        )

    def _measure_batch_memory_timed(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        measure_duration_sec: float = 3.0,
        num_warmups: int = 2,
    ) -> Tuple[float, bool]:
        """
        Run training steps for a fixed duration and average peak memory readings.

        This provides more stable measurements by:
        1. Running warmup iterations to stabilize memory
        2. Measuring over a time window (default 3 seconds)
        3. Averaging peak memory readings for stability

        Args:
            model: Model to test
            optimizer: Optimizer instance
            batch: Input batch dictionary with input_ids, attention_mask
            device: CUDA device for peak memory reset
            measure_duration_sec: How long to measure for averaging (default 3 seconds)
            num_warmups: Number of warmup iterations before measuring

        Returns:
            Tuple of (average_peak_utilization, oom_occurred)
        """
        try:
            # Warmup iterations to stabilize memory
            # This ensures optimizer states are allocated and CUDA kernels are cached
            for warmup_idx in range(num_warmups):
                model.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('input_ids'),
                )
                loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

            # Clear cache after warmups, before measurement
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.05)  # 50ms stabilization delay

            # Run for measure_duration_sec and collect peak memory readings
            measurements: List[float] = []
            start_time = time.monotonic()
            iteration = 0

            while time.monotonic() - start_time < measure_duration_sec:
                # Reset peak memory before each iteration
                torch.cuda.reset_peak_memory_stats(device)

                model.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('input_ids'),
                )
                loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

                util = self._memory_monitor.get_utilization(force_refresh=True)
                measurements.append(util)

            return max(measurements), False

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return 1.0, True
            raise

    def startup_calibration(
        self,
        model: nn.Module,
        sample_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
        optimizer_factory: Callable[[], torch.optim.Optimizer],
        target_memory: Optional[float] = None,
        max_time_seconds: float = 30.0,
        seq_len: int = 512,
        use_training_mode: bool = True,
        perform_optimizer_step: bool = True,
        rank: int = 0,
        world_size: int = 1,
        model_factory: Optional[Callable[[], nn.Module]] = None,
    ) -> int:
        """
        Binary search for optimal batch size at training start.

        IMPORTANT: This now creates a REAL OPTIMIZER during calibration and runs
        full training mode with forward + backward + optimizer.step() to accurately
        measure the complete memory footprint including:
        - Model parameters
        - Gradients
        - Activations (with gradient checkpointing if enabled)
        - Optimizer states (momentum, variance for Adam)

        This provides accurate calibration to the target memory threshold without
        needing to estimate or reserve headroom for optimizer states.

        Algorithm:
        1. Create fresh model (if model_factory provided) and optimizer from factory
        2. Start at min_batch_size (definitely works)
        3. Double until OOM or util > target
        4. Binary search between last_good and first_bad
        5. Clean up test model and optimizer
        6. Lock onto stable size

        Args:
            model: The model to calibrate with (used if model_factory is None)
            sample_batch_fn: Function that creates a batch given batch_size
            optimizer_factory: Callable that creates optimizer instances for testing.
                              If model_factory is provided, this should accept a model parameter:
                              optimizer_factory(model) -> Optimizer
            target_memory: Override target memory utilization (uses full target directly)
            max_time_seconds: Maximum time to spend calibrating
            seq_len: Sequence length for calibration batches (unused, kept for compatibility)
            use_training_mode: If True, run forward+backward+optimizer.step() for accurate memory
            perform_optimizer_step: If True, call optimizer.step() to allocate optimizer states
            rank: Current process rank for distributed training (0 for single GPU)
            world_size: Total number of processes for distributed training (1 for single GPU)
            model_factory: Optional callable that creates fresh model instances for each test.
                          If provided, each batch size test uses a completely fresh model
                          with new states, ensuring accurate memory measurements.

        Returns:
            Optimal batch size found
        """

        if target_memory is None:
            target_memory = self._target_memory

        # Apply calibration headroom to leave room for runtime overhead:
        # - DataLoader prefetch buffers
        # - Optimizer state growth over training steps
        # - Mixed precision scaler state
        # - CUDA memory pool expansion
        effective_target = target_memory * self._calibration_headroom
        logger.info(
            f"Calibration target: {target_memory:.0%} × {self._calibration_headroom:.0%} headroom = {effective_target:.0%}"
        )

        # Track whether we're using fresh models for each test
        use_fresh_models = model_factory is not None

        # Current test model and optimizer (will be recreated each iteration if use_fresh_models)
        test_model = model

        # Create initial test optimizer for accurate memory measurement
        # If model_factory is provided, optimizer_factory should accept model parameter
        if use_fresh_models:
            test_optimizer = optimizer_factory(test_model)
        else:
            test_optimizer = optimizer_factory()
        optimizer_name = test_optimizer.__class__.__name__

        if rank == 0:
            if use_fresh_models:
                logger.info(f"Calibrating with FRESH MODEL per test + optimizer: {optimizer_name}")
            else:
                logger.info(f"Calibrating with optimizer: {optimizer_name}")

        start_time = time.time()

        # Setup distributed synchronization if needed
        is_distributed = world_size > 1
        if is_distributed:
            import torch.distributed as dist
            if not dist.is_initialized():
                is_distributed = False  # Fallback if dist not initialized

        # Get device once at the beginning to avoid inconsistencies
        device = next(test_model.parameters()).device

        # Find the GPU with smallest VRAM - calibrate on that one
        calibrating_rank = rank  # Default: all ranks calibrate (single GPU)
        if is_distributed:
            total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            vram_tensor = torch.tensor([total_memory_gb], dtype=torch.float32, device=device)
            vram_list = [torch.zeros_like(vram_tensor) for _ in range(world_size)]
            dist.all_gather(vram_list, vram_tensor)

            # Find rank with minimum VRAM
            vram_sizes = [v.item() for v in vram_list]
            min_vram_rank = vram_sizes.index(min(vram_sizes))
            calibrating_rank = min_vram_rank

            if rank == 0:
                logger.info(
                    f"Starting batch size calibration with REAL OPTIMIZER "
                    f"(target={target_memory:.0%}, timeout={max_time_seconds}s, "
                    f"training_mode={use_training_mode}, perform_optimizer_step={perform_optimizer_step}, "
                    f"distributed={is_distributed}, world_size={world_size})"
                )
                logger.info(f"GPU VRAM sizes: {[f'{v:.1f}GB' for v in vram_sizes]}")
                logger.info(f"Calibrating on rank {calibrating_rank} (smallest GPU: {vram_sizes[calibrating_rank]:.1f}GB)")
        else:
            if rank == 0:
                logger.info(
                    f"Starting batch size calibration with REAL OPTIMIZER "
                    f"(target={target_memory:.0%}, timeout={max_time_seconds}s, "
                    f"training_mode={use_training_mode}, perform_optimizer_step={perform_optimizer_step}, "
                    f"distributed={is_distributed}, world_size={world_size})"
                )

        # Phase 1: Find upper bound by doubling
        if rank == 0:
            logger.info(f"Phase 1: Doubling ({optimizer_name}, target={effective_target:.0%})")
        test_size = self._min_batch_size
        last_good = test_size
        first_bad = self._max_batch_size + 1  # Sentinel for "not found"


        # Use training mode for accurate memory estimation
        was_training = test_model.training
        if use_training_mode:
            test_model.train()
        else:
            test_model.eval()



        # All ranks participate in forward/backward (required for DDP)
        # But only calibrating rank measures memory and decides on batch sizes
        if is_distributed and rank == calibrating_rank:
            logger.info(f"[Rank {rank}] Calibrating based on smallest GPU memory")
        elif is_distributed and rank != calibrating_rank:
            logger.info(f"[Rank {rank}] Participating in calibration (following rank {calibrating_rank})")

        # CRITICAL: Add barrier to ensure all ranks are synchronized before entering calibration loop
        if is_distributed:
            dist.barrier(timeout=_BARRIER_TIMEOUT)

        while test_size <= self._max_batch_size:
            # Synchronize test_size from calibrating rank to all others
            if is_distributed:
                test_size_tensor = torch.tensor([test_size], dtype=torch.int64, device=device)
                dist.broadcast(test_size_tensor, src=calibrating_rank)
                test_size = int(test_size_tensor.item())

            if time.time() - start_time > max_time_seconds:
                if rank == calibrating_rank:
                    logger.warning(f"Calibration timeout after {max_time_seconds}s")
                break

            # Track if this rank hit an OOM - must be set before try block
            oom_on_this_rank = False
            should_break = 0

            try:
                # FULL RESET: Recreate model (if using fresh models) and optimizer for accurate memory measurement
                if use_training_mode and perform_optimizer_step:
                    del test_optimizer
                    if use_fresh_models:
                        # Delete old model and create fresh one
                        del test_model
                    else:
                        test_model.zero_grad(set_to_none=True)
                    # Aggressive cleanup to prevent memory fragmentation
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(0.1)  # 100ms stabilization delay for GPU memory manager

                    if use_fresh_models:
                        # Create fresh model with completely new states
                        test_model = model_factory()
                        test_model = test_model.to(device)
                        if use_training_mode:
                            test_model.train()
                        else:
                            test_model.eval()
                        test_optimizer = optimizer_factory(test_model)
                    else:
                        test_optimizer = optimizer_factory()
                else:
                    torch.cuda.empty_cache()

                batch = sample_batch_fn(test_size)

                # Move to GPU if not already
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # CRITICAL: Reset peak memory stats before test to get accurate peak measurement
                torch.cuda.reset_peak_memory_stats(device)

                if use_training_mode:
                    # Forward pass only - check for OOM before backward
                    test_model.zero_grad(set_to_none=True)
                    outputs = test_model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('input_ids'),  # Use input_ids as labels for calibration
                    )
                    loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                else:
                    with torch.no_grad():
                        _ = test_model(
                            input_ids=batch.get('input_ids'),
                            attention_mask=batch.get('attention_mask'),
                        )

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    oom_on_this_rank = True
                    # AGGRESSIVE CLEANUP: Delete any partially created objects
                    try:
                        del test_optimizer
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del test_model
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del batch
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del loss
                    except (NameError, UnboundLocalError):
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    if rank == calibrating_rank:
                        first_bad = test_size
                        self._known_unsafe.add(test_size)
                        should_break = 1
                        logger.info(f"  BS={test_size}: OOM ✗")
                else:
                    raise

            # CRITICAL: Synchronize here so both ranks reach this point regardless of OOM
            # This prevents deadlock when one rank hits OOM and the other doesn't
            if is_distributed:
                dist.barrier(timeout=_BARRIER_TIMEOUT)

            # CRITICAL: Check if any rank hit OOM before calling backward/optimizer
            # With DDP, backward() has gradient synchronization that requires all ranks
            # With DeepSpeed, optimizer.step() requires collective communication
            # Both ranks must either call them together or skip them together
            if use_training_mode and not oom_on_this_rank:
                if is_distributed:
                    pass
                    # Check if ANY rank hit OOM during forward
                    oom_check_tensor = torch.tensor([1 if oom_on_this_rank else 0], dtype=torch.int64, device=device)
                    oom_check_list = [torch.zeros_like(oom_check_tensor) for _ in range(world_size)]
                    dist.all_gather(oom_check_list, oom_check_tensor)
                    any_rank_oom = any(t.item() > 0 for t in oom_check_list)

                    if any_rank_oom:
                        # Don't call backward - would deadlock with DDP gradient sync
                        try:
                            test_model.zero_grad(set_to_none=True)
                        except (NameError, UnboundLocalError):
                            pass
                    else:
                        # All ranks succeeded forward - safe to call backward()
                        loss.backward()
                        # CRITICAL: Use barrier instead of cuda.synchronize() to prevent NCCL deadlock
                        # cuda.synchronize() can block on pending NCCL ops while another rank
                        # moves to a different collective, causing deadlock
                        dist.barrier(timeout=_BARRIER_TIMEOUT)
                else:
                    # Single GPU - just check this rank, wrap backward in try-except
                    if not oom_on_this_rank:
                        try:
                            loss.backward()
                            torch.cuda.synchronize()
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                if rank == calibrating_rank:
                                    first_bad = test_size
                                    self._known_unsafe.add(test_size)
                                    should_break = 1
                                    logger.info(f"  BS={test_size}: OOM during backward ✗")
                            else:
                                raise

            # Now check for optimizer.step() - only if backward succeeded
            if use_training_mode and perform_optimizer_step and not oom_on_this_rank:
                if is_distributed:
                    # CRITICAL: Synchronize CUDA before collective ops
                    # DDP backward() triggers async gradient sync via NCCL hooks
                    # If these haven't completed, the next all_gather can deadlock
                    torch.cuda.synchronize()

                    # Check if ANY rank hit OOM
                    oom_check_tensor = torch.tensor([1 if oom_on_this_rank else 0], dtype=torch.int64, device=device)
                    oom_check_list = [torch.zeros_like(oom_check_tensor) for _ in range(world_size)]
                    dist.all_gather(oom_check_list, oom_check_tensor)
                    any_rank_oom = any(t.item() > 0 for t in oom_check_list)

                    if any_rank_oom:
                        # Clear gradients if we have any
                        try:
                            test_model.zero_grad(set_to_none=True)
                        except (NameError, UnboundLocalError):
                            pass
                    else:
                        # All ranks succeeded - safe to call optimizer.step()
                        try:
                            test_optimizer.step()
                            test_optimizer.zero_grad(set_to_none=True)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                logger.info(f"  BS={test_size}: OOM during optimizer.step() ✗")
                            else:
                                raise
                else:
                    # Single GPU - optimizer step with OOM handling
                    try:
                        test_optimizer.step()
                        test_optimizer.zero_grad(set_to_none=True)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "out of memory" in str(e).lower():
                            oom_on_this_rank = True
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            if rank == calibrating_rank:
                                first_bad = test_size
                                self._known_unsafe.add(test_size)
                                should_break = 1
                                logger.info(f"  BS={test_size}: OOM during optimizer.step() ✗")
                        else:
                            raise

            # CRITICAL: Wait for all GPU operations to complete before measuring memory
            # Optimizer states are lazily allocated and need sync to be fully materialized
            if not oom_on_this_rank:
                torch.cuda.synchronize()

                # Only calibrating rank measures memory and makes decisions
                if rank == calibrating_rank:
                    # Run for 3 seconds and average peak memory readings for stability
                    measurements: List[float] = []
                    measure_duration = 3.0  # seconds
                    measure_start = time.monotonic()

                    while time.monotonic() - measure_start < measure_duration:
                        try:
                            # Reset peak memory before each iteration
                            torch.cuda.reset_peak_memory_stats(device)

                            # Run one training iteration
                            test_model.zero_grad(set_to_none=True)
                            outputs = test_model(
                                input_ids=batch.get('input_ids'),
                                attention_mask=batch.get('attention_mask'),
                                labels=batch.get('input_ids'),
                            )
                            loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                            loss.backward()
                            if perform_optimizer_step:
                                test_optimizer.step()
                                test_optimizer.zero_grad(set_to_none=True)
                            else:
                                test_model.zero_grad(set_to_none=True)
                            torch.cuda.synchronize()

                            util = self._memory_monitor.get_utilization(force_refresh=True)
                            measurements.append(util)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                first_bad = test_size
                                self._known_unsafe.add(test_size)
                                should_break = 1
                                logger.info(f"  BS={test_size}: OOM during averaging ✗")
                                break
                            raise

                    if not oom_on_this_rank and measurements:
                        # Use average of measurements for stability
                        util = sum(measurements) / len(measurements)
                        num_samples = len(measurements)

                        status = "✓" if util <= effective_target else "✗"
                        logger.info(f"  BS={test_size}: {util:.1%} {status} (avg of {num_samples} samples over {measure_duration:.0f}s)")

                        if util > effective_target:
                            # Over target, found upper bound
                            first_bad = test_size
                            should_break = 1
                        else:
                            last_good = test_size
                            self._record_safe(test_size, util)
                            test_size *= 2

            # Now broadcast decision to all ranks - both ranks are synchronized here
            if is_distributed:
                break_tensor = torch.tensor([should_break, test_size, first_bad, last_good], dtype=torch.int64, device=device)
                dist.broadcast(break_tensor, src=calibrating_rank)
                should_break = int(break_tensor[0].item())
                test_size = int(break_tensor[1].item())
                first_bad = int(break_tensor[2].item())
                last_good = int(break_tensor[3].item())

            if should_break:
                break

        # Synchronize last_good and first_bad from calibrating rank to all others
        if is_distributed:
            sync_tensor = torch.tensor([last_good, first_bad], dtype=torch.int64, device=device)
            dist.broadcast(sync_tensor, src=calibrating_rank)
            last_good = int(sync_tensor[0].item())
            first_bad = int(sync_tensor[1].item())

        # Phase 2: Binary search for optimal with full reset before each test
        # Use calibration_step_size for alignment to avoid testing every batch size
        step = self._calibration_step_size
        if rank == calibrating_rank:
            logger.info(f"Phase 2: Binary search {last_good} → {first_bad} ({optimizer_name}, step={step})")

        while first_bad - last_good > step:

            # Calibrating rank computes mid, broadcasts to all
            if rank == calibrating_rank:
                # Simple midpoint aligned to calibration step size
                # This avoids the over-complicated interpolation that caused BS=65 issues
                mid = (last_good + first_bad) // 2
                # Align to calibration step size for efficient search
                mid = (mid // step) * step
                # Ensure we make progress (at least one step from last_good)
                mid = max(last_good + step, mid)
                # Don't exceed first_bad
                mid = min(mid, first_bad - 1)
            else:
                mid = 0

            if is_distributed:
                mid_tensor = torch.tensor([mid], dtype=torch.int64, device=device)
                dist.broadcast(mid_tensor, src=calibrating_rank)
                mid = int(mid_tensor.item())


            if mid <= last_good or mid >= first_bad:
                break

            if time.time() - start_time > max_time_seconds:
                if rank == calibrating_rank:
                    logger.warning(f"Calibration timeout during binary search")
                break

            # Track OOM for this iteration - same pattern as Phase 1
            oom_on_this_rank = False

            try:
                # FULL RESET: Always recreate model and optimizer for accurate memory measurement
                # Safely delete existing objects (may not exist after previous OOM)
                try:
                    del test_optimizer
                except (NameError, UnboundLocalError):
                    pass
                if use_fresh_models:
                    try:
                        del test_model
                    except (NameError, UnboundLocalError):
                        pass
                else:
                    try:
                        test_model.zero_grad(set_to_none=True)
                    except (NameError, UnboundLocalError):
                        pass

                # Aggressive cleanup to prevent memory fragmentation
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.1)  # 100ms stabilization delay for GPU memory manager

                # Create fresh model and optimizer
                if use_fresh_models:
                    test_model = model_factory()
                    test_model = test_model.to(device)
                    if use_training_mode:
                        test_model.train()
                    else:
                        test_model.eval()
                    test_optimizer = optimizer_factory(test_model)
                else:
                    test_optimizer = optimizer_factory()

                batch = sample_batch_fn(mid)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # CRITICAL: Reset peak memory stats before test to get accurate peak measurement
                torch.cuda.reset_peak_memory_stats(device)

                if use_training_mode:
                    test_model.zero_grad(set_to_none=True)
                    outputs = test_model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('input_ids'),
                    )
                    loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                else:
                    with torch.no_grad():
                        _ = test_model(
                            input_ids=batch.get('input_ids'),
                            attention_mask=batch.get('attention_mask'),
                        )

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    oom_on_this_rank = True
                    # AGGRESSIVE CLEANUP: Delete any partially created objects
                    try:
                        del test_optimizer
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del test_model
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del batch
                    except (NameError, UnboundLocalError):
                        pass
                    try:
                        del loss
                    except (NameError, UnboundLocalError):
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if rank == calibrating_rank:
                        first_bad = mid
                        self._known_unsafe.add(mid)
                        logger.info(f"  BS={mid}: OOM ✗ (continuing search {last_good} → {mid})")
                    # Continue binary search - don't break, just skip backward pass
                else:
                    raise

            # CRITICAL: Synchronize all ranks before backward (same as Phase 1)
            if is_distributed:
                dist.barrier(timeout=_BARRIER_TIMEOUT)

            # Check if ANY rank hit OOM before calling backward
            if use_training_mode and not oom_on_this_rank:
                if is_distributed:
                    torch.cuda.synchronize()
                    oom_check = torch.tensor([1 if oom_on_this_rank else 0], dtype=torch.int64, device=device)
                    oom_list = [torch.zeros_like(oom_check) for _ in range(world_size)]
                    dist.all_gather(oom_list, oom_check)
                    any_oom = any(t.item() > 0 for t in oom_list)

                    if any_oom:
                        try:
                            test_model.zero_grad(set_to_none=True)
                        except (NameError, UnboundLocalError):
                            pass
                    else:
                        loss.backward()
                        dist.barrier(timeout=_BARRIER_TIMEOUT)

                        if perform_optimizer_step:
                            test_optimizer.step()
                            test_optimizer.zero_grad(set_to_none=True)
                        else:
                            test_model.zero_grad(set_to_none=True)
                else:
                    # Single GPU path - wrap backward in try-except for OOM handling
                    if not oom_on_this_rank:
                        try:
                            loss.backward()
                            if perform_optimizer_step:
                                test_optimizer.step()
                                test_optimizer.zero_grad(set_to_none=True)
                            else:
                                test_model.zero_grad(set_to_none=True)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                if rank == calibrating_rank:
                                    first_bad = mid
                                    self._known_unsafe.add(mid)
                                    logger.info(f"  BS={mid}: OOM during backward ✗ (search {last_good} → {mid})")
                            else:
                                raise

            # Measure memory and make decisions
            if not oom_on_this_rank:
                torch.cuda.synchronize()
                if rank == calibrating_rank:
                    # Run for 3 seconds and average peak memory readings for stability
                    measurements: List[float] = []
                    measure_duration = 3.0  # seconds
                    measure_start = time.monotonic()

                    while time.monotonic() - measure_start < measure_duration:
                        try:
                            # Reset peak memory before each iteration
                            torch.cuda.reset_peak_memory_stats(device)

                            # Run one training iteration
                            test_model.zero_grad(set_to_none=True)
                            outputs = test_model(
                                input_ids=batch.get('input_ids'),
                                attention_mask=batch.get('attention_mask'),
                                labels=batch.get('input_ids'),
                            )
                            loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                            loss.backward()
                            if perform_optimizer_step:
                                test_optimizer.step()
                                test_optimizer.zero_grad(set_to_none=True)
                            else:
                                test_model.zero_grad(set_to_none=True)
                            torch.cuda.synchronize()

                            util = self._memory_monitor.get_utilization(force_refresh=True)
                            measurements.append(util)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                first_bad = mid
                                self._known_unsafe.add(mid)
                                logger.info(f"  BS={mid}: OOM during averaging ✗")
                                break
                            raise

                    if not oom_on_this_rank and measurements:
                        # Use average of measurements for stability
                        util = sum(measurements) / len(measurements)
                        num_samples = len(measurements)

                        status = "✓" if util <= effective_target else "✗"
                        logger.info(f"  BS={mid}: {util:.1%} {status} (avg of {num_samples} samples over {measure_duration:.0f}s)")

                        if util <= effective_target:
                            last_good = mid
                            self._record_safe(mid, util)
                        else:
                            first_bad = mid

            # Broadcast decision to all ranks
            if is_distributed:
                decision_tensor = torch.tensor([last_good, first_bad], dtype=torch.int64, device=device)
                dist.broadcast(decision_tensor, src=calibrating_rank)
                last_good = int(decision_tensor[0].item())
                first_bad = int(decision_tensor[1].item())

        # Phase 3: Fine-tuning - step-based search when gap is small but utilization is far from target
        # This catches cases where BS=48 gives 72% but BS=56 OOMs (gap too small for binary search)
        # Uses calibration_step_size for efficiency (avoids testing every single batch size)
        best_util = self._known_safe.get(last_good, 0.0)
        fine_tune_threshold = 0.90  # Fine-tune if util < 90% of target

        if first_bad - last_good <= step and best_util < effective_target * fine_tune_threshold:
            if rank == calibrating_rank:
                logger.info(f"Fine-tuning: util={best_util:.1%} is below {fine_tune_threshold:.0%} of target, trying sizes {last_good+step} to {first_bad-1} (step={step})")

            # Try each size from last_good+step to first_bad-1 with step increments
            for fine_size in range(last_good + step, first_bad, step):
                # Broadcast fine_size to all ranks
                if is_distributed:
                    fine_tensor = torch.tensor([fine_size], dtype=torch.int64, device=device)
                    dist.broadcast(fine_tensor, src=calibrating_rank)
                    fine_size = int(fine_tensor.item())

                if time.time() - start_time > max_time_seconds:
                    if rank == calibrating_rank:
                        logger.warning("Fine-tuning timeout")
                    break

                oom_on_this_rank = False
                should_stop = False

                try:
                    # FULL RESET: Safely delete existing objects (may not exist after previous OOM)
                    try:
                        del test_optimizer
                    except (NameError, UnboundLocalError):
                        pass
                    if use_fresh_models:
                        try:
                            del test_model
                        except (NameError, UnboundLocalError):
                            pass
                    else:
                        try:
                            test_model.zero_grad(set_to_none=True)
                        except (NameError, UnboundLocalError):
                            pass

                    # Aggressive cleanup to prevent memory fragmentation
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(0.1)  # 100ms stabilization delay for GPU memory manager

                    # Create fresh model and optimizer
                    if use_fresh_models:
                        test_model = model_factory()
                        test_model = test_model.to(device)
                        if use_training_mode:
                            test_model.train()
                        else:
                            test_model.eval()
                        test_optimizer = optimizer_factory(test_model)
                    else:
                        test_optimizer = optimizer_factory()

                    batch = sample_batch_fn(fine_size)
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # Run for 3 seconds and average peak memory readings for stability
                    fine_measurements: List[float] = []
                    fine_measure_duration = 3.0  # seconds
                    fine_measure_start = time.monotonic()

                    while time.monotonic() - fine_measure_start < fine_measure_duration:
                        try:
                            # CRITICAL: Reset peak memory stats before test to get accurate peak measurement
                            torch.cuda.reset_peak_memory_stats(device)

                            if use_training_mode:
                                test_model.zero_grad(set_to_none=True)
                                outputs = test_model(
                                    input_ids=batch.get('input_ids'),
                                    attention_mask=batch.get('attention_mask'),
                                    labels=batch.get('input_ids'),
                                )
                                loss = outputs.get('loss', outputs.get('logits', outputs).mean())
                                loss.backward()
                                if perform_optimizer_step:
                                    test_optimizer.step()
                                    test_optimizer.zero_grad(set_to_none=True)
                            else:
                                with torch.no_grad():
                                    _ = test_model(
                                        input_ids=batch.get('input_ids'),
                                        attention_mask=batch.get('attention_mask'),
                                    )
                            torch.cuda.synchronize()
                            util = self._memory_monitor.get_utilization(force_refresh=True)
                            fine_measurements.append(util)

                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                oom_on_this_rank = True
                                gc.collect()
                                torch.cuda.empty_cache()
                                break
                            raise

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        oom_on_this_rank = True
                        # AGGRESSIVE CLEANUP: Delete any partially created objects
                        try:
                            del test_optimizer
                        except (NameError, UnboundLocalError):
                            pass
                        try:
                            del test_model
                        except (NameError, UnboundLocalError):
                            pass
                        try:
                            del batch
                        except (NameError, UnboundLocalError):
                            pass
                        try:
                            del loss
                        except (NameError, UnboundLocalError):
                            pass
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if rank == calibrating_rank:
                            first_bad = fine_size
                            self._known_unsafe.add(fine_size)
                            logger.info(f"Fine-tune: OOM at BS={fine_size}")
                        should_stop = True
                    else:
                        raise

                # Handle OOM from measurement loop
                if oom_on_this_rank and not should_stop:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if rank == calibrating_rank:
                        first_bad = fine_size
                        self._known_unsafe.add(fine_size)
                        logger.info(f"Fine-tune: OOM at BS={fine_size}")
                    should_stop = True

                if is_distributed:
                    dist.barrier(timeout=_BARRIER_TIMEOUT)
                    # Check if any rank had OOM
                    torch.cuda.synchronize()
                    oom_check = torch.tensor([1 if oom_on_this_rank else 0], dtype=torch.int64, device=device)
                    oom_list = [torch.zeros_like(oom_check) for _ in range(world_size)]
                    dist.all_gather(oom_list, oom_check)
                    any_oom = any(t.item() > 0 for t in oom_list)
                    if any_oom:
                        should_stop = True

                if not oom_on_this_rank and not should_stop and fine_measurements:
                    if rank == calibrating_rank:
                        util = sum(fine_measurements) / len(fine_measurements)
                        num_samples = len(fine_measurements)
                        logger.info(f"Fine-tune: BS={fine_size}, util={util:.1%} (target={effective_target:.0%}, avg of {num_samples} samples)")

                        if util <= effective_target:
                            last_good = fine_size
                            best_util = util
                            self._record_safe(fine_size, util)
                        else:
                            # Over target, stop fine-tuning
                            should_stop = True

                # Broadcast result and stop flag
                if is_distributed:
                    fine_result = torch.tensor([last_good, 1 if should_stop else 0], dtype=torch.int64, device=device)
                    dist.broadcast(fine_result, src=calibrating_rank)
                    last_good = int(fine_result[0].item())
                    if fine_result[1].item() > 0:
                        break

                if should_stop:
                    break

        # Broadcast final result from calibrating rank to all others
        if is_distributed:
            result_tensor = torch.tensor([last_good], dtype=torch.int64, device=device)
            dist.broadcast(result_tensor, src=calibrating_rank)
            last_good = int(result_tensor.item())

        # Finalize
        self._current_batch_size = last_good
        self._stable_batch_size = last_good
        self._safe_ceiling = last_good
        self._mode = 'stable'

        # Clean up test optimizer and model (if using fresh models)
        # Use safe deletion since variables may not exist after OOM
        try:
            del test_optimizer
        except (NameError, UnboundLocalError):
            pass
        if use_fresh_models:
            try:
                del test_model
            except (NameError, UnboundLocalError):
                pass
        torch.cuda.empty_cache()

        # Restore original training state on the original model (not test_model)
        if was_training:
            model.train()
        else:
            model.eval()

        elapsed = time.time() - start_time
        if rank == 0:
            if is_distributed:
                logger.info(
                    f"✓ Calibration complete in {elapsed:.1f}s (calibrated on rank {calibrating_rank}): "
                    f"optimal batch size = {last_good}"
                )
            else:
                logger.info(
                    f"✓ Calibration complete in {elapsed:.1f}s: "
                    f"optimal batch size = {last_good}, "
                    f"tested {len(self._known_safe)} safe sizes, "
                    f"found {len(self._known_unsafe)} unsafe sizes"
                )

        return last_good

    def _record_safe(self, batch_size: int, memory_util: float) -> None:
        """Internal: Record a safe batch size."""
        self._known_safe[batch_size] = max(
            self._known_safe.get(batch_size, 0.0),
            memory_util
        )
        self._safe_ceiling = max(self._safe_ceiling, batch_size)

    def record_success(self, batch_size: int, memory_util: Optional[float] = None) -> None:
        """
        Record that a batch completed successfully.

        Call this after each successful forward+backward pass to help the
        controller learn which batch sizes are safe.

        Args:
            batch_size: Batch size that succeeded
            memory_util: Memory utilization during the batch (optional)
        """
        if memory_util is None:
            memory_util = self._memory_monitor.get_utilization()

        self._record_safe(batch_size, memory_util)
        self._consecutive_successes += 1

        # Transition from recovering to stable after enough successes
        if self._mode == 'recovering' and self._consecutive_successes >= self._stability_threshold:
            self._mode = 'stable'
            logger.info(f"Recovered to stable mode at BS={batch_size}")

    def record_failure(self, batch_size: int) -> int:
        """
        Record an OOM error and get a safe batch size to use.

        Call this when OOM occurs. Returns a known-safe batch size to use
        for the retry.

        Args:
            batch_size: Batch size that caused OOM

        Returns:
            Safe batch size to use for retry
        """
        self._known_unsafe.add(batch_size)
        self._consecutive_successes = 0
        self._mode = 'recovering'

        # Also mark all sizes >= batch_size as unsafe (they will also OOM)
        # This prevents the system from trying BS=44 again after it failed
        for unsafe_size in range(batch_size, self._max_batch_size + 1):
            self._known_unsafe.add(unsafe_size)

        # Find largest known-safe size below the failed size
        safe_sizes = [s for s in self._known_safe.keys() if s < batch_size and s not in self._known_unsafe]

        if safe_sizes:
            new_size = max(safe_sizes)
        else:
            pass
            # Fallback: reduce by oom_recovery_factor
            new_size = max(
                self._min_batch_size,
                int(batch_size * self._oom_recovery_factor)
            )
            # Round to multiple of min_batch_size
            new_size = (new_size // self._min_batch_size) * self._min_batch_size
            new_size = max(self._min_batch_size, new_size)

        self._current_batch_size = new_size

        # CRITICAL: Update max_batch_size to prevent future attempts at unsafe sizes
        self._max_batch_size = min(self._max_batch_size, batch_size - 1)

        # Update safe ceiling to be below the unsafe size
        self._safe_ceiling = min(self._safe_ceiling, batch_size - 1)

        logger.warning(
            f"OOM at BS={batch_size}, falling back to BS={new_size} "
            f"(known_safe={len(self._known_safe)}, known_unsafe={len(self._known_unsafe)}, "
            f"new_max={self._max_batch_size})"
        )

        return new_size

    def get_batch_size(self) -> int:
        """
        Get the current recommended batch size.

        After calibration, this returns a stable value. Only changes on OOM.

        Returns:
            Current batch size to use
        """
        return self._current_batch_size

    def get_safe_batch_for_seq_len(self, seq_len: int, base_seq_len: int = 512) -> int:
        """
        Get safe batch size adjusted for sequence length.

        Longer sequences require smaller batch sizes due to quadratic
        attention memory scaling.

        Args:
            seq_len: Current sequence length
            base_seq_len: Reference sequence length for current batch size

        Returns:
            Adjusted batch size safe for the given sequence length
        """
        if seq_len <= base_seq_len:
            return self._current_batch_size

        # Memory scales roughly as O(batch * seq^2) for attention
        # So batch_new = batch_old * (seq_old / seq_new)^2
        ratio = (base_seq_len / seq_len) ** 2
        adjusted = int(self._current_batch_size * ratio)

        # Round to multiple of min_batch_size
        adjusted = (adjusted // self._min_batch_size) * self._min_batch_size
        adjusted = max(self._min_batch_size, min(adjusted, self._current_batch_size))

        return adjusted

    def get_state(self) -> BatchSizeState:
        """Get current state for logging/debugging."""
        return BatchSizeState(
            current_batch_size=self._current_batch_size,
            min_batch_size=self._min_batch_size,
            max_batch_size=self._max_batch_size,
            safe_ceiling=self._safe_ceiling,
            mode=self._mode,
            known_safe_count=len(self._known_safe),
            known_unsafe_count=len(self._known_unsafe),
            consecutive_successes=self._consecutive_successes,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for logging."""
        return {
            'current_batch_size': self._current_batch_size,
            'safe_ceiling': self._safe_ceiling,
            'mode': self._mode,
            'known_safe_sizes': sorted(self._known_safe.keys()),
            'known_unsafe_sizes': sorted(self._known_unsafe),
            'consecutive_successes': self._consecutive_successes,
            'target_memory': self._target_memory,
        }

    @property
    def memory_monitor(self) -> MemoryMonitor:
        """Access the memory monitor for direct utilization queries."""
        return self._memory_monitor

    @property
    def is_stable(self) -> bool:
        """Check if batch size has stabilized."""
        return self._mode == 'stable'

    @property
    def mode(self) -> str:
        """Get current mode: 'calibrating', 'stable', or 'recovering'."""
        return self._mode

    @property
    def safe_ceiling(self) -> int:
        """Get the highest known-safe batch size."""
        return self._safe_ceiling

    @property
    def max_batch_size(self) -> int:
        """Get current maximum batch size (may be reduced after OOM)."""
        return self._max_batch_size

    def is_size_safe(self, batch_size: int) -> bool:
        """
        Check if a batch size is known to be safe (not in unsafe set).

        Args:
            batch_size: Size to check

        Returns:
            True if size is potentially safe, False if known to cause OOM
        """
        return batch_size not in self._known_unsafe


def create_batch_size_controller(config: Dict[str, Any]) -> Optional[BatchSizeController]:
    """
    Factory function to create BatchSizeController from config dict.

    Args:
        config: Configuration dictionary (full config or just dynamic_batching section)

    Returns:
        BatchSizeController instance, or None if disabled
    """
    # Handle nested config
    db_config = config.get('dynamic_batching', config)

    if not db_config.get('enabled', False):
        return None

    return BatchSizeController(
        min_batch_size=db_config.get('min_batch_size', 16),
        max_batch_size=db_config.get('max_batch_size', 256),
        target_memory=db_config.get('target_memory',
                                    db_config.get('target_memory_utilization',
                                    db_config.get('target_memory_threshold', 0.75))),
        oom_recovery_factor=db_config.get('oom_recovery_factor', 0.5),
        stability_threshold=db_config.get('stability_threshold', 10),
        calibration_headroom=db_config.get('calibration_headroom', 0.90),
        calibration_step_size=db_config.get('calibration_step_size', 8),
    )
