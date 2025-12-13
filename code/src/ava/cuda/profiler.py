"""
Nsight Profiling Integration for Ava Training Framework

This module provides comprehensive integration with NVIDIA Nsight Systems and Nsight Compute
for profiling GPU operations during training, with advanced analytics and bottleneck detection.

Features:
- NVTX annotations for Nsight Systems visualization
- PyTorch Profiler integration with TensorBoard export
- Per-kernel timing and memory statistics
- Layer-by-layer breakdown with activation memory tracking
- MoE expert routing analysis with load balance metrics
- Memory fragmentation detection and analysis
- Attention pattern profiling (sequence length, head utilization)
- Communication overhead tracking for distributed training
- Tensor shape and dtype statistics
- Automatic bottleneck detection with recommendations
- Timeline visualization and HTML report generation
- Gradient flow analysis
- Cache hit/miss tracking for MoE routing

Usage:
    # Basic profiling in training script
    from ava.cuda.profiler import NsightProfiler, DetailedProfiler

    # Simple profiler for Nsight integration
    profiler = NsightProfiler(
        enabled=True,
        output_dir="./profiles",
        profile_steps=(10, 20),  # Profile steps 10-20
        use_cuda_nvtx=True,
    )

    for step, batch in enumerate(dataloader):
        with profiler.step(step):
            loss = model(batch)
            loss.backward()

    # Advanced detailed profiler with full analytics
    detailed_profiler = DetailedProfiler(
        enabled=True,
        output_dir="./profiles",
        track_kernels=True,
        track_layers=True,
        track_moe=True,
        track_memory_timeline=True,
        track_attention=True,
        track_communication=True,
        track_gradients=True,
        detect_bottlenecks=True,
    )

    for step, batch in enumerate(dataloader):
        with detailed_profiler.step(step):
            with detailed_profiler.phase("forward"):
                output = model(batch)
            with detailed_profiler.phase("backward"):
                output.loss.backward()
            with detailed_profiler.phase("optimizer"):
                optimizer.step()

    detailed_profiler.finish()  # Generate reports

    # For Nsight Systems external profiling:
    # nsys profile -o my_profile python train.py --enable-profiling

    # For Nsight Compute kernel analysis:
    # ncu --set full -o kernel_profile python train.py --enable-profiling
"""

import os
import json
import logging
import subprocess
import shutil
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

import torch

# Import optimized CUDA stream utilities
try:
    from ava.cuda.streams import efficient_sync, optimal_sync_context
    _CUDA_STREAMS_AVAILABLE = True
except ImportError:
    _CUDA_STREAMS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for Nsight profiling."""
    enabled: bool = False
    output_dir: str = "./profiles"
    profile_steps: Tuple[int, int] = (0, 999999)  # (start_step, end_step) - profile all by default
    use_cuda_nvtx: bool = True  # NVTX annotations for Nsight Systems
    use_torch_profiler: bool = True  # PyTorch profiler (TensorBoard/Chrome trace)
    record_shapes: bool = True  # Record tensor shapes
    profile_memory: bool = True  # Profile memory allocation
    with_stack: bool = False  # Record Python stack (expensive)
    with_flops: bool = True  # Estimate FLOPS
    with_modules: bool = True  # Record module hierarchy

    # Nsight Systems specific
    capture_cuda_graph: bool = False
    capture_memory_operations: bool = True

    # Always emit NVTX even outside profile_steps range
    always_emit_nvtx: bool = True

    # Export options
    export_chrome_trace: bool = True
    export_tensorboard: bool = True
    export_stacks: bool = False
    export_sqlite: bool = True  # Export to SQLite for analysis


class NVTXAnnotator:
    """NVTX annotation manager for Nsight Systems integration."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._nvtx_available = False
        self._cudart_available = False

        if enabled:
            try:
                import torch.cuda.nvtx as nvtx
                self._nvtx_available = True
            except ImportError:
                logger.debug("NVTX not available - NVIDIA profiling annotations disabled")

            # Check for cuda runtime profiling API
            try:
                self._cudart_available = hasattr(torch.cuda, 'cudart')
            except Exception as e:
                logger.debug(f"CUDA runtime profiling API check failed: {e}")

    @contextmanager
    def range(self, name: str, color: str = "blue"):
        """Create an NVTX range for profiling.

        Args:
            name: Name of the range (visible in Nsight Systems)
            color: Color hint (not all colors supported)
        """
        if self.enabled and self._nvtx_available:
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield

    def mark(self, name: str):
        """Add an NVTX marker (instant event)."""
        if self.enabled and self._nvtx_available:
            # Use range with immediate pop for marker-like behavior
            torch.cuda.nvtx.range_push(name)
            torch.cuda.nvtx.range_pop()


class NsightProfiler:
    """
    Nsight-compatible profiler for PyTorch training.

    Integrates with:
    - Nsight Systems (nsys): System-wide profiling with NVTX annotations
    - Nsight Compute (ncu): Kernel-level analysis
    - PyTorch Profiler: Built-in profiler with TensorBoard export
    """

    def __init__(
        self,
        config: Optional[ProfilerConfig] = None,
        enabled: bool = False,
        output_dir: str = "./profiles",
        profile_steps: Tuple[int, int] = (10, 20),
        use_cuda_nvtx: bool = True,
        use_torch_profiler: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = True,
    ):
        """Initialize the Nsight profiler.

        Args:
            config: ProfilerConfig object (overrides other args if provided)
            enabled: Enable profiling
            output_dir: Directory to save profile outputs
            profile_steps: (start_step, end_step) range to profile
            use_cuda_nvtx: Enable NVTX annotations for Nsight Systems
            use_torch_profiler: Enable PyTorch profiler
            record_shapes: Record tensor shapes
            profile_memory: Profile memory allocations
            with_stack: Record Python stack traces (expensive)
            with_flops: Estimate FLOPS
            with_modules: Record module hierarchy
        """
        if config is not None:
            self.config = config
        else:
            self.config = ProfilerConfig(
                enabled=enabled,
                output_dir=output_dir,
                profile_steps=profile_steps,
                use_cuda_nvtx=use_cuda_nvtx,
                use_torch_profiler=use_torch_profiler,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
            )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nvtx = NVTXAnnotator(enabled=self.config.enabled and self.config.use_cuda_nvtx)
        self._profiler = None
        self._step_count = 0
        self._is_profiling = False
        self._profile_data: List[Dict[str, Any]] = []

        # Create timestamped output directory
        if self.config.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"profile_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Profiler] Output directory: {self.run_dir}")
            logger.info(f"[Profiler] Profile steps: {self.config.profile_steps[0]} - {self.config.profile_steps[1]}")

    def _should_profile(self, step: int) -> bool:
        """Check if current step should be profiled."""
        start, end = self.config.profile_steps
        return start <= step <= end

    def _create_torch_profiler(self):
        """Create PyTorch profiler with appropriate settings."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=self.config.profile_steps[1] - self.config.profile_steps[0],
            repeat=1,
        )

        def trace_handler(prof):
            """Handle profiler output."""
            # Export Chrome trace
            if self.config.export_chrome_trace:
                trace_path = self.run_dir / f"trace_step_{self._step_count}.json"
                prof.export_chrome_trace(str(trace_path))
                logger.info(f"[Profiler] Saved Chrome trace: {trace_path}")

            # Export TensorBoard
            if self.config.export_tensorboard:
                tb_path = self.run_dir / "tensorboard"
                tb_path.mkdir(exist_ok=True)
                prof.export_stacks(str(tb_path / "stacks.txt"), "self_cuda_time_total")

            # Print summary
            print(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))

            # Save key averages as JSON
            self._save_key_averages(prof)

        return torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )

    def _save_key_averages(self, prof):
        """Save key averages to JSON for analysis."""
        averages = prof.key_averages()

        data = []
        for avg in averages:
            entry = {
                "key": avg.key,
                "self_cpu_time_total": avg.self_cpu_time_total,
                "cpu_time_total": avg.cpu_time_total,
                "count": avg.count,
            }
            if torch.cuda.is_available():
                entry.update({
                    "self_cuda_time_total": getattr(avg, 'self_cuda_time_total', 0),
                    "cuda_time_total": getattr(avg, 'cuda_time_total', 0),
                    "cuda_memory_usage": getattr(avg, 'cuda_memory_usage', 0),
                })
            if self.config.with_flops:
                entry["flops"] = getattr(avg, 'flops', 0)
            data.append(entry)

        json_path = self.run_dir / f"key_averages_step_{self._step_count}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"[Profiler] Saved key averages: {json_path}")

    def start(self):
        """Start the profiler session."""
        if not self.config.enabled:
            return

        if self.config.use_torch_profiler:
            self._profiler = self._create_torch_profiler()
            self._profiler.__enter__()

        # Enable CUDA profiling for Nsight - use efficient sync
        if torch.cuda.is_available():
            if _CUDA_STREAMS_AVAILABLE:
                efficient_sync()  # Only syncs current stream
            else:
                torch.cuda.synchronize()

        self._is_profiling = True
        logger.info("[Profiler] Started profiling session")

    def stop(self):
        """Stop the profiler session and save results."""
        if not self._is_profiling:
            return

        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler = None

        # Use efficient sync when stopping profiler
        if torch.cuda.is_available():
            if _CUDA_STREAMS_AVAILABLE:
                efficient_sync()  # Only syncs current stream
            else:
                torch.cuda.synchronize()

        self._is_profiling = False
        logger.info("[Profiler] Stopped profiling session")

        # Save profile summary
        self._save_summary()

    def _save_summary(self):
        """Save profiling summary."""
        summary = {
            "config": {
                "profile_steps": self.config.profile_steps,
                "record_shapes": self.config.record_shapes,
                "profile_memory": self.config.profile_memory,
                "with_flops": self.config.with_flops,
            },
            "total_steps_profiled": self._step_count,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            summary["gpu_info"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            }

        summary_path = self.run_dir / "profile_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[Profiler] Saved summary: {summary_path}")

        # Export nsys-rep to SQLite if enabled and nsys is available
        if self.config.export_sqlite:
            self._export_nsys_to_sqlite()

    def _export_nsys_to_sqlite(self):
        """Export any nsys-rep files in the profile directory to SQLite."""
        # Check if nsys is available
        nsys_path = shutil.which('nsys')
        if not nsys_path:
            logger.debug("[Profiler] nsys not found, skipping SQLite export")
            return

        # Find nsys-rep files in run_dir or parent directories
        nsys_files = list(self.run_dir.glob("*.nsys-rep"))
        if not nsys_files:
            # Check parent (in case nsys output is in profiles folder)
            nsys_files = list(self.run_dir.parent.glob("*.nsys-rep"))

        for nsys_file in nsys_files:
            sqlite_path = nsys_file.with_suffix('.sqlite')
            if sqlite_path.exists():
                logger.info(f"[Profiler] SQLite already exists: {sqlite_path}")
                continue

            try:
                logger.info(f"[Profiler] Exporting {nsys_file.name} to SQLite...")
                result = subprocess.run(
                    [nsys_path, 'export', '-t', 'sqlite', '-o', str(sqlite_path), str(nsys_file)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode == 0:
                    logger.info(f"[Profiler] Exported to SQLite: {sqlite_path}")
                else:
                    logger.warning(f"[Profiler] nsys export failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning("[Profiler] nsys export timed out")
            except Exception as e:
                logger.warning(f"[Profiler] Failed to export to SQLite: {e}")

    @contextmanager
    def step(self, step: int):
        """Context manager for a training step.

        Usage:
            with profiler.step(step):
                loss = model(batch)
                loss.backward()
        """
        if not self.config.enabled:
            yield
            return

        self._step_count = step

        # Always emit NVTX annotations for nsys visibility
        should_profile_torch = self._should_profile(step)
        should_emit_nvtx = self.config.always_emit_nvtx or should_profile_torch

        # Start PyTorch profiler at first profile step
        start_step, end_step = self.config.profile_steps
        if step == start_step and not self._is_profiling and self.config.use_torch_profiler:
            self.start()

        # NVTX range for this step (always emitted if always_emit_nvtx is True)
        if should_emit_nvtx:
            with self.nvtx.range(f"step_{step}"):
                if self._profiler is not None and should_profile_torch:
                    self._profiler.step()
                yield
        else:
            if self._profiler is not None and should_profile_torch:
                self._profiler.step()
            yield

        # Stop PyTorch profiler at last profile step
        if step == end_step and self._is_profiling:
            self.stop()

    @contextmanager
    def range(self, name: str):
        """Create a named range for sub-step profiling.

        Usage:
            with profiler.range("forward_pass"):
                output = model(input)
        """
        with self.nvtx.range(name):
            yield

    def mark(self, name: str):
        """Add a marker at current point."""
        self.nvtx.mark(name)

    def record_metric(self, name: str, value: float):
        """Record a custom metric."""
        self._profile_data.append({
            "step": self._step_count,
            "name": name,
            "value": value,
        })


def create_profiler_from_config(config: Dict[str, Any]) -> NsightProfiler:
    """Create a profiler from a config dictionary.

    Args:
        config: Configuration dictionary with profiling settings

    Returns:
        NsightProfiler instance
    """
    profiling_config = config.get('profiling', {})

    if not profiling_config.get('enabled', False):
        return NsightProfiler(enabled=False)

    return NsightProfiler(
        enabled=True,
        output_dir=profiling_config.get('output_dir', './profiles'),
        profile_steps=tuple(profiling_config.get('profile_steps', [10, 20])),
        use_cuda_nvtx=profiling_config.get('use_cuda_nvtx', True),
        use_torch_profiler=profiling_config.get('use_torch_profiler', True),
        record_shapes=profiling_config.get('record_shapes', True),
        profile_memory=profiling_config.get('profile_memory', True),
        with_stack=profiling_config.get('with_stack', False),
        with_flops=profiling_config.get('with_flops', True),
        with_modules=profiling_config.get('with_modules', True),
    )


def enable_nsight_profiling():
    """Enable CUDA profiling hooks for Nsight Systems/Compute.

    Call this at the start of your script to enable profiling:
        nsys profile python train.py
        ncu python train.py

    This function:
    1. Synchronizes CUDA
    2. Enables profiling API
    3. Clears caches for clean profiling
    """
    if not torch.cuda.is_available():
        logger.warning("[Profiler] CUDA not available, profiling disabled")
        return

    # Synchronize and clear caches - use efficient sync
    if _CUDA_STREAMS_AVAILABLE:
        efficient_sync()  # Only syncs current stream
    else:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Enable cudnn benchmark for consistent kernel selection
    torch.backends.cudnn.benchmark = True

    logger.info("[Profiler] CUDA profiling hooks enabled")
    logger.info("[Profiler] Run with: nsys profile -o output python your_script.py")


def profile_training_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    profiler: NsightProfiler,
    step: int,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """Profile a single training step with detailed annotations.

    This is a reference implementation showing how to instrument
    a training step with NVTX annotations for Nsight profiling.

    Args:
        model: The model to train
        batch: Input batch dictionary
        optimizer: Optimizer
        profiler: NsightProfiler instance
        step: Current training step
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype
        gradient_accumulation_steps: Number of accumulation steps

    Returns:
        Dictionary with step metrics
    """
    with profiler.step(step):
        # Data transfer
        with profiler.range("data_to_gpu"):
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch.get('attention_mask')

        # Forward pass
        with profiler.range("forward"):
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss'] / gradient_accumulation_steps
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / gradient_accumulation_steps

        profiler.mark("forward_complete")

        # Backward pass
        with profiler.range("backward"):
            loss.backward()

        profiler.mark("backward_complete")

        # Optimizer step
        if (step + 1) % gradient_accumulation_steps == 0:
            with profiler.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

        return {
            'loss': loss.item() * gradient_accumulation_steps,
            'step': step,
        }


# =============================================================================
# DETAILED PROFILING COMPONENTS
# =============================================================================


@dataclass
class KernelStats:
    """Statistics for a single CUDA kernel."""
    name: str
    count: int = 0
    total_time_us: float = 0.0
    min_time_us: float = float('inf')
    max_time_us: float = 0.0
    times_us: List[float] = field(default_factory=list)

    # Memory stats
    total_memory_bytes: int = 0
    peak_memory_bytes: int = 0

    # Grid/block dimensions (for compute analysis)
    grid_dims: List[Tuple[int, int, int]] = field(default_factory=list)
    block_dims: List[Tuple[int, int, int]] = field(default_factory=list)

    def add_invocation(self, time_us: float, memory_bytes: int = 0,
                       grid: Tuple[int, int, int] = None,
                       block: Tuple[int, int, int] = None):
        """Record a kernel invocation."""
        self.count += 1
        self.total_time_us += time_us
        self.min_time_us = min(self.min_time_us, time_us)
        self.max_time_us = max(self.max_time_us, time_us)
        self.times_us.append(time_us)
        self.total_memory_bytes += memory_bytes
        self.peak_memory_bytes = max(self.peak_memory_bytes, memory_bytes)
        if grid:
            self.grid_dims.append(grid)
        if block:
            self.block_dims.append(block)

    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / self.count if self.count > 0 else 0

    @property
    def std_time_us(self) -> float:
        if len(self.times_us) < 2:
            return 0.0
        return statistics.stdev(self.times_us)

    @property
    def median_time_us(self) -> float:
        if not self.times_us:
            return 0.0
        return statistics.median(self.times_us)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_time_us": self.total_time_us,
            "avg_time_us": self.avg_time_us,
            "min_time_us": self.min_time_us if self.min_time_us != float('inf') else 0,
            "max_time_us": self.max_time_us,
            "std_time_us": self.std_time_us,
            "median_time_us": self.median_time_us,
            "total_memory_bytes": self.total_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass
class LayerProfile:
    """Profiling data for a single layer."""
    name: str
    layer_type: str

    # Timing
    forward_times_ms: List[float] = field(default_factory=list)
    backward_times_ms: List[float] = field(default_factory=list)

    # Memory
    activation_memory_mb: List[float] = field(default_factory=list)
    gradient_memory_mb: List[float] = field(default_factory=list)
    parameter_memory_mb: float = 0.0

    # Tensor shapes
    input_shapes: List[Tuple] = field(default_factory=list)
    output_shapes: List[Tuple] = field(default_factory=list)

    # FLOPS
    flops_per_forward: int = 0

    def add_forward(self, time_ms: float, activation_mb: float = 0,
                    input_shape: Tuple = None, output_shape: Tuple = None):
        self.forward_times_ms.append(time_ms)
        if activation_mb > 0:
            self.activation_memory_mb.append(activation_mb)
        if input_shape:
            self.input_shapes.append(input_shape)
        if output_shape:
            self.output_shapes.append(output_shape)

    def add_backward(self, time_ms: float, gradient_mb: float = 0):
        self.backward_times_ms.append(time_ms)
        if gradient_mb > 0:
            self.gradient_memory_mb.append(gradient_mb)

    def get_summary(self) -> Dict[str, Any]:
        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "std": 0, "total": 0}
            return {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "total": sum(values),
            }

        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "forward_ms": safe_stats(self.forward_times_ms),
            "backward_ms": safe_stats(self.backward_times_ms),
            "activation_memory_mb": safe_stats(self.activation_memory_mb),
            "gradient_memory_mb": safe_stats(self.gradient_memory_mb),
            "parameter_memory_mb": self.parameter_memory_mb,
            "flops_per_forward": self.flops_per_forward,
            "num_calls": len(self.forward_times_ms),
        }


@dataclass
class MoERoutingStats:
    """Statistics for MoE expert routing."""
    step: int
    num_experts: int
    top_k: int

    # Per-expert load
    expert_loads: List[int] = field(default_factory=list)  # tokens per expert
    expert_utilization: List[float] = field(default_factory=list)  # percentage

    # Routing quality metrics
    load_balance_loss: float = 0.0
    router_z_loss: float = 0.0
    auxiliary_loss: float = 0.0

    # Expert timing
    expert_compute_times_ms: List[float] = field(default_factory=list)
    routing_time_ms: float = 0.0
    combine_time_ms: float = 0.0

    # Token distribution
    tokens_per_expert_min: int = 0
    tokens_per_expert_max: int = 0
    tokens_per_expert_std: float = 0.0
    dropped_tokens: int = 0

    # Enhanced MoE metrics
    router_confidence: float = 0.0  # Average confidence of routing decisions
    expert_diversity: float = 0.0  # Entropy of expert selection
    capacity_factor: float = 1.0  # Expert capacity utilization
    expert_overlap: List[float] = field(default_factory=list)  # How often experts are co-selected
    cache_hit_rate: float = 0.0  # For cached routing decisions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_loads": self.expert_loads,
            "expert_utilization": self.expert_utilization,
            "load_balance_loss": self.load_balance_loss,
            "router_z_loss": self.router_z_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "expert_compute_times_ms": self.expert_compute_times_ms,
            "routing_time_ms": self.routing_time_ms,
            "combine_time_ms": self.combine_time_ms,
            "tokens_per_expert_min": self.tokens_per_expert_min,
            "tokens_per_expert_max": self.tokens_per_expert_max,
            "tokens_per_expert_std": self.tokens_per_expert_std,
            "dropped_tokens": self.dropped_tokens,
            "router_confidence": self.router_confidence,
            "expert_diversity": self.expert_diversity,
            "capacity_factor": self.capacity_factor,
            "expert_overlap": self.expert_overlap,
            "cache_hit_rate": self.cache_hit_rate,
        }


@dataclass
class AttentionStats:
    """Statistics for attention operations."""
    step: int
    layer_idx: int

    # Timing
    attention_time_ms: float = 0.0
    qkv_projection_time_ms: float = 0.0
    softmax_time_ms: float = 0.0
    output_projection_time_ms: float = 0.0

    # Memory
    kv_cache_size_mb: float = 0.0
    attention_scores_size_mb: float = 0.0

    # Shape information
    batch_size: int = 0
    seq_length: int = 0
    num_heads: int = 0
    head_dim: int = 0
    num_kv_heads: int = 0  # For GQA/MQA

    # Attention pattern metrics
    avg_attention_entropy: float = 0.0  # How distributed attention is
    max_attention_score: float = 0.0
    sparsity_ratio: float = 0.0  # Fraction of near-zero attention weights

    # Flash attention specific
    using_flash_attention: bool = False
    flash_attention_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "layer_idx": self.layer_idx,
            "attention_time_ms": self.attention_time_ms,
            "qkv_projection_time_ms": self.qkv_projection_time_ms,
            "softmax_time_ms": self.softmax_time_ms,
            "output_projection_time_ms": self.output_projection_time_ms,
            "kv_cache_size_mb": self.kv_cache_size_mb,
            "attention_scores_size_mb": self.attention_scores_size_mb,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "avg_attention_entropy": self.avg_attention_entropy,
            "max_attention_score": self.max_attention_score,
            "sparsity_ratio": self.sparsity_ratio,
            "using_flash_attention": self.using_flash_attention,
            "flash_attention_version": self.flash_attention_version,
        }


@dataclass
class CommunicationStats:
    """Statistics for distributed communication operations."""
    step: int
    operation: str  # all_reduce, all_gather, reduce_scatter, etc.

    # Timing
    time_ms: float = 0.0
    start_timestamp: float = 0.0

    # Data transfer
    bytes_transferred: int = 0
    tensor_shape: Tuple = ()
    tensor_dtype: str = ""

    # Topology
    world_size: int = 1
    local_rank: int = 0
    process_group: str = ""

    # Performance
    bandwidth_gbps: float = 0.0
    latency_ms: float = 0.0

    # Overlap
    overlapped_with_compute: bool = False
    overlap_efficiency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "operation": self.operation,
            "time_ms": self.time_ms,
            "bytes_transferred": self.bytes_transferred,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_ms": self.latency_ms,
            "overlapped_with_compute": self.overlapped_with_compute,
            "overlap_efficiency": self.overlap_efficiency,
        }


@dataclass
class GradientStats:
    """Statistics for gradient analysis."""
    step: int

    # Global gradient norms
    total_grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    min_grad_norm: float = 0.0

    # Per-layer gradient info
    layer_grad_norms: Dict[str, float] = field(default_factory=dict)

    # Gradient health metrics
    num_nan_grads: int = 0
    num_inf_grads: int = 0
    num_zero_grads: int = 0

    # Gradient flow analysis
    grad_norm_ratios: List[float] = field(default_factory=list)  # Layer-to-layer ratio
    vanishing_layers: List[str] = field(default_factory=list)  # Layers with very small grads
    exploding_layers: List[str] = field(default_factory=list)  # Layers with very large grads

    # Gradient statistics per parameter type
    embedding_grad_norm: float = 0.0
    attention_grad_norm: float = 0.0
    ffn_grad_norm: float = 0.0
    moe_grad_norm: float = 0.0
    norm_layer_grad_norm: float = 0.0

    # Gradient clipping info
    was_clipped: bool = False
    original_norm: float = 0.0
    clipped_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "total_grad_norm": self.total_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "min_grad_norm": self.min_grad_norm,
            "layer_grad_norms": self.layer_grad_norms,
            "num_nan_grads": self.num_nan_grads,
            "num_inf_grads": self.num_inf_grads,
            "num_zero_grads": self.num_zero_grads,
            "vanishing_layers": self.vanishing_layers,
            "exploding_layers": self.exploding_layers,
            "embedding_grad_norm": self.embedding_grad_norm,
            "attention_grad_norm": self.attention_grad_norm,
            "ffn_grad_norm": self.ffn_grad_norm,
            "moe_grad_norm": self.moe_grad_norm,
            "norm_layer_grad_norm": self.norm_layer_grad_norm,
            "was_clipped": self.was_clipped,
            "original_norm": self.original_norm,
            "clipped_norm": self.clipped_norm,
        }


@dataclass
class MemoryFragmentationStats:
    """Statistics for memory fragmentation analysis."""
    step: int
    timestamp: float = 0.0

    # Memory pool statistics
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0  # reserved - allocated

    # Fragmentation metrics
    fragmentation_ratio: float = 0.0  # free / reserved
    num_free_blocks: int = 0
    largest_free_block_mb: float = 0.0
    smallest_free_block_mb: float = 0.0

    # Allocation patterns
    num_allocations: int = 0
    num_deallocations: int = 0
    allocation_sizes_mb: List[float] = field(default_factory=list)

    # CUDA memory allocator stats
    num_ooms: int = 0
    num_retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "free_mb": self.free_mb,
            "fragmentation_ratio": self.fragmentation_ratio,
            "num_free_blocks": self.num_free_blocks,
            "largest_free_block_mb": self.largest_free_block_mb,
            "num_allocations": self.num_allocations,
            "num_deallocations": self.num_deallocations,
            "num_ooms": self.num_ooms,
            "num_retries": self.num_retries,
        }


@dataclass
class TensorStats:
    """Statistics for tensor shapes and dtypes."""
    name: str
    shapes: List[Tuple] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)

    # Memory
    sizes_mb: List[float] = field(default_factory=list)
    total_elements: List[int] = field(default_factory=list)

    # Statistics
    min_vals: List[float] = field(default_factory=list)
    max_vals: List[float] = field(default_factory=list)
    mean_vals: List[float] = field(default_factory=list)
    std_vals: List[float] = field(default_factory=list)
    num_nans: List[int] = field(default_factory=list)
    num_infs: List[int] = field(default_factory=list)

    def add_observation(self, tensor: torch.Tensor, compute_stats: bool = False):
        """Record a tensor observation."""
        self.shapes.append(tuple(tensor.shape))
        self.dtypes.append(str(tensor.dtype))
        self.devices.append(str(tensor.device))
        self.sizes_mb.append(tensor.numel() * tensor.element_size() / 1024**2)
        self.total_elements.append(tensor.numel())

        if compute_stats and tensor.numel() > 0:
            with torch.no_grad():
                flat = tensor.float().flatten()
                # GPU SYNC FIX: Batch all statistics computation on GPU, single .tolist() sync
                # instead of 6 separate .item() calls (reduces 6 cudaStreamSynchronize to 1)
                stats = torch.stack([
                    flat.min(),
                    flat.max(),
                    flat.mean(),
                    flat.std(),
                    torch.isnan(tensor).sum().float(),
                    torch.isinf(tensor).sum().float(),
                ]).tolist()
                self.min_vals.append(stats[0])
                self.max_vals.append(stats[1])
                self.mean_vals.append(stats[2])
                self.std_vals.append(stats[3])
                self.num_nans.append(int(stats[4]))
                self.num_infs.append(int(stats[5]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_observations": len(self.shapes),
            "shapes": self.shapes[-5:],  # Last 5 shapes
            "dtypes": list(set(self.dtypes)),
            "devices": list(set(self.devices)),
            "avg_size_mb": statistics.mean(self.sizes_mb) if self.sizes_mb else 0,
            "total_size_mb": sum(self.sizes_mb),
            "value_stats": {
                "min": min(self.min_vals) if self.min_vals else None,
                "max": max(self.max_vals) if self.max_vals else None,
                "mean": statistics.mean(self.mean_vals) if self.mean_vals else None,
                "total_nans": sum(self.num_nans),
                "total_infs": sum(self.num_infs),
            } if self.min_vals else None,
        }


@dataclass
class Bottleneck:
    """Represents a detected performance bottleneck."""
    category: str  # compute, memory, communication, io, etc.
    severity: str  # critical, high, medium, low
    description: str
    impact_ms: float = 0.0  # Estimated time impact
    recommendation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "impact_ms": self.impact_ms,
            "recommendation": self.recommendation,
            "details": self.details,
        }


class DetailedProfiler:
    """
    Enhanced profiler with detailed kernel, layer, and MoE statistics.

    Provides comprehensive profiling output including:
    - Per-kernel timing and memory statistics
    - Layer-by-layer breakdown with activation tracking
    - MoE expert routing analysis with load balance metrics
    - Memory timeline and fragmentation tracking
    - Attention pattern profiling (sequence length, head utilization)
    - Communication overhead tracking for distributed training
    - Tensor shape and dtype statistics
    - Gradient flow analysis
    - Automatic bottleneck detection with recommendations
    - Detailed formatted reports and HTML export
    """

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "./profiles",
        track_kernels: bool = True,
        track_layers: bool = True,
        track_moe: bool = True,
        track_memory_timeline: bool = True,
        track_attention: bool = True,
        track_communication: bool = True,
        track_gradients: bool = True,
        track_tensors: bool = False,  # Expensive - disabled by default
        track_fragmentation: bool = True,
        detect_bottlenecks: bool = True,
        print_step_summary: bool = True,
        print_final_report: bool = True,
        verbose: bool = False,
        step_summary_interval: int = 1,  # Print summary every N steps
        auto_detect_distributed: bool = True,
    ):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self.track_kernels = track_kernels
        self.track_layers = track_layers
        self.track_moe = track_moe
        self.track_memory_timeline = track_memory_timeline
        self.track_attention = track_attention
        self.track_communication = track_communication
        self.track_gradients = track_gradients
        self.track_tensors = track_tensors
        self.track_fragmentation = track_fragmentation
        self.detect_bottlenecks = detect_bottlenecks
        self.print_step_summary = print_step_summary
        self.print_final_report = print_final_report
        self.verbose = verbose
        self.step_summary_interval = step_summary_interval

        # Statistics storage - basic
        self.kernel_stats: Dict[str, KernelStats] = {}
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.moe_routing_stats: List[MoERoutingStats] = []
        self.memory_timeline: List[Dict[str, Any]] = []

        # Statistics storage - enhanced
        self.attention_stats: List[AttentionStats] = []
        self.communication_stats: List[CommunicationStats] = []
        self.gradient_stats: List[GradientStats] = []
        self.tensor_stats: Dict[str, TensorStats] = {}
        self.fragmentation_stats: List[MemoryFragmentationStats] = []
        self.bottlenecks: List[Bottleneck] = []

        # Step-level metrics
        self.step_metrics: List[Dict[str, Any]] = []
        self.current_step = 0
        self._step_start_time = 0.0

        # Phase timings within a step
        self._phase_timings: Dict[str, List[float]] = defaultdict(list)
        self._active_phases: Dict[str, float] = {}
        self._phase_memory_start: Dict[str, float] = {}

        # NVTX for Nsight integration
        self.nvtx = NVTXAnnotator(enabled=enabled)

        # CUDA events for precise timing
        self._cuda_events: Dict[str, Tuple] = {}
        self._cuda_event_pool: List[torch.cuda.Event] = []

        # Distributed training info
        self._world_size = 1
        self._local_rank = 0
        self._is_distributed = False
        if auto_detect_distributed:
            self._detect_distributed()

        # Compute utilization tracking
        self._compute_start_time = 0.0
        self._total_compute_time = 0.0
        self._total_idle_time = 0.0

        # Initialize run directory
        self.run_dir = None
        if enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"detailed_profile_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DetailedProfiler] Output directory: {self.run_dir}")

    def _detect_distributed(self):
        """Auto-detect distributed training configuration."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self._is_distributed = True
                self._world_size = dist.get_world_size()
                self._local_rank = dist.get_rank()
        except Exception as e:
            logger.debug(f"Distributed environment check failed: {e}")

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }

    def start_phase(self, name: str):
        """Start timing a phase (forward, backward, optimizer, etc.)."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._active_phases[name] = time.perf_counter()

        if self.nvtx._nvtx_available:
            torch.cuda.nvtx.range_push(name)

    def end_phase(self, name: str):
        """End timing a phase."""
        if not self.enabled or name not in self._active_phases:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - self._active_phases[name]) * 1000
        self._phase_timings[name].append(elapsed_ms)
        del self._active_phases[name]

        if self.nvtx._nvtx_available:
            torch.cuda.nvtx.range_pop()

    @contextmanager
    def phase(self, name: str):
        """Context manager for timing a phase."""
        self.start_phase(name)
        try:
            yield
        finally:
            self.end_phase(name)

    @contextmanager
    def step(self, step: int):
        """Profile a complete training step."""
        if not self.enabled:
            yield
            return

        self.current_step = step
        self._step_start_time = time.perf_counter()

        # Record memory at start
        start_memory = self._get_memory_stats()

        with self.nvtx.range(f"step_{step}"):
            yield

        # Record memory at end
        end_memory = self._get_memory_stats()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time_ms = (time.perf_counter() - self._step_start_time) * 1000

        # Compile step metrics
        step_data = {
            "step": step,
            "total_time_ms": step_time_ms,
            "phases": {k: v[-1] if v else 0 for k, v in self._phase_timings.items()},
            "memory_start": start_memory,
            "memory_end": end_memory,
            "memory_delta_mb": (
                end_memory.get("allocated_mb", 0) - start_memory.get("allocated_mb", 0)
            ),
        }
        self.step_metrics.append(step_data)

        # Track memory timeline
        if self.track_memory_timeline:
            self.memory_timeline.append({
                "step": step,
                "timestamp": time.time(),
                **end_memory,
            })

        # Print step summary if enabled
        if self.print_step_summary:
            self._print_step_summary(step_data)

    def record_kernel(
        self,
        name: str,
        time_us: float,
        memory_bytes: int = 0,
        grid: Tuple[int, int, int] = None,
        block: Tuple[int, int, int] = None,
    ):
        """Record a CUDA kernel execution."""
        if not self.enabled or not self.track_kernels:
            return

        if name not in self.kernel_stats:
            self.kernel_stats[name] = KernelStats(name=name)

        self.kernel_stats[name].add_invocation(time_us, memory_bytes, grid, block)

    def record_layer(
        self,
        name: str,
        layer_type: str,
        forward_time_ms: float = 0,
        backward_time_ms: float = 0,
        activation_mb: float = 0,
        gradient_mb: float = 0,
        input_shape: Tuple = None,
        output_shape: Tuple = None,
    ):
        """Record layer-level profiling data."""
        if not self.enabled or not self.track_layers:
            return

        if name not in self.layer_profiles:
            self.layer_profiles[name] = LayerProfile(name=name, layer_type=layer_type)

        profile = self.layer_profiles[name]
        if forward_time_ms > 0:
            profile.add_forward(forward_time_ms, activation_mb, input_shape, output_shape)
        if backward_time_ms > 0:
            profile.add_backward(backward_time_ms, gradient_mb)

    def record_moe_routing(
        self,
        num_experts: int,
        top_k: int,
        expert_loads: List[int],
        load_balance_loss: float = 0,
        router_z_loss: float = 0,
        routing_time_ms: float = 0,
        expert_times_ms: List[float] = None,
        dropped_tokens: int = 0,
        router_confidence: float = 0.0,
        expert_diversity: float = 0.0,
        capacity_factor: float = 1.0,
        expert_overlap: List[float] = None,
        cache_hit_rate: float = 0.0,
    ):
        """Record MoE routing statistics with enhanced metrics."""
        if not self.enabled or not self.track_moe:
            return

        total_tokens = sum(expert_loads) if expert_loads else 0
        utilization = [
            load / (total_tokens / num_experts) if total_tokens > 0 else 0
            for load in expert_loads
        ]

        stats = MoERoutingStats(
            step=self.current_step,
            num_experts=num_experts,
            top_k=top_k,
            expert_loads=expert_loads,
            expert_utilization=utilization,
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            routing_time_ms=routing_time_ms,
            expert_compute_times_ms=expert_times_ms or [],
            tokens_per_expert_min=min(expert_loads) if expert_loads else 0,
            tokens_per_expert_max=max(expert_loads) if expert_loads else 0,
            tokens_per_expert_std=statistics.stdev(expert_loads) if len(expert_loads) > 1 else 0,
            dropped_tokens=dropped_tokens,
            router_confidence=router_confidence,
            expert_diversity=expert_diversity,
            capacity_factor=capacity_factor,
            expert_overlap=expert_overlap or [],
            cache_hit_rate=cache_hit_rate,
        )
        self.moe_routing_stats.append(stats)

    def record_attention(
        self,
        layer_idx: int,
        attention_time_ms: float = 0.0,
        qkv_projection_time_ms: float = 0.0,
        softmax_time_ms: float = 0.0,
        output_projection_time_ms: float = 0.0,
        kv_cache_size_mb: float = 0.0,
        batch_size: int = 0,
        seq_length: int = 0,
        num_heads: int = 0,
        head_dim: int = 0,
        num_kv_heads: int = 0,
        avg_attention_entropy: float = 0.0,
        max_attention_score: float = 0.0,
        sparsity_ratio: float = 0.0,
        using_flash_attention: bool = False,
        flash_attention_version: str = "",
    ):
        """Record attention layer statistics."""
        if not self.enabled or not self.track_attention:
            return

        # Calculate attention scores memory
        attention_scores_size_mb = 0.0
        if batch_size > 0 and num_heads > 0 and seq_length > 0:
            # Attention scores: [batch, heads, seq, seq]
            attention_scores_size_mb = (batch_size * num_heads * seq_length * seq_length * 4) / 1024**2

        stats = AttentionStats(
            step=self.current_step,
            layer_idx=layer_idx,
            attention_time_ms=attention_time_ms,
            qkv_projection_time_ms=qkv_projection_time_ms,
            softmax_time_ms=softmax_time_ms,
            output_projection_time_ms=output_projection_time_ms,
            kv_cache_size_mb=kv_cache_size_mb,
            attention_scores_size_mb=attention_scores_size_mb,
            batch_size=batch_size,
            seq_length=seq_length,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            avg_attention_entropy=avg_attention_entropy,
            max_attention_score=max_attention_score,
            sparsity_ratio=sparsity_ratio,
            using_flash_attention=using_flash_attention,
            flash_attention_version=flash_attention_version,
        )
        self.attention_stats.append(stats)

    def record_communication(
        self,
        operation: str,
        time_ms: float,
        bytes_transferred: int,
        tensor_shape: Tuple = (),
        tensor_dtype: str = "",
        overlapped_with_compute: bool = False,
        overlap_efficiency: float = 0.0,
    ):
        """Record distributed communication statistics."""
        if not self.enabled or not self.track_communication:
            return

        # Calculate bandwidth
        bandwidth_gbps = 0.0
        if time_ms > 0 and bytes_transferred > 0:
            bandwidth_gbps = (bytes_transferred * 8 / 1e9) / (time_ms / 1000)

        stats = CommunicationStats(
            step=self.current_step,
            operation=operation,
            time_ms=time_ms,
            start_timestamp=time.time(),
            bytes_transferred=bytes_transferred,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            world_size=self._world_size,
            local_rank=self._local_rank,
            bandwidth_gbps=bandwidth_gbps,
            latency_ms=time_ms if bytes_transferred < 1024 else 0,  # Estimate latency for small transfers
            overlapped_with_compute=overlapped_with_compute,
            overlap_efficiency=overlap_efficiency,
        )
        self.communication_stats.append(stats)

    def record_gradients(
        self,
        model: torch.nn.Module,
        grad_clip_value: float = None,
    ):
        """Record gradient statistics from model parameters.

        GPU SYNC FIX: Batches all gradient statistics on GPU and syncs once,
        instead of calling .item() and .any() per parameter (N syncs → 1 sync).
        """
        if not self.enabled or not self.track_gradients:
            return

        stats = GradientStats(step=self.current_step)

        # GPU SYNC FIX: Collect all tensors on GPU, sync once at end
        param_names = []
        norm_tensors = []
        nan_tensors = []
        inf_tensors = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()
            param_names.append(name)
            # Accumulate on GPU - NO .item() or .any() calls here
            norm_tensors.append(grad.norm())
            nan_tensors.append(torch.isnan(grad).any().float())
            inf_tensors.append(torch.isinf(grad).any().float())

        if not norm_tensors:
            self.gradient_stats.append(stats)
            return

        # GPU SYNC FIX: Single sync - stack all and transfer with .tolist()
        all_norms_tensor = torch.stack(norm_tensors)
        all_nans_tensor = torch.stack(nan_tensors)
        all_infs_tensor = torch.stack(inf_tensors)

        # Combine into single transfer
        combined = torch.cat([
            all_norms_tensor,
            all_nans_tensor,
            all_infs_tensor,
        ]).tolist()

        n = len(param_names)
        all_norms = combined[:n]
        nan_flags = combined[n:2*n]
        inf_flags = combined[2*n:]

        # Build layer_norms dict and category lists
        layer_norms = {}
        embedding_norms = []
        attention_norms = []
        ffn_norms = []
        moe_norms = []
        norm_layer_norms = []

        num_nan = sum(1 for f in nan_flags if f > 0)
        num_inf = sum(1 for f in inf_flags if f > 0)
        num_zero = sum(1 for n in all_norms if n == 0)

        for i, name in enumerate(param_names):
            grad_norm = all_norms[i]
            layer_norms[name] = grad_norm

            # Categorize gradients
            name_lower = name.lower()
            if 'embed' in name_lower:
                embedding_norms.append(grad_norm)
            elif 'attention' in name_lower or 'attn' in name_lower:
                attention_norms.append(grad_norm)
            elif 'expert' in name_lower or 'moe' in name_lower:
                moe_norms.append(grad_norm)
            elif 'ffn' in name_lower or 'mlp' in name_lower or 'fc' in name_lower:
                ffn_norms.append(grad_norm)
            elif 'norm' in name_lower or 'ln' in name_lower:
                norm_layer_norms.append(grad_norm)

        if all_norms:
            stats.total_grad_norm = sum(n**2 for n in all_norms) ** 0.5
            stats.max_grad_norm = max(all_norms)
            stats.min_grad_norm = min(all_norms)
            stats.layer_grad_norms = layer_norms
            stats.num_nan_grads = num_nan
            stats.num_inf_grads = num_inf
            stats.num_zero_grads = num_zero

            # Per-category norms
            stats.embedding_grad_norm = sum(n**2 for n in embedding_norms) ** 0.5 if embedding_norms else 0
            stats.attention_grad_norm = sum(n**2 for n in attention_norms) ** 0.5 if attention_norms else 0
            stats.ffn_grad_norm = sum(n**2 for n in ffn_norms) ** 0.5 if ffn_norms else 0
            stats.moe_grad_norm = sum(n**2 for n in moe_norms) ** 0.5 if moe_norms else 0
            stats.norm_layer_grad_norm = sum(n**2 for n in norm_layer_norms) ** 0.5 if norm_layer_norms else 0

            # Detect vanishing/exploding gradients
            threshold_vanish = 1e-7
            threshold_explode = 100.0
            stats.vanishing_layers = [n for n, v in layer_norms.items() if v < threshold_vanish]
            stats.exploding_layers = [n for n, v in layer_norms.items() if v > threshold_explode]

            # Gradient clipping info
            if grad_clip_value is not None:
                stats.original_norm = stats.total_grad_norm
                stats.was_clipped = stats.total_grad_norm > grad_clip_value
                stats.clipped_norm = min(stats.total_grad_norm, grad_clip_value)

        self.gradient_stats.append(stats)

    def record_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        compute_stats: bool = False,
    ):
        """Record tensor shape and statistics."""
        if not self.enabled or not self.track_tensors:
            return

        if name not in self.tensor_stats:
            self.tensor_stats[name] = TensorStats(name=name)

        self.tensor_stats[name].add_observation(tensor, compute_stats)

    def record_memory_fragmentation(self):
        """Record memory fragmentation statistics."""
        if not self.enabled or not self.track_fragmentation:
            return

        if not torch.cuda.is_available():
            return

        stats = MemoryFragmentationStats(
            step=self.current_step,
            timestamp=time.time(),
        )

        # Basic memory stats
        stats.allocated_mb = torch.cuda.memory_allocated() / 1024**2
        stats.reserved_mb = torch.cuda.memory_reserved() / 1024**2
        stats.free_mb = stats.reserved_mb - stats.allocated_mb
        stats.fragmentation_ratio = stats.free_mb / stats.reserved_mb if stats.reserved_mb > 0 else 0

        # Try to get detailed memory stats
        try:
            memory_stats = torch.cuda.memory_stats()
            stats.num_allocations = memory_stats.get('allocation.all.current', 0)
            stats.num_ooms = memory_stats.get('num_ooms', 0)
            stats.num_retries = memory_stats.get('num_alloc_retries', 0)
        except Exception as e:
            logger.debug(f"Could not get detailed memory stats: {e}")

        self.fragmentation_stats.append(stats)

    def _detect_bottlenecks(self):
        """Analyze profiling data and detect performance bottlenecks."""
        if not self.detect_bottlenecks:
            return

        self.bottlenecks.clear()

        # Check for memory bottlenecks
        if self.memory_timeline:
            peak_mem = max(m.get('max_allocated_mb', 0) for m in self.memory_timeline)
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
                utilization = peak_mem / total_mem
                if utilization > 0.95:
                    self.bottlenecks.append(Bottleneck(
                        category="memory",
                        severity="critical",
                        description=f"Memory utilization at {utilization*100:.1f}% - risk of OOM",
                        recommendation="Enable gradient checkpointing, reduce batch size, or use model parallelism",
                        details={"peak_mb": peak_mem, "total_mb": total_mem, "utilization": utilization},
                    ))
                elif utilization > 0.85:
                    self.bottlenecks.append(Bottleneck(
                        category="memory",
                        severity="high",
                        description=f"Memory utilization at {utilization*100:.1f}%",
                        recommendation="Consider gradient checkpointing or reducing batch size for headroom",
                        details={"peak_mb": peak_mem, "total_mb": total_mem, "utilization": utilization},
                    ))

        # Check for MoE load imbalance
        if self.moe_routing_stats:
            recent_stats = self.moe_routing_stats[-10:]
            avg_std = statistics.mean(s.tokens_per_expert_std for s in recent_stats)
            avg_load = statistics.mean(statistics.mean(s.expert_loads) for s in recent_stats if s.expert_loads)
            imbalance = avg_std / avg_load if avg_load > 0 else 0

            if imbalance > 0.5:
                self.bottlenecks.append(Bottleneck(
                    category="moe",
                    severity="high",
                    description=f"MoE expert load imbalance: {imbalance:.2f} coefficient of variation",
                    recommendation="Increase load balance loss coefficient or adjust expert capacity",
                    details={"avg_std": avg_std, "avg_load": avg_load, "imbalance": imbalance},
                ))

            total_dropped = sum(s.dropped_tokens for s in recent_stats)
            if total_dropped > 0:
                self.bottlenecks.append(Bottleneck(
                    category="moe",
                    severity="medium",
                    description=f"Dropped {total_dropped} tokens in recent steps",
                    recommendation="Increase expert capacity factor or number of experts",
                    details={"dropped_tokens": total_dropped},
                ))

        # Check for gradient issues
        if self.gradient_stats:
            recent_grads = self.gradient_stats[-10:]
            nan_count = sum(g.num_nan_grads for g in recent_grads)
            inf_count = sum(g.num_inf_grads for g in recent_grads)

            if nan_count > 0:
                self.bottlenecks.append(Bottleneck(
                    category="gradient",
                    severity="critical",
                    description=f"NaN gradients detected ({nan_count} occurrences)",
                    recommendation="Check learning rate, loss scaling, or numerical stability",
                    details={"nan_count": nan_count},
                ))

            if inf_count > 0:
                self.bottlenecks.append(Bottleneck(
                    category="gradient",
                    severity="critical",
                    description=f"Inf gradients detected ({inf_count} occurrences)",
                    recommendation="Enable gradient clipping or reduce learning rate",
                    details={"inf_count": inf_count},
                ))

            # Check for vanishing gradients
            vanishing = set()
            for g in recent_grads:
                vanishing.update(g.vanishing_layers)
            if len(vanishing) > 3:
                self.bottlenecks.append(Bottleneck(
                    category="gradient",
                    severity="medium",
                    description=f"Vanishing gradients in {len(vanishing)} layers",
                    recommendation="Check initialization, consider skip connections, or adjust learning rate",
                    details={"layers": list(vanishing)[:5]},
                ))

        # Check for communication overhead
        if self.communication_stats and self._is_distributed:
            total_comm_time = sum(c.time_ms for c in self.communication_stats)
            total_step_time = sum(s.get('total_time_ms', 0) for s in self.step_metrics) if self.step_metrics else 1
            comm_ratio = total_comm_time / total_step_time if total_step_time > 0 else 0

            if comm_ratio > 0.3:
                self.bottlenecks.append(Bottleneck(
                    category="communication",
                    severity="high",
                    description=f"Communication overhead is {comm_ratio*100:.1f}% of step time",
                    recommendation="Enable gradient bucketing, overlap comm/compute, or reduce world size",
                    details={"comm_time_ms": total_comm_time, "step_time_ms": total_step_time, "ratio": comm_ratio},
                ))

        # Check for memory fragmentation
        if self.fragmentation_stats:
            recent_frag = self.fragmentation_stats[-5:]
            avg_frag = statistics.mean(f.fragmentation_ratio for f in recent_frag)
            if avg_frag > 0.3:
                self.bottlenecks.append(Bottleneck(
                    category="memory",
                    severity="medium",
                    description=f"Memory fragmentation at {avg_frag*100:.1f}%",
                    recommendation="Call torch.cuda.empty_cache() periodically or use memory pools",
                    details={"fragmentation_ratio": avg_frag},
                ))

        # Check for attention memory usage
        if self.attention_stats:
            recent_attn = self.attention_stats[-5:]
            avg_kv_cache = statistics.mean(a.kv_cache_size_mb for a in recent_attn)
            if avg_kv_cache > 1000:  # > 1GB
                self.bottlenecks.append(Bottleneck(
                    category="attention",
                    severity="medium",
                    description=f"KV cache using {avg_kv_cache:.0f} MB",
                    recommendation="Enable sliding window attention, reduce context length, or use GQA/MQA",
                    details={"kv_cache_mb": avg_kv_cache},
                ))

            # Check if not using flash attention
            not_flash = [a for a in recent_attn if not a.using_flash_attention]
            if not_flash:
                self.bottlenecks.append(Bottleneck(
                    category="attention",
                    severity="medium",
                    description="Flash Attention not detected",
                    recommendation="Enable Flash Attention for 40% memory savings and faster attention",
                    details={},
                ))

    def _print_step_summary(self, step_data: Dict[str, Any]):
        """Print a formatted step summary."""
        step = step_data["step"]
        total_ms = step_data["total_time_ms"]
        phases = step_data["phases"]
        mem_delta = step_data["memory_delta_mb"]

        print(f"\n{'='*70}")
        print(f"  STEP {step} SUMMARY")
        print(f"{'='*70}")
        print(f"  Total Time: {total_ms:.2f} ms")
        print(f"  Memory Delta: {mem_delta:+.2f} MB")

        if phases:
            print(f"\n  Phase Breakdown:")
            print(f"  {'-'*40}")
            for phase, time_ms in sorted(phases.items(), key=lambda x: -x[1]):
                pct = (time_ms / total_ms * 100) if total_ms > 0 else 0
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {phase:<20} {time_ms:>8.2f} ms ({pct:>5.1f}%) {bar}")

        mem_end = step_data["memory_end"]
        if mem_end:
            print(f"\n  Memory Status:")
            print(f"  {'-'*40}")
            print(f"  Allocated:     {mem_end.get('allocated_mb', 0):>8.1f} MB")
            print(f"  Reserved:      {mem_end.get('reserved_mb', 0):>8.1f} MB")
            print(f"  Peak Allocated:{mem_end.get('max_allocated_mb', 0):>8.1f} MB")

        print(f"{'='*70}\n")

    def print_kernel_report(self, top_n: int = 30):
        """Print detailed kernel statistics report."""
        if not self.kernel_stats:
            print("No kernel statistics recorded.")
            return

        print(f"\n{'='*100}")
        print(f"  CUDA KERNEL ANALYSIS REPORT")
        print(f"{'='*100}")

        # Sort by total time
        sorted_kernels = sorted(
            self.kernel_stats.values(),
            key=lambda k: k.total_time_us,
            reverse=True
        )[:top_n]

        total_time = sum(k.total_time_us for k in self.kernel_stats.values())

        print(f"\n  Top {top_n} Kernels by Total Time (Total: {total_time/1000:.2f} ms)")
        print(f"  {'-'*96}")
        print(f"  {'Kernel Name':<45} {'Count':>8} {'Total(ms)':>10} {'Avg(µs)':>10} {'Std(µs)':>10} {'%':>6}")
        print(f"  {'-'*96}")

        for kernel in sorted_kernels:
            pct = (kernel.total_time_us / total_time * 100) if total_time > 0 else 0
            name = kernel.name[:44] if len(kernel.name) > 44 else kernel.name
            print(
                f"  {name:<45} {kernel.count:>8} {kernel.total_time_us/1000:>10.2f} "
                f"{kernel.avg_time_us:>10.2f} {kernel.std_time_us:>10.2f} {pct:>5.1f}%"
            )

        print(f"  {'-'*96}")
        print(f"\n  Kernel Categories Summary:")

        # Categorize kernels
        categories = defaultdict(lambda: {"count": 0, "time_us": 0})
        for kernel in self.kernel_stats.values():
            # Simple categorization based on name patterns
            name_lower = kernel.name.lower()
            if "gemm" in name_lower or "matmul" in name_lower:
                cat = "GEMM/MatMul"
            elif "conv" in name_lower:
                cat = "Convolution"
            elif "softmax" in name_lower:
                cat = "Softmax"
            elif "attention" in name_lower or "flash" in name_lower:
                cat = "Attention"
            elif "layernorm" in name_lower or "rmsnorm" in name_lower:
                cat = "Normalization"
            elif "elementwise" in name_lower or "add" in name_lower or "mul" in name_lower:
                cat = "Elementwise"
            elif "reduce" in name_lower or "sum" in name_lower:
                cat = "Reduction"
            elif "copy" in name_lower or "memcpy" in name_lower:
                cat = "Memory Copy"
            else:
                cat = "Other"

            categories[cat]["count"] += kernel.count
            categories[cat]["time_us"] += kernel.total_time_us

        print(f"  {'-'*60}")
        print(f"  {'Category':<20} {'Count':>10} {'Time(ms)':>12} {'%':>8}")
        print(f"  {'-'*60}")

        for cat, data in sorted(categories.items(), key=lambda x: -x[1]["time_us"]):
            pct = (data["time_us"] / total_time * 100) if total_time > 0 else 0
            print(f"  {cat:<20} {data['count']:>10} {data['time_us']/1000:>12.2f} {pct:>7.1f}%")

        print(f"{'='*100}\n")

    def print_layer_report(self):
        """Print layer-by-layer profiling report."""
        if not self.layer_profiles:
            print("No layer profiles recorded.")
            return

        print(f"\n{'='*110}")
        print(f"  LAYER-BY-LAYER ANALYSIS REPORT")
        print(f"{'='*110}")

        # Sort by forward time
        sorted_layers = sorted(
            self.layer_profiles.values(),
            key=lambda l: sum(l.forward_times_ms),
            reverse=True
        )

        total_forward = sum(sum(l.forward_times_ms) for l in sorted_layers)
        total_backward = sum(sum(l.backward_times_ms) for l in sorted_layers)

        print(f"\n  Total Forward Time: {total_forward:.2f} ms")
        print(f"  Total Backward Time: {total_backward:.2f} ms")
        print(f"\n  {'-'*106}")
        print(
            f"  {'Layer Name':<35} {'Type':<15} {'Fwd(ms)':>10} {'Bwd(ms)':>10} "
            f"{'Act(MB)':>10} {'Calls':>6} {'Fwd%':>6}"
        )
        print(f"  {'-'*106}")

        for layer in sorted_layers[:25]:  # Top 25 layers
            summary = layer.get_summary()
            fwd_total = summary["forward_ms"]["total"]
            bwd_total = summary["backward_ms"]["total"]
            act_mean = summary["activation_memory_mb"]["mean"]
            calls = summary["num_calls"]
            pct = (fwd_total / total_forward * 100) if total_forward > 0 else 0

            name = layer.name[:34] if len(layer.name) > 34 else layer.name
            ltype = layer.layer_type[:14] if len(layer.layer_type) > 14 else layer.layer_type

            print(
                f"  {name:<35} {ltype:<15} {fwd_total:>10.2f} {bwd_total:>10.2f} "
                f"{act_mean:>10.2f} {calls:>6} {pct:>5.1f}%"
            )

        print(f"  {'-'*106}")

        # Layer type summary
        type_summary = defaultdict(lambda: {"forward": 0, "backward": 0, "count": 0})
        for layer in self.layer_profiles.values():
            summary = layer.get_summary()
            type_summary[layer.layer_type]["forward"] += summary["forward_ms"]["total"]
            type_summary[layer.layer_type]["backward"] += summary["backward_ms"]["total"]
            type_summary[layer.layer_type]["count"] += 1

        print(f"\n  Layer Type Summary:")
        print(f"  {'-'*70}")
        print(f"  {'Type':<20} {'Count':>8} {'Forward(ms)':>15} {'Backward(ms)':>15} {'%':>8}")
        print(f"  {'-'*70}")

        for ltype, data in sorted(type_summary.items(), key=lambda x: -x[1]["forward"]):
            pct = (data["forward"] / total_forward * 100) if total_forward > 0 else 0
            print(
                f"  {ltype:<20} {data['count']:>8} {data['forward']:>15.2f} "
                f"{data['backward']:>15.2f} {pct:>7.1f}%"
            )

        print(f"{'='*110}\n")

    def print_moe_report(self):
        """Print MoE expert routing analysis report."""
        if not self.moe_routing_stats:
            print("No MoE routing statistics recorded.")
            return

        print(f"\n{'='*100}")
        print(f"  MoE EXPERT ROUTING ANALYSIS REPORT")
        print(f"{'='*100}")

        # Aggregate statistics
        all_loads = []
        all_balance_losses = []
        all_z_losses = []
        total_dropped = 0

        for stats in self.moe_routing_stats:
            all_loads.extend(stats.expert_loads)
            all_balance_losses.append(stats.load_balance_loss)
            all_z_losses.append(stats.router_z_loss)
            total_dropped += stats.dropped_tokens

        if all_loads:
            print(f"\n  Global Statistics (across {len(self.moe_routing_stats)} steps):")
            print(f"  {'-'*60}")
            print(f"  Avg Load Balance Loss:  {statistics.mean(all_balance_losses):.6f}")
            print(f"  Avg Router Z-Loss:      {statistics.mean(all_z_losses):.6f}")
            print(f"  Total Dropped Tokens:   {total_dropped:,}")
            print(f"  Avg Tokens/Expert:      {statistics.mean(all_loads):.1f}")
            print(f"  Std Tokens/Expert:      {statistics.stdev(all_loads) if len(all_loads) > 1 else 0:.1f}")

        # Per-expert analysis for most recent step
        if self.moe_routing_stats:
            latest = self.moe_routing_stats[-1]
            print(f"\n  Latest Step ({latest.step}) Expert Distribution:")
            print(f"  {'-'*80}")
            print(f"  {'Expert':<10} {'Tokens':>10} {'Utilization':>15} {'Compute(ms)':>15} {'Bar':>25}")
            print(f"  {'-'*80}")

            max_load = max(latest.expert_loads) if latest.expert_loads else 1

            for i, (load, util) in enumerate(zip(latest.expert_loads, latest.expert_utilization)):
                compute_ms = latest.expert_compute_times_ms[i] if i < len(latest.expert_compute_times_ms) else 0
                bar_len = int(load / max_load * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  Expert {i:<3} {load:>10,} {util:>14.2f}x {compute_ms:>14.2f} {bar}")

            print(f"  {'-'*80}")
            print(f"  Routing Time: {latest.routing_time_ms:.2f} ms")
            print(f"  Combine Time: {latest.combine_time_ms:.2f} ms")

        # Load balance over time
        if len(self.moe_routing_stats) > 5:
            print(f"\n  Load Balance Trend (last 10 steps):")
            print(f"  {'-'*50}")
            for stats in self.moe_routing_stats[-10:]:
                std = statistics.stdev(stats.expert_loads) if len(stats.expert_loads) > 1 else 0
                bar_len = min(int(std / 100), 30)
                bar = "▓" * bar_len
                print(f"  Step {stats.step:>5}: std={std:>8.1f} {bar}")

        print(f"{'='*100}\n")

    def print_memory_report(self):
        """Print memory timeline and analysis report."""
        if not self.memory_timeline:
            print("No memory timeline recorded.")
            return

        print(f"\n{'='*80}")
        print(f"  MEMORY ANALYSIS REPORT")
        print(f"{'='*80}")

        # Extract peak values
        peak_allocated = max(m.get("max_allocated_mb", 0) for m in self.memory_timeline)
        peak_reserved = max(m.get("max_reserved_mb", 0) for m in self.memory_timeline)

        print(f"\n  Peak Memory Usage:")
        print(f"  {'-'*40}")
        print(f"  Peak Allocated: {peak_allocated:>10.1f} MB")
        print(f"  Peak Reserved:  {peak_reserved:>10.1f} MB")

        # Memory progression
        if len(self.memory_timeline) > 1:
            first = self.memory_timeline[0]
            last = self.memory_timeline[-1]
            growth = last.get("allocated_mb", 0) - first.get("allocated_mb", 0)

            print(f"\n  Memory Growth:")
            print(f"  {'-'*40}")
            print(f"  Start:  {first.get('allocated_mb', 0):>10.1f} MB")
            print(f"  End:    {last.get('allocated_mb', 0):>10.1f} MB")
            print(f"  Growth: {growth:>+10.1f} MB")

        # Memory timeline visualization (last 20 steps)
        print(f"\n  Memory Timeline (last 20 steps):")
        print(f"  {'-'*60}")

        timeline = self.memory_timeline[-20:]
        max_mem = max(m.get("allocated_mb", 1) for m in timeline)

        for entry in timeline:
            step = entry.get("step", 0)
            allocated = entry.get("allocated_mb", 0)
            bar_len = int(allocated / max_mem * 40)
            bar = "█" * bar_len
            print(f"  Step {step:>5}: {allocated:>8.1f} MB {bar}")

        # Fragmentation analysis
        if self.fragmentation_stats:
            print(f"\n  Memory Fragmentation Analysis:")
            print(f"  {'-'*50}")
            recent_frag = self.fragmentation_stats[-10:]
            avg_frag = statistics.mean(f.fragmentation_ratio for f in recent_frag)
            max_frag = max(f.fragmentation_ratio for f in recent_frag)
            total_ooms = sum(f.num_ooms for f in recent_frag)

            print(f"  Average Fragmentation: {avg_frag*100:>6.1f}%")
            print(f"  Peak Fragmentation:    {max_frag*100:>6.1f}%")
            print(f"  Total OOMs:            {total_ooms:>6}")

        print(f"{'='*80}\n")

    def print_attention_report(self):
        """Print attention layer profiling report."""
        if not self.attention_stats:
            print("No attention statistics recorded.")
            return

        print(f"\n{'='*100}")
        print(f"  ATTENTION PROFILING REPORT")
        print(f"{'='*100}")

        # Group by layer
        layers: Dict[int, List[AttentionStats]] = defaultdict(list)
        for stats in self.attention_stats:
            layers[stats.layer_idx].append(stats)

        print(f"\n  Per-Layer Attention Statistics:")
        print(f"  {'-'*96}")
        print(f"  {'Layer':>6} {'Attn(ms)':>10} {'QKV(ms)':>10} {'Softmax':>10} {'KV$(MB)':>10} {'SeqLen':>8} {'Heads':>6} {'Flash':>6}")
        print(f"  {'-'*96}")

        for layer_idx in sorted(layers.keys()):
            layer_stats = layers[layer_idx]
            avg_attn = statistics.mean(s.attention_time_ms for s in layer_stats)
            avg_qkv = statistics.mean(s.qkv_projection_time_ms for s in layer_stats)
            avg_softmax = statistics.mean(s.softmax_time_ms for s in layer_stats)
            avg_kv = statistics.mean(s.kv_cache_size_mb for s in layer_stats)
            avg_seq = int(statistics.mean(s.seq_length for s in layer_stats))
            heads = layer_stats[0].num_heads
            flash = "Yes" if layer_stats[0].using_flash_attention else "No"

            print(f"  {layer_idx:>6} {avg_attn:>10.2f} {avg_qkv:>10.2f} {avg_softmax:>10.2f} "
                  f"{avg_kv:>10.1f} {avg_seq:>8} {heads:>6} {flash:>6}")

        # Summary
        total_attn_time = sum(s.attention_time_ms for s in self.attention_stats)
        total_kv_cache = sum(s.kv_cache_size_mb for s in self.attention_stats) / len(layers) if layers else 0

        print(f"\n  Summary:")
        print(f"  {'-'*40}")
        print(f"  Total Attention Time: {total_attn_time:.2f} ms")
        print(f"  Average KV Cache:     {total_kv_cache:.1f} MB")

        # Flash attention recommendation
        non_flash = sum(1 for s in self.attention_stats if not s.using_flash_attention)
        if non_flash > 0:
            print(f"\n  ⚠ {non_flash} attention calls not using Flash Attention")
            print(f"    Recommendation: Enable Flash Attention for ~40% memory savings")

        print(f"{'='*100}\n")

    def print_communication_report(self):
        """Print distributed communication profiling report."""
        if not self.communication_stats:
            print("No communication statistics recorded.")
            return

        print(f"\n{'='*100}")
        print(f"  DISTRIBUTED COMMUNICATION REPORT")
        print(f"{'='*100}")

        print(f"\n  Configuration:")
        print(f"  {'-'*40}")
        print(f"  World Size:  {self._world_size}")
        print(f"  Local Rank:  {self._local_rank}")

        # Group by operation type
        ops: Dict[str, List[CommunicationStats]] = defaultdict(list)
        for stats in self.communication_stats:
            ops[stats.operation].append(stats)

        print(f"\n  Communication Breakdown by Operation:")
        print(f"  {'-'*90}")
        print(f"  {'Operation':<20} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'BW(Gbps)':>10} {'Bytes':>15}")
        print(f"  {'-'*90}")

        total_time = 0
        total_bytes = 0
        for op_name in sorted(ops.keys()):
            op_stats = ops[op_name]
            count = len(op_stats)
            total_op_time = sum(s.time_ms for s in op_stats)
            avg_time = total_op_time / count if count > 0 else 0
            avg_bw = statistics.mean(s.bandwidth_gbps for s in op_stats if s.bandwidth_gbps > 0) if op_stats else 0
            op_bytes = sum(s.bytes_transferred for s in op_stats)

            total_time += total_op_time
            total_bytes += op_bytes

            print(f"  {op_name:<20} {count:>8} {total_op_time:>12.2f} {avg_time:>10.2f} "
                  f"{avg_bw:>10.2f} {op_bytes:>15,}")

        print(f"  {'-'*90}")
        print(f"  {'TOTAL':<20} {len(self.communication_stats):>8} {total_time:>12.2f}")

        # Overlap analysis
        overlapped = sum(1 for s in self.communication_stats if s.overlapped_with_compute)
        if self.communication_stats:
            overlap_pct = overlapped / len(self.communication_stats) * 100
            print(f"\n  Compute/Communication Overlap: {overlap_pct:.1f}%")
            if overlap_pct < 50:
                print(f"  ⚠ Low overlap - consider enabling gradient bucketing or async allreduce")

        print(f"{'='*100}\n")

    def print_gradient_report(self):
        """Print gradient analysis report."""
        if not self.gradient_stats:
            print("No gradient statistics recorded.")
            return

        print(f"\n{'='*100}")
        print(f"  GRADIENT ANALYSIS REPORT")
        print(f"{'='*100}")

        # Overall statistics
        all_norms = [g.total_grad_norm for g in self.gradient_stats]
        print(f"\n  Gradient Norm Statistics:")
        print(f"  {'-'*50}")
        print(f"  Mean:   {statistics.mean(all_norms):>12.6f}")
        print(f"  Std:    {statistics.stdev(all_norms) if len(all_norms) > 1 else 0:>12.6f}")
        print(f"  Min:    {min(all_norms):>12.6f}")
        print(f"  Max:    {max(all_norms):>12.6f}")

        # Health check
        total_nan = sum(g.num_nan_grads for g in self.gradient_stats)
        total_inf = sum(g.num_inf_grads for g in self.gradient_stats)
        total_zero = sum(g.num_zero_grads for g in self.gradient_stats)

        print(f"\n  Gradient Health:")
        print(f"  {'-'*50}")
        print(f"  NaN gradients:   {total_nan:>8} {'❌ CRITICAL!' if total_nan > 0 else '✓'}")
        print(f"  Inf gradients:   {total_inf:>8} {'❌ CRITICAL!' if total_inf > 0 else '✓'}")
        print(f"  Zero gradients:  {total_zero:>8} {'⚠ Warning' if total_zero > 5 else '✓'}")

        # Per-component gradient norms
        print(f"\n  Gradient Norms by Component:")
        print(f"  {'-'*50}")
        components = ['embedding', 'attention', 'ffn', 'moe', 'norm_layer']
        for comp in components:
            comp_norms = [getattr(g, f'{comp}_grad_norm', 0) for g in self.gradient_stats]
            avg_norm = statistics.mean(n for n in comp_norms if n > 0) if any(n > 0 for n in comp_norms) else 0
            if avg_norm > 0:
                print(f"  {comp.replace('_', ' ').title():<20}: {avg_norm:>12.6f}")

        # Gradient flow visualization (last 10 steps)
        print(f"\n  Gradient Norm Trend (last 20 steps):")
        print(f"  {'-'*60}")
        recent = self.gradient_stats[-20:]
        max_norm = max(g.total_grad_norm for g in recent) if recent else 1

        for g in recent:
            bar_len = int(g.total_grad_norm / max_norm * 40)
            bar = "█" * bar_len
            status = ""
            if g.num_nan_grads > 0:
                status = " [NaN!]"
            elif g.num_inf_grads > 0:
                status = " [Inf!]"
            elif g.was_clipped:
                status = " [clipped]"
            print(f"  Step {g.step:>5}: {g.total_grad_norm:>10.4f} {bar}{status}")

        # Vanishing/exploding layers
        all_vanishing = set()
        all_exploding = set()
        for g in self.gradient_stats:
            all_vanishing.update(g.vanishing_layers)
            all_exploding.update(g.exploding_layers)

        if all_vanishing:
            print(f"\n  ⚠ Vanishing Gradient Layers ({len(all_vanishing)}):")
            for layer in list(all_vanishing)[:5]:
                print(f"    - {layer}")

        if all_exploding:
            print(f"\n  ❌ Exploding Gradient Layers ({len(all_exploding)}):")
            for layer in list(all_exploding)[:5]:
                print(f"    - {layer}")

        print(f"{'='*100}\n")

    def print_bottleneck_report(self):
        """Print detected bottlenecks and recommendations."""
        self._detect_bottlenecks()

        print(f"\n{'='*100}")
        print(f"  BOTTLENECK DETECTION REPORT")
        print(f"{'='*100}")

        if not self.bottlenecks:
            print(f"\n  ✓ No significant bottlenecks detected!")
            print(f"{'='*100}\n")
            return

        # Group by severity
        critical = [b for b in self.bottlenecks if b.severity == "critical"]
        high = [b for b in self.bottlenecks if b.severity == "high"]
        medium = [b for b in self.bottlenecks if b.severity == "medium"]
        low = [b for b in self.bottlenecks if b.severity == "low"]

        def print_bottleneck_group(bottlenecks: List[Bottleneck], header: str, symbol: str):
            if not bottlenecks:
                return
            print(f"\n  {symbol} {header} ({len(bottlenecks)} issues)")
            print(f"  {'-'*90}")
            for b in bottlenecks:
                print(f"\n  [{b.category.upper()}] {b.description}")
                print(f"    Recommendation: {b.recommendation}")
                if b.impact_ms > 0:
                    print(f"    Estimated Impact: {b.impact_ms:.2f} ms per step")

        print_bottleneck_group(critical, "CRITICAL ISSUES", "❌")
        print_bottleneck_group(high, "HIGH PRIORITY", "⚠")
        print_bottleneck_group(medium, "MEDIUM PRIORITY", "●")
        print_bottleneck_group(low, "LOW PRIORITY", "○")

        print(f"\n  Summary: {len(critical)} critical, {len(high)} high, {len(medium)} medium, {len(low)} low")
        print(f"{'='*100}\n")

    def print_full_report(self):
        """Print comprehensive profiling report."""
        print("\n")
        print("╔" + "═"*78 + "╗")
        print("║" + " DETAILED PROFILING REPORT ".center(78) + "║")
        print("╚" + "═"*78 + "╝")

        # Overview
        if self.step_metrics:
            total_steps = len(self.step_metrics)
            total_time = sum(s["total_time_ms"] for s in self.step_metrics)
            avg_time = total_time / total_steps if total_steps > 0 else 0

            print(f"\n  OVERVIEW")
            print(f"  {'-'*40}")
            print(f"  Total Steps Profiled: {total_steps}")
            print(f"  Total Time: {total_time/1000:.2f} s")
            print(f"  Average Step Time: {avg_time:.2f} ms")
            print(f"  Throughput: {1000/avg_time:.1f} steps/s" if avg_time > 0 else "")

            # Distributed training info
            if self._is_distributed:
                print(f"\n  Distributed Training:")
                print(f"  {'-'*40}")
                print(f"  World Size: {self._world_size}")
                print(f"  Local Rank: {self._local_rank}")

        # Phase timing summary
        if self._phase_timings:
            print(f"\n  PHASE TIMING SUMMARY")
            print(f"  {'-'*70}")
            print(f"  {'Phase':<25} {'Total(ms)':>12} {'Mean(ms)':>12} {'Std(ms)':>12} {'%':>8}")
            print(f"  {'-'*70}")

            total_phase_time = sum(sum(times) for times in self._phase_timings.values())

            for phase, times in sorted(self._phase_timings.items(), key=lambda x: -sum(x[1])):
                total = sum(times)
                mean = statistics.mean(times) if times else 0
                std = statistics.stdev(times) if len(times) > 1 else 0
                pct = (total / total_phase_time * 100) if total_phase_time > 0 else 0
                print(f"  {phase:<25} {total:>12.2f} {mean:>12.2f} {std:>12.2f} {pct:>7.1f}%")

        # Print bottleneck report FIRST (most actionable)
        self.print_bottleneck_report()

        # Individual reports
        self.print_kernel_report()
        self.print_layer_report()
        self.print_attention_report()
        self.print_moe_report()
        self.print_gradient_report()
        self.print_communication_report()
        self.print_memory_report()

    def save_report(self, filename: str = "profile_report.json"):
        """Save all profiling data to JSON."""
        if not self.enabled or not self.run_dir:
            return

        # Detect bottlenecks before saving
        self._detect_bottlenecks()

        total_time = sum(s["total_time_ms"] for s in self.step_metrics) if self.step_metrics else 0
        total_steps = len(self.step_metrics)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "total_steps": total_steps,
                "total_time_ms": total_time,
                "avg_step_time_ms": total_time / total_steps if total_steps > 0 else 0,
                "throughput_steps_per_sec": 1000 * total_steps / total_time if total_time > 0 else 0,
            },
            "distributed": {
                "is_distributed": self._is_distributed,
                "world_size": self._world_size,
                "local_rank": self._local_rank,
            },
            "step_metrics": self.step_metrics,
            "phase_timings": {k: list(v) for k, v in self._phase_timings.items()},
            "kernel_stats": {k: v.to_dict() for k, v in self.kernel_stats.items()},
            "layer_profiles": {k: v.get_summary() for k, v in self.layer_profiles.items()},
            "moe_routing_stats": [s.to_dict() for s in self.moe_routing_stats],
            "attention_stats": [s.to_dict() for s in self.attention_stats],
            "communication_stats": [s.to_dict() for s in self.communication_stats],
            "gradient_stats": [s.to_dict() for s in self.gradient_stats],
            "tensor_stats": {k: v.to_dict() for k, v in self.tensor_stats.items()},
            "fragmentation_stats": [s.to_dict() for s in self.fragmentation_stats],
            "memory_timeline": self.memory_timeline,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
        }

        # Add GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu_info"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": torch.cuda.get_device_capability(0),
                "multi_processor_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
            }

        path = self.run_dir / filename
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"[DetailedProfiler] Report saved to: {path}")
        return path

    def export_html_report(self, filename: str = "profile_report.html"):
        """Export an interactive HTML report with visualizations."""
        if not self.enabled or not self.run_dir:
            return

        self._detect_bottlenecks()

        total_time = sum(s["total_time_ms"] for s in self.step_metrics) if self.step_metrics else 0
        total_steps = len(self.step_metrics)
        avg_step_time = total_time / total_steps if total_steps > 0 else 0

        # Generate HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ava Training Profile Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; margin-bottom: 20px; text-align: center; }}
        h2 {{ color: #00d4ff; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
        h3 {{ color: #888; margin: 15px 0 8px; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric {{ text-align: center; padding: 15px; }}
        .metric .value {{ font-size: 2.5em; font-weight: bold; color: #00d4ff; }}
        .metric .label {{ color: #888; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        tr:hover {{ background: #1f4068; }}
        .bar {{ height: 20px; background: linear-gradient(90deg, #00d4ff, #0077b6); border-radius: 3px; }}
        .severity-critical {{ color: #ff4757; font-weight: bold; }}
        .severity-high {{ color: #ffa502; }}
        .severity-medium {{ color: #ffdd59; }}
        .severity-low {{ color: #7bed9f; }}
        .bottleneck {{ background: #1f2937; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid; }}
        .bottleneck.critical {{ border-color: #ff4757; }}
        .bottleneck.high {{ border-color: #ffa502; }}
        .bottleneck.medium {{ border-color: #ffdd59; }}
        .recommendation {{ color: #7bed9f; font-style: italic; margin-top: 8px; }}
        .timeline {{ display: flex; align-items: flex-end; height: 150px; gap: 2px; padding: 10px 0; }}
        .timeline-bar {{ flex: 1; background: #00d4ff; min-width: 10px; transition: height 0.3s; }}
        .good {{ color: #7bed9f; }}
        .warn {{ color: #ffdd59; }}
        .bad {{ color: #ff4757; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Ava Training Profile Report</h1>
        <p style="text-align:center;color:#888;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="card">
            <h2>Overview</h2>
            <div class="grid">
                <div class="metric">
                    <div class="value">{total_steps}</div>
                    <div class="label">Steps Profiled</div>
                </div>
                <div class="metric">
                    <div class="value">{total_time/1000:.1f}s</div>
                    <div class="label">Total Time</div>
                </div>
                <div class="metric">
                    <div class="value">{avg_step_time:.1f}ms</div>
                    <div class="label">Avg Step Time</div>
                </div>
                <div class="metric">
                    <div class="value">{1000/avg_step_time:.1f}</div>
                    <div class="label">Steps/Second</div>
                </div>
            </div>
        </div>
"""

        # Bottlenecks section
        if self.bottlenecks:
            html_content += """
        <div class="card">
            <h2>Detected Bottlenecks</h2>
"""
            for b in sorted(self.bottlenecks, key=lambda x: ['critical', 'high', 'medium', 'low'].index(x.severity)):
                html_content += f"""
            <div class="bottleneck {b.severity}">
                <strong class="severity-{b.severity}">[{b.category.upper()}]</strong> {b.description}
                <div class="recommendation">💡 {b.recommendation}</div>
            </div>
"""
            html_content += "        </div>\n"

        # Phase timings
        if self._phase_timings:
            total_phase_time = sum(sum(times) for times in self._phase_timings.values())
            html_content += """
        <div class="card">
            <h2>Phase Timing Breakdown</h2>
            <table>
                <tr><th>Phase</th><th>Total (ms)</th><th>Mean (ms)</th><th>%</th><th>Distribution</th></tr>
"""
            for phase, times in sorted(self._phase_timings.items(), key=lambda x: -sum(x[1])):
                total = sum(times)
                mean = statistics.mean(times) if times else 0
                pct = (total / total_phase_time * 100) if total_phase_time > 0 else 0
                html_content += f"""
                <tr>
                    <td>{phase}</td>
                    <td>{total:.2f}</td>
                    <td>{mean:.2f}</td>
                    <td>{pct:.1f}%</td>
                    <td><div class="bar" style="width:{pct}%"></div></td>
                </tr>
"""
            html_content += "            </table>\n        </div>\n"

        # Memory timeline
        if self.memory_timeline:
            peak_mem = max(m.get('max_allocated_mb', 0) for m in self.memory_timeline)
            html_content += f"""
        <div class="card">
            <h2>Memory Usage</h2>
            <div class="grid">
                <div class="metric">
                    <div class="value">{peak_mem:.0f} MB</div>
                    <div class="label">Peak Allocated</div>
                </div>
            </div>
            <h3>Memory Timeline</h3>
            <div class="timeline">
"""
            timeline = self.memory_timeline[-50:]  # Last 50 steps
            max_mem = max(m.get('allocated_mb', 1) for m in timeline)
            for m in timeline:
                height = (m.get('allocated_mb', 0) / max_mem * 100) if max_mem > 0 else 0
                html_content += f'                <div class="timeline-bar" style="height:{height}%" title="Step {m.get("step", 0)}: {m.get("allocated_mb", 0):.0f} MB"></div>\n'
            html_content += "            </div>\n        </div>\n"

        # Gradient health
        if self.gradient_stats:
            total_nan = sum(g.num_nan_grads for g in self.gradient_stats)
            total_inf = sum(g.num_inf_grads for g in self.gradient_stats)
            avg_norm = statistics.mean(g.total_grad_norm for g in self.gradient_stats)

            nan_class = "bad" if total_nan > 0 else "good"
            inf_class = "bad" if total_inf > 0 else "good"

            html_content += f"""
        <div class="card">
            <h2>Gradient Health</h2>
            <div class="grid">
                <div class="metric">
                    <div class="value {nan_class}">{total_nan}</div>
                    <div class="label">NaN Gradients</div>
                </div>
                <div class="metric">
                    <div class="value {inf_class}">{total_inf}</div>
                    <div class="label">Inf Gradients</div>
                </div>
                <div class="metric">
                    <div class="value">{avg_norm:.4f}</div>
                    <div class="label">Avg Grad Norm</div>
                </div>
            </div>
        </div>
"""

        # MoE stats
        if self.moe_routing_stats:
            recent = self.moe_routing_stats[-1]
            avg_balance_loss = statistics.mean(s.load_balance_loss for s in self.moe_routing_stats)
            total_dropped = sum(s.dropped_tokens for s in self.moe_routing_stats)

            html_content += f"""
        <div class="card">
            <h2>MoE Expert Routing</h2>
            <div class="grid">
                <div class="metric">
                    <div class="value">{recent.num_experts}</div>
                    <div class="label">Num Experts</div>
                </div>
                <div class="metric">
                    <div class="value">{avg_balance_loss:.6f}</div>
                    <div class="label">Avg Load Balance Loss</div>
                </div>
                <div class="metric">
                    <div class="value">{total_dropped:,}</div>
                    <div class="label">Total Dropped Tokens</div>
                </div>
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        path = self.run_dir / filename
        with open(path, 'w') as f:
            f.write(html_content)

        print(f"[DetailedProfiler] HTML report saved to: {path}")
        return path

    def finish(self):
        """Finalize profiling and generate reports."""
        if not self.enabled:
            return

        if self.print_final_report:
            self.print_full_report()

        # Save reports
        self.save_report()
        self.export_html_report()

        print(f"\n[DetailedProfiler] Profiling complete. Reports saved to: {self.run_dir}")


def create_detailed_profiler(
    enabled: bool = True,
    output_dir: str = "./profiles",
    track_kernels: bool = True,
    track_layers: bool = True,
    track_moe: bool = True,
    track_memory_timeline: bool = True,
    track_attention: bool = True,
    track_communication: bool = True,
    track_gradients: bool = True,
    track_tensors: bool = False,
    track_fragmentation: bool = True,
    detect_bottlenecks: bool = True,
    print_step_summary: bool = True,
    print_final_report: bool = True,
    verbose: bool = False,
    **kwargs
) -> DetailedProfiler:
    """Factory function to create a DetailedProfiler.

    Args:
        enabled: Enable profiling
        output_dir: Output directory for reports
        track_kernels: Track CUDA kernel statistics
        track_layers: Track layer-by-layer timing
        track_moe: Track MoE expert routing
        track_memory_timeline: Track memory usage over time
        track_attention: Track attention layer statistics
        track_communication: Track distributed communication
        track_gradients: Track gradient statistics
        track_tensors: Track tensor shapes/dtypes (expensive)
        track_fragmentation: Track memory fragmentation
        detect_bottlenecks: Enable automatic bottleneck detection
        print_step_summary: Print summary after each step
        print_final_report: Print full report at end
        verbose: Enable verbose output
        **kwargs: Additional arguments passed to DetailedProfiler

    Returns:
        DetailedProfiler instance
    """
    return DetailedProfiler(
        enabled=enabled,
        output_dir=output_dir,
        track_kernels=track_kernels,
        track_layers=track_layers,
        track_moe=track_moe,
        track_memory_timeline=track_memory_timeline,
        track_attention=track_attention,
        track_communication=track_communication,
        track_gradients=track_gradients,
        track_tensors=track_tensors,
        track_fragmentation=track_fragmentation,
        detect_bottlenecks=detect_bottlenecks,
        print_step_summary=print_step_summary,
        print_final_report=print_final_report,
        verbose=verbose,
        **kwargs
    )


# =============================================================================
# CUDA EVENT-BASED TIMING UTILITIES
# =============================================================================


class CUDATimer:
    """High-precision CUDA event-based timer for accurate GPU timing."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    def start(self, name: str):
        """Start timing a named operation."""
        if not self.enabled:
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._events[name] = (start_event, end_event)

    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        if not self.enabled or name not in self._events:
            return 0.0

        start_event, end_event = self._events[name]
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        del self._events[name]
        return elapsed_ms

    @contextmanager
    def time(self, name: str):
        """Context manager for timing."""
        self.start(name)
        try:
            yield
        finally:
            elapsed = self.stop(name)
            if self.enabled:
                logger.debug(f"[CUDATimer] {name}: {elapsed:.2f} ms")


class ProfilerHook:
    """Hook for automatically profiling model forward/backward passes."""

    def __init__(self, profiler: DetailedProfiler, prefix: str = ""):
        self.profiler = profiler
        self.prefix = prefix
        self._forward_start_times: Dict[str, float] = {}
        self._forward_start_memory: Dict[str, float] = {}

    def forward_pre_hook(self, module: torch.nn.Module, input):
        """Called before forward pass."""
        if not self.profiler.enabled:
            return

        name = f"{self.prefix}{module.__class__.__name__}"
        self._forward_start_times[name] = time.perf_counter()
        if torch.cuda.is_available():
            self._forward_start_memory[name] = torch.cuda.memory_allocated() / 1024**2

    def forward_hook(self, module: torch.nn.Module, input, output):
        """Called after forward pass."""
        if not self.profiler.enabled:
            return

        name = f"{self.prefix}{module.__class__.__name__}"
        if name in self._forward_start_times:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - self._forward_start_times[name]) * 1000
            activation_mb = 0.0
            if name in self._forward_start_memory and torch.cuda.is_available():
                activation_mb = torch.cuda.memory_allocated() / 1024**2 - self._forward_start_memory[name]

            # Determine layer type
            layer_type = module.__class__.__name__

            # Get input/output shapes
            input_shape = None
            output_shape = None
            if isinstance(input, tuple) and len(input) > 0 and torch.is_tensor(input[0]):
                input_shape = tuple(input[0].shape)
            if torch.is_tensor(output):
                output_shape = tuple(output.shape)
            elif isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                output_shape = tuple(output[0].shape)

            self.profiler.record_layer(
                name=name,
                layer_type=layer_type,
                forward_time_ms=elapsed_ms,
                activation_mb=activation_mb,
                input_shape=input_shape,
                output_shape=output_shape,
            )

            del self._forward_start_times[name]
            if name in self._forward_start_memory:
                del self._forward_start_memory[name]

    def register(self, model: torch.nn.Module, layer_types: List[type] = None):
        """Register hooks on model layers.

        Args:
            model: Model to profile
            layer_types: Optional list of layer types to profile.
                         If None, profiles common types (Linear, Attention, etc.)
        """
        if layer_types is None:
            layer_types = [
                torch.nn.Linear,
                torch.nn.LayerNorm,
                torch.nn.Embedding,
            ]
            # Try to add common transformer layers
            try:
                from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
                layer_types.extend([LlamaAttention, LlamaMLP])
            except ImportError:
                logger.debug("Transformers library not available for Llama layer profiling")

        handles = []
        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in layer_types):
                h1 = module.register_forward_pre_hook(self.forward_pre_hook)
                h2 = module.register_forward_hook(self.forward_hook)
                handles.extend([h1, h2])

        return handles


def profile_model_forward(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    profiler: DetailedProfiler = None,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Profile a model's forward pass with detailed timing.

    Args:
        model: Model to profile
        sample_input: Sample input dictionary
        profiler: Optional DetailedProfiler instance
        num_warmup: Number of warmup runs
        num_runs: Number of profiling runs

    Returns:
        Dictionary with timing statistics
    """
    if profiler is None:
        profiler = DetailedProfiler(enabled=True, print_step_summary=False)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Profile runs
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            with profiler.step(i):
                with profiler.phase("forward"):
                    start = time.perf_counter()
                    _ = model(**sample_input)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "times_ms": times,
    }
