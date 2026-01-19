"""
Data Loading Profiler for identifying bottlenecks.

Tracks timing and statistics for all stages of data loading:
- File I/O time (Arrow table loading)
- Data extraction time (to_pydict vs zero-copy)
- Tensor conversion time (numpy -> torch)
- Collation time (padding, stacking)
- GPU transfer time (CPU -> GPU)
- Cache hit/miss rates

Usage:
    from ava.data.profiling import DataLoaderProfiler, get_global_profiler

    # Enable profiling
    profiler = get_global_profiler()
    profiler.enable()

    # In your data loading code:
    profiler.start_timer('file_load')
    table = cache.get(file_path)
    profiler.stop_timer('file_load')

    # Print report
    print(profiler.report())
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Maximum history size for timing lists to prevent memory leaks
MAX_PROFILING_HISTORY = 200

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderStats:
    """Aggregated data loading statistics."""

    # Timing stats (in seconds) - bounded to prevent memory leaks
    file_load_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))
    extraction_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))
    tensor_conversion_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))
    collation_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))
    gpu_transfer_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))
    batch_total_times: deque = field(default_factory=lambda: deque(maxlen=MAX_PROFILING_HISTORY))

    # Throughput stats
    samples_processed: int = 0
    batches_processed: int = 0
    tokens_processed: int = 0

    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0

    # Data stats
    total_sequences: int = 0
    avg_sequence_length: float = 0.0
    packing_efficiency: float = 0.0

    def _avg(self, lst: List[float]) -> float:
        """Calculate average of a list."""
        return sum(lst) / len(lst) if lst else 0.0

    def _percentile(self, lst: List[float], p: float) -> float:
        """Calculate percentile of a list."""
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        k = (len(sorted_lst) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_lst) else f
        return sorted_lst[f] + (sorted_lst[c] - sorted_lst[f]) * (k - f)

    def summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        total_time = sum(self.batch_total_times)
        elapsed_time = total_time if total_time > 0 else 1.0

        return {
            # Timing (milliseconds)
            "avg_file_load_ms": self._avg(self.file_load_times) * 1000,
            "avg_extraction_ms": self._avg(self.extraction_times) * 1000,
            "avg_tensor_conv_ms": self._avg(self.tensor_conversion_times) * 1000,
            "avg_collation_ms": self._avg(self.collation_times) * 1000,
            "avg_gpu_transfer_ms": self._avg(self.gpu_transfer_times) * 1000,
            "avg_batch_total_ms": self._avg(self.batch_total_times) * 1000,
            # Percentiles (milliseconds)
            "p99_file_load_ms": self._percentile(self.file_load_times, 99) * 1000,
            "p99_extraction_ms": self._percentile(self.extraction_times, 99) * 1000,
            "p99_batch_total_ms": self._percentile(self.batch_total_times, 99) * 1000,
            # Throughput
            "samples_per_sec": self.samples_processed / elapsed_time,
            "batches_per_sec": self.batches_processed / elapsed_time,
            "tokens_per_sec": self.tokens_processed / elapsed_time,
            # Cache
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            # Data
            "total_batches": self.batches_processed,
            "total_samples": self.samples_processed,
            "avg_sequence_length": self.avg_sequence_length,
            "packing_efficiency": self.packing_efficiency,
        }

    def reset(self):
        """Reset all statistics."""
        self.file_load_times.clear()
        self.extraction_times.clear()
        self.tensor_conversion_times.clear()
        self.collation_times.clear()
        self.gpu_transfer_times.clear()
        self.batch_total_times.clear()
        self.samples_processed = 0
        self.batches_processed = 0
        self.tokens_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_sequences = 0
        self.avg_sequence_length = 0.0
        self.packing_efficiency = 0.0


class DataLoaderProfiler:
    """
    Thread-safe profiler for data loading operations.

    Provides timing instrumentation and statistics collection with
    minimal overhead when disabled.

    Example:
        >>> profiler = DataLoaderProfiler(enabled=True)
        >>> profiler.start_timer('file_load')
        >>> table = load_arrow_table(path)
        >>> profiler.stop_timer('file_load')
        >>> print(profiler.report())
    """

    def __init__(
        self,
        enabled: bool = False,
        warmup_batches: int = 10,
        max_samples: int = 10000,
    ):
        """
        Initialize the profiler.

        Args:
            enabled: Whether profiling is active
            warmup_batches: Number of batches to skip before recording
            max_samples: Maximum number of samples to keep per metric (prevents memory bloat)
        """
        self._enabled = enabled
        self._warmup_batches = warmup_batches
        self._max_samples = max_samples

        self._stats = DataLoaderStats()
        self._batch_count = 0
        self._current_timers: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Per-worker stats for multi-worker dataloaders
        self._worker_stats: Dict[int, DataLoaderStats] = defaultdict(DataLoaderStats)

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    def enable(self):
        """Enable profiling."""
        self._enabled = True
        logger.info("DataLoaderProfiler enabled")

    def disable(self):
        """Disable profiling."""
        self._enabled = False
        logger.info("DataLoaderProfiler disabled")

    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self._stats.reset()
            self._batch_count = 0
            self._current_timers.clear()
            self._worker_stats.clear()

    def start_timer(self, name: str):
        """Start a named timer."""
        if not self._enabled or self._batch_count < self._warmup_batches:
            return
        self._current_timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record the elapsed time.

        Returns:
            Elapsed time in seconds (0.0 if profiling disabled)
        """
        if not self._enabled or self._batch_count < self._warmup_batches:
            return 0.0

        if name not in self._current_timers:
            return 0.0

        elapsed = time.perf_counter() - self._current_timers[name]
        del self._current_timers[name]

        # Record to appropriate stats list
        with self._lock:
            if name == "file_load":
                if len(self._stats.file_load_times) < self._max_samples:
                    self._stats.file_load_times.append(elapsed)
            elif name == "extraction":
                if len(self._stats.extraction_times) < self._max_samples:
                    self._stats.extraction_times.append(elapsed)
            elif name == "tensor_conversion":
                if len(self._stats.tensor_conversion_times) < self._max_samples:
                    self._stats.tensor_conversion_times.append(elapsed)
            elif name == "collation":
                if len(self._stats.collation_times) < self._max_samples:
                    self._stats.collation_times.append(elapsed)
            elif name == "gpu_transfer":
                if len(self._stats.gpu_transfer_times) < self._max_samples:
                    self._stats.gpu_transfer_times.append(elapsed)
            elif name == "batch_total":
                if len(self._stats.batch_total_times) < self._max_samples:
                    self._stats.batch_total_times.append(elapsed)

        return elapsed

    def record_cache_hit(self):
        """Record a cache hit."""
        if self._enabled:
            with self._lock:
                self._stats.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        if self._enabled:
            with self._lock:
                self._stats.cache_misses += 1

    def record_batch(self, batch_size: int, num_tokens: int = 0):
        """
        Record batch processing.

        Args:
            batch_size: Number of samples in the batch
            num_tokens: Total tokens in the batch (for tokens/sec calculation)
        """
        with self._lock:
            self._batch_count += 1
            self._stats.batches_processed += 1
            self._stats.samples_processed += batch_size
            self._stats.tokens_processed += num_tokens

    def record_sequence_stats(self, avg_length: float, packing_efficiency: float = 0.0):
        """Record sequence statistics."""
        if self._enabled:
            with self._lock:
                self._stats.avg_sequence_length = avg_length
                self._stats.packing_efficiency = packing_efficiency

    def get_stats(self) -> DataLoaderStats:
        """Get current statistics."""
        with self._lock:
            return self._stats

    def summary(self) -> Dict[str, float]:
        """Get summary statistics dictionary."""
        with self._lock:
            return self._stats.summary()

    def report(self) -> str:
        """Generate human-readable profiling report."""
        s = self.summary()

        if s["total_batches"] == 0:
            return "DataLoaderProfiler: No batches profiled yet"

        lines = [
            "",
            "=" * 60,
            "DATA LOADING PROFILE",
            "=" * 60,
            f"Batches profiled: {s['total_batches']:,} ({s['total_samples']:,} samples)",
            f"Cache hit rate: {s['cache_hit_rate']:.1%} ({s['cache_hits']:,} hits, {s['cache_misses']:,} misses)",
            "",
            "TIMING BREAKDOWN (avg per batch):",
            f"  {'File I/O:':<20} {s['avg_file_load_ms']:>8.2f} ms (p99: {s['p99_file_load_ms']:.2f} ms)",
            f"  {'Extraction:':<20} {s['avg_extraction_ms']:>8.2f} ms (p99: {s['p99_extraction_ms']:.2f} ms)",
            f"  {'Tensor conversion:':<20} {s['avg_tensor_conv_ms']:>8.2f} ms",
            f"  {'Collation:':<20} {s['avg_collation_ms']:>8.2f} ms",
            f"  {'GPU transfer:':<20} {s['avg_gpu_transfer_ms']:>8.2f} ms",
            f"  {'---':<20} {'-' * 8}",
            f"  {'Batch total:':<20} {s['avg_batch_total_ms']:>8.2f} ms (p99: {s['p99_batch_total_ms']:.2f} ms)",
            "",
            "THROUGHPUT:",
            f"  {s['samples_per_sec']:.1f} samples/sec",
            f"  {s['batches_per_sec']:.1f} batches/sec",
            f"  {s['tokens_per_sec']:,.0f} tokens/sec",
            "",
            "BOTTLENECK ANALYSIS:",
        ]

        # Identify the biggest bottleneck
        timings = {
            "File I/O": s["avg_file_load_ms"],
            "Extraction": s["avg_extraction_ms"],
            "Tensor conversion": s["avg_tensor_conv_ms"],
            "Collation": s["avg_collation_ms"],
            "GPU transfer": s["avg_gpu_transfer_ms"],
        }

        total_time = sum(timings.values())
        if total_time > 0:
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
            for name, time_ms in sorted_timings:
                pct = (time_ms / total_time) * 100
                bar_len = int(pct / 2)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                lines.append(f"  {name:<20} {bar} {pct:>5.1f}%")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def log_summary(self, level: int = logging.INFO):
        """Log the profiling summary."""
        logger.log(level, self.report())


# Global profiler instance
_global_profiler: Optional[DataLoaderProfiler] = None
_global_profiler_lock = threading.Lock()


def get_global_profiler() -> DataLoaderProfiler:
    """Get or create the global DataLoaderProfiler instance."""
    global _global_profiler

    if _global_profiler is None:
        with _global_profiler_lock:
            if _global_profiler is None:
                _global_profiler = DataLoaderProfiler(enabled=False)

    return _global_profiler


def enable_profiling():
    """Enable the global data loader profiler."""
    get_global_profiler().enable()


def disable_profiling():
    """Disable the global data loader profiler."""
    get_global_profiler().disable()


def profile_report() -> str:
    """Get the global profiler report."""
    return get_global_profiler().report()


__all__ = [
    "DataLoaderStats",
    "DataLoaderProfiler",
    "get_global_profiler",
    "enable_profiling",
    "disable_profiling",
    "profile_report",
]
