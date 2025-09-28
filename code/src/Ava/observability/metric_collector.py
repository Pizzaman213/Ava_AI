"""
Optimized Metric Collector (Phase 7.1)

Fix metric collection overhead with intelligent sampling and async computation.
"""

import time
import threading
import queue
import asyncio
import math
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from enum import Enum


class SamplingStrategy(Enum):
    """Metric sampling strategies."""
    NONE = "none"                    # No sampling, collect all
    UNIFORM = "uniform"              # Uniform sampling at fixed rate
    ADAPTIVE = "adaptive"            # Adaptive sampling based on importance
    LOGARITHMIC = "logarithmic"      # Logarithmic sampling (more frequent early)
    EXPONENTIAL = "exponential"      # Exponential decay sampling
    SIGNIFICANCE = "significance"    # Sample based on metric significance


@dataclass
class MetricSample:
    """A sampled metric with metadata."""
    name: str
    value: Any
    timestamp: float
    step: Optional[int] = None
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0


@dataclass
class SamplingConfig:
    """Configuration for metric sampling."""
    strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE
    base_sample_rate: float = 0.1        # Base sampling rate (10%)
    min_sample_rate: float = 0.01        # Minimum sampling rate (1%)
    max_sample_rate: float = 1.0         # Maximum sampling rate (100%)
    importance_threshold: float = 0.5     # Threshold for importance-based sampling
    decay_factor: float = 0.95           # Decay factor for exponential sampling
    adaptation_interval: int = 100        # Interval for adaptive rate adjustment


class MetricSampler(ABC):
    """Abstract base class for metric samplers."""

    @abstractmethod
    def should_sample(self, metric_name: str, step: int, importance: float = 1.0) -> bool:
        """Determine if a metric should be sampled."""
        pass

    @abstractmethod
    def update_rates(self, performance_metrics: Dict[str, float]):
        """Update sampling rates based on performance."""
        pass


class AdaptiveSampler(MetricSampler):
    """Adaptive sampler that adjusts rates based on system performance."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.current_rates = defaultdict(lambda: config.base_sample_rate)
        self.performance_history = deque(maxlen=100)
        self.last_adaptation = 0

    def should_sample(self, metric_name: str, step: int, importance: float = 1.0) -> bool:
        """Adaptive sampling decision."""
        base_rate = self.current_rates[metric_name]

        # Adjust rate based on importance
        adjusted_rate = base_rate * importance

        # Clamp to bounds
        adjusted_rate = max(self.config.min_sample_rate,
                          min(adjusted_rate, self.config.max_sample_rate))

        return random.random() < adjusted_rate

    def update_rates(self, performance_metrics: Dict[str, float]):
        """Update sampling rates based on system performance."""
        self.performance_history.append(performance_metrics)

        # Adapt rates every N steps
        if len(self.performance_history) >= self.config.adaptation_interval:
            avg_collection_time = sum(p.get('metric_collection_time', 0)
                                    for p in self.performance_history) / len(self.performance_history)

            # If collection is taking too long, reduce sampling rates
            if avg_collection_time > 0.1:  # 100ms threshold
                for metric_name in self.current_rates:
                    self.current_rates[metric_name] *= 0.8  # Reduce by 20%
                    self.current_rates[metric_name] = max(
                        self.current_rates[metric_name],
                        self.config.min_sample_rate
                    )
            # If collection is fast, we can increase rates
            elif avg_collection_time < 0.05:  # 50ms threshold
                for metric_name in self.current_rates:
                    self.current_rates[metric_name] *= 1.1  # Increase by 10%
                    self.current_rates[metric_name] = min(
                        self.current_rates[metric_name],
                        self.config.max_sample_rate
                    )

            self.performance_history.clear()


class UniformSampler(MetricSampler):
    """Simple uniform sampling at fixed rate."""

    def __init__(self, config: SamplingConfig):
        self.config = config

    def should_sample(self, metric_name: str, step: int, importance: float = 1.0) -> bool:
        return random.random() < self.config.base_sample_rate

    def update_rates(self, performance_metrics: Dict[str, float]):
        pass  # No adaptation for uniform sampling


class LogarithmicSampler(MetricSampler):
    """Logarithmic sampling - more frequent early in training."""

    def __init__(self, config: SamplingConfig):
        self.config = config

    def should_sample(self, metric_name: str, step: int, importance: float = 1.0) -> bool:
        # Higher sampling rate early in training
        if step < 100:
            rate = self.config.max_sample_rate
        else:
            rate = self.config.base_sample_rate * math.log(100) / math.log(step)
            rate = max(self.config.min_sample_rate, min(rate, self.config.max_sample_rate))

        return random.random() < rate

    def update_rates(self, performance_metrics: Dict[str, float]):
        pass


class AsyncMetricComputer:
    """Asynchronous metric computation to reduce blocking."""

    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.max_workers = max_workers
        self.computation_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()

        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def submit_computation(self, metric_name: str, computation_func: Callable, *args, **kwargs):
        """Submit a metric computation for async processing."""
        try:
            self.computation_queue.put_nowait({
                'metric_name': metric_name,
                'func': computation_func,
                'args': args,
                'kwargs': kwargs,
                'submit_time': time.time()
            })
        except queue.Full:
            # Drop computation if queue is full
            pass

    def get_completed_results(self) -> List[MetricSample]:
        """Get all completed metric computations."""
        results = []
        try:
            while True:
                result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results

    def _worker_loop(self):
        """Worker thread loop for processing computations."""
        while not self.stop_event.is_set():
            try:
                task = self.computation_queue.get(timeout=1.0)
                start_time = time.time()

                try:
                    # Execute computation
                    result = task['func'](*task['args'], **task['kwargs'])
                    computation_time = time.time() - start_time

                    # Create metric sample
                    sample = MetricSample(
                        name=task['metric_name'],
                        value=result,
                        timestamp=time.time(),
                        computation_time=computation_time
                    )

                    self.result_queue.put(sample)

                except Exception as e:
                    # Log error but don't crash
                    error_sample = MetricSample(
                        name=f"{task['metric_name']}_error",
                        value=str(e),
                        timestamp=time.time(),
                        computation_time=time.time() - start_time
                    )
                    self.result_queue.put(error_sample)

            except queue.Empty:
                continue
            except Exception:
                continue

    def shutdown(self):
        """Shutdown async computation."""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=5.0)


class MetricRegistry:
    """Registry for metric computation functions with metadata."""

    def __init__(self):
        self.metrics = {}
        self.metadata = {}

    def register(
        self,
        name: str,
        computation_func: Callable,
        importance: float = 1.0,
        cost: float = 1.0,
        dependencies: List[str] = None,
        async_safe: bool = True
    ):
        """Register a metric computation function."""
        self.metrics[name] = computation_func
        self.metadata[name] = {
            'importance': importance,
            'cost': cost,
            'dependencies': dependencies or [],
            'async_safe': async_safe,
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

    def get_metric_func(self, name: str) -> Optional[Callable]:
        """Get metric computation function."""
        return self.metrics.get(name)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metric metadata."""
        return self.metadata.get(name, {})

    def update_timing(self, name: str, execution_time: float):
        """Update timing statistics for a metric."""
        if name in self.metadata:
            meta = self.metadata[name]
            meta['call_count'] += 1
            meta['total_time'] += execution_time
            meta['avg_time'] = meta['total_time'] / meta['call_count']

    def get_expensive_metrics(self, threshold: float = 0.1) -> List[str]:
        """Get list of metrics that are expensive to compute."""
        return [
            name for name, meta in self.metadata.items()
            if meta.get('avg_time', 0) > threshold
        ]


class OptimizedMetricCollector:
    """
    High-performance metric collector with intelligent sampling and async computation.

    Features:
    - Multiple sampling strategies
    - Async computation for expensive metrics
    - Intelligent rate adaptation
    - Overhead monitoring
    - Configurable importance weighting
    """

    def __init__(
        self,
        sampling_config: SamplingConfig = None,
        enable_async: bool = True,
        max_async_workers: int = 4,
        overhead_threshold: float = 0.05  # 5% of total time
    ):
        self.sampling_config = sampling_config or SamplingConfig()
        self.enable_async = enable_async
        self.overhead_threshold = overhead_threshold

        # Initialize sampler
        if self.sampling_config.strategy == SamplingStrategy.ADAPTIVE:
            self.sampler = AdaptiveSampler(self.sampling_config)
        elif self.sampling_config.strategy == SamplingStrategy.UNIFORM:
            self.sampler = UniformSampler(self.sampling_config)
        elif self.sampling_config.strategy == SamplingStrategy.LOGARITHMIC:
            self.sampler = LogarithmicSampler(self.sampling_config)
        else:
            self.sampler = UniformSampler(self.sampling_config)

        # Initialize async computer
        self.async_computer = AsyncMetricComputer(max_async_workers) if enable_async else None

        # Metric registry
        self.registry = MetricRegistry()

        # Storage
        self.collected_metrics = deque(maxlen=10000)
        self.performance_stats = defaultdict(float)

        # State
        self.current_step = 0
        self.total_collection_time = 0.0
        self.collection_count = 0

    def register_metric(
        self,
        name: str,
        computation_func: Callable,
        importance: float = 1.0,
        expensive: bool = False,
        **kwargs
    ):
        """Register a metric for collection."""
        self.registry.register(
            name=name,
            computation_func=computation_func,
            importance=importance,
            cost=10.0 if expensive else 1.0,
            async_safe=expensive,  # Expensive metrics should be async
            **kwargs
        )

    def collect_metrics(self, step: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Collect metrics with optimized sampling and async computation.

        Args:
            step: Current training step
            context: Additional context for metric computation

        Returns:
            Dictionary of collected metrics
        """
        self.current_step = step
        collection_start = time.time()

        collected = {}
        async_metrics = []

        # Process each registered metric
        for metric_name in self.registry.metrics:
            metadata = self.registry.get_metadata(metric_name)
            importance = metadata.get('importance', 1.0)

            # Decide whether to sample this metric
            if not self.sampler.should_sample(metric_name, step, importance):
                continue

            computation_func = self.registry.get_metric_func(metric_name)
            if not computation_func:
                continue

            # Decide whether to compute async
            should_async = (
                self.enable_async and
                metadata.get('async_safe', False) and
                metadata.get('cost', 1.0) > 5.0
            )

            if should_async:
                # Submit for async computation
                self.async_computer.submit_computation(
                    metric_name, computation_func, context or {}
                )
                async_metrics.append(metric_name)
            else:
                # Compute synchronously
                start_time = time.time()
                try:
                    value = computation_func(context or {})
                    execution_time = time.time() - start_time

                    collected[metric_name] = value
                    self.registry.update_timing(metric_name, execution_time)

                    # Store as sample
                    sample = MetricSample(
                        name=metric_name,
                        value=value,
                        timestamp=time.time(),
                        step=step,
                        importance=importance,
                        computation_time=execution_time
                    )
                    self.collected_metrics.append(sample)

                except Exception as e:
                    # Handle errors gracefully
                    collected[f"{metric_name}_error"] = str(e)

        # Collect completed async results
        if self.async_computer:
            async_results = self.async_computer.get_completed_results()
            for sample in async_results:
                collected[sample.name] = sample.value
                self.collected_metrics.append(sample)
                self.registry.update_timing(sample.name, sample.computation_time)

        # Update performance statistics
        total_time = time.time() - collection_start
        self.total_collection_time += total_time
        self.collection_count += 1

        self.performance_stats['avg_collection_time'] = (
            self.total_collection_time / self.collection_count
        )
        self.performance_stats['last_collection_time'] = total_time
        self.performance_stats['metrics_collected'] = len(collected)
        self.performance_stats['async_metrics_submitted'] = len(async_metrics)

        # Adapt sampling rates based on performance
        if self.collection_count % 50 == 0:  # Every 50 collections
            self._adapt_sampling_rates()

        return collected

    def _adapt_sampling_rates(self):
        """Adapt sampling rates based on performance."""
        # Check if we're exceeding overhead threshold
        avg_time = self.performance_stats.get('avg_collection_time', 0)

        performance_metrics = {
            'metric_collection_time': avg_time,
            'overhead_ratio': avg_time,  # This would be relative to training step time
            'expensive_metrics': len(self.registry.get_expensive_metrics())
        }

        self.sampler.update_rates(performance_metrics)

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get detailed collection statistics."""
        stats = dict(self.performance_stats)

        # Add registry statistics
        stats['registered_metrics'] = len(self.registry.metrics)
        stats['expensive_metrics'] = len(self.registry.get_expensive_metrics())

        # Add sampling statistics
        if hasattr(self.sampler, 'current_rates'):
            stats['sampling_rates'] = dict(self.sampler.current_rates)

        # Add async statistics
        if self.async_computer:
            stats['async_queue_size'] = self.async_computer.computation_queue.qsize()
            stats['async_results_pending'] = self.async_computer.result_queue.qsize()

        return stats

    def get_metric_performance(self) -> List[Dict[str, Any]]:
        """Get performance data for each metric."""
        performance = []

        for name, metadata in self.registry.metadata.items():
            performance.append({
                'name': name,
                'importance': metadata.get('importance', 1.0),
                'cost': metadata.get('cost', 1.0),
                'call_count': metadata.get('call_count', 0),
                'avg_time': metadata.get('avg_time', 0.0),
                'total_time': metadata.get('total_time', 0.0)
            })

        # Sort by total time (most expensive first)
        performance.sort(key=lambda x: x['total_time'], reverse=True)
        return performance

    def optimize_collection(self, target_overhead: float = 0.02):
        """Automatically optimize collection to meet target overhead."""
        current_overhead = self.performance_stats.get('avg_collection_time', 0)

        if current_overhead > target_overhead:
            # Reduce sampling rates for expensive metrics
            expensive_metrics = self.registry.get_expensive_metrics()
            for metric_name in expensive_metrics:
                if hasattr(self.sampler, 'current_rates'):
                    self.sampler.current_rates[metric_name] *= 0.5

            print(f"Reduced sampling for {len(expensive_metrics)} expensive metrics")

    def export_metrics(self, last_n: int = 1000) -> List[Dict[str, Any]]:
        """Export collected metrics for analysis."""
        recent_samples = list(self.collected_metrics)[-last_n:]

        return [
            {
                'name': sample.name,
                'value': sample.value,
                'timestamp': sample.timestamp,
                'step': sample.step,
                'importance': sample.importance,
                'computation_time': sample.computation_time,
                'metadata': sample.metadata
            }
            for sample in recent_samples
        ]

    def shutdown(self):
        """Shutdown the metric collector."""
        if self.async_computer:
            self.async_computer.shutdown()


# Convenience functions for common metrics
def create_training_metric_collector() -> OptimizedMetricCollector:
    """Create metric collector optimized for training monitoring."""
    config = SamplingConfig(
        strategy=SamplingStrategy.ADAPTIVE,
        base_sample_rate=0.2,  # 20% base rate
        min_sample_rate=0.05,   # 5% minimum
        max_sample_rate=1.0     # 100% maximum
    )

    collector = OptimizedMetricCollector(
        sampling_config=config,
        enable_async=True,
        max_async_workers=2,
        overhead_threshold=0.03  # 3% overhead target
    )

    return collector


def create_lightweight_collector() -> OptimizedMetricCollector:
    """Create lightweight collector for minimal overhead."""
    config = SamplingConfig(
        strategy=SamplingStrategy.UNIFORM,
        base_sample_rate=0.05,  # 5% sampling
    )

    collector = OptimizedMetricCollector(
        sampling_config=config,
        enable_async=False,
        overhead_threshold=0.01  # 1% overhead target
    )

    return collector


# Common metric computation functions
def compute_gradient_norm(context: Dict[str, Any]) -> float:
    """Compute gradient norm (placeholder)."""
    # This would compute actual gradient norm from model
    return random.uniform(0.1, 2.0)


def compute_memory_usage(context: Dict[str, Any]) -> Dict[str, float]:
    """Compute memory usage metrics."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                'gpu_memory_allocated': allocated,
                'gpu_memory_reserved': reserved
            }
    except:
        pass
    return {'gpu_memory_allocated': 0.0, 'gpu_memory_reserved': 0.0}


def compute_model_statistics(context: Dict[str, Any]) -> Dict[str, float]:
    """Compute model-level statistics (expensive)."""
    # Simulate expensive computation
    time.sleep(0.01)  # 10ms computation
    return {
        'parameter_count': 1000000,
        'activation_magnitude': random.uniform(0.5, 2.0),
        'weight_magnitude': random.uniform(0.8, 1.2)
    }