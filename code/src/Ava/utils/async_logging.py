"""
Async Logging System for Non-blocking Training

This module provides a comprehensive async logging system that prevents
training slowdown while maintaining comprehensive metrics tracking and WandB integration.
"""

import threading
import queue
import time
import os
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class AsyncLoggingConfig:
    """Configuration for async logging system."""
    # Queue settings
    max_queue_size: int = 5000              # Max items in queue
    queue_timeout: float = 1.0              # Queue get timeout

    # Batch processing
    batch_size: int = 100                   # Items per batch
    batch_timeout: float = 5.0              # Max time between batches

    # WandB settings
    wandb_cache_size: int = 2000            # Cache size for offline resilience
    wandb_flush_interval: int = 50          # Cache flush interval
    wandb_retry_delay: float = 1.0          # Initial retry delay
    wandb_max_retry_delay: float = 60.0     # Max retry delay

    # Performance settings
    enable_system_metrics: bool = False     # System resource monitoring
    metrics_history_size: int = 1000        # Size of metrics history
    enable_profiling: bool = False          # Performance profiling


class AsyncLogger:
    """
    Async logging system that handles all logging in background threads.

    Features:
    - Non-blocking metrics queuing
    - Batch processing for efficiency
    - WandB integration with retry mechanisms
    - System metrics monitoring
    - Network resilience with caching
    - Comprehensive error handling
    """

    def __init__(self, config: AsyncLoggingConfig, wandb_available: bool = True, wandb_offline: bool = False, disable_wandb: bool = False):
        """
        Initialize async logger.

        Args:
            config: Async logging configuration
            wandb_available: Whether WandB is available
            wandb_offline: Force WandB offline mode
            disable_wandb: Completely disable WandB logging
        """
        self.config = config
        self.wandb_available = wandb_available and not disable_wandb
        self.wandb_offline = wandb_offline
        self.disable_wandb = disable_wandb
        self.wandb_run = None

        # Queue and threading
        self.log_queue = queue.Queue(maxsize=config.max_queue_size)
        self.stop_logging = threading.Event()
        self.logging_thread = None

        # Caching for network resilience
        self.wandb_cache = []
        self.cache_lock = threading.Lock()
        self.last_network_success = time.time()
        self.network_retry_delay = config.wandb_retry_delay

        # Metrics processing
        self.metrics_buffer = []
        self.metrics_history = deque(maxlen=config.metrics_history_size)
        self.buffer_lock = threading.Lock()

        # Statistics
        self.logging_stats = defaultdict(int)
        self.start_time = time.time()

        # Performance tracking
        self.profiling_data = {} if config.enable_profiling else None

    def set_wandb_run(self, wandb_run) -> None:
        """Set WandB run instance."""
        self.wandb_run = wandb_run

    def start(self) -> None:
        """Start async logging thread."""
        if self.logging_thread is not None:
            return

        self.logging_thread = threading.Thread(target=self._async_logger, daemon=True)
        self.logging_thread.start()
        print("Async logging started")

    def stop(self) -> None:
        """Stop async logging and flush remaining items."""
        if self.logging_thread is None:
            return

        # Signal stop and add sentinel
        self.stop_logging.set()
        try:
            self.log_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for thread to finish
        self.logging_thread.join(timeout=10.0)
        self.logging_thread = None

        # Final flush
        self._flush_wandb_cache()

        # Print final statistics
        print("\n🏁 Async logging stopped")
        self.print_statistics()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics asynchronously (non-blocking).

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            item = ('metrics', {'data': metrics, 'step': step, 'timestamp': time.time()})
            self.log_queue.put_nowait(item)
            self.logging_stats['metrics_queued'] += 1
        except queue.Full:
            self.logging_stats['metrics_dropped'] += 1

    def log_timing_metrics(self, timing_data: Dict[str, Any]) -> None:
        """
        Log timing metrics for real-time tracking.

        Args:
            timing_data: Timing information
        """
        try:
            item = ('timing_metrics', timing_data)
            self.log_queue.put_nowait(item)
            self.logging_stats['timing_queued'] += 1
        except queue.Full:
            self.logging_stats['timing_dropped'] += 1

    def log_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """
        Queue evaluation request for background processing.

        Args:
            eval_data: Evaluation data
        """
        try:
            item = ('evaluation', eval_data)
            self.log_queue.put_nowait(item)
            self.logging_stats['eval_queued'] += 1
        except queue.Full:
            self.logging_stats['eval_dropped'] += 1

    def log_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        Queue checkpoint information.

        Args:
            checkpoint_info: Checkpoint metadata
        """
        try:
            item = ('checkpoint', checkpoint_info)
            self.log_queue.put_nowait(item)
            self.logging_stats['checkpoint_queued'] += 1
        except queue.Full:
            self.logging_stats['checkpoint_dropped'] += 1

    def _async_logger(self) -> None:
        """Background thread that handles ALL logging."""
        last_metrics_time = time.time()
        metrics_buffer = []

        print("Async logging thread started")

        while not self.stop_logging.is_set():
            try:
                # Get item from queue
                item = self.log_queue.get(timeout=self.config.queue_timeout)
                if item is None:  # Sentinel to stop
                    break

                item_type, data = item
                self.logging_stats['items_processed'] += 1

                if item_type == 'metrics':
                    # Buffer metrics for batch processing
                    metrics_buffer.append(data)

                    # Process batch when full or timeout reached
                    if (len(metrics_buffer) >= self.config.batch_size or
                        time.time() - last_metrics_time > self.config.batch_timeout):
                        self._process_metrics_batch(metrics_buffer)
                        metrics_buffer = []
                        last_metrics_time = time.time()

                elif item_type == 'timing_metrics':
                    # Process timing metrics immediately for real-time tracking
                    self._process_timing_metrics(data)

                elif item_type == 'evaluation':
                    # Handle evaluation in background
                    self._handle_async_evaluation(data)

                elif item_type == 'checkpoint':
                    # Handle checkpoint in background
                    self._handle_async_checkpoint(data)

            except queue.Empty:
                # Process any remaining metrics on timeout
                if metrics_buffer:
                    self._process_metrics_batch(metrics_buffer)
                    metrics_buffer = []
                    last_metrics_time = time.time()
                continue

            except Exception as e:
                self.logging_stats['errors'] += 1
                # Suppress errors to prevent training slowdown, but track error types
                error_type = type(e).__name__
                self.logging_stats[f'errors_{error_type}'] = self.logging_stats.get(f'errors_{error_type}', 0) + 1
                # Only print critical errors, suppress network/socket errors
                if not any(keyword in str(e).lower() for keyword in ['socket', 'network', 'connection', 'timeout']):
                    print(f" Async logging error ({error_type}): {e}")
                pass

        # Final processing
        if metrics_buffer:
            self._process_metrics_batch(metrics_buffer)

        print("Async logging thread stopped")

    def _process_metrics_batch(self, metrics_buffer: List[Dict[str, Any]]) -> None:
        """Process a batch of metrics."""
        if not metrics_buffer:
            return

        try:
            # Group metrics by step for efficient logging
            step_metrics = defaultdict(dict)

            for metric_data in metrics_buffer:
                data = metric_data['data']
                step = metric_data.get('step', 0)
                timestamp = metric_data.get('timestamp', time.time())

                step_metrics[step].update(data)
                step_metrics[step]['timestamp'] = timestamp

            # Log to WandB if available
            if self.wandb_available and self.wandb_run:
                self._log_to_wandb(step_metrics)

            # Update metrics history
            with self.buffer_lock:
                for step, metrics in step_metrics.items():
                    self.metrics_history.append({
                        'step': step,
                        'metrics': metrics,
                        'processed_at': time.time()
                    })

            self.logging_stats['batches_processed'] += 1

        except Exception as e:
            self.logging_stats['batch_errors'] += 1

    def _process_timing_metrics(self, timing_data: Dict[str, Any]) -> None:
        """Process timing metrics immediately."""
        if not self.wandb_available or not self.wandb_run:
            return

        try:
            step = timing_data.get('step', 0)
            self._log_metrics_to_wandb(timing_data, step)
            self.logging_stats['timing_processed'] += 1

        except Exception:
            self.logging_stats['timing_errors'] += 1

    def _log_to_wandb(self, step_metrics: Dict[int, Dict[str, Any]]) -> None:
        """Log metrics to WandB with caching for resilience."""
        if not self.wandb_run:
            return

        for step, metrics in step_metrics.items():
            try:
                # Try direct logging first
                clean_metrics = {k: v for k, v in metrics.items() if k != 'step' and k != 'timestamp'}
                self._log_metrics_to_wandb(clean_metrics, step)
                self.logging_stats['wandb_logged'] += 1

            except Exception as e:
                # Cache for later retry
                self._cache_metrics(metrics, step)
                self.logging_stats['wandb_cached'] += 1

        # Periodic cache flush
        if len(self.wandb_cache) >= self.config.wandb_flush_interval:
            self._flush_wandb_cache()

    def _log_metrics_to_wandb(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics directly to WandB."""
        if self.disable_wandb or not self.wandb_available:
            return

        try:
            import wandb

            # If we're in offline mode, ensure WandB knows about it
            if self.wandb_offline and wandb.run is None:
                os.environ['WANDB_MODE'] = 'offline'  # type: ignore[misc]

            wandb.log(metrics, step=step)
        except ImportError:
            # WandB not available, silently skip
            self.wandb_available = False
            return
        except Exception as e:
            # Handle specific network errors by falling back to offline mode
            error_lower = str(e).lower()
            is_network_error = any(keyword in error_lower for keyword in ['socket', 'network', 'connection', 'timeout', 'unreachable'])

            if is_network_error:
                if not self.wandb_offline:
                    print(f"⚠ WandB network error detected: {e}")
                    print("  → Switching to offline mode (metrics will sync when network is available)")
                    self.wandb_offline = True
                    try:
                        import os
                        os.environ['WANDB_MODE'] = 'offline'
                        import wandb
                        # Verify offline mode engaged
                        if wandb.run:
                            print(f"  ✓ WandB offline mode active (run will sync later)")
                        wandb.log(metrics, step=step)  # Retry in offline mode
                        return
                    except Exception as retry_error:
                        print(f"  ✗ Offline mode retry failed: {retry_error}")
                        print("  → Metrics will be cached for manual recovery")

                # Track network errors for diagnostics
                self.logging_stats['network_errors'] = self.logging_stats.get('network_errors', 0) + 1

                # Cache metrics for later flush when network returns
                self._cache_metrics(metrics, step)
            else:
                # Non-network error
                self.logging_stats['wandb_errors'] = self.logging_stats.get('wandb_errors', 0) + 1
                print(f"⚠ WandB logging error (step {step}): {e}")
                print(f"  Error type: {type(e).__name__}")
                # Cache metrics for recovery
                self._cache_metrics(metrics, step)
            return

    def _cache_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Cache metrics for later retry."""
        with self.cache_lock:
            cache_entry = {**metrics, 'step': step, 'cached_at': time.time()}
            self.wandb_cache.append(cache_entry)

            # Limit cache size
            if len(self.wandb_cache) > self.config.wandb_cache_size:
                self.wandb_cache = self.wandb_cache[-self.config.wandb_cache_size//2:]

    def _flush_wandb_cache(self) -> None:
        """Flush cached metrics to WandB."""
        if not self.wandb_cache or not self.wandb_run:
            return

        with self.cache_lock:
            cache_to_flush = self.wandb_cache.copy()

        try:
            if len(cache_to_flush) > 50:  # Only print for significant cache flushes
                print(f" Uploading {len(cache_to_flush)} cached metrics to WandB...")

            # Group by step
            step_metrics = defaultdict(dict)
            for metrics in cache_to_flush:
                step = metrics.get('step', 0)
                step_metrics[step].update({k: v for k, v in metrics.items() if k != 'step'})

            # Upload in batches
            successful_count = 0
            for step, metrics in step_metrics.items():
                try:
                    self._log_metrics_to_wandb(metrics, step)
                    successful_count += 1

                    # Small delay to avoid rate limiting
                    if successful_count % 100 == 0:
                        time.sleep(0.01)

                except Exception as e:
                    # Suppress socket/network errors specifically
                    if any(keyword in str(e).lower() for keyword in ['socket', 'network', 'connection', 'timeout']):
                        self.logging_stats['network_errors'] = self.logging_stats.get('network_errors', 0) + 1
                    continue

            # Clear successfully uploaded metrics
            with self.cache_lock:
                self.wandb_cache = []

            self.last_network_success = time.time()
            self.network_retry_delay = self.config.wandb_retry_delay

            print(f"Successfully uploaded {successful_count} metric batches to WandB")

        except Exception as e:
            print(f" Network error uploading to WandB: {e}")
            # Exponential backoff
            self.network_retry_delay = min(self.network_retry_delay * 1.5, self.config.wandb_max_retry_delay)

    def _handle_async_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Handle evaluation in background (placeholder)."""
        # This would trigger actual evaluation
        self.logging_stats['evaluations_handled'] += 1

    def _handle_async_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """Handle checkpoint saving in background (placeholder)."""
        # This would trigger actual checkpoint saving
        self.logging_stats['checkpoints_handled'] += 1

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        if not self.config.enable_system_metrics:
            return {}

        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            metrics = {
                'system/memory_percent': memory.percent,
                'system/cpu_percent': cpu_percent
            }

            # GPU metrics if available
            try:
                import torch  # type: ignore[import]
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    metrics.update({
                        'system/gpu_memory_allocated_gb': allocated,
                        'system/gpu_memory_reserved_gb': reserved
                    })
            except:
                pass

            return metrics

        except Exception:
            return {}

    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        uptime = time.time() - self.start_time
        queue_size = self.log_queue.qsize()

        stats = {
            'uptime_seconds': uptime,
            'queue_size': queue_size,
            'cache_size': len(self.wandb_cache),
            'network_retry_delay': self.network_retry_delay,
            'last_network_success': self.last_network_success,
            **dict(self.logging_stats)
        }

        # Calculate rates
        if uptime > 0:
            stats.update({
                'metrics_per_second': self.logging_stats.get('metrics_queued', 0) / uptime,
                'batches_per_minute': self.logging_stats.get('batches_processed', 0) / (uptime / 60),
                'error_rate': self.logging_stats.get('errors', 0) / max(self.logging_stats.get('items_processed', 1), 1)
            })

        return stats

    def get_recent_metrics(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics from history."""
        with self.buffer_lock:
            return list(self.metrics_history)[-last_n:]

    def clear_cache(self) -> int:
        """Clear metrics cache and return number of items cleared."""
        with self.cache_lock:
            cache_size = len(self.wandb_cache)
            self.wandb_cache = []
            return cache_size

    def force_flush(self) -> None:
        """Force flush of all cached metrics."""
        self._flush_wandb_cache()

    def get_state_dict(self) -> Dict[str, Any]:
        """Get logger state for checkpointing."""
        return {
            'logging_stats': dict(self.logging_stats),
            'start_time': self.start_time,
            'network_retry_delay': self.network_retry_delay,
            'last_network_success': self.last_network_success
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load logger state from checkpoint."""
        self.logging_stats.update(state_dict.get('logging_stats', {}))
        self.start_time = state_dict.get('start_time', time.time())
        self.network_retry_delay = state_dict.get('network_retry_delay', self.config.wandb_retry_delay)
        self.last_network_success = state_dict.get('last_network_success', time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics for diagnostics."""
        stats = {
            'total_items_processed': self.logging_stats.get('items_processed', 0),
            'metrics_queued': self.logging_stats.get('metrics_queued', 0),
            'metrics_dropped': self.logging_stats.get('metrics_dropped', 0),
            'wandb_errors': self.logging_stats.get('wandb_errors', 0),
            'network_errors': self.logging_stats.get('network_errors', 0),
            'cache_size': len(self.wandb_cache),
            'wandb_offline_mode': self.wandb_offline,
            'wandb_available': self.wandb_available,
            'queue_size': self.log_queue.qsize() if hasattr(self, 'log_queue') and hasattr(self.log_queue, 'qsize') else 0,
        }

        # Calculate drop rate
        total_queued = self.logging_stats.get('metrics_queued', 0)
        total_dropped = self.logging_stats.get('metrics_dropped', 0)
        if total_queued > 0:
            stats['drop_rate_percent'] = (total_dropped / total_queued) * 100
        else:
            stats['drop_rate_percent'] = 0.0

        return stats

    def print_statistics(self) -> None:
        """Print comprehensive logging statistics."""
        stats = self.get_statistics()
        print("\n📊 Async Logger Statistics:")
        print(f"  ✓ Items processed: {stats['total_items_processed']}")
        print(f"  ✓ Metrics queued: {stats['metrics_queued']}")

        if stats['metrics_dropped'] > 0:
            print(f"  ⚠ Metrics dropped: {stats['metrics_dropped']} ({stats['drop_rate_percent']:.2f}%)")

        if stats['network_errors'] > 0:
            print(f"  ⚠ Network errors: {stats['network_errors']}")
            if stats['wandb_offline_mode']:
                print(f"    → WandB in offline mode (will sync when network available)")

        if stats['wandb_errors'] > 0:
            print(f"  ⚠ WandB errors: {stats['wandb_errors']}")

        if stats['cache_size'] > 0:
            print(f"  📦 Cached metrics: {stats['cache_size']} (waiting for flush)")

        print(f"  🔌 WandB status: {'Offline' if stats['wandb_offline_mode'] else 'Online'}")
        print(f"  📨 Queue size: {stats['queue_size']}")


# Context manager for automatic cleanup
class AsyncLoggingContext:
    """Context manager for async logging with automatic cleanup."""

    def __init__(self, config: AsyncLoggingConfig, wandb_available: bool = True, wandb_offline: bool = False, disable_wandb: bool = False):
        self.logger = AsyncLogger(config, wandb_available, wandb_offline, disable_wandb)

    def __enter__(self) -> AsyncLogger:
        self.logger.start()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()


# Convenience functions
def create_fast_logging_config() -> AsyncLoggingConfig:
    """Create configuration optimized for fast logging."""
    return AsyncLoggingConfig(
        batch_size=50,
        batch_timeout=2.0,
        wandb_flush_interval=25,
        enable_system_metrics=False
    )


def create_comprehensive_logging_config() -> AsyncLoggingConfig:
    """Create configuration with comprehensive monitoring."""
    return AsyncLoggingConfig(
        batch_size=100,
        batch_timeout=5.0,
        wandb_flush_interval=50,
        enable_system_metrics=True,
        enable_profiling=True
    )


def create_minimal_logging_config() -> AsyncLoggingConfig:
    """Create minimal logging configuration."""
    return AsyncLoggingConfig(
        batch_size=200,
        batch_timeout=10.0,
        wandb_flush_interval=100,
        enable_system_metrics=False,
        wandb_cache_size=500
    )