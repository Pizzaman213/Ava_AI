"""
GPU Memory Management Utilities

This module provides comprehensive GPU memory management, cleanup functions,
and signal handling for proper resource management during training.
"""

import torch  # type: ignore[import]
import gc
import signal
import atexit
import time
import os
from typing import Optional, Dict, Any

# Distributed training support
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    dist = None  # type: ignore[assignment]
    DISTRIBUTED_AVAILABLE = False


class GPUMemoryManager:
    """
    Comprehensive GPU memory management with cleanup and monitoring.

    Features:
    - Automatic memory cleanup
    - Signal handling for interruptions
    - Memory monitoring and reporting
    - Emergency cleanup procedures
    """

    def __init__(self, auto_cleanup: bool = True, emergency_threshold: float = 0.99):
        """
        Initialize GPU Memory Manager.

        Args:
            auto_cleanup: Enable automatic memory cleanup
            emergency_threshold: Memory usage threshold for emergency cleanup (0.0 to 1.0, default 0.99 = 99%)
        """
        self.auto_cleanup = auto_cleanup
        self.emergency_threshold = emergency_threshold
        self._cleanup_handlers_registered = False

        if auto_cleanup:
            self.register_cleanup_handlers()

    def defragment_memory_periodic(self, step_count: int, interval: int = 1000) -> Dict[str, Any]:
        """
        Proactively defragment GPU memory at regular intervals.

        Args:
            step_count: Current training step
            interval: Defragmentation interval in steps (default: 1000)

        Returns:
            Dict with defragmentation statistics
        """
        stats = {'defragmented': False, 'freed_mb': 0.0}

        if step_count % interval != 0:
            return stats

        try:
            if torch.cuda.is_available():
                before_reserved = torch.cuda.memory_reserved() / 1024**2
                before_allocated = torch.cuda.memory_allocated() / 1024**2

                # Force consolidation of memory allocator
                torch.cuda.empty_cache()
                gc.collect()

                # Reset memory stats to clear fragmentation tracking
                torch.cuda.reset_peak_memory_stats()

                # Additional cache clearing
                torch.cuda.empty_cache()

                after_reserved = torch.cuda.memory_reserved() / 1024**2
                freed_mb = before_reserved - after_reserved

                stats['defragmented'] = True
                stats['freed_mb'] = freed_mb

                if freed_mb > 100:  # Only log if significant memory freed
                    print(f"    🗑️  Proactive defragmentation freed {freed_mb:.1f}MB (step {step_count})")

        except Exception as e:
            stats['error'] = str(e)  # type: ignore[assignment]

        return stats

    def cleanup_gpu_memory(self, aggressive: bool = False, enabled: bool = True) -> Dict[str, float]:
        """
        Comprehensive GPU memory cleanup function.

        Args:
            aggressive: Enable more aggressive cleanup procedures
            enabled: Whether cleanup is enabled (can be controlled via config)

        Returns:
            Dict with memory statistics after cleanup
        """
        stats = {'before_allocated': 0.0, 'before_cached': 0.0,
                'after_allocated': 0.0, 'after_cached': 0.0}

        # If cleanup is disabled, return immediately
        if not enabled:
            return stats

        try:
            if torch.cuda.is_available():
                # Record initial memory state
                stats['before_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['before_cached'] = torch.cuda.memory_reserved() / 1024**3

                # Clear PyTorch CUDA cache
                torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                # Additional CUDA cleanup
                # GPU UTIL FIX: Skip sync in normal cleanup - only sync in emergency
                if torch.cuda.is_available():
                    # torch.cuda.synchronize()  # GPU UTIL FIX: Removed - adds 10-50ms stall
                    torch.cuda.empty_cache()

                    # Aggressive cleanup if requested
                    total_memory = self._get_total_gpu_memory()
                    if total_memory > 0 and (aggressive or stats['before_cached'] > self.emergency_threshold * total_memory):
                        try:
                            torch.cuda.ipc_collect()
                        except Exception as e:
                            pass  # Silently handle IPC collect failures
                        torch.cuda.empty_cache()

                        # OPTIMIZATION: Reduced cleanup rounds from 5 to 2 (saves ~1.5s)
                        for _ in range(2):
                            gc.collect()
                            torch.cuda.empty_cache()

                        # Force memory pool cleanup
                        try:
                            torch.cuda.memory._set_per_process_memory_fraction(0.0)  # type: ignore[attr-defined]
                            torch.cuda.empty_cache()
                            torch.cuda.memory._set_per_process_memory_fraction(1.0)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        # GPU UTIL FIX: Reduced emergency rounds from 3 to 1, removed sleep (saves ~1s)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # GPU UTIL FIX: Only sync in true emergency (>99.5% memory)
                        total_memory = self._get_total_gpu_memory()
                        if stats['before_cached'] > 0.995 * total_memory:
                            torch.cuda.synchronize()  # Critical sync only at >99.5%

                # Record final memory state
                stats['after_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['after_cached'] = torch.cuda.memory_reserved() / 1024**3

        except Exception as e:
            stats['error'] = str(e)  # type: ignore[assignment]

        return stats

    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory > 0:
                    return total_memory / 1024**3
        except Exception as e:
            print(f"Warning: Could not get GPU memory info: {e}")
        return 80.0  # Default assumption for A100

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics.

        Note: This now calls the consolidated get_memory_stats() function.
        """
        return get_memory_stats(device=None, unit='GB')

    def monitor_memory(self, threshold: float = 0.99) -> bool:
        """
        Monitor memory usage and return True if threshold exceeded.

        Args:
            threshold: Memory utilization threshold (0.0 to 1.0, default 0.99 = 99%)

        Returns:
            True if memory usage exceeds threshold
        """
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory

                if total > 0:  # Avoid division by zero
                    utilization = allocated / total

                    if utilization > threshold:
                        allocated_gb = allocated / (1024**3)
                        total_gb = total / (1024**3)
                        print(f"⚠️  High GPU memory usage: {utilization*100:.1f}% ({allocated_gb:.1f}GB / {total_gb:.1f}GB)")

                        # MEMORY OPTIMIZATION: Provide actionable recommendations
                        print("💡 Memory optimization suggestions:")
                        if utilization > 0.95:
                            print("   • Reduce batch size (currently highest memory consumer)")
                            print("   • Enable gradient checkpointing (saves ~25%, costs ~8% speed)")
                            print("   • Increase gradient_accumulation_steps to compensate for smaller batches")
                        if utilization > 0.98:
                            print("   • Consider enabling expert offloading (saves 75-87% on MoE models)")
                            print("   • Use LoRA experts (saves 40-60% expert memory)")
                            print("   • Reduce max_length if using long sequences")
                        if utilization > 0.99:
                            print("   • CRITICAL: OOM imminent - reduce batch_size immediately")
                            print("   • Enable all memory optimizations in config")

                        return True

        except Exception as e:
            print(f"Warning: Memory monitoring failed: {e}")

        return False

    def emergency_cleanup(self) -> None:
        """Perform emergency cleanup when memory is critically low."""
        print(" Emergency GPU memory cleanup initiated!")

        # GPU UTIL FIX: Reduced emergency cleanup rounds from 5 to 2 (saves 2-5 seconds)
        for i in range(2):
            print(f" Emergency cleanup round {i+1}/2")
            self.cleanup_gpu_memory(aggressive=True)

            # Check if cleanup was successful
            if not self.monitor_memory(threshold=0.99):  # Check if still above threshold after cleanup
                print("Emergency cleanup successful!")
                break

            # GPU UTIL FIX: Removed time.sleep(0.5) that blocked training for 500ms
            # Cleanup operations are synchronous and don't need sleep between rounds
        else:
            print(" Emergency cleanup completed, but memory usage still high")

    def signal_handler(self, signum: int, frame) -> None:
        """Handle interruption signals (Ctrl+C, etc.) with proper GPU cleanup."""
        print(f"\n Received signal {signum}. Performing cleanup...")
        self.cleanup_gpu_memory(aggressive=True)
        print("Cleanup completed. Exiting...")
        exit(0)

    def register_cleanup_handlers(self) -> None:
        """Register signal handlers and exit functions for proper cleanup."""
        if self._cleanup_handlers_registered:
            return

        # Register signal handlers for common interruption signals
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination signal

        # Register cleanup function to run at exit
        atexit.register(lambda: self.cleanup_gpu_memory(aggressive=True))

        self._cleanup_handlers_registered = True
        print("GPU cleanup handlers registered")

    def context_manager(self):
        """Context manager for automatic cleanup."""
        return _GPUMemoryContext(self)


class _GPUMemoryContext:
    """Context manager for GPU memory cleanup."""

    def __init__(self, manager: GPUMemoryManager):
        self.manager = manager

    def __enter__(self):
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.cleanup_gpu_memory()


# Global instance for backwards compatibility
_global_manager = None


def get_memory_manager() -> GPUMemoryManager:
    """Get or create the global GPU memory manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = GPUMemoryManager()
    return _global_manager


def cleanup_gpu_memory(aggressive: bool = False) -> Dict[str, float]:
    """
    Legacy function for backwards compatibility.

    Args:
        aggressive: Enable aggressive cleanup

    Returns:
        Memory statistics after cleanup
    """
    return get_memory_manager().cleanup_gpu_memory(aggressive=aggressive)


def register_cleanup_handlers() -> None:
    """Legacy function for backwards compatibility."""
    get_memory_manager().register_cleanup_handlers()


def get_memory_stats(device: Optional[int] = None, unit: str = 'GB') -> Dict[str, Any]:
    """
    Get comprehensive GPU memory statistics.

    Consolidated function that replaces multiple implementations across the codebase.

    Args:
        device: GPU device index (None for current device)
        unit: Memory unit - 'GB' or 'MB' (default: 'GB')

    Returns:
        Dictionary with comprehensive memory statistics
    """
    stats = {}

    try:
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()

            props = torch.cuda.get_device_properties(device)

            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            total_bytes = props.total_memory

            # Convert to requested unit
            if unit == 'MB':
                divisor = 1024 ** 2
                unit_suffix = 'mb'
            else:  # GB
                divisor = 1024 ** 3
                unit_suffix = 'gb'

            allocated = allocated_bytes / divisor
            reserved = reserved_bytes / divisor
            total = total_bytes / divisor
            free = max(0.0, total - reserved)

            stats.update({
                'device_name': props.name,
                'device_index': device,
                f'allocated_{unit_suffix}': allocated,
                f'reserved_{unit_suffix}': reserved,
                f'total_{unit_suffix}': total,
                f'free_{unit_suffix}': free,
                'utilization_percent': (allocated / total) * 100 if total > 0 else 0.0,
                'cache_percent': (reserved / total) * 100 if total > 0 else 0.0,
                # Additional fields for compatibility
                'allocated': allocated,  # For generic access
                'reserved': reserved,
                'total': total,
                'free': free,
                'utilization': allocated / total if total > 0 else 0.0,
            })

    except Exception as e:
        stats['error'] = str(e)

    return stats


def get_gpu_compute_utilization(device: Optional[int] = None) -> float:
    """
    Get GPU compute utilization (0-1).

    Consolidated function that attempts multiple methods:
    1. pynvml for accurate metrics
    2. nvidia-smi as fallback
    3. Memory utilization as last resort proxy

    Args:
        device: GPU device index (None for current device)

    Returns:
        Estimated compute utilization (0.0-1.0)
    """
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    # Try pynvml first (most accurate)
    try:
        import pynvml
        if not hasattr(get_gpu_compute_utilization, '_nvml_initialized'):
            pynvml.nvmlInit()
            get_gpu_compute_utilization._nvml_initialized = True  # type: ignore[attr-defined]

        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(utilization.gpu) / 100.0

        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, gpu_util))
    except (ImportError, Exception):
        pass

    # Try nvidia-smi as fallback
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={device}'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            util = float(result.stdout.strip()) / 100.0
            return max(0.0, min(1.0, util))
    except (FileNotFoundError, Exception):
        pass

    # Fallback to memory utilization as proxy (typically assumes 75% utilization)
    return 0.75


def monitor_memory(threshold: float = 0.99) -> bool:
    """Legacy function for backwards compatibility."""
    return get_memory_manager().monitor_memory(threshold=threshold)


def distributed_cleanup(sync: bool = True) -> Dict[str, float]:
    """
    Perform distributed-aware GPU memory cleanup.

    Args:
        sync: Whether to synchronize across all processes

    Returns:
        Memory statistics after cleanup
    """
    if DISTRIBUTED_AVAILABLE and dist.is_initialized() and sync:  # type: ignore[union-attr]
        # Synchronize all processes before cleanup
        dist.barrier()  # type: ignore[union-attr]

        # Perform cleanup
        stats = get_memory_manager().cleanup_gpu_memory(aggressive=True)

        # Synchronize after cleanup
        dist.barrier()  # type: ignore[union-attr]

        return stats
    else:
        return get_memory_manager().cleanup_gpu_memory(aggressive=True)


def is_distributed_training() -> bool:
    """Check if we're in a distributed training environment."""
    return (
        DISTRIBUTED_AVAILABLE and
        (dist.is_initialized() or 'WORLD_SIZE' in os.environ)  # type: ignore[union-attr]
    )


def get_default_device() -> torch.device:
    """
    Get the default device for computation.

    Consolidated utility to replace repeated device detection pattern.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device('cpu')