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

    def cleanup_gpu_memory(self, aggressive: bool = False) -> Dict[str, float]:
        """
        Comprehensive GPU memory cleanup function.

        Args:
            aggressive: Enable more aggressive cleanup procedures

        Returns:
            Dict with memory statistics after cleanup
        """
        stats = {'before_allocated': 0.0, 'before_cached': 0.0,
                'after_allocated': 0.0, 'after_cached': 0.0}

        try:
            if torch.cuda.is_available():
                # Record initial memory state
                stats['before_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['before_cached'] = torch.cuda.memory_reserved() / 1024**3

                print("Cleaning up GPU memory...")

                # Clear PyTorch CUDA cache
                torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                # Additional CUDA cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # Aggressive cleanup if requested
                    total_memory = self._get_total_gpu_memory()
                    if total_memory > 0 and (aggressive or stats['before_cached'] > self.emergency_threshold * total_memory):
                        print(" Performing aggressive GPU cleanup...")
                        try:
                            torch.cuda.ipc_collect()
                        except Exception as e:
                            print(f"Warning: IPC collect failed: {e}")
                        torch.cuda.empty_cache()

                        # Additional aggressive cleanup
                        for _ in range(5):
                            gc.collect()
                            torch.cuda.empty_cache()
                            time.sleep(0.1)

                        # Force memory pool cleanup
                        try:
                            torch.cuda.memory._set_per_process_memory_fraction(0.0)  # type: ignore[attr-defined]
                            torch.cuda.empty_cache()
                            torch.cuda.memory._set_per_process_memory_fraction(1.0)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        # Final emergency cleanup rounds
                        for _ in range(3):
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            time.sleep(0.2)

                # Record final memory state
                stats['after_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['after_cached'] = torch.cuda.memory_reserved() / 1024**3

                print(f"GPU Memory after cleanup: {stats['after_allocated']:.2f}GB allocated, {stats['after_cached']:.2f}GB cached")

                # Check if emergency cleanup is still needed
                if stats['after_cached'] > 1.0:
                    print(f" High cache usage detected: {stats['after_cached']:.2f}GB still cached")

        except Exception as e:
            print(f" Error during GPU cleanup: {e}")
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
        """Get current GPU memory statistics."""
        stats = {}

        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)

                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = props.total_memory / 1024**3

                stats.update({
                    'device_name': props.name,
                    'total_memory_gb': total,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'free_gb': max(0.0, total - reserved),  # Ensure non-negative
                    'utilization_percent': (allocated / total) * 100 if total > 0 else 0.0,
                    'cache_percent': (reserved / total) * 100 if total > 0 else 0.0,
                })

        except Exception as e:
            stats['error'] = str(e)

        return stats

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
                        print(f" High GPU memory usage: {utilization*100:.1f}% (>{threshold*100:.1f}%)")
                        return True

        except Exception as e:
            print(f"Warning: Memory monitoring failed: {e}")

        return False

    def emergency_cleanup(self) -> None:
        """Perform emergency cleanup when memory is critically low."""
        print(" Emergency GPU memory cleanup initiated!")

        # Multiple rounds of aggressive cleanup
        for i in range(5):
            print(f" Emergency cleanup round {i+1}/5")
            self.cleanup_gpu_memory(aggressive=True)

            # Check if cleanup was successful
            if not self.monitor_memory(threshold=0.99):  # Check if still above threshold after cleanup
                print("Emergency cleanup successful!")
                break

            time.sleep(0.5)
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


def get_memory_stats() -> Dict[str, Any]:
    """Legacy function for backwards compatibility."""
    return get_memory_manager().get_memory_stats()


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