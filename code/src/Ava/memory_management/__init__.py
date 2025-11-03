"""
Ava Memory Management Module

Comprehensive memory monitoring, optimization, and management utilities
for training large language models efficiently.

This module provides:
- Real-time GPU/CPU memory monitoring with NVML support
- Proactive OOM prevention and emergency cleanup
- Memory optimization tracking (gradient checkpointing, flash attention, etc.)
- Detailed memory breakdowns and profiling
- Activation memory estimation for transformer models

Components:
    MemoryMonitor: Main class for memory monitoring and management

Example:
    >>> from Ava.memory_management import MemoryMonitor
    >>> monitor = MemoryMonitor(target_utilization=0.90)
    >>> health = monitor.check_memory_health(batch_size=12)
    >>> print(f"Status: {health['status']}, Usage: {health['gpu_utilization']:.1%}")

For detailed documentation, see:
    - docs/03_MEMORY_OPTIMIZATION.md - Complete optimization guide
    - docs/01_ARCHITECTURE.md - System architecture
"""

from .memory_monitor import MemoryMonitor

__all__ = ['MemoryMonitor']

__version__ = '1.0.0'
__author__ = 'Ava Team'
