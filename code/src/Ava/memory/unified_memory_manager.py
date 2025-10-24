"""
Unified Memory Management System

Consolidates all memory management functionality:
- Episodic memory (continual learning)
- GPU memory monitoring and OOM prevention
- Memory cleanup utilities

From files:
- code/src/Ava/memory/episodic_memory.py
- code/src/Ava/training/memory_monitor.py
- code/src/Ava/utils/gpu_memory.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
import signal
import atexit
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
import random
import math

logger = logging.getLogger(__name__)

# Try NVML for GPU monitoring
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except Exception:
        NVML_AVAILABLE = False
        pynvml = None
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

# Try distributed support
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    dist = None
    DISTRIBUTED_AVAILABLE = False


# ============================================================================
# Episodic Memory (Continual Learning)
# ============================================================================

@dataclass
class MemoryEntry:
    """Single episodic memory entry for continual learning."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    task_id: int
    importance_score: float
    timestamp: int
    gradient_norm: Optional[float] = None
    loss_value: Optional[float] = None


class MemoryRetriever(nn.Module):
    """Retrieves relevant episodic memories."""

    def __init__(self, hidden_size: int, memory_size: int, retrieval_method: str = "cosine"):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.retrieval_method = retrieval_method
        self.query_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_bank: List[MemoryEntry],
        k: int = 10
    ) -> Tuple[List[MemoryEntry], torch.Tensor]:
        """Retrieve k most relevant memories."""
        if not memory_bank:
            return [], torch.tensor([])

        query = self.query_proj(query_states.mean(dim=1))
        similarities = []

        for memory in memory_bank:
            memory_repr = memory.hidden_states.mean(dim=1)

            if self.retrieval_method == "cosine":
                sim = F.cosine_similarity(
                    query.unsqueeze(1), memory_repr.unsqueeze(0), dim=-1
                ).mean()
            elif self.retrieval_method == "euclidean":
                sim = -torch.norm(query - memory_repr, dim=-1).mean()
            else:
                sim = torch.mm(query, memory_repr.t()).mean()

            similarities.append(sim.item())

        similarities = torch.tensor(similarities)
        top_k_indices = torch.topk(similarities, min(k, len(memory_bank))).indices

        retrieved_memories = [memory_bank[i] for i in top_k_indices]
        retrieved_scores = similarities[top_k_indices]

        return retrieved_memories, retrieved_scores


class EpisodicMemoryBank(nn.Module):
    """Episodic memory bank for continual learning."""

    def __init__(
        self,
        capacity: int = 1000,
        hidden_size: int = 768,
        selection_strategy: str = "importance",
        importance_threshold: float = 0.5,
        retrieval_method: str = "cosine"
    ):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.selection_strategy = selection_strategy
        self.importance_threshold = importance_threshold

        self.memories: List[MemoryEntry] = []
        self.memory_index = 0

        self.retriever = MemoryRetriever(
            hidden_size=hidden_size,
            memory_size=capacity,
            retrieval_method=retrieval_method
        )

        self.task_memory_counts: Dict[int, int] = {}

    def compute_importance(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_value: Optional[float] = None,
        gradient_norm: Optional[float] = None
    ) -> float:
        """Compute importance score for memory entry."""
        importance = 0.0

        # Loss-based importance
        if loss_value is not None:
            importance += min(loss_value, 10.0) / 10.0

        # Gradient-based importance
        if gradient_norm is not None:
            importance += min(gradient_norm, 5.0) / 5.0

        # Activation magnitude
        activation_mean = hidden_states.abs().mean().item()
        importance += min(activation_mean, 1.0)

        return importance / 3.0  # Normalize

    def add_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        task_id: int = 0,
        importance_score: Optional[float] = None,
        loss_value: Optional[float] = None,
        gradient_norm: Optional[float] = None
    ):
        """Add memory to bank."""
        if importance_score is None:
            importance_score = self.compute_importance(
                hidden_states, labels, loss_value, gradient_norm
            )

        if importance_score < self.importance_threshold:
            return

        memory = MemoryEntry(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            labels=labels.detach().cpu() if labels is not None else None,
            hidden_states=hidden_states.detach().cpu(),
            task_id=task_id,
            importance_score=importance_score,
            timestamp=self.memory_index,
            gradient_norm=gradient_norm,
            loss_value=loss_value
        )

        if len(self.memories) < self.capacity:
            self.memories.append(memory)
        else:
            # Replace least important memory
            min_idx = min(range(len(self.memories)),
                         key=lambda i: self.memories[i].importance_score)
            if importance_score > self.memories[min_idx].importance_score:
                self.memories[min_idx] = memory

        self.memory_index += 1
        self.task_memory_counts[task_id] = self.task_memory_counts.get(task_id, 0) + 1

    def retrieve(
        self,
        query_states: torch.Tensor,
        k: int = 10
    ) -> Tuple[List[MemoryEntry], torch.Tensor]:
        """Retrieve k most relevant memories."""
        return self.retriever(query_states, self.memories, k)


# ============================================================================
# GPU Memory Monitoring & OOM Prevention
# ============================================================================

def get_gpu_compute_utilization(device: int = 0) -> float:
    """Get actual GPU compute utilization."""
    if not NVML_AVAILABLE:
        return 0.75  # Default assumption

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(utilization.gpu) / 100.0
        return max(0.0, min(1.0, gpu_util))
    except Exception:
        return 0.75


class MemoryMonitor:
    """Monitor and manage GPU/CPU memory to prevent OOM."""

    def __init__(
        self,
        target_utilization: float = 0.85,
        warning_threshold: float = 0.90,
        critical_threshold: float = 0.95,
        emergency_threshold: float = 0.98,
        history_size: int = 100,
        memory_headroom_gb: float = 1.0,
        silent_mode: bool = False
    ):
        self.target_utilization = target_utilization
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.memory_headroom_gb = memory_headroom_gb
        self.silent_mode = silent_mode

        # Memory history
        self.gpu_memory_history = deque(maxlen=history_size)
        self.cpu_memory_history = deque(maxlen=history_size)
        self.batch_size_history = deque(maxlen=history_size)

        # Statistics
        self.oom_predictions = []
        self.false_alarms = 0
        self.successful_interventions = 0

        # Device info
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.total_gpu_memory = {}
        self.reserved_memory = {}

        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                self.total_gpu_memory[i] = total_memory
                self.reserved_memory[i] = min(memory_headroom_gb, total_memory * 0.1)

                log_fn = logger.debug if silent_mode else logger.info
                log_fn(f"GPU {i}: {total_memory:.1f}GB total, {self.reserved_memory[i]:.1f}GB reserved")

        self.total_cpu_memory = psutil.virtual_memory().total / (1024**3)
        log_fn = logger.debug if silent_mode else logger.info
        log_fn(f"CPU: {self.total_cpu_memory:.1f}GB total")

    def get_memory_stats(self, device: Optional[int] = None, skip_sync: bool = False) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}

        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()

            if not skip_sync:
                torch.cuda.synchronize(device)

            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            cached = torch.cuda.memory_reserved(device) / (1024**3)
            total = self.total_gpu_memory.get(device, 0.0)

            stats['gpu_allocated_gb'] = allocated
            stats['gpu_cached_gb'] = cached
            stats['gpu_total_gb'] = total
            stats['gpu_utilization'] = allocated / total if total > 0 else 0.0

        # CPU memory
        cpu_mem = psutil.virtual_memory()
        stats['cpu_used_gb'] = cpu_mem.used / (1024**3)
        stats['cpu_total_gb'] = cpu_mem.total / (1024**3)
        stats['cpu_utilization'] = cpu_mem.percent / 100.0

        return stats

    def check_memory_status(self, device: Optional[int] = None) -> Dict[str, Any]:
        """Check memory status and return recommendations."""
        stats = self.get_memory_stats(device)
        self.gpu_memory_history.append(stats.get('gpu_utilization', 0.0))

        status = {
            'level': 'normal',
            'gpu_utilization': stats.get('gpu_utilization', 0.0),
            'recommendation': None
        }

        gpu_util = stats.get('gpu_utilization', 0.0)

        if gpu_util >= self.emergency_threshold:
            status['level'] = 'emergency'
            status['recommendation'] = 'emergency_cleanup'
        elif gpu_util >= self.critical_threshold:
            status['level'] = 'critical'
            status['recommendation'] = 'reduce_batch_size'
        elif gpu_util >= self.warning_threshold:
            status['level'] = 'warning'
            status['recommendation'] = 'monitor_closely'

        return status

    def cleanup_memory(self, aggressive: bool = False):
        """Trigger memory cleanup."""
        torch.cuda.empty_cache()
        gc.collect()

        if aggressive:
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time.sleep(0.1)


# ============================================================================
# GPU Memory Cleanup Utilities
# ============================================================================

class GPUMemoryManager:
    """Comprehensive GPU memory management."""

    def __init__(self, auto_cleanup: bool = True, emergency_threshold: float = 0.95):
        self.auto_cleanup = auto_cleanup
        self.emergency_threshold = emergency_threshold
        self._cleanup_handlers_registered = False

        if auto_cleanup:
            self.register_cleanup_handlers()

    def cleanup_gpu_memory(self, aggressive: bool = False) -> Dict[str, float]:
        """Comprehensive GPU memory cleanup."""
        stats = {'before_allocated': 0.0, 'before_cached': 0.0,
                'after_allocated': 0.0, 'after_cached': 0.0}

        try:
            if torch.cuda.is_available():
                stats['before_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['before_cached'] = torch.cuda.memory_reserved() / 1024**3

                torch.cuda.empty_cache()
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    if aggressive:
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass

                        for _ in range(5):
                            gc.collect()
                            torch.cuda.empty_cache()
                            time.sleep(0.1)

                stats['after_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['after_cached'] = torch.cuda.memory_reserved() / 1024**3

        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
            stats['error'] = str(e)

        return stats

    def register_cleanup_handlers(self):
        """Register cleanup handlers for process termination."""
        if self._cleanup_handlers_registered:
            return

        def cleanup_handler(signum=None, frame=None):
            logger.info("Cleanup handler triggered")
            self.cleanup_gpu_memory(aggressive=True)

        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, cleanup_handler)
            except Exception:
                pass

        # Register exit handler
        atexit.register(cleanup_handler)
        self._cleanup_handlers_registered = True

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics."""
        stats = {}

        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)

                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                total = props.total_memory / 1024**3

                stats['allocated_gb'] = allocated
                stats['cached_gb'] = cached
                stats['total_gb'] = total
                stats['utilization'] = allocated / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")

        return stats


# ============================================================================
# Unified Memory Manager (Main Interface)
# ============================================================================

class UnifiedMemoryManager:
    """
    Main unified memory manager combining all memory management features.

    Provides:
    - Episodic memory for continual learning
    - GPU memory monitoring and OOM prevention
    - Memory cleanup utilities
    """

    def __init__(
        self,
        # Episodic memory config
        episodic_memory_capacity: int = 1000,
        hidden_size: int = 768,
        enable_episodic_memory: bool = False,

        # GPU monitoring config
        target_utilization: float = 0.85,
        warning_threshold: float = 0.90,
        critical_threshold: float = 0.95,
        emergency_threshold: float = 0.98,
        silent_mode: bool = False,

        # Cleanup config
        auto_cleanup: bool = True
    ):
        """Initialize unified memory manager."""

        # Episodic memory (optional)
        self.episodic_memory = None
        if enable_episodic_memory:
            self.episodic_memory = EpisodicMemoryBank(
                capacity=episodic_memory_capacity,
                hidden_size=hidden_size
            )

        # GPU memory monitoring
        self.monitor = MemoryMonitor(
            target_utilization=target_utilization,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            emergency_threshold=emergency_threshold,
            silent_mode=silent_mode
        )

        # GPU memory manager
        self.gpu_manager = GPUMemoryManager(
            auto_cleanup=auto_cleanup,
            emergency_threshold=emergency_threshold
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {}
        stats['monitor'] = self.monitor.get_memory_stats()
        stats['gpu'] = self.gpu_manager.get_memory_stats()

        if self.episodic_memory is not None:
            stats['episodic'] = {
                'num_memories': len(self.episodic_memory.memories),
                'capacity': self.episodic_memory.capacity,
                'utilization': len(self.episodic_memory.memories) / self.episodic_memory.capacity
            }

        return stats

    def check_and_cleanup(self) -> Dict[str, Any]:
        """Check memory status and cleanup if needed."""
        status = self.monitor.check_memory_status()

        if status['recommendation'] == 'emergency_cleanup':
            logger.warning("Emergency memory cleanup triggered!")
            self.gpu_manager.cleanup_gpu_memory(aggressive=True)
        elif status['recommendation'] == 'reduce_batch_size':
            logger.warning("Critical memory usage detected!")
            self.monitor.cleanup_memory()

        return status

    def cleanup(self, aggressive: bool = False):
        """Manual memory cleanup."""
        return self.gpu_manager.cleanup_gpu_memory(aggressive=aggressive)


# Convenience function
def create_memory_manager(
    enable_episodic: bool = False,
    silent_mode: bool = False,
    **kwargs
) -> UnifiedMemoryManager:
    """Convenience function to create unified memory manager."""
    return UnifiedMemoryManager(
        enable_episodic_memory=enable_episodic,
        silent_mode=silent_mode,
        **kwargs
    )
