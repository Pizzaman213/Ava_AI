"""
Proactive Memory Management and OOM Prevention

Implements intelligent memory monitoring and management to prevent
out-of-memory errors during training:
- Real-time GPU memory monitoring
- Proactive batch size reduction
- Emergency memory cleanup
- Memory headroom reservation
- Memory usage forecasting
"""

import torch  # type: ignore[import]
import psutil
import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import logging
import numpy as np

# Try to import NVML for actual GPU utilization monitoring
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except Exception as e:
        # NVML library exists but initialization failed (no GPU, permissions, etc.)
        NVML_AVAILABLE = False
        pynvml = None
        logger = logging.getLogger(__name__)
        logger.debug(f"NVML initialization failed: {e}")
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


def get_gpu_compute_utilization(device: int = 0) -> float:
    """
    Get actual GPU compute utilization (not memory utilization).

    Args:
        device: GPU device index

    Returns:
        GPU compute utilization as a float between 0.0 and 1.0
    """
    if not NVML_AVAILABLE:
        # Fallback: estimate based on training speed
        return 0.75  # Assume reasonable utilization if NVML not available

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)  # type: ignore[union-attr]
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[union-attr]
        gpu_util = float(utilization.gpu) / 100.0  # Convert percentage to fraction

        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, gpu_util))
    except Exception as e:
        logger.debug(f"Failed to get GPU utilization via NVML: {e}")
        return 0.75  # Fallback assumption


class MemoryMonitor:
    """
    Monitor and manage GPU/CPU memory usage to prevent OOM errors.

    Tracks memory usage patterns and takes proactive action when
    memory usage approaches dangerous levels.
    """

    def __init__(
        self,
        target_utilization: float = 0.85,
        warning_threshold: float = 0.99,  # Set to 99% to allow maximum GPU utilization
        critical_threshold: float = 0.99,  # Set to 99% to allow maximum GPU utilization
        emergency_threshold: float = 0.99,  # Set to 99% to trigger cleanup only at near-full capacity
        history_size: int = 100,
        memory_headroom_gb: float = 1.0,
        silent_mode: bool = False  # NEW: Suppress memory warnings
    ):
        """
        Initialize memory monitor.

        Args:
            target_utilization: Target GPU memory utilization (0.85 = 85%)
            warning_threshold: Threshold to trigger warnings (default 0.99 = 99%)
            critical_threshold: Threshold to trigger batch size reduction (default 0.99 = 99%)
            emergency_threshold: Threshold to trigger emergency cleanup (default 0.99 = 99%)
            history_size: Number of memory measurements to keep
            memory_headroom_gb: GB of memory to reserve for safety
            silent_mode: If True, suppress memory warning messages (only log at DEBUG level)
        """
        self.target_utilization = target_utilization
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.memory_headroom_gb = memory_headroom_gb
        self.silent_mode = silent_mode

        # Memory history tracking
        self.gpu_memory_history = deque(maxlen=history_size)
        self.cpu_memory_history = deque(maxlen=history_size)
        self.batch_size_history = deque(maxlen=history_size)

        # OOM prediction
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
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                self.total_gpu_memory[i] = total_memory
                self.reserved_memory[i] = min(
                    self.memory_headroom_gb,
                    total_memory * 0.1  # Reserve at most 10% of total memory
                )
                if not self.silent_mode:
                    logger.info(f"GPU {i}: {total_memory:.1f}GB total, {self.reserved_memory[i]:.1f}GB reserved")
                else:
                    logger.debug(f"GPU {i}: {total_memory:.1f}GB total, {self.reserved_memory[i]:.1f}GB reserved")

        # CPU memory info
        self.total_cpu_memory = psutil.virtual_memory().total / (1024**3)
        if not self.silent_mode:
            logger.info(f"CPU: {self.total_cpu_memory:.1f}GB total")
        else:
            logger.debug(f"CPU: {self.total_cpu_memory:.1f}GB total")

    def get_memory_stats(self, device: Optional[int] = None, skip_sync: bool = False) -> Dict[str, float]:
        """
        Get current memory statistics.

        Args:
            device: GPU device index (None for current device)
            skip_sync: Skip synchronization for performance (use cached values)

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        # GPU memory stats
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()

            try:
                current_device = device  # Type assertion - device is not None here
                # SPEED OPTIMIZATION: Skip synchronization unless explicitly needed
                if not skip_sync:
                    torch.cuda.synchronize(current_device)
                allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
                cached = torch.cuda.memory_reserved(current_device) / (1024**3)
                total = self.total_gpu_memory.get(current_device, 0)

                # Get actual GPU compute utilization (not memory utilization)
                gpu_compute_util = get_gpu_compute_utilization(current_device)

                # Calculate memory utilization (what we should monitor for memory warnings)
                memory_utilization = cached / total if total > 0 else 0  # Use cached (reserved) memory

                stats.update({
                    'gpu_allocated_gb': allocated,
                    'gpu_cached_gb': cached,
                    'gpu_total_gb': total,
                    'gpu_utilization': memory_utilization,  # Use MEMORY utilization for memory warnings
                    'gpu_compute_utilization': gpu_compute_util,  # Separate compute util
                    'gpu_memory_utilization': allocated / total if total > 0 else 0,  # Allocated memory util
                    'gpu_cached_utilization': cached / total if total > 0 else 0,
                    'gpu_available_gb': total - cached,
                    'gpu_device': device
                })
            except Exception as e:
                logger.debug(f"Failed to get GPU memory stats (using fallbacks): {e}")
                # Use reasonable fallback that won't trigger false warnings
                # Fallback utilization assumes GPU is actively training
                fallback_utilization = 0.75  # Assume reasonable utilization
                stats.update({
                    'gpu_allocated_gb': 0,
                    'gpu_cached_gb': 0,
                    'gpu_total_gb': 0,
                    'gpu_utilization': fallback_utilization,
                    'gpu_memory_utilization': 0,
                    'gpu_cached_utilization': 0,
                    'gpu_available_gb': 0,
                    'gpu_device': -1
                })
        else:
            # CUDA not available - provide fallback stats
            stats.update({
                'gpu_allocated_gb': 0,
                'gpu_cached_gb': 0,
                'gpu_total_gb': 0,
                'gpu_utilization': 0,
                'gpu_memory_utilization': 0,
                'gpu_cached_utilization': 0,
                'gpu_available_gb': 0,
                'gpu_device': -1
            })

        # CPU memory stats
        try:
            cpu_mem = psutil.virtual_memory()
            stats.update({
                'cpu_used_gb': cpu_mem.used / (1024**3),
                'cpu_available_gb': cpu_mem.available / (1024**3),
                'cpu_total_gb': self.total_cpu_memory,
                'cpu_utilization': cpu_mem.percent / 100.0
            })
        except Exception as e:
            logger.warning(f"Failed to get CPU memory stats: {e}")
            stats.update({
                'cpu_used_gb': 0,
                'cpu_available_gb': 0,
                'cpu_total_gb': 0,
                'cpu_utilization': 0
            })

        return stats

    def update_memory_history(self, batch_size: int, device: Optional[int] = None):
        """
        Update memory usage history.

        Args:
            batch_size: Current batch size
            device: GPU device index
        """
        stats = self.get_memory_stats(device)

        self.gpu_memory_history.append({
            'timestamp': time.time(),
            'utilization': stats['gpu_utilization'],
            'allocated_gb': stats['gpu_allocated_gb'],
            'cached_gb': stats['gpu_cached_gb'],
            'available_gb': stats['gpu_available_gb'],
            'batch_size': batch_size
        })

        self.cpu_memory_history.append({
            'timestamp': time.time(),
            'utilization': stats['cpu_utilization'],
            'used_gb': stats['cpu_used_gb'],
            'available_gb': stats['cpu_available_gb'],
            'batch_size': batch_size
        })

        self.batch_size_history.append(batch_size)

    def check_memory_health(self, batch_size: int, device: Optional[int] = None) -> Dict[str, Any]:
        """
        Check memory health and recommend actions.

        Args:
            batch_size: Current batch size
            device: GPU device index

        Returns:
            Dictionary with memory health assessment and recommendations
        """
        # Update history
        self.update_memory_history(batch_size, device)

        # Get current stats
        current_stats = self.get_memory_stats(device)
        gpu_util = current_stats['gpu_utilization']
        cpu_util = current_stats['cpu_utilization']

        # Determine memory status
        status = 'healthy'
        action_needed = False
        recommended_batch_size = batch_size

        if gpu_util >= self.emergency_threshold:
            status = 'emergency'
            action_needed = True
            recommended_batch_size = max(1, batch_size // 4)  # Aggressive reduction
        elif gpu_util >= self.critical_threshold:
            status = 'critical'
            action_needed = True
            recommended_batch_size = max(1, batch_size // 2)  # Significant reduction
        elif gpu_util >= self.warning_threshold:
            status = 'warning'
            action_needed = True
            recommended_batch_size = max(1, int(batch_size * 0.8))  # Modest reduction
        elif gpu_util < self.target_utilization * 0.8:
            # Memory usage is low, could potentially increase batch size
            status = 'underutilized'
            if len(self.gpu_memory_history) >= 10:
                # Check if memory has been stable at low utilization
                recent_utils = [h['utilization'] for h in list(self.gpu_memory_history)[-10:]]
                if all(u < self.target_utilization * 0.8 for u in recent_utils):
                    recommended_batch_size = min(batch_size * 2, 64)  # Conservative increase

        # Predict OOM risk
        oom_risk = self._predict_oom_risk(current_stats, batch_size)

        # Check for memory leaks
        leak_detected = self._detect_memory_leak()

        return {
            'status': status,
            'action_needed': action_needed,
            'recommended_batch_size': recommended_batch_size,
            'current_batch_size': batch_size,
            'gpu_utilization': gpu_util,
            'cpu_utilization': cpu_util,
            'oom_risk': oom_risk,
            'leak_detected': leak_detected,
            'available_gb': current_stats['gpu_available_gb'],
            'allocated_gb': current_stats['gpu_allocated_gb'],
            'cached_gb': current_stats['gpu_cached_gb'],
            **current_stats
        }

    def _predict_oom_risk(self, current_stats: Dict[str, float], batch_size: int) -> float:
        """
        Predict OOM risk based on current memory usage and trends.

        Args:
            current_stats: Current memory statistics
            batch_size: Current batch size

        Returns:
            OOM risk score (0.0 to 1.0)
        """
        if len(self.gpu_memory_history) < 5:
            return 0.0

        # Get recent memory usage trend
        recent_history = list(self.gpu_memory_history)[-5:]
        utilizations = [h['utilization'] for h in recent_history]

        # Calculate trend
        if len(utilizations) >= 2:
            trend = np.mean(np.diff(utilizations))  # Average change per step
        else:
            trend = 0.0

        # Base risk from current utilization
        current_util = current_stats['gpu_utilization']
        base_risk = max(0.0, (current_util - self.target_utilization) / (1.0 - self.target_utilization))

        # Trend risk - if memory is increasing rapidly
        trend_risk = max(0.0, trend * 10)  # Scale trend impact

        # Batch size risk - larger batches are riskier
        batch_risk = min(1.0, batch_size / 64.0) * 0.2  # Max 0.2 risk from batch size

        # Combine risks
        total_risk = min(1.0, base_risk + trend_risk + batch_risk)

        return float(total_risk)

    def _detect_memory_leak(self) -> bool:
        """
        Detect potential memory leaks.

        Returns:
            True if a memory leak is detected
        """
        if len(self.gpu_memory_history) < 20:
            return False

        # Check if memory usage is consistently increasing
        recent_history = list(self.gpu_memory_history)[-20:]
        utilizations = [h['utilization'] for h in recent_history]
        batch_sizes = [h['batch_size'] for h in recent_history]

        # If batch sizes are stable but memory keeps increasing, possible leak
        batch_size_stable = len(set(batch_sizes)) <= 2  # At most 2 different batch sizes

        if batch_size_stable:
            # Check for increasing trend
            if len(utilizations) >= 10:
                first_half = np.mean(utilizations[:10])
                second_half = np.mean(utilizations[10:])
                increase = second_half - first_half

                # If memory increased by more than 5% with stable batch size
                if increase > 0.05:
                    return True

        return False

    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, float]:
        """
        Perform memory cleanup.

        Args:
            aggressive: Whether to perform aggressive cleanup

        Returns:
            Memory stats before and after cleanup
        """
        # SPEED OPTIMIZATION: Use skip_sync for before stats (we don't need exact sync here)
        before_stats = self.get_memory_stats(skip_sync=True)

        # SPEED OPTIMIZATION: Only do cleanup if there's significant cached memory (> 1GB, not 0.5GB)
        # Increased threshold to reduce unnecessary cleanup operations
        if torch.cuda.is_available():
            cached_gb = before_stats.get('gpu_cached_gb', 0)
            if cached_gb > 1.0:  # Only cleanup if > 1GB cached (was 0.5GB)
                torch.cuda.empty_cache()

        # CPU cleanup
        gc.collect()

        if aggressive:
            # More aggressive cleanup (but remove expensive synchronization)
            try:
                # Clear PyTorch internal caches
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()

                # Force garbage collection 2x (reduced from 3x)
                for _ in range(2):
                    gc.collect()

                # SPEED OPTIMIZATION: Skip synchronization even in aggressive mode unless emergency
                # Synchronize only current device (not all devices - major bottleneck removed)
                if torch.cuda.is_available():
                    # Only sync if we're in true emergency (>99% memory usage)
                    if before_stats.get('gpu_cached_gb', 0) / self.total_gpu_memory.get(torch.cuda.current_device(), 12.0) > 0.99:
                        torch.cuda.synchronize()  # Current device only
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Aggressive cleanup failed: {e}")

        # SPEED OPTIMIZATION: Use skip_sync for after stats too
        after_stats = self.get_memory_stats(skip_sync=True)

        freed_gb = before_stats['gpu_cached_gb'] - after_stats['gpu_cached_gb']
        if not self.silent_mode:
            logger.info(f"Memory cleanup: freed {freed_gb:.2f}GB GPU memory")
        else:
            logger.debug(f"Memory cleanup: freed {freed_gb:.2f}GB GPU memory")

        return {
            'before_allocated': before_stats['gpu_allocated_gb'],
            'after_allocated': after_stats['gpu_allocated_gb'],
            'before_cached': before_stats['gpu_cached_gb'],
            'after_cached': after_stats['gpu_cached_gb'],
            'freed_gb': freed_gb
        }

    def get_optimal_batch_size(
        self,
        current_batch_size: int,
        memory_per_sample_mb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal batch size based on current memory usage.

        Args:
            current_batch_size: Current batch size
            memory_per_sample_mb: Memory usage per sample in MB (estimated if None)

        Returns:
            Recommended optimal batch size
        """
        if not torch.cuda.is_available():
            return current_batch_size

        current_stats = self.get_memory_stats()
        available_gb = current_stats['gpu_available_gb']
        target_available_gb = self.total_gpu_memory.get(
            current_stats['gpu_device'], 0
        ) * (1.0 - self.target_utilization)

        # If we don't have memory per sample, estimate it
        if memory_per_sample_mb is None and len(self.gpu_memory_history) >= 2:
            # Estimate based on recent memory usage and batch sizes
            recent = list(self.gpu_memory_history)[-2:]
            if len(set(h['batch_size'] for h in recent)) == 2:  # Different batch sizes
                mem_diff = recent[1]['allocated_gb'] - recent[0]['allocated_gb']
                batch_diff = recent[1]['batch_size'] - recent[0]['batch_size']
                if batch_diff != 0:
                    memory_per_sample_mb = abs(mem_diff * 1024 / batch_diff)

        if memory_per_sample_mb is None:
            # Fallback: use conservative estimate
            memory_per_sample_mb = 50.0  # 50MB per sample

        # Calculate optimal batch size
        memory_budget_gb = target_available_gb - self.reserved_memory.get(
            current_stats['gpu_device'], 0
        )
        memory_budget_mb = memory_budget_gb * 1024

        optimal_batch_size = max(1, int(memory_budget_mb / memory_per_sample_mb))

        # Don't change batch size too dramatically
        max_change_factor = 2.0
        min_batch_size = max(1, int(current_batch_size / max_change_factor))
        max_batch_size = int(current_batch_size * max_change_factor)

        optimal_batch_size = max(min_batch_size, min(optimal_batch_size, max_batch_size))

        return optimal_batch_size

    def should_emergency_stop(self) -> bool:
        """
        Check if training should be emergency stopped due to memory issues.

        Returns:
            True if emergency stop is recommended
        """
        if not torch.cuda.is_available():
            return False

        current_stats = self.get_memory_stats()

        # Emergency stop if we're consistently at critical memory levels
        if len(self.gpu_memory_history) >= 5:
            recent_utils = [h['utilization'] for h in list(self.gpu_memory_history)[-5:]]
            if all(u >= self.emergency_threshold for u in recent_utils):
                return True

        # Emergency stop if memory leak detected and at high utilization
        if self._detect_memory_leak() and current_stats['gpu_utilization'] > self.critical_threshold:
            return True

        return False

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        current_stats = self.get_memory_stats()

        summary = {
            'current_status': current_stats,
            'thresholds': {
                'target': self.target_utilization,
                'warning': self.warning_threshold,
                'critical': self.critical_threshold,
                'emergency': self.emergency_threshold
            },
            'interventions': {
                'successful': self.successful_interventions,
                'false_alarms': self.false_alarms
            }
        }

        if len(self.gpu_memory_history) > 0:
            recent_utils = [h['utilization'] for h in list(self.gpu_memory_history)[-10:]]
            summary['recent_utilization'] = {
                'mean': float(np.mean(recent_utils)),
                'max': float(np.max(recent_utils)),
                'min': float(np.min(recent_utils)),
                'std': float(np.std(recent_utils))
            }

        return summary

    def get_detailed_memory_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed memory breakdown by component.

        Returns comprehensive memory usage breakdown including:
        - Model parameters
        - Gradients
        - Optimizer states
        - Activations (estimated)
        - Reserved vs allocated memory

        Returns:
            Dictionary with detailed memory breakdown
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        breakdown = {}

        try:
            # Get PyTorch memory stats
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB

            # Get memory stats from PyTorch
            memory_stats = torch.cuda.memory_stats(device)

            # Calculate component sizes
            active_bytes = memory_stats.get('active_bytes.all.current', 0)
            inactive_bytes = memory_stats.get('inactive_split_bytes.all.current', 0)

            breakdown['total'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'free_reserved_gb': reserved - allocated,
            }

            breakdown['memory_pools'] = {
                'active_gb': active_bytes / (1024**3),
                'inactive_gb': inactive_bytes / (1024**3),
            }

            # Estimate fragmentation
            fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0
            breakdown['fragmentation'] = {
                'ratio': fragmentation,
                'wasted_gb': reserved - allocated,
            }

            # Add allocation/free counts
            breakdown['allocations'] = {
                'num_alloc': memory_stats.get('num_alloc_retries', 0),
                'num_ooms': memory_stats.get('num_ooms', 0),
            }

        except Exception as e:
            breakdown['error'] = str(e)

        return breakdown

    def estimate_activation_memory(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        use_gradient_checkpointing: bool = False,
        use_flash_attention: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate activation memory usage for transformer models.

        Args:
            batch_size: Training batch size
            sequence_length: Sequence length
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            use_gradient_checkpointing: Whether gradient checkpointing is enabled
            use_flash_attention: Whether flash attention is enabled

        Returns:
            Dictionary with memory estimates in GB
        """
        estimates = {}

        # Attention memory: O(batch * heads * seq^2) for standard attention
        if use_flash_attention:
            # Flash attention: O(batch * seq) memory
            attention_mem = batch_size * sequence_length * hidden_size * 2  # bytes (fp16)
            estimates['attention_type'] = 'flash'
        else:
            # Standard attention: O(batch * heads * seq^2) memory
            attention_mem = (
                batch_size
                * num_attention_heads
                * sequence_length
                * sequence_length
                * 2  # bytes (fp16)
            )
            estimates['attention_type'] = 'standard'

        attention_mem_gb = (attention_mem * num_layers) / (1024**3)

        # Activation memory for feed-forward layers
        # Each layer stores: 2 activations of size (batch * seq * hidden * 4) for FFN
        ffn_mem = batch_size * sequence_length * hidden_size * 4 * 2 * 2  # bytes (fp16)
        ffn_mem_gb = (ffn_mem * num_layers) / (1024**3)

        # Total activation memory without checkpointing
        total_activation_mem_gb = attention_mem_gb + ffn_mem_gb

        if use_gradient_checkpointing:
            # Gradient checkpointing saves ~80% of activation memory
            # Only stores activations at checkpoints, recomputes during backward
            checkpoint_savings = 0.8
            effective_mem_gb = total_activation_mem_gb * (1 - checkpoint_savings)
            estimates['checkpointing_enabled'] = True
            estimates['savings_gb'] = total_activation_mem_gb * checkpoint_savings
        else:
            effective_mem_gb = total_activation_mem_gb
            estimates['checkpointing_enabled'] = False
            estimates['savings_gb'] = 0.0

        estimates['attention_memory_gb'] = attention_mem_gb
        estimates['ffn_memory_gb'] = ffn_mem_gb
        estimates['total_without_checkpointing_gb'] = total_activation_mem_gb
        estimates['effective_memory_gb'] = effective_mem_gb

        # Calculate potential savings
        if not use_flash_attention:
            # Estimate flash attention savings
            flash_attention_mem = (
                batch_size * sequence_length * hidden_size * 2 * num_layers
            ) / (1024**3)
            estimates['potential_flash_attention_savings_gb'] = (
                attention_mem_gb - flash_attention_mem
            )

        if not use_gradient_checkpointing:
            estimates['potential_checkpointing_savings_gb'] = (
                total_activation_mem_gb * 0.8
            )

        return estimates

    def track_memory_optimizations(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track which memory optimizations are enabled and their impact.

        Args:
            config: Training configuration dictionary

        Returns:
            Dictionary summarizing enabled optimizations
        """
        optimizations = {}

        # Check gradient checkpointing
        grad_checkpoint = config.get('gradient_checkpointing', False) or config.get(
            'output', {}
        ).get('gradient_checkpointing', False)
        optimizations['gradient_checkpointing'] = {
            'enabled': grad_checkpoint,
            'memory_savings': '60-80%' if grad_checkpoint else 'N/A',
        }

        # Check flash attention
        flash_attn = config.get('model', {}).get('use_flash_attention', False)
        optimizations['flash_attention'] = {
            'enabled': flash_attn,
            'memory_savings': '50-70%' if flash_attn else 'N/A',
        }

        # Check mixed precision
        mixed_precision = config.get('training', {}).get('mixed_precision', 'fp32')
        optimizations['mixed_precision'] = {
            'enabled': mixed_precision in ['fp16', 'bf16'],
            'type': mixed_precision,
            'memory_savings': '50%' if mixed_precision != 'fp32' else 'N/A',
        }

        # Check DeepSpeed ZeRO
        zero_stage = config.get('training', {}).get('deepspeed', {}).get(
            'zero_stage', 0
        )
        optimizations['deepspeed_zero'] = {
            'enabled': zero_stage > 0,
            'stage': zero_stage,
            'memory_savings': (
                f'Stage {zero_stage}: '
                + (
                    '4x' if zero_stage == 3 else ('2x' if zero_stage == 2 else '1.5x')
                )
                if zero_stage > 0
                else 'N/A'
            ),
        }

        # Check activation checkpointing
        activation_checkpoint = (
            config.get('training', {})
            .get('deepspeed', {})
            .get('activation_checkpointing', False)
        )
        optimizations['activation_checkpointing'] = {
            'enabled': activation_checkpoint,
            'memory_savings': '30-50%' if activation_checkpoint else 'N/A',
        }

        # Calculate combined estimated savings
        total_savings = 0
        if grad_checkpoint:
            total_savings += 70  # ~70% average
        if flash_attn:
            total_savings += 60  # ~60% average for attention
        if mixed_precision in ['fp16', 'bf16']:
            total_savings += 50  # 50% for parameters/gradients

        optimizations['estimated_combined_savings'] = f'{min(total_savings, 90)}%'

        return optimizations


# Alias for backward compatibility
GPUMemoryManager = MemoryMonitor