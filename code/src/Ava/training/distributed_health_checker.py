"""
Distributed Training Health Checker

Monitors the health of distributed training across all ranks with loss synchronization,
gradient analysis, and performance monitoring.
"""

import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import time
import threading
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetrics:
    """Health metrics for a rank."""
    rank: int
    timestamp: float
    loss: float
    gradient_norm: float
    learning_rate: float
    memory_usage: float
    compute_time: float
    status: HealthStatus
    anomaly_score: float = 0.0


@dataclass
class SynchronizedHealthData:
    """Synchronized health data across all ranks."""
    timestamp: float
    rank_metrics: List[HealthMetrics]
    global_loss_mean: float
    global_loss_std: float
    global_gradient_norm_mean: float
    global_gradient_norm_std: float
    healthy_ranks: List[int]
    warning_ranks: List[int]
    degraded_ranks: List[int]
    critical_ranks: List[int]
    failed_ranks: List[int]
    overall_status: HealthStatus


class DistributedHealthChecker:
    """
    Monitors distributed training health with loss synchronization and anomaly detection.

    Features:
    - Loss synchronization across ranks
    - Gradient norm monitoring
    - Performance tracking
    - Anomaly detection
    - Health status reporting
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        check_interval: float = 30.0,
        loss_history_size: int = 100,
        anomaly_threshold: float = 2.5,
        enable_gradient_sync: bool = True,
        enable_performance_sync: bool = True
    ):
        """
        Initialize distributed health checker.

        Args:
            world_size: Total number of ranks
            rank: Current rank
            check_interval: Health check interval in seconds
            loss_history_size: Number of historical values to keep
            anomaly_threshold: Threshold for anomaly detection (std deviations)
            enable_gradient_sync: Enable gradient norm synchronization
            enable_performance_sync: Enable performance metrics synchronization
        """
        self.world_size = world_size
        self.rank = rank
        self.check_interval = check_interval
        self.loss_history_size = loss_history_size
        self.anomaly_threshold = anomaly_threshold
        self.enable_gradient_sync = enable_gradient_sync
        self.enable_performance_sync = enable_performance_sync

        # Health tracking
        self.local_metrics_history: deque = deque(maxlen=loss_history_size)
        self.global_health_history: deque = deque(maxlen=loss_history_size)
        self.current_metrics: Optional[HealthMetrics] = None

        # Anomaly detection state
        self.loss_baseline_mean: Optional[float] = None
        self.loss_baseline_std: Optional[float] = None
        self.gradient_baseline_mean: Optional[float] = None
        self.gradient_baseline_std: Optional[float] = None

        # Synchronization
        self._health_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Monitoring
        self.health_check_thread: Optional[threading.Thread] = None
        self.last_health_check = time.time()

        # Statistics
        self.total_checks = 0
        self.anomaly_count = 0
        self.synchronization_failures = 0

        # Start monitoring if distributed
        if self.world_size > 1:
            self._start_health_monitoring()

    def _start_health_monitoring(self):
        """Start background health monitoring."""
        self.health_check_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name=f"DistributedHealthCheck-{self.rank}"
        )
        self.health_check_thread.start()
        logger.info(f"Rank {self.rank}: Distributed health monitoring started")

    def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while not self._shutdown_event.wait(self.check_interval):
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.synchronization_failures += 1

    def _perform_health_check(self):
        """Perform distributed health check."""
        if not dist.is_initialized():
            return

        try:
            # Synchronize current metrics across all ranks
            synchronized_data = self._synchronize_health_metrics()

            if synchronized_data:
                # Analyze global health
                self._analyze_global_health(synchronized_data)

                # Update history
                with self._health_lock:
                    self.global_health_history.append(synchronized_data)

                self.last_health_check = time.time()
                self.total_checks += 1

        except Exception as e:
            logger.error(f"Health check synchronization failed: {e}")
            self.synchronization_failures += 1

    def record_training_metrics(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        memory_usage: float,
        compute_time: float
    ):
        """
        Record training metrics for health monitoring.

        Args:
            loss: Current training loss
            gradient_norm: Gradient norm
            learning_rate: Current learning rate
            memory_usage: GPU memory usage (0-1)
            compute_time: Time for forward+backward pass
        """
        # Create health metrics
        metrics = HealthMetrics(
            rank=self.rank,
            timestamp=time.time(),
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            memory_usage=memory_usage,
            compute_time=compute_time,
            status=HealthStatus.HEALTHY  # Will be updated by analysis
        )

        # Analyze local health
        metrics.status, metrics.anomaly_score = self._analyze_local_health(metrics)

        # Store metrics
        with self._health_lock:
            self.current_metrics = metrics
            self.local_metrics_history.append(metrics)

    def _analyze_local_health(self, metrics: HealthMetrics) -> Tuple[HealthStatus, float]:
        """
        Analyze local health based on current metrics.

        Returns:
            Tuple of (health_status, anomaly_score)
        """
        anomaly_score = 0.0
        status = HealthStatus.HEALTHY

        # Check for obvious problems
        if not torch.isfinite(torch.tensor(metrics.loss)):
            return HealthStatus.FAILED, 10.0

        if metrics.gradient_norm > 1000.0:  # Very large gradients
            return HealthStatus.CRITICAL, 5.0

        if metrics.memory_usage > 0.95:  # Very high memory usage
            status = HealthStatus.WARNING
            anomaly_score += 1.0

        # Check against baselines if available
        if len(self.local_metrics_history) > 10:
            recent_losses = [m.loss for m in list(self.local_metrics_history)[-10:]]
            recent_grads = [m.gradient_norm for m in list(self.local_metrics_history)[-10:]]

            # Loss anomaly detection
            if len(recent_losses) > 1:
                loss_mean = np.mean(recent_losses)
                loss_std = np.std(recent_losses)

                if loss_std > 0:
                    loss_z_score = abs(metrics.loss - loss_mean) / loss_std
                    if loss_z_score > self.anomaly_threshold:
                        anomaly_score += loss_z_score
                        if loss_z_score > 5.0:
                            status = HealthStatus.DEGRADED
                        elif status == HealthStatus.HEALTHY:
                            status = HealthStatus.WARNING

            # Gradient anomaly detection
            if len(recent_grads) > 1:
                grad_mean = np.mean(recent_grads)
                grad_std = np.std(recent_grads)

                if grad_std > 0:
                    grad_z_score = abs(metrics.gradient_norm - grad_mean) / grad_std
                    if grad_z_score > self.anomaly_threshold:
                        anomaly_score += grad_z_score
                        if grad_z_score > 5.0:
                            status = HealthStatus.DEGRADED
                        elif status == HealthStatus.HEALTHY:
                            status = HealthStatus.WARNING

        return status, float(anomaly_score)

    def _synchronize_health_metrics(self) -> Optional[SynchronizedHealthData]:
        """Synchronize health metrics across all ranks."""
        if not self.current_metrics:
            return None

        try:
            # Serialize current metrics
            metrics_data = asdict(self.current_metrics)
            metrics_json = json.dumps(metrics_data).encode('utf-8')

            # Create tensors for synchronization
            max_size = 1024  # Maximum size for JSON data
            data_tensor = torch.zeros(max_size, dtype=torch.uint8, device='cuda' if torch.cuda.is_available() else 'cpu')
            size_tensor = torch.tensor([len(metrics_json)], dtype=torch.int64, device=data_tensor.device)

            # Pack data into tensor
            data_tensor[:len(metrics_json)] = torch.frombuffer(metrics_json, dtype=torch.uint8)

            # All-gather sizes and data
            all_sizes = [torch.zeros_like(size_tensor) for _ in range(self.world_size)]
            all_data = [torch.zeros_like(data_tensor) for _ in range(self.world_size)]

            dist.all_gather(all_sizes, size_tensor)
            dist.all_gather(all_data, data_tensor)

            # Reconstruct metrics from all ranks
            all_metrics = []
            for rank in range(self.world_size):
                try:
                    size = all_sizes[rank].item()
                    data_bytes = all_data[rank][:size].cpu().numpy().tobytes()
                    rank_data = json.loads(data_bytes.decode('utf-8'))

                    # Convert back to HealthMetrics
                    rank_data['status'] = HealthStatus(rank_data['status'])
                    metrics = HealthMetrics(**rank_data)
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Failed to decode metrics from rank {rank}: {e}")

            if not all_metrics:
                return None

            # Calculate global statistics
            losses = [m.loss for m in all_metrics if torch.isfinite(torch.tensor(m.loss))]
            gradient_norms = [m.gradient_norm for m in all_metrics if torch.isfinite(torch.tensor(m.gradient_norm))]

            global_loss_mean = np.mean(losses) if losses else 0.0
            global_loss_std = np.std(losses) if len(losses) > 1 else 0.0
            global_gradient_norm_mean = np.mean(gradient_norms) if gradient_norms else 0.0
            global_gradient_norm_std = np.std(gradient_norms) if len(gradient_norms) > 1 else 0.0

            # Categorize ranks by health status
            healthy_ranks = [m.rank for m in all_metrics if m.status == HealthStatus.HEALTHY]
            warning_ranks = [m.rank for m in all_metrics if m.status == HealthStatus.WARNING]
            degraded_ranks = [m.rank for m in all_metrics if m.status == HealthStatus.DEGRADED]
            critical_ranks = [m.rank for m in all_metrics if m.status == HealthStatus.CRITICAL]
            failed_ranks = [m.rank for m in all_metrics if m.status == HealthStatus.FAILED]

            # Determine overall health status
            if failed_ranks:
                overall_status = HealthStatus.FAILED
            elif critical_ranks:
                overall_status = HealthStatus.CRITICAL
            elif len(degraded_ranks) > self.world_size // 2:
                overall_status = HealthStatus.DEGRADED
            elif degraded_ranks or warning_ranks:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY

            return SynchronizedHealthData(
                timestamp=time.time(),
                rank_metrics=all_metrics,
                global_loss_mean=float(global_loss_mean),
                global_loss_std=float(global_loss_std),
                global_gradient_norm_mean=float(global_gradient_norm_mean),
                global_gradient_norm_std=float(global_gradient_norm_std),
                healthy_ranks=healthy_ranks,
                warning_ranks=warning_ranks,
                degraded_ranks=degraded_ranks,
                critical_ranks=critical_ranks,
                failed_ranks=failed_ranks,
                overall_status=overall_status
            )

        except Exception as e:
            logger.error(f"Health metrics synchronization failed: {e}")
            return None

    def _analyze_global_health(self, synchronized_data: SynchronizedHealthData):
        """Analyze global health and log findings."""
        # Log health status if not healthy
        if synchronized_data.overall_status != HealthStatus.HEALTHY:
            logger.warning(f"Distributed training health: {synchronized_data.overall_status.value}")

            if synchronized_data.failed_ranks:
                logger.error(f"Failed ranks: {synchronized_data.failed_ranks}")
            if synchronized_data.critical_ranks:
                logger.error(f"Critical ranks: {synchronized_data.critical_ranks}")
            if synchronized_data.degraded_ranks:
                logger.warning(f"Degraded ranks: {synchronized_data.degraded_ranks}")
            if synchronized_data.warning_ranks:
                logger.info(f"Warning ranks: {synchronized_data.warning_ranks}")

        # Detect global anomalies
        if synchronized_data.global_loss_std > 0:
            # Check for high loss variance across ranks
            loss_cv = synchronized_data.global_loss_std / max(synchronized_data.global_loss_mean, 1e-8)
            if loss_cv > 0.5:  # Coefficient of variation > 50%
                logger.warning(f"High loss variance across ranks: CV={loss_cv:.3f}")
                self.anomaly_count += 1

        # Update baselines
        self._update_health_baselines(synchronized_data)

        # Log periodic health summary
        if self.total_checks % 10 == 0 and self.rank == 0:  # Master rank logs summary
            self._log_health_summary(synchronized_data)

    def _update_health_baselines(self, synchronized_data: SynchronizedHealthData):
        """Update health baselines for anomaly detection."""
        # Use exponential moving average for baselines
        alpha = 0.1

        if self.loss_baseline_mean is None:
            self.loss_baseline_mean = synchronized_data.global_loss_mean
            self.loss_baseline_std = synchronized_data.global_loss_std
        else:
            self.loss_baseline_mean = (1 - alpha) * self.loss_baseline_mean + alpha * synchronized_data.global_loss_mean
            if self.loss_baseline_std is not None:
                self.loss_baseline_std = (1 - alpha) * self.loss_baseline_std + alpha * synchronized_data.global_loss_std

        if self.gradient_baseline_mean is None:
            self.gradient_baseline_mean = synchronized_data.global_gradient_norm_mean
            self.gradient_baseline_std = synchronized_data.global_gradient_norm_std
        else:
            self.gradient_baseline_mean = (1 - alpha) * self.gradient_baseline_mean + alpha * synchronized_data.global_gradient_norm_mean
            if self.gradient_baseline_std is not None:
                self.gradient_baseline_std = (1 - alpha) * self.gradient_baseline_std + alpha * synchronized_data.global_gradient_norm_std

    def _log_health_summary(self, synchronized_data: SynchronizedHealthData):
        """Log health summary (master rank only)."""
        logger.info("=== Distributed Training Health Summary ===")
        logger.info(f"Overall Status: {synchronized_data.overall_status.value}")
        logger.info(f"Healthy Ranks: {len(synchronized_data.healthy_ranks)}/{self.world_size}")
        logger.info(f"Global Loss: {synchronized_data.global_loss_mean:.6f} ± {synchronized_data.global_loss_std:.6f}")
        logger.info(f"Global Grad Norm: {synchronized_data.global_gradient_norm_mean:.3f} ± {synchronized_data.global_gradient_norm_std:.3f}")
        logger.info(f"Total Health Checks: {self.total_checks}")
        logger.info(f"Anomalies Detected: {self.anomaly_count}")
        logger.info(f"Sync Failures: {self.synchronization_failures}")
        logger.info("===========================================")

    def get_current_health_status(self) -> HealthStatus:
        """Get current health status for this rank."""
        with self._health_lock:
            if self.current_metrics:
                return self.current_metrics.status
            return HealthStatus.HEALTHY

    def get_global_health_status(self) -> Optional[HealthStatus]:
        """Get latest global health status."""
        with self._health_lock:
            if self.global_health_history:
                return self.global_health_history[-1].overall_status
            return None

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._health_lock:
            current_status = self.get_current_health_status()
            global_status = self.get_global_health_status()

            # Recent metrics statistics
            if self.local_metrics_history:
                recent_losses = [m.loss for m in list(self.local_metrics_history)[-10:]]
                recent_grads = [m.gradient_norm for m in list(self.local_metrics_history)[-10:]]
                recent_anomaly_scores = [m.anomaly_score for m in list(self.local_metrics_history)[-10:]]
            else:
                recent_losses = recent_grads = recent_anomaly_scores = []

            return {
                'rank': self.rank,
                'world_size': self.world_size,
                'current_status': current_status.value,
                'global_status': global_status.value if global_status else None,
                'total_checks': self.total_checks,
                'anomaly_count': self.anomaly_count,
                'synchronization_failures': self.synchronization_failures,
                'last_health_check': self.last_health_check,
                'monitoring_active': not self._shutdown_event.is_set(),
                'recent_loss_mean': np.mean(recent_losses) if recent_losses else None,
                'recent_loss_std': np.std(recent_losses) if len(recent_losses) > 1 else None,
                'recent_grad_norm_mean': np.mean(recent_grads) if recent_grads else None,
                'recent_grad_norm_std': np.std(recent_grads) if len(recent_grads) > 1 else None,
                'recent_anomaly_score_mean': np.mean(recent_anomaly_scores) if recent_anomaly_scores else None,
                'health_history_size': len(self.local_metrics_history),
                'global_health_history_size': len(self.global_health_history)
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status including loss health information (alias for compatibility)."""
        summary = self.get_health_summary()

        # Add loss health info for test compatibility
        return {
            'status': summary.get('current_status', 'healthy'),
            'loss_health': {
                'is_healthy': summary.get('current_status') not in ['critical', 'failed'],
                'anomaly_count': summary.get('anomaly_count', 0),
                'recent_loss_mean': summary.get('recent_loss_mean'),
                'recent_loss_std': summary.get('recent_loss_std')
            },
            'global_status': summary.get('global_status'),
            'rank': summary.get('rank'),
            'total_checks': summary.get('total_checks', 0)
        }

    def should_abort_training(self) -> bool:
        """Check if training should be aborted due to health issues."""
        global_status = self.get_global_health_status()

        # Abort if global status is failed
        if global_status == HealthStatus.FAILED:
            return True

        # Abort if too many synchronization failures
        if self.synchronization_failures > 10:
            return True

        # Abort if local status is failed
        if self.get_current_health_status() == HealthStatus.FAILED:
            return True

        return False

    def cleanup(self):
        """Clean up health checker resources."""
        logger.info(f"Rank {self.rank}: Cleaning up health checker")

        # Stop monitoring
        self._shutdown_event.set()

        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5.0)

        logger.info(f"Rank {self.rank}: Health checker cleanup completed")


# Global health checker instance
_global_health_checker: Optional[DistributedHealthChecker] = None


def get_health_checker(
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    **kwargs
) -> DistributedHealthChecker:
    """Get or create global distributed health checker."""
    global _global_health_checker

    if _global_health_checker is None:
        # Auto-detect world_size and rank if not provided
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        _global_health_checker = DistributedHealthChecker(world_size, rank, **kwargs)

    return _global_health_checker


def record_training_metrics(
    loss: float,
    gradient_norm: float,
    learning_rate: float,
    memory_usage: float = 0.0,
    compute_time: float = 0.0
):
    """Convenience function to record training metrics."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        health_checker = get_health_checker()
        health_checker.record_training_metrics(
            loss, gradient_norm, learning_rate, memory_usage, compute_time
        )