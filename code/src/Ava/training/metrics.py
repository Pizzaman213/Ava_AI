"""
Training Metrics System

This module provides comprehensive training metrics collection, processing,
and analysis for enhanced monitoring and debugging.
"""

import time
import torch  # type: ignore[import]
import psutil
from typing import Dict, Any, List, Optional, Tuple, Union, cast
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    LOSS = "loss"
    LEARNING_RATE = "learning_rate"
    GRADIENT = "gradient"
    MEMORY = "memory"
    TIMING = "timing"
    SYSTEM = "system"
    TRAINING = "training"
    EVALUATION = "evaluation"


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    # Collection settings
    collect_gradients: bool = True
    collect_memory: bool = True
    collect_system: bool = False
    collect_timing: bool = True

    # History settings
    history_size: int = 1000
    detailed_history_size: int = 100

    # Frequency settings
    gradient_freq: int = 100          # Collect gradients every N steps
    memory_freq: int = 50             # Collect memory every N steps
    system_freq: int = 500            # Collect system metrics every N steps

    # Analysis settings
    enable_trend_analysis: bool = True
    trend_window_size: int = 50
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0    # Standard deviations for anomaly


@dataclass
class TrainingStep:
    """Container for training step information."""
    step: int
    epoch: int
    batch_idx: int
    timestamp: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    memory_allocated: Optional[float] = None
    memory_cached: Optional[float] = None
    batch_time: Optional[float] = None
    forward_time: Optional[float] = None
    backward_time: Optional[float] = None
    optimizer_time: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class TrainingMetricsCollector:
    """
    Comprehensive training metrics collector and analyzer.

    Features:
    - Real-time metrics collection
    - Performance analytics
    - Memory monitoring
    - Gradient analysis
    - Trend detection
    - Anomaly detection
    """

    def __init__(self, config: MetricConfig):
        """
        Initialize training metrics collector.

        Args:
            config: Metrics collection configuration
        """
        self.config = config

        # Metrics storage
        self.metrics_history = deque(maxlen=config.history_size)
        self.detailed_history = deque(maxlen=config.detailed_history_size)

        # Running statistics - using Dict[str, Any] for flexibility with different stat types
        self.running_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0, 'sum': 0.0, 'sum_sq': 0.0,
            'min': float('inf'), 'max': float('-inf'),
            'recent': deque(maxlen=config.trend_window_size)
        })

        # Timing trackers
        self.timing_contexts = {}
        self.step_start_time = None

        # Counters
        self.total_steps = 0
        self.collection_start_time = time.time()

        # Analysis results
        self.trends = {}
        self.anomalies = []

    def start_step(self, step: int, epoch: int, batch_idx: int) -> None:
        """Start tracking a training step."""
        self.step_start_time = time.time()
        self.current_step_info = {
            'step': step,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'start_time': self.step_start_time
        }

    def end_step(self, loss: float, learning_rate: float, **kwargs) -> TrainingStep:
        """
        End tracking a training step and create step record.

        Args:
            loss: Training loss for this step
            learning_rate: Current learning rate
            **kwargs: Additional metrics

        Returns:
            TrainingStep object with collected metrics
        """
        if self.step_start_time is None:
            self.step_start_time = time.time()

        end_time = time.time()
        step_duration = end_time - self.step_start_time

        # Create training step record
        step_info = TrainingStep(
            step=self.current_step_info.get('step', self.total_steps),
            epoch=self.current_step_info.get('epoch', 0),
            batch_idx=self.current_step_info.get('batch_idx', 0),
            timestamp=end_time,
            loss=loss,
            learning_rate=learning_rate,
            batch_time=step_duration,
            additional_metrics=kwargs
        )

        # Collect additional metrics
        self._collect_memory_metrics(step_info)
        if self.total_steps % self.config.system_freq == 0:
            self._collect_system_metrics(step_info)

        # Update statistics
        self._update_statistics(step_info)

        # Store in history
        self.metrics_history.append(step_info)
        if self.total_steps % 10 == 0:  # Store detailed every 10 steps
            self.detailed_history.append(step_info)

        # Analysis
        if self.config.enable_trend_analysis:
            self._analyze_trends(step_info)
        if self.config.enable_anomaly_detection:
            self._detect_anomalies(step_info)

        self.total_steps += 1
        self.step_start_time = None

        return step_info

    def collect_gradient_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Collect gradient-related metrics.

        Args:
            model: PyTorch model

        Returns:
            Dictionary of gradient metrics
        """
        if not self.config.collect_gradients or self.total_steps % self.config.gradient_freq != 0:
            return {}

        try:
            total_norm = 0.0
            param_count = 0
            grad_norms = []
            zero_grads = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    grad_norms.append(grad_norm)
                    param_count += 1
                else:
                    zero_grads += 1

            if param_count > 0:
                total_norm = total_norm ** 0.5
                grad_norms = np.array(grad_norms)

                metrics = {
                    'grad_norm_total': total_norm,
                    'grad_norm_mean': grad_norms.mean(),
                    'grad_norm_std': grad_norms.std(),
                    'grad_norm_max': grad_norms.max(),
                    'grad_norm_min': grad_norms.min(),
                    'params_with_grad': param_count,
                    'params_zero_grad': zero_grads,
                    'grad_norm_ratio': grad_norms.max() / max(grad_norms.mean(), 1e-8)
                }

                return metrics

        except Exception:
            pass

        return {}

    def _collect_memory_metrics(self, step_info: TrainingStep) -> None:
        """Collect memory usage metrics."""
        if not self.config.collect_memory or self.total_steps % self.config.memory_freq != 0:
            return

        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                step_info.memory_allocated = allocated
                step_info.memory_cached = cached

        except Exception:
            pass

    def _collect_system_metrics(self, step_info: TrainingStep) -> None:
        """Collect system resource metrics."""
        if not self.config.collect_system:
            return

        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            step_info.additional_metrics.update({
                'system_memory_percent': memory.percent,
                'system_cpu_percent': cpu_percent
            })

        except Exception:
            pass

    def _update_statistics(self, step_info: TrainingStep) -> None:
        """Update running statistics for all metrics."""
        # Core metrics
        metrics_to_track = {
            'loss': step_info.loss,
            'learning_rate': step_info.learning_rate
        }

        # Optional metrics
        if step_info.grad_norm is not None:
            metrics_to_track['grad_norm'] = step_info.grad_norm
        if step_info.memory_allocated is not None:
            metrics_to_track['memory_allocated'] = step_info.memory_allocated
        if step_info.batch_time is not None:
            metrics_to_track['batch_time'] = step_info.batch_time

        # Additional metrics
        metrics_to_track.update(step_info.additional_metrics)

        # Update running stats
        for name, value in metrics_to_track.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                stats = self.running_stats[name]
                stats['count'] = int(stats['count']) + 1  # type: ignore[assignment]
                stats['sum'] = float(stats['sum']) + value  # type: ignore[assignment]
                stats['sum_sq'] = float(stats['sum_sq']) + value ** 2  # type: ignore[assignment]
                stats['min'] = min(float(stats['min']), value)
                stats['max'] = max(float(stats['max']), value)
                cast(deque, stats['recent']).append(value)

    def _analyze_trends(self, step_info: TrainingStep) -> None:
        """Analyze trends in metrics."""
        for metric_name, stats in self.running_stats.items():
            recent = cast(deque, stats['recent'])
            if len(recent) >= self.config.trend_window_size:
                recent_values = list(recent)

                # Simple linear trend
                x = np.arange(len(recent_values))
                y = np.array(recent_values)

                try:
                    slope = np.polyfit(x, y, 1)[0]
                    self.trends[metric_name] = {
                        'slope': slope,
                        'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'magnitude': abs(slope),
                        'updated_at': step_info.step
                    }
                except:
                    pass

    def _detect_anomalies(self, step_info: TrainingStep) -> None:
        """Detect anomalies in metrics."""
        for metric_name, stats in self.running_stats.items():
            count = float(stats['count'])
            if count >= 10:  # Need sufficient history
                mean = float(stats['sum']) / count
                variance = (float(stats['sum_sq']) / count) - (mean ** 2)
                std = variance ** 0.5 if variance > 0 else 0

                # Get current value
                current_value = None
                if metric_name == 'loss':
                    current_value = step_info.loss
                elif metric_name == 'learning_rate':
                    current_value = step_info.learning_rate
                elif metric_name in step_info.additional_metrics:
                    current_value = step_info.additional_metrics[metric_name]

                # Check for anomaly
                if current_value is not None and std > 0:
                    z_score = abs(current_value - mean) / std
                    if z_score > self.config.anomaly_threshold:
                        anomaly = {
                            'step': step_info.step,
                            'metric': metric_name,
                            'value': current_value,
                            'mean': mean,
                            'std': std,
                            'z_score': z_score,
                            'timestamp': step_info.timestamp
                        }
                        self.anomalies.append(anomaly)

                        # Keep only recent anomalies
                        if len(self.anomalies) > 100:
                            self.anomalies = self.anomalies[-50:]

    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current statistics for all tracked metrics."""
        stats_summary = {}

        for metric_name, stats in self.running_stats.items():
            count = float(stats['count'])
            if count > 0:
                mean = float(stats['sum']) / count
                variance = (float(stats['sum_sq']) / count) - (mean ** 2)
                std = variance ** 0.5 if variance > 0 else 0

                recent = cast(deque, stats['recent'])
                stats_summary[metric_name] = {
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'latest': recent[-1] if recent else None
                }

        return stats_summary

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.metrics_history:
            return {}

        current_time = time.time()
        total_time = current_time - self.collection_start_time

        # Calculate rates
        steps_per_second = self.total_steps / total_time if total_time > 0 else 0

        # Get recent metrics
        recent_steps = list(self.metrics_history)[-min(100, len(self.metrics_history)):]
        recent_losses = [step.loss for step in recent_steps if step.loss is not None]

        summary = {
            'total_steps': self.total_steps,
            'total_time_minutes': total_time / 60,
            'steps_per_second': steps_per_second,
            'steps_per_minute': steps_per_second * 60,
            'current_loss': recent_losses[-1] if recent_losses else None,
            'loss_trend': self.trends.get('loss', {}).get('direction', 'unknown'),
            'recent_anomalies': len([a for a in self.anomalies if current_time - a['timestamp'] < 300]),  # Last 5 minutes
            'memory_usage_gb': self.metrics_history[-1].memory_allocated if self.metrics_history and self.metrics_history[-1].memory_allocated else None
        }

        # Add trend information
        if self.trends:
            summary['trends'] = {name: trend['direction'] for name, trend in self.trends.items()}

        return summary

    def get_recent_anomalies(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent anomalies."""
        return self.anomalies[-last_n:] if self.anomalies else []

    def get_metric_history(self, metric_name: str, last_n: Optional[int] = None) -> List[Tuple[int, float]]:
        """Get history for a specific metric."""
        history = []
        steps_to_check = self.metrics_history if last_n is None else list(self.metrics_history)[-last_n:]

        for step in steps_to_check:
            value = None
            if metric_name == 'loss':
                value = step.loss
            elif metric_name == 'learning_rate':
                value = step.learning_rate
            elif metric_name == 'grad_norm':
                value = step.grad_norm
            elif metric_name == 'memory_allocated':
                value = step.memory_allocated
            elif metric_name == 'batch_time':
                value = step.batch_time
            elif metric_name in step.additional_metrics:
                value = step.additional_metrics[metric_name]

            if value is not None:
                history.append((step.step, value))

        return history

    def create_timing_context(self, name: str):
        """Create a timing context manager."""
        return TimingContext(self, name)

    def record_timing(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        cast(deque, self.running_stats[f'timing_{name}']['recent']).append(duration)

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.running_stats.clear()
        self.trends.clear()
        self.anomalies.clear()
        self.total_steps = 0
        self.collection_start_time = time.time()

    def record_batch_size_change(self, step: int, new_batch_size: int, reason: str) -> None:
        """
        Record a dynamic batch size change event.

        Args:
            step: Training step number
            new_batch_size: New batch size after adjustment
            reason: Reason for the adjustment
        """
        # Initialize batch size tracking if not exists
        if 'batch_size_history' not in self.running_stats:
            self.running_stats['batch_size_history'] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'recent': deque(maxlen=self.config.trend_window_size),
                'changes': deque(maxlen=100),  # Track last 100 changes
                'increases': 0,
                'decreases': 0
            }

        stats = self.running_stats['batch_size_history']

        # Determine if increase or decrease
        recent = cast(deque, stats['recent'])
        if recent:
            prev_batch_size = recent[-1]
            if new_batch_size > prev_batch_size:
                stats['increases'] = int(stats['increases']) + 1  # type: ignore[assignment]
                direction = 'increase'
            elif new_batch_size < prev_batch_size:
                stats['decreases'] = int(stats['decreases']) + 1  # type: ignore[assignment]
                direction = 'decrease'
            else:
                direction = 'unchanged'
        else:
            direction = 'initial'

        # Update statistics
        stats['count'] = int(stats['count']) + 1  # type: ignore[assignment]
        stats['sum'] = float(stats['sum']) + new_batch_size  # type: ignore[assignment]
        stats['sum_sq'] = float(stats['sum_sq']) + new_batch_size ** 2  # type: ignore[assignment]
        stats['min'] = min(float(stats['min']), new_batch_size)
        stats['max'] = max(float(stats['max']), new_batch_size)
        recent.append(new_batch_size)

        # Record change event
        cast(deque, stats['changes']).append({
            'step': step,
            'batch_size': new_batch_size,
            'reason': reason,
            'direction': direction,
            'timestamp': time.time()
        })

    def get_batch_size_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive batch size statistics.

        Returns:
            Dictionary with batch size statistics
        """
        if 'batch_size_history' not in self.running_stats:
            return {
                'enabled': False,
                'current_batch_size': None,
                'avg_batch_size': None,
                'min_batch_size': None,
                'max_batch_size': None,
                'total_changes': 0,
                'increases': 0,
                'decreases': 0,
                'adjustment_rate': 0.0
            }

        stats = self.running_stats['batch_size_history']

        # Calculate average
        count = float(stats['count'])
        avg_batch_size = float(stats['sum']) / count if count > 0 else 0.0

        # Get current batch size
        recent = cast(deque, stats['recent'])
        current_batch_size = recent[-1] if recent else None

        # Calculate adjustment rate (changes per step)
        adjustment_rate = count / max(self.total_steps, 1)

        return {
            'enabled': True,
            'current_batch_size': current_batch_size,
            'avg_batch_size': avg_batch_size,
            'min_batch_size': stats['min'] if stats['min'] != float('inf') else None,
            'max_batch_size': stats['max'] if stats['max'] != float('-inf') else None,
            'total_changes': stats['count'],
            'increases': stats['increases'],
            'decreases': stats['decreases'],
            'adjustment_rate': adjustment_rate,
            'recent_changes': list(cast(deque, stats['changes']))[-5:] if stats['changes'] else []
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get collector state for checkpointing."""
        return {
            'total_steps': self.total_steps,
            'collection_start_time': self.collection_start_time,
            'running_stats': {
                name: {
                    'count': stats['count'],
                    'sum': stats['sum'],
                    'sum_sq': stats['sum_sq'],
                    'min': stats['min'],
                    'max': stats['max']
                }
                for name, stats in self.running_stats.items()
            },
            'trends': self.trends,
            'recent_anomalies': self.anomalies[-10:] if self.anomalies else []
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load collector state from checkpoint."""
        self.total_steps = state_dict.get('total_steps', 0)
        self.collection_start_time = state_dict.get('collection_start_time', time.time())
        self.trends = state_dict.get('trends', {})
        self.anomalies = state_dict.get('recent_anomalies', [])

        # Restore running stats
        saved_stats = state_dict.get('running_stats', {})
        for name, stats in saved_stats.items():
            self.running_stats[name].update(stats)
            # Initialize recent deque
            self.running_stats[name]['recent'] = deque(maxlen=self.config.trend_window_size)


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, collector: TrainingMetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timing(self.name, duration)


# Convenience functions for creating configurations
def create_comprehensive_metrics_config() -> MetricConfig:
    """Create configuration for comprehensive metrics collection."""
    return MetricConfig(
        collect_gradients=True,
        collect_memory=True,
        collect_system=True,
        collect_timing=True,
        gradient_freq=50,
        memory_freq=25,
        system_freq=100,
        enable_trend_analysis=True,
        enable_anomaly_detection=True
    )


def create_fast_metrics_config() -> MetricConfig:
    """Create configuration optimized for speed."""
    return MetricConfig(
        collect_gradients=False,
        collect_memory=True,
        collect_system=False,
        collect_timing=True,
        memory_freq=100,
        gradient_freq=1000,
        enable_trend_analysis=False,
        enable_anomaly_detection=False
    )


def create_minimal_metrics_config() -> MetricConfig:
    """Create minimal metrics configuration."""
    return MetricConfig(
        collect_gradients=False,
        collect_memory=False,
        collect_system=False,
        collect_timing=False,
        enable_trend_analysis=False,
        enable_anomaly_detection=False
    )


# MoE-Specific Metrics
@dataclass
class MoEMetricsConfig:
    """Configuration for MoE-specific metrics tracking."""
    track_expert_utilization: bool = True
    track_routing_entropy: bool = True
    track_load_balance: bool = True
    track_expert_specialization: bool = False
    log_frequency: int = 500


class MoEMetricsTracker:
    """
    Mixture-of-Experts specific metrics tracker.

    Monitors expert behavior to detect issues like:
    - Expert collapse (all tokens routed to few experts)
    - Poor load balancing
    - Low routing diversity
    """

    def __init__(self, config: MoEMetricsConfig, num_experts: int = 8):
        """
        Initialize MoE metrics tracker.

        Args:
            config: MoE metrics configuration
            num_experts: Number of experts in the model
        """
        self.config = config
        self.num_experts = num_experts

        # Expert utilization tracking
        self.expert_counts = defaultdict(int)
        self.expert_utilization_history = deque(maxlen=1000)

        # Routing entropy tracking
        self.routing_entropy_history = deque(maxlen=1000)

        # Load balance tracking
        self.load_balance_history = deque(maxlen=1000)

        # Specialization tracking (if enabled)
        self.expert_token_affinity = defaultdict(lambda: defaultdict(int))

    def update_from_router_outputs(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """
        Update metrics from router outputs.

        Args:
            router_logits: Router logits [batch_size, seq_len, num_experts]
            selected_experts: Selected expert indices [batch_size, seq_len, top_k]
            step: Current training step

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        if self.config.track_expert_utilization:
            utilization = self._compute_expert_utilization(selected_experts)
            metrics['moe/expert_utilization'] = utilization
            self.expert_utilization_history.append(utilization)

        if self.config.track_routing_entropy:
            entropy = self._compute_routing_entropy(router_logits)
            metrics['moe/routing_entropy'] = entropy
            self.routing_entropy_history.append(entropy)

        if self.config.track_load_balance:
            balance = self._compute_load_balance(selected_experts)
            metrics['moe/load_balance'] = balance
            self.load_balance_history.append(balance)

        return metrics

    def _compute_expert_utilization(self, selected_experts: torch.Tensor) -> float:
        """
        Compute percentage of experts being actively used.

        Returns:
            Utilization ratio (0.0 to 1.0, higher is better)
        """
        # Flatten to get all selected experts
        flat_experts = selected_experts.flatten()
        unique_experts = torch.unique(flat_experts)

        utilization = len(unique_experts) / self.num_experts
        return float(utilization)

    def _compute_routing_entropy(self, router_logits: torch.Tensor) -> float:
        """
        Compute entropy of routing distribution.

        Higher entropy = more diverse routing (better)
        Lower entropy = concentrated routing (risk of expert collapse)

        Returns:
            Average routing entropy across batch
        """
        # Apply softmax to get probabilities
        routing_probs = torch.softmax(router_logits, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-10), dim=-1)

        # Average across batch and sequence
        avg_entropy = entropy.mean().item()

        return avg_entropy

    def _compute_load_balance(self, selected_experts: torch.Tensor) -> float:
        """
        Compute load balance metric.

        Perfect balance = 1.0 (all experts used equally)
        Poor balance = lower values (some experts overused)

        Returns:
            Load balance score (0.0 to 1.0)
        """
        # Count how many times each expert was selected
        flat_experts = selected_experts.flatten()
        expert_counts = torch.bincount(flat_experts, minlength=self.num_experts)

        # Compute coefficient of variation (inverse measure of balance)
        mean_count = expert_counts.float().mean()
        std_count = expert_counts.float().std()

        if mean_count > 0:
            cv = std_count / mean_count
            # Convert to 0-1 score (lower CV = better balance)
            balance_score = 1.0 / (1.0 + cv)
        else:
            balance_score = 0.0

        return float(balance_score)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of MoE metrics."""
        summary = {}

        if self.expert_utilization_history:
            summary['expert_utilization_mean'] = np.mean(self.expert_utilization_history)
            summary['expert_utilization_std'] = np.std(self.expert_utilization_history)
            summary['expert_utilization_min'] = np.min(self.expert_utilization_history)

        if self.routing_entropy_history:
            summary['routing_entropy_mean'] = np.mean(self.routing_entropy_history)
            summary['routing_entropy_std'] = np.std(self.routing_entropy_history)

        if self.load_balance_history:
            summary['load_balance_mean'] = np.mean(self.load_balance_history)
            summary['load_balance_min'] = np.min(self.load_balance_history)

        return summary

    def detect_expert_collapse(self, threshold: float = 0.5) -> bool:
        """
        Detect if expert collapse is occurring.

        Args:
            threshold: Minimum utilization ratio to consider healthy

        Returns:
            True if expert collapse detected
        """
        if not self.expert_utilization_history:
            return False

        recent_utilization = list(self.expert_utilization_history)[-10:]
        avg_utilization = np.mean(recent_utilization)

        return bool(avg_utilization < threshold)