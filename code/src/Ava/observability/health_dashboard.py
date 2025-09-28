"""
Training Health Dashboard (Phase 7.1)

Real-time training health monitoring with visualizations and alerts.
"""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
import math


@dataclass
class HealthMetrics:
    """Container for training health metrics."""
    # Training metrics
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None

    # Performance metrics
    samples_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None

    # System metrics
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None
    system_memory_used: Optional[float] = None

    # Training state
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    # Health indicators
    is_training_stable: bool = True
    is_performance_good: bool = True
    is_resource_healthy: bool = True

    # Alerts
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class HealthThresholds:
    """Thresholds for health monitoring."""
    # Loss thresholds
    loss_divergence_threshold: float = 10.0
    loss_stagnation_steps: int = 100
    loss_spike_multiplier: float = 3.0

    # Gradient thresholds
    gradient_explosion_threshold: float = 100.0
    gradient_vanishing_threshold: float = 1e-8

    # Performance thresholds
    min_samples_per_second: float = 1.0
    performance_drop_threshold: float = 0.5  # 50% drop

    # Resource thresholds
    gpu_memory_warning: float = 0.85  # 85%
    gpu_memory_critical: float = 0.95  # 95%
    cpu_utilization_warning: float = 0.9  # 90%

    # Stability thresholds
    loss_variance_threshold: float = 2.0
    performance_variance_threshold: float = 0.3


class HealthAnalyzer:
    """Analyzes health metrics and generates alerts."""

    def __init__(self, thresholds: HealthThresholds = None):
        self.thresholds = thresholds or HealthThresholds()
        self.history = deque(maxlen=1000)
        self.baseline_metrics = {}
        self.alert_cooldowns = defaultdict(float)

    def analyze_metrics(self, metrics: HealthMetrics) -> HealthMetrics:
        """Analyze metrics and update health status."""
        # Store history
        self.history.append(metrics)

        # Clear existing alerts/warnings
        metrics.alerts = []
        metrics.warnings = []

        # Analyze training stability
        metrics.is_training_stable = self._check_training_stability(metrics)

        # Analyze performance
        metrics.is_performance_good = self._check_performance_health(metrics)

        # Analyze resource usage
        metrics.is_resource_healthy = self._check_resource_health(metrics)

        return metrics

    def _check_training_stability(self, metrics: HealthMetrics) -> bool:
        """Check if training is stable."""
        stable = True

        if metrics.loss is not None:
            # Check for loss divergence
            if metrics.loss > self.thresholds.loss_divergence_threshold:
                metrics.alerts.append(f"Loss diverging: {metrics.loss:.4f}")
                stable = False

            # Check for loss spikes
            if len(self.history) > 10:
                recent_losses = [h.loss for h in list(self.history)[-10:] if h.loss is not None]
                if recent_losses:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    if metrics.loss > avg_loss * self.thresholds.loss_spike_multiplier:
                        metrics.warnings.append(f"Loss spike detected: {metrics.loss:.4f} vs avg {avg_loss:.4f}")

            # Check for loss stagnation
            if len(self.history) >= self.thresholds.loss_stagnation_steps:
                recent_losses = [h.loss for h in list(self.history)[-self.thresholds.loss_stagnation_steps:] if h.loss is not None]
                if recent_losses and len(set(f"{loss:.4f}" for loss in recent_losses)) == 1:
                    metrics.warnings.append("Loss appears to be stagnating")

        if metrics.gradient_norm is not None:
            # Check for gradient explosion
            if metrics.gradient_norm > self.thresholds.gradient_explosion_threshold:
                metrics.alerts.append(f"Gradient explosion: {metrics.gradient_norm:.6f}")
                stable = False

            # Check for gradient vanishing
            if metrics.gradient_norm < self.thresholds.gradient_vanishing_threshold:
                metrics.warnings.append(f"Very small gradients: {metrics.gradient_norm:.6f}")

        return stable

    def _check_performance_health(self, metrics: HealthMetrics) -> bool:
        """Check if performance is healthy."""
        healthy = True

        if metrics.samples_per_second is not None:
            # Check minimum performance
            if metrics.samples_per_second < self.thresholds.min_samples_per_second:
                metrics.warnings.append(f"Low throughput: {metrics.samples_per_second:.2f} samples/sec")
                healthy = False

            # Check for performance drops
            if len(self.history) > 20:
                recent_perf = [h.samples_per_second for h in list(self.history)[-20:] if h.samples_per_second is not None]
                if recent_perf:
                    avg_perf = sum(recent_perf) / len(recent_perf)
                    if metrics.samples_per_second < avg_perf * self.thresholds.performance_drop_threshold:
                        metrics.warnings.append(f"Performance drop: {metrics.samples_per_second:.2f} vs avg {avg_perf:.2f}")

        return healthy

    def _check_resource_health(self, metrics: HealthMetrics) -> bool:
        """Check if resource usage is healthy."""
        healthy = True

        # GPU memory checks
        if metrics.gpu_memory_used is not None and metrics.gpu_memory_total is not None:
            memory_ratio = metrics.gpu_memory_used / metrics.gpu_memory_total

            if memory_ratio > self.thresholds.gpu_memory_critical:
                metrics.alerts.append(f"Critical GPU memory: {memory_ratio:.1%}")
                healthy = False
            elif memory_ratio > self.thresholds.gpu_memory_warning:
                metrics.warnings.append(f"High GPU memory: {memory_ratio:.1%}")

        # GPU utilization (now measuring actual compute utilization)
        if metrics.gpu_utilization is not None:
            # Only warn if GPU utilization is critically low (< 10%)
            # Some periods of lower utilization are normal (data loading, checkpointing, etc.)
            if metrics.gpu_utilization < 0.1:  # Less than 10% - indicates real underutilization
                metrics.warnings.append(f"Low GPU utilization: {metrics.gpu_utilization:.1%}")
            elif metrics.gpu_utilization > 0.98:  # Over 98% - may indicate bottleneck
                metrics.warnings.append(f"Very high GPU utilization: {metrics.gpu_utilization:.1%}")

        # CPU utilization
        if metrics.cpu_utilization is not None:
            if metrics.cpu_utilization > self.thresholds.cpu_utilization_warning:
                metrics.warnings.append(f"High CPU utilization: {metrics.cpu_utilization:.1%}")

        return healthy

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.history:
            return {"status": "no_data", "message": "No metrics available"}

        latest = self.history[-1]

        # Overall health score (0-100)
        health_score = 100
        if not latest.is_training_stable:
            health_score -= 40
        if not latest.is_performance_good:
            health_score -= 30
        if not latest.is_resource_healthy:
            health_score -= 30

        health_score -= len(latest.alerts) * 10
        health_score -= len(latest.warnings) * 5
        health_score = max(0, health_score)

        # Determine status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "warning"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "training_stable": latest.is_training_stable,
            "performance_good": latest.is_performance_good,
            "resources_healthy": latest.is_resource_healthy,
            "alerts": latest.alerts,
            "warnings": latest.warnings,
            "last_update": latest.timestamp
        }


class DashboardPlotter:
    """Creates real-time plots and visualizations."""

    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.plot_data = defaultdict(lambda: deque(maxlen=max_points))

    def add_metric(self, name: str, value: float, timestamp: float = None):
        """Add a metric point for plotting."""
        if timestamp is None:
            timestamp = time.time()

        self.plot_data[name].append((timestamp, value))

    def get_plot_data(self, metric_name: str, last_n_points: int = None) -> List[Tuple[float, float]]:
        """Get plot data for a metric."""
        data = list(self.plot_data[metric_name])
        if last_n_points:
            data = data[-last_n_points:]
        return data

    def generate_ascii_plot(self, metric_name: str, width: int = 80, height: int = 20) -> str:
        """Generate ASCII plot for console display."""
        data = self.get_plot_data(metric_name, last_n_points=width)

        if not data:
            return f"No data for {metric_name}"

        values = [point[1] for point in data]
        min_val, max_val = min(values), max(values)

        if min_val == max_val:
            # All values are the same
            lines = [" " * width for _ in range(height)]
            lines[height // 2] = "─" * width
            return f"{metric_name}:\n" + "\n".join(lines)

        # Normalize values to fit in height
        normalized = []
        for val in values:
            norm_val = (val - min_val) / (max_val - min_val)
            y_pos = int(norm_val * (height - 1))
            normalized.append(height - 1 - y_pos)  # Flip Y axis

        # Create plot
        lines = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, y in enumerate(normalized):
            if i < width:
                lines[y][i] = "●"

        # Convert to strings
        plot_lines = ["".join(line) for line in lines]

        # Add labels
        result = f"{metric_name} (min: {min_val:.4f}, max: {max_val:.4f}):\n"
        for i, line in enumerate(plot_lines):
            # Add Y-axis labels
            if i == 0:
                result += f"{max_val:8.3f}│{line}\n"
            elif i == height - 1:
                result += f"{min_val:8.3f}│{line}\n"
            else:
                result += f"        │{line}\n"

        # Add X-axis
        result += "        └" + "─" * width + "\n"

        return result

    def get_dashboard_html(self, metrics_list: List[str]) -> str:
        """Generate HTML dashboard (placeholder for web interface)."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Health Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-container { margin: 20px 0; }
                .plot { border: 1px solid #ccc; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>Training Health Dashboard</h1>
        """

        for metric in metrics_list:
            data = self.get_plot_data(metric)
            if data:
                html += f"""
                <div class="metric-container">
                    <h3>{metric}</h3>
                    <div class="plot">
                        <!-- Placeholder for interactive plot -->
                        <p>Latest: {data[-1][1]:.4f}</p>
                        <p>Points: {len(data)}</p>
                    </div>
                </div>
                """

        html += """
        </body>
        </html>
        """
        return html


class HealthDashboard:
    """
    Main training health dashboard that combines analysis and visualization.

    Features:
    - Real-time health monitoring
    - Interactive visualizations
    - Alert system
    - Performance tracking
    - Resource monitoring
    """

    def __init__(
        self,
        thresholds: HealthThresholds = None,
        update_interval: float = 5.0,
        enable_plots: bool = True,
        max_history: int = 10000
    ):
        self.analyzer = HealthAnalyzer(thresholds)
        self.plotter = DashboardPlotter(max_history) if enable_plots else None
        self.update_interval = update_interval

        # Dashboard state
        self.current_metrics: Optional[HealthMetrics] = None
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Callbacks
        self.alert_callbacks: List[Callable[[str], None]] = []
        self.update_callbacks: List[Callable[[HealthMetrics], None]] = []

        # Statistics
        self.stats = {
            'updates': 0,
            'alerts_generated': 0,
            'warnings_generated': 0,
            'start_time': time.time()
        }

    def start(self):
        """Start the dashboard monitoring."""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        print("Training health dashboard started")

    def stop(self):
        """Stop the dashboard monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.update_thread:
            self.update_thread.join(timeout=5.0)

        print("Training health dashboard stopped")

    def update_metrics(self, **kwargs):
        """Update health metrics."""
        # Create metrics from kwargs
        metrics = HealthMetrics()
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        # Analyze metrics
        self.current_metrics = self.analyzer.analyze_metrics(metrics)

        # Update plots
        if self.plotter:
            self._update_plots(self.current_metrics)

        # Handle alerts
        self._handle_alerts(self.current_metrics)

        # Update statistics
        self.stats['updates'] += 1

        # Notify callbacks
        for callback in self.update_callbacks:
            try:
                callback(self.current_metrics)
            except Exception as e:
                print(f"Dashboard callback error: {e}")

    def _update_loop(self):
        """Background update loop."""
        while not self.stop_event.wait(self.update_interval):
            # This would typically fetch latest metrics from the training system
            # For now, it's a placeholder for periodic health checks
            if self.current_metrics:
                health_summary = self.analyzer.get_health_summary()
                # Could trigger periodic health reports here

    def _update_plots(self, metrics: HealthMetrics):
        """Update plot data with new metrics."""
        timestamp = metrics.timestamp

        # Add all numeric metrics to plots
        for field_name in ['loss', 'learning_rate', 'gradient_norm', 'samples_per_second',
                          'tokens_per_second', 'gpu_memory_used', 'gpu_utilization',
                          'cpu_utilization', 'system_memory_used']:
            value = getattr(metrics, field_name)
            if value is not None:
                self.plotter.add_metric(field_name, value, timestamp)

    def _handle_alerts(self, metrics: HealthMetrics):
        """Handle alerts and warnings."""
        # Process alerts
        for alert in metrics.alerts:
            self.stats['alerts_generated'] += 1
            for callback in self.alert_callbacks:
                try:
                    callback(f"ALERT: {alert}")
                except Exception as e:
                    print(f"Alert callback error: {e}")

        # Process warnings
        for warning in metrics.warnings:
            self.stats['warnings_generated'] += 1
            for callback in self.alert_callbacks:
                try:
                    callback(f"WARNING: {warning}")
                except Exception as e:
                    print(f"Warning callback error: {e}")

    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)

    def add_update_callback(self, callback: Callable[[HealthMetrics], None]):
        """Add callback for metric updates."""
        self.update_callbacks.append(callback)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        return self.analyzer.get_health_summary()

    def get_console_dashboard(self) -> str:
        """Generate console-based dashboard."""
        if not self.current_metrics:
            return "No metrics available"

        lines = []
        lines.append("=" * 80)
        lines.append("TRAINING HEALTH DASHBOARD")
        lines.append("=" * 80)

        # Health summary
        summary = self.get_health_summary()
        lines.append(f"Overall Status: {summary['status'].upper()} (Score: {summary['health_score']}/100)")
        lines.append(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(summary['last_update']))}")
        lines.append("")

        # Current metrics
        m = self.current_metrics
        lines.append("CURRENT METRICS:")
        if m.step is not None:
            lines.append(f"  Step: {m.step}")
        if m.epoch is not None:
            lines.append(f"  Epoch: {m.epoch}")
        if m.loss is not None:
            lines.append(f"  Loss: {m.loss:.6f}")
        if m.learning_rate is not None:
            lines.append(f"  Learning Rate: {m.learning_rate:.6f}")
        if m.gradient_norm is not None:
            lines.append(f"  Gradient Norm: {m.gradient_norm:.6f}")
        if m.samples_per_second is not None:
            lines.append(f"  Throughput: {m.samples_per_second:.2f} samples/sec")
        if m.gpu_memory_used is not None and m.gpu_memory_total is not None:
            ratio = m.gpu_memory_used / m.gpu_memory_total
            lines.append(f"  GPU Memory: {m.gpu_memory_used:.1f}GB / {m.gpu_memory_total:.1f}GB ({ratio:.1%})")
        if m.gpu_utilization is not None:
            lines.append(f"  GPU Utilization: {m.gpu_utilization:.1%}")

        lines.append("")

        # Alerts and warnings
        if m.alerts:
            lines.append("🚨 ALERTS:")
            for alert in m.alerts:
                lines.append(f"  - {alert}")
            lines.append("")

        if m.warnings:
            lines.append("⚠️ WARNINGS:")
            for warning in m.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        # Plots
        if self.plotter:
            for metric in ['loss', 'samples_per_second', 'gpu_utilization']:
                if getattr(m, metric) is not None:
                    plot = self.plotter.generate_ascii_plot(metric, width=60, height=10)
                    lines.append(plot)
                    lines.append("")

        # Statistics
        uptime = time.time() - self.stats['start_time']
        lines.append("DASHBOARD STATISTICS:")
        lines.append(f"  Uptime: {uptime:.0f} seconds")
        lines.append(f"  Updates: {self.stats['updates']}")
        lines.append(f"  Alerts: {self.stats['alerts_generated']}")
        lines.append(f"  Warnings: {self.stats['warnings_generated']}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_dashboard_html(self, filepath: str):
        """Save HTML dashboard to file."""
        if not self.plotter:
            return

        metrics = ['loss', 'learning_rate', 'gradient_norm', 'samples_per_second',
                  'gpu_memory_used', 'gpu_utilization', 'cpu_utilization']

        html = self.plotter.get_dashboard_html(metrics)

        with open(filepath, 'w') as f:
            f.write(html)

    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics history."""
        data = {
            'metrics_history': [
                {
                    'timestamp': entry.timestamp,
                    'step': entry.step,
                    'epoch': entry.epoch,
                    'loss': entry.loss,
                    'learning_rate': entry.learning_rate,
                    'gradient_norm': entry.gradient_norm,
                    'samples_per_second': entry.samples_per_second,
                    'tokens_per_second': entry.tokens_per_second,
                    'gpu_memory_used': entry.gpu_memory_used,
                    'gpu_utilization': entry.gpu_utilization,
                    'cpu_utilization': entry.cpu_utilization,
                    'alerts': entry.alerts,
                    'warnings': entry.warnings
                }
                for entry in self.analyzer.history
            ],
            'statistics': self.stats
        }

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions
def create_health_dashboard(
    alert_callback: Callable[[str], None] = None,
    **kwargs
) -> HealthDashboard:
    """Create and configure a health dashboard."""
    dashboard = HealthDashboard(**kwargs)

    if alert_callback:
        dashboard.add_alert_callback(alert_callback)
    else:
        # Default alert handler
        dashboard.add_alert_callback(lambda msg: print(f"🔔 {msg}"))

    return dashboard


def create_console_dashboard() -> HealthDashboard:
    """Create dashboard optimized for console output."""
    return HealthDashboard(
        update_interval=2.0,
        enable_plots=True,
        max_history=1000
    )