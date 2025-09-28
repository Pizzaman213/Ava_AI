"""
Health Monitoring API (Phase 7.2)

External health monitoring API for training systems.
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime, timedelta


class MonitoringEndpoint(Enum):
    """Available monitoring endpoints."""
    HEALTH_STATUS = "/api/v1/health"
    METRICS = "/api/v1/metrics"
    ALERTS = "/api/v1/alerts"
    LOGS = "/api/v1/logs"
    DIAGNOSTICS = "/api/v1/diagnostics"
    REPORTS = "/api/v1/reports"
    SYSTEM_INFO = "/api/v1/system"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthStatus:
    """Overall system health status."""
    status: str  # "healthy", "warning", "error", "critical"
    score: float  # 0.0 to 1.0
    timestamp: float
    uptime_seconds: float
    components: Dict[str, str]  # component -> status
    active_alerts: int
    last_update: float


@dataclass
class MetricSnapshot:
    """Snapshot of training metrics."""
    timestamp: float
    step: Optional[int]
    epoch: Optional[int]
    loss: Optional[float]
    learning_rate: Optional[float]
    throughput: Optional[float]
    gpu_utilization: Optional[float]
    memory_usage: Optional[float]
    custom_metrics: Dict[str, Any]


@dataclass
class Alert:
    """System alert."""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolve_timestamp: Optional[float] = None


@dataclass
class DiagnosticInfo:
    """System diagnostic information."""
    timestamp: float
    training_stable: bool
    performance_good: bool
    resources_healthy: bool
    error_count_1h: int
    warning_count_1h: int
    avg_step_time: float
    memory_efficiency: float
    recommendations: List[str]


class MonitoringDataProvider(ABC):
    """Abstract base class for monitoring data providers."""

    @abstractmethod
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        pass

    @abstractmethod
    def get_current_metrics(self) -> MetricSnapshot:
        """Get current metrics snapshot."""
        pass

    @abstractmethod
    def get_recent_metrics(self, count: int = 100) -> List[MetricSnapshot]:
        """Get recent metrics history."""
        pass

    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        pass

    @abstractmethod
    def get_diagnostic_info(self) -> DiagnosticInfo:
        """Get diagnostic information."""
        pass


class TrainingMonitoringProvider(MonitoringDataProvider):
    """Monitoring data provider for training systems."""

    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.alerts = {}
        self.components_status = {
            "training": "unknown",
            "data_pipeline": "unknown",
            "model": "unknown",
            "optimizer": "unknown",
            "system": "unknown"
        }
        self.last_update = time.time()

        # Simulated state (in real implementation, this would connect to actual systems)
        self.current_metrics = MetricSnapshot(
            timestamp=time.time(),
            step=None,
            epoch=None,
            loss=None,
            learning_rate=None,
            throughput=None,
            gpu_utilization=None,
            memory_usage=None,
            custom_metrics={}
        )

    def update_metrics(self, **kwargs):
        """Update current metrics."""
        self.current_metrics = MetricSnapshot(
            timestamp=time.time(),
            step=kwargs.get('step'),
            epoch=kwargs.get('epoch'),
            loss=kwargs.get('loss'),
            learning_rate=kwargs.get('learning_rate'),
            throughput=kwargs.get('throughput'),
            gpu_utilization=kwargs.get('gpu_utilization'),
            memory_usage=kwargs.get('memory_usage'),
            custom_metrics=kwargs.get('custom_metrics', {})
        )

        # Store in history
        self.metrics_history.append(self.current_metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]

        self.last_update = time.time()

    def update_component_status(self, component: str, status: str):
        """Update component status."""
        if component in self.components_status:
            self.components_status[component] = status

    def add_alert(self, level: AlertLevel, title: str, message: str, component: str):
        """Add new alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            level=level,
            title=title,
            message=message,
            component=component,
            timestamp=time.time()
        )
        self.alerts[alert.id] = alert

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolve_timestamp = time.time()

    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        # Calculate overall status
        component_scores = {
            "healthy": 1.0,
            "warning": 0.7,
            "error": 0.3,
            "critical": 0.0,
            "unknown": 0.5
        }

        scores = [component_scores.get(status, 0.5) for status in self.components_status.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.5

        # Determine status
        if overall_score >= 0.9:
            status = "healthy"
        elif overall_score >= 0.7:
            status = "warning"
        elif overall_score >= 0.3:
            status = "error"
        else:
            status = "critical"

        active_alerts = len([a for a in self.alerts.values() if not a.resolved])

        return HealthStatus(
            status=status,
            score=overall_score,
            timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
            components=self.components_status.copy(),
            active_alerts=active_alerts,
            last_update=self.last_update
        )

    def get_current_metrics(self) -> MetricSnapshot:
        """Get current metrics snapshot."""
        return self.current_metrics

    def get_recent_metrics(self, count: int = 100) -> List[MetricSnapshot]:
        """Get recent metrics history."""
        return self.metrics_history[-count:]

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_diagnostic_info(self) -> DiagnosticInfo:
        """Get diagnostic information."""
        recent_metrics = self.get_recent_metrics(100)

        # Calculate diagnostics
        training_stable = True
        performance_good = True
        resources_healthy = True

        if recent_metrics:
            # Check training stability
            recent_losses = [m.loss for m in recent_metrics[-10:] if m.loss is not None]
            if recent_losses:
                import numpy as np
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                if loss_std > loss_mean * 0.5:
                    training_stable = False

            # Check performance
            recent_throughput = [m.throughput for m in recent_metrics[-10:] if m.throughput is not None]
            if recent_throughput:
                avg_throughput = sum(recent_throughput) / len(recent_throughput)
                if avg_throughput < 1.0:  # Less than 1 sample/sec
                    performance_good = False

            # Check resources
            recent_memory = [m.memory_usage for m in recent_metrics[-10:] if m.memory_usage is not None]
            if recent_memory:
                avg_memory = sum(recent_memory) / len(recent_memory)
                if avg_memory > 0.9:  # More than 90% memory usage
                    resources_healthy = False

        # Count recent alerts
        hour_ago = time.time() - 3600
        recent_alerts = [a for a in self.alerts.values() if a.timestamp > hour_ago]
        error_count = len([a for a in recent_alerts if a.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]])
        warning_count = len([a for a in recent_alerts if a.level == AlertLevel.WARNING])

        # Calculate average step time
        avg_step_time = 0.1  # Placeholder

        # Calculate memory efficiency
        memory_efficiency = 0.8  # Placeholder

        # Generate recommendations
        recommendations = []
        if not training_stable:
            recommendations.append("Consider reducing learning rate for stability")
        if not performance_good:
            recommendations.append("Optimize data pipeline or increase batch size")
        if not resources_healthy:
            recommendations.append("Reduce memory usage or upgrade hardware")

        return DiagnosticInfo(
            timestamp=time.time(),
            training_stable=training_stable,
            performance_good=performance_good,
            resources_healthy=resources_healthy,
            error_count_1h=error_count,
            warning_count_1h=warning_count,
            avg_step_time=avg_step_time,
            memory_efficiency=memory_efficiency,
            recommendations=recommendations
        )


class MonitoringServer:
    """Simple monitoring server for external access."""

    def __init__(self, data_provider: MonitoringDataProvider, port: int = 8888):
        self.data_provider = data_provider
        self.port = port
        self.is_running = False
        self.server_thread = None

        # Request handlers
        self.handlers = {
            MonitoringEndpoint.HEALTH_STATUS: self._handle_health_status,
            MonitoringEndpoint.METRICS: self._handle_metrics,
            MonitoringEndpoint.ALERTS: self._handle_alerts,
            MonitoringEndpoint.DIAGNOSTICS: self._handle_diagnostics,
            MonitoringEndpoint.SYSTEM_INFO: self._handle_system_info
        }

        # Access log
        self.access_log = []

    def start(self):
        """Start the monitoring server."""
        if self.is_running:
            return

        self.is_running = True
        print(f"🌐 Starting monitoring server on port {self.port}")
        print("Available endpoints:")
        for endpoint in MonitoringEndpoint:
            print(f"  http://localhost:{self.port}{endpoint.value}")

        # In a real implementation, this would start an actual HTTP server
        # For now, we simulate it
        print("✓ Monitoring server started (simulated)")

    def stop(self):
        """Stop the monitoring server."""
        if not self.is_running:
            return

        self.is_running = False
        print("🛑 Monitoring server stopped")

    def handle_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a monitoring request (simulated).

        Args:
            endpoint: API endpoint path
            params: Request parameters

        Returns:
            JSON response
        """
        params = params or {}

        # Log access
        self.access_log.append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'params': params
        })

        # Keep access log size reasonable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]

        # Find handler
        endpoint_enum = None
        for ep in MonitoringEndpoint:
            if ep.value == endpoint:
                endpoint_enum = ep
                break

        if endpoint_enum not in self.handlers:
            return {
                'error': 'Endpoint not found',
                'available_endpoints': [ep.value for ep in MonitoringEndpoint]
            }

        try:
            return self.handlers[endpoint_enum](params)
        except Exception as e:
            return {
                'error': f'Internal server error: {str(e)}',
                'timestamp': time.time()
            }

    def _handle_health_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health status request."""
        health_status = self.data_provider.get_health_status()

        return {
            'status': 'success',
            'data': asdict(health_status),
            'timestamp': time.time()
        }

    def _handle_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle metrics request."""
        count = params.get('count', 100)
        current_only = params.get('current_only', False)

        if current_only:
            current_metrics = self.data_provider.get_current_metrics()
            return {
                'status': 'success',
                'data': asdict(current_metrics),
                'timestamp': time.time()
            }
        else:
            recent_metrics = self.data_provider.get_recent_metrics(count)
            return {
                'status': 'success',
                'data': {
                    'count': len(recent_metrics),
                    'metrics': [asdict(m) for m in recent_metrics]
                },
                'timestamp': time.time()
            }

    def _handle_alerts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alerts request."""
        active_only = params.get('active_only', True)

        if active_only:
            alerts = self.data_provider.get_active_alerts()
        else:
            # In a real implementation, this would get all alerts from storage
            alerts = self.data_provider.get_active_alerts()

        return {
            'status': 'success',
            'data': {
                'count': len(alerts),
                'alerts': [asdict(alert) for alert in alerts]
            },
            'timestamp': time.time()
        }

    def _handle_diagnostics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle diagnostics request."""
        diagnostic_info = self.data_provider.get_diagnostic_info()

        return {
            'status': 'success',
            'data': asdict(diagnostic_info),
            'timestamp': time.time()
        }

    def _handle_system_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system info request."""
        # Collect system information
        system_info = {
            'python_version': '3.8+',
            'torch_version': 'torch.__version__',
            'gpu_available': True,  # torch.cuda.is_available()
            'gpu_count': 1,  # torch.cuda.device_count()
            'monitoring_uptime': time.time() - getattr(self.data_provider, 'start_time', time.time()),
            'server_uptime': time.time() - getattr(self, 'start_time', time.time()),
            'total_requests': len(self.access_log)
        }

        return {
            'status': 'success',
            'data': system_info,
            'timestamp': time.time()
        }

    def get_server_stats(self) -> Dict[str, Any]:
        """Get monitoring server statistics."""
        if not self.access_log:
            return {'message': 'No requests received yet'}

        # Count requests by endpoint
        endpoint_counts = {}
        for log_entry in self.access_log:
            endpoint = log_entry['endpoint']
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        # Calculate request rate
        if len(self.access_log) > 1:
            time_span = self.access_log[-1]['timestamp'] - self.access_log[0]['timestamp']
            request_rate = len(self.access_log) / max(time_span, 1)
        else:
            request_rate = 0

        return {
            'total_requests': len(self.access_log),
            'request_rate_per_second': request_rate,
            'endpoint_counts': endpoint_counts,
            'last_request': self.access_log[-1]['timestamp'] if self.access_log else None,
            'server_running': self.is_running
        }


class HealthMonitoringAPI:
    """
    Main health monitoring API that integrates all components.

    Features:
    - RESTful API for health monitoring
    - Real-time metrics access
    - Alert management
    - Diagnostic information
    - External monitoring integration
    """

    def __init__(self, port: int = 8888, enable_server: bool = True):
        self.data_provider = TrainingMonitoringProvider()
        self.server = MonitoringServer(self.data_provider, port) if enable_server else None

        # Integration callbacks
        self.health_callbacks = []
        self.alert_callbacks = []

        # Monitoring state
        self.monitoring_active = False

    def start_monitoring(self):
        """Start the monitoring API."""
        self.monitoring_active = True

        if self.server:
            self.server.start()

        print("✓ Health Monitoring API started")

    def stop_monitoring(self):
        """Stop the monitoring API."""
        self.monitoring_active = False

        if self.server:
            self.server.stop()

        print("✓ Health Monitoring API stopped")

    def update_training_metrics(self, **kwargs):
        """Update training metrics."""
        self.data_provider.update_metrics(**kwargs)

        # Trigger health callbacks
        if self.health_callbacks:
            health_status = self.data_provider.get_health_status()
            for callback in self.health_callbacks:
                try:
                    callback(health_status)
                except Exception as e:
                    print(f"Health callback error: {e}")

    def update_component_status(self, component: str, status: str):
        """Update component status."""
        self.data_provider.update_component_status(component, status)

    def add_alert(self, level: str, title: str, message: str, component: str = "system"):
        """Add new alert."""
        alert_level = AlertLevel(level.lower())
        self.data_provider.add_alert(alert_level, title, message, component)

        # Trigger alert callbacks
        if self.alert_callbacks:
            alerts = self.data_provider.get_active_alerts()
            latest_alert = alerts[-1] if alerts else None
            for callback in self.alert_callbacks:
                try:
                    callback(latest_alert)
                except Exception as e:
                    print(f"Alert callback error: {e}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        self.data_provider.resolve_alert(alert_id)

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if self.server:
            return self.server.handle_request(MonitoringEndpoint.HEALTH_STATUS.value)
        else:
            health_status = self.data_provider.get_health_status()
            return asdict(health_status)

    def get_metrics(self, count: int = 100, current_only: bool = False) -> Dict[str, Any]:
        """Get training metrics."""
        params = {'count': count, 'current_only': current_only}
        if self.server:
            return self.server.handle_request(MonitoringEndpoint.METRICS.value, params)
        else:
            if current_only:
                metrics = self.data_provider.get_current_metrics()
                return asdict(metrics)
            else:
                metrics = self.data_provider.get_recent_metrics(count)
                return {'metrics': [asdict(m) for m in metrics]}

    def get_alerts(self, active_only: bool = True) -> Dict[str, Any]:
        """Get system alerts."""
        params = {'active_only': active_only}
        if self.server:
            return self.server.handle_request(MonitoringEndpoint.ALERTS.value, params)
        else:
            alerts = self.data_provider.get_active_alerts()
            return {'alerts': [asdict(a) for a in alerts]}

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        if self.server:
            return self.server.handle_request(MonitoringEndpoint.DIAGNOSTICS.value)
        else:
            diagnostics = self.data_provider.get_diagnostic_info()
            return asdict(diagnostics)

    def add_health_callback(self, callback: Callable[[HealthStatus], None]):
        """Add callback for health status changes."""
        self.health_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for new alerts."""
        self.alert_callbacks.append(callback)

    def export_monitoring_data(self, filepath: str, format: str = 'json'):
        """Export all monitoring data."""
        export_data = {
            'export_timestamp': time.time(),
            'health_status': asdict(self.data_provider.get_health_status()),
            'recent_metrics': [asdict(m) for m in self.data_provider.get_recent_metrics(1000)],
            'active_alerts': [asdict(a) for a in self.data_provider.get_active_alerts()],
            'diagnostics': asdict(self.data_provider.get_diagnostic_info())
        }

        if self.server:
            export_data['server_stats'] = self.server.get_server_stats()

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_monitoring_dashboard_url(self) -> Optional[str]:
        """Create URL for monitoring dashboard."""
        if self.server and self.server.is_running:
            return f"http://localhost:{self.server.port}/api/v1/health"
        return None

    def get_api_status(self) -> Dict[str, Any]:
        """Get API status information."""
        status = {
            'monitoring_active': self.monitoring_active,
            'server_running': self.server.is_running if self.server else False,
            'health_callbacks': len(self.health_callbacks),
            'alert_callbacks': len(self.alert_callbacks),
            'data_provider_type': type(self.data_provider).__name__
        }

        if self.server:
            status['server_stats'] = self.server.get_server_stats()
            status['dashboard_url'] = self.create_monitoring_dashboard_url()

        return status


# Convenience functions
def create_monitoring_api(port: int = 8888, auto_start: bool = True) -> HealthMonitoringAPI:
    """Create and optionally start monitoring API."""
    api = HealthMonitoringAPI(port=port, enable_server=True)

    if auto_start:
        api.start_monitoring()

    return api


def create_simple_monitoring() -> HealthMonitoringAPI:
    """Create simple monitoring without server."""
    return HealthMonitoringAPI(enable_server=False)


# Integration helpers
def integrate_with_wandb(api: HealthMonitoringAPI):
    """Integrate monitoring API with Weights & Biases."""
    def log_health_to_wandb(health_status: HealthStatus):
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    'monitoring/health_score': health_status.score,
                    'monitoring/active_alerts': health_status.active_alerts,
                    'monitoring/uptime_hours': health_status.uptime_seconds / 3600
                })
        except ImportError:
            pass

    api.add_health_callback(log_health_to_wandb)


def integrate_with_tensorboard(api: HealthMonitoringAPI, log_dir: str = "logs"):
    """Integrate monitoring API with TensorBoard."""
    def log_health_to_tensorboard(health_status: HealthStatus):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
            step = int(health_status.timestamp)

            writer.add_scalar('monitoring/health_score', health_status.score, step)
            writer.add_scalar('monitoring/active_alerts', health_status.active_alerts, step)
            writer.add_scalar('monitoring/uptime_hours', health_status.uptime_seconds / 3600, step)

            writer.close()
        except ImportError:
            pass

    api.add_health_callback(log_health_to_tensorboard)