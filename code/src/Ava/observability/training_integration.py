"""
Training Integration for Observability (Phase 7)

Integration layer that connects all observability components with the training system.
"""

import time
import torch
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path

# Import all observability components
from .hierarchical_logging import HierarchicalLogger, LogLevel, setup_hierarchical_logging, log_performance
from .health_dashboard import HealthDashboard, HealthMetrics, create_health_dashboard
from .metric_collector import OptimizedMetricCollector, create_training_metric_collector
from .training_validator import TrainingValidator, create_training_validator
from .post_mortem import PostMortemAnalyzer, create_post_mortem_analyzer
from .monitoring_api import HealthMonitoringAPI, create_monitoring_api


@dataclass
class ObservabilityConfig:
    """Configuration for observability integration."""
    # Logging
    enable_hierarchical_logging: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs/observability"

    # Health Dashboard
    enable_health_dashboard: bool = True
    dashboard_update_interval: float = 5.0

    # Metric Collection
    enable_optimized_metrics: bool = True
    metric_sampling_rate: float = 0.2
    enable_async_metrics: bool = True

    # Validation
    enable_training_validation: bool = True
    enable_continuous_monitoring: bool = True

    # Post-Mortem Analysis
    enable_post_mortem: bool = True
    enable_auto_failure_detection: bool = True

    # Monitoring API
    enable_monitoring_api: bool = True
    api_port: int = 8888

    # Integration
    export_interval: int = 100  # Export data every N steps
    checkpoint_observability: bool = True


class ObservabilityIntegration:
    """
    Main integration class that orchestrates all observability components.

    This class provides a unified interface for integrating comprehensive
    observability into training systems.
    """

    def __init__(self, config: ObservabilityConfig = None):
        self.config = config or ObservabilityConfig()

        # Initialize components
        self.logger: Optional[HierarchicalLogger] = None
        self.dashboard: Optional[HealthDashboard] = None
        self.metric_collector: Optional[OptimizedMetricCollector] = None
        self.validator: Optional[TrainingValidator] = None
        self.post_mortem: Optional[PostMortemAnalyzer] = None
        self.monitoring_api: Optional[HealthMonitoringAPI] = None

        # State
        self.is_initialized = False
        self.training_active = False
        self.last_export = 0

        # Storage
        self.training_history = []
        self.validation_reports = []

    def initialize(self, training_context: Dict[str, Any] = None):
        """
        Initialize all observability components.

        Args:
            training_context: Context containing model, optimizer, data loader, etc.
        """
        print("🔧 Initializing Phase 7 Observability & Debugging...")

        training_context = training_context or {}

        # Initialize hierarchical logging
        if self.config.enable_hierarchical_logging:
            self.logger = setup_hierarchical_logging(
                name="training_observability",
                min_level=LogLevel[self.config.log_level.upper()],
                log_dir=self.config.log_dir
            )
            self.logger.info("Hierarchical logging system initialized")

        # Initialize health dashboard
        if self.config.enable_health_dashboard:
            self.dashboard = create_health_dashboard()
            self.dashboard.start()
            self.logger.info("Health dashboard started") if self.logger else None

        # Initialize optimized metric collector
        if self.config.enable_optimized_metrics:
            self.metric_collector = create_training_metric_collector()

            # Register common training metrics
            self._register_training_metrics()
            self.logger.info("Optimized metric collector initialized") if self.logger else None

        # Initialize training validator
        if self.config.enable_training_validation:
            self.validator = create_training_validator(
                enable_monitoring=self.config.enable_continuous_monitoring
            )

            # Run pre-flight checks if training context provided
            if training_context:
                try:
                    report = self.validator.run_pre_flight_checks(training_context)
                    self.validation_reports.append(report)
                    self.logger.info(f"Pre-flight validation: {report.overall_status}") if self.logger else None
                except Exception as e:
                    self.logger.error(f"Pre-flight validation failed: {e}") if self.logger else None

        # Initialize post-mortem analyzer
        if self.config.enable_post_mortem:
            self.post_mortem = create_post_mortem_analyzer(
                auto_detect=self.config.enable_auto_failure_detection
            )
            self.logger.info("Post-mortem analyzer initialized") if self.logger else None

        # Initialize monitoring API
        if self.config.enable_monitoring_api:
            self.monitoring_api = create_monitoring_api(
                port=self.config.api_port,
                auto_start=True
            )
            self.logger.info(f"Monitoring API started on port {self.config.api_port}") if self.logger else None

        # Setup cross-component integration
        self._setup_integrations()

        self.is_initialized = True
        self.logger.info("✓ Observability integration initialized successfully") if self.logger else None

        print("✓ Phase 7 Observability & Debugging initialized")

    def start_training_observation(self, training_context: Dict[str, Any] = None):
        """Start observing training process."""
        if not self.is_initialized:
            self.initialize(training_context)

        self.training_active = True

        # Start continuous monitoring if enabled
        if self.validator and self.config.enable_continuous_monitoring:
            self.validator.start_continuous_monitoring(training_context or {})

        self.logger.info("🚀 Training observation started") if self.logger else None

    def stop_training_observation(self):
        """Stop observing training process."""
        self.training_active = False

        # Stop components
        if self.validator:
            self.validator.stop_continuous_monitoring()

        if self.dashboard:
            self.dashboard.stop()

        if self.monitoring_api:
            self.monitoring_api.stop_monitoring()

        self.logger.info("🛑 Training observation stopped") if self.logger else None

    def update_training_step(
        self,
        step: int,
        epoch: int = None,
        loss: float = None,
        learning_rate: float = None,
        batch_size: int = None,
        sequence_length: int = None,
        **kwargs
    ):
        """
        Update observability with training step information.

        This is the main method called from training loops.
        """
        timestamp = time.time()

        # Update logger context
        if self.logger:
            self.logger.update_context(
                step=step,
                epoch=epoch,
                batch_size=batch_size,
                sequence_length=sequence_length
            )

        # Collect metrics
        metrics = {}
        if self.metric_collector:
            context = {
                'step': step,
                'epoch': epoch,
                'loss': loss,
                'learning_rate': learning_rate,
                **kwargs
            }
            metrics = self.metric_collector.collect_metrics(step, context)

        # Update health dashboard
        if self.dashboard:
            # Merge metrics and kwargs to avoid duplicate keys
            merged_kwargs = {**metrics, **kwargs}
            self.dashboard.update_metrics(
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                batch_size=batch_size,
                sequence_length=sequence_length,
                **merged_kwargs
            )

        # Update post-mortem analyzer
        if self.post_mortem:
            self.post_mortem.update_training_metrics(
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                **kwargs
            )

        # Update monitoring API
        if self.monitoring_api:
            self.monitoring_api.update_training_metrics(
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                throughput=kwargs.get('samples_per_second'),
                gpu_utilization=kwargs.get('gpu_utilization'),
                memory_usage=kwargs.get('gpu_memory_used'),
                custom_metrics=metrics
            )

        # Update validator state
        if self.validator:
            self.validator.update_training_state(
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                **kwargs
            )

        # Store training history
        self.training_history.append({
            'timestamp': timestamp,
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'learning_rate': learning_rate,
            'metrics': metrics,
            **kwargs
        })

        # Periodic export
        if step - self.last_export >= self.config.export_interval:
            self._periodic_export(step)
            self.last_export = step

        # Log performance metrics
        if self.logger and step % 50 == 0:  # Every 50 steps
            self.logger.debug(f"Training step {step} completed", step=step, loss=loss)

    def handle_training_error(self, error: Exception, step: int, context: Dict[str, Any] = None):
        """Handle training errors with comprehensive analysis."""
        self.logger.error(f"Training error at step {step}: {error}", exception=error, step=step) if self.logger else None

        # Add alert to monitoring API
        if self.monitoring_api:
            self.monitoring_api.add_alert(
                level="error",
                title="Training Error",
                message=f"Error at step {step}: {str(error)}",
                component="training"
            )

        # Trigger post-mortem analysis if enabled
        if self.post_mortem and self.config.enable_auto_failure_detection:
            try:
                training_context = context or {}
                training_context.update({'step': step, 'error': error})

                failure_report = self.post_mortem.analyze_failure(training_context=training_context)
                self.logger.info(f"Post-mortem analysis completed: {failure_report.summary}") if self.logger else None

            except Exception as analysis_error:
                self.logger.error(f"Post-mortem analysis failed: {analysis_error}") if self.logger else None

    def handle_out_of_memory(self, step: int, context: Dict[str, Any] = None):
        """Handle OOM errors specifically."""
        self.logger.critical("Out of memory error detected", step=step) if self.logger else None

        # Add critical alert
        if self.monitoring_api:
            self.monitoring_api.add_alert(
                level="critical",
                title="Out of Memory",
                message=f"OOM error at step {step}",
                component="system"
            )

        # Trigger specific OOM analysis
        if self.post_mortem:
            from .post_mortem import FailureType
            evidence_data = {
                'memory_usage': context.get('gpu_memory_used', 0),
                'batch_size': context.get('batch_size', 0),
                'sequence_length': context.get('sequence_length', 0)
            }

            failure_report = self.post_mortem.analyzer.analyze_failure(
                FailureType.OUT_OF_MEMORY,
                evidence_data,
                context
            )

            self.logger.info(f"OOM analysis: {failure_report.summary}") if self.logger else None

    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        summary = {
            'timestamp': time.time(),
            'initialized': self.is_initialized,
            'training_active': self.training_active,
            'config': {
                'hierarchical_logging': self.config.enable_hierarchical_logging,
                'health_dashboard': self.config.enable_health_dashboard,
                'optimized_metrics': self.config.enable_optimized_metrics,
                'training_validation': self.config.enable_training_validation,
                'post_mortem': self.config.enable_post_mortem,
                'monitoring_api': self.config.enable_monitoring_api
            }
        }

        # Add component summaries
        if self.logger:
            summary['logging'] = self.logger.get_statistics()

        if self.dashboard:
            summary['health'] = self.dashboard.get_health_summary()

        if self.metric_collector:
            summary['metrics'] = self.metric_collector.get_collection_statistics()

        if self.validator:
            summary['validation'] = self.validator.get_validation_summary()

        if self.post_mortem:
            summary['post_mortem'] = self.post_mortem.get_failure_statistics()

        if self.monitoring_api:
            summary['monitoring_api'] = self.monitoring_api.get_api_status()

        summary['training_history_size'] = len(self.training_history)
        summary['validation_reports'] = len(self.validation_reports)

        return summary

    def export_all_data(self, export_dir: str = "observability_export"):
        """Export all observability data."""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Export training history
        if self.training_history:
            import json
            with open(export_path / f"training_history_{timestamp}.json", 'w') as f:
                json.dump(self.training_history, f, indent=2)

        # Export validation reports
        if self.validation_reports and self.validator:
            self.validator.export_report(
                export_path / f"validation_report_{timestamp}.json"
            )

        # Export post-mortem analysis
        if self.post_mortem:
            self.post_mortem.export_analysis(
                export_path / f"failure_analysis_{timestamp}.json"
            )

        # Export monitoring data
        if self.monitoring_api:
            self.monitoring_api.export_monitoring_data(
                export_path / f"monitoring_data_{timestamp}.json"
            )

        # Export health dashboard data
        if self.dashboard:
            self.dashboard.export_metrics(
                export_path / f"health_metrics_{timestamp}.json"
            )

        # Export metric collection data
        if self.metric_collector:
            metrics_data = self.metric_collector.export_metrics()
            if metrics_data:
                import json
                with open(export_path / f"collected_metrics_{timestamp}.json", 'w') as f:
                    json.dump(metrics_data, f, indent=2)

        self.logger.info(f"Exported all observability data to {export_path}") if self.logger else None
        print(f"📁 Exported observability data to {export_path}")

    def create_checkpoint_data(self) -> Dict[str, Any]:
        """Create checkpoint data for observability state."""
        if not self.config.checkpoint_observability:
            return {}

        checkpoint_data = {
            'observability_version': '1.0',
            'config': {
                'log_level': self.config.log_level,
                'enable_auto_failure_detection': self.config.enable_auto_failure_detection,
                'metric_sampling_rate': self.config.metric_sampling_rate
            },
            'training_step_count': len(self.training_history),
            'validation_report_count': len(self.validation_reports)
        }

        # Add component states
        if self.logger:
            checkpoint_data['logging_stats'] = self.logger.get_statistics()

        if self.metric_collector:
            checkpoint_data['metric_collector_stats'] = self.metric_collector.get_collection_statistics()

        if self.post_mortem:
            checkpoint_data['failure_stats'] = self.post_mortem.get_failure_statistics()

        return checkpoint_data

    def _register_training_metrics(self):
        """Register common training metrics with the collector."""
        if not self.metric_collector:
            return

        # Import metric computation functions
        from .metric_collector import compute_gradient_norm, compute_memory_usage, compute_model_statistics

        # Register metrics with different importance levels
        self.metric_collector.register_metric(
            name="gradient_norm",
            computation_func=compute_gradient_norm,
            importance=0.9,
            expensive=False
        )

        self.metric_collector.register_metric(
            name="memory_usage",
            computation_func=compute_memory_usage,
            importance=0.8,
            expensive=False
        )

        self.metric_collector.register_metric(
            name="model_statistics",
            computation_func=compute_model_statistics,
            importance=0.5,
            expensive=True  # This will be computed asynchronously
        )

    def _setup_integrations(self):
        """Setup cross-component integrations."""
        # Connect health dashboard alerts to monitoring API
        if self.dashboard and self.monitoring_api:
            def forward_alert(message: str):
                level = "warning" if "WARNING" in message else "error" if "ALERT" in message else "info"
                self.monitoring_api.add_alert(
                    level=level,
                    title="Health Dashboard Alert",
                    message=message,
                    component="health_dashboard"
                )

            self.dashboard.add_alert_callback(forward_alert)

        # Connect post-mortem alerts to monitoring API
        if self.post_mortem and self.monitoring_api:
            # This would be set up in the post-mortem analyzer to trigger alerts
            pass

        # Connect validator alerts to monitoring API
        if self.validator and self.monitoring_api:
            # This would be set up in the validator to trigger alerts
            pass

    def _periodic_export(self, step: int):
        """Perform periodic data export."""
        if self.logger:
            self.logger.debug(f"Performing periodic export at step {step}")

        # Update component statuses in monitoring API
        if self.monitoring_api:
            self.monitoring_api.update_component_status("training", "healthy")
            self.monitoring_api.update_component_status("data_pipeline", "healthy")
            self.monitoring_api.update_component_status("model", "healthy")

        # Keep training history size manageable
        if len(self.training_history) > 10000:
            self.training_history = self.training_history[-5000:]

    def shutdown(self):
        """Shutdown all observability components."""
        self.logger.info("🔧 Shutting down observability integration") if self.logger else None

        # Stop training observation
        self.stop_training_observation()

        # Shutdown individual components
        if self.metric_collector:
            self.metric_collector.shutdown()

        if self.logger:
            self.logger.shutdown()

        self.logger.info("✓ Observability integration shutdown complete") if self.logger else None


# Context manager for automatic lifecycle management
class ObservabilityContext:
    """Context manager for observability integration."""

    def __init__(self, config: ObservabilityConfig = None, training_context: Dict[str, Any] = None):
        self.integration = ObservabilityIntegration(config)
        self.training_context = training_context

    def __enter__(self) -> ObservabilityIntegration:
        self.integration.initialize(self.training_context)
        self.integration.start_training_observation(self.training_context)
        return self.integration

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Handle any exceptions that occurred during training
            self.integration.handle_training_error(exc_val, step=0, context={'exception_type': exc_type})

        self.integration.shutdown()


# Convenience functions
def create_observability_integration(
    enable_all: bool = True,
    log_level: str = "INFO",
    api_port: int = 8888,
    **kwargs
) -> ObservabilityIntegration:
    """Create observability integration with sensible defaults."""
    config = ObservabilityConfig(
        enable_hierarchical_logging=enable_all,
        enable_health_dashboard=enable_all,
        enable_optimized_metrics=enable_all,
        enable_training_validation=enable_all,
        enable_post_mortem=enable_all,
        enable_monitoring_api=enable_all,
        log_level=log_level,
        api_port=api_port,
        **kwargs
    )

    return ObservabilityIntegration(config)


def create_lightweight_observability() -> ObservabilityIntegration:
    """Create lightweight observability for minimal overhead."""
    config = ObservabilityConfig(
        enable_hierarchical_logging=True,
        enable_health_dashboard=False,
        enable_optimized_metrics=True,
        enable_training_validation=False,
        enable_post_mortem=True,
        enable_monitoring_api=False,
        log_level="WARNING",
        metric_sampling_rate=0.05  # Very low sampling
    )

    return ObservabilityIntegration(config)


# Decorator for automatic observability
def with_observability(
    config: ObservabilityConfig = None,
    auto_export: bool = True,
    export_interval: int = 1000
):
    """Decorator to add observability to training functions."""
    def decorator(training_func):
        def wrapper(*args, **kwargs):
            # Extract training context from kwargs
            training_context = kwargs.pop('observability_context', {})

            with ObservabilityContext(config, training_context) as obs:
                try:
                    # Add observability integration to kwargs
                    kwargs['observability'] = obs

                    # Execute training function
                    result = training_func(*args, **kwargs)

                    # Export data if requested
                    if auto_export:
                        obs.export_all_data()

                    return result

                except Exception as e:
                    # Handle training errors
                    obs.handle_training_error(e, step=0)
                    raise

        return wrapper
    return decorator