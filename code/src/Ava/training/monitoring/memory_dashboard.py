"""
Real-time Memory Profiling Dashboard for LLM Training

Provides comprehensive memory tracking and visualization for:
- GPU memory allocation by component
- Memory usage over time
- Optimization recommendations
- Memory leak detection
- Per-layer memory breakdown

Usage:
    >>> from Ava.training.monitoring.memory_dashboard import MemoryDashboard
    >>>
    >>> dashboard = MemoryDashboard(model)
    >>> dashboard.start_profiling()
    >>>
    >>> # During training
    >>> dashboard.record_step(step=100, loss=0.5)
    >>>
    >>> # Get recommendations
    >>> dashboard.print_recommendations()
    >>>
    >>> # Save report
    >>> dashboard.save_report('memory_report.html')
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import time
import logging
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""
    timestamp: float
    step: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    # Breakdown by component
    model_mb: float = 0.0
    optimizer_mb: float = 0.0
    gradients_mb: float = 0.0
    activations_mb: float = 0.0
    other_mb: float = 0.0
    # Training info
    loss: Optional[float] = None
    lr: Optional[float] = None
    # Additional metrics
    fragmentation_ratio: float = 0.0


@dataclass
class ComponentMemory:
    """Memory usage for a specific component."""
    name: str
    size_mb: float
    size_bytes: int
    dtype: str
    shape: Tuple[int, ...]
    device: str
    category: str  # 'model', 'optimizer', 'gradient', 'activation', 'other'


class MemoryDashboard:
    """
    Real-time memory profiling and visualization dashboard.

    Tracks memory usage across training, identifies bottlenecks,
    and provides optimization recommendations.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        track_gradients: bool = True,
        track_activations: bool = False,  # Expensive, disabled by default
        history_size: int = 1000,
        sample_interval: int = 10,  # Record every N steps
    ):
        """
        Initialize memory dashboard.

        Args:
            model: PyTorch model to profile
            track_gradients: Track gradient memory
            track_activations: Track activation memory (expensive)
            history_size: Number of snapshots to keep
            sample_interval: Record every N steps
        """
        self.model = model
        self.track_gradients = track_gradients
        self.track_activations = track_activations
        self.sample_interval = sample_interval

        # Memory history
        self.history: deque = deque(maxlen=history_size)
        self.component_history: Dict[str, List[float]] = defaultdict(list)

        # Current state
        self.current_step = 0
        self.start_time = time.time()
        self.is_profiling = False

        # Component tracking
        self.tracked_tensors: Dict[str, ComponentMemory] = {}
        self.allocation_tracker: Dict[int, str] = {}  # tensor id -> name

        # Leak detection
        self.leak_candidates: List[Tuple[str, float]] = []
        self.baseline_memory: Optional[float] = None

        # Recommendations cache
        self._recommendations: Optional[List[str]] = None

    def start_profiling(self):
        """Start memory profiling."""
        self.is_profiling = True
        self.start_time = time.time()
        self.baseline_memory = self._get_current_memory_mb()
        logger.info("Memory profiling started")

        if self.model is not None:
            self._profile_model()

    def stop_profiling(self):
        """Stop memory profiling."""
        self.is_profiling = False
        logger.info("Memory profiling stopped")

    def record_step(
        self,
        step: int,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        force: bool = False
    ):
        """
        Record memory snapshot for current step.

        Args:
            step: Training step number
            loss: Current loss value
            lr: Current learning rate
            force: Force recording even if not at sample interval
        """
        if not self.is_profiling:
            return

        self.current_step = step

        # Sample at intervals
        if not force and step % self.sample_interval != 0:
            return

        snapshot = self._take_snapshot(step, loss, lr)
        self.history.append(snapshot)

        # Check for memory leaks
        if len(self.history) > 100:
            self._detect_leaks()

    def _take_snapshot(
        self,
        step: int,
        loss: Optional[float] = None,
        lr: Optional[float] = None
    ) -> MemorySnapshot:
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=time.time(),
                step=step,
                allocated_mb=0.0,
                reserved_mb=0.0,
                max_allocated_mb=0.0,
                loss=loss,
                lr=lr
            )

        # Get basic memory stats
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Calculate fragmentation
        fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0.0

        # Estimate memory breakdown
        model_mb, optimizer_mb, gradients_mb = self._estimate_memory_breakdown()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            step=step,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            model_mb=model_mb,
            optimizer_mb=optimizer_mb,
            gradients_mb=gradients_mb,
            activations_mb=max(0, allocated - model_mb - optimizer_mb - gradients_mb),
            loss=loss,
            lr=lr,
            fragmentation_ratio=fragmentation
        )

        return snapshot

    def _get_current_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    def _profile_model(self):
        """Profile model parameters and buffers."""
        if self.model is None:
            return

        total_params = 0
        total_bytes = 0

        for name, param in self.model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            size_mb = size_bytes / (1024 ** 2)

            component = ComponentMemory(
                name=name,
                size_mb=size_mb,
                size_bytes=size_bytes,
                dtype=str(param.dtype),
                shape=tuple(param.shape),
                device=str(param.device),
                category='model'
            )

            self.tracked_tensors[name] = component
            total_params += param.numel()
            total_bytes += size_bytes

        logger.info(
            f"Model profiling: {total_params:,} parameters, "
            f"{total_bytes / (1024 ** 2):.2f} MB"
        )

    def _estimate_memory_breakdown(self) -> Tuple[float, float, float]:
        """
        Estimate memory usage breakdown.

        Returns:
            (model_mb, optimizer_mb, gradients_mb)
        """
        model_mb = 0.0
        gradients_mb = 0.0

        if self.model is not None:
            # Model parameters
            for param in self.model.parameters():
                param_mb = param.numel() * param.element_size() / (1024 ** 2)
                model_mb += param_mb

                # Gradients (if tracking)
                if self.track_gradients and param.grad is not None:
                    gradients_mb += param_mb  # Gradients same size as params

        # Estimate optimizer state (typically 2x params for Adam, 1x for Lion)
        # This is a rough estimate
        optimizer_mb = model_mb * 2.0

        return model_mb, optimizer_mb, gradients_mb

    def _detect_leaks(self):
        """Detect potential memory leaks."""
        if len(self.history) < 100:
            return

        # Check if memory is steadily increasing
        recent_memory = [s.allocated_mb for s in list(self.history)[-50:]]
        early_memory = [s.allocated_mb for s in list(self.history)[-100:-50]]

        recent_avg = sum(recent_memory) / len(recent_memory)
        early_avg = sum(early_memory) / len(early_memory)

        # If memory increased by >20% over time, possible leak
        if recent_avg > early_avg * 1.2:
            leak_mb = recent_avg - early_avg
            self.leak_candidates.append((f"Step {self.current_step}", leak_mb))
            logger.warning(
                f"Potential memory leak detected: {leak_mb:.2f} MB increase "
                f"over last 50 steps"
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self.history:
            return {}

        latest = self.history[-1]

        # Peak memory
        peak_allocated = max(s.allocated_mb for s in self.history)
        peak_reserved = max(s.reserved_mb for s in self.history)

        # Average memory
        avg_allocated = sum(s.allocated_mb for s in self.history) / len(self.history)
        avg_fragmentation = sum(s.fragmentation_ratio for s in self.history) / len(self.history)

        return {
            'current_allocated_mb': latest.allocated_mb,
            'current_reserved_mb': latest.reserved_mb,
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'avg_allocated_mb': avg_allocated,
            'avg_fragmentation': avg_fragmentation,
            'total_steps': self.current_step,
            'snapshots_recorded': len(self.history),
            'model_mb': latest.model_mb,
            'optimizer_mb': latest.optimizer_mb,
            'gradients_mb': latest.gradients_mb,
            'activations_mb': latest.activations_mb,
        }

    def get_recommendations(self) -> List[str]:
        """
        Generate memory optimization recommendations.

        Returns:
            List of recommendations
        """
        if self._recommendations is not None:
            return self._recommendations

        recommendations = []

        if not self.history:
            return recommendations

        latest = self.history[-1]
        summary = self.get_summary()

        # High fragmentation
        if summary['avg_fragmentation'] > 0.3:
            recommendations.append(
                f"⚠️ High memory fragmentation ({summary['avg_fragmentation']:.1%}). "
                f"Consider using `torch.cuda.empty_cache()` periodically."
            )

        # Large activation memory
        if latest.activations_mb > latest.model_mb:
            recommendations.append(
                f"💡 Activation memory ({latest.activations_mb:.0f} MB) exceeds model size. "
                f"Enable gradient checkpointing for 60-80% reduction."
            )

        # Large optimizer state
        if latest.optimizer_mb > latest.model_mb * 1.5:
            recommendations.append(
                f"💡 Optimizer state ({latest.optimizer_mb:.0f} MB) is large. "
                f"Consider 8-bit optimizers (Lion8bit, AdamW8bit) for 75-87% reduction."
            )

        # Large gradients
        if latest.gradients_mb > latest.model_mb * 0.5:
            recommendations.append(
                f"💡 Gradient memory ({latest.gradients_mb:.0f} MB) is significant. "
                f"Consider GaLore optimizer for 50-65% gradient memory reduction."
            )

        # Memory leaks
        if self.leak_candidates:
            recommendations.append(
                f"⚠️ Detected {len(self.leak_candidates)} potential memory leaks. "
                f"Check for unintended references to tensors."
            )

        # Peak memory usage
        peak_ratio = summary['peak_allocated_mb'] / summary['avg_allocated_mb'] if summary['avg_allocated_mb'] > 0 else 1.0
        if peak_ratio > 1.5:
            recommendations.append(
                f"💡 Peak memory ({summary['peak_allocated_mb']:.0f} MB) is {peak_ratio:.1f}x average. "
                f"Consider reducing batch size or sequence length."
            )

        # General recommendations based on total memory
        total_mb = latest.allocated_mb
        if total_mb > 16000:  # >16GB
            recommendations.append(
                f"💡 High memory usage ({total_mb:.0f} MB). Consider enabling: "
                f"1) Flash Attention, 2) DeepSpeed ZeRO-3, 3) CPU offloading"
            )

        self._recommendations = recommendations
        return recommendations

    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary()

        if not summary:
            print("No profiling data available")
            return

        print("\n" + "=" * 70)
        print("Memory Profiling Summary")
        print("=" * 70)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Snapshots: {summary['snapshots_recorded']}")
        print(f"\nCurrent Memory Usage:")
        print(f"  Allocated: {summary['current_allocated_mb']:>10.2f} MB")
        print(f"  Reserved:  {summary['current_reserved_mb']:>10.2f} MB")
        print(f"\nPeak Memory Usage:")
        print(f"  Allocated: {summary['peak_allocated_mb']:>10.2f} MB")
        print(f"  Reserved:  {summary['peak_reserved_mb']:>10.2f} MB")
        print(f"\nMemory Breakdown:")
        print(f"  Model:       {summary['model_mb']:>10.2f} MB")
        print(f"  Optimizer:   {summary['optimizer_mb']:>10.2f} MB")
        print(f"  Gradients:   {summary['gradients_mb']:>10.2f} MB")
        print(f"  Activations: {summary['activations_mb']:>10.2f} MB")
        print(f"\nFragmentation: {summary['avg_fragmentation']:.1%}")
        print("=" * 70)

    def print_recommendations(self):
        """Print optimization recommendations."""
        recommendations = self.get_recommendations()

        if not recommendations:
            print("\n✅ No specific recommendations - memory usage looks good!")
            return

        print("\n" + "=" * 70)
        print("Memory Optimization Recommendations")
        print("=" * 70)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
        print("\n" + "=" * 70)

    def save_report(self, filepath: str = 'memory_report.html'):
        """
        Save detailed HTML report.

        Args:
            filepath: Output file path
        """
        html_content = self._generate_html_report()

        with open(filepath, 'w') as f:
            f.write(html_content)

        logger.info(f"Memory report saved to {filepath}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        summary = self.get_summary()
        recommendations = self.get_recommendations()

        # Generate memory timeline data
        timeline_data = [
            {
                'step': s.step,
                'allocated': s.allocated_mb,
                'reserved': s.reserved_mb,
                'model': s.model_mb,
                'optimizer': s.optimizer_mb,
                'gradients': s.gradients_mb,
                'activations': s.activations_mb,
            }
            for s in self.history
        ]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Profiling Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .chart {{
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Memory Profiling Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Summary Statistics</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-label">Current Memory</div>
                <div class="metric-value">{summary.get('current_allocated_mb', 0):.0f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Peak Memory</div>
                <div class="metric-value">{summary.get('peak_allocated_mb', 0):.0f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">{summary.get('model_mb', 0):.0f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Fragmentation</div>
                <div class="metric-value">{summary.get('avg_fragmentation', 0):.1%}</div>
            </div>
        </div>

        <h2>Memory Timeline</h2>
        <div id="timeline-chart" class="chart"></div>

        <h2>Memory Breakdown</h2>
        <div id="breakdown-chart" class="chart"></div>

        <h2>Optimization Recommendations</h2>
        {''.join(f'<div class="recommendation">{rec}</div>' for rec in recommendations)}
    </div>

    <script>
        // Timeline chart
        const timelineData = {json.dumps(timeline_data)};
        const steps = timelineData.map(d => d.step);

        Plotly.newPlot('timeline-chart', [
            {{
                x: steps,
                y: timelineData.map(d => d.allocated),
                name: 'Allocated',
                type: 'scatter',
                mode: 'lines',
                line: {{color: '#007bff'}}
            }},
            {{
                x: steps,
                y: timelineData.map(d => d.reserved),
                name: 'Reserved',
                type: 'scatter',
                mode: 'lines',
                line: {{color: '#6c757d', dash: 'dash'}}
            }}
        ], {{
            title: 'Memory Usage Over Time',
            xaxis: {{title: 'Training Step'}},
            yaxis: {{title: 'Memory (MB)'}}
        }});

        // Breakdown chart
        const latest = timelineData[timelineData.length - 1];
        Plotly.newPlot('breakdown-chart', [{{
            values: [latest.model, latest.optimizer, latest.gradients, latest.activations],
            labels: ['Model', 'Optimizer', 'Gradients', 'Activations'],
            type: 'pie',
            marker: {{
                colors: ['#007bff', '#28a745', '#ffc107', '#dc3545']
            }}
        }}], {{
            title: 'Current Memory Breakdown'
        }});
    </script>
</body>
</html>
"""
        return html

    def export_json(self, filepath: str = 'memory_profile.json'):
        """
        Export profiling data as JSON.

        Args:
            filepath: Output file path
        """
        data = {
            'summary': self.get_summary(),
            'recommendations': self.get_recommendations(),
            'history': [
                {
                    'step': s.step,
                    'timestamp': s.timestamp,
                    'allocated_mb': s.allocated_mb,
                    'reserved_mb': s.reserved_mb,
                    'max_allocated_mb': s.max_allocated_mb,
                    'model_mb': s.model_mb,
                    'optimizer_mb': s.optimizer_mb,
                    'gradients_mb': s.gradients_mb,
                    'activations_mb': s.activations_mb,
                    'fragmentation': s.fragmentation_ratio,
                    'loss': s.loss,
                    'lr': s.lr,
                }
                for s in self.history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Memory profile exported to {filepath}")


# Convenience functions
def profile_training(
    model: nn.Module,
    output_path: str = 'memory_report.html'
) -> MemoryDashboard:
    """
    Quick start function for profiling training.

    Args:
        model: Model to profile
        output_path: Where to save the report

    Returns:
        MemoryDashboard instance

    Example:
        >>> dashboard = profile_training(model)
        >>> # ... training loop ...
        >>> dashboard.record_step(step=i, loss=loss.item())
        >>> # ... end of training ...
        >>> dashboard.save_report()
    """
    dashboard = MemoryDashboard(model)
    dashboard.start_profiling()
    return dashboard


__all__ = [
    'MemoryDashboard',
    'MemorySnapshot',
    'ComponentMemory',
    'profile_training'
]
