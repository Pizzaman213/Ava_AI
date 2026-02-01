"""
Training Diagnostics Module.

Provides comprehensive diagnostic collection during training.

Features:
- Per-layer gradient statistics (norm, mean, std, max, min)
- Expert routing diagnostics for MoE models
- Memory breakdown by component (params, grads, optimizer, activations)
- Timing profiling for training phases

Usage:
    from ava.logging.diagnostics import DiagnosticsManager, LayerGradientStats

    diagnostics = DiagnosticsManager(context)
    diagnostics.configure(DiagnosticsConfig(enabled=True))

    with diagnostics.time_phase('forward'):
        outputs = model(inputs)

    layer_stats = diagnostics.collect_per_layer_gradients(model, step=100)
"""

from ava.logging.diagnostics.training import (
    DiagnosticsManager,
    LayerGradientStats,
    ExpertRoutingStats,
    MemoryBreakdown,
    TimingProfile,
)

__all__ = [
    'DiagnosticsManager',
    'LayerGradientStats',
    'ExpertRoutingStats',
    'MemoryBreakdown',
    'TimingProfile',
]
