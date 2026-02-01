"""
Diagnostics Manager for detailed training insights.

Provides comprehensive diagnostic collection during training:
- Per-layer gradient statistics
- Expert routing diagnostics for MoE models
- Memory breakdown by component
- Timing profiling for training phases

Example:
    >>> from ava.config.training_config import DiagnosticsConfig
    >>> diagnostics = DiagnosticsManager(context)
    >>> diagnostics.configure(DiagnosticsConfig(enabled=True))
    >>> with diagnostics.time_phase('forward'):
    ...     outputs = model(inputs)
    >>> layer_stats = diagnostics.collect_per_layer_gradients(model, step=100)
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from ava.training.context import ManagerInterface, TrainingContext

if TYPE_CHECKING:
    from ava.config.training_config import DiagnosticsConfig

logger = logging.getLogger(__name__)


@dataclass
class LayerGradientStats:
    """Per-layer gradient statistics.

    Attributes:
        layer_name: Full name of the layer (e.g., 'model.layers.0.mlp')
        grad_norm: L2 norm of gradients
        grad_mean: Mean gradient value
        grad_std: Standard deviation of gradients
        grad_max: Maximum gradient value
        grad_min: Minimum gradient value
        num_params: Number of parameters in this layer
        has_grad: Whether gradients were present
    """
    layer_name: str
    grad_norm: float
    grad_mean: float
    grad_std: float
    grad_max: float
    grad_min: float
    num_params: int
    has_grad: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'layer_name': self.layer_name,
            'grad_norm': self.grad_norm,
            'grad_mean': self.grad_mean,
            'grad_std': self.grad_std,
            'grad_max': self.grad_max,
            'grad_min': self.grad_min,
            'num_params': self.num_params,
            'has_grad': self.has_grad,
        }


@dataclass
class ExpertRoutingStats:
    """Expert routing statistics for one step.

    Attributes:
        step: Training step this was collected at
        per_expert_load: Dict mapping expert ID to load fraction
        routing_entropy: Entropy of routing decisions (higher = more uniform)
        capacity_usage: Dict mapping expert ID to capacity usage fraction
        dropped_tokens: Number of tokens dropped due to capacity limits
        balance_score: Load balance score 0-1 (1 = perfectly balanced)
    """
    step: int
    per_expert_load: Dict[int, float] = field(default_factory=dict)
    routing_entropy: float = 0.0
    capacity_usage: Dict[int, float] = field(default_factory=dict)
    dropped_tokens: int = 0
    balance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'step': self.step,
            'per_expert_load': self.per_expert_load,
            'routing_entropy': self.routing_entropy,
            'capacity_usage': self.capacity_usage,
            'dropped_tokens': self.dropped_tokens,
            'balance_score': self.balance_score,
        }


@dataclass
class MemoryBreakdown:
    """Memory usage breakdown by component.

    All values are in gigabytes (GB).

    Attributes:
        total_allocated_gb: Total GPU memory allocated
        total_reserved_gb: Total GPU memory reserved (includes fragmentation)
        parameters_gb: Memory used by model parameters
        gradients_gb: Memory used by gradients
        optimizer_states_gb: Memory used by optimizer states
        activations_gb: Estimated memory for activations (computed as remainder)
        other_gb: Other/unaccounted memory
    """
    total_allocated_gb: float = 0.0
    total_reserved_gb: float = 0.0
    parameters_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_states_gb: float = 0.0
    activations_gb: float = 0.0
    other_gb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'total_allocated_gb': self.total_allocated_gb,
            'total_reserved_gb': self.total_reserved_gb,
            'parameters_gb': self.parameters_gb,
            'gradients_gb': self.gradients_gb,
            'optimizer_states_gb': self.optimizer_states_gb,
            'activations_gb': self.activations_gb,
            'other_gb': self.other_gb,
        }


@dataclass
class TimingProfile:
    """Timing breakdown for a training step.

    All values are in milliseconds (ms).

    Attributes:
        step: Training step this was collected at
        data_loading_ms: Time spent loading data
        forward_ms: Time spent in forward pass
        backward_ms: Time spent in backward pass
        optimizer_step_ms: Time spent in optimizer step
        total_step_ms: Total step time
    """
    step: int = 0
    data_loading_ms: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_step_ms: float = 0.0
    total_step_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'step': self.step,
            'data_loading_ms': self.data_loading_ms,
            'forward_ms': self.forward_ms,
            'backward_ms': self.backward_ms,
            'optimizer_step_ms': self.optimizer_step_ms,
            'total_step_ms': self.total_step_ms,
        }


class DiagnosticsManager(ManagerInterface):
    """
    Manages detailed diagnostic collection during training.

    Features:
    - Per-layer gradient statistics (norm, mean, std, max, min)
    - Expert routing analysis for MoE models
    - Memory breakdown by component (params, grads, optimizer, activations)
    - Step timing profiling (data, forward, backward, optimizer)

    All diagnostic collection is optional and controlled by DiagnosticsConfig.
    Diagnostics are only collected at configured intervals to minimize overhead.

    Example:
        >>> diagnostics = DiagnosticsManager(context)
        >>> diagnostics.initialize()
        >>> diagnostics.configure(config.diagnostics)
        >>> # In training loop:
        >>> with diagnostics.time_phase('forward'):
        ...     outputs = model(inputs)
        >>> stats = diagnostics.collect_per_layer_gradients(model, step)
    """

    def __init__(self, context: TrainingContext):
        """Initialize diagnostics manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self._config: Optional['DiagnosticsConfig'] = None
        self._timing_accumulator: Dict[str, float] = {}
        self._pending_events: Dict[str, tuple] = {}  # phase -> (start_event, end_event)
        self._current_step: int = 0

    def initialize(self) -> None:
        """Initialize the diagnostics manager."""
        self._initialized = True
        self.logger.debug("DiagnosticsManager initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._timing_accumulator.clear()
        self._pending_events.clear()

    def configure(self, config: 'DiagnosticsConfig') -> None:
        """Configure diagnostics collection.

        Args:
            config: DiagnosticsConfig with enabled flags and frequencies
        """
        self._config = config
        if config.enabled:
            enabled_features = []
            if config.enable_per_layer_gradients:
                enabled_features.append("per-layer gradients")
            if config.enable_routing_diagnostics:
                enabled_features.append("routing diagnostics")
            if config.enable_memory_breakdown:
                enabled_features.append("memory breakdown")
            if config.enable_timing_profiling:
                enabled_features.append("timing profiling")
            self.logger.info(f"Diagnostics enabled: {', '.join(enabled_features) or 'none'}")

    def set_current_step(self, step: int) -> None:
        """Set current training step for timing collection.

        Args:
            step: Current training step
        """
        self._current_step = step

    # ========================
    # Per-Layer Gradient Stats
    # ========================

    def collect_per_layer_gradients(
        self,
        model: nn.Module,
        step: int,
    ) -> List[LayerGradientStats]:
        """Collect gradient statistics per layer.

        Only collects at configured frequency to minimize overhead.

        Args:
            model: Model to collect gradient stats from
            step: Current training step

        Returns:
            List of LayerGradientStats for each matched layer
        """
        if not self._config or not self._config.enable_per_layer_gradients:
            return []

        if step % self._config.per_layer_log_freq != 0:
            return []

        stats = []
        patterns = self._config.layer_name_patterns

        for name, module in model.named_modules():
            # Filter by pattern
            if not any(p in name for p in patterns):
                continue

            # Collect gradients from parameters
            grads = []
            num_params = 0
            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    grads.append(param.grad.data.flatten())
                    num_params += param.numel()

            if grads:
                all_grads = torch.cat(grads)
                # Single GPU→CPU sync for all 5 stats instead of 5 .item() calls
                stats_tensor = torch.stack([
                    all_grads.norm(),
                    all_grads.mean(),
                    all_grads.std(),
                    all_grads.max(),
                    all_grads.min()
                ])
                norm, mean, std, max_val, min_val = stats_tensor.tolist()

                layer_stat = LayerGradientStats(
                    layer_name=name,
                    grad_norm=norm,
                    grad_mean=mean,
                    grad_std=std,
                    grad_max=max_val,
                    grad_min=min_val,
                    num_params=num_params,
                    has_grad=True,
                )
                stats.append(layer_stat)

        return stats

    # ========================
    # Expert Routing Diagnostics
    # ========================

    def collect_routing_stats(
        self,
        model: nn.Module,
        step: int,
    ) -> Optional[ExpertRoutingStats]:
        """Collect expert routing statistics.

        Only collects at configured frequency. Searches for MoE layers
        in the model and aggregates their routing metrics.

        Args:
            model: Model to collect routing stats from
            step: Current training step

        Returns:
            ExpertRoutingStats or None if not enabled/not time yet
        """
        if not self._config or not self._config.enable_routing_diagnostics:
            return None

        if step % self._config.routing_log_freq != 0:
            return None

        stats = ExpertRoutingStats(step=step)

        # Find MoE layers and aggregate their metrics
        for name, module in model.named_modules():
            # Check for common MoE layer attributes
            if hasattr(module, 'get_expert_usage_stats'):
                try:
                    usage = module.get_expert_usage_stats()
                    if 'expert_usage_normalized' in usage:
                        tensor = usage['expert_usage_normalized']
                        if isinstance(tensor, torch.Tensor):
                            for i, load in enumerate(tensor.tolist()):
                                stats.per_expert_load[i] = load
                except Exception as e:
                    self.logger.debug(f"Could not get expert stats from {name}: {e}")

            # Also check for routing entropy if available
            if hasattr(module, 'routing_entropy'):
                try:
                    stats.routing_entropy = float(module.routing_entropy)
                except Exception as e:
                    logger.debug(f"Could not get routing entropy: {e}")

        # Calculate balance score from loads
        if stats.per_expert_load:
            loads = list(stats.per_expert_load.values())
            if loads:
                avg_load = sum(loads) / len(loads)
                if avg_load > 0:
                    # Balance score: 1 / (1 + variance), normalized
                    variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
                    stats.balance_score = 1.0 / (1.0 + variance * 100)  # Scale variance
                else:
                    stats.balance_score = 0.0

        return stats

    # ========================
    # Memory Breakdown
    # ========================

    def collect_memory_breakdown(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> Optional[MemoryBreakdown]:
        """Collect detailed memory breakdown.

        Only collects at configured frequency.

        Args:
            model: Model to analyze
            optimizer: Optimizer to analyze
            step: Current training step

        Returns:
            MemoryBreakdown or None if not enabled/not time yet
        """
        if not self._config or not self._config.enable_memory_breakdown:
            return None

        if step % self._config.memory_log_freq != 0:
            return None

        if not torch.cuda.is_available():
            return None

        breakdown = MemoryBreakdown()

        # Total memory
        breakdown.total_allocated_gb = torch.cuda.memory_allocated() / 1e9
        breakdown.total_reserved_gb = torch.cuda.memory_reserved() / 1e9

        # Parameter memory
        if self._config.track_parameter_memory:
            param_bytes = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            breakdown.parameters_gb = param_bytes / 1e9

        # Gradient memory
        if self._config.track_gradient_memory:
            grad_bytes = sum(
                p.grad.numel() * p.grad.element_size()
                for p in model.parameters()
                if p.grad is not None
            )
            breakdown.gradients_gb = grad_bytes / 1e9

        # Optimizer state memory (approximate)
        if self._config.track_optimizer_state_memory:
            opt_bytes = 0
            for state in optimizer.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        opt_bytes += v.numel() * v.element_size()
            breakdown.optimizer_states_gb = opt_bytes / 1e9

        # Activations (estimated as remainder)
        if self._config.track_activation_memory:
            accounted = (
                breakdown.parameters_gb +
                breakdown.gradients_gb +
                breakdown.optimizer_states_gb
            )
            breakdown.activations_gb = max(0, breakdown.total_allocated_gb - accounted)

        breakdown.other_gb = max(0, (
            breakdown.total_reserved_gb - breakdown.total_allocated_gb
        ))

        return breakdown

    # ========================
    # Timing Profiling
    # ========================

    @contextmanager
    def time_phase(self, phase: str) -> Generator[None, None, None]:
        """Context manager for timing a training phase.

        Usage:
            with diagnostics.time_phase('forward'):
                outputs = model(inputs)

        Args:
            phase: Phase name ('data_loading', 'forward', 'backward', 'optimizer_step')

        Yields:
            None
        """
        if not self._config or not self._config.enable_timing_profiling:
            yield
            return

        # Use CUDA events for non-blocking GPU timing (avoids cudaStreamSynchronize)
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            yield
            end_event.record()
            # Store events for deferred timing extraction (non-blocking)
            self._pending_events[phase] = (start_event, end_event)
        else:
            # CPU fallback - no sync needed
            start = time.perf_counter()
            yield
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._timing_accumulator[phase] = elapsed_ms

    def get_timing_profile(self, step: int) -> Optional[TimingProfile]:
        """Get timing profile for current step.

        Only returns profile at configured frequency.
        Extracts deferred CUDA event timings (single sync instead of 8 per step).

        Args:
            step: Current training step

        Returns:
            TimingProfile or None if not enabled/not time yet
        """
        if not self._config or not self._config.enable_timing_profiling:
            return None

        if step % self._config.timing_log_freq != 0:
            return None

        # Extract timings from pending CUDA events (single sync at log time)
        if self._pending_events and torch.cuda.is_available():
            # Single synchronize to ensure all events are complete
            torch.cuda.synchronize()
            for phase, (start_event, end_event) in self._pending_events.items():
                try:
                    elapsed_ms = start_event.elapsed_time(end_event)
                    self._timing_accumulator[phase] = elapsed_ms
                except RuntimeError:
                    # Event timing failed, skip this phase
                    pass
            self._pending_events.clear()

        profile = TimingProfile(step=step)
        profile.data_loading_ms = self._timing_accumulator.get('data_loading', 0.0)
        profile.forward_ms = self._timing_accumulator.get('forward', 0.0)
        profile.backward_ms = self._timing_accumulator.get('backward', 0.0)
        profile.optimizer_step_ms = self._timing_accumulator.get('optimizer_step', 0.0)
        profile.total_step_ms = sum([
            profile.data_loading_ms,
            profile.forward_ms,
            profile.backward_ms,
            profile.optimizer_step_ms,
        ])

        # Clear timing accumulator for next collection
        self._timing_accumulator.clear()

        return profile

    def reset_timing(self) -> None:
        """Reset timing accumulator and pending events."""
        self._timing_accumulator.clear()
        self._pending_events.clear()

    # ========================
    # Status
    # ========================

    def get_status(self) -> Dict[str, Any]:
        """Return current diagnostics status."""
        return {
            'enabled': self._config.enabled if self._config else False,
            'per_layer_gradients': (
                self._config.enable_per_layer_gradients if self._config else False
            ),
            'routing_diagnostics': (
                self._config.enable_routing_diagnostics if self._config else False
            ),
            'memory_breakdown': (
                self._config.enable_memory_breakdown if self._config else False
            ),
            'timing_profiling': (
                self._config.enable_timing_profiling if self._config else False
            ),
        }
