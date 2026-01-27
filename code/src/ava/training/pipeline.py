"""
Training pipeline orchestrator for the Ava framework.

This module implements a component-based orchestration pattern that coordinates
all training components through a unified lifecycle interface.

Architecture Overview:
    TrainingPipeline (Orchestrator)
        ├── TrainingContext (Shared State Hub)
        │   └── model, optimizer, scheduler, device, config, metadata
        │
        └── Registered Components (in order):
            ├── ModelBuilder      → Model creation & optimization
            ├── OptimizerManager  → Optimizer & scheduler setup
            ├── DataLoaderManager → Data loading & preprocessing
            ├── TrainingLoopManager → Main training loop
            ├── ValidationManager → Validation & model selection
            ├── GenerationManager → Sample generation (async)
            └── MetricsManager    → WandB, CSV, logging

Lifecycle Hooks:
    - initialize(): Called once at startup (FATAL severity - must succeed)
    - cleanup(): Called once at shutdown (WARNING severity - best effort)
    - on_epoch_start/end(): Called at epoch boundaries
    - on_step_start/end(): Called at step boundaries
    - on_error(): Called when training fails (for emergency saves)

Error Handling:
    - FATAL errors stop training immediately and trigger cleanup
    - WARNING errors are logged but training continues
    - RETRY is reserved for future automatic recovery logic

Usage:
    pipeline = TrainingPipeline(context)
    pipeline.register('model', ModelBuilder(context))
    pipeline.register('optimizer', OptimizerManager(context))
    pipeline.initialize_all()  # Calls initialize() on each in order
    # ... training loop ...
    pipeline.cleanup_all()     # Calls cleanup() in reverse order
"""

import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from .context import ManagerInterface, TrainingComponent, TrainingContext

# Unified memory cache for reduced GPU sync overhead
try:
    from ava.cuda.memory_cache import get_memory_cache
    MEMORY_CACHE_AVAILABLE = True
except ImportError:
    MEMORY_CACHE_AVAILABLE = False


@dataclass
class RetryState:
    """Tracks retry attempts for a component."""
    attempt_count: int = 0
    last_backoff: float = 0.0
    total_retries: int = 0
    last_error: Optional[Exception] = None


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_backoff: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff: float = 60.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on_oom: bool = True
    reduce_batch_on_oom: bool = True

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """
    Severity levels for component errors during training.

    - FATAL: Stop training immediately (e.g., model build failure)
    - WARNING: Log error but continue (e.g., metrics logging failed)
    - RETRY: Attempt recovery (reserved for future retry logic)
    """
    FATAL = "fatal"
    WARNING = "warning"
    RETRY = "retry"


class ComponentError:
    """
    Wrapper for component errors with severity and context.

    Used internally by TrainingPipeline to wrap exceptions from components
    with metadata about which component failed, how severe the error is,
    and any contextual information (e.g., epoch number if it failed during
    on_epoch_start).
    """

    def __init__(
        self,
        component_name: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.component_name = component_name  # e.g., 'model', 'optimizer', 'metrics'
        self.error = error                    # The underlying exception
        self.severity = severity              # FATAL/WARNING/RETRY
        self.context = context or {}          # Additional context (epoch, step, etc.)

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.component_name}: {self.error}"


# Default severity mapping for hooks
# These define how strictly to enforce errors in each lifecycle stage.
# FATAL errors stop training immediately, WARNING errors are logged but continue.
DEFAULT_HOOK_SEVERITIES: Dict[str, ErrorSeverity] = {
    'initialize': ErrorSeverity.FATAL,     # Must succeed before training starts
    'cleanup': ErrorSeverity.WARNING,       # Best-effort cleanup, continue anyway
    'on_epoch_start': ErrorSeverity.WARNING,  # Log but continue to next epoch
    'on_epoch_end': ErrorSeverity.WARNING,    # Log but continue to next epoch
    'on_step_start': ErrorSeverity.WARNING,   # Log but continue to next step
    'on_step_end': ErrorSeverity.WARNING,     # Log but continue to next step
    'on_error': ErrorSeverity.WARNING,        # Don't fail during error handling
}


class TrainingPipeline:
    """
    Central manager for all training components and their lifecycle.

    The TrainingPipeline orchestrates training by:

    1. **Component Registration**: Components are registered by name and can
       be retrieved via pipeline.get('name') or pipeline['name'].

    2. **Lifecycle Hooks**: Calls lifecycle methods on all registered components
       - initialize(): Called once before training (FATAL if fails)
       - cleanup(): Called once after training (WARNING if fails)
       - on_epoch_start(epoch): Called at epoch start (WARNING if fails)
       - on_epoch_end(epoch): Called at epoch end (WARNING if fails)
       - on_step_start(step): Called at step start (WARNING if fails)
       - on_step_end(step, loss): Called at step end (WARNING if fails)
       - on_error(error): Called when training fails (WARNING if fails)

    3. **Error Handling**: Wraps component errors with severity levels
       and context. FATAL errors stop training, WARNING errors are logged.

    4. **State Management**: Tracks context, registered components, and
       error history for debugging.

    Example:
        >>> context = TrainingContext(model=None, device=device)
        >>> pipeline = TrainingPipeline(context)
        >>> # Register components
        >>> pipeline.register('model', ModelBuilder(context))
        >>> pipeline.register('optimizer', OptimizerManager(context))
        >>> pipeline.register('data', DataLoaderManager(context))
        >>> pipeline.initialize_all()
        >>> # Training loop
        >>> for epoch in range(num_epochs):
        ...     pipeline.on_epoch_start(epoch)
        ...     train_loss = pipeline.get('training').train_epoch(...)
        ...     pipeline.on_epoch_end(epoch)
        >>> pipeline.cleanup_all()
    """

    def __init__(self, context: TrainingContext, retry_config: Optional[RetryConfig] = None):
        """
        Initialize the training pipeline.

        Args:
            context: Training context with shared state (model, device, config, etc.)
            retry_config: Optional retry configuration for error recovery

        Attributes:
            _components: OrderedDict of registered components (order preserved)
            _initialized: Flag indicating if all components have been initialized
            _errors: List of ComponentError objects for debugging
            _hook_severities: Maps hook names to ErrorSeverity levels
            _retry_config: Configuration for retry logic
            _retry_states: Per-component retry state tracking
        """
        self.context = context
        self._components: OrderedDict[str, TrainingComponent] = OrderedDict()
        self._initialized = False
        self._errors: List[ComponentError] = []
        self._hook_severities: Dict[str, ErrorSeverity] = DEFAULT_HOOK_SEVERITIES.copy()
        self._retry_config = retry_config or RetryConfig()
        self._retry_states: Dict[str, RetryState] = {}
        # Status caching to avoid redundant get_status() calls (reduces sync overhead)
        self._status_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._status_cache_time: float = 0.0
        self._status_cache_ttl: float = 5.0  # 5 second TTL (reduced overhead)

    def set_hook_severity(self, hook_name: str, severity: ErrorSeverity) -> None:
        """
        Set the error severity for a specific hook.

        Args:
            hook_name: Name of the hook (e.g., 'on_epoch_start')
            severity: ErrorSeverity level
        """
        self._hook_severities[hook_name] = severity

    def get_errors(self) -> List[ComponentError]:
        """Get all recorded errors."""
        return self._errors.copy()

    def clear_errors(self) -> None:
        """Clear recorded errors."""
        self._errors.clear()

    def _compute_backoff(self, component_name: str) -> float:
        """
        Compute exponential backoff delay with optional jitter.

        Args:
            component_name: Name of the component for retry state lookup

        Returns:
            Backoff delay in seconds
        """
        state = self._retry_states.get(component_name, RetryState())
        config = self._retry_config

        # Exponential backoff: initial * (multiplier ^ attempt)
        backoff = config.initial_backoff * (config.backoff_multiplier ** state.attempt_count)
        backoff = min(backoff, config.max_backoff)

        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = backoff * config.jitter_factor
            backoff += random.uniform(-jitter_range, jitter_range)
            backoff = max(0.1, backoff)  # Ensure minimum delay

        return backoff

    def _is_oom_error(self, error: Exception) -> bool:
        """Check if error is a CUDA out-of-memory error."""
        error_str = str(error).lower()
        return (
            'out of memory' in error_str or
            'cuda out of memory' in error_str or
            'cudaerroroutofmemory' in error_str or
            isinstance(error, RuntimeError) and 'CUDA' in str(error)
        )

    def _should_retry(self, component_name: str, error: Exception) -> bool:
        """
        Determine if we should retry based on error type and retry count.

        Args:
            component_name: Name of the component
            error: The exception that occurred

        Returns:
            True if we should retry, False otherwise
        """
        state = self._retry_states.get(component_name, RetryState())
        config = self._retry_config

        # Check if we've exceeded max retries
        if state.attempt_count >= config.max_retries:
            return False

        # Check for OOM errors
        if self._is_oom_error(error):
            return config.retry_on_oom

        return True

    def _handle_hook_error(
        self,
        hook_name: str,
        component_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle an error from a component hook.

        Args:
            hook_name: Name of the hook that failed
            component_name: Name of the component
            error: The exception that occurred
            context: Additional context

        Returns:
            True if the error was handled and should retry, False otherwise

        Raises:
            RuntimeError: If severity is FATAL and cannot recover
        """
        severity = self._hook_severities.get(hook_name, ErrorSeverity.WARNING)
        comp_error = ComponentError(component_name, error, severity, context)
        self._errors.append(comp_error)

        # Initialize retry state for component if not exists
        if component_name not in self._retry_states:
            self._retry_states[component_name] = RetryState()
        state = self._retry_states[component_name]
        state.last_error = error

        # Handle error based on severity level:
        # - FATAL: Critical failure, cannot continue (e.g., model build failed)
        # - WARNING: Non-critical, log and continue (e.g., metrics logging failed)
        # - RETRY: Automatic retry with exponential backoff
        if severity == ErrorSeverity.FATAL:
            logger.error(f"FATAL error in {hook_name} for '{component_name}': {error}")
            raise RuntimeError(f"Fatal pipeline error: {comp_error}")

        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"{hook_name} failed for '{component_name}': {error}")
            return False

        elif severity == ErrorSeverity.RETRY:
            # Check if we should retry
            if not self._should_retry(component_name, error):
                logger.error(
                    f"Max retries ({self._retry_config.max_retries}) exceeded for "
                    f"'{component_name}' in {hook_name}. Last error: {error}"
                )
                # Reset retry state and escalate to WARNING
                state.attempt_count = 0
                return False

            # Compute and apply backoff
            backoff = self._compute_backoff(component_name)
            state.attempt_count += 1
            state.total_retries += 1
            state.last_backoff = backoff

            logger.info(
                f"{hook_name} failed for '{component_name}' "
                f"(attempt {state.attempt_count}/{self._retry_config.max_retries}), "
                f"retrying in {backoff:.2f}s: {error}"
            )

            # Handle OOM-specific recovery
            if self._is_oom_error(error) and self._retry_config.reduce_batch_on_oom:
                self._handle_oom_recovery(component_name)

            return True

        return False

    def _handle_oom_recovery(self, component_name: str) -> None:
        """
        Handle OOM recovery by reducing batch size or clearing cache.

        Args:
            component_name: Name of the component that failed
        """
        import torch

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleared CUDA cache for OOM recovery in '{component_name}'")

        # Try to reduce batch size in context if available
        if hasattr(self.context, 'batch_size') and self.context.batch_size:
            old_batch = self.context.batch_size
            new_batch = max(1, old_batch // 2)
            self.context.batch_size = new_batch
            logger.info(f"Reduced batch size from {old_batch} to {new_batch} for OOM recovery")

    def reset_retry_state(self, component_name: Optional[str] = None) -> None:
        """
        Reset retry state for a component or all components.

        Args:
            component_name: Specific component to reset, or None for all
        """
        if component_name:
            if component_name in self._retry_states:
                self._retry_states[component_name] = RetryState()
        else:
            self._retry_states.clear()

    def get_retry_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get retry statistics for all components.

        Returns:
            Dictionary mapping component names to their retry stats
        """
        return {
            name: {
                'attempt_count': state.attempt_count,
                'total_retries': state.total_retries,
                'last_backoff': state.last_backoff,
                'last_error': str(state.last_error) if state.last_error else None,
            }
            for name, state in self._retry_states.items()
        }

    def register(self, name: str, component: TrainingComponent) -> 'TrainingPipeline':
        """
        Register a component with the pipeline.

        Components are registered in order and initialize_all() calls them
        in registration order. Common components:
        - 'model': ModelBuilder
        - 'optimizer': OptimizerManager
        - 'data': DataLoaderManager
        - 'training': TrainingLoopManager
        - 'validation': ValidationManager
        - 'generation': GenerationManager
        - 'metrics': MetricsManager

        Args:
            name: Unique name for the component (string key)
            component: Must be instance of TrainingComponent

        Returns:
            Self for method chaining (allows: pipeline.register(...).register(...))

        Raises:
            ValueError: If component name already registered
            TypeError: If component is not a TrainingComponent
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered")

        # Duck typing: check for required interface instead of strict isinstance
        # This allows MetricsManager and other compatible classes to work
        required_methods = ['initialize', 'cleanup']
        missing = [m for m in required_methods if not hasattr(component, m)]
        if missing and not isinstance(component, TrainingComponent):
            raise TypeError(
                f"Component must be a TrainingComponent or have {required_methods}, "
                f"got {type(component).__name__} missing {missing}"
            )

        self._components[name] = component
        logger.debug(f"Registered component: {name} ({type(component).__name__})")

        return self

    def get(self, name: str) -> TrainingComponent:
        """
        Get a registered component by name.

        Args:
            name: Component name

        Returns:
            The registered component

        Raises:
            KeyError: If component not found
        """
        if name not in self._components:
            raise KeyError(
                f"Component '{name}' not found. "
                f"Registered: {list(self._components.keys())}"
            )
        return self._components[name]

    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def __getitem__(self, name: str) -> TrainingComponent:
        """Allow dict-like access: pipeline['model']."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Allow 'in' operator: 'model' in pipeline."""
        return self.has(name)

    def __iter__(self) -> Iterator[Tuple[str, TrainingComponent]]:
        """Iterate over (name, component) pairs."""
        return iter(self._components.items())

    def initialize_all(self) -> None:
        """
        Initialize all registered components in registration order.

        This must be called before training begins. If any component fails
        with FATAL severity, stops immediately and cleans up already-
        initialized components.

        Order matters: usually model → optimizer → data → training → metrics

        Raises:
            RuntimeError: If initialization fails (FATAL error from component)
        """
        logger.info("Initializing pipeline components...")

        # OPTIMIZATION: Initialize persistent pinned buffer pool if not already set
        # This provides 2-5% speedup on epochs 2+ by avoiding reallocation
        if self.context.pinned_buffer_pool is None and torch.cuda.is_available():
            try:
                from ..cuda.streams import PersistentPinnedBufferPool, _auto_size_pinned_buffer_pool

                pool_size_mb = _auto_size_pinned_buffer_pool()
                self.context.pinned_buffer_pool = PersistentPinnedBufferPool(
                    max_buffers_per_shape=4,
                    max_total_memory_mb=pool_size_mb,
                    trim_unused_epochs=2,
                )

                # Warmup with common shapes if batch_size and max_length known
                batch_size = self.context.metadata.get('batch_size', 32)
                max_length = self.context.metadata.get('max_length', 512)
                if batch_size and max_length:
                    common_shapes = [
                        (torch.long, (batch_size, max_length)),  # input_ids
                        (torch.long, (batch_size, max_length)),  # attention_mask
                        (torch.long, (batch_size, max_length)),  # labels
                    ]
                    self.context.pinned_buffer_pool.warmup(common_shapes)

                logger.info(f"Initialized persistent pinned buffer pool ({pool_size_mb:.0f}MB)")
            except Exception as e:
                logger.debug(f"Persistent buffer pool initialization skipped: {e}")

        for name, component in self._components.items():
            try:
                logger.debug(f"Initializing: {name}")
                component.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize '{name}': {e}")
                # Cleanup already initialized components
                self._cleanup_initialized()
                raise RuntimeError(f"Pipeline initialization failed at '{name}': {e}")

        self._initialized = True
        logger.info(f"Pipeline initialized ({len(self._components)} components)")

    def cleanup_all(self) -> None:
        """
        Cleanup all components in reverse registration order.

        Called after training completes (or fails). Cleans up in reverse
        order to properly deallocate resources (e.g., close dataloaders
        before destroying model). Continues cleanup even if individual
        components fail so all get a chance to cleanup.

        Cleanup errors are logged as WARNING but don't stop the process.
        """
        logger.info("Cleaning up pipeline components...")

        # Cleanup in reverse order
        for name, component in reversed(list(self._components.items())):
            try:
                logger.debug(f"Cleaning up: {name}")
                component.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for '{name}': {e}")
                # Continue cleanup for other components

        self._initialized = False
        logger.info("Pipeline cleanup complete")

    def _cleanup_initialized(self) -> None:
        """Cleanup only initialized components (for error recovery)."""
        for name, component in reversed(list(self._components.items())):
            if component.is_initialized():
                try:
                    component.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleanup failed for '{name}': {e}")

    def on_epoch_start(self, epoch: int) -> None:
        """
        Notify all components of epoch start.

        Called at the beginning of each training epoch. Components can use
        this hook to reset epoch-level state (e.g., metrics). Errors are
        logged as WARNING but don't stop training.

        Args:
            epoch: Starting epoch number (0-indexed)
        """
        self.context.epoch = epoch

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_epoch_start(epoch)
                except Exception as e:
                    self._handle_hook_error(
                        'on_epoch_start', name, e, {'epoch': epoch}
                    )

    def on_epoch_end(self, epoch: int) -> None:
        """
        Notify all components of epoch end.

        Called at the end of each training epoch. Components can use this
        hook to finalize epoch metrics, save checkpoints, or log progress.
        Errors are logged as WARNING but don't stop training.

        Args:
            epoch: Ending epoch number (0-indexed)
        """
        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_epoch_end(epoch)
                except Exception as e:
                    self._handle_hook_error(
                        'on_epoch_end', name, e, {'epoch': epoch}
                    )

        # OPTIMIZATION: Trim unused buffers from persistent pinned buffer pool
        # This frees memory from buffer shapes no longer used (e.g., batch size changed)
        if self.context.pinned_buffer_pool is not None:
            try:
                self.context.pinned_buffer_pool.on_epoch_end()
            except Exception as e:
                logger.debug(f"Buffer pool epoch cleanup: {e}")

        # PERF FIX: Conditionally clear CUDA cache only when fragmentation is high
        # empty_cache() takes 50-500ms, so only call when necessary
        # OPTIMIZATION: Use unified memory cache to avoid additional GPU sync
        if torch.cuda.is_available():
            if MEMORY_CACHE_AVAILABLE:
                # Use unified cache (no additional GPU sync)
                stats = get_memory_cache().get_stats()
                allocated_gb = stats.get('allocated_gb', 0)
                reserved_gb = stats.get('reserved_gb', 0)
                if reserved_gb > 0:
                    fragmentation = 1.0 - (allocated_gb / reserved_gb)
                    if fragmentation > 0.3:  # Only clear if >30% fragmented
                        torch.cuda.empty_cache()
                        # Invalidate cache since we cleared CUDA cache
                        get_memory_cache().invalidate()
                        logger.debug(f"Cleared CUDA cache (fragmentation: {fragmentation:.1%})")
            else:
                # Fallback: direct query
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if reserved > 0:
                    fragmentation = 1.0 - (allocated / reserved)
                    if fragmentation > 0.3:  # Only clear if >30% fragmented
                        torch.cuda.empty_cache()
                        logger.debug(f"Cleared CUDA cache (fragmentation: {fragmentation:.1%})")

    def on_step_start(self, step: int) -> None:
        """
        Notify all components of step start.

        Args:
            step: Starting step number
        """
        self.context.step = step

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_step_start(step)
                except Exception as e:
                    self._handle_hook_error(
                        'on_step_start', name, e, {'step': step}
                    )

    def on_step_end(self, step: int, loss: float) -> None:
        """
        Notify all components of step end.

        Args:
            step: Ending step number
            loss: Loss value for this step
        """
        self.context.current_loss = loss

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_step_end(step, loss)
                except Exception as e:
                    self._handle_hook_error(
                        'on_step_end', name, e, {'step': step, 'loss': loss}
                    )

    def on_error(self, error: Exception) -> None:
        """
        Notify all components of a training error.

        Called when training fails. Components can use this hook to save
        emergency checkpoints, log final metrics, or attempt recovery.
        Errors from on_error handlers are logged as WARNING and don't
        prevent other components from handling the error.

        Args:
            error: The exception that caused training to fail
        """
        logger.error(f"Pipeline error: {error}")

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_error(error)
                except Exception as e:
                    self._handle_hook_error(
                        'on_error', name, e, {'original_error': str(error)}
                    )

    def get_status(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get status from all components (with caching to reduce overhead).

        Aggregates status dictionaries from all registered components.
        Useful for debugging, monitoring dashboards, or health checks.

        Args:
            force_refresh: If True, bypass cache and refresh status.

        Returns:
            Dictionary mapping component names to their status dictionaries.
            Each component's status may include: metrics, counters, flags, etc.
            If a component's get_status() fails, returns {'error': str(e)}.
        """
        # OPTIMIZATION: Cache status to avoid repeated calls to component.get_status()
        # Each component's get_status() may involve GPU sync operations
        current_time = time.time()
        if not force_refresh and self._status_cache is not None:
            if (current_time - self._status_cache_time) < self._status_cache_ttl:
                return self._status_cache

        # Aggregate status from all components for unified monitoring
        # Each ManagerInterface component provides its own status dict
        # TrainingComponent (base) only reports initialization state
        status = {}

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    status[name] = component.get_status()
                except Exception as e:
                    status[name] = {'error': str(e)}
            else:
                status[name] = {'initialized': component.is_initialized()}

        # Update cache
        self._status_cache = status
        self._status_cache_time = current_time

        return status

    def is_initialized(self) -> bool:
        """Check if pipeline has been initialized."""
        return self._initialized

    @property
    def component_names(self) -> list:
        """Get list of registered component names."""
        return list(self._components.keys())

    def __len__(self) -> int:
        """Get number of registered components."""
        return len(self._components)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrainingPipeline("
            f"components={list(self._components.keys())}, "
            f"initialized={self._initialized})"
        )
