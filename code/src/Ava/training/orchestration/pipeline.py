"""
Training pipeline orchestrator for the Ava framework.

Coordinates all training components through a unified interface.
"""

import logging
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..train.base import ManagerInterface, TrainingComponent, TrainingContext

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for component errors."""
    FATAL = "fatal"       # Stop training immediately
    WARNING = "warning"   # Log and continue
    RETRY = "retry"       # Attempt recovery


class ComponentError:
    """Wrapper for component errors with severity and context."""

    def __init__(
        self,
        component_name: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.component_name = component_name
        self.error = error
        self.severity = severity
        self.context = context or {}

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.component_name}: {self.error}"


# Default severity mapping for hooks
DEFAULT_HOOK_SEVERITIES: Dict[str, ErrorSeverity] = {
    'initialize': ErrorSeverity.FATAL,
    'cleanup': ErrorSeverity.WARNING,
    'on_epoch_start': ErrorSeverity.WARNING,
    'on_epoch_end': ErrorSeverity.WARNING,
    'on_step_start': ErrorSeverity.WARNING,
    'on_step_end': ErrorSeverity.WARNING,
    'on_error': ErrorSeverity.WARNING,
}


class TrainingPipeline:
    """
    Manages component lifecycle and coordination for training.

    The TrainingPipeline:
        - Registers and manages all training components
        - Calls lifecycle hooks (initialize, cleanup, on_epoch_*, on_step_*, on_error)
        - Provides unified access to components
        - Ensures proper cleanup on error

    Example:
        >>> context = TrainingContext(model=None, device=device)
        >>> pipeline = TrainingPipeline(context)
        >>> pipeline.register('model', ModelBuilder(context))
        >>> pipeline.register('optimizer', OptimizerManager(context))
        >>> pipeline.register('data', DataLoaderManager(context))
        >>> pipeline.register('training', TrainingLoopManager(context))
        >>> pipeline.register('validation', ValidationManager(context))
        >>> pipeline.register('metrics', MetricsManager(context))
        >>>
        >>> pipeline.initialize_all()
        >>>
        >>> for epoch in range(num_epochs):
        ...     pipeline.on_epoch_start(epoch)
        ...     train_loss = pipeline.get('training').train_epoch(...)
        ...     pipeline.on_epoch_end(epoch)
        >>>
        >>> pipeline.cleanup_all()
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the training pipeline.

        Args:
            context: Training context with shared state
        """
        self.context = context
        self._components: OrderedDict[str, TrainingComponent] = OrderedDict()
        self._initialized = False
        self._errors: List[ComponentError] = []
        self._hook_severities: Dict[str, ErrorSeverity] = DEFAULT_HOOK_SEVERITIES.copy()

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

    def _handle_hook_error(
        self,
        hook_name: str,
        component_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Handle an error from a component hook.

        Args:
            hook_name: Name of the hook that failed
            component_name: Name of the component
            error: The exception that occurred
            context: Additional context

        Raises:
            RuntimeError: If severity is FATAL
        """
        severity = self._hook_severities.get(hook_name, ErrorSeverity.WARNING)
        comp_error = ComponentError(component_name, error, severity, context)
        self._errors.append(comp_error)

        if severity == ErrorSeverity.FATAL:
            logger.error(f"FATAL error in {hook_name} for '{component_name}': {error}")
            raise RuntimeError(f"Fatal pipeline error: {comp_error}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"{hook_name} failed for '{component_name}': {error}")
        elif severity == ErrorSeverity.RETRY:
            logger.info(f"{hook_name} failed for '{component_name}', will retry: {error}")

    def register(self, name: str, component: TrainingComponent) -> 'TrainingPipeline':
        """
        Register a component with the pipeline.

        Args:
            name: Unique name for the component
            component: TrainingComponent instance

        Returns:
            Self for method chaining

        Raises:
            ValueError: If component name already registered
            TypeError: If component is not a TrainingComponent
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered")

        if not isinstance(component, TrainingComponent):
            raise TypeError(
                f"Component must be a TrainingComponent, got {type(component).__name__}"
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

        Raises:
            RuntimeError: If initialization fails
        """
        logger.info("Initializing pipeline components...")

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

        Continues cleanup even if individual components fail.
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

        Args:
            epoch: Starting epoch number
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

        Args:
            epoch: Ending epoch number
        """
        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    component.on_epoch_end(epoch)
                except Exception as e:
                    self._handle_hook_error(
                        'on_epoch_end', name, e, {'epoch': epoch}
                    )

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
        Notify all components of an error.

        Args:
            error: The exception that occurred
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

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status from all components.

        Returns:
            Dictionary mapping component names to their status dictionaries
        """
        status = {}

        for name, component in self._components.items():
            if isinstance(component, ManagerInterface):
                try:
                    status[name] = component.get_status()
                except Exception as e:
                    status[name] = {'error': str(e)}
            else:
                status[name] = {'initialized': component.is_initialized()}

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
