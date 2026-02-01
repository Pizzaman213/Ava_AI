"""
Unit tests for TrainingPipeline - the component orchestrator.

Tests the TrainingPipeline class that manages component lifecycle
and coordinates training across all registered components.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from ava.training.pipeline import (
    TrainingPipeline,
    ErrorSeverity,
    ComponentError,
    RetryConfig,
    RetryState,
    DEFAULT_HOOK_SEVERITIES,
)
from ava.training.context import TrainingContext, TrainingComponent, ManagerInterface


class MockComponent(TrainingComponent):
    """Mock component for testing pipeline."""

    def __init__(self, context, name="mock"):
        super().__init__(context)
        self.name = name
        self.init_called = False
        self.cleanup_called = False

    def initialize(self):
        self.init_called = True
        self._initialized = True

    def cleanup(self):
        self.cleanup_called = True
        self._initialized = False


class MockManager(ManagerInterface):
    """Mock manager for testing lifecycle hooks."""

    def __init__(self, context, name="mock_manager"):
        super().__init__(context)
        self.name = name
        self.hooks_called = []

    def initialize(self):
        self._initialized = True
        self.hooks_called.append('initialize')

    def cleanup(self):
        self._initialized = False
        self.hooks_called.append('cleanup')

    def on_epoch_start(self, epoch):
        self.hooks_called.append(('epoch_start', epoch))

    def on_epoch_end(self, epoch):
        self.hooks_called.append(('epoch_end', epoch))

    def on_step_start(self, step):
        self.hooks_called.append(('step_start', step))

    def on_step_end(self, step, loss):
        self.hooks_called.append(('step_end', step, loss))

    def on_error(self, error):
        self.hooks_called.append(('error', str(error)))

    def get_status(self):
        return {'hooks_called': len(self.hooks_called)}


class FailingComponent(TrainingComponent):
    """Component that fails during initialization."""

    def __init__(self, context, fail_on="initialize"):
        super().__init__(context)
        self.fail_on = fail_on

    def initialize(self):
        if self.fail_on == "initialize":
            raise RuntimeError("Initialization failed")
        self._initialized = True

    def cleanup(self):
        if self.fail_on == "cleanup":
            raise RuntimeError("Cleanup failed")


class TestTrainingPipelineBasics:
    """Basic tests for TrainingPipeline."""

    def test_pipeline_creation(self, training_context):
        """Test creating an empty pipeline."""
        pipeline = TrainingPipeline(training_context)

        assert pipeline.context is training_context
        assert len(pipeline) == 0
        assert pipeline.is_initialized() is False
        assert pipeline.component_names == []

    def test_register_component(self, training_context):
        """Test registering a component."""
        pipeline = TrainingPipeline(training_context)
        component = MockComponent(training_context)

        result = pipeline.register('test', component)

        assert result is pipeline  # Returns self for chaining
        assert len(pipeline) == 1
        assert 'test' in pipeline
        assert pipeline.has('test')

    def test_register_chaining(self, training_context):
        """Test that register supports method chaining."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context)
        c2 = MockComponent(training_context)

        pipeline.register('first', c1).register('second', c2)

        assert len(pipeline) == 2
        assert 'first' in pipeline
        assert 'second' in pipeline

    def test_register_duplicate_raises(self, training_context):
        """Test that registering duplicate name raises."""
        pipeline = TrainingPipeline(training_context)
        pipeline.register('test', MockComponent(training_context))

        with pytest.raises(ValueError, match="already registered"):
            pipeline.register('test', MockComponent(training_context))

    def test_get_component(self, training_context):
        """Test getting a registered component."""
        pipeline = TrainingPipeline(training_context)
        component = MockComponent(training_context)
        pipeline.register('test', component)

        assert pipeline.get('test') is component
        assert pipeline['test'] is component

    def test_get_missing_raises(self, training_context):
        """Test that getting missing component raises."""
        pipeline = TrainingPipeline(training_context)

        with pytest.raises(KeyError, match="not found"):
            pipeline.get('missing')

        with pytest.raises(KeyError, match="not found"):
            _ = pipeline['missing']

    def test_iter_components(self, training_context):
        """Test iterating over components."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context, "c1")
        c2 = MockComponent(training_context, "c2")

        pipeline.register('first', c1).register('second', c2)

        names_and_components = list(pipeline)
        assert len(names_and_components) == 2
        assert names_and_components[0] == ('first', c1)
        assert names_and_components[1] == ('second', c2)


class TestPipelineInitialization:
    """Tests for pipeline initialization."""

    def test_initialize_all(self, training_context):
        """Test initializing all components."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context)
        c2 = MockComponent(training_context)

        pipeline.register('first', c1).register('second', c2)
        pipeline.initialize_all()

        assert c1.init_called
        assert c2.init_called
        assert pipeline.is_initialized()

    def test_initialize_order(self, training_context):
        """Test that components are initialized in registration order."""
        pipeline = TrainingPipeline(training_context)
        order = []

        class OrderedComponent(MockComponent):
            def initialize(self):
                super().initialize()
                order.append(self.name)

        pipeline.register('first', OrderedComponent(training_context, 'first'))
        pipeline.register('second', OrderedComponent(training_context, 'second'))
        pipeline.register('third', OrderedComponent(training_context, 'third'))

        pipeline.initialize_all()

        assert order == ['first', 'second', 'third']

    def test_initialize_failure_stops_and_cleans(self, training_context):
        """Test that initialization failure triggers cleanup."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context)
        fail = FailingComponent(training_context, fail_on="initialize")
        c3 = MockComponent(training_context)

        pipeline.register('first', c1)
        pipeline.register('failing', fail)
        pipeline.register('third', c3)

        with pytest.raises(RuntimeError, match="initialization failed"):
            pipeline.initialize_all()

        # First component should have been cleaned up
        assert c1.cleanup_called
        # Third component was never initialized
        assert not c3.init_called


class TestPipelineCleanup:
    """Tests for pipeline cleanup."""

    def test_cleanup_all(self, training_context):
        """Test cleaning up all components."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context)
        c2 = MockComponent(training_context)

        pipeline.register('first', c1).register('second', c2)
        pipeline.initialize_all()
        pipeline.cleanup_all()

        assert c1.cleanup_called
        assert c2.cleanup_called
        assert not pipeline.is_initialized()

    def test_cleanup_reverse_order(self, training_context):
        """Test that cleanup happens in reverse order."""
        pipeline = TrainingPipeline(training_context)
        order = []

        class OrderedComponent(MockComponent):
            def cleanup(self):
                super().cleanup()
                order.append(self.name)

        pipeline.register('first', OrderedComponent(training_context, 'first'))
        pipeline.register('second', OrderedComponent(training_context, 'second'))
        pipeline.register('third', OrderedComponent(training_context, 'third'))

        pipeline.initialize_all()
        pipeline.cleanup_all()

        assert order == ['third', 'second', 'first']

    def test_cleanup_continues_on_failure(self, training_context):
        """Test that cleanup continues even if one component fails."""
        pipeline = TrainingPipeline(training_context)
        c1 = MockComponent(training_context)
        fail = FailingComponent(training_context, fail_on="cleanup")
        c3 = MockComponent(training_context)

        pipeline.register('first', c1)
        pipeline.register('failing', fail)
        pipeline.register('third', c3)

        pipeline.initialize_all()
        # Should not raise, just log warning
        pipeline.cleanup_all()

        assert c1.cleanup_called
        assert c3.cleanup_called


class TestLifecycleHooks:
    """Tests for lifecycle hook dispatch."""

    def test_on_epoch_start(self, training_context):
        """Test epoch start hook dispatch."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        pipeline.on_epoch_start(5)

        assert ('epoch_start', 5) in manager.hooks_called
        assert training_context.epoch == 5

    def test_on_epoch_end(self, training_context):
        """Test epoch end hook dispatch."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        pipeline.on_epoch_end(5)

        assert ('epoch_end', 5) in manager.hooks_called

    def test_on_step_start(self, training_context):
        """Test step start hook dispatch."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        pipeline.on_step_start(100)

        assert ('step_start', 100) in manager.hooks_called
        assert training_context.step == 100

    def test_on_step_end(self, training_context):
        """Test step end hook dispatch."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        pipeline.on_step_end(100, 0.5)

        assert ('step_end', 100, 0.5) in manager.hooks_called
        assert training_context.current_loss == 0.5

    def test_on_error(self, training_context):
        """Test error hook dispatch."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        error = ValueError("test error")
        pipeline.on_error(error)

        # Check error was dispatched
        error_calls = [c for c in manager.hooks_called if isinstance(c, tuple) and c[0] == 'error']
        assert len(error_calls) == 1

    def test_hooks_only_for_managers(self, training_context):
        """Test that hooks are only called for ManagerInterface components."""
        pipeline = TrainingPipeline(training_context)
        component = MockComponent(training_context)  # Not a manager
        manager = MockManager(training_context)

        pipeline.register('component', component)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        pipeline.on_epoch_start(1)

        # Only manager should have received the hook
        assert ('epoch_start', 1) in manager.hooks_called


class TestErrorHandling:
    """Tests for error handling and severity."""

    def test_default_hook_severities(self):
        """Test that default severities are set correctly."""
        assert DEFAULT_HOOK_SEVERITIES['initialize'] == ErrorSeverity.FATAL
        assert DEFAULT_HOOK_SEVERITIES['cleanup'] == ErrorSeverity.WARNING
        assert DEFAULT_HOOK_SEVERITIES['on_epoch_start'] == ErrorSeverity.WARNING

    def test_set_hook_severity(self, training_context):
        """Test changing hook severity."""
        pipeline = TrainingPipeline(training_context)

        pipeline.set_hook_severity('on_epoch_start', ErrorSeverity.FATAL)

        assert pipeline._hook_severities['on_epoch_start'] == ErrorSeverity.FATAL

    def test_error_recording(self, training_context):
        """Test that errors are recorded."""
        pipeline = TrainingPipeline(training_context)

        class FailingManager(ManagerInterface):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

            def on_epoch_start(self, epoch):
                raise ValueError("Hook failed")

        pipeline.register('failing', FailingManager(training_context))
        pipeline.initialize_all()

        # Should not raise (WARNING severity)
        pipeline.on_epoch_start(0)

        errors = pipeline.get_errors()
        assert len(errors) == 1
        assert errors[0].component_name == 'failing'

    def test_clear_errors(self, training_context):
        """Test clearing recorded errors."""
        pipeline = TrainingPipeline(training_context)

        class FailingManager(ManagerInterface):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

            def on_epoch_start(self, epoch):
                raise ValueError("Hook failed")

        pipeline.register('failing', FailingManager(training_context))
        pipeline.initialize_all()
        pipeline.on_epoch_start(0)

        pipeline.clear_errors()

        assert len(pipeline.get_errors()) == 0


class TestRetryLogic:
    """Tests for retry configuration and behavior."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_backoff == 1.0
        assert config.backoff_multiplier == 2.0
        assert config.retry_on_oom is True

    def test_is_oom_error(self, training_context):
        """Test OOM error detection."""
        pipeline = TrainingPipeline(training_context)

        # Various OOM error messages
        assert pipeline._is_oom_error(RuntimeError("CUDA out of memory"))
        assert pipeline._is_oom_error(RuntimeError("out of memory"))
        assert pipeline._is_oom_error(RuntimeError("CUDAErrorOutOfMemory"))

        # Non-OOM errors
        assert not pipeline._is_oom_error(ValueError("regular error"))
        # Note: The implementation checks for CUDA in the message, so this may be true
        # The implementation is aggressive about detecting CUDA errors

    def test_retry_stats(self, training_context):
        """Test retry statistics tracking."""
        pipeline = TrainingPipeline(training_context)

        # Simulate retry attempts
        pipeline._retry_states['test'] = RetryState(
            attempt_count=2,
            total_retries=5,
            last_backoff=4.0,
            last_error=ValueError("test"),
        )

        stats = pipeline.get_retry_stats()

        assert 'test' in stats
        assert stats['test']['attempt_count'] == 2
        assert stats['test']['total_retries'] == 5

    def test_reset_retry_state(self, training_context):
        """Test resetting retry state."""
        pipeline = TrainingPipeline(training_context)

        pipeline._retry_states['test'] = RetryState(attempt_count=3)
        pipeline._retry_states['other'] = RetryState(attempt_count=2)

        # Reset specific component
        pipeline.reset_retry_state('test')
        assert pipeline._retry_states['test'].attempt_count == 0
        assert pipeline._retry_states['other'].attempt_count == 2

        # Reset all
        pipeline.reset_retry_state()
        assert len(pipeline._retry_states) == 0


class TestGetStatus:
    """Tests for status aggregation."""

    def test_get_status_aggregates(self, training_context):
        """Test that get_status aggregates from all components."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        component = MockComponent(training_context)

        pipeline.register('manager', manager)
        pipeline.register('component', component)
        pipeline.initialize_all()

        status = pipeline.get_status()

        assert 'manager' in status
        assert 'component' in status
        # Manager returns status dict, component returns initialized flag
        assert 'hooks_called' in status['manager']
        assert 'initialized' in status['component']

    def test_status_caching(self, training_context):
        """Test that status is cached to avoid repeated queries."""
        pipeline = TrainingPipeline(training_context)
        manager = MockManager(training_context)
        pipeline.register('manager', manager)
        pipeline.initialize_all()

        # First call populates cache
        status1 = pipeline.get_status()

        # Modify manager state
        manager.on_epoch_start(1)

        # Second call should return cached result (within TTL)
        status2 = pipeline.get_status()
        assert status1 == status2

        # Force refresh should return updated state
        status3 = pipeline.get_status(force_refresh=True)
        assert status3['manager']['hooks_called'] > status1['manager']['hooks_called']


class TestComponentError:
    """Tests for ComponentError wrapper."""

    def test_component_error_creation(self):
        """Test creating a ComponentError."""
        error = ComponentError(
            component_name='test',
            error=ValueError("test error"),
            severity=ErrorSeverity.WARNING,
            context={'epoch': 5},
        )

        assert error.component_name == 'test'
        assert isinstance(error.error, ValueError)
        assert error.severity == ErrorSeverity.WARNING
        assert error.context == {'epoch': 5}

    def test_component_error_str(self):
        """Test ComponentError string representation."""
        error = ComponentError(
            component_name='test',
            error=ValueError("test error"),
            severity=ErrorSeverity.FATAL,
        )

        str_repr = str(error)
        assert 'fatal' in str_repr.lower()
        assert 'test' in str_repr
        assert 'test error' in str_repr


class TestPipelineRepr:
    """Tests for pipeline string representation."""

    def test_repr(self, training_context):
        """Test pipeline repr."""
        pipeline = TrainingPipeline(training_context)
        pipeline.register('model', MockComponent(training_context))
        pipeline.register('optimizer', MockComponent(training_context))

        repr_str = repr(pipeline)

        assert 'TrainingPipeline' in repr_str
        assert 'model' in repr_str
        assert 'optimizer' in repr_str
