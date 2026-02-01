"""
Unit tests for TrainingContext - the shared state hub.

Tests the TrainingContext dataclass that provides shared state
between all training components.
"""

import pytest
import torch
import torch.nn as nn

from ava.training.context import TrainingContext, TrainingComponent, ManagerInterface


class TestTrainingContext:
    """Tests for TrainingContext dataclass."""

    def test_creation_with_model_only(self, simple_moe_model):
        """Test creating context with just a model."""
        context = TrainingContext(model=simple_moe_model)

        assert context.model is simple_moe_model
        assert context.optimizer is None
        assert context.scheduler is None
        assert context.epoch == 0
        assert context.step == 0
        assert context.current_loss == 0.0
        assert context.best_loss == float('inf')
        assert context.rank == 0
        assert context.world_size == 1
        assert context.is_main_process is True

    def test_creation_with_all_fields(self, simple_moe_model, device, optimizer, scheduler):
        """Test creating context with all fields specified."""
        context = TrainingContext(
            model=simple_moe_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config={'model': {'hidden_size': 64}},
            epoch=5,
            step=100,
            current_loss=0.5,
            best_loss=0.3,
            rank=2,
            world_size=4,
            is_main_process=False,
        )

        assert context.model is simple_moe_model
        assert context.optimizer is optimizer
        assert context.scheduler is scheduler
        assert context.device == device
        assert context.epoch == 5
        assert context.step == 100
        assert context.current_loss == 0.5
        assert context.best_loss == 0.3
        assert context.rank == 2
        assert context.world_size == 4
        assert context.is_main_process is False

    def test_post_init_updates_is_main_process(self, simple_moe_model):
        """Test that __post_init__ corrects is_main_process based on rank."""
        # Non-zero rank should force is_main_process to False
        context = TrainingContext(
            model=simple_moe_model,
            rank=1,
            is_main_process=True,  # This should be corrected
        )

        assert context.is_main_process is False

    def test_update_from_config(self, simple_moe_model, minimal_config):
        """Test updating context from configuration dict."""
        context = TrainingContext(model=simple_moe_model)
        context.update_from_config(minimal_config)

        # Check gradient accumulation was updated
        expected_accum = minimal_config['training']['batching']['gradient_accumulation_steps']
        assert context.gradient_accumulation_steps == expected_accum

        # Check metadata was populated
        assert 'batch_size' in context.metadata or 'max_length' in context.metadata

    def test_update_from_config_precision_bf16(self, simple_moe_model):
        """Test that bf16 precision is correctly parsed."""
        config = {
            'training': {
                'precision': {'mixed_precision': 'bf16'},
            },
        }
        context = TrainingContext(model=simple_moe_model)
        context.update_from_config(config)

        assert context.use_amp is True
        assert context.amp_dtype == torch.bfloat16

    def test_update_from_config_precision_fp16(self, simple_moe_model):
        """Test that fp16 precision is correctly parsed."""
        config = {
            'training': {
                'precision': {'mixed_precision': 'fp16'},
            },
        }
        context = TrainingContext(model=simple_moe_model)
        context.update_from_config(config)

        assert context.use_amp is True
        assert context.amp_dtype == torch.float16

    def test_update_from_config_precision_fp32(self, simple_moe_model):
        """Test that fp32 precision disables AMP."""
        config = {
            'training': {
                'precision': {'mixed_precision': 'fp32'},
            },
        }
        context = TrainingContext(model=simple_moe_model)
        context.update_from_config(config)

        assert context.use_amp is False
        assert context.amp_dtype == torch.float32

    def test_metadata_storage(self, simple_moe_model):
        """Test that metadata dict stores custom values."""
        context = TrainingContext(model=simple_moe_model)

        context.metadata['custom_key'] = 'custom_value'
        context.metadata['batch_size'] = 32

        assert context.metadata['custom_key'] == 'custom_value'
        assert context.metadata['batch_size'] == 32

    def test_loss_tracking(self, simple_moe_model):
        """Test loss value tracking."""
        context = TrainingContext(model=simple_moe_model)

        # Update current loss
        context.current_loss = 2.5
        assert context.current_loss == 2.5

        # Update best loss
        context.best_loss = 1.0
        assert context.best_loss == 1.0

    def test_step_and_epoch_tracking(self, simple_moe_model):
        """Test step and epoch counters."""
        context = TrainingContext(model=simple_moe_model)

        context.epoch = 5
        context.step = 1000
        context.micro_step = 4000

        assert context.epoch == 5
        assert context.step == 1000
        assert context.micro_step == 4000


class TestTrainingComponent:
    """Tests for TrainingComponent base class."""

    def test_component_initialization(self, training_context):
        """Test that components receive context and start uninitialized."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

        component = TestComponent(training_context)

        assert component.context is training_context
        assert component._initialized is False
        assert component.is_initialized() is False

    def test_component_model_shortcut(self, training_context):
        """Test model property shortcut."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                pass

            def cleanup(self):
                pass

        component = TestComponent(training_context)
        assert component.model is training_context.model

    def test_component_device_shortcut(self, training_context):
        """Test device property shortcut."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                pass

            def cleanup(self):
                pass

        component = TestComponent(training_context)
        assert component.device == training_context.device

    def test_component_config_shortcut(self, training_context):
        """Test config property shortcut."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                pass

            def cleanup(self):
                pass

        component = TestComponent(training_context)
        assert component.config is training_context.config

    def test_assert_initialized_raises(self, training_context):
        """Test that assert_initialized raises if not initialized."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

        component = TestComponent(training_context)

        with pytest.raises(RuntimeError, match="not initialized"):
            component.assert_initialized()

    def test_assert_initialized_passes(self, training_context):
        """Test that assert_initialized passes after initialization."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

        component = TestComponent(training_context)
        component.initialize()

        # Should not raise
        component.assert_initialized()

    def test_component_logger(self, training_context):
        """Test that components have a logger."""
        class TestComponent(TrainingComponent):
            def initialize(self):
                pass

            def cleanup(self):
                pass

        component = TestComponent(training_context)
        assert component.logger is not None


class TestManagerInterface:
    """Tests for ManagerInterface - extended component interface."""

    def test_manager_lifecycle_hooks(self, training_context):
        """Test that lifecycle hooks can be called."""
        class TestManager(ManagerInterface):
            def __init__(self, context):
                super().__init__(context)
                self.epoch_start_called = False
                self.epoch_end_called = False
                self.step_start_called = False
                self.step_end_called = False
                self.error_called = False

            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

            def on_epoch_start(self, epoch):
                self.epoch_start_called = True
                self.last_epoch = epoch

            def on_epoch_end(self, epoch):
                self.epoch_end_called = True

            def on_step_start(self, step):
                self.step_start_called = True
                self.last_step = step

            def on_step_end(self, step, loss):
                self.step_end_called = True
                self.last_loss = loss

            def on_error(self, error):
                self.error_called = True
                self.last_error = error

        manager = TestManager(training_context)
        manager.initialize()

        # Test epoch hooks
        manager.on_epoch_start(5)
        assert manager.epoch_start_called
        assert manager.last_epoch == 5

        manager.on_epoch_end(5)
        assert manager.epoch_end_called

        # Test step hooks
        manager.on_step_start(100)
        assert manager.step_start_called
        assert manager.last_step == 100

        manager.on_step_end(100, 0.5)
        assert manager.step_end_called
        assert manager.last_loss == 0.5

        # Test error hook
        manager.on_error(ValueError("test error"))
        assert manager.error_called

    def test_manager_get_status(self, training_context):
        """Test get_status returns dict."""
        class TestManager(ManagerInterface):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

            def get_status(self):
                return {'custom': 'status', 'value': 42}

        manager = TestManager(training_context)
        status = manager.get_status()

        assert isinstance(status, dict)
        assert status['custom'] == 'status'
        assert status['value'] == 42

    def test_manager_default_hooks_are_noop(self, training_context):
        """Test that default hook implementations don't raise."""
        class MinimalManager(ManagerInterface):
            def initialize(self):
                self._initialized = True

            def cleanup(self):
                pass

        manager = MinimalManager(training_context)

        # These should all be no-ops (not raise)
        manager.on_epoch_start(0)
        manager.on_epoch_end(0)
        manager.on_step_start(0)
        manager.on_step_end(0, 0.0)

        # Default get_status returns empty dict
        assert manager.get_status() == {}
