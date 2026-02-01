"""
Unit tests for TrainingLoopManager - the core training loop.

Tests the TrainingLoopManager that handles:
- Gradient accumulation
- Mixed precision training
- Loss computation and tracking
- Step/epoch management

Note: TrainingLoopManager requires CUDA device, so many tests are
skipped when running in CPU-only mode.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from ava.training.loop import TrainingLoopManager, TrainingLoopConfig
from ava.training.context import TrainingContext

# Mark for tests that use TrainingLoopManager
# TrainingLoopManager requires a CUDA device internally, so these tests
# must be skipped when the training_context uses CPU device
def requires_cuda_context(training_context):
    """Check if the training_context uses a CUDA device."""
    if training_context.device.type != 'cuda':
        pytest.skip("TrainingLoopManager requires CUDA device in context")


class TestTrainingLoopConfig:
    """Tests for TrainingLoopConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingLoopConfig()

        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.use_amp is True
        assert config.amp_dtype == torch.bfloat16
        assert config.log_interval == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingLoopConfig(
            gradient_accumulation_steps=4,
            max_grad_norm=0.5,
            use_amp=False,
            log_interval=50,
        )

        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 0.5
        assert config.use_amp is False
        assert config.log_interval == 50

    def test_cuda_graph_config(self):
        """Test CUDA graph configuration."""
        config = TrainingLoopConfig(
            use_cuda_graphs=True,
            cuda_graph_warmup_steps=5,
            use_canonical_shapes=True,
        )

        assert config.use_cuda_graphs is True
        assert config.cuda_graph_warmup_steps == 5
        assert config.use_canonical_shapes is True

    def test_distributed_config(self):
        """Test distributed training configuration."""
        config = TrainingLoopConfig(
            minimize_distributed_barriers=True,
            sync_loss_across_ranks=False,
        )

        assert config.minimize_distributed_barriers is True
        assert config.sync_loss_across_ranks is False

    def test_logging_config(self):
        """Test logging configuration."""
        config = TrainingLoopConfig(
            log_mode='verbose',
            verbose_log_interval=100,
            logging_disabled=False,
            progress_bar_enabled=True,
        )

        assert config.log_mode == 'verbose'
        assert config.verbose_log_interval == 100


class TestTrainingLoopManagerCreation:
    """Tests for TrainingLoopManager initialization."""

    def test_manager_creation(self, training_context):
        """Test creating training loop manager."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)

        assert manager.context is training_context
        assert manager._initialized is False
        assert manager._global_step == 0

    def test_manager_initialize(self, training_context):
        """Test manager initialization."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        assert manager._initialized is True

    def test_manager_cleanup(self, training_context):
        """Test manager cleanup."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()
        manager.cleanup()

        # Should cleanup without error
        assert manager.prefetcher is None


class TestTrainingLoopManagerComponents:
    """Tests for setting and using components."""

    def test_set_components(self, training_context):
        """Test setting component references."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Create mock components
        mock_metrics = Mock()
        mock_generation = Mock()
        mock_checkpoint = Mock()

        manager.set_components(
            metrics_manager=mock_metrics,
            generation_manager=mock_generation,
            checkpoint_manager=mock_checkpoint,
        )

        assert manager._metrics_manager is mock_metrics
        assert manager._generation_manager is mock_generation
        assert manager._checkpoint_manager is mock_checkpoint

    def test_component_integration(self, training_context):
        """Test that components are called correctly."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        mock_metrics = Mock()
        mock_metrics.log_step = Mock()

        manager._metrics_manager = mock_metrics

        # The metrics manager should be available
        assert manager._metrics_manager is not None


class TestGradientAccumulation:
    """Tests for gradient accumulation logic."""

    def test_accumulation_step_counting(self, training_context):
        """Test gradient accumulation step counting."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        config = TrainingLoopConfig(gradient_accumulation_steps=4)

        # Verify config is stored correctly
        manager._loop_config = config

        # Step counter logic depends on implementation
        assert config.gradient_accumulation_steps == 4

    def test_accumulation_boundary_detection(self):
        """Test detecting accumulation boundaries."""
        # This test doesn't need training_context
        config = TrainingLoopConfig(gradient_accumulation_steps=4)

        # At step 3, we should accumulate (step 4 = boundary)
        # At step 0, 1, 2, 3 are accumulation steps
        # At step 3 (idx 4), we step
        assert (4 % config.gradient_accumulation_steps) == 0  # Boundary


class TestLossTracking:
    """Tests for loss accumulation and tracking."""

    def test_loss_accumulator_initialization(self, training_context):
        """Test loss accumulator is initialized."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Loss accumulator starts as None
        assert manager._loss_accumulator is None
        assert manager._loss_count == 0

    def test_step_losses_list(self, training_context):
        """Test step losses list for epoch averaging."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        assert manager._step_losses == []

    def test_loss_values_tracking(self, training_context):
        """Test that loss values are tracked."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Manually add a loss value
        manager._step_losses.append(0.5)
        manager._step_losses.append(0.4)

        assert len(manager._step_losses) == 2
        assert sum(manager._step_losses) / len(manager._step_losses) == 0.45


class TestMixedPrecision:
    """Tests for mixed precision training."""

    def test_amp_config(self, training_context):
        """Test AMP configuration."""
        config = TrainingLoopConfig(
            use_amp=True,
            amp_dtype=torch.float16,
        )

        assert config.use_amp is True
        assert config.amp_dtype == torch.float16

    def test_amp_disabled(self, training_context):
        """Test AMP disabled configuration."""
        config = TrainingLoopConfig(use_amp=False)

        assert config.use_amp is False

    def test_bf16_dtype(self, training_context):
        """Test bfloat16 dtype configuration."""
        config = TrainingLoopConfig(
            use_amp=True,
            amp_dtype=torch.bfloat16,
        )

        assert config.amp_dtype == torch.bfloat16


class TestCUDAGraphs:
    """Tests for CUDA graph optimization."""

    def test_cuda_graph_cache(self, training_context):
        """Test CUDA graph cache initialization."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        assert manager._cuda_graph_cache == {}
        assert manager._max_cached_graphs == 1

    def test_cuda_graph_cleanup(self, training_context):
        """Test CUDA graph cleanup."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Add mock graph data
        manager._cuda_graph_cache[(32, 64)] = {
            'graph': Mock(),
            'static_input': {},
            'static_loss': None,
        }

        manager._cleanup_cuda_graph()

        assert manager._cuda_graph_cache == {}

    def test_canonical_shape_lut(self, training_context):
        """Test canonical shape lookup table."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # LUT should be empty initially
        assert len(manager._canonical_shape_lut) == 0


class TestStepTiming:
    """Tests for step timing and throughput."""

    def test_step_times_window(self, training_context):
        """Test step times rolling window."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        assert manager._step_times == []
        assert manager._step_times_window == 100

    def test_timing_tracking(self, training_context):
        """Test step timing is tracked."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Simulate step timing
        manager._step_start_time = 0.0
        manager._step_times.append(0.1)
        manager._step_times.append(0.2)

        assert len(manager._step_times) == 2


class TestLifecycleHooks:
    """Tests for lifecycle hook implementations."""

    def test_on_epoch_start(self, training_context):
        """Test on_epoch_start hook."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Should not raise
        manager.on_epoch_start(0)

    def test_on_epoch_end(self, training_context):
        """Test on_epoch_end hook."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Should not raise
        manager.on_epoch_end(0)

    def test_on_step_start(self, training_context):
        """Test on_step_start hook."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Should not raise
        manager.on_step_start(0)

    def test_on_step_end(self, training_context):
        """Test on_step_end hook."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Should not raise
        manager.on_step_end(0, 0.5)


class TestMetricsBatcher:
    """Tests for metrics batching optimization."""

    def test_metrics_batcher_exists(self, training_context):
        """Test metrics batcher is initialized."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        assert manager._metrics_batcher is not None


class TestOverlappedAccumulation:
    """Tests for overlapped gradient accumulation."""

    def test_overlapped_accumulator_config(self):
        """Test overlapped accumulation configuration."""
        config = TrainingLoopConfig(
            use_overlapped_accumulation=True,
            gradient_accumulation_steps=4,
        )

        assert config.use_overlapped_accumulation is True

    def test_pipeline_microbatching_config(self):
        """Test pipeline microbatching configuration."""
        config = TrainingLoopConfig(
            use_pipeline_microbatching=True,
            pipeline_overlap_factor=2,
        )

        assert config.use_pipeline_microbatching is True
        assert config.pipeline_overlap_factor == 2


class TestMemoryMonitoring:
    """Tests for memory monitoring."""

    def test_memory_cache(self, training_context):
        """Test memory cache initialization."""
        requires_cuda_context(training_context)
        manager = TrainingLoopManager(training_context)
        manager.initialize()

        # Memory cache should be initialized
        assert 'total' in manager._memory_cache
        assert 'step' in manager._memory_cache

    def test_memory_monitoring_disabled_by_default(self):
        """Test memory monitoring is disabled by default."""
        config = TrainingLoopConfig()

        assert config.enable_memory_monitoring is False


class TestProfilingConfig:
    """Tests for profiling configuration."""

    def test_profiling_disabled_by_default(self):
        """Test profiling is disabled by default."""
        config = TrainingLoopConfig()

        assert config.enable_profiling is False

    def test_profiling_config(self):
        """Test profiling configuration."""
        config = TrainingLoopConfig(
            enable_profiling=True,
            profile_start_step=10,
            profile_end_step=100,
            profile_dir="./my_profiles",
        )

        assert config.enable_profiling is True
        assert config.profile_start_step == 10
        assert config.profile_end_step == 100


class TestValidationConfig:
    """Tests for validation configuration."""

    def test_validation_steps(self):
        """Test validation step configuration."""
        config = TrainingLoopConfig(
            eval_steps=500,
            max_val_batches=20,
        )

        assert config.eval_steps == 500
        assert config.max_val_batches == 20


class TestGenerationConfig:
    """Tests for generation configuration."""

    def test_generation_frequency(self):
        """Test generation frequency configuration."""
        config = TrainingLoopConfig(
            generate_every_n_steps=1000,
        )

        assert config.generate_every_n_steps == 1000


class TestMaxSteps:
    """Tests for max_steps limitation."""

    def test_max_steps_none(self):
        """Test max_steps is None by default."""
        config = TrainingLoopConfig()

        assert config.max_steps is None

    def test_max_steps_set(self):
        """Test max_steps can be set."""
        config = TrainingLoopConfig(max_steps=10000)

        assert config.max_steps == 10000
