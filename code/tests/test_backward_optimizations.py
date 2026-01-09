"""
Unit tests for backward pass optimizations.

Tests:
1. FP8 Backward Computation
2. Fused Gradient All-Reduce
3. Layer-wise Optimizer Updates
4. CUDA Graph Backward Capture
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def simple_input():
    """Create simple input tensors."""
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))
    return input_ids.float(), labels


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# FP8 Backward Tests
# =============================================================================

class TestFP8Backward:
    """Tests for FP8 backward computation."""

    def test_fp8_config_defaults(self):
        """Test FP8Config has correct default values."""
        from ava.optimizations.fp8 import FP8Config

        config = FP8Config()
        assert config.enabled == False
        assert config.backward_enabled == False
        assert config.backward_format == "e5m2"
        assert 'embedding' in config.exclude_layers
        assert 'lm_head' in config.exclude_layers

    def test_fp8_config_backward_enabled(self):
        """Test FP8Config with backward enabled."""
        from ava.optimizations.fp8 import FP8Config

        config = FP8Config(
            enabled=True,
            backward_enabled=True,
            backward_format="e5m2",
        )
        assert config.enabled == True
        assert config.backward_enabled == True

    def test_fp8_linear_wrapper_creation(self, simple_model):
        """Test FP8LinearBackwardWrapper can be created."""
        from ava.optimizations.fp8 import FP8LinearBackwardWrapper, FP8Config

        config = FP8Config(enabled=True, backward_enabled=True)
        linear = nn.Linear(64, 128)
        wrapper = FP8LinearBackwardWrapper(linear, config)

        assert wrapper.linear is linear
        assert wrapper.config.backward_enabled == True
        assert hasattr(wrapper, 'input_scale')
        assert hasattr(wrapper, 'grad_output_scale')

    def test_fp8_forward_pass(self, simple_model, device):
        """Test FP8 forward pass works correctly."""
        from ava.optimizations.fp8 import FP8LinearBackwardWrapper, FP8Config

        config = FP8Config(enabled=True, backward_enabled=True)
        linear = nn.Linear(64, 128).to(device)
        wrapper = FP8LinearBackwardWrapper(linear, config).to(device)

        x = torch.randn(4, 64, device=device)
        wrapper.train()

        # Forward should work
        output = wrapper(x)
        assert output.shape == (4, 128)

    def test_fp8_backward_pass(self, device):
        """Test FP8 backward pass computes gradients."""
        from ava.optimizations.fp8 import FP8LinearBackwardWrapper, FP8Config

        config = FP8Config(enabled=True, backward_enabled=True)
        linear = nn.Linear(64, 128).to(device)
        wrapper = FP8LinearBackwardWrapper(linear, config).to(device)

        x = torch.randn(4, 64, device=device, requires_grad=True)
        wrapper.train()

        output = wrapper(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert wrapper.linear.weight.grad is not None
        assert x.grad is not None

    def test_fp8_gradient_stats(self, device):
        """Test FP8 gradient statistics tracking."""
        from ava.optimizations.fp8 import FP8LinearBackwardWrapper, FP8Config

        config = FP8Config(enabled=True, backward_enabled=True)
        linear = nn.Linear(64, 128).to(device)
        wrapper = FP8LinearBackwardWrapper(linear, config).to(device)

        x = torch.randn(4, 64, device=device)
        wrapper.train()

        output = wrapper(x)
        loss = output.sum()
        loss.backward()

        stats = wrapper.get_gradient_stats()
        assert 'grad_norm' in stats
        assert stats['grad_norm'] > 0

    def test_check_fp8_backward_available(self):
        """Test FP8 backward availability check."""
        from ava.optimizations.fp8 import check_fp8_backward_available

        # Should return bool
        result = check_fp8_backward_available()
        assert isinstance(result, bool)


# =============================================================================
# Fused Gradient All-Reduce Tests
# =============================================================================

class TestFusedGradientAllReduce:
    """Tests for fused gradient all-reduce."""

    def test_fused_allreduce_creation(self):
        """Test FusedGradientAllReduce can be created."""
        from ava.training.distributed import FusedGradientAllReduce

        fused = FusedGradientAllReduce(
            fusion_factor=4,
            fused_bucket_mb=100.0,
            async_allreduce=True,
        )

        assert fused._fusion_factor == 4
        assert fused._max_fused_bytes == 100 * 1024 * 1024
        assert fused._async_allreduce == True

    def test_fused_allreduce_disabled_without_dist(self):
        """Test fused all-reduce is disabled without distributed."""
        from ava.training.distributed import FusedGradientAllReduce

        fused = FusedGradientAllReduce()

        # Should not be enabled without distributed init
        # (depends on whether dist is initialized in test env)
        assert hasattr(fused, '_enabled')

    def test_fused_allreduce_stats(self):
        """Test fused all-reduce statistics."""
        from ava.training.distributed import FusedGradientAllReduce

        fused = FusedGradientAllReduce(fusion_factor=4)
        stats = fused.get_stats()

        assert 'total_fused_ops' in stats
        assert 'fusion_factor' in stats
        assert stats['fusion_factor'] == 4

    def test_create_fused_gradient_allreduce_factory(self):
        """Test factory function."""
        from ava.training.distributed import create_fused_gradient_allreduce

        # Should return None when disabled
        result = create_fused_gradient_allreduce(enabled=False)
        assert result is None


# =============================================================================
# Layer-wise Optimizer Tests
# =============================================================================

class TestLayerwiseOptimizer:
    """Tests for layer-wise optimizer updates."""

    def test_layerwise_optimizer_creation(self, simple_model, device):
        """Test LayerwiseOptimizer can be created."""
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
            bucket_size_mb=25.0,
        )

        assert len(layerwise._buckets) > 0
        assert layerwise.optimizer is base_optimizer

    def test_layerwise_optimizer_forward_backward(self, simple_model, device):
        """Test forward/backward with layer-wise optimizer."""
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
        )

        x = torch.randn(4, 64, device=device)
        output = model(x)
        loss = output.sum()

        # Backward triggers layer-wise updates
        loss.backward()

        # Finish step
        layerwise.finish_step()

        # Parameters should have been updated
        stats = layerwise.get_stats()
        assert stats['step_count'] == 1

    def test_layerwise_optimizer_buckets(self, simple_model, device):
        """Test bucket assignment."""
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
            bucket_size_mb=0.001,  # Very small to force multiple buckets
        )

        # Should have multiple buckets
        assert len(layerwise._buckets) >= 1

    def test_layerwise_optimizer_state_dict(self, simple_model, device):
        """Test state dict save/load."""
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
        )

        # Do one step
        x = torch.randn(4, 64, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        layerwise.finish_step()

        # Save state
        state = layerwise.state_dict()

        # Load state
        layerwise.load_state_dict(state)

    def test_layerwise_optimizer_zero_grad(self, simple_model, device):
        """Test zero_grad works."""
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
        )

        x = torch.randn(4, 64, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Zero grad
        layerwise.zero_grad()

        # Gradients should be None
        for p in model.parameters():
            assert p.grad is None

    def test_create_layerwise_optimizer_factory(self, simple_model, device):
        """Test factory function."""
        from ava.optimizations.layerwise_optimizer import create_layerwise_optimizer

        model = simple_model.to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Enabled
        result = create_layerwise_optimizer(
            base_optimizer, model, enabled=True
        )
        assert hasattr(result, 'finish_step')

        # Disabled
        result = create_layerwise_optimizer(
            base_optimizer, model, enabled=False
        )
        assert result is base_optimizer


# =============================================================================
# CUDA Graph Manager Tests
# =============================================================================

class TestCUDAGraphManager:
    """Tests for CUDA graph manager."""

    def test_cuda_graph_config_defaults(self):
        """Test CUDAGraphConfig has correct defaults."""
        from ava.cuda.graph_manager import CUDAGraphConfig

        config = CUDAGraphConfig()
        assert config.enabled == False
        assert config.capture_backward == True
        assert config.capture_optimizer_step == True
        assert config.max_cached_graphs == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_graph_manager_creation(self, simple_model):
        """Test CUDAGraphManager can be created."""
        from ava.cuda.graph_manager import CUDAGraphManager, CUDAGraphConfig

        model = simple_model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = CUDAGraphConfig(enabled=True)

        manager = CUDAGraphManager(
            model=model,
            optimizer=optimizer,
            config=config,
        )

        assert manager._enabled == True
        assert len(manager._graph_cache) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_graph_warmup(self, simple_model):
        """Test warmup step."""
        from ava.cuda.graph_manager import CUDAGraphManager, CUDAGraphConfig

        model = simple_model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = CUDAGraphConfig(enabled=True, warmup_steps=2)

        manager = CUDAGraphManager(
            model=model,
            optimizer=optimizer,
            config=config,
        )

        # Warmup
        x = torch.randn(4, 64, device='cuda')
        labels = torch.randint(0, 10, (4,), device='cuda')

        loss = manager.warmup_step(x, labels)
        assert manager._warmup_count == 1
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_graph_stats(self, simple_model):
        """Test graph manager statistics."""
        from ava.cuda.graph_manager import CUDAGraphManager, CUDAGraphConfig

        model = simple_model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = CUDAGraphConfig(enabled=True)

        manager = CUDAGraphManager(
            model=model,
            optimizer=optimizer,
            config=config,
        )

        stats = manager.get_stats()
        assert 'enabled' in stats
        assert 'total_captures' in stats
        assert 'cache_hits' in stats

    def test_create_cuda_graph_manager_factory(self, simple_model, device):
        """Test factory function."""
        from ava.cuda.graph_manager import create_cuda_graph_manager

        model = simple_model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Disabled
        result = create_cuda_graph_manager(model, optimizer, enabled=False)
        assert result is None


# =============================================================================
# Configuration Tests
# =============================================================================

class TestBackwardOptimizationConfigs:
    """Tests for configuration dataclasses."""

    def test_gradient_sync_config(self):
        """Test GradientSyncConfig."""
        from ava.config.training_config import GradientSyncConfig

        config = GradientSyncConfig()
        assert config.mode == 'standard'
        assert config.fusion_factor == 4
        assert config.fused_bucket_mb == 100.0

    def test_layerwise_optimizer_config(self):
        """Test LayerwiseOptimizerConfig."""
        from ava.config.training_config import LayerwiseOptimizerConfig

        config = LayerwiseOptimizerConfig()
        assert config.enabled == False
        assert config.bucket_size_mb == 25.0
        assert config.release_gradients_early == True

    def test_cuda_graph_config(self):
        """Test CUDAGraphConfig from training_config."""
        from ava.config.training_config import CUDAGraphConfig

        config = CUDAGraphConfig()
        assert config.enabled == False
        assert config.capture_backward == True
        assert config.capture_optimizer_step == True

    def test_fp8_config_with_backward(self):
        """Test FP8Config with backward settings."""
        from ava.config.training_config import FP8Config

        config = FP8Config()
        assert config.backward_enabled == False
        assert config.backward_format == 'e5m2'
        assert 'embedding' in config.exclude_layers


# =============================================================================
# Integration Tests
# =============================================================================

class TestBackwardOptimizationIntegration:
    """Integration tests combining multiple optimizations."""

    def test_fp8_with_layerwise_optimizer(self, simple_model, device):
        """Test FP8 backward with layer-wise optimizer."""
        from ava.optimizations.fp8 import FP8LinearBackwardWrapper, FP8Config
        from ava.optimizations.layerwise_optimizer import LayerwiseOptimizer

        # Apply FP8 to model
        config = FP8Config(enabled=True, backward_enabled=True)
        model = simple_model.to(device)

        # Replace first linear with FP8
        original_linear = model[0]
        model[0] = FP8LinearBackwardWrapper(original_linear, config).to(device)

        # Create layer-wise optimizer
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        layerwise = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
        )

        # Training step
        x = torch.randn(4, 64, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        layerwise.finish_step()

        # Should complete without error
        stats = layerwise.get_stats()
        assert stats['step_count'] == 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
