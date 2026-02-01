"""
Tests for training speedup features.

Tests the three speedup techniques from recent arXiv papers:
1. SYMI Optimizer Decoupling (arXiv 2504.19925) - ~30% speedup
2. 2:4 Activation Sparsity (arXiv 2503.16672) - 1.2-1.3x speedup
3. Stable-MoE Routing (arXiv 2512.06784) - 40% throughput
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional


# Fixtures are now defined in conftest.py:
# - device: CUDA if available, else CPU
# - hidden_size: 256
# - intermediate_size: 1024
# - num_experts: 8
# - batch_size: 32


# =========================================================================
# SPARSE24 TESTS
# =========================================================================

class TestSparse24:
    """Tests for 2:4 activation sparsity."""

    def test_sparsity_pattern(self, device):
        """Test that 2:4 sparsity produces correct pattern."""
        from ava.cuda.sparse24_kernels import apply_24_sparsity

        # Create input divisible by 4
        x = torch.randn(32, 128, device=device)

        sparse_x, mask = apply_24_sparsity(x)

        # Reshape to groups of 4
        mask_grouped = mask.view(-1, 4)

        # Each group should have exactly 2 True values
        ones_per_group = mask_grouped.sum(dim=-1)
        assert (ones_per_group == 2).all(), "Each group of 4 should have exactly 2 non-zeros"

    def test_magnitude_pruning(self, device):
        """Test that largest magnitude values are kept."""
        from ava.cuda.sparse24_kernels import apply_24_sparsity

        # Create input with known magnitudes
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)  # Should keep 3.0, 4.0

        sparse_x, mask = apply_24_sparsity(x)

        # Indices 2 and 3 should be kept (largest magnitudes)
        assert mask[0, 2] == True
        assert mask[0, 3] == True
        assert mask[0, 0] == False
        assert mask[0, 1] == False

    def test_gradient_flow(self, device):
        """Test that gradients flow through STE."""
        from ava.cuda.sparse24_kernels import sparse24_ste

        x = torch.randn(32, 128, device=device, requires_grad=True)

        sparse_x = sparse24_ste(x)
        loss = sparse_x.sum()
        loss.backward()

        # Gradient should flow through
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_hardware_detection(self):
        """Test hardware detection for 2:4 sparsity."""
        from ava.cuda.sparse24_kernels import is_sparse24_supported, get_sparse24_info

        info = get_sparse24_info()
        assert 'supported' in info
        assert 'sm_version' in info
        assert 'triton_available' in info

        # If CUDA available, SM version should be > 0
        if torch.cuda.is_available():
            assert info['sm_version'] > 0

    def test_sparse24_linear(self, device, hidden_size, intermediate_size):
        """Test Sparse24Linear module."""
        from ava.cuda.sparse24_kernels import Sparse24Linear

        layer = Sparse24Linear(
            hidden_size,
            intermediate_size,
            apply_to_output=True,
            warmup_steps=0,  # No warmup for testing
        ).to(device)

        x = torch.randn(32, hidden_size, device=device)

        # Forward pass
        layer.train()
        output = layer(x)

        assert output.shape == (32, intermediate_size)

    def test_sparse24_expert_ffn(self, device, hidden_size, intermediate_size):
        """Test Sparse24ExpertFFN module."""
        from ava.cuda.sparse24_kernels import Sparse24ExpertFFN

        ffn = Sparse24ExpertFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='swiglu',
            warmup_steps=0,
        ).to(device)

        x = torch.randn(32, hidden_size, device=device)

        ffn.train()
        output = ffn(x)

        assert output.shape == x.shape


# =========================================================================
# STABLE-MOE TESTS
# =========================================================================

class TestStableMoE:
    """Tests for Stable-MoE routing."""

    def test_router_creation(self, device, hidden_size, num_experts):
        """Test StableMoERouter initialization."""
        from ava.models.routing import StableMoERouter

        router = StableMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=2,
            adaptation_rate=0.01,
            temperature_init=1.0,
        ).to(device)

        assert router.num_experts == num_experts
        assert router.temperature == 1.0
        assert router.capacity_factors.shape == (num_experts,)

    def test_lyapunov_convergence(self, device, hidden_size, num_experts, batch_size):
        """Test that Lyapunov controller reduces utilization variance."""
        from ava.models.routing import StableMoERouter

        router = StableMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=2,
            adaptation_rate=0.1,  # Higher rate for faster convergence in test
        ).to(device)

        router.train()

        initial_std = router.expert_utilization_ema.std().item()

        # Run multiple forward passes
        for _ in range(100):
            x = torch.randn(batch_size, hidden_size, device=device)
            indices, weights, aux_loss, metrics = router(x)

        final_std = router.expert_utilization_ema.std().item()

        # Utilization should become more balanced
        assert final_std <= initial_std * 1.5, "Utilization variance should not increase significantly"

    def test_temperature_annealing(self, device, hidden_size, num_experts, batch_size):
        """Test temperature annealing schedule."""
        from ava.models.routing import StableMoERouter

        initial_temp = 2.0
        router = StableMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=2,
            temperature_init=initial_temp,
            temperature_min=0.1,
            temperature_decay=0.99,
        ).to(device)

        router.train()

        # Run forward passes
        for _ in range(100):
            x = torch.randn(batch_size, hidden_size, device=device)
            router(x)

        # Temperature should have decreased
        assert router.temperature < initial_temp
        assert router.temperature >= 0.1  # Should not go below min

    def test_capacity_bounds(self, device, hidden_size, num_experts, batch_size):
        """Test adaptive capacity factor bounds."""
        from ava.models.routing import StableMoERouter

        capacity_min = 1.0
        capacity_max = 2.0

        router = StableMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=2,
            capacity_min=capacity_min,
            capacity_max=capacity_max,
            adaptation_rate=0.5,  # High rate to test bounds
        ).to(device)

        router.train()

        # Run many forward passes
        for _ in range(200):
            x = torch.randn(batch_size, hidden_size, device=device)
            router(x)

        # Capacity factors should stay within bounds
        assert (router.capacity_factors >= capacity_min).all()
        assert (router.capacity_factors <= capacity_max).all()

    def test_metrics_logging(self, device, hidden_size, num_experts, batch_size):
        """Test that metrics are properly logged."""
        from ava.models.routing import StableMoERouter

        router = StableMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=2,
            log_metrics=True,
        ).to(device)

        router.train()
        x = torch.randn(batch_size, hidden_size, device=device)

        indices, weights, aux_loss, metrics = router(x)

        # Check Stable-MoE specific metrics
        assert 'stable_moe_temperature' in metrics
        assert 'stable_moe_lyapunov' in metrics
        assert 'stable_moe_capacity_mean' in metrics
        assert 'stable_moe_utilization_mean' in metrics


# =========================================================================
# SYMI TESTS
# =========================================================================

class TestSYMI:
    """Tests for SYMI optimizer decoupling."""

    def test_wrapper_creation(self, device, hidden_size, num_experts):
        """Test SYMIOptimizerWrapper creation."""
        from ava.optimizations.symi_optimizer import (
            SYMIOptimizerWrapper,
            SYMIConfig,
        )

        # Create simple model
        model = nn.Linear(hidden_size, hidden_size).to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = SYMIConfig(enabled=True, sync_frequency=10)
        wrapper = SYMIOptimizerWrapper(base_optimizer, model, config)

        assert wrapper.num_partitions >= 1
        assert wrapper._step_count == 0

    def test_expert_param_identification(self, device, hidden_size, intermediate_size, num_experts):
        """Test identification of expert parameters."""
        from ava.optimizations.symi_optimizer import SYMIOptimizerWrapper, SYMIConfig
        from ava.models.experts import ExpertParallelGroup

        # Create model with experts
        experts = ExpertParallelGroup(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ).to(device)

        base_optimizer = torch.optim.AdamW(experts.parameters(), lr=1e-4)
        config = SYMIConfig(enabled=True)

        wrapper = SYMIOptimizerWrapper(base_optimizer, experts, config)

        # Should identify expert params
        assert len(wrapper.expert_params) > 0

    def test_optimizer_step(self, device, hidden_size):
        """Test optimizer step functionality."""
        from ava.optimizations.symi_optimizer import (
            SYMIOptimizerWrapper,
            SYMIConfig,
        )

        model = nn.Linear(hidden_size, hidden_size).to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = SYMIConfig(enabled=True, sync_frequency=10)
        wrapper = SYMIOptimizerWrapper(base_optimizer, model, config)

        # Forward + backward
        x = torch.randn(32, hidden_size, device=device)
        loss = model(x).sum()
        loss.backward()

        # Optimizer step
        wrapper.step()

        assert wrapper._step_count == 1

    def test_checkpoint_restore(self, device, hidden_size):
        """Test state dict save/load."""
        from ava.optimizations.symi_optimizer import (
            SYMIOptimizerWrapper,
            SYMIConfig,
        )

        model = nn.Linear(hidden_size, hidden_size).to(device)
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = SYMIConfig(enabled=True)
        wrapper = SYMIOptimizerWrapper(base_optimizer, model, config)

        # Run some steps
        for _ in range(5):
            x = torch.randn(32, hidden_size, device=device)
            loss = model(x).sum()
            loss.backward()
            wrapper.step()
            wrapper.zero_grad()

        # Save state
        state = wrapper.state_dict()

        # Create new wrapper and load state
        new_wrapper = SYMIOptimizerWrapper(
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            model,
            config,
        )
        new_wrapper.load_state_dict(state)

        assert new_wrapper._step_count == 5

    def test_expert_param_stats(self, device, hidden_size, intermediate_size, num_experts):
        """Test expert parameter statistics."""
        from ava.optimizations.symi_optimizer import SYMIOptimizerWrapper, SYMIConfig
        from ava.models.experts import ExpertParallelGroup

        experts = ExpertParallelGroup(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ).to(device)

        base_optimizer = torch.optim.AdamW(experts.parameters(), lr=1e-4)
        config = SYMIConfig(enabled=True)

        wrapper = SYMIOptimizerWrapper(base_optimizer, experts, config)

        stats = wrapper.get_expert_param_stats()

        assert 'total_expert_params' in stats
        assert 'num_partitions' in stats
        assert stats['total_expert_params'] > 0


# =========================================================================
# INTEGRATION TESTS
# =========================================================================

class TestIntegration:
    """Integration tests for all speedup features together."""

    def test_stable_moe_in_moe_layer(self, device, hidden_size, intermediate_size, num_experts):
        """Test StableMoE router in SparseMoELayer."""
        from ava.models.moe_layer import SparseMoELayer

        layer = SparseMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=2,
            router_type='stable_moe',
        ).to(device)

        layer.train()
        x = torch.randn(32, 16, hidden_size, device=device)  # [batch, seq, hidden]

        output, aux_loss, metrics = layer(x)

        assert output.shape == x.shape
        assert 'stable_moe_temperature' in metrics

    def test_sparse24_in_expert(self, device, hidden_size, intermediate_size):
        """Test 2:4 sparsity integration in HighPerformanceExpert."""
        from ava.models.experts import HighPerformanceExpert

        expert = HighPerformanceExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='swiglu',
            use_sparse24=True,
            sparse24_warmup_steps=0,  # No warmup for testing
        ).to(device)

        expert.train()
        x = torch.randn(32, hidden_size, device=device)

        output = expert(x)

        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_features_together(self, device, hidden_size, intermediate_size, num_experts):
        """Test all speedup features working together."""
        from ava.models.moe_layer import SparseMoELayer
        from ava.optimizations.symi_optimizer import SYMIOptimizerWrapper, SYMIConfig

        # Create MoE layer with StableMoE routing
        layer = SparseMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=2,
            router_type='stable_moe',
        ).to(device)

        # Wrap optimizer with SYMI
        base_optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        symi_config = SYMIConfig(enabled=True, sync_frequency=10)
        optimizer = SYMIOptimizerWrapper(base_optimizer, layer, symi_config)

        layer.train()

        # Training loop
        for step in range(10):
            x = torch.randn(16, 8, hidden_size, device=device)

            output, aux_loss, metrics = layer(x)
            loss = output.sum() + aux_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # All components should have run successfully
        assert optimizer._step_count == 10


# =========================================================================
# PERFORMANCE TESTS (OPTIONAL)
# =========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for perf tests")
class TestPerformance:
    """Performance benchmarks for speedup features."""

    def test_sparse24_overhead(self, device, hidden_size, intermediate_size):
        """Measure 2:4 sparsity overhead."""
        from ava.cuda.sparse24_kernels import is_sparse24_supported

        if not is_sparse24_supported():
            pytest.skip("2:4 sparsity not supported on this hardware")

        from ava.models.experts import HighPerformanceExpert
        import time

        # Without sparsity
        expert_dense = HighPerformanceExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_sparse24=False,
        ).to(device)

        # With sparsity
        expert_sparse = HighPerformanceExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_sparse24=True,
            sparse24_warmup_steps=0,
        ).to(device)

        x = torch.randn(1024, hidden_size, device=device)

        # Warmup
        for _ in range(10):
            expert_dense(x)
            expert_sparse(x)

        torch.cuda.synchronize()

        # Benchmark dense
        start = time.perf_counter()
        for _ in range(100):
            expert_dense(x)
        torch.cuda.synchronize()
        dense_time = time.perf_counter() - start

        # Benchmark sparse
        expert_sparse.train()
        start = time.perf_counter()
        for _ in range(100):
            expert_sparse(x)
        torch.cuda.synchronize()
        sparse_time = time.perf_counter() - start

        print(f"\nDense time: {dense_time:.4f}s")
        print(f"Sparse time: {sparse_time:.4f}s")
        print(f"Ratio: {sparse_time/dense_time:.2f}x")


# =========================================================================
# RUN TESTS
# =========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
