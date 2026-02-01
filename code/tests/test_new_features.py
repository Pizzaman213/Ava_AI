"""
Comprehensive Test Suite for New Ava Features.

Tests newly implemented features:
- Gradient Surgery
- Auxiliary-Free Load Balancing
- Sequential Expert Group

Run with: python -m pytest code/tests/test_new_features.py -v
"""

import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGradientSurgery(unittest.TestCase):
    """Tests for gradient surgery methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_dim = 100

    def test_pcgrad(self):
        """Test PCGrad."""
        from ava.optimizations.gradient_surgery import PCGrad

        pcgrad = PCGrad()

        # Create conflicting gradients
        grad1 = torch.randn(self.grad_dim, device=self.device)
        grad2 = -grad1 + torch.randn(self.grad_dim, device=self.device) * 0.1  # Mostly opposite

        modified = pcgrad([grad1, grad2])

        self.assertEqual(len(modified), 2)
        self.assertEqual(modified[0].shape, (self.grad_dim,))

    def test_cagrad(self):
        """Test CAGrad."""
        from ava.optimizations.gradient_surgery import CAGrad

        cagrad = CAGrad(c=0.5)

        grad1 = torch.randn(self.grad_dim, device=self.device)
        grad2 = torch.randn(self.grad_dim, device=self.device)
        grad3 = torch.randn(self.grad_dim, device=self.device)

        combined = cagrad([grad1, grad2, grad3])

        self.assertEqual(combined.shape, (self.grad_dim,))

    def test_mgda(self):
        """Test MGDA."""
        from ava.optimizations.gradient_surgery import MGDA

        mgda = MGDA(normalize=True)

        grad1 = torch.randn(self.grad_dim, device=self.device)
        grad2 = torch.randn(self.grad_dim, device=self.device)

        combined = mgda([grad1, grad2])

        self.assertEqual(combined.shape, (self.grad_dim,))

    def test_gradnorm(self):
        """Test GradNorm."""
        from ava.optimizations.gradient_surgery import GradNorm

        gradnorm = GradNorm(num_tasks=3, alpha=1.5)

        # Create dummy task losses and gradients (with requires_grad for backward)
        task_losses = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
        ]
        task_grads = [
            torch.randn(self.grad_dim),
            torch.randn(self.grad_dim),
            torch.randn(self.grad_dim),
        ]

        weights = gradnorm.update(task_losses, task_grads)

        self.assertEqual(weights.shape, (3,))
        self.assertTrue((weights > 0).all())


class TestAuxFreeRouter(unittest.TestCase):
    """Tests for auxiliary-free MoE routing."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.hidden_size = 256
        self.num_experts = 8

    def test_aux_free_load_balancer(self):
        """Test AuxFreeLoadBalancer."""
        from ava.models.routing import AuxFreeLoadBalancer

        balancer = AuxFreeLoadBalancer(
            num_experts=self.num_experts,
            balance_factor=0.1,
        )

        # Simulate routing updates
        expert_indices = torch.randint(0, self.num_experts, (self.batch_size, 2), device=self.device)
        balancer.update_utilization(expert_indices, self.batch_size)

        stats = balancer.get_stats()
        self.assertIn('expert_utilization', stats)
        self.assertEqual(stats['expert_utilization'].shape, (self.num_experts,))

    def test_aux_free_router(self):
        """Test AuxFreeRouter."""
        from ava.models.routing import AuxFreeRouter

        router = AuxFreeRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_selected_experts=2,
            balance_factor=0.1,
        ).to(self.device)

        hidden_states = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        indices, weights, aux_loss, metrics = router(hidden_states, training=True)

        self.assertEqual(indices.shape, (self.batch_size, 2))
        self.assertEqual(weights.shape, (self.batch_size, 2))
        # Aux loss should be minimal (only z-loss, no load balance loss)
        self.assertIn('balance_utilization_std', metrics)


class TestSequentialExpertGroup(unittest.TestCase):
    """Tests for SequentialExpertGroup fallback."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_experts = 4

    def test_sequential_expert_group_creation(self):
        """Test SequentialExpertGroup can be created."""
        from ava.models.experts import SequentialExpertGroup

        expert_group = SequentialExpertGroup(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        ).to(self.device)

        self.assertEqual(expert_group.num_experts, self.num_experts)

    def test_sequential_expert_group_forward(self):
        """Test SequentialExpertGroup forward pass."""
        from ava.models.experts import SequentialExpertGroup

        expert_group = SequentialExpertGroup(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        ).to(self.device)

        # Create input and routing
        k = 2  # Number of experts per token
        hidden_states = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        expert_indices = torch.randint(0, self.num_experts, (self.batch_size, k), device=self.device)
        expert_weights = F.softmax(torch.randn(self.batch_size, k, device=self.device), dim=-1)

        output = expert_group(hidden_states, expert_indices, expert_weights)

        # Output is [batch, k, hidden] - one output per selected expert
        self.assertEqual(output.shape, (self.batch_size, k, self.hidden_size))

        # Sum over k dimension to get final output [batch, hidden]
        final_output = output.sum(dim=1)
        self.assertEqual(final_output.shape, (self.batch_size, self.hidden_size))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestGradientSurgery,
        TestAuxFreeRouter,
        TestSequentialExpertGroup,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
