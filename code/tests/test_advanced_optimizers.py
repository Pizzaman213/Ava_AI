"""
Unit Tests for Advanced Optimizers Module

Tests for Lion, Sophia, and AdaFactor optimizers with comprehensive validation
of parameter updates, memory efficiency, and convergence properties.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from tests import TEST_CONFIG
from src.Ava.optimization.advanced_optimizers import (
    LionOptimizer,
    SophiaOptimizer,
    AdaFactorOptimizer,
    OptimizerFactory,
    OptimizerConfig
)


class TestLionOptimizer(unittest.TestCase):
    """Test Lion optimizer implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(self.device)

        # Set deterministic weights for consistent testing
        torch.manual_seed(TEST_CONFIG["seed"])
        for param in self.model.parameters():
            nn.init.normal_(param, 0, 0.1)

    def test_lion_initialization(self):
        """Test Lion optimizer initialization with various parameters"""
        optimizer = LionOptimizer(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.01
        )

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['betas'], (0.9, 0.99))
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)

    def test_lion_step(self):
        """Test Lion optimizer parameter updates"""
        optimizer = LionOptimizer(self.model.parameters(), lr=1e-3)

        # Create dummy input and target
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)

        # Initial parameters
        initial_params = [param.clone() for param in self.model.parameters()]

        # Forward pass and loss
        output = self.model(x)
        loss = nn.MSELoss()(output, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Check parameters were updated
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current))

    def test_lion_momentum_state(self):
        """Test Lion momentum state management"""
        optimizer = LionOptimizer(self.model.parameters(), lr=1e-3)

        # First step to initialize momentum
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)

        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Check momentum state is created
        self.assertEqual(len(optimizer.state), len(list(self.model.parameters())))

        for param in self.model.parameters():
            state = optimizer.state[param]
            self.assertIn('exp_avg', state)
            self.assertEqual(state['exp_avg'].shape, param.shape)

    def test_lion_convergence(self):
        """Test Lion optimizer convergence on simple problem"""
        torch.manual_seed(TEST_CONFIG["seed"])

        # Simple quadratic function: f(x) = (x - 2)^2
        x = torch.tensor([0.0], requires_grad=True, device=self.device)
        optimizer = LionOptimizer([x], lr=0.1)

        losses = []
        for _ in range(100):
            loss = (x - 2) ** 2
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should converge to x = 2
        self.assertLess(abs(x.item() - 2.0), 0.1)
        # Loss should decrease
        self.assertLess(losses[-1], losses[0])


class TestSophiaOptimizer(unittest.TestCase):
    """Test Sophia optimizer implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(self.device)

        torch.manual_seed(TEST_CONFIG["seed"])
        for param in self.model.parameters():
            nn.init.normal_(param, 0, 0.1)

    def test_sophia_initialization(self):
        """Test Sophia optimizer initialization"""
        optimizer = SophiaOptimizer(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.965, 0.99),
            rho=0.04,
            weight_decay=1e-1
        )

        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['betas'], (0.965, 0.99))
        self.assertEqual(optimizer.param_groups[0]['rho'], 0.04)

    def test_sophia_hessian_update(self):
        """Test Sophia Hessian diagonal estimation"""
        optimizer = SophiaOptimizer(self.model.parameters(), lr=1e-3)

        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)

        # First step to initialize states
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Check Hessian diagonal states
        for param in self.model.parameters():
            state = optimizer.state[param]
            self.assertIn('exp_hessian_diag_sq', state)
            self.assertEqual(state['exp_hessian_diag_sq'].shape, param.shape)

    def test_sophia_memory_efficiency(self):
        """Test Sophia memory usage compared to standard optimizers"""
        # Create larger model for memory testing
        large_model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.Linear(500, 100),
            nn.Linear(100, 1)
        ).to(self.device)

        sophia_optimizer = SophiaOptimizer(large_model.parameters(), lr=1e-3)

        # Count optimizer state parameters
        total_params = sum(p.numel() for p in large_model.parameters())

        x = torch.randn(64, 1000, device=self.device)
        y = torch.randn(64, 1, device=self.device)

        output = large_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        sophia_optimizer.step()

        # Sophia should have 2 state tensors per parameter (momentum + hessian)
        state_params = sum(
            sum(state_tensor.numel() for state_tensor in state.values() if isinstance(state_tensor, torch.Tensor))
            for state in sophia_optimizer.state.values()
        )

        # Should be approximately 2x the model parameters
        self.assertLessEqual(state_params / total_params, 2.5)


class TestAdaFactorOptimizer(unittest.TestCase):
    """Test AdaFactor optimizer implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(self.device)

        torch.manual_seed(TEST_CONFIG["seed"])
        for param in self.model.parameters():
            nn.init.normal_(param, 0, 0.1)

    def test_adafactor_initialization(self):
        """Test AdaFactor optimizer initialization"""
        optimizer = AdaFactorOptimizer(
            self.model.parameters(),
            lr=None,  # Use automatic scaling
            eps2=1e-30,
            cliping_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=True,
            relative_step=True
        )

        self.assertIsNone(optimizer.param_groups[0]['lr'])
        self.assertTrue(optimizer.param_groups[0]['scale_parameter'])
        self.assertTrue(optimizer.param_groups[0]['relative_step'])

    def test_adafactor_factorization(self):
        """Test AdaFactor second moment factorization"""
        # Create 2D parameter for factorization testing
        large_param = nn.Parameter(torch.randn(100, 50, device=self.device))
        optimizer = AdaFactorOptimizer([large_param], relative_step=True)

        # Simulate gradient
        large_param.grad = torch.randn_like(large_param)
        optimizer.step()

        state = optimizer.state[large_param]

        # Should have factorized second moments for 2D tensors
        if len(large_param.shape) >= 2:
            self.assertIn('exp_avg_sq_row', state)
            self.assertIn('exp_avg_sq_col', state)

    def test_adafactor_relative_step(self):
        """Test AdaFactor relative step size computation"""
        optimizer = AdaFactorOptimizer(
            self.model.parameters(),
            relative_step=True,
            scale_parameter=True
        )

        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)

        # Multiple steps to test step size adaptation
        for i in range(5):
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            # Capture learning rate before step
            lr_before = optimizer._get_lr(optimizer.param_groups[0], optimizer.state[next(iter(self.model.parameters()))])

            optimizer.step()
            optimizer.zero_grad()

            # Learning rate should be automatically determined
            self.assertGreater(lr_before, 0)


class TestOptimizerFactory(unittest.TestCase):
    """Test OptimizerFactory and OptimizerConfig"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)

    def test_factory_lion_creation(self):
        """Test factory creation of Lion optimizer"""
        config = OptimizerConfig(
            name="lion",
            lr=1e-4,
            weight_decay=0.01,
            lion_betas=(0.9, 0.99)
        )

        optimizer = OptimizerFactory.create_optimizer(config, self.model.parameters())
        self.assertIsInstance(optimizer, LionOptimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)

    def test_factory_sophia_creation(self):
        """Test factory creation of Sophia optimizer"""
        config = OptimizerConfig(
            name="sophia",
            lr=1e-4,
            sophia_betas=(0.965, 0.99),
            sophia_rho=0.04
        )

        optimizer = OptimizerFactory.create_optimizer(config, self.model.parameters())
        self.assertIsInstance(optimizer, SophiaOptimizer)
        self.assertEqual(optimizer.param_groups[0]['rho'], 0.04)

    def test_factory_adafactor_creation(self):
        """Test factory creation of AdaFactor optimizer"""
        config = OptimizerConfig(
            name="adafactor",
            adafactor_relative_step=True,
            adafactor_scale_parameter=True,
            adafactor_warmup_init=False
        )

        optimizer = OptimizerFactory.create_optimizer(config, self.model.parameters())
        self.assertIsInstance(optimizer, AdaFactorOptimizer)
        self.assertTrue(optimizer.param_groups[0]['relative_step'])

    def test_factory_invalid_optimizer(self):
        """Test factory error handling for invalid optimizer"""
        config = OptimizerConfig(name="invalid_optimizer")

        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer(config, self.model.parameters())

    def test_config_validation(self):
        """Test OptimizerConfig validation"""
        # Valid config
        config = OptimizerConfig(name="lion", lr=1e-4)
        self.assertEqual(config.name, "lion")
        self.assertEqual(config.lr, 1e-4)

        # Config with all parameters
        full_config = OptimizerConfig(
            name="sophia",
            lr=1e-3,
            weight_decay=0.01,
            sophia_betas=(0.9, 0.95),
            sophia_rho=0.05,
            adafactor_relative_step=False
        )
        self.assertEqual(full_config.sophia_betas, (0.9, 0.95))


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for all optimizers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_optimizer_comparison(self):
        """Compare convergence of different optimizers on same problem"""
        torch.manual_seed(TEST_CONFIG["seed"])

        # Create simple regression problem
        true_weight = torch.tensor([[2.0], [3.0]], device=self.device)
        true_bias = torch.tensor([1.0], device=self.device)

        def create_model():
            model = nn.Linear(2, 1).to(self.device)
            nn.init.normal_(model.weight, 0, 0.1)
            nn.init.normal_(model.bias, 0, 0.1)
            return model

        def train_model(optimizer_class, optimizer_kwargs, steps=100):
            model = create_model()
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

            losses = []
            for _ in range(steps):
                x = torch.randn(32, 2, device=self.device)
                y = x @ true_weight + true_bias + 0.1 * torch.randn(32, 1, device=self.device)

                pred = model(x)
                loss = nn.MSELoss()(pred, y)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return losses, model

        # Test all optimizers
        results = {}

        # Lion
        losses_lion, model_lion = train_model(LionOptimizer, {'lr': 1e-3})
        results['lion'] = losses_lion

        # Sophia
        losses_sophia, model_sophia = train_model(SophiaOptimizer, {'lr': 1e-3})
        results['sophia'] = losses_sophia

        # AdaFactor
        losses_adafactor, model_adafactor = train_model(AdaFactorOptimizer, {'relative_step': True})
        results['adafactor'] = losses_adafactor

        # All optimizers should converge (loss should decrease)
        for name, losses in results.items():
            with self.subTest(optimizer=name):
                self.assertLess(losses[-1], losses[0], f"{name} should converge")
                self.assertLess(losses[-1], 1.0, f"{name} should reach reasonable loss")

    def test_memory_usage_comparison(self):
        """Compare memory usage of different optimizers"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Create larger model for meaningful memory comparison
        model = nn.Sequential(
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        ).to(self.device)

        def measure_optimizer_memory(optimizer_class, optimizer_kwargs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

            # Run a few steps to initialize optimizer state
            for _ in range(3):
                x = torch.randn(64, 500, device=self.device)
                y = torch.randn(64, 1, device=self.device)

                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            return torch.cuda.max_memory_allocated()

        # Measure memory for each optimizer
        memory_usage = {}

        memory_usage['lion'] = measure_optimizer_memory(LionOptimizer, {'lr': 1e-3})
        memory_usage['sophia'] = measure_optimizer_memory(SophiaOptimizer, {'lr': 1e-3})
        memory_usage['adafactor'] = measure_optimizer_memory(AdaFactorOptimizer, {'relative_step': True})

        # AdaFactor should use less memory than others due to factorization
        self.assertLess(
            memory_usage['adafactor'],
            max(memory_usage['lion'], memory_usage['sophia']),
            "AdaFactor should be more memory efficient"
        )


if __name__ == '__main__':
    unittest.main()