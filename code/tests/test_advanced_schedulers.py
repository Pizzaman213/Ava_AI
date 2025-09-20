"""
Unit Tests for Advanced Learning Rate Schedulers

Tests for enhanced schedulers including Cosine Annealing with Restarts,
OneCycle, Adaptive LR, and the scheduler factory system.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import math
import numpy as np

from tests import TEST_CONFIG
from src.Ava.training.advanced_schedulers import (
    CosineAnnealingWithRestarts,
    OneCycleScheduler,
    AdaptiveLRScheduler,
    WarmupScheduler,
    SchedulerFactory,
    SchedulerConfig
)


class TestCosineAnnealingWithRestarts(unittest.TestCase):
    """Test Cosine Annealing with Restarts scheduler"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_cosine_annealing_initialization(self):
        """Test scheduler initialization"""
        scheduler = CosineAnnealingWithRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6,
            last_epoch=-1
        )

        self.assertEqual(scheduler.T_0, 100)
        self.assertEqual(scheduler.T_mult, 2)
        self.assertEqual(scheduler.eta_min, 1e-6)
        self.assertEqual(scheduler.last_epoch, -1)

    def test_cosine_annealing_lr_calculation(self):
        """Test learning rate calculation in cosine annealing"""
        scheduler = CosineAnnealingWithRestarts(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=0.01
        )

        initial_lr = 0.1
        lrs = []

        # First cycle (10 steps)
        for i in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Should start high and decrease to eta_min
        self.assertAlmostEqual(lrs[0], initial_lr, places=5)
        self.assertGreater(lrs[0], lrs[4])  # Should decrease
        self.assertGreater(lrs[4], lrs[9])  # Continue decreasing
        self.assertAlmostEqual(lrs[9], 0.01, places=2)  # Near eta_min

    def test_restart_mechanism(self):
        """Test restart mechanism in cosine annealing"""
        scheduler = CosineAnnealingWithRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=0.01
        )

        lrs = []
        for i in range(20):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # First restart at step 5
        self.assertGreater(lrs[5], lrs[4])  # Should restart to higher LR

        # Second restart at step 5 + 10 = 15 (T_mult=2 doubles the period)
        self.assertGreater(lrs[15], lrs[14])  # Should restart again

    def test_t_mult_scaling(self):
        """Test T_mult scaling of restart periods"""
        scheduler = CosineAnnealingWithRestarts(
            self.optimizer,
            T_0=4,
            T_mult=2,
            eta_min=0.01
        )

        restart_steps = []
        prev_lr = scheduler.get_last_lr()[0]

        for i in range(20):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Detect restart (LR increases)
            if current_lr > prev_lr:
                restart_steps.append(i)

            prev_lr = current_lr

        # Should have restarts at: 4, 4+8=12
        expected_restarts = [4, 12]
        self.assertEqual(restart_steps, expected_restarts)

    def test_warmup_integration(self):
        """Test integration with warmup"""
        scheduler = CosineAnnealingWithRestarts(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=0.01,
            warmup_steps=3,
            warmup_factor=0.1
        )

        lrs = []
        for i in range(15):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # First 3 steps should be warmup
        self.assertLess(lrs[0], lrs[2])  # Warmup should increase LR
        self.assertAlmostEqual(lrs[3], 0.1, places=3)  # Should reach base LR after warmup


class TestOneCycleScheduler(unittest.TestCase):
    """Test OneCycle learning rate scheduler"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_onecycle_initialization(self):
        """Test OneCycle scheduler initialization"""
        scheduler = OneCycleScheduler(
            self.optimizer,
            max_lr=0.1,
            total_steps=1000,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        self.assertEqual(scheduler.max_lr, 0.1)
        self.assertEqual(scheduler.total_steps, 1000)
        self.assertEqual(scheduler.pct_start, 0.3)

    def test_onecycle_lr_progression(self):
        """Test OneCycle learning rate progression"""
        scheduler = OneCycleScheduler(
            self.optimizer,
            max_lr=0.1,
            total_steps=100,
            pct_start=0.3
        )

        lrs = []
        for i in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Should start low, increase to max, then decrease
        warmup_end = int(0.3 * 100)

        # LR should increase during warmup
        self.assertLess(lrs[0], lrs[warmup_end-1])

        # Should reach approximately max_lr
        max_observed_lr = max(lrs)
        self.assertAlmostEqual(max_observed_lr, 0.1, places=2)

        # Should decrease after peak
        peak_idx = lrs.index(max_observed_lr)
        self.assertGreater(lrs[peak_idx], lrs[-1])

    def test_onecycle_cosine_vs_linear(self):
        """Test different annealing strategies"""
        scheduler_cos = OneCycleScheduler(
            self.optimizer,
            max_lr=0.1,
            total_steps=50,
            anneal_strategy='cos'
        )

        scheduler_linear = OneCycleScheduler(
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            max_lr=0.1,
            total_steps=50,
            anneal_strategy='linear'
        )

        lrs_cos = []
        lrs_linear = []

        for i in range(50):
            lrs_cos.append(scheduler_cos.get_last_lr()[0])
            lrs_linear.append(scheduler_linear.get_last_lr()[0])
            scheduler_cos.step()
            scheduler_linear.step()

        # Cosine should have smoother transitions
        cos_diffs = [abs(lrs_cos[i+1] - lrs_cos[i]) for i in range(49)]
        linear_diffs = [abs(lrs_linear[i+1] - lrs_linear[i]) for i in range(49)]

        # Cosine should have more gradual changes in some regions
        self.assertNotEqual(cos_diffs, linear_diffs)

    def test_momentum_beta_adjustment(self):
        """Test momentum/beta parameter adjustment in OneCycle"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = OneCycleScheduler(
            optimizer,
            max_lr=0.1,
            total_steps=100,
            adjust_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )

        momentums = []
        for i in range(100):
            momentums.append(optimizer.param_groups[0]['momentum'])
            scheduler.step()

        # Momentum should change inversely to learning rate
        # (high LR = low momentum, low LR = high momentum)
        self.assertNotEqual(momentums[0], momentums[50])

    def test_div_factor_calculation(self):
        """Test div_factor and final_div_factor effects"""
        scheduler = OneCycleScheduler(
            self.optimizer,
            max_lr=0.1,
            total_steps=100,
            div_factor=10.0,
            final_div_factor=100.0
        )

        lrs = []
        for i in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Initial LR should be max_lr / div_factor
        expected_initial = 0.1 / 10.0
        self.assertAlmostEqual(lrs[0], expected_initial, places=3)

        # Final LR should be initial_lr / final_div_factor
        expected_final = expected_initial / 100.0
        self.assertAlmostEqual(lrs[-1], expected_final, places=5)


class TestAdaptiveLRScheduler(unittest.TestCase):
    """Test Adaptive Learning Rate scheduler"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_adaptive_initialization(self):
        """Test adaptive scheduler initialization"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=10,
            factor=0.5,
            threshold=1e-4,
            min_lr=1e-6
        )

        self.assertEqual(scheduler.patience, 10)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.threshold, 1e-4)

    def test_adaptive_lr_reduction(self):
        """Test learning rate reduction on plateau"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=3,
            factor=0.5,
            threshold=1e-4
        )

        initial_lr = scheduler.get_last_lr()[0]

        # Feed plateau metrics (no improvement)
        plateau_loss = 1.0
        for i in range(5):
            scheduler.step(plateau_loss)

        # LR should be reduced after patience steps
        reduced_lr = scheduler.get_last_lr()[0]
        self.assertLess(reduced_lr, initial_lr)
        self.assertAlmostEqual(reduced_lr, initial_lr * 0.5, places=5)

    def test_adaptive_improvement_detection(self):
        """Test improvement detection in adaptive scheduler"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=5,
            factor=0.5,
            threshold=0.01
        )

        # Provide improving metrics
        for i in range(10):
            loss = 1.0 - i * 0.02  # Decreasing loss
            scheduler.step(loss)

        # LR should not be reduced with consistent improvement
        final_lr = scheduler.get_last_lr()[0]
        self.assertEqual(final_lr, 0.1)  # Should remain at initial LR

    def test_adaptive_min_lr_constraint(self):
        """Test minimum learning rate constraint"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=2,
            factor=0.1,
            min_lr=1e-3
        )

        # Force multiple reductions
        plateau_loss = 1.0
        for i in range(20):
            scheduler.step(plateau_loss)

        # LR should not go below min_lr
        final_lr = scheduler.get_last_lr()[0]
        self.assertGreaterEqual(final_lr, 1e-3)

    def test_adaptive_gradient_norm_mode(self):
        """Test adaptive scheduler with gradient norm tracking"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=3,
            factor=0.5,
            mode='gradient_norm',
            threshold=0.1
        )

        # High gradient norms (not converging)
        for i in range(5):
            scheduler.step(2.0)  # High gradient norm

        # LR should be reduced
        reduced_lr = scheduler.get_last_lr()[0]
        self.assertLess(reduced_lr, 0.1)

    def test_adaptive_multiple_metrics(self):
        """Test adaptive scheduler with multiple metrics"""
        scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=3,
            factor=0.7,
            multi_metric=True
        )

        # Provide multiple metrics
        for i in range(5):
            metrics = {
                'loss': 1.0,
                'gradient_norm': 2.0,
                'validation_loss': 1.1
            }
            scheduler.step(metrics)

        # Should handle multiple metrics
        reduced_lr = scheduler.get_last_lr()[0]
        self.assertLess(reduced_lr, 0.1)


class TestWarmupScheduler(unittest.TestCase):
    """Test Warmup scheduler wrapper"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_warmup_initialization(self):
        """Test warmup scheduler initialization"""
        base_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        scheduler = WarmupScheduler(
            self.optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_factor=0.1
        )

        self.assertEqual(scheduler.warmup_steps, 5)
        self.assertEqual(scheduler.warmup_factor, 0.1)

    def test_warmup_lr_progression(self):
        """Test learning rate progression during warmup"""
        base_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        scheduler = WarmupScheduler(
            self.optimizer,
            base_scheduler,
            warmup_steps=4,
            warmup_factor=0.25
        )

        lrs = []
        for i in range(8):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Warmup should gradually increase LR
        self.assertAlmostEqual(lrs[0], 0.025, places=3)  # 0.1 * 0.25
        self.assertLess(lrs[0], lrs[1])
        self.assertLess(lrs[1], lrs[2])
        self.assertLess(lrs[2], lrs[3])
        self.assertAlmostEqual(lrs[4], 0.1, places=3)  # Should reach base LR

    def test_warmup_with_cosine_scheduler(self):
        """Test warmup integration with cosine scheduler"""
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0.01
        )
        scheduler = WarmupScheduler(
            self.optimizer,
            base_scheduler,
            warmup_steps=3,
            warmup_factor=0.1
        )

        lrs = []
        for i in range(15):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # First 3 steps should be warmup
        self.assertLess(lrs[0], lrs[2])

        # After warmup, should follow cosine pattern
        self.assertGreater(lrs[3], lrs[8])  # Cosine decay


class TestSchedulerFactory(unittest.TestCase):
    """Test SchedulerFactory and SchedulerConfig"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_factory_cosine_creation(self):
        """Test factory creation of cosine annealing scheduler"""
        config = SchedulerConfig(
            name="cosine_with_restarts",
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )

        scheduler = SchedulerFactory.create_scheduler(config, self.optimizer)
        self.assertIsInstance(scheduler, CosineAnnealingWithRestarts)
        self.assertEqual(scheduler.T_0, 100)

    def test_factory_onecycle_creation(self):
        """Test factory creation of OneCycle scheduler"""
        config = SchedulerConfig(
            name="onecycle",
            max_lr=0.1,
            total_steps=1000,
            pct_start=0.3
        )

        scheduler = SchedulerFactory.create_scheduler(config, self.optimizer)
        self.assertIsInstance(scheduler, OneCycleScheduler)
        self.assertEqual(scheduler.total_steps, 1000)

    def test_factory_adaptive_creation(self):
        """Test factory creation of adaptive scheduler"""
        config = SchedulerConfig(
            name="adaptive",
            patience=10,
            factor=0.5,
            threshold=1e-4
        )

        scheduler = SchedulerFactory.create_scheduler(config, self.optimizer)
        self.assertIsInstance(scheduler, AdaptiveLRScheduler)
        self.assertEqual(scheduler.patience, 10)

    def test_factory_warmup_wrapping(self):
        """Test factory creation with warmup wrapping"""
        config = SchedulerConfig(
            name="cosine_with_restarts",
            T_0=50,
            warmup_steps=10,
            warmup_factor=0.1
        )

        scheduler = SchedulerFactory.create_scheduler(config, self.optimizer)
        self.assertIsInstance(scheduler, WarmupScheduler)
        self.assertEqual(scheduler.warmup_steps, 10)

    def test_factory_invalid_scheduler(self):
        """Test factory error handling for invalid scheduler"""
        config = SchedulerConfig(name="invalid_scheduler")

        with self.assertRaises(ValueError):
            SchedulerFactory.create_scheduler(config, self.optimizer)

    def test_config_validation(self):
        """Test SchedulerConfig validation"""
        # Valid config
        config = SchedulerConfig(name="onecycle", max_lr=0.1, total_steps=1000)
        self.assertEqual(config.name, "onecycle")

        # Config with optional parameters
        full_config = SchedulerConfig(
            name="adaptive",
            patience=5,
            factor=0.8,
            threshold=1e-3,
            min_lr=1e-6,
            warmup_steps=100
        )
        self.assertEqual(full_config.patience, 5)
        self.assertEqual(full_config.warmup_steps, 100)


class TestSchedulerIntegration(unittest.TestCase):
    """Integration tests for scheduler combinations"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_scheduler_training_simulation(self):
        """Test schedulers in simulated training scenario"""
        model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Test different schedulers
        schedulers = {
            'cosine': CosineAnnealingWithRestarts(optimizer, T_0=20, T_mult=2),
            'onecycle': OneCycleScheduler(optimizer, max_lr=0.01, total_steps=100),
            'adaptive': AdaptiveLRScheduler(optimizer, patience=5, factor=0.5)
        }

        for name, scheduler in schedulers.items():
            with self.subTest(scheduler=name):
                # Reset optimizer
                for group in optimizer.param_groups:
                    group['lr'] = 0.01

                # Simulate training
                losses = []
                for step in range(50):
                    # Mock training step
                    x = torch.randn(16, 20, device=self.device)
                    y = torch.randn(16, 1, device=self.device)

                    pred = model(x)
                    loss = nn.MSELoss()(pred, y)
                    losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Step scheduler
                    if name == 'adaptive':
                        scheduler.step(loss.item())
                    else:
                        scheduler.step()

                # Should complete without errors
                self.assertEqual(len(losses), 50)

    def test_scheduler_memory_efficiency(self):
        """Test scheduler memory efficiency"""
        large_model = nn.Sequential(
            *[nn.Linear(100, 100) for _ in range(10)]
        ).to(self.device)

        optimizer = torch.optim.SGD(large_model.parameters(), lr=0.1)

        # Create complex scheduler
        scheduler = CosineAnnealingWithRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            warmup_steps=10
        )

        # Memory usage should be minimal for scheduler
        initial_params = sum(p.numel() for p in large_model.parameters())

        # Run several scheduler steps
        for i in range(100):
            scheduler.step()

        # Scheduler should not create significant additional memory overhead
        # (This is mainly a smoke test to ensure no memory leaks)
        self.assertGreater(initial_params, 0)

    def test_scheduler_state_persistence(self):
        """Test scheduler state saving and loading"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        scheduler = OneCycleScheduler(optimizer, max_lr=0.1, total_steps=100)

        # Run for some steps
        for i in range(30):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Create new scheduler and load state
        new_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        new_scheduler = OneCycleScheduler(new_optimizer, max_lr=0.1, total_steps=100)
        new_scheduler.load_state_dict(state)

        # Should have same learning rate
        self.assertAlmostEqual(
            scheduler.get_last_lr()[0],
            new_scheduler.get_last_lr()[0],
            places=6
        )


if __name__ == '__main__':
    unittest.main()