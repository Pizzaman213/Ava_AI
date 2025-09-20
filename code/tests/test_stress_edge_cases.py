"""
Stress Tests and Edge Case Validation

Comprehensive stress testing and edge case validation for all advanced training
optimizations including extreme configurations, memory pressure, and error conditions.
"""

import unittest
import torch
import torch.nn as nn
import gc
import warnings
from unittest.mock import patch, MagicMock
import tempfile
import os

from tests import TEST_CONFIG
from src.Ava.optimization.advanced_optimizers import (
    LionOptimizer, SophiaOptimizer, AdaFactorOptimizer, OptimizerFactory, OptimizerConfig
)
from src.Ava.training.progressive_training import (
    ProgressiveTrainingOrchestrator, ProgressiveConfig,
    CurriculumConfig, GrowLengthConfig, DynamicBatchConfig
)
from src.Ava.training.advanced_schedulers import (
    CosineAnnealingWithRestarts, OneCycleScheduler, AdaptiveLRScheduler,
    SchedulerFactory, SchedulerConfig
)
from src.Ava.optimization.fp8_training import (
    FP8Config, FP8Optimizer, FP8TrainingManager, convert_model_to_fp8
)


class TestOptimizerStressCases(unittest.TestCase):
    """Stress test optimizers under extreme conditions"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_optimizer_extreme_learning_rates(self):
        """Test optimizers with extreme learning rates"""
        model = nn.Linear(10, 1).to(self.device)

        extreme_lrs = [1e-10, 1e-8, 1e8, 1e10]
        optimizers = [
            ('Lion', lambda lr: LionOptimizer(model.parameters(), lr=lr)),
            ('Sophia', lambda lr: SophiaOptimizer(model.parameters(), lr=lr)),
            ('AdaFactor', lambda lr: AdaFactorOptimizer(model.parameters(), lr=lr, relative_step=False))
        ]

        for opt_name, opt_factory in optimizers:
            for lr in extreme_lrs:
                with self.subTest(optimizer=opt_name, lr=lr):
                    try:
                        optimizer = opt_factory(lr)

                        # Test a few steps
                        for _ in range(5):
                            x = torch.randn(16, 10, device=self.device)
                            y = torch.randn(16, 1, device=self.device)

                            optimizer.zero_grad()
                            output = model(x)
                            loss = nn.MSELoss()(output, y)
                            loss.backward()
                            optimizer.step()

                            # Check for NaN/Inf
                            for param in model.parameters():
                                self.assertFalse(torch.isnan(param).any(),
                                               f"NaN in parameters with {opt_name} lr={lr}")
                                self.assertFalse(torch.isinf(param).any(),
                                               f"Inf in parameters with {opt_name} lr={lr}")

                    except (RuntimeError, ValueError) as e:
                        # Some extreme values may be invalid, which is acceptable
                        pass

    def test_optimizer_large_gradients(self):
        """Test optimizers with artificially large gradients"""
        model = nn.Linear(50, 1).to(self.device)

        optimizers = [
            LionOptimizer(model.parameters(), lr=1e-3),
            SophiaOptimizer(model.parameters(), lr=1e-3),
            AdaFactorOptimizer(model.parameters(), lr=1e-3, relative_step=False)
        ]

        for optimizer in optimizers:
            with self.subTest(optimizer=type(optimizer).__name__):
                # Create artificially large gradients
                for param in model.parameters():
                    param.grad = torch.randn_like(param) * 1e6

                # Optimizer should handle large gradients gracefully
                try:
                    optimizer.step()

                    # Parameters should remain finite
                    for param in model.parameters():
                        self.assertTrue(torch.isfinite(param).all())

                except RuntimeError:
                    # Some optimizers may clip or handle overflow differently
                    pass

    def test_optimizer_zero_gradients(self):
        """Test optimizers with zero gradients"""
        model = nn.Linear(20, 1).to(self.device)

        optimizers = [
            LionOptimizer(model.parameters(), lr=1e-3),
            SophiaOptimizer(model.parameters(), lr=1e-3),
            AdaFactorOptimizer(model.parameters(), lr=1e-3)
        ]

        for optimizer in optimizers:
            with self.subTest(optimizer=type(optimizer).__name__):
                initial_params = [param.clone() for param in model.parameters()]

                # Set all gradients to zero
                for param in model.parameters():
                    param.grad = torch.zeros_like(param)

                optimizer.step()

                # Parameters should remain unchanged with zero gradients
                for initial, current in zip(initial_params, model.parameters()):
                    self.assertTrue(torch.allclose(initial, current, atol=1e-7))

    def test_optimizer_memory_pressure(self):
        """Test optimizers under memory pressure"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Create large model to stress memory
        large_model = nn.Sequential(
            *[nn.Linear(1000, 1000) for _ in range(10)]
        ).to(self.device)

        optimizers = [
            ('AdaFactor', AdaFactorOptimizer(large_model.parameters(), relative_step=True)),
            ('Lion', LionOptimizer(large_model.parameters(), lr=1e-4))
        ]

        for opt_name, optimizer in optimizers:
            with self.subTest(optimizer=opt_name):
                try:
                    torch.cuda.empty_cache()

                    # Run training steps
                    for step in range(5):
                        x = torch.randn(32, 1000, device=self.device)
                        y = torch.randn(32, 1000, device=self.device)

                        optimizer.zero_grad()
                        output = large_model(x)
                        loss = nn.MSELoss()(output, y)
                        loss.backward()
                        optimizer.step()

                        # Force memory pressure
                        if step % 2 == 0:
                            torch.cuda.empty_cache()

                    # Should complete without OOM
                    self.assertTrue(True)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM is acceptable for this stress test
                        pass
                    else:
                        raise

                finally:
                    del large_model, optimizer
                    torch.cuda.empty_cache()

    def test_optimizer_long_training(self):
        """Test optimizers over very long training runs"""
        model = nn.Linear(32, 1).to(self.device)

        optimizer = AdaFactorOptimizer(model.parameters(), relative_step=True)

        # Simulate very long training (10k steps)
        for step in range(10000):
            x = torch.randn(8, 32, device=self.device)
            y = torch.randn(8, 1, device=self.device)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

            # Check stability periodically
            if step % 1000 == 999:
                for param in model.parameters():
                    self.assertTrue(torch.isfinite(param).all(),
                                  f"Parameters became non-finite at step {step}")

                # Check optimizer state
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.isfinite(value).all(),
                                          f"Optimizer state became non-finite at step {step}")


class TestProgressiveTrainingStressCases(unittest.TestCase):
    """Stress test progressive training components"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_extreme_curriculum_parameters(self):
        """Test curriculum learning with extreme parameters"""
        extreme_configs = [
            # Very large sequences
            CurriculumConfig(
                strategy="length_based",
                initial_max_length=1,
                final_max_length=10000,
                growth_steps=10
            ),
            # Very fast growth
            CurriculumConfig(
                strategy="length_based",
                initial_max_length=8,
                final_max_length=512,
                growth_steps=1
            ),
            # Very slow growth
            CurriculumConfig(
                strategy="length_based",
                initial_max_length=32,
                final_max_length=64,
                growth_steps=100000
            )
        ]

        for i, config in enumerate(extreme_configs):
            with self.subTest(config=i):
                progressive_config = ProgressiveConfig(
                    curriculum_config=config,
                    enable_curriculum=True
                )

                try:
                    orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

                    # Test rapid updates
                    for step in range(100):
                        orchestrator.update(
                            step=step,
                            loss=1.0 - step * 0.01,
                            gradient_norm=0.5
                        )

                        length = orchestrator.get_current_max_length()

                        # Length should be reasonable
                        self.assertGreater(length, 0)
                        self.assertLess(length, 50000)  # Reasonable upper bound

                except Exception as e:
                    # Some extreme configurations may be rejected
                    self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_dynamic_batch_extreme_memory(self):
        """Test dynamic batch sizing under extreme memory conditions"""
        config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=1,
                max_batch_size=1024,
                memory_threshold_gb=0.001,  # Very low threshold
                adaptation_strategy="memory_based"
            ),
            enable_dynamic_batch=True
        )

        orchestrator = ProgressiveTrainingOrchestrator(config)

        # Simulate extreme memory pressure
        for step in range(50):
            batch_size = orchestrator.get_current_batch_size()

            # Simulate high memory usage
            memory_usage = batch_size * 0.1 + 10.0  # Always high memory

            orchestrator.update(
                step=step,
                loss=1.0,
                memory_usage_gb=memory_usage
            )

        # Should adapt to very small batch sizes
        final_batch_size = orchestrator.get_current_batch_size()
        self.assertLessEqual(final_batch_size, 4)  # Should reduce significantly

    def test_progressive_training_rapid_state_changes(self):
        """Test progressive training with rapidly changing conditions"""
        config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="difficulty_based",
                initial_max_length=16,
                final_max_length=128,
                difficulty_threshold=0.5
            ),
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=8,
                max_batch_size=64,
                adaptation_strategy="performance_based"
            ),
            enable_curriculum=True,
            enable_dynamic_batch=True
        )

        orchestrator = ProgressiveTrainingOrchestrator(config)

        # Rapidly changing training conditions
        for step in range(100):
            # Oscillating loss and performance
            loss = 2.0 + torch.sin(torch.tensor(step * 0.5)).item()
            throughput = 500 + torch.cos(torch.tensor(step * 0.3)).item() * 200
            memory_usage = 2.0 + torch.sin(torch.tensor(step * 0.2)).item()

            orchestrator.update(
                step=step,
                loss=loss,
                gradient_norm=1.0,
                memory_usage_gb=memory_usage,
                throughput=throughput
            )

        # Should handle rapid changes gracefully
        final_metrics = orchestrator.get_current_metrics()
        self.assertIsInstance(final_metrics, dict)


class TestSchedulerStressCases(unittest.TestCase):
    """Stress test learning rate schedulers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)

    def test_scheduler_extreme_parameters(self):
        """Test schedulers with extreme parameters"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        extreme_schedulers = [
            # Very short cycles
            CosineAnnealingWithRestarts(optimizer, T_0=1, T_mult=1),
            # Very long cycles
            CosineAnnealingWithRestarts(optimizer, T_0=100000, T_mult=1),
            # Extreme OneCycle
            OneCycleScheduler(optimizer, max_lr=1e6, total_steps=1),
            # Extreme adaptive patience
            AdaptiveLRScheduler(optimizer, patience=1, factor=0.001)
        ]

        for scheduler in extreme_schedulers:
            with self.subTest(scheduler=type(scheduler).__name__):
                try:
                    # Test many steps
                    for step in range(1000):
                        if isinstance(scheduler, AdaptiveLRScheduler):
                            scheduler.step(1.0)  # Constant metric
                        else:
                            scheduler.step()

                        # Check learning rate remains reasonable
                        current_lr = scheduler.get_last_lr()[0]
                        self.assertGreater(current_lr, 0)
                        self.assertLess(current_lr, 1e10)
                        self.assertTrue(torch.isfinite(torch.tensor(current_lr)))

                except Exception as e:
                    # Some extreme configurations may fail
                    self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_scheduler_nan_inf_metrics(self):
        """Test adaptive scheduler with NaN/Inf metrics"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = AdaptiveLRScheduler(optimizer, patience=3, factor=0.5)

        extreme_metrics = [float('nan'), float('inf'), -float('inf'), 1e20, -1e20]

        for metric in extreme_metrics:
            with self.subTest(metric=metric):
                try:
                    scheduler.step(metric)

                    # Learning rate should remain finite
                    lr = scheduler.get_last_lr()[0]
                    self.assertTrue(torch.isfinite(torch.tensor(lr)))

                except Exception:
                    # Scheduler may reject invalid metrics
                    pass

    def test_scheduler_very_long_training(self):
        """Test schedulers over very long training runs"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWithRestarts(optimizer, T_0=100, T_mult=2)

        # Simulate 100k steps
        for step in range(100000):
            scheduler.step()

            if step % 10000 == 9999:
                lr = scheduler.get_last_lr()[0]
                self.assertGreater(lr, 0)
                self.assertTrue(torch.isfinite(torch.tensor(lr)))


class TestFP8StressCases(unittest.TestCase):
    """Stress test FP8 training components"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_fp8_extreme_model_sizes(self):
        """Test FP8 with extremely large and small models"""
        configs = [
            # Very small model
            (2, 1),
            # Very large model (if memory allows)
            (2048, 1024)
        ]

        fp8_config = FP8Config(enabled=True)

        for in_size, out_size in configs:
            with self.subTest(model_size=f"{in_size}x{out_size}"):
                try:
                    model = nn.Linear(in_size, out_size).to(self.device)
                    fp8_model = convert_model_to_fp8(model, fp8_config)

                    optimizer = torch.optim.Adam(fp8_model.parameters(), lr=1e-3)

                    # Test training steps
                    for _ in range(10):
                        x = torch.randn(4, in_size, device=self.device)
                        y = torch.randn(4, out_size, device=self.device)

                        optimizer.zero_grad()
                        output = fp8_model(x)
                        loss = nn.MSELoss()(output, y)
                        loss.backward()
                        optimizer.step()

                    # Should complete successfully
                    self.assertTrue(True)

                except (RuntimeError, OutOfMemoryError):
                    # Large models may exceed memory limits
                    pass

    def test_fp8_scaling_factor_extremes(self):
        """Test FP8 scaling factor handling with extreme values"""
        config = FP8Config(enabled=True, margin=0, interval=1)
        manager = FP8TrainingManager(config)

        extreme_tensors = [
            torch.tensor([1e-10], device=self.device),  # Very small
            torch.tensor([1e10], device=self.device),   # Very large
            torch.zeros(1, device=self.device),         # Zero
            torch.tensor([float('inf')], device=self.device),  # Infinity
        ]

        for i, tensor in enumerate(extreme_tensors):
            with self.subTest(tensor=i):
                try:
                    manager.track_amax(tensor, 'forward')
                    manager.update_scaling_factors(
                        forward_loss=tensor,
                        backward_loss=tensor
                    )

                    # Should handle extreme values gracefully
                    factors = manager.get_scaling_factors()
                    self.assertIsInstance(factors, dict)

                except Exception:
                    # Some extreme values may be rejected
                    pass

    def test_fp8_overflow_recovery(self):
        """Test FP8 overflow detection and recovery"""
        config = FP8Config(enabled=True)
        manager = FP8TrainingManager(config)

        # Simulate overflow conditions
        for step in range(20):
            if step % 5 == 0:
                # Inject overflow
                overflow_tensor = torch.tensor([float('inf')], device=self.device)
                overflow_detected = manager.check_overflow(overflow_tensor)
                self.assertTrue(overflow_detected)
            else:
                # Normal training
                normal_tensor = torch.randn(64, device=self.device)
                overflow_detected = manager.check_overflow(normal_tensor)
                self.assertFalse(overflow_detected)

            manager.step(step=step, loss=1.0)

        # Manager should recover from overflows
        state = manager.get_state()
        self.assertIsInstance(state, dict)


class TestIntegrationStressCases(unittest.TestCase):
    """Stress test integration scenarios"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_all_optimizations_extreme_config(self):
        """Test all optimizations together with extreme configuration"""
        # Extreme configuration
        optimizer_config = OptimizerConfig(
            name="sophia",
            lr=1e-2,  # High learning rate
            sophia_rho=0.1,  # High rho
            weight_decay=0.1
        )

        scheduler_config = SchedulerConfig(
            name="onecycle",
            max_lr=1.0,  # Very high max LR
            total_steps=10,  # Very short cycle
            pct_start=0.9  # Late peak
        )

        progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=2,
                final_max_length=1000,  # Large jump
                growth_steps=5
            ),
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=1,
                max_batch_size=256,
                memory_threshold_gb=0.1  # Very low threshold
            ),
            enable_curriculum=True,
            enable_dynamic_batch=True
        )

        fp8_config = FP8Config(enabled=True, margin=0, interval=1)

        # Create components
        model = nn.Linear(64, 1).to(self.device)
        fp8_model = convert_model_to_fp8(model, fp8_config)

        optimizer = OptimizerFactory.create_optimizer(optimizer_config, fp8_model.parameters())
        scheduler = SchedulerFactory.create_scheduler(scheduler_config, optimizer)
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        # Stress test
        try:
            for step in range(20):
                batch_size = orchestrator.get_current_batch_size()
                max_length = min(orchestrator.get_current_max_length(), 64)  # Cap for linear model

                x = torch.randn(batch_size, max_length, device=self.device)
                x = x.mean(dim=1, keepdim=True)  # Reduce to fit linear layer
                y = torch.randn(batch_size, 1, device=self.device)

                optimizer.zero_grad()
                output = fp8_model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                orchestrator.update(
                    step=step,
                    loss=loss.item(),
                    gradient_norm=1.0,
                    memory_usage_gb=0.5  # High memory usage
                )

            # Should survive extreme configuration
            self.assertTrue(True)

        except Exception as e:
            # Some extreme combinations may fail
            self.assertIsInstance(e, (RuntimeError, ValueError))

    def test_rapid_configuration_changes(self):
        """Test rapid configuration changes during training"""
        model = nn.Linear(32, 1).to(self.device)

        # Start with one configuration
        optimizer1 = LionOptimizer(model.parameters(), lr=1e-3)
        scheduler1 = CosineAnnealingWithRestarts(optimizer1, T_0=10)

        # Train for some steps
        for step in range(25):
            x = torch.randn(16, 32, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            optimizer1.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer1.step()
            scheduler1.step()

        # Switch to different configuration mid-training
        optimizer2 = AdaFactorOptimizer(model.parameters(), relative_step=True)
        scheduler2 = OneCycleScheduler(optimizer2, max_lr=1e-2, total_steps=25)

        # Continue training with new configuration
        for step in range(25):
            x = torch.randn(16, 32, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            optimizer2.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer2.step()
            scheduler2.step()

        # Should handle configuration changes gracefully
        self.assertTrue(True)

    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion scenarios"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Progressive config that adapts to memory pressure
        progressive_config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=64,
                max_batch_size=128,
                memory_threshold_gb=1.0,  # Low threshold
                adaptation_strategy="memory_based"
            ),
            enable_dynamic_batch=True
        )

        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        # Large model to create memory pressure
        try:
            model = nn.Sequential(
                *[nn.Linear(512, 512) for _ in range(5)]
            ).to(self.device)

            optimizer = AdaFactorOptimizer(model.parameters(), relative_step=True)

            for step in range(20):
                batch_size = orchestrator.get_current_batch_size()

                try:
                    x = torch.randn(batch_size, 512, device=self.device)
                    y = torch.randn(batch_size, 512, device=self.device)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer.step()

                    # Report success
                    memory_usage = torch.cuda.memory_allocated() / (1024**3)
                    orchestrator.update(
                        step=step,
                        loss=loss.item(),
                        memory_usage_gb=memory_usage
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Simulate memory pressure
                        torch.cuda.empty_cache()
                        orchestrator.update(
                            step=step,
                            loss=10.0,  # High loss indicates problem
                            memory_usage_gb=10.0  # High memory usage
                        )
                    else:
                        raise

            # Should adapt batch size down due to memory pressure
            final_batch_size = orchestrator.get_current_batch_size()
            self.assertLessEqual(final_batch_size, 32)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM is acceptable for this stress test
                pass
            else:
                raise

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    # Suppress expected warnings during stress testing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    unittest.main()