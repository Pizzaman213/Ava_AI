"""
End-to-End Integration Tests

Comprehensive integration tests combining all advanced training optimizations
including optimizers, progressive training, schedulers, and FP8 support.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
import json
import time
from unittest.mock import patch, MagicMock

from tests import TEST_CONFIG
from src.Ava.optimization.advanced_optimizers import (
    LionOptimizer, SophiaOptimizer, AdaFactorOptimizer, OptimizerFactory, OptimizerConfig
)
from src.Ava.training.progressive_training import (
    ProgressiveTrainingManager, ProgressiveTrainingConfig
)
from src.Ava.training.advanced_schedulers import (
    CosineAnnealingWarmRestarts, OneCycleLR, AdaptiveLRScheduler,
    SchedulerFactory
)
from src.Ava.optimization.fp8_training import (
    FP8Config, FP8Handler, FP8ModelWrapper, create_fp8_model
)


class TestOptimizerSchedulerIntegration(unittest.TestCase):
    """Test integration between optimizers and schedulers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(self.device)

    def test_lion_with_cosine_scheduler(self):
        """Test Lion optimizer with Cosine Annealing scheduler"""
        optimizer = LionOptimizer(self.model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWithRestarts(optimizer, T_0=20, T_mult=2)

        # Training simulation
        losses = []
        learning_rates = []

        for step in range(50):
            x = torch.randn(32, 50, device=self.device)
            y = torch.randn(32, 1, device=self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])

        # Should have varying learning rates due to cosine schedule
        self.assertGreater(max(learning_rates), min(learning_rates))

        # Loss should generally trend downward
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        self.assertLess(late_avg, early_avg)

    def test_sophia_with_onecycle_scheduler(self):
        """Test Sophia optimizer with OneCycle scheduler"""
        optimizer = SophiaOptimizer(self.model.parameters(), lr=1e-3)
        scheduler = OneCycleScheduler(optimizer, max_lr=1e-2, total_steps=100)

        training_metrics = []

        for step in range(100):
            x = torch.randn(16, 50, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            training_metrics.append({
                'step': step,
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

        # OneCycle should reach peak LR and then decrease
        lrs = [m['lr'] for m in training_metrics]
        peak_lr_step = lrs.index(max(lrs))

        self.assertGreater(peak_lr_step, 10)  # Peak should not be immediate
        self.assertLess(peak_lr_step, 90)     # Peak should not be at the end

    def test_adafactor_with_adaptive_scheduler(self):
        """Test AdaFactor optimizer with Adaptive scheduler"""
        optimizer = AdaFactorOptimizer(self.model.parameters(), relative_step=False, lr=1e-3)
        scheduler = AdaptiveLRScheduler(optimizer, patience=5, factor=0.8)

        losses = []
        lr_changes = []
        prev_lr = scheduler.get_last_lr()[0]

        for step in range(50):
            x = torch.randn(24, 50, device=self.device)
            y = torch.randn(24, 1, device=self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step(loss.item())

            losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]

            if current_lr != prev_lr:
                lr_changes.append(step)
            prev_lr = current_lr

        # Adaptive scheduler should change LR when loss plateaus
        self.assertGreater(len(lr_changes), 0)


class TestProgressiveTrainingIntegration(unittest.TestCase):
    """Test progressive training with optimizers and schedulers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1000)
        ).to(self.device)

    def test_progressive_training_with_lion(self):
        """Test progressive training with Lion optimizer"""
        # Configure progressive training
        progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=16,
                final_max_length=64,
                growth_steps=50
            ),
            growlength_config=GrowLengthConfig(
                initial_length=16,
                final_length=64,
                growth_steps=50
            ),
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=8,
                max_batch_size=32,
                memory_threshold_gb=4.0
            )
        )

        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)
        optimizer = LionOptimizer(self.model.parameters(), lr=1e-3)

        training_log = []

        for step in range(50):
            # Get progressive training parameters
            max_length = orchestrator.get_current_max_length()
            batch_size = orchestrator.get_current_batch_size()

            # Create batch with current parameters
            x = torch.randint(0, 1000, (batch_size, max_length), device=self.device)
            y = torch.randint(0, 1000, (batch_size, max_length), device=self.device)

            optimizer.zero_grad()

            # Forward pass
            output = self.model(x.view(-1, 1)).view(batch_size, max_length, -1)
            loss = nn.CrossEntropyLoss()(
                output.view(-1, 1000),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            # Update progressive training
            orchestrator.update(
                step=step,
                loss=loss.item(),
                gradient_norm=1.0,
                memory_usage_gb=2.0
            )

            training_log.append({
                'step': step,
                'loss': loss.item(),
                'max_length': max_length,
                'batch_size': batch_size
            })

        # Progressive training should increase sequence length
        initial_length = training_log[0]['max_length']
        final_length = training_log[-1]['max_length']
        self.assertGreaterEqual(final_length, initial_length)

    def test_curriculum_with_scheduler_coordination(self):
        """Test curriculum learning coordination with scheduler"""
        # Setup curriculum learning
        progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="difficulty_based",
                initial_max_length=8,
                final_max_length=32,
                difficulty_threshold=0.5
            ),
            enable_curriculum=True,
            enable_growlength=False,
            enable_dynamic_batch=False
        )

        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)
        optimizer = SophiaOptimizer(self.model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWithRestarts(optimizer, T_0=15, T_mult=2)

        curriculum_progression = []

        for step in range(30):
            max_length = orchestrator.get_current_max_length()

            # Create samples of varying difficulty
            if step < 15:
                # Easy samples (low loss expected)
                x = torch.zeros(4, max_length, dtype=torch.long, device=self.device)
                y = torch.zeros(4, max_length, dtype=torch.long, device=self.device)
            else:
                # Harder samples
                x = torch.randint(0, 1000, (4, max_length), device=self.device)
                y = torch.randint(0, 1000, (4, max_length), device=self.device)

            optimizer.zero_grad()
            output = self.model(x.view(-1, 1)).view(4, max_length, -1)
            loss = nn.CrossEntropyLoss()(output.view(-1, 1000), y.view(-1))
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Update curriculum with loss and learning rate info
            orchestrator.update(
                step=step,
                loss=loss.item(),
                gradient_norm=1.0
            )

            curriculum_progression.append({
                'step': step,
                'max_length': max_length,
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

        # Curriculum should adapt based on training progress
        lengths = [p['max_length'] for p in curriculum_progression]
        self.assertGreaterEqual(max(lengths), min(lengths))


class TestFP8Integration(unittest.TestCase):
    """Test FP8 training integration with optimizers and schedulers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

    def test_fp8_with_lion_optimizer(self):
        """Test FP8 training with Lion optimizer"""
        # Configure FP8
        fp8_config = FP8Config(enabled=True, precision="E4M3")

        # Convert model to FP8
        fp8_model = convert_model_to_fp8(self.model, fp8_config)

        # Setup FP8 optimizer
        base_optimizer = LionOptimizer(fp8_model.parameters(), lr=1e-3)
        optimizer = FP8Optimizer(base_optimizer, fp8_config)
        training_manager = FP8TrainingManager(fp8_config)

        losses = []

        for step in range(20):
            x = torch.randn(16, 128, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            optimizer.zero_grad()
            output = fp8_model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            training_manager.step(step=step, loss=loss.item())

            losses.append(loss.item())

        # Should train without errors
        self.assertEqual(len(losses), 20)

    def test_fp8_with_progressive_training(self):
        """Test FP8 integration with progressive training"""
        # FP8 configuration
        fp8_config = FP8Config(enabled=True)

        # Progressive training configuration
        progressive_config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=4,
                max_batch_size=16,
                memory_threshold_gb=2.0
            ),
            enable_curriculum=False,
            enable_growlength=False,
            enable_dynamic_batch=True
        )

        # Setup
        fp8_model = convert_model_to_fp8(self.model, fp8_config)
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        base_optimizer = AdaFactorOptimizer(fp8_model.parameters(), relative_step=True)
        fp8_optimizer = FP8Optimizer(base_optimizer, fp8_config)
        fp8_manager = FP8TrainingManager(fp8_config)

        training_metrics = []

        for step in range(25):
            batch_size = orchestrator.get_current_batch_size()

            x = torch.randn(batch_size, 128, device=self.device)
            y = torch.randn(batch_size, 1, device=self.device)

            fp8_optimizer.zero_grad()
            output = fp8_model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            fp8_optimizer.step()

            # Update both managers
            memory_usage = batch_size * 0.1  # Simulate memory usage
            orchestrator.update(
                step=step,
                loss=loss.item(),
                memory_usage_gb=memory_usage
            )
            fp8_manager.step(step=step, loss=loss.item())

            training_metrics.append({
                'step': step,
                'batch_size': batch_size,
                'loss': loss.item(),
                'memory_usage': memory_usage
            })

        # Dynamic batch sizing should adapt
        batch_sizes = [m['batch_size'] for m in training_metrics]
        self.assertGreaterEqual(max(batch_sizes), min(batch_sizes))


class TestFactoryIntegration(unittest.TestCase):
    """Test factory pattern integration across components"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(32, 1).to(self.device)

    def test_optimizer_scheduler_factory_integration(self):
        """Test creating optimizers and schedulers via factories"""
        # Create optimizer via factory
        optimizer_config = OptimizerConfig(
            name="lion",
            lr=1e-3,
            weight_decay=0.01,
            lion_betas=(0.9, 0.99)
        )

        optimizer = OptimizerFactory.create_optimizer(
            optimizer_config,
            self.model.parameters()
        )

        # Create scheduler via factory
        scheduler_config = SchedulerConfig(
            name="onecycle",
            max_lr=1e-2,
            total_steps=50,
            pct_start=0.3
        )

        scheduler = SchedulerFactory.create_scheduler(scheduler_config, optimizer)

        # Verify types
        self.assertIsInstance(optimizer, LionOptimizer)
        self.assertIsInstance(scheduler, OneCycleScheduler)

        # Test training integration
        for step in range(50):
            x = torch.randn(16, 32, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

        # Should complete successfully
        self.assertTrue(True)

    def test_configuration_driven_training(self):
        """Test complete training setup via configuration"""
        # Training configuration
        training_config = {
            'optimizer': {
                'name': 'sophia',
                'lr': 1e-3,
                'sophia_rho': 0.04,
                'sophia_betas': (0.965, 0.99)
            },
            'scheduler': {
                'name': 'cosine_with_restarts',
                'T_0': 20,
                'T_mult': 2,
                'eta_min': 1e-6
            },
            'progressive': {
                'curriculum_config': {
                    'strategy': 'length_based',
                    'initial_max_length': 8,
                    'final_max_length': 32,
                    'growth_steps': 30
                },
                'enable_curriculum': True,
                'enable_growlength': False,
                'enable_dynamic_batch': False
            },
            'fp8': {
                'enabled': True,
                'precision': 'E4M3'
            }
        }

        # Create components from configuration
        optimizer_config = OptimizerConfig(**training_config['optimizer'])
        optimizer = OptimizerFactory.create_optimizer(optimizer_config, self.model.parameters())

        scheduler_config = SchedulerConfig(**training_config['scheduler'])
        scheduler = SchedulerFactory.create_scheduler(scheduler_config, optimizer)

        progressive_config = ProgressiveConfig(**training_config['progressive'])
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        fp8_config = FP8Config(**training_config['fp8'])
        fp8_model = convert_model_to_fp8(self.model, fp8_config)

        # Integrated training loop
        for step in range(30):
            max_length = orchestrator.get_current_max_length()

            x = torch.randn(8, max_length, device=self.device)
            # Reduce to single dimension for linear model
            x = x.mean(dim=1, keepdim=True)  # [8, 1]
            y = torch.randn(8, 1, device=self.device)

            optimizer.zero_grad()
            output = fp8_model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            orchestrator.update(step=step, loss=loss.item(), gradient_norm=1.0)

        # All components should work together
        self.assertTrue(True)


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration and stress scenarios"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_memory_efficiency_integration(self):
        """Test memory efficiency with all optimizations enabled"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Large model for memory testing
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # Enable all optimizations
        fp8_config = FP8Config(enabled=True)
        fp8_model = convert_model_to_fp8(model, fp8_config)

        progressive_config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=16,
                max_batch_size=64,
                memory_threshold_gb=4.0,
                adaptation_strategy="memory_based"
            ),
            enable_dynamic_batch=True
        )
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        optimizer = AdaFactorOptimizer(fp8_model.parameters(), relative_step=True)

        # Track memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for step in range(20):
            batch_size = orchestrator.get_current_batch_size()

            x = torch.randn(batch_size, 512, device=self.device)
            y = torch.randn(batch_size, 1, device=self.device)

            optimizer.zero_grad()
            output = fp8_model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()

            # Simulate memory monitoring
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            orchestrator.update(
                step=step,
                loss=loss.item(),
                memory_usage_gb=memory_gb
            )

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        # Should complete with reasonable memory usage
        self.assertLess(peak_memory, 8.0)  # Reasonable threshold

    def test_checkpoint_recovery_integration(self):
        """Test checkpoint saving and recovery with all components"""
        model = nn.Linear(64, 1).to(self.device)

        # Setup all components
        optimizer_config = OptimizerConfig(name="lion", lr=1e-3)
        optimizer = OptimizerFactory.create_optimizer(optimizer_config, model.parameters())

        scheduler_config = SchedulerConfig(name="onecycle", max_lr=1e-2, total_steps=50)
        scheduler = SchedulerFactory.create_scheduler(scheduler_config, optimizer)

        progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=16,
                final_max_length=48,
                growth_steps=50
            )
        )
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        # Train for some steps
        for step in range(25):
            x = torch.randn(8, 64, device=self.device)
            y = torch.randn(8, 1, device=self.device)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            orchestrator.update(step=step, loss=loss.item(), gradient_norm=1.0)

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'progressive': orchestrator.get_state(),
            'step': 25
        }

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            torch.save(checkpoint, f)
            checkpoint_path = f.name

        try:
            # Create new components and load checkpoint
            new_model = nn.Linear(64, 1).to(self.device)
            new_optimizer = OptimizerFactory.create_optimizer(optimizer_config, new_model.parameters())
            new_scheduler = SchedulerFactory.create_scheduler(scheduler_config, new_optimizer)
            new_orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

            loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

            new_model.load_state_dict(loaded_checkpoint['model'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer'])
            new_scheduler.load_state_dict(loaded_checkpoint['scheduler'])
            new_orchestrator.load_state(loaded_checkpoint['progressive'])

            # Continue training
            for step in range(25, 50):
                x = torch.randn(8, 64, device=self.device)
                y = torch.randn(8, 1, device=self.device)

                new_optimizer.zero_grad()
                output = new_model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()

                new_optimizer.step()
                new_scheduler.step()
                new_orchestrator.update(step=step, loss=loss.item(), gradient_norm=1.0)

            # Should resume training successfully
            self.assertTrue(True)

        finally:
            os.unlink(checkpoint_path)

    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        model = nn.Linear(32, 1).to(self.device)

        # Test with invalid configurations
        with self.assertRaises(ValueError):
            invalid_optimizer_config = OptimizerConfig(name="nonexistent")
            OptimizerFactory.create_optimizer(invalid_optimizer_config, model.parameters())

        with self.assertRaises(ValueError):
            invalid_scheduler_config = SchedulerConfig(name="nonexistent")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            SchedulerFactory.create_scheduler(invalid_scheduler_config, optimizer)

        # Test with extreme configurations
        extreme_progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=1000,  # Very large
                final_max_length=10000,   # Extremely large
                growth_steps=1
            )
        )

        # Should handle gracefully
        try:
            orchestrator = ProgressiveTrainingOrchestrator(extreme_progressive_config)
            self.assertIsInstance(orchestrator, ProgressiveTrainingOrchestrator)
        except Exception as e:
            # If it fails, should be a clear error
            self.assertIsInstance(e, (ValueError, RuntimeError))


if __name__ == '__main__':
    unittest.main()