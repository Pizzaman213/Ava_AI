"""
Unit Tests for Progressive Training Framework

Tests for curriculum learning, GrowLength training, dynamic batch sizing,
and progressive training orchestration.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

from tests import TEST_CONFIG
from src.Ava.training.progressive_training import (
    CurriculumLearning,
    GrowLengthTraining,
    DynamicBatchSizing,
    ProgressiveTrainingOrchestrator,
    ProgressiveConfig,
    CurriculumConfig,
    GrowLengthConfig,
    DynamicBatchConfig
)


class TestCurriculumLearning(unittest.TestCase):
    """Test curriculum learning implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = CurriculumConfig(
            strategy="length_based",
            initial_max_length=128,
            final_max_length=512,
            growth_steps=1000,
            difficulty_metrics=["loss", "gradient_norm"]
        )

    def test_curriculum_initialization(self):
        """Test curriculum learning initialization"""
        curriculum = CurriculumLearning(self.config)

        self.assertEqual(curriculum.config.strategy, "length_based")
        self.assertEqual(curriculum.current_max_length, 128)
        self.assertEqual(curriculum.step_count, 0)

    def test_length_based_curriculum(self):
        """Test length-based curriculum progression"""
        curriculum = CurriculumLearning(self.config)

        # Test initial state
        self.assertEqual(curriculum.get_current_max_length(), 128)

        # Simulate training steps
        for step in range(500):
            curriculum.update(step=step, loss=1.0, gradient_norm=0.5)

        # Should be halfway to final length
        expected_length = 128 + (512 - 128) * 0.5
        self.assertAlmostEqual(curriculum.get_current_max_length(), expected_length, delta=10)

        # Complete training
        for step in range(500, 1000):
            curriculum.update(step=step, loss=0.5, gradient_norm=0.3)

        # Should reach final length
        self.assertAlmostEqual(curriculum.get_current_max_length(), 512, delta=5)

    def test_difficulty_based_curriculum(self):
        """Test difficulty-based curriculum adaptation"""
        config = CurriculumConfig(
            strategy="difficulty_based",
            initial_max_length=128,
            final_max_length=512,
            growth_steps=1000,
            difficulty_threshold=0.8
        )
        curriculum = CurriculumLearning(config)

        # High loss - should not increase difficulty
        curriculum.update(step=100, loss=2.0, gradient_norm=1.0)
        length_1 = curriculum.get_current_max_length()

        # Low loss - should increase difficulty
        for _ in range(10):
            curriculum.update(step=110, loss=0.1, gradient_norm=0.1)
        length_2 = curriculum.get_current_max_length()

        self.assertGreaterEqual(length_2, length_1)

    def test_sample_filtering(self):
        """Test curriculum sample filtering"""
        curriculum = CurriculumLearning(self.config)
        curriculum.current_max_length = 256

        # Create mock samples with different lengths
        samples = [
            {"input_ids": torch.zeros(128), "length": 128},
            {"input_ids": torch.zeros(256), "length": 256},
            {"input_ids": torch.zeros(512), "length": 512},
            {"input_ids": torch.zeros(64), "length": 64}
        ]

        filtered = curriculum.filter_samples(samples)

        # Should only include samples with length <= current_max_length
        self.assertEqual(len(filtered), 3)
        for sample in filtered:
            self.assertLessEqual(sample["length"], 256)

    def test_adaptive_threshold(self):
        """Test adaptive difficulty threshold"""
        config = CurriculumConfig(
            strategy="difficulty_based",
            initial_max_length=128,
            final_max_length=512,
            adaptive_threshold=True
        )
        curriculum = CurriculumLearning(config)

        # Feed consistent loss values
        losses = [1.0] * 50
        for i, loss in enumerate(losses):
            curriculum.update(step=i, loss=loss, gradient_norm=0.5)

        # Threshold should adapt to observed loss distribution
        self.assertIsNotNone(curriculum.adaptive_threshold_value)


class TestGrowLengthTraining(unittest.TestCase):
    """Test GrowLength training implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = GrowLengthConfig(
            initial_length=128,
            final_length=1024,
            growth_schedule="linear",
            growth_steps=2000,
            warmup_steps=100
        )

    def test_growlength_initialization(self):
        """Test GrowLength training initialization"""
        grow_length = GrowLengthTraining(self.config)

        self.assertEqual(grow_length.current_length, 128)
        self.assertEqual(grow_length.step_count, 0)

    def test_linear_growth_schedule(self):
        """Test linear growth schedule"""
        grow_length = GrowLengthTraining(self.config)

        # Test warmup phase
        grow_length.update(step=50)
        self.assertEqual(grow_length.get_current_length(), 128)

        # Test growth phase
        grow_length.update(step=1100)  # Halfway through growth
        expected = 128 + (1024 - 128) * 0.5
        self.assertAlmostEqual(grow_length.get_current_length(), expected, delta=10)

        # Test final phase
        grow_length.update(step=2100)
        self.assertEqual(grow_length.get_current_length(), 1024)

    def test_exponential_growth_schedule(self):
        """Test exponential growth schedule"""
        config = GrowLengthConfig(
            initial_length=128,
            final_length=1024,
            growth_schedule="exponential",
            growth_steps=1000
        )
        grow_length = GrowLengthTraining(config)

        lengths = []
        for step in range(0, 1000, 100):
            grow_length.update(step=step)
            lengths.append(grow_length.get_current_length())

        # Exponential growth should have larger increases toward the end
        differences = [lengths[i+1] - lengths[i] for i in range(len(lengths)-1)]
        self.assertGreater(differences[-1], differences[0])

    def test_cosine_growth_schedule(self):
        """Test cosine growth schedule"""
        config = GrowLengthConfig(
            initial_length=128,
            final_length=512,
            growth_schedule="cosine",
            growth_steps=1000
        )
        grow_length = GrowLengthTraining(config)

        # Test smooth cosine progression
        grow_length.update(step=250)  # 1/4 through
        length_quarter = grow_length.get_current_length()

        grow_length.update(step=500)  # 1/2 through
        length_half = grow_length.get_current_length()

        grow_length.update(step=750)  # 3/4 through
        length_three_quarter = grow_length.get_current_length()

        # Cosine should have smooth progression
        self.assertGreater(length_quarter, 128)
        self.assertGreater(length_half, length_quarter)
        self.assertGreater(length_three_quarter, length_half)
        self.assertLess(length_three_quarter, 512)

    def test_position_interpolation(self):
        """Test position embedding interpolation"""
        grow_length = GrowLengthTraining(self.config)

        # Mock model with position embeddings
        model = MagicMock()
        model.config.max_position_embeddings = 128
        pos_emb = torch.randn(128, 64)  # 128 positions, 64 dims
        model.transformer.wpe.weight = nn.Parameter(pos_emb)

        # Update to new length
        grow_length.update(step=1000)  # Should be at 512 length
        grow_length.interpolate_position_embeddings(model)

        # Position embeddings should be interpolated to new size
        new_pos_emb = model.transformer.wpe.weight
        self.assertEqual(new_pos_emb.shape[0], 512)
        self.assertEqual(new_pos_emb.shape[1], 64)

    def test_attention_mask_adjustment(self):
        """Test attention mask adjustment for new sequence lengths"""
        grow_length = GrowLengthTraining(self.config)
        grow_length.update(step=500)  # Mid-training

        # Create attention mask for current length
        current_length = grow_length.get_current_length()
        batch_size = 4

        mask = grow_length.create_attention_mask(batch_size, current_length)

        self.assertEqual(mask.shape, (batch_size, current_length, current_length))
        # Should be causal mask
        self.assertTrue(torch.allclose(mask, torch.tril(torch.ones_like(mask))))


class TestDynamicBatchSizing(unittest.TestCase):
    """Test dynamic batch sizing implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = DynamicBatchConfig(
            initial_batch_size=16,
            max_batch_size=128,
            memory_threshold_gb=8.0,
            adaptation_strategy="memory_based",
            warmup_steps=100
        )

    def test_dynamic_batch_initialization(self):
        """Test dynamic batch sizing initialization"""
        dynamic_batch = DynamicBatchSizing(self.config)

        self.assertEqual(dynamic_batch.current_batch_size, 16)
        self.assertEqual(dynamic_batch.step_count, 0)

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_memory_based_adaptation(self, mock_props, mock_memory):
        """Test memory-based batch size adaptation"""
        # Mock GPU with 16GB memory
        mock_props.return_value.total_memory = 16 * 1024**3
        mock_memory.return_value = 4 * 1024**3  # 4GB used

        dynamic_batch = DynamicBatchSizing(self.config)

        # Low memory usage - should increase batch size
        dynamic_batch.update(step=200, loss=1.0, memory_usage_gb=4.0)
        increased_size = dynamic_batch.get_current_batch_size()
        self.assertGreater(increased_size, 16)

        # High memory usage - should decrease batch size
        dynamic_batch.update(step=300, loss=1.0, memory_usage_gb=12.0)
        decreased_size = dynamic_batch.get_current_batch_size()
        self.assertLess(decreased_size, increased_size)

    def test_performance_based_adaptation(self):
        """Test performance-based batch size adaptation"""
        config = DynamicBatchConfig(
            initial_batch_size=32,
            max_batch_size=128,
            adaptation_strategy="performance_based",
            target_throughput=1000
        )
        dynamic_batch = DynamicBatchSizing(config)

        # Low throughput - should adjust batch size
        dynamic_batch.update(step=200, loss=1.0, throughput=500)
        size_1 = dynamic_batch.get_current_batch_size()

        # Good throughput - should maintain or increase
        dynamic_batch.update(step=300, loss=1.0, throughput=1200)
        size_2 = dynamic_batch.get_current_batch_size()

        self.assertGreaterEqual(size_2, size_1)

    def test_gradient_accumulation_adjustment(self):
        """Test gradient accumulation adjustment with batch size changes"""
        dynamic_batch = DynamicBatchSizing(self.config)

        target_effective_batch = 64

        # Small batch size should have high accumulation steps
        dynamic_batch.current_batch_size = 16
        acc_steps_1 = dynamic_batch.get_gradient_accumulation_steps(target_effective_batch)
        self.assertEqual(acc_steps_1, 4)  # 64 / 16

        # Large batch size should have low accumulation steps
        dynamic_batch.current_batch_size = 64
        acc_steps_2 = dynamic_batch.get_gradient_accumulation_steps(target_effective_batch)
        self.assertEqual(acc_steps_2, 1)  # 64 / 64

    def test_batch_size_constraints(self):
        """Test batch size constraint enforcement"""
        dynamic_batch = DynamicBatchSizing(self.config)

        # Try to set batch size above maximum
        dynamic_batch._adjust_batch_size(200)
        self.assertEqual(dynamic_batch.get_current_batch_size(), 128)

        # Try to set batch size below minimum
        dynamic_batch._adjust_batch_size(2)
        self.assertEqual(dynamic_batch.get_current_batch_size(), 4)  # Should respect minimum


class TestProgressiveTrainingOrchestrator(unittest.TestCase):
    """Test progressive training orchestrator"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=128,
                final_max_length=512,
                growth_steps=1000
            ),
            growlength_config=GrowLengthConfig(
                initial_length=128,
                final_length=512,
                growth_steps=1000
            ),
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=16,
                max_batch_size=64
            ),
            enable_curriculum=True,
            enable_growlength=True,
            enable_dynamic_batch=True
        )

    def test_orchestrator_initialization(self):
        """Test progressive training orchestrator initialization"""
        orchestrator = ProgressiveTrainingOrchestrator(self.config)

        self.assertIsNotNone(orchestrator.curriculum)
        self.assertIsNotNone(orchestrator.grow_length)
        self.assertIsNotNone(orchestrator.dynamic_batch)

    def test_orchestrator_update(self):
        """Test orchestrator coordinated update"""
        orchestrator = ProgressiveTrainingOrchestrator(self.config)

        initial_length = orchestrator.get_current_max_length()
        initial_batch = orchestrator.get_current_batch_size()

        # Update all components
        orchestrator.update(
            step=500,
            loss=1.0,
            gradient_norm=0.5,
            memory_usage_gb=4.0,
            throughput=800
        )

        # All components should have updated
        new_length = orchestrator.get_current_max_length()
        new_batch = orchestrator.get_current_batch_size()

        self.assertGreaterEqual(new_length, initial_length)
        # Batch size may increase or decrease based on conditions

    def test_state_saving_loading(self):
        """Test progressive training state persistence"""
        orchestrator = ProgressiveTrainingOrchestrator(self.config)

        # Update to mid-training state
        orchestrator.update(step=500, loss=1.0, gradient_norm=0.5)

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            state = orchestrator.get_state()
            json.dump(state, f)
            state_file = f.name

        try:
            # Create new orchestrator and load state
            new_orchestrator = ProgressiveTrainingOrchestrator(self.config)

            with open(state_file, 'r') as f:
                loaded_state = json.load(f)
            new_orchestrator.load_state(loaded_state)

            # States should match
            self.assertEqual(
                orchestrator.get_current_max_length(),
                new_orchestrator.get_current_max_length()
            )
            self.assertEqual(
                orchestrator.curriculum.step_count,
                new_orchestrator.curriculum.step_count
            )

        finally:
            os.unlink(state_file)

    def test_selective_component_enabling(self):
        """Test selective enabling/disabling of components"""
        # Only curriculum enabled
        config = ProgressiveConfig(
            curriculum_config=self.config.curriculum_config,
            enable_curriculum=True,
            enable_growlength=False,
            enable_dynamic_batch=False
        )

        orchestrator = ProgressiveTrainingOrchestrator(config)

        self.assertIsNotNone(orchestrator.curriculum)
        self.assertIsNone(orchestrator.grow_length)
        self.assertIsNone(orchestrator.dynamic_batch)

    def test_training_metrics_tracking(self):
        """Test training metrics tracking and reporting"""
        orchestrator = ProgressiveTrainingOrchestrator(self.config)

        # Update with metrics multiple times
        metrics_history = []
        for step in range(0, 1000, 100):
            loss = max(0.1, 2.0 * (1 - step/1000))  # Decreasing loss
            orchestrator.update(step=step, loss=loss, gradient_norm=0.5)

            metrics = orchestrator.get_current_metrics()
            metrics_history.append(metrics)

        # Should track progression
        self.assertLess(
            metrics_history[-1]["curriculum_max_length"],
            metrics_history[0]["curriculum_max_length"] + 200
        )


class TestProgressiveTrainingIntegration(unittest.TestCase):
    """Integration tests for progressive training components"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_end_to_end_progressive_training(self):
        """Test complete progressive training workflow"""
        # Create simple model for testing
        model = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1000)
        ).to(self.device)

        config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=32,
                final_max_length=128,
                growth_steps=100
            ),
            growlength_config=GrowLengthConfig(
                initial_length=32,
                final_length=128,
                growth_steps=100
            ),
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=8,
                max_batch_size=32
            )
        )

        orchestrator = ProgressiveTrainingOrchestrator(config)

        # Simulate training loop
        for step in range(100):
            # Get current training parameters
            max_length = orchestrator.get_current_max_length()
            batch_size = orchestrator.get_current_batch_size()

            # Create mock batch
            x = torch.randint(0, 1000, (batch_size, max_length), device=self.device)

            # Mock forward pass
            output = model(x.view(-1, 1)).view(batch_size, max_length, -1)
            loss = torch.randn(1).item()

            # Update progressive training
            orchestrator.update(
                step=step,
                loss=loss,
                gradient_norm=1.0,
                memory_usage_gb=2.0
            )

        # Should have progressed to larger sequences
        final_length = orchestrator.get_current_max_length()
        self.assertGreater(final_length, 32)

    def test_memory_efficiency_scaling(self):
        """Test progressive training memory efficiency"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=8,
                max_batch_size=64,
                memory_threshold_gb=4.0,
                adaptation_strategy="memory_based"
            ),
            enable_curriculum=False,
            enable_growlength=False,
            enable_dynamic_batch=True
        )

        orchestrator = ProgressiveTrainingOrchestrator(config)

        # Track memory usage across different batch sizes
        memory_usages = []

        for step in range(20):
            batch_size = orchestrator.get_current_batch_size()

            # Simulate memory usage proportional to batch size
            simulated_memory_gb = batch_size * 0.1  # 100MB per sample

            orchestrator.update(
                step=step,
                loss=1.0,
                memory_usage_gb=simulated_memory_gb
            )

            memory_usages.append(simulated_memory_gb)

        # Dynamic batch sizing should help control memory usage
        self.assertLess(max(memory_usages), config.dynamic_batch_config.memory_threshold_gb * 1.5)


if __name__ == '__main__':
    unittest.main()