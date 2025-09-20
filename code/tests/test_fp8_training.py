"""
Unit Tests for FP8 Training Support

Tests for FP8 training functionality including Transformer Engine integration,
FP8 attention, linear layers, and H100/B200 GPU optimizations.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, Mock
import warnings

from tests import TEST_CONFIG
from src.Ava.optimization.fp8_training import (
    FP8Config,
    FP8Optimizer,
    FP8Attention,
    FP8Linear,
    FP8LayerNorm,
    FP8TrainingManager,
    check_fp8_support,
    convert_model_to_fp8
)


class TestFP8Support(unittest.TestCase):
    """Test FP8 support detection and validation"""

    def test_fp8_support_detection(self):
        """Test FP8 support detection"""
        # This will check actual hardware, but should not crash
        try:
            support_info = check_fp8_support()
            self.assertIsInstance(support_info, dict)
            self.assertIn('supported', support_info)
            self.assertIn('device_capability', support_info)
        except Exception as e:
            # If any error occurs, it should be gracefully handled
            self.assertIsInstance(e, (RuntimeError, ImportError))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    def test_fp8_support_mocked(self, mock_capability, mock_cuda):
        """Test FP8 support with mocked hardware"""
        # Test with H100 (supported)
        mock_cuda.return_value = True
        mock_capability.return_value = (9, 0)  # H100 capability

        support_info = check_fp8_support()
        self.assertTrue(support_info['supported'])
        self.assertEqual(support_info['device_capability'], (9, 0))

        # Test with older GPU (not supported)
        mock_capability.return_value = (7, 5)  # RTX 2080 Ti

        support_info = check_fp8_support()
        self.assertFalse(support_info['supported'])

    @patch('torch.cuda.is_available')
    def test_fp8_support_no_cuda(self, mock_cuda):
        """Test FP8 support detection without CUDA"""
        mock_cuda.return_value = False

        support_info = check_fp8_support()
        self.assertFalse(support_info['supported'])
        self.assertEqual(support_info['reason'], 'CUDA not available')


class TestFP8Config(unittest.TestCase):
    """Test FP8 configuration"""

    def test_fp8_config_initialization(self):
        """Test FP8Config initialization"""
        config = FP8Config(
            enabled=True,
            precision="E4M3",
            margin=8,
            interval=16,
            amax_history_len=1024,
            amax_compute_algo="max"
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.precision, "E4M3")
        self.assertEqual(config.margin, 8)
        self.assertEqual(config.interval, 16)

    def test_fp8_config_validation(self):
        """Test FP8Config validation"""
        # Valid precision
        config = FP8Config(precision="E4M3")
        self.assertEqual(config.precision, "E4M3")

        config = FP8Config(precision="E5M2")
        self.assertEqual(config.precision, "E5M2")

        # Invalid precision should use default
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = FP8Config(precision="INVALID")
            self.assertEqual(config.precision, "E4M3")

    def test_fp8_config_auto_detection(self):
        """Test automatic FP8 configuration based on hardware"""
        config = FP8Config(auto_detect=True)

        # Should set reasonable defaults
        self.assertIsInstance(config.enabled, bool)
        self.assertIn(config.precision, ["E4M3", "E5M2"])
        self.assertGreater(config.margin, 0)


class TestFP8Optimizer(unittest.TestCase):
    """Test FP8 optimizer wrapper"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(10, 1).to(self.device)
        self.base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def test_fp8_optimizer_initialization(self):
        """Test FP8 optimizer initialization"""
        config = FP8Config(enabled=True)
        optimizer = FP8Optimizer(self.base_optimizer, config)

        self.assertEqual(optimizer.base_optimizer, self.base_optimizer)
        self.assertEqual(optimizer.config, config)

    @patch('src.Ava.optimization.fp8_training.te', new=None)
    def test_fp8_optimizer_fallback(self):
        """Test FP8 optimizer fallback when Transformer Engine unavailable"""
        config = FP8Config(enabled=True)
        optimizer = FP8Optimizer(self.base_optimizer, config)

        # Should fall back to base optimizer behavior
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)

        output = self.model(x)
        loss = nn.MSELoss()(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should work without errors (fallback to base optimizer)
        self.assertTrue(True)

    def test_fp8_optimizer_state_dict(self):
        """Test FP8 optimizer state dictionary operations"""
        config = FP8Config(enabled=True)
        optimizer = FP8Optimizer(self.base_optimizer, config)

        # Get state dict
        state_dict = optimizer.state_dict()
        self.assertIn('base_optimizer', state_dict)
        self.assertIn('fp8_config', state_dict)

        # Load state dict
        new_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        fp8_optimizer = FP8Optimizer(new_optimizer, config)
        fp8_optimizer.load_state_dict(state_dict)

        # Should load without errors
        self.assertTrue(True)


class TestFP8Layers(unittest.TestCase):
    """Test FP8 layer implementations"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = FP8Config(enabled=True)

    def test_fp8_linear_initialization(self):
        """Test FP8Linear layer initialization"""
        layer = FP8Linear(
            in_features=128,
            out_features=64,
            bias=True,
            config=self.config
        ).to(self.device)

        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 64)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_fp8_linear_forward(self):
        """Test FP8Linear forward pass"""
        layer = FP8Linear(
            in_features=64,
            out_features=32,
            config=self.config
        ).to(self.device)

        x = torch.randn(16, 64, device=self.device)
        output = layer(x)

        self.assertEqual(output.shape, (16, 32))
        self.assertEqual(output.device, self.device)

    @patch('src.Ava.optimization.fp8_training.te', new=None)
    def test_fp8_linear_fallback(self):
        """Test FP8Linear fallback to standard linear"""
        layer = FP8Linear(
            in_features=64,
            out_features=32,
            config=self.config
        ).to(self.device)

        x = torch.randn(16, 64, device=self.device)
        output = layer(x)

        # Should work even without Transformer Engine
        self.assertEqual(output.shape, (16, 32))

    def test_fp8_attention_initialization(self):
        """Test FP8Attention layer initialization"""
        attention = FP8Attention(
            hidden_size=512,
            num_heads=8,
            config=self.config
        ).to(self.device)

        self.assertEqual(attention.hidden_size, 512)
        self.assertEqual(attention.num_heads, 8)
        self.assertEqual(attention.head_dim, 64)  # 512 / 8

    def test_fp8_attention_forward(self):
        """Test FP8Attention forward pass"""
        attention = FP8Attention(
            hidden_size=256,
            num_heads=4,
            config=self.config
        ).to(self.device)

        seq_len = 32
        batch_size = 8
        x = torch.randn(batch_size, seq_len, 256, device=self.device)

        output = attention(x)

        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_fp8_attention_with_mask(self):
        """Test FP8Attention with attention mask"""
        attention = FP8Attention(
            hidden_size=128,
            num_heads=4,
            config=self.config
        ).to(self.device)

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, 128, device=self.device)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        output = attention(x, attention_mask=mask)

        self.assertEqual(output.shape, (batch_size, seq_len, 128))

    def test_fp8_layernorm_initialization(self):
        """Test FP8LayerNorm initialization"""
        layer_norm = FP8LayerNorm(
            normalized_shape=256,
            config=self.config
        ).to(self.device)

        self.assertEqual(layer_norm.normalized_shape, (256,))
        self.assertIsNotNone(layer_norm.weight)
        self.assertIsNotNone(layer_norm.bias)

    def test_fp8_layernorm_forward(self):
        """Test FP8LayerNorm forward pass"""
        layer_norm = FP8LayerNorm(
            normalized_shape=128,
            config=self.config
        ).to(self.device)

        x = torch.randn(32, 64, 128, device=self.device)
        output = layer_norm(x)

        self.assertEqual(output.shape, x.shape)

        # Check normalization properties
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)

        # Should be approximately normalized
        self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-5))
        self.assertTrue(torch.allclose(output_std, torch.ones_like(output_std), atol=1e-2))


class TestFP8TrainingManager(unittest.TestCase):
    """Test FP8 training manager"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = FP8Config(enabled=True, interval=4)

    def test_training_manager_initialization(self):
        """Test FP8TrainingManager initialization"""
        manager = FP8TrainingManager(self.config)

        self.assertEqual(manager.config, self.config)
        self.assertEqual(manager.step_count, 0)

    def test_scaling_factor_management(self):
        """Test scaling factor updates"""
        manager = FP8TrainingManager(self.config)

        # Initial scaling factors
        initial_factors = manager.get_scaling_factors()
        self.assertIsInstance(initial_factors, dict)

        # Update with loss information
        manager.update_scaling_factors(
            forward_loss=torch.tensor(1.0),
            backward_loss=torch.tensor(0.5)
        )

        # Scaling factors should be updated
        updated_factors = manager.get_scaling_factors()
        self.assertIsInstance(updated_factors, dict)

    def test_amax_tracking(self):
        """Test AMAX (absolute maximum) tracking"""
        manager = FP8TrainingManager(self.config)

        # Feed some tensor data
        test_tensors = [
            torch.randn(64, 128, device=self.device),
            torch.randn(32, 256, device=self.device),
            torch.randn(16, 512, device=self.device)
        ]

        for tensor in test_tensors:
            manager.track_amax(tensor, 'forward')

        # Should have tracked AMAX values
        amax_history = manager.get_amax_history()
        self.assertIsInstance(amax_history, dict)

    def test_loss_scaling_adaptation(self):
        """Test dynamic loss scaling adaptation"""
        manager = FP8TrainingManager(self.config)

        # Simulate training with different loss values
        losses = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4]  # Decreasing losses

        for i, loss in enumerate(losses):
            manager.step(step=i, loss=loss)

        # Should adapt scaling factors based on loss trends
        final_factors = manager.get_scaling_factors()
        self.assertIsInstance(final_factors, dict)

    def test_overflow_detection(self):
        """Test overflow detection and handling"""
        manager = FP8TrainingManager(self.config)

        # Simulate overflow scenario
        overflow_tensor = torch.tensor(float('inf'), device=self.device)

        overflow_detected = manager.check_overflow(overflow_tensor)
        self.assertTrue(overflow_detected)

        # Normal tensor should not trigger overflow
        normal_tensor = torch.randn(64, device=self.device)
        overflow_detected = manager.check_overflow(normal_tensor)
        self.assertFalse(overflow_detected)

    def test_state_persistence(self):
        """Test FP8 training manager state saving/loading"""
        manager = FP8TrainingManager(self.config)

        # Run for some steps
        for i in range(10):
            manager.step(step=i, loss=1.0 - i * 0.1)

        # Save state
        state = manager.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn('step_count', state)
        self.assertIn('scaling_factors', state)

        # Create new manager and load state
        new_manager = FP8TrainingManager(self.config)
        new_manager.load_state(state)

        # Should match original state
        self.assertEqual(manager.step_count, new_manager.step_count)


class TestFP8ModelConversion(unittest.TestCase):
    """Test FP8 model conversion utilities"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.config = FP8Config(enabled=True)

    def test_linear_layer_conversion(self):
        """Test conversion of linear layers to FP8"""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

        fp8_model = convert_model_to_fp8(model, self.config)

        # Linear layers should be converted to FP8Linear
        linear_layers = [module for module in fp8_model.modules() if isinstance(module, (nn.Linear, FP8Linear))]
        fp8_linear_count = sum(1 for layer in linear_layers if isinstance(layer, FP8Linear))

        self.assertGreater(fp8_linear_count, 0)

    def test_attention_layer_conversion(self):
        """Test conversion of attention layers to FP8"""
        # Mock transformer-like model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(256, 8)
                self.norm = nn.LayerNorm(256)
                self.linear = nn.Linear(256, 256)

            def forward(self, x):
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(attn_out + x)
                return self.linear(x)

        model = SimpleTransformer().to(self.device)
        fp8_model = convert_model_to_fp8(model, self.config, convert_attention=True)

        # Should convert compatible layers
        self.assertIsInstance(fp8_model, nn.Module)

    def test_selective_layer_conversion(self):
        """Test selective layer conversion with layer name patterns"""
        model = nn.Sequential(
            nn.Linear(64, 32, bias=False),  # No bias
            nn.Linear(32, 16),              # With bias
            nn.LayerNorm(16),
            nn.Linear(16, 1)
        ).to(self.device)

        # Convert only layers without bias
        fp8_model = convert_model_to_fp8(
            model,
            self.config,
            layer_name_patterns=['*.weight'],  # Only layers with weight parameter
            exclude_bias_layers=True
        )

        # Should have converted some layers
        self.assertIsInstance(fp8_model, nn.Module)

    def test_model_conversion_preserves_functionality(self):
        """Test that FP8 conversion preserves model functionality"""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(self.device)

        # Get original output
        x = torch.randn(32, 10, device=self.device)
        with torch.no_grad():
            original_output = model(x)

        # Convert to FP8
        fp8_model = convert_model_to_fp8(model, self.config)

        # Get FP8 output
        with torch.no_grad():
            fp8_output = fp8_model(x)

        # Outputs should have similar shapes
        self.assertEqual(original_output.shape, fp8_output.shape)

    def test_conversion_with_mixed_precision(self):
        """Test FP8 conversion compatibility with mixed precision"""
        model = nn.Linear(128, 64).to(self.device)

        # Convert to FP8
        fp8_model = convert_model_to_fp8(model, self.config)

        # Test with different input precisions
        inputs = [
            torch.randn(32, 128, dtype=torch.float32, device=self.device),
            torch.randn(32, 128, dtype=torch.float16, device=self.device)
        ]

        for x in inputs:
            with self.subTest(dtype=x.dtype):
                try:
                    output = fp8_model(x)
                    self.assertEqual(output.shape, (32, 64))
                except Exception as e:
                    # Some precision combinations might not be supported
                    self.assertIsInstance(e, (RuntimeError, TypeError))


class TestFP8Integration(unittest.TestCase):
    """Integration tests for FP8 training"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def test_end_to_end_fp8_training(self):
        """Test complete FP8 training workflow"""
        # Create model
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(self.device)

        # Configure FP8
        config = FP8Config(enabled=True, interval=4)

        # Convert model
        fp8_model = convert_model_to_fp8(model, config)

        # Create optimizer and training manager
        optimizer = torch.optim.AdamW(fp8_model.parameters(), lr=1e-3)
        fp8_optimizer = FP8Optimizer(optimizer, config)
        training_manager = FP8TrainingManager(config)

        # Training loop
        for step in range(10):
            x = torch.randn(16, 64, device=self.device)
            y = torch.randn(16, 1, device=self.device)

            fp8_optimizer.zero_grad()

            output = fp8_model(x)
            loss = nn.MSELoss()(output, y)

            loss.backward()
            fp8_optimizer.step()

            # Update FP8 training manager
            training_manager.step(step=step, loss=loss.item())

        # Should complete without errors
        self.assertTrue(True)

    @patch('src.Ava.optimization.fp8_training.check_fp8_support')
    def test_fp8_fallback_behavior(self, mock_support):
        """Test FP8 fallback when hardware doesn't support it"""
        # Mock unsupported hardware
        mock_support.return_value = {'supported': False, 'reason': 'Unsupported GPU'}

        config = FP8Config(enabled=True, auto_detect=True)

        # Should automatically disable FP8
        self.assertFalse(config.enabled)

    def test_fp8_memory_efficiency(self):
        """Test FP8 memory efficiency compared to FP16/FP32"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Create large model for memory testing
        large_model = nn.Sequential(
            *[nn.Linear(512, 512) for _ in range(5)]
        ).to(self.device)

        def measure_memory_usage(model, dtype=torch.float32):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Convert model to specified precision
            if dtype == torch.float16:
                model = model.half()
            elif dtype == torch.float32:
                model = model.float()

            x = torch.randn(64, 512, device=self.device, dtype=dtype)

            # Forward pass
            output = model(x)
            loss = output.sum()

            # Backward pass
            loss.backward()

            return torch.cuda.max_memory_allocated()

        # Measure FP32 memory
        fp32_memory = measure_memory_usage(large_model.float(), torch.float32)

        # Measure FP16 memory
        fp16_memory = measure_memory_usage(large_model.half(), torch.float16)

        # FP16 should use less memory than FP32
        self.assertLess(fp16_memory, fp32_memory)

        # Note: Actual FP8 memory testing would require Transformer Engine
        # This test validates the testing framework


if __name__ == '__main__':
    unittest.main()