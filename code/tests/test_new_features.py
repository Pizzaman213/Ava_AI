"""
Comprehensive Test Suite for New Ava Features.

Tests all newly implemented features:
- ALiBi Positional Encoding
- Cross-Attention and Multi-Modal Fusion
- Modality Encoders (Vision, Audio)
- Advanced Loss Functions
- Gradient Surgery
- Auxiliary-Free Load Balancing
- Media Processors (Audio, Video)

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


class TestALiBi(unittest.TestCase):
    """Tests for ALiBi positional encoding."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.seq_len = 32
        self.hidden_size = 256
        self.num_heads = 8

    def test_alibi_slopes(self):
        """Test ALiBi slope computation."""
        from ava.models.alibi import get_alibi_slopes

        # Power of 2
        slopes = get_alibi_slopes(8)
        self.assertEqual(slopes.shape, (8,))
        self.assertTrue((slopes > 0).all())

        # Non-power of 2
        slopes = get_alibi_slopes(12)
        self.assertEqual(slopes.shape, (12,))

    def test_alibi_bias_matrix(self):
        """Test ALiBi bias matrix construction."""
        from ava.models.alibi import build_alibi_bias, get_alibi_slopes

        slopes = get_alibi_slopes(self.num_heads)
        bias = build_alibi_bias(self.seq_len, self.num_heads, slopes, device=self.device)

        self.assertEqual(bias.shape, (1, self.num_heads, self.seq_len, self.seq_len))
        # Diagonal should be 0
        diagonal = torch.diagonal(bias[0], dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

    def test_alibi_module(self):
        """Test ALiBiPositionalBias module."""
        from ava.models.alibi import ALiBiPositionalBias

        alibi = ALiBiPositionalBias(num_heads=self.num_heads).to(self.device)
        bias = alibi(self.seq_len, device=self.device)

        self.assertEqual(bias.shape, (1, self.num_heads, self.seq_len, self.seq_len))

    def test_alibi_attention(self):
        """Test ALiBiAttention layer."""
        from ava.models.alibi import ALiBiAttention

        attn = ALiBiAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        output = attn(x, x, x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))


class TestCrossAttention(unittest.TestCase):
    """Tests for Cross-Attention layers."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.seq_q = 32
        self.seq_kv = 64
        self.hidden_size = 256
        self.encoder_hidden_size = 128

    def test_cross_attention_creation(self):
        """Test CrossAttention can be created."""
        from ava.models.cross_attention import CrossAttentionConfig, CrossAttention

        config = CrossAttentionConfig(
            hidden_size=self.hidden_size,
            encoder_hidden_size=self.encoder_hidden_size,
            num_heads=8,
        )
        cross_attn = CrossAttention(config).to(self.device)

        self.assertEqual(cross_attn.hidden_size, self.hidden_size)
        self.assertEqual(cross_attn.encoder_hidden_size, self.encoder_hidden_size)

    def test_cross_attention_forward(self):
        """Test CrossAttention forward pass."""
        from ava.models.cross_attention import CrossAttentionConfig, CrossAttention

        config = CrossAttentionConfig(
            hidden_size=self.hidden_size,
            encoder_hidden_size=self.encoder_hidden_size,
            num_heads=8,
        )
        cross_attn = CrossAttention(config).to(self.device)

        hidden_states = torch.randn(self.batch_size, self.seq_q, self.hidden_size, device=self.device)
        encoder_hidden = torch.randn(self.batch_size, self.seq_kv, self.encoder_hidden_size, device=self.device)

        output, attn_weights = cross_attn(hidden_states, encoder_hidden)

        self.assertEqual(output.shape, (self.batch_size, self.seq_q, self.hidden_size))

    def test_gated_cross_attention(self):
        """Test GatedCrossAttention."""
        from ava.models.cross_attention import CrossAttentionConfig, GatedCrossAttention

        config = CrossAttentionConfig(
            hidden_size=self.hidden_size,
            encoder_hidden_size=self.encoder_hidden_size,
            num_heads=8,
            gated_cross_attention=True,
        )
        gated_attn = GatedCrossAttention(config).to(self.device)

        hidden_states = torch.randn(self.batch_size, self.seq_q, self.hidden_size, device=self.device)
        encoder_hidden = torch.randn(self.batch_size, self.seq_kv, self.encoder_hidden_size, device=self.device)

        output, _ = gated_attn(hidden_states, encoder_hidden)

        self.assertEqual(output.shape, (self.batch_size, self.seq_q, self.hidden_size))
        # Gate should be bounded by tanh
        gate_val = gated_attn.get_gate_value()
        self.assertTrue(-1 <= gate_val <= 1)

    def test_multi_modal_fusion(self):
        """Test MultiModalFusion with multiple modalities."""
        from ava.models.cross_attention import MultiModalFusion

        fusion = MultiModalFusion(
            hidden_size=self.hidden_size,
            modality_configs={
                'vision': {'encoder_hidden_size': 128},
                'audio': {'encoder_hidden_size': 64},
            },
            fusion_type='sequential',
        ).to(self.device)

        text_hidden = torch.randn(self.batch_size, self.seq_q, self.hidden_size, device=self.device)
        encoder_hidden = {
            'vision': torch.randn(self.batch_size, 49, 128, device=self.device),
            'audio': torch.randn(self.batch_size, 100, 64, device=self.device),
        }

        output, attn_weights = fusion(text_hidden, encoder_hidden)

        self.assertEqual(output.shape, (self.batch_size, self.seq_q, self.hidden_size))
        self.assertIn('vision', attn_weights)
        self.assertIn('audio', attn_weights)


class TestModalityEncoders(unittest.TestCase):
    """Tests for modality encoders."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2

    def test_vision_encoder(self):
        """Test VisionEncoder."""
        from ava.models.modality_encoders import VisionEncoderConfig, VisionEncoder

        config = VisionEncoderConfig(
            image_size=64,
            patch_size=8,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
        )
        encoder = VisionEncoder(config).to(self.device)

        # Create fake image batch
        images = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        features = encoder(images)

        # Expected: (64/8)^2 = 64 patches + 1 CLS token = 65
        self.assertEqual(features.shape, (self.batch_size, 65, 128))

    def test_audio_encoder(self):
        """Test AudioEncoder."""
        from ava.models.modality_encoders import AudioEncoderConfig, AudioEncoder

        config = AudioEncoderConfig(
            input_size=80,  # Mel bins
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            conv_downsample=True,
        )
        encoder = AudioEncoder(config).to(self.device)

        # Create fake spectrogram [batch, mel_bins, frames]
        spectrogram = torch.randn(self.batch_size, 80, 400, device=self.device)
        features, mask = encoder(spectrogram)

        # Output should be downsampled
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertEqual(features.shape[2], 128)  # hidden_size

    def test_patch_embedding(self):
        """Test PatchEmbedding."""
        from ava.models.modality_encoders import PatchEmbedding

        patch_embed = PatchEmbedding(
            image_size=64,
            patch_size=8,
            in_channels=3,
            hidden_size=128,
        ).to(self.device)

        images = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        patches = patch_embed(images)

        # (64/8)^2 = 64 patches
        self.assertEqual(patches.shape, (self.batch_size, 64, 128))


class TestAdvancedLosses(unittest.TestCase):
    """Tests for advanced loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.num_classes = 100
        self.hidden_size = 256

    def test_focal_loss(self):
        """Test FocalLoss."""
        from ava.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        loss = loss_fn(logits, targets)

        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

    def test_contrastive_loss(self):
        """Test ContrastiveLoss."""
        from ava.training.losses import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.07)

        # Create embeddings with some positive pairs (same labels)
        embeddings = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        labels = torch.tensor([0, 0, 1, 1, 2, 2] + [i for i in range(3, self.batch_size - 6 + 3)], device=self.device)

        loss = loss_fn(embeddings, labels)

        self.assertEqual(loss.dim(), 0)  # Scalar

    def test_diversity_loss(self):
        """Test DiversityLoss."""
        from ava.training.losses import DiversityLoss

        loss_fn = DiversityLoss(weight=0.01)

        embeddings = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        loss = loss_fn(embeddings)

        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertTrue(loss.item() >= 0)

    def test_adaptive_temperature_loss(self):
        """Test AdaptiveTemperatureLoss."""
        from ava.training.losses import AdaptiveTemperatureLoss

        loss_fn = AdaptiveTemperatureLoss(
            initial_temperature=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
        )
        loss_fn.train()

        logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        # Run multiple steps to test temperature adaptation
        initial_temp = loss_fn.get_temperature()
        for _ in range(10):
            loss = loss_fn(logits, targets)

        # Temperature should have been updated
        self.assertEqual(loss.dim(), 0)

    def test_label_smoothing(self):
        """Test LabelSmoothingCrossEntropy."""
        from ava.training.losses import LabelSmoothingCrossEntropy

        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        loss = loss_fn(logits, targets)

        self.assertEqual(loss.dim(), 0)

    def test_distillation_loss(self):
        """Test DistillationLoss."""
        from ava.training.losses import DistillationLoss

        loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)

        student_logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        teacher_logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        loss, info = loss_fn(student_logits, teacher_logits, targets)

        self.assertEqual(loss.dim(), 0)
        self.assertIn('hard_loss', info)
        self.assertIn('soft_loss', info)


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


class TestMediaProcessors(unittest.TestCase):
    """Tests for media processors."""

    def test_audio_processor_creation(self):
        """Test AudioProcessor can be created."""
        try:
            from ava.data.media_processors import AudioProcessor, AudioConfig
        except (ImportError, RuntimeError, AttributeError) as e:
            self.skipTest(f"Media processors not available: {e}")
            return

        config = AudioConfig(sample_rate=16000, n_mels=80)
        processor = AudioProcessor(config)

        self.assertEqual(processor.config.sample_rate, 16000)
        self.assertEqual(processor.config.n_mels, 80)

    def test_video_processor_creation(self):
        """Test VideoProcessor can be created."""
        try:
            from ava.data.media_processors import VideoProcessor, VideoConfig
        except (ImportError, RuntimeError, AttributeError) as e:
            self.skipTest(f"Media processors not available: {e}")
            return

        config = VideoConfig(num_frames=8, frame_size=(224, 224))
        processor = VideoProcessor(config)

        self.assertEqual(processor.config.num_frames, 8)
        self.assertEqual(processor.config.frame_size, (224, 224))


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
        TestALiBi,
        TestCrossAttention,
        TestModalityEncoders,
        TestAdvancedLosses,
        TestGradientSurgery,
        TestAuxFreeRouter,
        TestMediaProcessors,
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
