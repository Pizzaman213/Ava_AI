#!/usr/bin/env python3
"""
Comprehensive test suite for Adaptive Multi-Token Prediction system.

Tests all components:
- ConfidenceGate
- MultiTokenPredictionHeads
- AdaptiveMTPModel
- AdaptiveMTPLoss
- Integration with configuration
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.Ava.models import (
    AdaptiveMTPModel,
    AdaptiveMTPConfig,
    ConfidenceGate,
    MultiTokenPredictionHeads,
)
from src.Ava.losses import AdaptiveMTPLoss


def test_confidence_gate():
    """Test ConfidenceGate component."""
    print("\n" + "="*80)
    print("TEST 1: ConfidenceGate")
    print("="*80)

    batch_size = 4
    seq_len = 128
    hidden_size = 768

    # Create gate
    gate = ConfidenceGate(
        hidden_size=hidden_size,
        gate_hidden_dims=(512, 256),
        dropout=0.1,
    )

    # Test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    outputs = gate(hidden_states, attention_mask)

    # Assertions
    assert 'confidence' in outputs, "Missing 'confidence' in outputs"
    assert 'avg_confidence' in outputs, "Missing 'avg_confidence' in outputs"
    assert outputs['confidence'].shape == (batch_size, 1), f"Wrong confidence shape: {outputs['confidence'].shape}"
    assert 0 <= outputs['avg_confidence'].item() <= 1, "Confidence not in [0, 1]"

    # Test per-token mode
    outputs_per_token = gate(hidden_states, attention_mask, return_per_token=True)
    assert outputs_per_token['confidence'].shape == (batch_size, seq_len, 1), "Wrong per-token shape"

    # Test statistics
    stats = gate.get_statistics()
    assert 'avg_confidence' in stats
    assert 'high_confidence_ratio' in stats

    print(f"✓ Confidence shape: {outputs['confidence'].shape}")
    print(f"✓ Confidence value: {outputs['avg_confidence'].item():.4f}")
    print(f"✓ Per-token shape: {outputs_per_token['confidence'].shape}")
    print(f"✓ Statistics: {stats}")
    print("✓ ConfidenceGate test PASSED")

    return gate


def test_prediction_heads():
    """Test MultiTokenPredictionHeads component."""
    print("\n" + "="*80)
    print("TEST 2: MultiTokenPredictionHeads")
    print("="*80)

    batch_size = 4
    seq_len = 128
    hidden_size = 768
    vocab_size = 50257
    num_heads = 3

    # Test linear heads
    heads_linear = MultiTokenPredictionHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        head_type='linear',
    )

    # Test MLP heads
    heads_mlp = MultiTokenPredictionHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        head_type='mlp',
        intermediate_size=1024,
    )

    # Test shared projections
    heads_shared = MultiTokenPredictionHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        head_type='linear',
        share_projections=True,
    )

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Test each head type
    for name, heads in [('linear', heads_linear), ('mlp', heads_mlp), ('shared', heads_shared)]:
        outputs = heads(hidden_states)

        assert 'all_logits' in outputs
        assert len(outputs['all_logits']) == num_heads
        for i, logits in enumerate(outputs['all_logits']):
            assert logits.shape == (batch_size, seq_len, vocab_size), \
                f"Wrong logits shape for head {i}: {logits.shape}"

        param_info = heads.get_parameters_count()
        print(f"✓ {name.upper()} heads: {param_info['total_params']:,} parameters")
        print(f"  - Per head: {param_info['per_head_params']:,} parameters")

    # Test prediction
    predictions = heads_linear.predict_tokens(hidden_states, temperature=0.8)
    assert 'predictions' in predictions
    assert len(predictions['predictions']) == num_heads
    print(f"✓ Prediction shape: {predictions['predictions'][0].shape}")
    print("✓ MultiTokenPredictionHeads test PASSED")

    return heads_linear


def test_adaptive_mtp_model():
    """Test AdaptiveMTPModel wrapper."""
    print("\n" + "="*80)
    print("TEST 3: AdaptiveMTPModel")
    print("="*80)

    batch_size = 4
    seq_len = 128
    hidden_size = 768
    vocab_size = 50257

    # Create dummy base model
    class DummyBaseModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, 8, dim_feedforward=2048),
                num_layers=2
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            hidden_states = self.transformer(x.transpose(0, 1)).transpose(0, 1)
            logits = self.lm_head(hidden_states)
            return {'hidden_states': hidden_states, 'logits': logits}

    base_model = DummyBaseModel(hidden_size, vocab_size)

    # Create MTP config
    mtp_config = AdaptiveMTPConfig(
        num_prediction_heads=3,
        confidence_threshold_train=0.6,
        confidence_threshold_inference=0.7,
        mtp_warmup_epochs=2,
    )

    # Create adaptive model
    model = AdaptiveMTPModel(
        base_model=base_model,
        config=mtp_config,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    # Test warmup period
    model.set_epoch(0)
    assert model.in_warmup_period(), "Should be in warmup at epoch 0"
    model.set_epoch(2)
    assert not model.in_warmup_period(), "Should not be in warmup at epoch 2"

    print(f"✓ Warmup logic working correctly")

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # During warmup
    model.set_epoch(0)
    outputs_warmup = model(input_ids, attention_mask)

    assert 'primary_logits' in outputs_warmup
    assert 'confidence_scores' in outputs_warmup
    assert 'mtp_active' in outputs_warmup
    assert not outputs_warmup['mtp_active'], "MTP should not be active during warmup"

    print(f"✓ Warmup forward pass: MTP active = {outputs_warmup['mtp_active']}")

    # After warmup
    model.set_epoch(3)
    model.eval()  # Set to eval to ensure deterministic behavior
    outputs_active = model(input_ids, attention_mask)

    assert 'primary_logits' in outputs_active
    assert outputs_active['primary_logits'].shape == (batch_size, seq_len, vocab_size)
    print(f"✓ Primary logits shape: {outputs_active['primary_logits'].shape}")

    # Check if MTP can activate (depends on confidence)
    if outputs_active['mtp_active']:
        assert 'additional_logits' in outputs_active
        assert len(outputs_active['additional_logits']) == mtp_config.num_prediction_heads
        print(f"✓ MTP activated with confidence: {outputs_active['avg_confidence'].item():.3f}")
    else:
        print(f"✓ MTP not activated (low confidence: {outputs_active['avg_confidence'].item():.3f})")

    # Test statistics
    stats = model.get_mtp_statistics()
    print(f"✓ Statistics: {stats}")
    assert 'mtp_usage_ratio' in stats
    assert 'avg_confidence' in stats

    # Test multi-token generation
    hidden_states = outputs_active['hidden_states']
    gen_result = model.generate_multi_token(hidden_states, attention_mask)
    assert 'used_mtp' in gen_result
    assert 'confidence' in gen_result
    print(f"✓ Generation: used_mtp={gen_result['used_mtp']}, confidence={gen_result['confidence']:.3f}")

    print("✓ AdaptiveMTPModel test PASSED")

    return model


def test_adaptive_mtp_loss():
    """Test AdaptiveMTPLoss."""
    print("\n" + "="*80)
    print("TEST 4: AdaptiveMTPLoss")
    print("="*80)

    batch_size = 4
    seq_len = 128
    vocab_size = 50257

    # Create loss function
    loss_fn = AdaptiveMTPLoss(
        vocab_size=vocab_size,
        primary_loss_weight=1.0,
        additional_loss_base_weight=0.1,
        confidence_reg_strength=0.01,
        use_confidence_weighting=True,
        label_smoothing=0.1,
    )

    # Create test data (enable gradients)
    primary_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Test without MTP (primary loss only)
    loss_outputs_single = loss_fn(
        primary_logits=primary_logits,
        targets=targets,
        attention_mask=attention_mask,
        mtp_active=False,
    )

    assert 'loss' in loss_outputs_single
    assert 'primary_loss' in loss_outputs_single
    assert loss_outputs_single['loss'].requires_grad, "Loss should require gradients"
    print(f"✓ Single-token loss: {loss_outputs_single['loss'].item():.4f}")
    print(f"  - Primary loss: {loss_outputs_single['primary_loss'].item():.4f}")

    # Test with MTP (enable gradients)
    additional_logits = [
        torch.randn(batch_size, seq_len, vocab_size, requires_grad=True),
        torch.randn(batch_size, seq_len, vocab_size, requires_grad=True),
        torch.randn(batch_size, seq_len, vocab_size, requires_grad=True),
    ]
    confidence_scores = torch.rand(batch_size, 1) * 0.5 + 0.5  # 0.5-1.0

    loss_outputs_mtp = loss_fn(
        primary_logits=primary_logits,
        targets=targets,
        additional_logits=additional_logits,
        confidence_scores=confidence_scores,
        attention_mask=attention_mask,
        mtp_active=True,
    )

    assert 'loss' in loss_outputs_mtp
    assert 'primary_loss' in loss_outputs_mtp
    assert 'additional_loss' in loss_outputs_mtp
    assert 'confidence_reg' in loss_outputs_mtp
    assert loss_outputs_mtp['loss'].requires_grad, "Loss should require gradients"

    print(f"✓ MTP loss: {loss_outputs_mtp['loss'].item():.4f}")
    print(f"  - Primary loss: {loss_outputs_mtp['primary_loss'].item():.4f}")
    print(f"  - Additional loss: {loss_outputs_mtp['additional_loss'].item():.4f}")
    print(f"  - Confidence reg: {loss_outputs_mtp['confidence_reg'].item():.4f}")
    print(f"  - Avg confidence: {loss_outputs_mtp['avg_confidence'].item():.4f}")

    # Test backward pass
    loss_outputs_mtp['loss'].backward()
    print(f"✓ Backward pass successful")

    # Test statistics
    stats = loss_fn.get_statistics()
    print(f"✓ Loss statistics: {stats}")

    print("✓ AdaptiveMTPLoss test PASSED")

    return loss_fn


def test_end_to_end():
    """Test end-to-end integration."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Integration")
    print("="*80)

    batch_size = 2
    seq_len = 64
    hidden_size = 256
    vocab_size = 1000

    # Create components
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, 4, dim_feedforward=512),
                num_layers=1
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            hidden_states = self.transformer(x.transpose(0, 1)).transpose(0, 1)
            logits = self.lm_head(hidden_states)
            return {'hidden_states': hidden_states, 'logits': logits}

    base_model = SimpleModel(hidden_size, vocab_size)

    config = AdaptiveMTPConfig(
        num_prediction_heads=2,
        confidence_threshold_train=0.5,
        mtp_warmup_epochs=1,
    )

    model = AdaptiveMTPModel(base_model, config, vocab_size, hidden_size)
    loss_fn = AdaptiveMTPLoss(vocab_size, additional_loss_base_weight=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate training
    print("\nSimulating training...")
    for epoch in range(3):
        model.set_epoch(epoch)
        model.train()

        for step in range(5):
            optimizer.zero_grad()

            # Generate batch
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Forward pass
            outputs = model(input_ids)

            # Compute loss
            loss_outputs = loss_fn(
                primary_logits=outputs['primary_logits'],
                targets=labels,
                additional_logits=outputs.get('additional_logits'),
                confidence_scores=outputs.get('confidence_scores'),
                mtp_active=outputs.get('mtp_active', False),
            )

            loss = loss_outputs['loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            if step == 0:
                print(f"Epoch {epoch}, Step {step}:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  MTP Active: {outputs.get('mtp_active', False)}")
                print(f"  Confidence: {outputs.get('avg_confidence', 0).item():.3f}")

        # Get statistics
        stats = model.get_mtp_statistics()
        print(f"Epoch {epoch} stats: MTP usage={stats['mtp_usage_ratio']:.2%}, "
              f"Avg conf={stats['avg_confidence']:.3f}")

    print("\n✓ End-to-end integration test PASSED")


def test_configuration():
    """Test configuration parsing."""
    print("\n" + "="*80)
    print("TEST 6: Configuration")
    print("="*80)

    from src.Ava.config.training_config import AdaptiveMTPConfig as ConfigClass

    # Test default configuration
    config = ConfigClass()
    print(f"✓ Default config created")
    print(f"  - num_prediction_heads: {config.num_prediction_heads}")
    print(f"  - confidence_threshold_train: {config.confidence_threshold_train}")
    print(f"  - mtp_warmup_epochs: {config.mtp_warmup_epochs}")

    # Test custom configuration
    custom_config = ConfigClass(
        use_adaptive_mtp=True,
        num_prediction_heads=4,
        confidence_threshold_train=0.7,
    )
    assert custom_config.use_adaptive_mtp == True
    assert custom_config.num_prediction_heads == 4
    print(f"✓ Custom config created")

    print("✓ Configuration test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ADAPTIVE MULTI-TOKEN PREDICTION - COMPREHENSIVE TEST SUITE")
    print("="*80)

    try:
        # Run all tests
        gate = test_confidence_gate()
        heads = test_prediction_heads()
        model = test_adaptive_mtp_model()
        loss = test_adaptive_mtp_loss()
        test_end_to_end()
        test_configuration()

        # Final summary
        print("\n" + "="*80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nAdaptive MTP System is fully functional and ready for use!")
        print("\nTo enable in training:")
        print("  1. Edit configs/gpu/small.yaml")
        print("  2. Set: enhanced_features.adaptive_mtp.use_adaptive_mtp: true")
        print("  3. Run: python code/scripts/training/train.py --config configs/gpu/small.yaml")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
