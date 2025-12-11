"""
Test coherence measurement module.
"""

import sys
from pathlib import Path

# Add code/src to path for src.Ava imports
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir))

import torch
import torch.nn as nn


def test_coherence_metrics_import():
    """Test that coherence module imports correctly."""
    from src.Ava.evaluation.coherence import (
        CoherenceMetrics,
        CoherenceMeasurer,
        CoherenceConfig,
        measure_coherence,
        measure_batch_coherence,
    )
    print("✓ Coherence module imports successfully")


def test_coherence_metrics_dataclass():
    """Test CoherenceMetrics dataclass."""
    from src.Ava.evaluation.coherence import CoherenceMetrics

    metrics = CoherenceMetrics(
        perplexity=25.5,
        repetition_score=0.15,
        sentence_flow_score=0.82,
        topic_consistency=0.78,
        coherence_score=0.75,
        num_samples=10,
        avg_sequence_length=128.0,
        unique_token_ratio=0.65,
    )

    # Test to_dict
    metrics_dict = metrics.to_dict()
    assert 'coherence/perplexity' in metrics_dict
    assert 'coherence/overall_score' in metrics_dict
    assert metrics_dict['coherence/perplexity'] == 25.5
    assert metrics_dict['coherence/overall_score'] == 0.75

    print("✓ CoherenceMetrics dataclass works correctly")


def test_coherence_config():
    """Test CoherenceConfig dataclass."""
    from src.Ava.evaluation.coherence import CoherenceConfig

    # Default config
    config = CoherenceConfig()
    assert config.enabled is True
    assert config.eval_every_n_steps == 500
    assert config.num_samples == 10
    assert config.temperature == 0.8

    # Custom config
    custom_config = CoherenceConfig(
        enabled=True,
        eval_every_n_steps=250,
        num_samples=5,
        temperature=0.7,
    )
    assert custom_config.eval_every_n_steps == 250
    assert custom_config.num_samples == 5

    print("✓ CoherenceConfig works correctly")


class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""

    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, output_hidden_states=False):
        h = self.embedding(x)
        hidden_states = [h]
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)

        logits = self.output_proj(h)

        if output_hidden_states:
            return type('Output', (), {
                'logits': logits,
                'hidden_states': hidden_states
            })()
        return logits


def test_coherence_measurer_basic():
    """Test basic CoherenceMeasurer functionality."""
    from src.Ava.evaluation.coherence import CoherenceMeasurer, CoherenceConfig

    # Create simple model
    model = SimpleTransformer(vocab_size=1000, hidden_size=64)
    device = torch.device('cpu')

    config = CoherenceConfig(
        num_samples=2,
        max_generation_length=32,
    )

    measurer = CoherenceMeasurer(
        model=model,
        tokenizer=None,
        config=config,
        device=device,
    )

    # Create some fake input
    input_ids = torch.randint(0, 1000, (4, 64))

    # Measure coherence
    metrics = measurer.measure(input_ids=input_ids)

    # Check metrics are reasonable
    assert metrics.perplexity > 0, "Perplexity should be positive"
    assert 0 <= metrics.repetition_score <= 1, "Repetition should be 0-1"
    assert 0 <= metrics.coherence_score <= 1, "Coherence score should be 0-1"
    assert metrics.num_samples == 4, "Should have 4 samples"

    print(f"✓ CoherenceMeasurer basic test passed")
    print(f"  - Perplexity: {metrics.perplexity:.2f}")
    print(f"  - Repetition: {metrics.repetition_score:.4f}")
    print(f"  - Coherence: {metrics.coherence_score:.4f}")


def test_repetition_score():
    """Test repetition score calculation."""
    from src.Ava.evaluation.coherence import CoherenceMeasurer, CoherenceConfig

    model = SimpleTransformer(vocab_size=1000, hidden_size=64)
    config = CoherenceConfig()
    measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

    # Highly repetitive sequence (same token repeated)
    repetitive = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    rep_score = measurer._compute_repetition_score(repetitive)
    assert rep_score > 0.5, f"Repetitive sequence should have high rep score, got {rep_score}"

    # More varied sequence
    varied = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    varied_score = measurer._compute_repetition_score(varied)
    assert varied_score < rep_score, "Varied sequence should have lower rep score"

    print(f"✓ Repetition score calculation works")
    print(f"  - Repetitive: {rep_score:.4f}")
    print(f"  - Varied: {varied_score:.4f}")


def test_perplexity_calculation():
    """Test perplexity calculation."""
    from src.Ava.evaluation.coherence import CoherenceMeasurer, CoherenceConfig

    model = SimpleTransformer(vocab_size=1000, hidden_size=64)
    config = CoherenceConfig()
    measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

    # Random input
    input_ids = torch.randint(0, 1000, (2, 32))
    ppl = measurer._compute_perplexity(input_ids)

    assert ppl > 0, "Perplexity should be positive"
    assert ppl < 10000, "Perplexity should be reasonable for random model"

    print(f"✓ Perplexity calculation works: {ppl:.2f}")


def test_unique_ratio():
    """Test unique token ratio calculation."""
    from src.Ava.evaluation.coherence import CoherenceMeasurer, CoherenceConfig

    model = SimpleTransformer(vocab_size=1000, hidden_size=64)
    config = CoherenceConfig()
    measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

    # All unique tokens
    unique = torch.arange(16).unsqueeze(0)
    unique_ratio = measurer._compute_unique_ratio(unique)
    assert unique_ratio == 1.0, f"All unique should be 1.0, got {unique_ratio}"

    # All same token
    same = torch.ones(1, 16, dtype=torch.long)
    same_ratio = measurer._compute_unique_ratio(same)
    assert same_ratio == 1/16, f"All same should be 1/16, got {same_ratio}"

    print(f"✓ Unique ratio calculation works")


def test_config_from_training_config():
    """Test that CoherenceConfig works with training config."""
    from src.Ava.config.training_config import CoherenceConfig as TrainingCoherenceConfig

    # Create config from training_config module
    config = TrainingCoherenceConfig(
        enabled=True,
        eval_every_n_steps=250,
        num_samples=5,
        log_to_wandb=True,
    )

    assert config.enabled is True
    assert config.eval_every_n_steps == 250
    assert config.log_to_wandb is True

    print("✓ TrainingCoherenceConfig works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Coherence Measurement Module")
    print("=" * 60)

    tests = [
        test_coherence_metrics_import,
        test_coherence_metrics_dataclass,
        test_coherence_config,
        test_coherence_measurer_basic,
        test_repetition_score,
        test_perplexity_calculation,
        test_unique_ratio,
        test_config_from_training_config,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
