"""
Test suite for UnifiedLoss module
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from src.Ava.losses import UnifiedLoss, create_unified_loss


def test_unified_loss_import():
    """Test that UnifiedLoss can be imported."""
    assert UnifiedLoss is not None
    assert create_unified_loss is not None


def test_standard_loss():
    """Test standard cross-entropy loss."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="standard"
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert loss.requires_grad


def test_deepseek_loss():
    """Test DeepSeek-style loss."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="deepseek",
        initial_temperature=1.0,
        adaptive_temperature=True,
        label_smoothing=0.1
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_unified_loss_with_mtp():
    """Test unified loss with multi-token prediction."""
    vocab_size = 100
    hidden_size = 64
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        primary_loss_type="deepseek",
        use_mtp=True,
        num_future_tokens=3,
        mtp_weight=0.1,
        mtp_type="deepseek"
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(
        logits=logits,
        targets=targets,
        hidden_states=hidden_states
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_unified_loss_with_repetition_penalties():
    """Test unified loss with repetition penalties."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="standard",
        use_ngram_penalty=True,
        ngram_size=3,
        ngram_penalty_weight=0.1,
        use_immediate_repetition_penalty=True,
        immediate_repetition_weight=0.5
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_unified_loss_with_moe():
    """Test unified loss with MoE balancing."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    num_experts = 4
    top_k = 2

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="standard",
        num_experts=num_experts,
        use_moe_balancing=True,
        gradient_balance_weight=0.1
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    gate_logits = torch.randn(batch_size * seq_len, num_experts)
    expert_indices = torch.randint(0, num_experts, (batch_size * seq_len, top_k))

    loss = loss_fn(
        logits=logits,
        targets=targets,
        gate_logits=gate_logits,
        expert_indices=expert_indices
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_detailed_loss_output():
    """Test detailed loss output."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="deepseek",
        use_ngram_penalty=True,
        use_immediate_repetition_penalty=True
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss_dict = loss_fn(logits, targets, return_detailed=True)

    assert isinstance(loss_dict, dict)
    assert 'total_loss' in loss_dict
    assert 'main_loss' in loss_dict
    assert 'ngram_penalty' in loss_dict
    assert 'immediate_repetition_penalty' in loss_dict
    assert isinstance(loss_dict['total_loss'], torch.Tensor)


def test_create_unified_loss_from_config():
    """Test creating unified loss from config dict."""
    config = {
        'vocab_size': 100,
        'hidden_size': 64,
        'primary_loss_type': 'deepseek',
        'use_mtp': True,
        'num_future_tokens': 3,
        'use_ngram_penalty': True
    }

    loss_fn = create_unified_loss(config)

    assert isinstance(loss_fn, UnifiedLoss)
    assert loss_fn.vocab_size == 100
    assert loss_fn.use_mtp == True
    assert loss_fn.use_ngram_penalty == True


def test_loss_statistics():
    """Test loss statistics retrieval."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    num_experts = 4

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        num_experts=num_experts,
        use_moe_balancing=True
    )

    stats = loss_fn.get_loss_statistics()

    assert isinstance(stats, dict)

    # Reset statistics should not raise error
    loss_fn.reset_statistics()


def test_backward_pass():
    """Test that loss supports backward pass."""
    vocab_size = 100
    hidden_size = 64
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        primary_loss_type="deepseek",
        use_mtp=True,
        use_ngram_penalty=True
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(
        logits=logits,
        targets=targets,
        hidden_states=hidden_states
    )

    # Backward pass should work
    loss.backward()

    assert logits.grad is not None
    assert hidden_states.grad is not None


def test_attention_mask():
    """Test loss with attention mask."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="standard"
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # Mask half the sequence

    loss = loss_fn(logits, targets, attention_mask=attention_mask)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_eos_penalty():
    """Test EOS penalty."""
    vocab_size = 100
    batch_size = 2
    seq_len = 30
    eos_token_id = 99

    loss_fn = UnifiedLoss(
        vocab_size=vocab_size,
        primary_loss_type="deepseek",
        eos_token_id=eos_token_id,
        min_sequence_length=20,
        eos_penalty_weight=0.1
    )

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Place EOS early in sequence (should be penalized)
    targets[0, 5] = eos_token_id

    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


if __name__ == '__main__':
    # Run tests
    print("Running UnifiedLoss tests...")

    test_unified_loss_import()
    print("✓ Import test passed")

    test_standard_loss()
    print("✓ Standard loss test passed")

    test_deepseek_loss()
    print("✓ DeepSeek loss test passed")

    test_unified_loss_with_mtp()
    print("✓ MTP test passed")

    test_unified_loss_with_repetition_penalties()
    print("✓ Repetition penalties test passed")

    test_unified_loss_with_moe()
    print("✓ MoE balancing test passed")

    test_detailed_loss_output()
    print("✓ Detailed output test passed")

    test_create_unified_loss_from_config()
    print("✓ Config creation test passed")

    test_loss_statistics()
    print("✓ Statistics test passed")

    test_backward_pass()
    print("✓ Backward pass test passed")

    test_attention_mask()
    print("✓ Attention mask test passed")

    test_eos_penalty()
    print("✓ EOS penalty test passed")

    print("\nAll tests passed! ✓")
