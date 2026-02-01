"""
Shared pytest fixtures for Ava tests.

This module provides common fixtures used across multiple test files,
reducing code duplication and ensuring consistent test setup.

Usage:
    Fixtures are automatically discovered by pytest when placed in conftest.py.
    Import is not needed - just use the fixture name as a parameter.

    def test_example(device, simple_model):
        model = simple_model.to(device)
        ...
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device for tests that must run on CPU."""
    return torch.device('cpu')


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple feedforward model for testing."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def simple_transformer_config():
    """Create a minimal transformer config for testing."""
    return {
        'vocab_size': 1000,
        'hidden_size': 64,
        'num_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'max_position_embeddings': 128,
    }


# =============================================================================
# Input Fixtures
# =============================================================================

@pytest.fixture
def simple_input():
    """Create simple input tensors for testing."""
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))
    return input_ids.float(), labels


@pytest.fixture
def batch_input():
    """Create a batch of inputs with attention masks."""
    batch_size = 4
    seq_len = 128
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'labels': torch.randint(0, 1000, (batch_size, seq_len)),
    }


# =============================================================================
# MoE-specific Fixtures
# =============================================================================

@pytest.fixture
def hidden_size():
    """Standard hidden size for MoE tests."""
    return 256


@pytest.fixture
def intermediate_size():
    """Standard intermediate size for MoE tests."""
    return 1024


@pytest.fixture
def num_experts():
    """Standard number of experts for MoE tests."""
    return 8


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32
