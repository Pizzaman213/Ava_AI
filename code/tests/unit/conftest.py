"""
Shared pytest fixtures for Ava unit tests.

This module provides comprehensive fixtures used across all unit test files.
Fixtures are organized by category for easy discovery and maintenance.

Categories:
    - Device fixtures: CPU/CUDA device management
    - Config fixtures: Training and model configurations
    - Model fixtures: Mock and simple MoE models
    - Data fixtures: Dataloaders and sample batches
    - Context fixtures: TrainingContext and pipeline setup
    - Optimizer fixtures: Optimizer and scheduler setup
    - Distributed fixtures: Mock distributed environment
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TESTS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

# Import mocks using relative imports
from mocks.cuda_mocks import (
    mock_cuda_context,
    MockCUDAStream,
    MockCUDAEvent,
    mock_amp_scaler,
)
from mocks.model_mocks import (
    MockMoEConfig,
    SimpleMoEModel,
    create_mock_moe_model,
    create_simple_moe_model,
)
from mocks.data_mocks import (
    MockDataset,
    MockTokenizer,
    create_mock_dataloader,
    create_sample_batch,
)
from mocks.distributed_mocks import (
    mock_distributed_context,
    MockDDP,
    MockProcessGroup,
)


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (prefers CPU for unit tests)."""
    return torch.device('cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device."""
    return torch.device('cpu')


@pytest.fixture
def cuda_device():
    """Get CUDA device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def test_device():
    """
    Get test device per development guidelines (RTX 3060 only).

    Falls back to CPU if running in CI or if CUDA not available.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # Check for RTX 3060 (device 1 per guidelines)
    if torch.cuda.device_count() > 1:
        return torch.device('cuda:1')

    return torch.device('cuda:0')


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal training configuration for fast tests."""
    return {
        'model': {
            'vocab_size': 1000,
            'hidden_size': 64,
            'num_layers': 2,
            'num_attention_heads': 2,
            'intermediate_size': 256,
            'max_position_embeddings': 128,
            'num_experts': 4,
            'num_experts_per_token': 2,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'bos_token_id': 2,
            'gradient_checkpointing': False,
            'use_flash_attention': False,
        },
        'training': {
            'batching': {
                'batch_size': 4,
                'gradient_accumulation_steps': 1,
            },
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
            },
            'schedule': {
                'num_epochs': 1,
                'warmup_steps': 10,
            },
            'precision': {
                'mixed_precision': 'fp32',  # CPU-safe
            },
        },
        'data': {
            'max_length': 64,
            'num_workers': 0,
        },
        'compute': {
            'device': {
                'type': 'cpu',
            },
        },
        'logging': {
            'log_interval': 10,
        },
    }


@pytest.fixture
def moe_config() -> MockMoEConfig:
    """MoE model configuration for testing."""
    return MockMoEConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_token=2,
    )


@pytest.fixture
def full_config(minimal_config) -> Dict[str, Any]:
    """Full training configuration with all options."""
    config = minimal_config.copy()
    config['training']['validation'] = {
        'enabled': True,
        'batch_size': 4,
        'max_batches': 5,
    }
    config['training']['generation'] = {
        'enabled': True,
        'generate_every_n_steps': 100,
    }
    config['checkpoints'] = {
        'save_steps': 100,
        'keep_last_n': 2,
    }
    return config


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def simple_model(device):
    """Create a simple feedforward model for basic tests."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    return model.to(device)


@pytest.fixture
def simple_moe_model(moe_config, device):
    """Create a simple but functional MoE model."""
    model = create_simple_moe_model(config=moe_config, device=device)
    return model


@pytest.fixture
def mock_moe_model(device):
    """Create a fully mocked MoE model."""
    return create_mock_moe_model(device=device)


@pytest.fixture
def simple_transformer_model(minimal_config, device):
    """Create a simple transformer model for testing."""
    config = minimal_config['model']
    model = SimpleMoEModel(MockMoEConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_attention_heads=config['num_attention_heads'],
    ))
    return model.to(device)


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_batch(device):
    """Create a sample training batch."""
    return create_sample_batch(
        batch_size=4,
        seq_len=64,
        vocab_size=1000,
        device=device,
    )


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader for testing."""
    return create_mock_dataloader(
        num_batches=10,
        batch_size=4,
        seq_len=64,
    )


@pytest.fixture
def mock_dataset():
    """Create a mock Dataset."""
    return MockDataset(size=100, seq_len=64, vocab_size=1000)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer(vocab_size=1000)


# =============================================================================
# Context Fixtures
# =============================================================================

@pytest.fixture
def training_context(simple_moe_model, device, minimal_config):
    """Create a TrainingContext for testing."""
    from ava.training.context import TrainingContext

    context = TrainingContext(
        model=simple_moe_model,
        device=device,
        config=minimal_config,
        rank=0,
        world_size=1,
        is_main_process=True,
    )
    context.update_from_config(minimal_config)
    return context


@pytest.fixture
def training_context_with_optimizer(training_context):
    """Create a TrainingContext with optimizer and scheduler."""
    import torch.optim as optim

    optimizer = optim.AdamW(
        training_context.model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000
    )

    training_context.optimizer = optimizer
    training_context.scheduler = scheduler

    return training_context


# =============================================================================
# Optimizer Fixtures
# =============================================================================

@pytest.fixture
def optimizer(simple_moe_model):
    """Create an optimizer for testing."""
    return torch.optim.AdamW(simple_moe_model.parameters(), lr=1e-4)


@pytest.fixture
def scheduler(optimizer):
    """Create a scheduler for testing."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)


@pytest.fixture
def grad_scaler():
    """Create a mock gradient scaler for AMP testing."""
    return mock_amp_scaler()


# =============================================================================
# Pipeline Fixtures
# =============================================================================

@pytest.fixture
def training_pipeline(training_context):
    """Create a TrainingPipeline for testing."""
    from ava.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(training_context)
    return pipeline


# =============================================================================
# Distributed Fixtures
# =============================================================================

@pytest.fixture
def mock_distributed():
    """Set up mock distributed environment."""
    with mock_distributed_context(rank=0, world_size=1):
        yield


@pytest.fixture
def multi_gpu_distributed():
    """Set up mock multi-GPU distributed environment."""
    with mock_distributed_context(rank=0, world_size=4):
        yield


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = temp_dir / 'checkpoints'
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create a temporary directory for training outputs."""
    output_dir = temp_dir / 'outputs'
    output_dir.mkdir()
    return output_dir


# =============================================================================
# CUDA Mock Fixtures
# =============================================================================

@pytest.fixture
def no_cuda():
    """Context that mocks CUDA as unavailable."""
    with mock_cuda_context(force_cpu=True):
        yield


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return 42


@pytest.fixture
def deterministic(seed):
    """Enable deterministic mode for reproducible tests."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (use -m 'not slow' to skip)")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "distributed: mark test as requiring distributed setup")
    config.addinivalue_line("markers", "integration: mark test as integration test")


# =============================================================================
# Auto-use Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    torch.manual_seed(42)
    yield
