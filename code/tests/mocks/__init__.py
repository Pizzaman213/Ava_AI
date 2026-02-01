"""
Reusable mock objects for Ava unit tests.

This package provides mock implementations for testing components
without requiring actual CUDA, distributed training, or data loading.

Modules:
    cuda_mocks: Mock CUDA operations for CPU-safe testing
    model_mocks: Mock MoE models and components
    data_mocks: Mock dataloaders and datasets
    distributed_mocks: Mock distributed training operations
"""

from .cuda_mocks import (
    mock_cuda_context,
    MockCUDAStream,
    MockCUDAEvent,
    mock_triton_or_run,
)
from .model_mocks import (
    create_mock_moe_model,
    create_simple_moe_model,
    MockMoEForward,
    create_mock_router,
)
from .data_mocks import (
    create_mock_dataloader,
    create_sample_batch,
    MockDataset,
    MockIterableDataset,
)
from .distributed_mocks import (
    mock_distributed_context,
    MockProcessGroup,
    setup_mock_distributed,
)

__all__ = [
    # CUDA mocks
    'mock_cuda_context',
    'MockCUDAStream',
    'MockCUDAEvent',
    'mock_triton_or_run',
    # Model mocks
    'create_mock_moe_model',
    'create_simple_moe_model',
    'MockMoEForward',
    'create_mock_router',
    # Data mocks
    'create_mock_dataloader',
    'create_sample_batch',
    'MockDataset',
    'MockIterableDataset',
    # Distributed mocks
    'mock_distributed_context',
    'MockProcessGroup',
    'setup_mock_distributed',
]
