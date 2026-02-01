"""
Mock distributed training operations for single-GPU testing.

These mocks allow testing distributed training code paths without
actually running multiple processes or GPUs.
"""

import contextlib
from contextlib import contextmanager
from typing import Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.distributed as dist


class MockProcessGroup:
    """Mock process group for distributed operations."""

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        backend: str = 'nccl',
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self._name = f"mock_pg_{rank}"

    def size(self) -> int:
        return self.world_size

    def rank(self) -> int:
        return self.rank


def setup_mock_distributed(
    rank: int = 0,
    world_size: int = 1,
    backend: str = 'nccl',
) -> dict:
    """
    Set up mock distributed environment variables.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend

    Returns:
        Dictionary of environment variables that were set
    """
    import os

    env_vars = {
        'RANK': str(rank),
        'LOCAL_RANK': str(rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355',
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    return env_vars


def cleanup_mock_distributed() -> None:
    """Clean up mock distributed environment variables."""
    import os

    for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        os.environ.pop(key, None)


@contextmanager
def mock_distributed_context(
    rank: int = 0,
    world_size: int = 1,
    is_initialized: bool = True,
):
    """
    Context manager for mocking distributed operations.

    Usage:
        with mock_distributed_context(rank=0, world_size=4):
            # dist.is_initialized() returns True
            # dist.get_rank() returns 0
            # dist.get_world_size() returns 4
            train_distributed()

    Args:
        rank: Process rank
        world_size: Total number of processes
        is_initialized: Whether distributed is considered initialized
    """
    mock_pg = MockProcessGroup(rank=rank, world_size=world_size)

    def mock_all_reduce(tensor, op=None, group=None, async_op=False):
        """Mock all_reduce that returns the tensor unchanged."""
        if async_op:
            work = Mock()
            work.wait = Mock()
            return work
        return None

    def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
        """Mock all_gather that duplicates the tensor."""
        for t in tensor_list:
            t.copy_(tensor)
        if async_op:
            work = Mock()
            work.wait = Mock()
            return work
        return None

    def mock_broadcast(tensor, src=0, group=None, async_op=False):
        """Mock broadcast that does nothing."""
        if async_op:
            work = Mock()
            work.wait = Mock()
            return work
        return None

    def mock_barrier(group=None, async_op=False, device_ids=None, timeout=None):
        """Mock barrier that does nothing."""
        if async_op:
            work = Mock()
            work.wait = Mock()
            return work
        return None

    patches = [
        patch('torch.distributed.is_initialized', return_value=is_initialized),
        patch('torch.distributed.is_available', return_value=True),
        patch('torch.distributed.get_rank', return_value=rank),
        patch('torch.distributed.get_world_size', return_value=world_size),
        patch('torch.distributed.get_backend', return_value='nccl'),
        patch('torch.distributed.all_reduce', side_effect=mock_all_reduce),
        patch('torch.distributed.all_gather', side_effect=mock_all_gather),
        patch('torch.distributed.broadcast', side_effect=mock_broadcast),
        patch('torch.distributed.barrier', side_effect=mock_barrier),
        patch('torch.distributed.new_group', return_value=mock_pg),
    ]

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        setup_mock_distributed(rank, world_size)
        try:
            yield mock_pg
        finally:
            cleanup_mock_distributed()


class MockDDP(torch.nn.Module):
    """
    Mock DistributedDataParallel wrapper.

    Wraps a model without actually setting up distributed training.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self.device_ids = device_ids or [0]
        self.output_device = output_device or self.device_ids[0]
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode: bool = True):
        self.training = mode
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)


class MockFSDP(torch.nn.Module):
    """
    Mock FullyShardedDataParallel wrapper.

    Wraps a model without actually setting up FSDP.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self._fsdp_wrapped_module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)


@contextmanager
def mock_ddp_context():
    """
    Context manager that replaces DDP with MockDDP.

    Usage:
        with mock_ddp_context():
            model = DDP(model)  # Actually creates MockDDP
    """
    with patch('torch.nn.parallel.DistributedDataParallel', MockDDP):
        yield


@contextmanager
def mock_fsdp_context():
    """
    Context manager that replaces FSDP with MockFSDP.

    Usage:
        with mock_fsdp_context():
            model = FSDP(model)  # Actually creates MockFSDP
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        with patch('torch.distributed.fsdp.FullyShardedDataParallel', MockFSDP):
            yield
    except ImportError:
        # FSDP not available, just yield
        yield


class MockGradBucket:
    """Mock gradient bucket for testing communication hooks."""

    def __init__(
        self,
        index: int = 0,
        tensors: Optional[List[torch.Tensor]] = None,
    ):
        self._index = index
        self._tensors = tensors or [torch.randn(100)]

    def index(self) -> int:
        return self._index

    def buffer(self) -> torch.Tensor:
        return torch.cat([t.flatten() for t in self._tensors])

    def gradients(self) -> List[torch.Tensor]:
        return self._tensors

    def parameters(self) -> List[torch.Tensor]:
        return self._tensors

    def is_last(self) -> bool:
        return True

    def set_buffer(self, buffer: torch.Tensor) -> None:
        pass


def create_mock_gradient_hook():
    """
    Create a mock gradient communication hook.

    Returns:
        A hook function that can be registered with DDP
    """
    def hook(state: Any, bucket: MockGradBucket) -> torch.futures.Future:
        future = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    return hook


class MockWork:
    """Mock distributed work handle."""

    def __init__(self, result: Any = None):
        self._result = result
        self._done = True

    def wait(self, timeout: Optional[float] = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return self._done

    def get_future(self) -> 'torch.futures.Future':
        future = torch.futures.Future()
        future.set_result(self._result)
        return future
