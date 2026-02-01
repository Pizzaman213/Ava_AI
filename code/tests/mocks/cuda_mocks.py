"""
Mock CUDA operations for CPU-safe testing.

This module provides mock implementations of CUDA operations that allow
tests to run without actual GPU hardware. All operations are designed
to be transparent - they accept the same inputs and return similar outputs.
"""

import contextlib
from contextlib import ExitStack, contextmanager
from typing import Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import torch


class MockCUDAStream:
    """Mock CUDA stream that does nothing."""

    def __init__(self, device: Optional[int] = None, priority: int = 0):
        self.device = device
        self.priority = priority
        self._is_recording = False

    def synchronize(self) -> None:
        """No-op synchronize."""
        pass

    def wait_stream(self, other: 'MockCUDAStream') -> None:
        """No-op wait."""
        pass

    def record_event(self, event: Optional['MockCUDAEvent'] = None) -> 'MockCUDAEvent':
        """Record a mock event."""
        return event or MockCUDAEvent()

    def wait_event(self, event: 'MockCUDAEvent') -> None:
        """No-op wait for event."""
        pass

    def __enter__(self) -> 'MockCUDAStream':
        return self

    def __exit__(self, *args) -> None:
        pass

    def query(self) -> bool:
        """Stream is always ready."""
        return True


class MockCUDAEvent:
    """Mock CUDA event that does nothing."""

    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self.enable_timing = enable_timing
        self.blocking = blocking
        self._recorded = False

    def record(self, stream: Optional[MockCUDAStream] = None) -> None:
        """Record event."""
        self._recorded = True

    def wait(self, stream: Optional[MockCUDAStream] = None) -> None:
        """No-op wait."""
        pass

    def synchronize(self) -> None:
        """No-op synchronize."""
        pass

    def query(self) -> bool:
        """Event is always ready."""
        return True

    def elapsed_time(self, end_event: 'MockCUDAEvent') -> float:
        """Return mock elapsed time (0.0)."""
        return 0.0


class MockCUDAGraph:
    """Mock CUDA graph for testing."""

    def __init__(self):
        self._captured = False

    def capture_begin(self, **kwargs) -> None:
        self._captured = True

    def capture_end(self) -> None:
        pass

    def replay(self) -> None:
        pass

    def reset(self) -> None:
        self._captured = False


@contextmanager
def mock_cuda_context(force_cpu: bool = True):
    """
    Context manager that mocks all CUDA operations for CPU testing.

    Usage:
        with mock_cuda_context():
            # All CUDA operations are mocked
            model = MyModel()  # Won't try to use GPU

    Args:
        force_cpu: If True, torch.cuda.is_available() returns False
    """
    patches = [
        patch('torch.cuda.is_available', return_value=not force_cpu),
        patch('torch.cuda.empty_cache'),
        patch('torch.cuda.synchronize'),
        patch('torch.cuda.Stream', MockCUDAStream),
        patch('torch.cuda.Event', MockCUDAEvent),
        patch('torch.cuda.memory_allocated', return_value=0),
        patch('torch.cuda.memory_reserved', return_value=0),
        patch('torch.cuda.max_memory_allocated', return_value=0),
        patch('torch.cuda.max_memory_reserved', return_value=0),
        patch('torch.cuda.reset_peak_memory_stats'),
        patch('torch.cuda.current_device', return_value=0),
        patch('torch.cuda.device_count', return_value=1 if not force_cpu else 0),
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@contextmanager
def mock_triton_or_run(use_reference: bool = True):
    """
    Context manager for testing Triton kernels.

    Either runs the Triton kernel (if GPU available) or falls back
    to a PyTorch reference implementation for testing.

    Args:
        use_reference: If True, always use PyTorch reference

    Usage:
        with mock_triton_or_run():
            result = my_triton_kernel(x)  # Uses reference on CPU
    """
    if use_reference or not torch.cuda.is_available():
        # Patch Triton availability flag
        with patch.dict('sys.modules', {'triton': MagicMock(), 'triton.language': MagicMock()}):
            yield False  # Indicates reference was used
    else:
        yield True  # Indicates Triton was used


class MockMemoryCache:
    """Mock memory cache for GPU memory tracking tests."""

    def __init__(
        self,
        allocated_gb: float = 4.0,
        reserved_gb: float = 6.0,
        total_gb: float = 24.0,
    ):
        self.allocated_gb = allocated_gb
        self.reserved_gb = reserved_gb
        self.total_gb = total_gb

    def get_stats(self, force_refresh: bool = False) -> dict:
        return {
            'allocated_gb': self.allocated_gb,
            'reserved_gb': self.reserved_gb,
            'total_gb': self.total_gb,
            'utilization': self.reserved_gb / self.total_gb,
        }

    def get_utilization(self, force_refresh: bool = False) -> float:
        return self.reserved_gb / self.total_gb

    def invalidate(self) -> None:
        pass


def create_mock_cuda_device_properties(
    name: str = "Mock GPU",
    total_memory: int = 24 * 1024**3,  # 24 GB
    major: int = 8,
    minor: int = 6,
    multi_processor_count: int = 82,
) -> Mock:
    """Create a mock CUDA device properties object."""
    props = Mock()
    props.name = name
    props.total_memory = total_memory
    props.major = major
    props.minor = minor
    props.multi_processor_count = multi_processor_count
    return props


@contextmanager
def mock_autocast():
    """
    Mock torch.cuda.amp.autocast for CPU testing.

    Provides a no-op autocast context that works on CPU.
    """
    # On CPU, autocast should be a no-op
    yield


def mock_amp_scaler() -> Mock:
    """
    Create a mock GradScaler for AMP testing.

    Returns:
        Mock GradScaler with functional methods
    """
    scaler = Mock()

    def scale_fn(loss):
        return loss

    def step_fn(optimizer, *args, **kwargs):
        optimizer.step()

    def update_fn():
        pass

    def unscale_fn(optimizer):
        pass

    def get_scale_fn():
        return 1.0

    scaler.scale = Mock(side_effect=scale_fn)
    scaler.step = Mock(side_effect=step_fn)
    scaler.update = Mock(side_effect=update_fn)
    scaler.unscale_ = Mock(side_effect=unscale_fn)
    scaler.get_scale = Mock(side_effect=get_scale_fn)
    scaler.is_enabled = Mock(return_value=True)

    return scaler
