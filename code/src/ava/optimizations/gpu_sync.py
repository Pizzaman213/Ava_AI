"""
GPU synchronization utilities for minimizing cudaStreamSynchronize overhead.

This module provides utilities to batch GPU tensor→CPU scalar conversions,
reducing the number of expensive synchronization points during training.

Key Patterns:
    Instead of: loss.item(), grad_norm.item(), entropy.item()  # 3 syncs
    Use:        batcher.add('loss', loss); batcher.add('grad', grad); batcher.flush()  # 1 sync

    Instead of: [t.item() for t in tensors]  # N syncs
    Use:        batch_to_list(tensors)  # 1 sync

Performance Impact:
    Each cudaStreamSynchronize adds 10-50µs latency. Over millions of training
    steps with multiple metrics per step, this accumulates to significant overhead.
    Batching syncs can save 5-15% of training time for metric-heavy pipelines.

Example:
    >>> batcher = MetricsBatcher()
    >>> batcher.add('loss', loss_tensor)
    >>> batcher.add('grad_norm', grad_norm_tensor)
    >>> values = batcher.flush()  # ONE GPU→CPU sync
    >>> print(values)  # {'loss': 0.5, 'grad_norm': 1.2}
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class MetricsBatcher:
    """
    Batches GPU tensor→scalar conversions to minimize cudaStreamSynchronize overhead.

    Instead of calling .item() on each metric tensor separately (each causing a sync),
    this class collects all scalar tensors and extracts them in a single .tolist() call.

    Usage:
        batcher = MetricsBatcher()
        batcher.add('loss', loss_tensor)
        batcher.add('grad_norm', grad_norm_tensor)
        batcher.add('entropy', entropy_tensor)

        # ONE sync for all metrics
        values = batcher.flush()
        # values = {'loss': 0.5, 'grad_norm': 1.2, 'entropy': 0.8}

    Thread Safety:
        This class is NOT thread-safe. Use separate instances for different threads.
    """

    def __init__(self):
        """Initialize empty metrics batcher."""
        self._pending: List[tuple] = []  # List of (key, tensor) pairs

    def add(self, key: str, value: Union[torch.Tensor, float, int]) -> None:
        """
        Queue a scalar tensor or Python numeric for batched extraction.

        Args:
            key: Identifier for this metric
            value: A scalar tensor (0-dim or 1-element) on GPU, or Python float/int
        """
        if value is None:
            return
        # Handle both tensors and Python scalars
        if isinstance(value, (int, float)):
            # Python scalars don't need GPU sync, store directly
            self._pending.append((key, torch.tensor(value)))
        else:
            # Detach to avoid holding onto computation graph
            self._pending.append((key, value.detach()))

    def add_multi(self, prefix: str, tensors: Dict[str, torch.Tensor]) -> None:
        """
        Queue multiple tensors with a common prefix.

        Args:
            prefix: Prefix for all keys (e.g., 'grad' -> 'grad/norm', 'grad/max')
            tensors: Dict of name -> tensor pairs
        """
        for name, tensor in tensors.items():
            if tensor is not None:
                self.add(f"{prefix}/{name}", tensor.detach())

    def flush(self) -> Dict[str, float]:
        """
        Extract all queued tensors in a single GPU→CPU sync.

        Returns:
            Dictionary mapping keys to their scalar float values.
            Clears the internal queue after extraction.
        """
        if not self._pending:
            return {}

        keys = [k for k, _ in self._pending]
        # Ensure all tensors are scalar (0-dim) or squeeze to scalar
        tensors = []
        for _, t in self._pending:
            if t.dim() == 0:
                tensors.append(t.view(1))
            elif t.numel() == 1:
                tensors.append(t.view(1))
            else:
                # Multi-element tensor - take mean
                tensors.append(t.mean().view(1))

        # ONE sync point: concatenate and transfer to CPU
        try:
            stacked = torch.cat(tensors)
            values = stacked.tolist()
            del stacked  # Explicitly free GPU tensor after transfer
        except Exception:
            # Fallback: individual extraction if cat fails (mixed devices)
            def extract_value(t):
                if t.dim() == 0:
                    return t.item()
                elif t.numel() == 1:
                    return t.view(-1)[0].item()
                else:
                    return t.mean().item()
            values = [extract_value(t) for _, t in self._pending]

        # Explicitly free tensor references before clearing
        del tensors
        self._pending.clear()
        return dict(zip(keys, values))

    def pending_count(self) -> int:
        """Return number of pending tensors."""
        return len(self._pending)

    def clear(self) -> None:
        """Clear pending tensors without extracting."""
        self._pending.clear()


def batch_to_list(
    tensors: Union[torch.Tensor, List[torch.Tensor]],
    as_int: bool = False
) -> List[Union[float, int]]:
    """
    Convert GPU tensors to Python list with a single sync.

    Instead of: [t.item() for t in tensors]  # N syncs
    Use: batch_to_list(tensors)  # 1 sync

    Args:
        tensors: Either a 1D tensor or list of scalar tensors
        as_int: If True, return integers instead of floats

    Returns:
        List of scalar values

    Example:
        >>> losses = torch.tensor([0.5, 0.6, 0.7], device='cuda')
        >>> batch_to_list(losses)  # One sync
        [0.5, 0.6, 0.7]

        >>> token_ids = torch.tensor([101, 2023, 102], device='cuda')
        >>> batch_to_list(token_ids, as_int=True)
        [101, 2023, 102]
    """
    if isinstance(tensors, list):
        if not tensors:
            return []
        # Stack list of tensors into single tensor
        stacked = torch.stack([t.view(1) if t.dim() == 0 else t.view(-1) for t in tensors])
        stacked = stacked.view(-1)
    else:
        stacked = tensors.view(-1) if tensors.dim() > 0 else tensors.view(1)

    # Single GPU→CPU sync
    values = stacked.tolist()

    if as_int:
        return [int(v) for v in values]
    return values


def batch_mean_to_list(
    tensors: List[torch.Tensor],
) -> List[float]:
    """
    Compute mean of each tensor and return as list with single sync.

    Useful for per-example losses, entropies, etc.

    Args:
        tensors: List of tensors (any shape, will be reduced via mean)

    Returns:
        List of mean values

    Example:
        >>> per_token_losses = [torch.tensor([0.5, 0.6]), torch.tensor([0.4, 0.5, 0.3])]
        >>> batch_mean_to_list(per_token_losses)  # One sync
        [0.55, 0.4]
    """
    if not tensors:
        return []

    # Compute means on GPU
    means = torch.stack([t.mean() for t in tensors])

    # Single GPU→CPU sync
    return means.tolist()


def accumulate_and_sync(
    values: List[torch.Tensor],
    device: Optional[torch.device] = None
) -> List[float]:
    """
    Accumulate values on GPU and sync once at the end.

    Pattern for curriculum learning difficulty scoring, attention entropy, etc.

    Args:
        values: List of scalar GPU tensors to accumulate
        device: Device for accumulation (inferred from first value if None)

    Returns:
        List of accumulated values

    Example:
        >>> # Instead of: results = [v.item() for v in values]  # N syncs
        >>> results = accumulate_and_sync(values)  # 1 sync
    """
    if not values:
        return []

    # Stack and sync in one operation
    stacked = torch.stack(values)
    return stacked.tolist()


class DeferredMetrics:
    """
    Collect GPU metrics during training and extract them lazily.

    This is useful when metrics are computed at different points in the
    training loop but should only be synced to CPU at log intervals.

    Usage:
        deferred = DeferredMetrics()

        # During training step (no sync)
        deferred.record('loss', loss_tensor)
        deferred.record('grad_norm', grad_tensor)

        # At log interval (single sync)
        if step % log_interval == 0:
            metrics = deferred.extract_all()
            log_metrics(metrics)

    Note:
        Unlike MetricsBatcher which uses add/flush pattern, DeferredMetrics
        accumulates multiple values per key and provides aggregations.
    """

    def __init__(self):
        """Initialize deferred metrics collector."""
        self._metrics: Dict[str, List[torch.Tensor]] = {}

    def record(self, key: str, value: torch.Tensor) -> None:
        """
        Record a metric value (stays on GPU).

        Args:
            key: Metric name
            value: Scalar tensor
        """
        if value is None:
            return
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value.detach())

    def extract_all(self, reduce: str = 'mean') -> Dict[str, float]:
        """
        Extract all metrics with a single GPU→CPU sync.

        Args:
            reduce: Reduction method for multi-value metrics ('mean', 'sum', 'last')

        Returns:
            Dictionary of metric names to scalar values
        """
        if not self._metrics:
            return {}

        result = {}
        all_tensors = []
        all_keys = []

        for key, values in self._metrics.items():
            if not values:
                continue

            if reduce == 'last':
                all_tensors.append(values[-1])
            elif reduce == 'sum':
                all_tensors.append(torch.stack(values).sum())
            else:  # mean
                all_tensors.append(torch.stack(values).mean())

            all_keys.append(key)

        if all_tensors:
            # Single GPU→CPU sync
            stacked = torch.stack(all_tensors)
            values = stacked.tolist()
            result = dict(zip(all_keys, values))

        return result

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()

    def keys(self) -> List[str]:
        """Return list of recorded metric keys."""
        return list(self._metrics.keys())


__all__ = [
    'MetricsBatcher',
    'DeferredMetrics',
    'batch_to_list',
    'batch_mean_to_list',
    'accumulate_and_sync',
    # Gradient utilities (merged from gradients.py)
    'check_gradients',
    'check_gradients_deferred',
    'detect_gradient_issues',
    'ZERO_GRAD_CHECK_SAMPLE_RATE',
]


# ============================================================================
# Gradient Monitoring Utilities
# ============================================================================
# Originally from gradients.py - merged here for single-module GPU sync utilities
#
# GPU SYNC OPTIMIZATION: These functions batch all GPU->CPU transfers to minimize
# cudaStreamSynchronize calls. Instead of calling .item() per parameter (N syncs),
# we accumulate tensors on GPU and sync once at the end (1 sync).
#
# SAMPLING OPTIMIZATION: The zero-gradient check uses sampling (10% of params by default)
# to avoid full tensor scans on every parameter, saving 10-30ms per check.

import random
import torch.nn as nn

# Sampling rate for zero-gradient checks (0.1 = 10% of parameters)
# This avoids expensive .all() calls on every parameter
ZERO_GRAD_CHECK_SAMPLE_RATE = 0.1


def check_gradients(
    model: nn.Module,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Check gradient statistics and return metrics.

    Useful for detecting vanishing/exploding gradients during training.

    GPU SYNC FIX: Batches all GPU operations and syncs once at the end,
    instead of calling .item() per parameter which causes N separate syncs.

    Args:
        model: The model to check gradients for
        logger: Optional logger for detailed gradient reporting

    Returns:
        Dictionary containing gradient statistics:
            - total_norm: Sum of all gradient values
            - max_grad: Maximum gradient value
            - min_grad: Minimum gradient value
            - num_zero_grads: Number of parameters with near-zero gradients
            - num_params: Number of parameters with gradients
            - num_grads: Total number of gradient elements
    """
    grad_stats = {
        'total_norm': 0.0,
        'max_grad': 0.0,
        'min_grad': float('inf'),
        'num_zero_grads': 0,
        'num_params': 0,
        'num_grads': 0,
    }

    # GPU SYNC FIX: Collect all values on GPU first, then sync once
    sum_tensors: List[torch.Tensor] = []
    max_tensors: List[torch.Tensor] = []
    min_tensors: List[torch.Tensor] = []

    for param in model.parameters():
        if param.grad is not None:
            grad_stats['num_params'] += 1
            grad_val = param.grad.data.abs()

            # Accumulate on GPU - NO .item() calls here
            sum_tensors.append(grad_val.sum())
            max_tensors.append(grad_val.max())
            min_tensors.append(grad_val.min())
            grad_stats['num_grads'] += param.grad.numel()

            # SAMPLING OPTIMIZATION: Only check a subset of parameters for zero gradients
            # The .all() call forces a GPU sync, so we sample to reduce overhead
            if random.random() < ZERO_GRAD_CHECK_SAMPLE_RATE:
                if (param.grad.abs() < 1e-10).all():
                    # Scale up to estimate total (since we're sampling)
                    grad_stats['num_zero_grads'] += int(1.0 / ZERO_GRAD_CHECK_SAMPLE_RATE)

    # GPU SYNC FIX: Single sync point - stack and transfer all at once
    if sum_tensors:
        # Stack tensors and compute aggregates on GPU
        all_sums = torch.stack(sum_tensors)
        all_maxs = torch.stack(max_tensors)
        all_mins = torch.stack(min_tensors)

        # Compute final values on GPU
        total_norm = all_sums.sum()
        max_grad = all_maxs.max()
        min_grad = all_mins.min()

        # GPU SYNC FIX: Stack all 3 scalars and transfer with single .tolist()
        # This is 1 sync instead of 3 separate .item() syncs
        results = torch.stack([total_norm, max_grad, min_grad]).tolist()
        grad_stats['total_norm'] = results[0]
        grad_stats['max_grad'] = results[1]
        grad_stats['min_grad'] = results[2]

    if grad_stats['min_grad'] == float('inf'):
        grad_stats['min_grad'] = 0.0

    if logger is not None and grad_stats['num_params'] > 0:
        avg_grad = grad_stats['total_norm'] / grad_stats['num_grads'] if grad_stats['num_grads'] > 0 else 0.0
        logger.info(
            f"Gradient stats: avg={avg_grad:.2e}, max={grad_stats['max_grad']:.2e}, "
            f"min={grad_stats['min_grad']:.2e}, zero_grads={grad_stats['num_zero_grads']}, "
            f"params_with_grads={grad_stats['num_params']}"
        )

    return grad_stats


def check_gradients_deferred(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Compute gradient statistics but return GPU tensors instead of extracting to CPU.

    This function is designed to work with MetricsBatcher for single-sync extraction.
    It computes the same statistics as check_gradients() but keeps values on GPU.

    Args:
        model: The model to check gradients for

    Returns:
        Dictionary containing gradient statistics as GPU tensors:
            - grad_total_norm: Sum of all gradient absolute values
            - grad_max: Maximum gradient absolute value
            - grad_min: Minimum gradient absolute value
            - grad_num_params: Count of parameters with gradients (as tensor)
            - grad_num_grads: Total count of gradient elements (as tensor)
            - grad_num_zero: Count of parameters with near-zero gradients (as tensor)
    """
    # Collect gradient values on GPU
    sum_tensors: List[torch.Tensor] = []
    max_tensors: List[torch.Tensor] = []
    min_tensors: List[torch.Tensor] = []
    num_params = 0
    num_grads = 0
    num_zero_grads = 0
    device = None

    for param in model.parameters():
        if param.grad is not None:
            num_params += 1
            device = param.grad.device
            grad_val = param.grad.data.abs()

            sum_tensors.append(grad_val.sum())
            max_tensors.append(grad_val.max())
            min_tensors.append(grad_val.min())
            num_grads += param.grad.numel()

            # SAMPLING OPTIMIZATION: Only check subset of parameters for zero gradients
            if random.random() < ZERO_GRAD_CHECK_SAMPLE_RATE:
                if (param.grad.abs() < 1e-10).all():
                    num_zero_grads += int(1.0 / ZERO_GRAD_CHECK_SAMPLE_RATE)

    result: Dict[str, torch.Tensor] = {}

    if sum_tensors and device is not None:
        # Stack and compute aggregates on GPU (no sync)
        all_sums = torch.stack(sum_tensors)
        all_maxs = torch.stack(max_tensors)
        all_mins = torch.stack(min_tensors)

        result['grad_total_norm'] = all_sums.sum()
        result['grad_max'] = all_maxs.max()
        result['grad_min'] = all_mins.min()
        # Store counts as tensors for consistent batching
        result['grad_num_params'] = torch.tensor(float(num_params), device=device)
        result['grad_num_grads'] = torch.tensor(float(num_grads), device=device)
        result['grad_num_zero'] = torch.tensor(float(num_zero_grads), device=device)

    return result


def detect_gradient_issues(grad_stats: Dict[str, float]) -> Dict[str, bool]:
    """
    Detect potential gradient issues from statistics.

    Args:
        grad_stats: Output from check_gradients()

    Returns:
        Dictionary with detected issues:
            - vanishing: True if gradients are vanishing
            - exploding: True if gradients are exploding
            - zero_grads: True if many parameters have zero gradients
    """
    issues = {
        'vanishing': False,
        'exploding': False,
        'zero_grads': False,
    }

    if grad_stats['num_params'] == 0:
        return issues

    avg_grad = grad_stats['total_norm'] / grad_stats['num_grads'] if grad_stats['num_grads'] > 0 else 0.0

    # Detect vanishing gradients
    if avg_grad < 1e-8 or grad_stats['max_grad'] < 1e-7:
        issues['vanishing'] = True

    # Detect exploding gradients
    if grad_stats['max_grad'] > 1e3 or avg_grad > 1e2:
        issues['exploding'] = True

    # Detect too many zero gradients
    zero_ratio = grad_stats['num_zero_grads'] / grad_stats['num_params']
    if zero_ratio > 0.5:
        issues['zero_grads'] = True

    return issues
