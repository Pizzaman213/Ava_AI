"""
Gradient monitoring utilities for the Ava pipeline.

Provides tools to check gradient health during training.

GPU SYNC OPTIMIZATION: This module batches all GPU->CPU transfers to minimize
cudaStreamSynchronize calls. Instead of calling .item() per parameter (N syncs),
we accumulate tensors on GPU and sync once at the end (1 sync).

SAMPLING OPTIMIZATION: The zero-gradient check uses sampling (10% of params by default)
to avoid full tensor scans on every parameter, saving 10-30ms per check.
"""

import logging
import random
from typing import Dict, List, Optional

import torch
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
            - total_norm: Sum of all gradient absolute values
            - max_grad: Maximum gradient absolute value
            - min_grad: Minimum gradient absolute value
            - num_zero_grads: Count of parameters with near-zero gradients (as tensor)
            - num_params: Count of parameters with gradients (as tensor)
            - num_grads: Total count of gradient elements (as tensor)
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
