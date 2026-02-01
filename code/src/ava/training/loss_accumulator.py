"""
Numerically stable loss accumulation for training.

This module provides the KahanAccumulator class which uses Kahan summation
for numerically stable loss accumulation during training.

Kahan summation tracks accumulated floating-point error and compensates for it,
achieving near-full precision regardless of the number of additions. This is
important for long training runs with many micro-batches.

Example:
    >>> accumulator = KahanAccumulator()
    >>> for loss in losses:
    ...     accumulator.add(loss)
    >>> mean_loss = accumulator.get_mean()
"""

from typing import Optional

import torch


class KahanAccumulator:
    """
    Kahan summation algorithm for numerically stable loss accumulation.

    Standard floating-point addition accumulates error over many operations.
    For long training runs with many micro-batches, this can result in
    noticeable drift in reported loss values.

    Kahan summation tracks the accumulated error and compensates for it,
    achieving near-full precision regardless of the number of additions.

    Example:
        >>> accumulator = KahanAccumulator()
        >>> for loss in losses:
        ...     accumulator.add(loss)
        >>> mean_loss = accumulator.get_mean()
    """

    def __init__(self):
        """Initialize empty accumulator."""
        self.sum: Optional[torch.Tensor] = None
        self.compensation: Optional[torch.Tensor] = None
        self.count: int = 0

    def add(self, value: torch.Tensor) -> None:
        """
        Add a value to the accumulator using Kahan summation.

        Args:
            value: Tensor value to add (should be scalar or will be summed)
        """
        if value.numel() != 1:
            value = value.mean()

        if self.sum is None:
            # detach first to disconnect from graph, then clone for safe accumulation
            self.sum = value.detach().clone()
            self.compensation = torch.zeros_like(self.sum)
        else:
            # Kahan summation algorithm
            y = value - self.compensation
            t = self.sum + y
            self.compensation = (t - self.sum) - y
            self.sum = t
        self.count += 1

    def get_sum(self) -> Optional[torch.Tensor]:
        """Get the accumulated sum."""
        return self.sum

    def get_mean(self) -> Optional[torch.Tensor]:
        """Get the mean of accumulated values."""
        if self.sum is None or self.count == 0:
            return None
        return self.sum / self.count

    def get_count(self) -> int:
        """Get the number of accumulated values."""
        return self.count

    def reset(self) -> None:
        """Reset the accumulator to empty state."""
        self.sum = None
        self.compensation = None
        self.count = 0


__all__ = ['KahanAccumulator']
