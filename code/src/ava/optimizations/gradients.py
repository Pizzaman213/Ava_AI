"""
Gradient monitoring utilities for the Ava pipeline.

NOTE: This module is a backward-compatibility shim. The actual implementations
have been merged into gpu_sync.py for consolidated GPU synchronization utilities.

All exports are re-exported from gpu_sync.py:
    - check_gradients
    - check_gradients_deferred
    - detect_gradient_issues
    - ZERO_GRAD_CHECK_SAMPLE_RATE
"""

# Re-export from gpu_sync for backward compatibility
from .gpu_sync import (
    check_gradients,
    check_gradients_deferred,
    detect_gradient_issues,
    ZERO_GRAD_CHECK_SAMPLE_RATE,
)

__all__ = [
    'check_gradients',
    'check_gradients_deferred',
    'detect_gradient_issues',
    'ZERO_GRAD_CHECK_SAMPLE_RATE',
]
