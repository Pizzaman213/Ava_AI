"""
Evaluation utilities for model assessment.

This package provides:
- Coherence metrics
- Perplexity calculation
- Generation quality evaluation
"""

from .coherence import (
    CoherenceConfig,
    CoherenceMetrics,
    CoherenceMeasurer,
    measure_coherence,
    measure_batch_coherence,
)

__all__ = [
    'CoherenceConfig',
    'CoherenceMetrics',
    'CoherenceMeasurer',
    'measure_coherence',
    'measure_batch_coherence',
]
