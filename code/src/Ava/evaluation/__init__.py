"""
Evaluation module for Ava LLM Training Framework.

Provides metrics and evaluation tools for assessing model quality
during and after training.
"""

from .coherence import (
    CoherenceMetrics,
    CoherenceMeasurer,
    measure_coherence,
    measure_batch_coherence,
)

__all__ = [
    'CoherenceMetrics',
    'CoherenceMeasurer',
    'measure_coherence',
    'measure_batch_coherence',
]
