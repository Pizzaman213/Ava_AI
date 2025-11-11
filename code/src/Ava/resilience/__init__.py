"""
Resilience Components for Ava

This module provides resilience patterns for handling external service failures
and ensuring training continues even when optional services are unavailable.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
)

__all__ = [
    'CircuitBreaker',
    'CircuitBreakerManager',
    'CircuitState',
]
