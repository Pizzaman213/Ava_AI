"""
Circuit Breaker Pattern for Resilient External Service Integration

This module implements the circuit breaker pattern to provide graceful
degradation when external services (like WandB) fail or become unavailable.

The circuit breaker prevents cascading failures by:
1. Tracking failure rates for external services
2. Opening the circuit after threshold failures (skip service temporarily)
3. Periodically testing if service has recovered (half-open state)
4. Closing circuit when service is healthy again

States:
- CLOSED: Normal operation, requests go through
- OPEN: Circuit is open, requests are skipped (fail-fast)
- HALF_OPEN: Testing recovery, limited requests go through
"""

import time
import logging
from enum import Enum
from typing import Callable, Optional, Any, Dict, TypeVar, Generic
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, skip calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for external service calls.

    Provides graceful degradation when external services fail:
    - Tracks failure rates
    - Opens circuit after threshold failures
    - Periodically tests service recovery
    - Closes circuit when service is healthy

    Example:
        breaker = CircuitBreaker(name="wandb", failure_threshold=5)

        @breaker.protected
        def log_to_wandb(metrics):
            wandb.log(metrics)

        # Or use call() method
        result = breaker.call(wandb.log, metrics)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        half_open_timeout: float = 10.0,
        exclude_exceptions: Optional[tuple] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the service being protected
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout_seconds: Time to wait before trying again (open → half-open)
            half_open_timeout: Max time in half-open state before reopening
            exclude_exceptions: Exception types that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_timeout = half_open_timeout
        self.exclude_exceptions = exclude_exceptions or ()

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.half_open_at: Optional[float] = None

        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_opens = 0
        self.circuit_closes = 0

        logger.info(f"CircuitBreaker '{name}' initialized (threshold: {failure_threshold})")

    def call(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result, or None if circuit is open

        Raises:
            Exception: If function raises and circuit allows it through
        """
        self.total_calls += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.debug(
                    f"CircuitBreaker '{self.name}' is OPEN, skipping call "
                    f"(opens: {self.circuit_opens})"
                )
                return None

        # In HALF_OPEN state, check timeout
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_at and time.time() - self.half_open_at > self.half_open_timeout:
                logger.warning(
                    f"CircuitBreaker '{self.name}' timed out in HALF_OPEN, "
                    f"reopening circuit"
                )
                self._transition_to_open()
                return None

        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.exclude_exceptions as e:
            # Don't count these as failures
            logger.debug(f"CircuitBreaker '{self.name}' excluded exception: {e}")
            raise
        except Exception as e:
            self._on_failure(e)
            raise

    def protected(self, func: Callable[..., T]) -> Callable[..., Optional[T]]:
        """
        Decorator to protect a function with circuit breaker.

        Example:
            @breaker.protected
            def risky_call():
                # ... external service call
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call_with_fallback(
        self,
        func: Callable[..., T],
        fallback: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Call function with fallback if circuit is open.

        Args:
            func: Primary function to try
            fallback: Fallback function if circuit is open
            *args: Arguments for functions
            **kwargs: Keyword arguments for functions

        Returns:
            Result from func or fallback
        """
        if self.state == CircuitState.OPEN and not self._should_attempt_reset():
            logger.debug(f"CircuitBreaker '{self.name}' using fallback")
            return fallback(*args, **kwargs)

        try:
            return self.call(func, *args, **kwargs) or fallback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"CircuitBreaker '{self.name}' error, using fallback: {e}")
            return fallback(*args, **kwargs)

    def _on_success(self):
        """Handle successful call."""
        self.total_successes += 1
        self.last_failure_time = None

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"CircuitBreaker '{self.name}' success in HALF_OPEN "
                f"({self.success_count}/{self.success_threshold})"
            )

            if self.success_count >= self.success_threshold:
                self._transition_to_closed()

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            f"CircuitBreaker '{self.name}' failure "
            f"({self.failure_count}/{self.failure_threshold}): {exception}"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open immediately reopens circuit
            logger.warning(f"CircuitBreaker '{self.name}' failed in HALF_OPEN, reopening")
            self._transition_to_open()

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        self.circuit_opens += 1
        logger.error(
            f"CircuitBreaker '{self.name}' OPENED after {self.failure_count} failures "
            f"(will retry in {self.timeout_seconds}s)"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_at = time.time()
        self.success_count = 0
        self.failure_count = 0
        logger.info(f"CircuitBreaker '{self.name}' entering HALF_OPEN (testing recovery)")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        self.half_open_at = None
        self.circuit_closes += 1
        logger.info(f"CircuitBreaker '{self.name}' CLOSED (service recovered)")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit."""
        if self.opened_at is None:
            return False
        return time.time() - self.opened_at >= self.timeout_seconds

    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        return self.state == CircuitState.OPEN

    def is_closed(self) -> bool:
        """Check if circuit is currently closed."""
        return self.state == CircuitState.CLOSED

    def is_half_open(self) -> bool:
        """Check if circuit is currently half-open."""
        return self.state == CircuitState.HALF_OPEN

    def reset(self):
        """Manually reset circuit to closed state."""
        logger.info(f"CircuitBreaker '{self.name}' manually reset")
        self._transition_to_closed()

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        uptime_seconds = 0.0
        if self.opened_at:
            uptime_seconds = time.time() - self.opened_at

        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': self.total_successes / max(1, self.total_calls),
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'circuit_opens': self.circuit_opens,
            'circuit_closes': self.circuit_closes,
            'opened_at': self.opened_at,
            'uptime_seconds': uptime_seconds,
            'last_failure_time': self.last_failure_time,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
            f"failures={self.failure_count}/{self.failure_threshold})"
        )


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized management of circuit breakers for different services.

    Example:
        manager = CircuitBreakerManager()
        manager.register("wandb", failure_threshold=5)
        manager.register("tensorboard", failure_threshold=3)

        # Get breaker and use it
        wandb_breaker = manager.get("wandb")
        wandb_breaker.call(wandb.log, metrics)

        # Or use manager directly
        manager.call("wandb", wandb.log, metrics)
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        logger.info("CircuitBreakerManager initialized")

    def register(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        half_open_timeout: float = 10.0,
        exclude_exceptions: Optional[tuple] = None,
    ) -> CircuitBreaker:
        """
        Register a new circuit breaker.

        Args:
            name: Service name
            failure_threshold: Failures before opening
            success_threshold: Successes in half-open before closing
            timeout_seconds: Time before retry
            half_open_timeout: Max time in half-open
            exclude_exceptions: Exceptions to exclude

        Returns:
            Created circuit breaker
        """
        if name in self.breakers:
            logger.warning(f"CircuitBreaker '{name}' already registered, returning existing")
            return self.breakers[name]

        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            half_open_timeout=half_open_timeout,
            exclude_exceptions=exclude_exceptions,
        )
        self.breakers[name] = breaker
        logger.info(f"Registered CircuitBreaker '{name}'")
        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)

    def call(self, name: str, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """
        Call function through named circuit breaker.

        Args:
            name: Circuit breaker name
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or None if circuit is open
        """
        breaker = self.get(name)
        if not breaker:
            logger.warning(f"CircuitBreaker '{name}' not found, calling directly")
            return func(*args, **kwargs)

        return breaker.call(func, *args, **kwargs)

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_statistics()
            for name, breaker in self.breakers.items()
        }

    def print_status(self):
        """Print status of all circuit breakers."""
        print("\n=== Circuit Breaker Status ===")
        for name, breaker in self.breakers.items():
            stats = breaker.get_statistics()
            print(f"\n{name}:")
            print(f"  State: {stats['state'].upper()}")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"  Total Calls: {stats['total_calls']}")
            print(f"  Circuit Opens: {stats['circuit_opens']}")
            if stats['state'] == 'open':
                print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
