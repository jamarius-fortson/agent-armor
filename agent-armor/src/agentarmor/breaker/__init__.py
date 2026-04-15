"""Circuit breaker state machine for AI agents."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from . import metrics

logger = logging.getLogger("agentarmor")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """Statistics tracked by the circuit breaker."""

    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change_time: float = 0.0
    total_latency: float = 0.0
    state_changes: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.total_successes / self.total_calls

    @property
    def avg_latency(self) -> float:
        if self.total_successes == 0:
            return 0.0
        return self.total_latency / self.total_successes

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_s": round(self.avg_latency, 2),
            "state_changes": self.state_changes,
        }


class CircuitBreaker:
    """Production-grade circuit breaker for AI agents.

    States:
        CLOSED  — Normal operation. Failures counted.
        OPEN    — All calls fail-fast. Fallback returned.
        HALF_OPEN — Limited test calls. Success → CLOSED, failure → OPEN.

    Usage::

        breaker = CircuitBreaker(name="my-agent", failure_threshold=5)
        result = await breaker.call(my_agent_func, "input")
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 2,
        retry_policy: Optional[Any] = None,  # RetryPolicy
        fallback_chain: Optional[Any] = None,  # FallbackChain
        fallback_response: Optional[str] = None,
        on_state_change: Optional[Callable] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
        on_fallback: Optional[Callable] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.retry_policy = retry_policy
        self.fallback_chain = fallback_chain
        self.fallback_response = fallback_response

        # Callbacks
        self._on_state_change = on_state_change
        self._on_success = on_success
        self._on_failure = on_failure
        self._on_fallback = on_fallback

        # State
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        self._stats = CircuitStats()
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        # Check if open circuit should transition to half-open
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
        return self._state.value

    @property
    def failure_count(self) -> int:
        return self._stats.consecutive_failures

    @property
    def success_count(self) -> int:
        return self._stats.consecutive_successes

    @property
    def total_calls(self) -> int:
        return self._stats.total_calls

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            **self._stats.to_dict(),
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        # Force state check (may transition open → half_open)
        current_state = self.state

        if self._state == CircuitState.OPEN:
            logger.info(f"[{self.name}] Circuit OPEN — returning fallback")
            return self._get_fallback(*args, **kwargs)

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                logger.info(f"[{self.name}] Half-open limit reached — returning fallback")
                return self._get_fallback(*args, **kwargs)
            self._half_open_calls += 1

        # Try to execute
        self._stats.total_calls += 1
        metrics.record_call(self.name)
        start = time.monotonic()

        try:
            # Apply retry policy if configured
            if self.retry_policy:
                result = await self.retry_policy.execute(func, *args, **kwargs)
            else:
                result = await func(*args, **kwargs)

            latency = time.monotonic() - start
            self._record_success(latency)
            return result

        except Exception as e:
            latency = time.monotonic() - start
            self._record_failure(e)

            # Try fallback chain
            if self.fallback_chain:
                try:
                    result = await self.fallback_chain.execute(*args, **kwargs)
                    if self._on_fallback:
                        self._on_fallback()
                    return result
                except Exception:
                    pass

            # Return static fallback
            if self.fallback_response is not None:
                if self._on_fallback:
                    self._on_fallback()
                return self.fallback_response

            raise

    def _record_success(self, latency: float) -> None:
        self._stats.total_successes += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = time.monotonic()
        self._stats.total_latency += latency
        
        metrics.record_success(self.name, latency)

        if self._on_success:
            self._on_success(latency)

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.half_open_max_calls:
                self._transition(CircuitState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        self._stats.total_failures += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = time.monotonic()

        metrics.record_failure(self.name)

        if self._on_failure:
            self._on_failure(error)

        if self._state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
        elif self._stats.consecutive_failures >= self.failure_threshold:
            self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        if old == new_state:
            return

        self._state = new_state
        self._stats.state_changes += 1
        self._stats.last_state_change_time = time.monotonic()

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            logger.warning(f"[{self.name}] Circuit OPENED after {self.failure_threshold} failures")
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            logger.info(f"[{self.name}] Circuit → HALF_OPEN (testing recovery)")
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            logger.info(f"[{self.name}] Circuit CLOSED (recovered)")

        metrics.record_state_change(self.name, new_state.value)

        if self._on_state_change:
            self._on_state_change(old.value, new_state.value)

    def _get_fallback(self, *args, **kwargs) -> Any:
        metrics.record_fallback(self.name, "static_response")
        if self._on_fallback:
            self._on_fallback()
        if self.fallback_response is not None:
            return self.fallback_response
        raise CircuitOpenError(
            f"Circuit '{self.name}' is OPEN. "
            f"{self._stats.consecutive_failures} consecutive failures."
        )

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        self._transition(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0


class CircuitOpenError(Exception):
    """Raised when calling through an open circuit with no fallback."""
