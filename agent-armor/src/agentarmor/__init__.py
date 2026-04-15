"""agentarmor: Production fault-tolerance for AI agents."""

from .breaker import CircuitBreaker, CircuitOpenError, CircuitState, CircuitStats
from .middleware import FallbackChain, FallbackExhaustedError, RetryPolicy
from .monitor import HealthMonitor, shield
from .bulkhead import Bulkhead, BulkheadRejectedError

__version__ = "0.1.0"
__all__ = [
    "CircuitBreaker", "CircuitOpenError", "CircuitState", "CircuitStats",
    "FallbackChain", "FallbackExhaustedError", "HealthMonitor",
    "RetryPolicy", "shield",
    "Bulkhead", "BulkheadRejectedError",
]
