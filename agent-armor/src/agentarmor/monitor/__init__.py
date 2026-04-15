"""Health monitor and @shield decorator."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

from .breaker import CircuitBreaker
from .middleware import FallbackChain, RetryPolicy
from .bulkhead import Bulkhead

logger = logging.getLogger("agentarmor")


def shield(
    retries: int = 3,
    backoff: str = "exponential",
    jitter: bool = True,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    fallback_model: Optional[str] = None,
    fallback_response: Optional[str] = None,
    circuit_threshold: int = 5,
    circuit_timeout: float = 60.0,
    timeout: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    max_wait_queue: int = 0,
    name: Optional[str] = None,
    on_fallback: Optional[Callable] = None,
    on_circuit_open: Optional[Callable] = None,
) -> Callable:
    """Decorator that wraps an async function with full fault tolerance.

    Combines retry + fallback + circuit breaker + timeout in one decorator.

    Usage::

        @shield(retries=3, fallback_model="gpt-4o-mini", timeout=30)
        async def my_agent(input: str) -> str:
            return await call_llm(input)
    """

    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__name__

        retry_policy = RetryPolicy(
            max_retries=retries,
            backoff=backoff,
            jitter=jitter,
            base_delay=base_delay,
            max_delay=max_delay,
        )

        fallback_chain = None
        if fallback_model:
            fallback_chain = FallbackChain([
                {"model": fallback_model, "provider": "openai"},
            ])
            
        bulkhead = None
        if max_concurrency is not None:
            bulkhead = Bulkhead(
                max_concurrency=max_concurrency, 
                max_wait_queue=max_wait_queue
            )

        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=circuit_threshold,
            recovery_timeout=circuit_timeout,
            retry_policy=retry_policy,
            fallback_chain=fallback_chain,
            fallback_response=fallback_response,
            on_state_change=lambda old, new: (
                on_circuit_open() if new == "open" and on_circuit_open else None
            ),
            on_fallback=on_fallback,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async def _execute(*a, **kw):
                if timeout:
                    return await asyncio.wait_for(
                        func(*a, **kw), timeout=timeout
                    )
                return await func(*a, **kw)
                
            async def _bulkhead_execute(*a, **kw):
                if bulkhead:
                    return await bulkhead.call(_execute, *a, **kw)
                return await _execute(*a, **kw)

            return await breaker.call(_bulkhead_execute, *args, **kwargs)

        # Expose breaker for inspection
        wrapper.breaker = breaker  # type: ignore
        wrapper.stats = lambda: breaker.stats  # type: ignore
        wrapper.reset = breaker.reset  # type: ignore

        return wrapper

    return decorator


class HealthMonitor:
    """Monitor multiple circuit breakers in one dashboard.

    Usage::

        monitor = HealthMonitor()
        monitor.register(breaker_1)
        monitor.register(breaker_2)
        print(monitor.status())
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker) -> None:
        self._breakers[breaker.name] = breaker

    def status(self) -> dict[str, dict]:
        return {name: b.stats for name, b in self._breakers.items()}

    def all_healthy(self) -> bool:
        return all(
            b.state == "closed" for b in self._breakers.values()
        )

    def unhealthy(self) -> list[str]:
        return [
            name for name, b in self._breakers.items()
            if b.state != "closed"
        ]

    def print_status(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(title="Circuit Breaker Health", box=box.ROUNDED)
            table.add_column("Agent", style="bold")
            table.add_column("State", justify="center")
            table.add_column("Failures", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Calls", justify="right")

            for name, b in self._breakers.items():
                s = b.stats
                state = s["state"]
                state_str = (
                    "[green]CLOSED ✅[/]" if state == "closed"
                    else "[red]OPEN ❌[/]" if state == "open"
                    else "[yellow]HALF_OPEN ⚠️[/]"
                )
                table.add_row(
                    name, state_str,
                    str(s["consecutive_failures"]),
                    f"{s['success_rate']:.1%}",
                    str(s["total_calls"]),
                )
            console.print(table)
        except ImportError:
            for name, b in self._breakers.items():
                s = b.stats
                print(f"  {name}: {s['state']} | {s['success_rate']:.1%} success")
