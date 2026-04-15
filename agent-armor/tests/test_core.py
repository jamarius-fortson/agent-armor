"""Comprehensive tests for agentarmor."""

import asyncio
import time

import pytest

from agentarmor.breaker import (
    CircuitBreaker, CircuitOpenError, CircuitState, CircuitStats,
)
from agentarmor.middleware import (
    FallbackChain, FallbackExhaustedError, RetryPolicy,
)
from agentarmor.monitor import HealthMonitor, shield


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

class FlakyService:
    """Service that fails N times then succeeds."""

    def __init__(self, fail_count: int = 0):
        self.fail_count = fail_count
        self.calls = 0

    async def call(self, input: str = "") -> str:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise ConnectionError(f"Failure #{self.calls}")
        return f"Success after {self.calls} attempts"


class AlwaysFailService:
    async def call(self, input: str = "") -> str:
        raise ConnectionError("Always fails")


class AlwaysSucceedService:
    async def call(self, input: str = "") -> str:
        return "Always succeeds"


# ─────────────────────────────────────────────
# Circuit Breaker State Machine
# ─────────────────────────────────────────────

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_closed_state_on_success(self):
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        svc = AlwaysSucceedService()
        result = await breaker.call(svc.call, "hello")
        assert result == "Always succeeds"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=3,
            fallback_response="fallback",
        )
        svc = AlwaysFailService()

        for _ in range(3):
            await breaker.call(svc.call, "test")

        assert breaker.state == "open"
        assert breaker.failure_count >= 3

    @pytest.mark.asyncio
    async def test_returns_fallback_when_open(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=1000,  # Long timeout
            fallback_response="I'm the fallback",
        )
        svc = AlwaysFailService()

        # Trip the breaker
        for _ in range(2):
            await breaker.call(svc.call)

        # Now it should return fallback immediately
        result = await breaker.call(svc.call)
        assert result == "I'm the fallback"

    @pytest.mark.asyncio
    async def test_raises_when_open_no_fallback(self):
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=1000)
        svc = AlwaysFailService()

        with pytest.raises(ConnectionError):
            await breaker.call(svc.call)
        with pytest.raises(ConnectionError):
            await breaker.call(svc.call)

        # Circuit is now open, no fallback configured
        with pytest.raises(CircuitOpenError):
            await breaker.call(svc.call)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=0.1,  # 100ms
            fallback_response="fb",
        )
        svc = AlwaysFailService()

        # Trip it
        for _ in range(2):
            await breaker.call(svc.call)
        assert breaker._state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should be half-open now
        assert breaker.state == "half_open"

    @pytest.mark.asyncio
    async def test_closes_after_half_open_success(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
            fallback_response="fb",
        )

        # Trip with failing service
        fail_svc = AlwaysFailService()
        for _ in range(2):
            await breaker.call(fail_svc.call)

        await asyncio.sleep(0.15)
        assert breaker.state == "half_open"

        # Succeed in half-open → should close
        ok_svc = AlwaysSucceedService()
        result = await breaker.call(ok_svc.call)
        assert result == "Always succeeds"
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_reopens_after_half_open_failure(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
            fallback_response="fb",
        )

        fail_svc = AlwaysFailService()
        for _ in range(2):
            await breaker.call(fail_svc.call)

        await asyncio.sleep(0.15)
        assert breaker.state == "half_open"

        # Fail in half-open → should re-open
        await breaker.call(fail_svc.call)
        assert breaker._state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self):
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            fallback_response="fb",
        )
        fail_svc = AlwaysFailService()
        for _ in range(2):
            await breaker.call(fail_svc.call)
        assert breaker._state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        breaker = CircuitBreaker(name="test", failure_threshold=10)

        ok_svc = AlwaysSucceedService()
        for _ in range(5):
            await breaker.call(ok_svc.call)

        stats = breaker.stats
        assert stats["total_calls"] == 5
        assert stats["total_successes"] == 5
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_callbacks_fired(self):
        events = []
        breaker = CircuitBreaker(
            name="test", failure_threshold=2,
            fallback_response="fb",
            on_state_change=lambda old, new: events.append(("state", old, new)),
            on_success=lambda lat: events.append(("success", lat)),
            on_failure=lambda e: events.append(("failure", str(e))),
        )

        ok_svc = AlwaysSucceedService()
        await breaker.call(ok_svc.call)
        assert any(e[0] == "success" for e in events)

        fail_svc = AlwaysFailService()
        for _ in range(2):
            await breaker.call(fail_svc.call)
        assert any(e[0] == "failure" for e in events)
        assert any(e[0] == "state" and e[2] == "open" for e in events)


# ─────────────────────────────────────────────
# Retry Policy
# ─────────────────────────────────────────────

class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self):
        retry = RetryPolicy(max_retries=3)
        svc = AlwaysSucceedService()
        result = await retry.execute(svc.call, "test")
        assert result == "Always succeeds"

    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self):
        retry = RetryPolicy(max_retries=5, base_delay=0.01)
        svc = FlakyService(fail_count=3)
        result = await retry.execute(svc.call, "test")
        assert "Success" in result
        assert svc.calls == 4  # 3 failures + 1 success

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        retry = RetryPolicy(max_retries=2, base_delay=0.01)
        svc = FlakyService(fail_count=10)
        with pytest.raises(ConnectionError):
            await retry.execute(svc.call, "test")
        assert svc.calls == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        retry = RetryPolicy(max_retries=3, backoff="exponential", base_delay=0.01)
        delays = [retry._compute_delay(i) for i in range(3)]
        # Exponential: each delay should be roughly double
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]

    @pytest.mark.asyncio
    async def test_linear_backoff(self):
        retry = RetryPolicy(backoff="linear", base_delay=1.0, jitter=False)
        delays = [retry._compute_delay(i) for i in range(3)]
        assert delays == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_constant_backoff(self):
        retry = RetryPolicy(backoff="constant", base_delay=1.0, jitter=False)
        delays = [retry._compute_delay(i) for i in range(3)]
        assert all(d == 1.0 for d in delays)

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        retry = RetryPolicy(
            backoff="exponential", base_delay=10.0,
            max_delay=15.0, jitter=False,
        )
        delay = retry._compute_delay(5)  # Would be 320s without cap
        assert delay == 15.0

    @pytest.mark.asyncio
    async def test_jitter_adds_variance(self):
        retry = RetryPolicy(backoff="constant", base_delay=1.0, jitter=True)
        delays = [retry._compute_delay(0) for _ in range(20)]
        # Should have some variance
        assert len(set(round(d, 4) for d in delays)) > 1


# ─────────────────────────────────────────────
# Fallback Chain
# ─────────────────────────────────────────────

class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_first_model_succeeds(self):
        call_log = []

        async def mock_call(config, *args, **kwargs):
            call_log.append(config["model"])
            return f"result from {config['model']}"

        chain = FallbackChain(
            [{"model": "primary"}, {"model": "backup"}],
            call_fn=mock_call,
        )
        result = await chain.execute("test")
        assert result == "result from primary"
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_falls_to_second(self):
        call_count = 0

        async def mock_call(config, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if config["model"] == "primary":
                raise ConnectionError("Primary down")
            return f"result from {config['model']}"

        chain = FallbackChain(
            [{"model": "primary"}, {"model": "backup"}],
            call_fn=mock_call,
        )
        result = await chain.execute("test")
        assert result == "result from backup"
        assert chain.last_used_model == "backup"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        async def mock_call(config, *args, **kwargs):
            raise ConnectionError(f"{config['model']} failed")

        chain = FallbackChain(
            [{"model": "a"}, {"model": "b"}],
            call_fn=mock_call,
        )
        with pytest.raises(FallbackExhaustedError):
            await chain.execute("test")


# ─────────────────────────────────────────────
# Shield Decorator
# ─────────────────────────────────────────────

class TestShield:
    @pytest.mark.asyncio
    async def test_basic_shield(self):
        @shield(retries=2, circuit_threshold=5)
        async def my_agent(input: str) -> str:
            return f"Result: {input}"

        result = await my_agent("hello")
        assert result == "Result: hello"

    @pytest.mark.asyncio
    async def test_shield_with_fallback_response(self):
        call_count = 0

        @shield(
            retries=1, base_delay=0.01,
            circuit_threshold=2,
            fallback_response="I'm the fallback",
        )
        async def failing_agent(input: str) -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        # First two calls trip the breaker (retry exhaustion counts)
        result1 = await failing_agent("test")
        result2 = await failing_agent("test")

        # Should get fallback once circuit opens
        result3 = await failing_agent("test")
        assert result3 == "I'm the fallback"

    @pytest.mark.asyncio
    async def test_shield_exposes_stats(self):
        @shield(retries=0, circuit_threshold=10)
        async def my_agent(input: str) -> str:
            return "ok"

        await my_agent("test")
        stats = my_agent.stats()
        assert stats["total_calls"] == 1
        assert stats["state"] == "closed"

    @pytest.mark.asyncio
    async def test_shield_timeout(self):
        @shield(timeout=0.1, retries=0, fallback_response="timed out")
        async def slow_agent(input: str) -> str:
            await asyncio.sleep(5)
            return "never"

        result = await slow_agent("test")
        # Should timeout and return fallback
        assert result == "timed out"


# ─────────────────────────────────────────────
# Health Monitor
# ─────────────────────────────────────────────

class TestHealthMonitor:
    def test_register_and_status(self):
        monitor = HealthMonitor()
        b1 = CircuitBreaker(name="agent-1")
        b2 = CircuitBreaker(name="agent-2")
        monitor.register(b1)
        monitor.register(b2)

        status = monitor.status()
        assert "agent-1" in status
        assert "agent-2" in status

    def test_all_healthy(self):
        monitor = HealthMonitor()
        monitor.register(CircuitBreaker(name="a"))
        monitor.register(CircuitBreaker(name="b"))
        assert monitor.all_healthy() is True

    @pytest.mark.asyncio
    async def test_unhealthy_detection(self):
        monitor = HealthMonitor()
        b1 = CircuitBreaker(name="healthy")
        b2 = CircuitBreaker(name="broken", failure_threshold=2, fallback_response="fb")
        monitor.register(b1)
        monitor.register(b2)

        # Trip b2
        svc = AlwaysFailService()
        for _ in range(2):
            await b2.call(svc.call)

        assert monitor.all_healthy() is False
        assert "broken" in monitor.unhealthy()
        assert "healthy" not in monitor.unhealthy()


# ─────────────────────────────────────────────
# CircuitStats
# ─────────────────────────────────────────────

class TestCircuitStats:
    def test_success_rate_empty(self):
        stats = CircuitStats()
        assert stats.success_rate == 1.0

    def test_success_rate_with_data(self):
        stats = CircuitStats(total_calls=10, total_successes=8)
        assert stats.success_rate == 0.8

    def test_avg_latency(self):
        stats = CircuitStats(total_successes=5, total_latency=10.0)
        assert stats.avg_latency == 2.0

    def test_to_dict(self):
        stats = CircuitStats(total_calls=100, total_successes=95, total_failures=5)
        d = stats.to_dict()
        assert d["total_calls"] == 100
        assert d["success_rate"] == 0.95
