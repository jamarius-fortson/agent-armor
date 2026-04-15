"""Prometheus metrics exporter for AgentArmor."""

import logging
from typing import Dict

logger = logging.getLogger("agentarmor")

try:
    from prometheus_client import Counter, Gauge, Histogram

    # Prometheus metrics
    _AGENT_CALLS = Counter(
        "agentarmor_calls_total",
        "Total calls to the agent bounded by the circuit breaker",
        ["agent_name"]
    )
    _AGENT_SUCCESSES = Counter(
        "agentarmor_successes_total",
        "Total successful calls",
        ["agent_name"]
    )
    _AGENT_FAILURES = Counter(
        "agentarmor_failures_total",
        "Total failed calls",
        ["agent_name"]
    )
    _AGENT_FALLBACKS = Counter(
        "agentarmor_fallbacks_total",
        "Total fallback events triggered",
        ["agent_name", "fallback_model"]
    )
    _CIRCUIT_STATE = Gauge(
        "agentarmor_circuit_state",
        "Current state of the circuit (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
        ["agent_name"]
    )
    _CALL_LATENCY = Histogram(
        "agentarmor_call_latency_seconds",
        "Latency of agent calls in seconds",
        ["agent_name"]
    )
    
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.debug("prometheus-client not installed. Prometheus metrics disabled.")

def record_call(agent_name: str) -> None:
    if HAS_PROMETHEUS:
        _AGENT_CALLS.labels(agent_name=agent_name).inc()

def record_success(agent_name: str, latency: float) -> None:
    if HAS_PROMETHEUS:
        _AGENT_SUCCESSES.labels(agent_name=agent_name).inc()
        _CALL_LATENCY.labels(agent_name=agent_name).observe(latency)

def record_failure(agent_name: str) -> None:
    if HAS_PROMETHEUS:
        _AGENT_FAILURES.labels(agent_name=agent_name).inc()

def record_fallback(agent_name: str, fallback_model: str) -> None:
    if HAS_PROMETHEUS:
        _AGENT_FALLBACKS.labels(agent_name=agent_name, fallback_model=fallback_model).inc()

def record_state_change(agent_name: str, state_value: str) -> None:
    if not HAS_PROMETHEUS:
        return
    
    state_map = {"closed": 0, "open": 1, "half_open": 2}
    _CIRCUIT_STATE.labels(agent_name=agent_name).set(state_map.get(state_value, 0))
