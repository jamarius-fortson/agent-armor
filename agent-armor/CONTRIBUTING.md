# Contributing to agent-circuit-breaker

## Setup
```bash
git clone https://github.com/daniellopez882/agent-circuit-breaker.git
cd agent-circuit-breaker
pip install -e ".[dev,all]"
pytest tests/ -v
```

## High-Impact Contributions
- **Redis-backed shared state** — circuit state shared across processes
- **Prometheus metrics** — export circuit state as Prometheus gauges
- **Bulkhead pattern** — concurrency limiting per agent
- **Cost-aware fallback** — choose cheapest working model
- **Checkpoint/resume** — save pipeline state on crash
- **Dashboard** — web UI for monitoring circuit states
