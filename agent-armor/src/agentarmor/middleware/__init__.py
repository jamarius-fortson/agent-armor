"""Retry policy and fallback chain for agent resilience."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import metrics

logger = logging.getLogger("agentarmor")


# HTTP status codes considered retryable
DEFAULT_RETRYABLE = {429, 500, 502, 503, 504}
DEFAULT_NON_RETRYABLE = {400, 401, 403, 404}


class RetryPolicy:
    """Configurable retry with exponential backoff and jitter.

    Usage::

        retry = RetryPolicy(max_retries=3, backoff="exponential", jitter=True)
        result = await retry.execute(my_func, arg1, arg2)
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff: str = "exponential",  # "exponential" | "linear" | "constant"
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
        jitter_range: float = 0.2,
        retry_on: Optional[list[int]] = None,  # HTTP status codes
        retry_on_exceptions: Optional[list[type]] = None,
        dont_retry_on: Optional[list[int]] = None,
    ):
        self.max_retries = max_retries
        self.backoff = backoff
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.retry_on = set(retry_on or DEFAULT_RETRYABLE)
        self.retry_on_exceptions = tuple(
            retry_on_exceptions or [TimeoutError, ConnectionError, OSError]
        )
        self.dont_retry_on = set(dont_retry_on or DEFAULT_NON_RETRYABLE)
        self.retry_count = 0

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"Succeeded on attempt {attempt + 1}/{self.max_retries + 1}"
                    )
                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e):
                    logger.debug(f"Non-retryable error: {type(e).__name__}: {e}")
                    raise

                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.info(
                        f"Attempt {attempt + 1} failed ({type(e).__name__}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    self.retry_count += 1
                else:
                    logger.warning(
                        f"All {self.max_retries + 1} attempts exhausted. "
                        f"Last error: {e}"
                    )

        raise last_exception  # type: ignore

    def _should_retry(self, error: Exception) -> bool:
        """Determine if this error type should be retried."""
        # Check exception type
        if isinstance(error, self.retry_on_exceptions):
            return True

        # Check for HTTP status code in error
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status:
            if status in self.dont_retry_on:
                return False
            if status in self.retry_on:
                return True

        # Default: retry on unknown errors
        return True

    def _compute_delay(self, attempt: int) -> float:
        """Compute the delay before the next retry."""
        if self.backoff == "exponential":
            delay = self.base_delay * (2 ** attempt)
        elif self.backoff == "linear":
            delay = self.base_delay * (attempt + 1)
        else:  # constant
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)

        return delay


class FallbackChain:
    """Ordered list of model/provider fallbacks.

    Tries each in order until one succeeds.

    Usage::

        chain = FallbackChain([
            {"model": "gpt-4o", "provider": "openai"},
            {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
            {"model": "gpt-4o-mini", "provider": "openai"},
        ])
        result = await chain.execute("My prompt")
    """

    def __init__(
        self,
        models: list[dict],
        call_fn: Optional[Callable] = None,
    ):
        self.models = models
        self._call_fn = call_fn or self._default_call
        self.last_used_model: str = ""

    async def execute(self, *args, **kwargs) -> Any:
        """Try each model in the chain until one succeeds."""
        errors = []

        for model_config in self.models:
            model_name = model_config.get("model", "unknown")
            try:
                logger.info(f"Trying fallback: {model_name}")
                result = await self._call_fn(model_config, *args, **kwargs)
                self.last_used_model = model_name
                metrics.record_fallback("chain", model_name)
                return result
            except Exception as e:
                logger.warning(f"Fallback {model_name} failed: {e}")
                errors.append((model_name, e))

        raise FallbackExhaustedError(
            f"All {len(self.models)} fallback models failed: "
            + ", ".join(f"{m}: {e}" for m, e in errors)
        )

    async def _default_call(self, model_config: dict, *args, **kwargs) -> str:
        """Default call using OpenAI-compatible API."""
        model = model_config.get("model", "gpt-4o-mini")
        provider = model_config.get("provider", "openai")

        prompt = args[0] if args else kwargs.get("input", "")

        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=model, max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        else:
            import openai
            client = openai.AsyncOpenAI()
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""


class FallbackExhaustedError(Exception):
    """Raised when all models in the fallback chain have failed."""
