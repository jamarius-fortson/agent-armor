"""Bulkhead pattern for concurrency isolation."""

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("agentarmor")

class BulkheadRejectedError(Exception):
    """Raised when the bulkhead limit is exceeded."""

class Bulkhead:
    """Limits the number of concurrent executions to prevent overwhelming the downstream agent.
    
    Usage::
        bulkhead = Bulkhead(max_concurrency=10, max_wait_queue=50)
        result = await bulkhead.call(my_agent, "args")
    """

    def __init__(self, max_concurrency: int, max_wait_queue: int = 0):
        self.max_concurrency = max_concurrency
        self.max_wait_queue = max_wait_queue
        
        # We use a semaphore for active concurrency and a counter for the queue
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._queued_count = 0

    @property
    def active_calls(self) -> int:
        # Not completely thread-safe atomic, but good enough in AsyncIO event loop
        return self.max_concurrency - self._semaphore._value

    @property
    def queued_calls(self) -> int:
        return self._queued_count

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self._queued_count >= self.max_wait_queue:
            logger.warning(
                f"Bulkhead rejected: Queue full (max_wait={self.max_wait_queue}, "
                f"active={self.active_calls}/{self.max_concurrency})"
            )
            raise BulkheadRejectedError("Bulkhead queue is full.")

        self._queued_count += 1
        try:
            async with self._semaphore:
                # We have acquired a concurrent slot
                self._queued_count -= 1
                return await func(*args, **kwargs)
        except asyncio.CancelledError:
            # The wait was cancelled (e.g., timeout from shield)
            self._queued_count = max(0, self._queued_count - 1)
            raise
