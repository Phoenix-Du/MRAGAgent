from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential


T = TypeVar("T")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
async def with_retry(func: Callable[[], Awaitable[T]]) -> T:
    return await func()

