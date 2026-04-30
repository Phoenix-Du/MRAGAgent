from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class ParserCache(Generic[T]):
    ttl_seconds: int
    max_entries: int
    _items: dict[str, tuple[float, T]] = field(default_factory=dict)

    def get(self, key: str, now: float) -> T | None:
        cached = self._items.get(key)
        if not cached:
            return None
        ts, value = cached
        if now - ts > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        return value

    def put(self, key: str, value: T, now: float) -> None:
        self._items[key] = (now, value)
        max_entries = max(1, int(self.max_entries))
        overflow = len(self._items) - max_entries
        if overflow <= 0:
            return
        for stale_key, _ in sorted(self._items.items(), key=lambda item: item[1][0])[:overflow]:
            self._items.pop(stale_key, None)

    def clear(self) -> None:
        self._items.clear()

    def keys(self) -> set[str]:
        return set(self._items)
