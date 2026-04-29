from __future__ import annotations

from contextvars import ContextVar

_runtime_flags_ctx: ContextVar[set[str]] = ContextVar("runtime_flags", default=set())


def reset_runtime_flags() -> None:
    _runtime_flags_ctx.set(set())


def add_runtime_flag(flag: str) -> None:
    current = set(_runtime_flags_ctx.get())
    current.add(flag)
    _runtime_flags_ctx.set(current)


def get_runtime_flags() -> list[str]:
    return sorted(_runtime_flags_ctx.get())
