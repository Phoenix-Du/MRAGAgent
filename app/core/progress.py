from __future__ import annotations

import time
from threading import Lock
from typing import Any


_LOCK = Lock()
_MAX_TASKS = 300
_TASKS: dict[str, dict[str, Any]] = {}


def progress_start(request_id: str, *, query: str, intent: str | None) -> None:
    now = int(time.time() * 1000)
    with _LOCK:
        if len(_TASKS) >= _MAX_TASKS:
            oldest = sorted(_TASKS.items(), key=lambda kv: kv[1].get("created_at_ms", 0))[:50]
            for key, _ in oldest:
                _TASKS.pop(key, None)
        _TASKS[request_id] = {
            "request_id": request_id,
            "status": "running",
            "created_at_ms": now,
            "updated_at_ms": now,
            "seq": 0,
            "query": query,
            "intent": intent,
            "events": [
                {
                    "ts_ms": now,
                    "elapsed_ms": 0,
                    "seq": 0,
                    "stage": "start",
                    "message": "收到请求，开始处理。",
                    "data": {"intent": intent, "query": query[:200]},
                }
            ],
        }
def progress_event(request_id: str, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
    now = int(time.time() * 1000)
    with _LOCK:
        item = _TASKS.get(request_id)
        if not item:
            return
        item["updated_at_ms"] = now
        item["seq"] = int(item.get("seq", 0)) + 1
        elapsed = max(0, now - int(item.get("created_at_ms", now)))
        item["events"].append(
            {
                "ts_ms": now,
                "elapsed_ms": elapsed,
                "seq": item["seq"],
                "stage": stage,
                "message": message,
                "data": data or {},
            }
        )
        # Keep the latest events only.
        if len(item["events"]) > 120:
            item["events"] = item["events"][-120:]


def progress_complete(request_id: str, data: dict[str, Any] | None = None) -> None:
    now = int(time.time() * 1000)
    with _LOCK:
        item = _TASKS.get(request_id)
        if not item:
            return
        item["updated_at_ms"] = now
        item["status"] = "completed"
        item["seq"] = int(item.get("seq", 0)) + 1
        elapsed = max(0, now - int(item.get("created_at_ms", now)))
        item["events"].append(
            {
                "ts_ms": now,
                "elapsed_ms": elapsed,
                "seq": item["seq"],
                "stage": "done",
                "message": "处理完成。",
                "data": data or {},
            }
        )


def progress_error(request_id: str, error: str) -> None:
    now = int(time.time() * 1000)
    with _LOCK:
        item = _TASKS.get(request_id)
        if not item:
            return
        item["updated_at_ms"] = now
        item["status"] = "error"
        item["seq"] = int(item.get("seq", 0)) + 1
        elapsed = max(0, now - int(item.get("created_at_ms", now)))
        item["events"].append(
            {
                "ts_ms": now,
                "elapsed_ms": elapsed,
                "seq": item["seq"],
                "stage": "error",
                "message": "处理失败。",
                "data": {"error": error[:300]},
            }
        )


def progress_get(request_id: str) -> dict[str, Any] | None:
    with _LOCK:
        item = _TASKS.get(request_id)
        if not item:
            return None
        return {
            "request_id": item["request_id"],
            "status": item["status"],
            "created_at_ms": item["created_at_ms"],
            "updated_at_ms": item["updated_at_ms"],
            "events": list(item["events"]),
        }
