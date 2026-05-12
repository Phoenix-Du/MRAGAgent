from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.integrations.bridge_settings import bridge_settings

logger = logging.getLogger(__name__)


def llm_env() -> tuple[str, str, str]:
    api_key = (bridge_settings.qwen_api_key or "").strip() or (
        bridge_settings.openai_api_key or ""
    ).strip()
    base = (
        (bridge_settings.qwen_base_url or "").strip()
        or (bridge_settings.openai_base_url or "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = (bridge_settings.qwen_parser_model or "").strip() or "qwen3.5-plus"
    return api_key, base, model


def extract_json_obj(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    obj_text = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def post_llm_json(
    *,
    base: str,
    api_key: str,
    payload: dict[str, Any],
    retries: int = 1,
) -> dict[str, Any] | None:
    timeout = httpx.Timeout(
        connect=float(bridge_settings.request_timeout_seconds),
        read=float(
            min(
                bridge_settings.parser_read_timeout_seconds,
                bridge_settings.request_timeout_seconds,
            )
        ),
        write=float(bridge_settings.request_timeout_seconds),
        pool=float(bridge_settings.request_timeout_seconds),
    )
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(
                    f"{base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message", {}).get("content", ""))
            obj = extract_json_obj(str(content))
            if obj:
                return obj
            logger.warning("llm_json_bad_json attempt=%s content_prefix=%s", attempt + 1, str(content)[:240])
            return None
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
    if last_exc:
        raise last_exc
    return None
