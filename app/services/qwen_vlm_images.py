"""DashScope/OpenAI-compatible VLM helpers: rank CLIP-filtered images and answer from images."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from app.integrations.bridge_settings import bridge_settings

logger = logging.getLogger(__name__)


def _vlm_env() -> tuple[str, str, str]:
    api_key = (bridge_settings.qwen_api_key or "").strip() or (
        bridge_settings.openai_api_key or ""
    ).strip()
    base = (
        (bridge_settings.qwen_base_url or "").strip()
        or (bridge_settings.openai_base_url or "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = (
        (bridge_settings.qwen_vlm_model or "").strip()
        or (bridge_settings.raganything_vision_model or "").strip()
        or "qwen-vl-plus"
    )
    return api_key, base, model


def has_vlm_credentials() -> bool:
    api_key, _, _ = _vlm_env()
    return bool(api_key)


async def _fetch_image_bytes(url: str, timeout: float = 25.0) -> tuple[bytes, str] | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, trust_env=False, headers=headers
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        ct = (resp.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
        if not ct.lower().startswith("image/"):
            return None
        return resp.content, ct
    except (httpx.HTTPError, OSError, ValueError):
        return None


def _guess_ext(content_type: str, url: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    suffix = Path(url).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return suffix
    return ".jpg"


async def _cache_image_locally(url: str, timeout: float = 20.0) -> str | None:
    cache_dir = Path(bridge_settings.image_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    # fast path: common extensions lookup
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        p = cache_dir / f"{digest}{ext}"
        if p.exists():
            return str(p)
    blob = await _fetch_image_bytes(url, timeout=timeout)
    if blob is None:
        return None
    content, ctype = blob
    path = cache_dir / f"{digest}{_guess_ext(ctype, url)}"
    try:
        if not path.exists():
            path.write_bytes(content)
        return str(path)
    except OSError:
        return None


def _read_local_image_bytes(local_path: str | None) -> tuple[bytes, str] | None:
    if not local_path:
        return None
    try:
        path = Path(local_path)
        if not path.is_file():
            return None
        suffix = path.suffix.lower()
        if suffix == ".png":
            ctype = "image/png"
        elif suffix == ".webp":
            ctype = "image/webp"
        elif suffix == ".gif":
            ctype = "image/gif"
        else:
            ctype = "image/jpeg"
        return path.read_bytes(), ctype
    except OSError:
        return None


async def _fetch_many_image_bytes(
    rows: list[tuple[str, str | None, str | None]],
    *,
    timeout: float = 25.0,
    concurrency: int = 6,
) -> list[tuple[bytes, str] | None]:
    sem = asyncio.Semaphore(max(1, concurrency))
    out: list[tuple[bytes, str] | None] = [None] * len(rows)

    async def _worker(i: int, url: str, local_path: str | None) -> None:
        local_blob = _read_local_image_bytes(local_path)
        if local_blob is not None:
            out[i] = local_blob
            return
        async with sem:
            out[i] = await _fetch_image_bytes(url, timeout=timeout)

    await asyncio.gather(*[_worker(i, u, lp) for i, (u, _d, lp) in enumerate(rows)])
    return out


async def filter_reachable_image_rows(
    rows: list[tuple[str, str | None, str | None]],
) -> list[tuple[str, str | None, str | None]]:
    normalized: list[tuple[str, str | None, str | None]] = []
    for url, desc, local_path in rows:
        lp = local_path
        if not _read_local_image_bytes(lp):
            lp = await _cache_image_locally(url, timeout=20.0)
        if lp is not None:
            normalized.append((url, desc, lp))
    return normalized


def _bytes_to_data_url(content: bytes, content_type: str) -> str:
    b64 = base64.standard_b64encode(content).decode("ascii")
    ct = content_type or "image/jpeg"
    return f"data:{ct};base64,{b64}"


async def _chat_vlm(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    api_key, base, model = _vlm_env()
    if not api_key:
        return ""
    endpoint = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
    return (
        (data.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    ).strip()


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _complete_index_order(order: list[Any], n: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for x in order:
        if not isinstance(x, int):
            continue
        if 0 <= x < n and x not in seen:
            out.append(x)
            seen.add(x)
    for i in range(n):
        if i not in seen:
            out.append(i)
    return out


def _normalize_index_subset(values: Any, n: int, limit: int) -> list[int]:
    if not isinstance(values, list):
        return []
    seen: set[int] = set()
    out: list[int] = []
    for x in values:
        if not isinstance(x, int):
            continue
        if 0 <= x < n and x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= limit:
            break
    return out


def _has_spatial_constraint(query: str) -> bool:
    text = query.lower()
    return any(token in text for token in ("左边", "右边", "左侧", "右侧", "left", "right"))


async def vlm_filter_strict_match_indices(
    query: str,
    image_rows: list[tuple[str, str | None, str | None]],
    *,
    max_keep: int,
) -> list[int] | None:
    api_key, _, _ = _vlm_env()
    if not api_key or not image_rows:
        return None

    n = len(image_rows)
    k = max(1, min(max_keep, n))
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"你会收到 {n} 张候选图（编号 0~{n - 1}）和用户问题。"
                "请严格判断每张图是否完全符合用户要求，尤其是左右位置、主体关系、数量关系等硬约束。"
                "不符合就不要选。"
                "只输出一个 JSON 对象，不要其它文字。格式：\n"
                f'{{"matches":[最多{k}个严格符合要求的编号]}}\n'
                f"用户问题：{query}"
            ),
        }
    ]
    blobs = await _fetch_many_image_bytes(image_rows, timeout=25.0, concurrency=6)
    for idx, (url, desc, _local_path) in enumerate(image_rows):
        parts.append(
            {
                "type": "text",
                "text": f"[候选图{idx}] {desc or url}"[:500],
            }
        )
        blob = blobs[idx]
        if blob:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _bytes_to_data_url(blob[0], blob[1])},
                }
            )
    try:
        content = await _chat_vlm(
            [{"role": "user", "content": parts}],
            max_tokens=900,
            temperature=0.0,
            timeout=120.0,
        )
    except Exception:
        logger.exception("vlm_filter_strict_match_indices_failed")
        return None

    data = _extract_json_obj(content)
    if not data:
        return None
    matches = _normalize_index_subset(data.get("matches"), n, k)
    return matches or None


async def vlm_rank_and_answer_from_image_urls(
    query: str,
    image_rows: list[tuple[str, str | None, str | None]],
    *,
    top_k: int,
) -> tuple[list[int] | None, list[int] | None, str]:
    """
    One VLM call that first ranks then answers.
    Returns (ranked_indices_for_rows, selected_indices_for_answer, answer).
    """
    api_key, _, _ = _vlm_env()
    if not api_key or not image_rows:
        return None, None, ""

    n = len(image_rows)
    k = max(1, min(top_k, n))
    spatial_hint = ""
    if _has_spatial_constraint(query):
        spatial_hint = (
            "用户问题包含明确的左右/方位约束，你必须严格依据图片内容判断左右位置。"
            "如果图片中左右关系不符合要求，就不要选入 selected，也不要在 answer 中描述为符合要求。"
        )
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"你会收到 {n} 张候选图（编号 0~{n - 1}）和用户问题。"
                f"请先按问题相关性排序，再选出最多 {k} 张最符合问题的图片，并给出仅基于这些入选图片的连贯回答。"
                "如果有候选图不符合约束或明显不相关，不要放入 selected。"
                "只输出一个 JSON 对象，不要其它文字。格式：\n"
                '{"order":[按相关性降序的编号],'
                f'"selected":[用于回答的前{k}个编号],'
                '"answer":"最终回答文本"}\n'
                f"用户问题：{query}\n"
                f"{spatial_hint}\n"
                "answer 必须只总结 selected 中的图片，不要描述未入选的图片。"
            ),
        }
    ]
    blobs = await _fetch_many_image_bytes(image_rows, timeout=25.0, concurrency=6)
    for idx, (url, desc, _local_path) in enumerate(image_rows):
        parts.append(
            {
                "type": "text",
                "text": f"[候选图{idx}] {desc or url}"[:500],
            }
        )
        blob = blobs[idx]
        if blob:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _bytes_to_data_url(blob[0], blob[1])},
                }
            )
        else:
            parts.append(
                {
                    "type": "text",
                    "text": f"[图{idx}无法加载] {desc or url}"[:700],
                }
            )

    try:
        content = await _chat_vlm(
            [{"role": "user", "content": parts}],
            max_tokens=2200,
            temperature=0.2,
            timeout=150.0,
        )
    except Exception:
        logger.exception("vlm_rank_and_answer_from_image_urls_failed")
        return None, None, ""

    data = _extract_json_obj(content)
    if not data:
        return None, None, content.strip()

    raw_order = data.get("order")
    indices: list[int] | None = None
    if isinstance(raw_order, list):
        indices = _complete_index_order(raw_order, n)
    selected = _normalize_index_subset(data.get("selected"), n, k)

    answer = data.get("answer")
    answer_text = str(answer).strip() if isinstance(answer, str) else ""
    return indices, (selected or None), answer_text


async def vlm_answer_from_image_urls(
    query: str,
    image_rows: list[tuple[str, str | None, str | None]],
    *,
    max_images: int,
) -> str:
    """Multimodal answer: user question + up to max_images (url, desc)."""
    api_key, _, _ = _vlm_env()
    if not api_key:
        return ""

    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "请根据下列图片回答用户问题。要求：结论清晰；若图片不足以回答请说明；"
                "不要编造图中不存在的信息。\n\n"
                f"用户问题：\n{query}"
            ),
        }
    ]

    selected = image_rows[:max_images]
    blobs = await _fetch_many_image_bytes(selected, timeout=25.0, concurrency=6)
    for i, (url, desc, _local_path) in enumerate(selected):
        parts.append(
            {
                "type": "text",
                "text": f"[候选图{i}] {desc or url}"[:500],
            }
        )
        blob = blobs[i]
        if blob:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _bytes_to_data_url(blob[0], blob[1])},
                }
            )
        else:
            parts.append(
                {
                    "type": "text",
                    "text": f"[图片无法加载] {desc or url}"[:600],
                }
            )

    messages = [{"role": "user", "content": parts}]
    try:
        return await _chat_vlm(messages, max_tokens=2048, temperature=0.25, timeout=120.0)
    except Exception:
        logger.exception("vlm_answer_from_image_urls_failed")
        return ""
