from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
import logging
import re
import hashlib
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, Field
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from app.integrations.bridge_settings import bridge_settings

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Image Pipeline Bridge", version="0.1.0")


class ImageSearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None)


@dataclass
class ImageCandidate:
    url: str
    title: str = ""
    desc: str = ""
    source: str = "unknown"
    score: float = 0.0
    local_path: str | None = None


_clip_model: ChineseCLIPModel | None = None
_clip_processor: ChineseCLIPProcessor | None = None
_clip_device = "cuda" if torch.cuda.is_available() else "cpu"
_last_cache_cleanup_ts = 0.0


def _clamp_int(value: int, min_v: int | None = None, max_v: int | None = None) -> int:
    if min_v is not None:
        value = max(min_v, value)
    if max_v is not None:
        value = min(max_v, value)
    return value


def _cfg_int(value: int | None, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    raw = default if value is None else int(value)
    return _clamp_int(raw, min_v=min_v, max_v=max_v)


def _cfg_float(value: float | None, default: float) -> float:
    return float(default if value is None else value)


def _image_cache_dir() -> Path:
    p = Path(bridge_settings.image_cache_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _maybe_cleanup_image_cache() -> None:
    global _last_cache_cleanup_ts
    ttl_hours = max(0, int(bridge_settings.image_cache_ttl_hours))
    if ttl_hours <= 0:
        return
    now = time.time()
    interval_seconds = max(60, int(bridge_settings.image_cache_cleanup_interval_seconds))
    if now - _last_cache_cleanup_ts < interval_seconds:
        return
    _last_cache_cleanup_ts = now

    expire_before = now - ttl_hours * 3600
    cache_dir = _image_cache_dir()
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < expire_before:
                path.unlink()
        except OSError:
            continue


def _guess_image_ext(url: str, content_type: str | None) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}:
        return suffix
    return ".jpg"


def _cached_image_path(url: str, content_type: str | None = None) -> Path:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return _image_cache_dir() / f"{digest}{_guess_image_ext(url, content_type)}"


async def _ensure_cached_image_file(url: str, timeout_seconds: int = 15) -> str | None:
    _maybe_cleanup_image_cache()
    existing = _cached_image_path(url)
    if existing.exists():
        try:
            existing.touch()
        except OSError:
            pass
        return str(existing)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not ctype.startswith("image/"):
            return None
        path = _cached_image_path(url, ctype)
        if not path.exists():
            path.write_bytes(resp.content)
        return str(path)
    except Exception:
        return None


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _lexical_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    hit = sum(1 for tok in q if tok in t)
    return hit / max(len(q), 1)


def _load_chinese_clip() -> tuple[ChineseCLIPModel | None, ChineseCLIPProcessor | None]:
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    model_name = bridge_settings.chinese_clip_model.strip()
    local_only = bridge_settings.chinese_clip_local_files_only
    local_candidates = [
        str((Path.cwd() / "models" / "chinese-clip-vit-base-patch16").resolve()),
        model_name,
    ]
    model_candidates = list(dict.fromkeys([x for x in local_candidates if x.strip()]))

    for candidate in model_candidates:
        attempts = [local_only]
        if local_only:
            # Fallback to remote loading when local cache is missing.
            attempts.append(False)
        for local_flag in attempts:
            try:
                processor = ChineseCLIPProcessor.from_pretrained(
                    candidate, local_files_only=local_flag
                )
                model = ChineseCLIPModel.from_pretrained(
                    candidate, local_files_only=local_flag
                ).to(_clip_device)
                model.eval()
                _clip_model = model
                _clip_processor = processor
                logger.info(
                    "chinese_clip_load_ok model=%s local_only=%s device=%s",
                    candidate,
                    local_flag,
                    _clip_device,
                )
                return _clip_model, _clip_processor
            except Exception:
                logger.exception(
                    "chinese_clip_load_failed model=%s local_only=%s",
                    candidate,
                    local_flag,
                )
                continue
    return None, None


async def _search_serpapi(query: str, limit: int) -> list[ImageCandidate]:
    # Support key rotation/fallback:
    # 1) SERPAPI_API_KEYS (comma-separated)
    # 2) SERPAPI_API_KEY (single key)
    raw_multi = (bridge_settings.serpapi_api_keys or "").strip()
    key_candidates: list[str] = []
    if raw_multi:
        key_candidates.extend([k.strip() for k in raw_multi.split(",") if k.strip()])
    single = (bridge_settings.serpapi_api_key or "").strip()
    if single:
        key_candidates.append(single)
    # dedupe while preserving order
    api_keys = list(dict.fromkeys(key_candidates))
    if not api_keys:
        return []

    async with httpx.AsyncClient(timeout=25, trust_env=False) as client:
        for api_key in api_keys:
            params = {
                "engine": "google_images",
                "q": query,
                "api_key": api_key,
                "ijn": "0",
            }
            try:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                # If key is rate-limited/invalid, try next key.
                if resp.status_code in {401, 403, 429}:
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    continue
            except (httpx.HTTPError, ValueError, TypeError):
                continue

            images = data.get("images_results") or []
            candidates: list[ImageCandidate] = []
            for item in images[: max(limit, 10)]:
                if not isinstance(item, dict):
                    continue
                url = item.get("original") or item.get("thumbnail")
                if not url:
                    continue
                candidates.append(
                    ImageCandidate(
                        url=str(url),
                        title=str(item.get("title") or ""),
                        desc=str(item.get("snippet") or ""),
                        source="serpapi",
                    )
                )
            if candidates:
                return candidates

    return []


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


async def _search_serpapi_with_debug(
    query: str,
    limit: int,
    min_accessible: int,
) -> tuple[list[ImageCandidate], list[dict[str, Any]]]:
    raw_multi = (bridge_settings.serpapi_api_keys or "").strip()
    key_candidates: list[str] = []
    if raw_multi:
        key_candidates.extend([k.strip() for k in raw_multi.split(",") if k.strip()])
    single = (bridge_settings.serpapi_api_key or "").strip()
    if single:
        key_candidates.append(single)
    api_keys = list(dict.fromkeys(key_candidates))

    debug_items: list[dict[str, Any]] = []
    if not api_keys:
        debug_items.append(
            {
                "key": "none",
                "usable": False,
                "status_code": None,
                "reason": "no_serpapi_key_configured",
            }
        )
        return [], debug_items

    async with httpx.AsyncClient(timeout=25, trust_env=False) as client:
        for idx, api_key in enumerate(api_keys):
            params = {
                "engine": "google_images",
                "q": query,
                "api_key": api_key,
                "ijn": "0",
            }
            masked = _mask_key(api_key)
            try:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                status_code = int(resp.status_code)
                data = resp.json()
                error_msg = data.get("error") if isinstance(data, dict) else None

                if status_code in {401, 403, 429} or error_msg:
                    debug_items.append(
                        {
                            "key": masked,
                            "index": idx,
                            "usable": False,
                            "status_code": status_code,
                            "reason": str(error_msg or f"http_{status_code}"),
                        }
                    )
                    continue

                images = data.get("images_results") or []
                candidates: list[ImageCandidate] = []
                for item in images[: max(limit, 10)]:
                    if not isinstance(item, dict):
                        continue
                    url = item.get("original") or item.get("thumbnail")
                    if not url:
                        continue
                    candidates.append(
                        ImageCandidate(
                            url=str(url),
                            title=str(item.get("title") or ""),
                            desc=str(item.get("snippet") or ""),
                            source="serpapi",
                        )
                    )

                accessible = await _filter_accessible_candidates(
                    candidates,
                    min_keep=max(1, min_accessible),
                    max_check=_cfg_int(bridge_settings.image_source_max_check, default=max(limit, min_accessible * 2), min_v=min_accessible, max_v=100),
                    concurrency=_cfg_int(bridge_settings.image_access_check_concurrency, default=8, min_v=1, max_v=24),
                    timeout_seconds=_cfg_int(bridge_settings.image_access_check_timeout, default=10, min_v=3, max_v=30),
                )
                if accessible:
                    debug_items.append(
                        {
                            "key": masked,
                            "index": idx,
                            "usable": True,
                            "status_code": status_code,
                            "reason": f"ok_accessible_{len(accessible)}_from_raw_{len(candidates)}",
                        }
                    )
                    return accessible, debug_items

                debug_items.append(
                    {
                        "key": masked,
                        "index": idx,
                        "usable": False,
                        "status_code": status_code,
                        "reason": f"no_accessible_images(min={max(1, min_accessible)},raw={len(candidates)})",
                    }
                )
            except Exception as e:
                debug_items.append(
                    {
                        "key": masked,
                        "index": idx,
                        "usable": False,
                        "status_code": None,
                        "reason": f"request_exception:{type(e).__name__}",
                    }
                )

    return [], debug_items


async def _search_unsplash_source(query: str, limit: int) -> list[ImageCandidate]:
    # Public query-driven image endpoint, no API key required.
    q = query.strip().replace(" ", ",")
    return [
        ImageCandidate(
            url=f"https://source.unsplash.com/featured/1024x768/?{q}&sig={idx}",
            title=f"unsplash result {idx}",
            desc=query,
            source="unsplash_source",
        )
        for idx in range(1, max(limit, 1) + 1)
    ]


def _clip_like_filter(query: str, candidates: list[ImageCandidate], top_k: int) -> list[ImageCandidate]:
    keep_n = _cfg_int(bridge_settings.image_clip_keep, default=top_k * 4, min_v=top_k)
    min_score = _cfg_float(bridge_settings.image_clip_min_score, default=0.18)
    for c in candidates:
        c.score = _lexical_score(query, f"{c.title} {c.desc}")
    ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
    filtered = [c for c in ranked if c.score >= min_score]
    if not filtered:
        filtered = ranked[:top_k]
    return filtered[:keep_n]


async def _download_image(url: str) -> Image.Image | None:
    try:
        local_path = await _ensure_cached_image_file(url, timeout_seconds=15)
        if not local_path:
            return None
        return Image.open(local_path).convert("RGB")
    except Exception:
        return None


async def _is_accessible_image_url(url: str, timeout_seconds: int) -> bool:
    return (await _ensure_cached_image_file(url, timeout_seconds=timeout_seconds)) is not None


async def _filter_accessible_candidates(
    candidates: list[ImageCandidate],
    *,
    min_keep: int,
    max_check: int,
    concurrency: int,
    timeout_seconds: int,
) -> list[ImageCandidate]:
    # Keep stable order and avoid repeated probing for duplicated URLs.
    unique: list[ImageCandidate] = []
    seen_urls: set[str] = set()
    for cand in candidates:
        if cand.url in seen_urls:
            continue
        seen_urls.add(cand.url)
        unique.append(cand)

    probe_pool = unique[: max(1, max_check)]
    if not probe_pool:
        return []

    sem = asyncio.Semaphore(max(1, concurrency))
    ok_map: dict[int, bool] = {}

    async def _probe(i: int, cand: ImageCandidate) -> None:
        async with sem:
            local_path = await _ensure_cached_image_file(cand.url, timeout_seconds=timeout_seconds)
            cand.local_path = local_path
            ok_map[i] = local_path is not None

    await asyncio.gather(*[_probe(i, c) for i, c in enumerate(probe_pool)])

    accessible = [c for i, c in enumerate(probe_pool) if ok_map.get(i, False)]
    if len(accessible) >= max(1, min_keep):
        return accessible
    return accessible


async def _ensure_accessible_topk(
    preferred: list[ImageCandidate],
    fallback_pool: list[ImageCandidate],
    top_k: int,
) -> list[ImageCandidate]:
    merged: list[ImageCandidate] = []
    seen: set[str] = set()
    for cand in [*preferred, *fallback_pool]:
        if cand.url in seen:
            continue
        seen.add(cand.url)
        merged.append(cand)

    checked = await _filter_accessible_candidates(
        merged,
        min_keep=top_k,
        max_check=_cfg_int(bridge_settings.image_final_max_check, default=max(top_k * 3, 15), min_v=top_k, max_v=80),
        concurrency=_cfg_int(bridge_settings.image_access_check_concurrency, default=8, min_v=1, max_v=24),
        timeout_seconds=_cfg_int(bridge_settings.image_access_check_timeout, default=10, min_v=3, max_v=30),
    )

    preferred_urls = {c.url for c in preferred}
    preferred_ok = [c for c in checked if c.url in preferred_urls]
    fallback_ok = [c for c in checked if c.url not in preferred_urls]
    return [*preferred_ok, *fallback_ok][:top_k]


async def _chinese_clip_filter(query: str, candidates: list[ImageCandidate], top_k: int) -> list[ImageCandidate]:
    model, processor = _load_chinese_clip()
    if model is None or processor is None:
        return _clip_like_filter(query, candidates, top_k=top_k)

    max_eval = _cfg_int(
        bridge_settings.image_clip_eval,
        default=min(len(candidates), top_k * 10),
        min_v=top_k,
    )
    keep_n = _cfg_int(bridge_settings.image_clip_keep, default=top_k * 4, min_v=top_k)
    min_score = _cfg_float(bridge_settings.image_clip_min_score, default=0.18)

    clip_download_concurrency = _cfg_int(
        bridge_settings.image_clip_download_concurrency, default=6, min_v=1, max_v=24
    )
    eval_pool = candidates[:max_eval]
    sem = asyncio.Semaphore(clip_download_concurrency)
    images_map: dict[int, Image.Image] = {}

    async def _dl(i: int, url: str) -> None:
        async with sem:
            img = await _download_image(url)
            if img is not None:
                images_map[i] = img

    await asyncio.gather(*[_dl(i, c.url) for i, c in enumerate(eval_pool)])
    valid_indices = sorted(images_map.keys())
    valid_images = [images_map[i] for i in valid_indices]

    if not valid_images:
        return _clip_like_filter(query, candidates, top_k=top_k)

    try:
        image_inputs = processor(images=valid_images, return_tensors="pt")
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        image_inputs = {k: v.to(_clip_device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(_clip_device) for k, v in text_inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            # Chinese-CLIP hf weights can have empty pooler_output; use CLS token projection directly.
            text_model_output = model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                token_type_ids=text_inputs.get("token_type_ids"),
            )
            text_features = model.text_projection(text_model_output.last_hidden_state[:, 0, :])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(-1)

        for i, score in enumerate(sims.tolist()):
            candidates[valid_indices[i]].score = float(score)
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        filtered = [c for c in ranked if c.score >= min_score]
        if not filtered:
            filtered = ranked[:top_k]
        return filtered[:keep_n]
    except Exception:
        return _clip_like_filter(query, candidates, top_k=top_k)


def _build_query_variants(query: str) -> list[str]:
    base = " ".join(query.strip().split())
    if not base:
        return []
    variants = [base]
    if bridge_settings.image_multi_query_enabled:
        # Variant 1: bias toward real-world scene photos.
        variants.append(f"{base} 实拍 场景")
        # Variant 2: lightweight bilingual expansion for common dog breeds.
        en = base
        en = en.replace("金毛", "Golden Retriever")
        en = en.replace("边牧", "Border Collie")
        en = en.replace("同框", "together photo")
        en = en.replace("左边", "left")
        en = en.replace("右边", "right")
        if en != base:
            variants.append(en)
    uniq = list(dict.fromkeys([v.strip() for v in variants if v.strip()]))
    max_n = _cfg_int(bridge_settings.image_multi_query_max_variants, default=2, min_v=1, max_v=4)
    return uniq[:max_n]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search-rank")
async def search_rank(req: ImageSearchRequest) -> dict[str, Any]:
    default_top_k = _cfg_int(bridge_settings.image_top_k_default, default=5, min_v=1, max_v=20)
    request_top_k = req.top_k if req.top_k is not None else default_top_k
    top_k = max(1, min(request_top_k, 20))
    retrieval_k = _cfg_int(bridge_settings.image_retrieval_k, default=top_k * 10, min_v=top_k, max_v=50)
    source_min_accessible = _cfg_int(
        bridge_settings.image_source_min_accessible, default=top_k * 2, min_v=top_k, max_v=50
    )
    provider = bridge_settings.image_search_provider.strip().lower()

    candidates: list[ImageCandidate] = []
    serpapi_debug: list[dict[str, Any]] = []
    query_variants = _build_query_variants(req.query)
    if provider == "serpapi":
        merged: list[ImageCandidate] = []
        seen_urls: set[str] = set()
        for q in query_variants:
            one, dbg = await _search_serpapi_with_debug(
                q,
                retrieval_k,
                min_accessible=source_min_accessible,
            )
            serpapi_debug.append({"query": q, "attempts": dbg, "count": len(one)})
            for c in one:
                if c.url in seen_urls:
                    continue
                seen_urls.add(c.url)
                merged.append(c)
        candidates = merged

    fallback_used = False
    if not candidates:
        candidates = await _search_unsplash_source(req.query, retrieval_k)
        fallback_used = True

    filtered = await _chinese_clip_filter(req.query, candidates, top_k=top_k)

    final_images = await _ensure_accessible_topk(
        preferred=filtered[:top_k],
        fallback_pool=filtered,
        top_k=top_k,
    )

    return {
        "images": [
            {
                "url": c.url,
                "desc": c.desc or c.title,
                "title": c.title,
                "score": c.score,
                "source": c.source,
                "local_path": c.local_path,
            }
            for c in final_images
        ],
        "debug": {
            "provider": provider,
            "fallback_used": fallback_used,
            "query_variants": query_variants,
            "serpapi_keys": serpapi_debug,
        },
    }
