from __future__ import annotations

import asyncio
import ipaddress
import logging
import mimetypes
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
import httpx
from fastapi import APIRouter, HTTPException, Query, Response

from app.adapters.min_adapter import MinAdapter
from app.core.metrics import metrics
from app.core.runtime_flags import add_runtime_flag, reset_runtime_flags
from app.core.settings import settings
from app.integrations.bridge_settings import bridge_settings
from app.models.schemas import GeneralQueryConstraints, ImageSearchConstraints, QueryRequest, QueryResponse
from app.services.connectors import BGERerankClient, CrawlClient, ImagePipelineClient, SearchClient
from app.services.clarification import maybe_resolve_pending, should_clarify
from app.services.dispatcher import TaskDispatcher
from app.services.image_query_parser import (
    parse_general_query_constraints,
    parse_image_search_constraints,
)
from app.services.memory_client import MemoryClient
from app.services.query_optimizer import optimize_image_query, optimize_image_query_with_constraints
from app.services.rag_client import RagClient
from app.services.rasa_client import RasaClient


router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# Singleton-like service instances for skeleton stage.
memory_client = MemoryClient(max_turns=settings.memory_max_turns)
rag_client = RagClient()
dispatcher = TaskDispatcher(
    search_client=SearchClient(),
    crawl_client=CrawlClient(),
    bge_rerank_client=BGERerankClient(),
    image_pipeline=ImagePipelineClient(),
)
adapter = MinAdapter(
    memory_client=memory_client,
    rag_client=rag_client,
    dispatcher=dispatcher,
)
rasa_client = RasaClient()


def _allowed_local_image_roots() -> list[Path]:
    root = Path(bridge_settings.raganything_working_dir).resolve()
    cache_dir = Path(bridge_settings.image_cache_dir).resolve()
    remote_images = root / "remote_images"
    allowed = [cache_dir, remote_images]
    out: list[Path] = []
    for path in allowed:
        try:
            out.append(path.resolve())
        except OSError:
            continue
    return out


def _is_allowed_local_image_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in _allowed_local_image_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_probable_image_file(path: Path) -> bool:
    ctype, _ = mimetypes.guess_type(str(path))
    return bool(ctype and ctype.startswith("image/"))


def _is_blocked_proxy_ip(raw: str) -> bool:
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


async def _is_blocked_image_proxy_host(hostname: str | None) -> bool:
    if not hostname:
        return True
    host = hostname.strip().rstrip(".").lower()
    if host in {"localhost"} or host.endswith(".localhost"):
        return True
    if _is_blocked_proxy_ip(host):
        return True

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror:
        return True

    resolved_ips = {info[4][0] for info in infos if info and info[4]}
    return any(_is_blocked_proxy_ip(ip) for ip in resolved_ips)


def _heuristic_image_search_intent(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    strong_phrases = [
        "给我几张",
        "来几张",
        "找几张",
        "搜几张",
        "发几张",
        "看几张",
        "图片",
        "照片",
        "壁纸",
        "表情包",
        "截图",
        "image",
        "photo",
        "photos",
        "picture",
        "pictures",
    ]
    return any(token in text for token in strong_phrases)


def _heuristic_general_qa_intent(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    general_markers = [
        "天气",
        "温度",
        "下雨",
        "对比",
        "区别",
        "哪个好",
        "为什么",
        "怎么",
        "如何",
        "是什么",
        "未来",
        "今天",
        "明天",
    ]
    return any(token in text for token in general_markers)


def _apply_image_constraints(
    req: QueryRequest,
    constraints: ImageSearchConstraints | None,
    entities: dict[str, str],
) -> QueryRequest:
    if constraints is None:
        return _apply_image_entities(req, entities)

    max_images = req.max_images
    if isinstance(constraints.count, int):
        max_images = max(1, min(int(constraints.count), 12))
    rewritten = optimize_image_query_with_constraints(
        req.original_query or req.query,
        constraints,
        entities,
    )
    return req.model_copy(
        update={
            # Keep req.query as user semantic query for downstream answering/context,
            # use dedicated image_search_query for retrieval to avoid intent/answer drift.
            "image_search_query": rewritten,
            "max_images": max_images,
            "image_constraints": constraints,
        }
    )


def _apply_general_constraints(
    req: QueryRequest,
    constraints: GeneralQueryConstraints | None,
) -> QueryRequest:
    if constraints is None:
        return req
    rewritten = (constraints.search_rewrite or "").strip() or req.query
    return req.model_copy(
        update={
            "query": rewritten,
            "general_constraints": constraints,
        }
    )


@router.get("/chat/image-proxy")
async def chat_image_proxy(
    url: str | None = Query(default=None, min_length=8, max_length=2048),
    local_path: str | None = Query(default=None, min_length=3, max_length=4096),
) -> Response:
    if local_path:
        path = Path(local_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Local image not found")
        if not _is_allowed_local_image_path(path):
            raise HTTPException(status_code=403, detail="Local image path is not allowed")
        if not _is_probable_image_file(path):
            raise HTTPException(status_code=415, detail="Local file is not an image")
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        try:
            return Response(
                content=path.read_bytes(),
                media_type=content_type,
                headers={"Cache-Control": "public, max-age=86400"},
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Failed to read local image") from exc

    if not url:
        raise HTTPException(status_code=400, detail="Either url or local_path is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https image URL is allowed")
    if await _is_blocked_image_proxy_host(parsed.hostname):
        raise HTTPException(status_code=400, detail="Image proxy host is not allowed")
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds, follow_redirects=True, trust_env=False
        ) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Upstream URL is not an image")
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {type(exc).__name__}") from exc


@router.post("/chat/query", response_model=QueryResponse)
async def chat_query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    reset_runtime_flags()
    metrics.inc("requests_total")
    intent_source = "request"
    intent_fallback = False
    effective_intent = req.intent
    parsed_entities: dict[str, str] = {}
    if effective_intent is None and req.use_rasa_intent:
        intent_source = "rasa"
        parsed_intent, confidence, parsed_entities = await rasa_client.parse(req.query)
        if (
            parsed_intent is not None
            and confidence >= req.intent_confidence_threshold
        ):
            # Guardrail: reject weakly supported image_search classification.
            if (
                parsed_intent == "image_search"
                and not _heuristic_image_search_intent(req.query)
                and _heuristic_general_qa_intent(req.query)
            ):
                intent_fallback = True
                add_runtime_flag("intent_rasa_image_rejected")
            else:
                effective_intent = parsed_intent
        else:
            intent_fallback = True

    if effective_intent is None:
        if _heuristic_image_search_intent(req.query):
            effective_intent = "image_search"
            add_runtime_flag("intent_heuristic_image_search")
        elif _heuristic_general_qa_intent(req.query):
            effective_intent = "general_qa"
            add_runtime_flag("intent_heuristic_general_qa")
        else:
            effective_intent = "general_qa"
        if req.use_rasa_intent:
            intent_fallback = True
    if intent_fallback:
        add_runtime_flag("intent_fallback")
        metrics.inc("intent_fallback_total")

    try:
        context = await memory_client.get_context(req.uid)
        preferences = context.get("preferences", {})
        req = req.model_copy(update={"original_query": req.original_query or req.query})

        # If user is replying to a prior clarification question, try to resolve it first.
        resolved, merged_query, next_pending, resolved_intent = maybe_resolve_pending(
            query=req.query,
            pending=preferences.get("pending_clarification"),
            preferences=preferences,
        )
        if resolved and merged_query:
            req = req.model_copy(update={"query": merged_query, "original_query": merged_query})
            if resolved_intent in {"general_qa", "image_search"}:
                effective_intent = resolved_intent
            await memory_client.set_preference(req.uid, "pending_clarification", {})
        elif next_pending:
            # Keep pending unchanged.
            await memory_client.set_preference(req.uid, "pending_clarification", next_pending)

        req = req.model_copy(update={"intent": effective_intent})
        if effective_intent == "image_search":
            constraints = await parse_image_search_constraints(
                req.original_query or req.query,
                parsed_entities,
            )
            req = _apply_image_constraints(req, constraints, parsed_entities)
            if constraints.parser_source == "llm":
                add_runtime_flag("image_query_llm_parser")
            else:
                add_runtime_flag("image_query_heuristic_parser")
            if constraints.search_rewrite and constraints.search_rewrite != (req.original_query or ""):
                add_runtime_flag("image_query_rewritten")
            if constraints.needs_clarification and constraints.clarification_question:
                add_runtime_flag("clarification_needed")
                metrics.inc("clarification_needed_total")
                await memory_client.set_preference(
                    req.uid,
                    "pending_clarification",
                    {
                        "scenario": "image_constraints",
                        "missing_slots": [],
                        "original_query": req.original_query or req.query,
                    },
                )
                return QueryResponse(
                    answer=constraints.clarification_question,
                    evidence=[],
                    images=[],
                    trace_id=f"tr_{uuid4().hex[:10]}",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    route=effective_intent,
                    runtime_flags=["clarification_needed"],
                )
        else:
            general_constraints = await parse_general_query_constraints(
                req.original_query or req.query
            )
            req = _apply_general_constraints(req, general_constraints)
            if general_constraints.parser_source == "llm":
                add_runtime_flag("general_query_llm_parser")
            else:
                add_runtime_flag("general_query_heuristic_parser")
            if general_constraints.search_rewrite and general_constraints.search_rewrite != (req.original_query or ""):
                add_runtime_flag("general_query_rewritten")
            if general_constraints.needs_clarification and general_constraints.clarification_question:
                add_runtime_flag("clarification_needed")
                metrics.inc("clarification_needed_total")
                await memory_client.set_preference(
                    req.uid,
                    "pending_clarification",
                    {
                        "scenario": "general_constraints",
                        "missing_slots": [],
                        "original_query": req.original_query or req.query,
                    },
                )
                return QueryResponse(
                    answer=general_constraints.clarification_question,
                    evidence=[],
                    images=[],
                    trace_id=f"tr_{uuid4().hex[:10]}",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    route=effective_intent,
                    runtime_flags=["clarification_needed"],
                )

        decision = should_clarify(
            query=req.query,
            intent=effective_intent,
            entities=parsed_entities,
            preferences=preferences,
        )
        if decision.should_ask:
            add_runtime_flag("clarification_needed")
            metrics.inc("clarification_needed_total")
            await memory_client.set_preference(
                req.uid,
                "pending_clarification",
                {
                    "scenario": decision.scenario,
                    "missing_slots": decision.missing_slots or [],
                    "original_query": req.query,
                },
            )
            return QueryResponse(
                answer=decision.question,
                evidence=[],
                images=[],
                trace_id=f"tr_{uuid4().hex[:10]}",
                latency_ms=int((time.perf_counter() - started) * 1000),
                route=effective_intent,
                runtime_flags=["clarification_needed"],
            )
        if decision.rewritten_query:
            req = req.model_copy(update={"query": decision.rewritten_query})

        normalized = await adapter.normalize_input(req)
        await adapter.ingest_to_rag(normalized)
        result = await adapter.query_with_context(normalized)
        total_latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.inc("requests_success_total")
        metrics.add_latency(total_latency_ms=total_latency_ms, rag_latency_ms=result.latency_ms)
        logger.info(
            "chat_query_done uid=%s trace_id=%s route=%s success=true total_latency_ms=%d rag_latency_ms=%d intent_source=%s intent_fallback=%s",
            req.uid,
            result.trace_id,
            result.route,
            total_latency_ms,
            result.latency_ms,
            intent_source,
            intent_fallback,
        )
        return result
    except Exception:
        total_latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.inc("requests_failed_total")
        logger.exception(
            "chat_query_done uid=%s success=false total_latency_ms=%d intent_source=%s intent_fallback=%s",
            req.uid,
            total_latency_ms,
            intent_source,
            intent_fallback,
        )
        raise


def _apply_image_entities(req: QueryRequest, entities: dict[str, str]) -> QueryRequest:
    landmark = entities.get("landmark")
    time_of_day = entities.get("time_of_day")
    image_count = entities.get("image_count")

    max_images = req.max_images
    if image_count:
        parsed_count = _parse_count(image_count)
        if parsed_count is not None:
            max_images = max(1, min(parsed_count, 12))

    if not landmark and not time_of_day:
        return req.model_copy(update={"max_images": max_images, "image_search_query": req.query})

    # Use entities to rewrite image query for better recall precision.
    rewritten = optimize_image_query(req.query, entities)
    return req.model_copy(update={"image_search_query": rewritten, "max_images": max_images})


def _parse_count(raw: str) -> int | None:
    m = re.search(r"\d{1,2}", raw)
    if m:
        return int(m.group(0))
    mapping = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "几": 5,
    }
    for k, v in mapping.items():
        if k in raw:
            return v
    return None


async def close_services() -> None:
    await memory_client.aclose()
