from __future__ import annotations

import asyncio
import ipaddress
import logging
import mimetypes
import re
import socket
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4
import httpx
from fastapi import APIRouter, HTTPException, Query, Response

from app.adapters.min_adapter import MinAdapter
from app.core.metrics import metrics
from app.core.progress import progress_complete, progress_error, progress_event, progress_get, progress_start
from app.core.runtime_flags import add_runtime_flag, reset_runtime_flags
from app.core.settings import settings
from app.integrations.bridge_settings import bridge_settings
from app.models.schemas import (
    GeneralQueryConstraints,
    ImageSearchConstraints,
    QueryExecutionContext,
    QueryRequest,
    QueryResponse,
)
from app.services.connectors import BGERerankClient, CrawlClient, ImagePipelineClient, SearchClient
from app.services.clarification import build_pending_state, decide_clarification, maybe_resolve_pending
from app.services.dispatcher import TaskDispatcher
from app.services.image_query_parser import (
    parse_general_query_constraints,
    parse_image_search_constraints,
)
from app.services.memory_client import MemoryClient
from app.services.query_planner import plan_query
from app.services.query_optimizer import optimize_image_query
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


async def _validate_remote_image_proxy_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https image URL is allowed")
    if await _is_blocked_image_proxy_host(parsed.hostname):
        raise HTTPException(status_code=400, detail="Image proxy host is not allowed")


async def _fetch_remote_image_for_proxy(raw_url: str) -> httpx.Response:
    current_url = raw_url
    max_redirects = max(0, int(settings.image_proxy_max_redirects))
    async with httpx.AsyncClient(
        timeout=settings.request_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for redirect_count in range(max_redirects + 1):
            await _validate_remote_image_proxy_url(current_url)
            resp = await client.get(current_url)
            if resp.status_code not in {301, 302, 303, 307, 308}:
                return resp
            location = resp.headers.get("location")
            if not location:
                return resp
            if redirect_count >= max_redirects:
                raise HTTPException(status_code=400, detail="Image proxy redirect limit exceeded")
            current_url = urljoin(current_url, location)
    raise HTTPException(status_code=400, detail="Image proxy redirect limit exceeded")


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
    ctx: QueryExecutionContext,
    constraints: ImageSearchConstraints | None,
    entities: dict[str, str],
) -> QueryExecutionContext:
    if constraints is None:
        return _apply_image_entities(ctx, entities)

    max_images = ctx.max_images
    if isinstance(constraints.count, int):
        max_images = max(1, min(int(constraints.count), 12))
    rewritten = (constraints.search_rewrite or "").strip() or ctx.query
    return ctx.model_copy(
        update={
            # Keep req.query as user semantic query for downstream answering/context,
            # use dedicated image_search_query for retrieval to avoid intent/answer drift.
            "image_search_query": rewritten,
            "max_images": max_images,
            "image_constraints": constraints,
        }
    )


def _apply_general_constraints(
    ctx: QueryExecutionContext,
    constraints: GeneralQueryConstraints | None,
) -> QueryExecutionContext:
    if constraints is None:
        return ctx
    rewritten = (constraints.search_rewrite or "").strip() or ctx.query
    return ctx.model_copy(
        update={
            "query": rewritten,
            "general_constraints": constraints,
        }
    )


def _direct_general_constraints_for_context(ctx: QueryExecutionContext) -> GeneralQueryConstraints:
    return GeneralQueryConstraints(
        raw_query=ctx.original_query or ctx.query,
        search_rewrite=ctx.query,
        parser_source="request_context",
    )


def _planner_user_context(preferences: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("profile", "response", "retrieval"):
        value = preferences.get(key)
        if isinstance(value, dict) and value:
            out[key] = value
    # Backward-compatible migration hints for older memory keys.
    legacy_location = preferences.get("default_city") or preferences.get("location")
    if legacy_location:
        profile = dict(out.get("profile") or {})
        profile.setdefault("location", legacy_location)
        out["profile"] = profile
    legacy_style = preferences.get("answer_style")
    if legacy_style:
        response = dict(out.get("response") or {})
        response.setdefault("style", legacy_style)
        out["response"] = response
    return out


async def _return_clarification_response(
    *,
    ctx: QueryExecutionContext,
    decision,
    request_id: str,
    started: float,
    route,
) -> QueryResponse:
    add_runtime_flag("clarification_needed")
    metrics.inc("clarification_needed_total")
    pending = build_pending_state(
        decision,
        original_query=ctx.original_query or ctx.query,
    )
    await memory_client.set_preference(ctx.uid, "pending_clarification", pending)
    progress_event(
        request_id,
        "clarification.ask",
        "需要补充关键信息。",
        {
            "route": pending.get("route"),
            "missing": pending.get("missing", []),
        },
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    progress_complete(
        request_id,
        {
            "clarification_needed": True,
            "route": route,
            "latency_ms": latency_ms,
        },
    )
    return QueryResponse(
        answer=decision.question,
        evidence=[],
        images=[],
        trace_id=f"tr_{uuid4().hex[:10]}",
        latency_ms=latency_ms,
        route=route,
        runtime_flags=["clarification_needed"],
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
    try:
        resp = await _fetch_remote_image_for_proxy(url)
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


@router.get("/chat/progress")
async def chat_progress(
    request_id: str = Query(..., min_length=6, max_length=128),
) -> dict:
    item = progress_get(request_id)
    if not item:
        return {"request_id": request_id, "status": "not_found", "events": []}
    return item


@router.post("/chat/query", response_model=QueryResponse)
async def chat_query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    reset_runtime_flags()
    metrics.inc("requests_total")
    request_id = (req.request_id or f"req_{uuid4().hex[:12]}").strip()
    req = req.model_copy(update={"request_id": request_id})
    progress_start(request_id, query=req.query, intent=req.intent)
    intent_source = "request"
    intent_fallback = False
    effective_intent = req.intent
    parsed_entities: dict[str, str] = {}
    planned_image_constraints: ImageSearchConstraints | None = None
    planned_general_constraints: GeneralQueryConstraints | None = None
    pending_intent_hint: str | None = None
    original_query = req.query
    planner_attempted = False
    planner_succeeded = False
    try:
        context = await memory_client.get_context(req.uid)
        preferences = context.get("preferences", {})

        # Resolve clarification replies before planning so the planner sees a complete query.
        resolved, merged_query, next_pending, resolved_intent = maybe_resolve_pending(
            query=req.query,
            pending=preferences.get("pending_clarification"),
            preferences=preferences,
        )
        if resolved and merged_query:
            req = req.model_copy(update={"query": merged_query})
            original_query = merged_query
            if resolved_intent in {"general_qa", "image_search"}:
                pending_intent_hint = resolved_intent
            await memory_client.set_preference(req.uid, "pending_clarification", {})
            progress_event(
                request_id,
                "clarification.resolved",
                "已合并上一轮澄清回复。",
                {"intent_hint": pending_intent_hint},
            )
        elif next_pending:
            await memory_client.set_preference(req.uid, "pending_clarification", next_pending)
    except Exception as exc:
        metrics.inc("requests_failed_total")
        logger.exception("chat_query_memory_or_clarification_failed uid=%s", req.uid)
        progress_error(request_id, "chat_query_memory_or_clarification_failed")
        raise HTTPException(status_code=500, detail="chat_query_memory_or_clarification_failed") from exc

    if effective_intent is None:
        planner_attempted = True
        progress_event(request_id, "intent.planning", "正在进行意图规划与结构化解析。")
        plan = await plan_query(req, user_context=_planner_user_context(preferences))
        if plan is not None and plan.confidence >= req.intent_confidence_threshold:
            planner_succeeded = True
            intent_source = plan.source
            effective_intent = plan.intent
            parsed_entities = plan.entities
            planned_image_constraints = plan.image_constraints
            planned_general_constraints = plan.general_constraints
            for flag in plan.flags:
                add_runtime_flag(flag)
            progress_event(
                request_id,
                "intent.planning_done",
                "LLM 规划完成。",
                {"intent": effective_intent, "confidence": plan.confidence},
            )
        elif req.use_rasa_intent:
            intent_fallback = True

    if effective_intent is None and req.use_rasa_intent:
        progress_event(request_id, "intent.rasa", "LLM 规划未命中，回退到 Rasa 意图识别。")
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

    if effective_intent is None and pending_intent_hint in {"general_qa", "image_search"}:
        effective_intent = pending_intent_hint  # planner/Rasa did not resolve; use the stored scenario hint.
        intent_source = "pending_clarification"
        add_runtime_flag("intent_pending_clarification")

    if effective_intent is None:
        progress_event(request_id, "intent.heuristic", "回退到启发式意图识别。")
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
        ctx = QueryExecutionContext.from_request(
            req,
            intent=effective_intent,
            original_query=original_query,
        )
        progress_event(
            request_id,
            "intent.finalized",
            "意图判定完成，开始进入执行链路。",
            {"intent": effective_intent},
        )
        image_constraints: ImageSearchConstraints | None = None
        general_constraints: GeneralQueryConstraints | None = None
        allow_legacy_parser_llm = not planner_attempted or planner_succeeded
        if effective_intent == "image_search":
            progress_event(request_id, "image_search.parse_constraints", "正在解析图像检索约束。")
            image_constraints = planned_image_constraints or await parse_image_search_constraints(
                ctx.original_query or ctx.query,
                parsed_entities,
                allow_llm=allow_legacy_parser_llm,
            )
            ctx = _apply_image_constraints(ctx, image_constraints, parsed_entities)
            progress_event(
                request_id,
                "image_search.constraints_done",
                "图像约束解析完成。",
                {
                    "search_rewrite": image_constraints.search_rewrite,
                    "subjects": image_constraints.subjects,
                    "spatial_relations_count": len(image_constraints.spatial_relations),
                    "action_relations_count": len(image_constraints.action_relations),
                },
            )
            if image_constraints.parser_source in {"llm", "llm_planner"}:
                add_runtime_flag("image_query_llm_parser")
            else:
                add_runtime_flag("image_query_heuristic_parser")
            if image_constraints.search_rewrite and image_constraints.search_rewrite != (ctx.original_query or ""):
                add_runtime_flag("image_query_rewritten")
        else:
            progress_event(request_id, "general_qa.parse_constraints", "正在解析通用问答约束。")
            if planned_general_constraints is not None:
                general_constraints = planned_general_constraints
            elif ctx.source_docs or ctx.url:
                general_constraints = _direct_general_constraints_for_context(ctx)
            else:
                general_constraints = await parse_general_query_constraints(
                    ctx.original_query or ctx.query,
                    allow_llm=allow_legacy_parser_llm,
                )
            ctx = _apply_general_constraints(ctx, general_constraints)
            progress_event(
                request_id,
                "general_qa.constraints_done",
                "通用问答约束解析完成。",
                {
                    "search_rewrite": general_constraints.search_rewrite,
                    "city": general_constraints.city,
                    "compare_targets": general_constraints.compare_targets,
                },
            )
            if general_constraints.parser_source in {"llm", "llm_planner"}:
                add_runtime_flag("general_query_llm_parser")
            elif general_constraints.parser_source == "request_context":
                add_runtime_flag("general_query_context_direct")
            else:
                add_runtime_flag("general_query_heuristic_parser")
            if general_constraints.search_rewrite and general_constraints.search_rewrite != (ctx.original_query or ""):
                add_runtime_flag("general_query_rewritten")

        decision = decide_clarification(
            query=ctx.original_query or ctx.query,
            intent=effective_intent,
            entities=parsed_entities,
            preferences=preferences,
            image_constraints=image_constraints,
            general_constraints=general_constraints,
        )
        if decision.should_ask:
            return await _return_clarification_response(
                ctx=ctx,
                decision=decision,
                request_id=request_id,
                started=started,
                route=effective_intent,
            )
        if decision.rewritten_query:
            ctx = ctx.model_copy(update={"query": decision.rewritten_query})

        normalized = await adapter.normalize_input(ctx)
        total_images = sum(
            1 for d in normalized.documents for m in (d.modal_elements or []) if getattr(m, "type", None) == "image"
        )
        progress_event(
            request_id,
            "pipeline.normalized",
            "输入归一化完成。",
            {"documents_count": len(normalized.documents), "images_count": total_images},
        )
        progress_event(request_id, "pipeline.ingest", "正在将材料写入 RAG/索引层。")
        await adapter.ingest_to_rag(normalized)
        progress_event(request_id, "pipeline.query", "正在执行最终问答生成。")
        result = await adapter.query_with_context(normalized)
        total_latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.inc("requests_success_total")
        metrics.add_latency(total_latency_ms=total_latency_ms, rag_latency_ms=result.latency_ms)
        logger.info(
            "chat_query_done uid=%s trace_id=%s route=%s success=true total_latency_ms=%d rag_latency_ms=%d intent_source=%s intent_fallback=%s",
            ctx.uid,
            result.trace_id,
            result.route,
            total_latency_ms,
            result.latency_ms,
            intent_source,
            intent_fallback,
        )
        progress_complete(
            request_id,
            {
                "route": result.route,
                "latency_ms": total_latency_ms,
                "evidence_count": len(result.evidence or []),
                "images_count": len(result.images or []),
            },
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
        progress_error(request_id, "chat_query_exception")
        raise


def _apply_image_entities(ctx: QueryExecutionContext, entities: dict[str, str]) -> QueryExecutionContext:
    landmark = entities.get("landmark")
    time_of_day = entities.get("time_of_day")
    image_count = entities.get("image_count")

    max_images = ctx.max_images
    if image_count:
        parsed_count = _parse_count(image_count)
        if parsed_count is not None:
            max_images = max(1, min(parsed_count, 12))

    if not landmark and not time_of_day:
        return ctx.model_copy(update={"max_images": max_images, "image_search_query": ctx.query})

    # Use entities to rewrite image query for better recall precision.
    rewritten = optimize_image_query(ctx.query, entities)
    return ctx.model_copy(update={"image_search_query": rewritten, "max_images": max_images})


def _parse_count(raw: str) -> int | None:
    m = re.search(r"\d{1,2}", raw)
    if m:
        return int(m.group(0))
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "几": 5,
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


_IMAGE_INTENT_MARKERS_CLEAN = (
    "\u7ed9\u6211\u51e0\u5f20",
    "\u6765\u51e0\u5f20",
    "\u627e\u51e0\u5f20",
    "\u641c\u51e0\u5f20",
    "\u53d1\u51e0\u5f20",
    "\u770b\u51e0\u5f20",
    "\u56fe\u7247",
    "\u7167\u7247",
    "\u58c1\u7eb8",
    "\u8868\u60c5\u5305",
    "\u622a\u56fe",
    "image",
    "photo",
    "photos",
    "picture",
    "pictures",
)
_GENERAL_QA_INTENT_MARKERS_CLEAN = (
    "\u5929\u6c14",
    "\u6e29\u5ea6",
    "\u4e0b\u96e8",
    "\u5bf9\u6bd4",
    "\u533a\u522b",
    "\u54ea\u4e2a\u597d",
    "\u4e3a\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u5982\u4f55",
    "\u662f\u4ec0\u4e48",
    "\u672a\u6765",
    "\u4eca\u5929",
    "\u660e\u5929",
)
_GENERAL_QA_STRONG_MARKERS = (
    "\u603b\u7ed3",
    "\u5206\u6790",
    "\u89e3\u91ca",
    "\u8bf4\u660e",
    "\u63d0\u70bc",
    "\u5bf9\u6bd4",
    "\u533a\u522b",
    "\u5224\u65ad",
    "\u63a8\u8350",
    "\u600e\u4e48\u505a",
    "\u5982\u4f55",
    "\u4e3a\u4ec0\u4e48",
    "summarize",
    "summary",
    "explain",
    "compare",
    "analyze",
)
_IMAGE_SEARCH_STRONG_MARKERS = (
    "\u627e\u56fe",
    "\u641c\u56fe",
    "\u641c\u7d22\u56fe\u7247",
    "\u627e\u51e0\u5f20",
    "\u7ed9\u6211\u51e0\u5f20",
    "\u6765\u51e0\u5f20",
    "\u53d1\u51e0\u5f20",
    "\u56fe\u7247\u641c\u7d22",
    "\u6587\u641c\u56fe",
    "find images",
    "search images",
    "show me images",
)
_IMAGE_REFERENCE_MARKERS = (
    "\u8fd9\u5f20\u56fe",
    "\u56fe\u91cc",
    "\u56fe\u4e2d",
    "\u7167\u7247\u91cc",
    "\u770b\u56fe",
)
_CHINESE_COUNT_WORDS_CLEAN = {
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10,
    "\u51e0": 5,
}


def _heuristic_image_search_intent(query: str) -> bool:
    return _score_intent(query)["image_search"] > _score_intent(query)["general_qa"]


def _heuristic_general_qa_intent(query: str) -> bool:
    scores = _score_intent(query)
    return scores["general_qa"] >= scores["image_search"] and scores["general_qa"] > 0


def _score_intent(query: str, *, has_url: bool = False, has_source_docs: bool = False, has_images: bool = False) -> dict[str, int]:
    text = (query or "").strip().lower()
    scores = {"image_search": 0, "general_qa": 0}
    if not text:
        return scores

    scores["image_search"] += 4 * sum(marker in text for marker in _IMAGE_SEARCH_STRONG_MARKERS)
    scores["image_search"] += 2 * sum(marker in text for marker in _IMAGE_INTENT_MARKERS_CLEAN)
    scores["general_qa"] += 3 * sum(marker in text for marker in _GENERAL_QA_STRONG_MARKERS)
    scores["general_qa"] += 2 * sum(marker in text for marker in _GENERAL_QA_INTENT_MARKERS_CLEAN)
    scores["general_qa"] += 2 * sum(marker in text for marker in _IMAGE_REFERENCE_MARKERS)

    if has_url or has_source_docs:
        scores["general_qa"] += 4
    if has_images:
        scores["general_qa"] += 3

    if re.search(r"(\d{1,2}|[\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u51e0])\s*(\u5f20|\u5f20\u56fe|\u5f20\u7167\u7247|images?|photos?|pictures?)", text):
        scores["image_search"] += 3
    if re.search(r"(\?|？)$", text):
        scores["general_qa"] += 1
    return scores


def _infer_intent(req: QueryRequest) -> IntentType:
    scores = _score_intent(
        req.query,
        has_url=bool(req.url),
        has_source_docs=bool(req.source_docs),
        has_images=bool(req.images),
    )
    if scores["image_search"] >= scores["general_qa"] + 2 and scores["image_search"] > 0:
        add_runtime_flag("intent_heuristic_image_search")
        return "image_search"
    if scores["general_qa"] > 0:
        add_runtime_flag("intent_heuristic_general_qa")
    return "general_qa"


def _should_reject_rasa_intent(req: QueryRequest, parsed_intent: str) -> bool:
    scores = _score_intent(
        req.query,
        has_url=bool(req.url),
        has_source_docs=bool(req.source_docs),
        has_images=bool(req.images),
    )
    if parsed_intent == "image_search":
        return scores["general_qa"] >= scores["image_search"] + 2
    if parsed_intent == "general_qa":
        return scores["image_search"] >= scores["general_qa"] + 3
    return False


def _parse_count(raw: str) -> int | None:
    m = re.search(r"\d{1,2}", raw)
    if m:
        return int(m.group(0))
    for key, value in _CHINESE_COUNT_WORDS_CLEAN.items():
        if key in raw:
            return value
    return None


async def close_services() -> None:
    await memory_client.aclose()
