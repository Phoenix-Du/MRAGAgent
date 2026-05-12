from __future__ import annotations

import re
import time
from uuid import uuid4

from app.core.progress import progress_event
from app.core.runtime_flags import add_runtime_flag
from app.core.retry import with_retry
from app.core.runtime_flags import get_runtime_flags
from app.core.settings import settings
from app.models.schemas import (
    NormalizedDocument,
    NormalizedPayload,
    QueryExecutionContext,
    QueryResponse,
    SourceDoc,
)
from app.services.dispatcher import TaskDispatcher
from app.services.image_search_vlm_answer import build_image_search_vlm_response
from app.services.memory_client import MemoryClient
from app.services.rag_client import RagClient


class MinAdapter:
    """
    Minimal adapter with 3 core functions:
    1) normalize_input
    2) ingest_to_rag
    3) query_with_context
    """

    def __init__(
        self,
        memory_client: MemoryClient,
        rag_client: RagClient,
        dispatcher: TaskDispatcher,
    ) -> None:
        self.memory_client = memory_client
        self.rag_client = rag_client
        self.dispatcher = dispatcher

    async def normalize_input(self, payload: QueryExecutionContext) -> NormalizedPayload:
        documents: list[NormalizedDocument] = []

        source_docs, prepared_images = await self.dispatcher.prepare_documents(payload)

        for src in source_docs:
            documents.append(self._from_source_doc(src))

        # If no source docs but user submitted image hints, build a lightweight document.
        if not documents and prepared_images:
            documents.append(
                NormalizedDocument(
                    doc_id=f"img-{uuid4().hex[:8]}",
                    text=payload.query,
                    modal_elements=prepared_images,
                    metadata={"source": "image_branch", "url": payload.url},
                )
            )

        # Fallback: ensure each query can become one searchable doc.
        if not documents:
            documents.append(
                NormalizedDocument(
                    doc_id=f"txt-{uuid4().hex[:8]}",
                    text=payload.query,
                    modal_elements=[],
                    metadata={"source": "direct_query", "url": payload.url},
                )
            )

        return NormalizedPayload(
            uid=payload.uid,
            request_id=payload.request_id,
            intent=payload.intent,
            query=payload.query,
            image_search_query=payload.image_search_query,
            original_query=payload.original_query or payload.query,
            max_images=payload.max_images,
            image_constraints=payload.image_constraints,
            general_constraints=payload.general_constraints,
            documents=documents,
        )

    async def ingest_to_rag(self, normalized: NormalizedPayload) -> list[str]:
        if normalized.intent == "image_search" and not settings.image_search_ingest_enabled:
            add_runtime_flag("image_search_ingest_skipped")
            if normalized.request_id:
                progress_event(
                    normalized.request_id,
                    "pipeline.ingest_skipped",
                    "图搜链路已按配置跳过 RAG ingest。",
                )
            return []

        tags = {"uid": normalized.uid, "intent": normalized.intent}
        if normalized.request_id:
            progress_event(
                normalized.request_id,
                "pipeline.ingest_running",
                "RAG ingest 正在执行。",
                {"documents_count": len(normalized.documents)},
            )

        async def _ingest() -> list[str]:
            return await self.rag_client.ingest_documents(normalized.documents, tags)

        indexed = await with_retry(_ingest)
        if normalized.request_id:
            progress_event(
                normalized.request_id,
                "pipeline.ingest_done",
                "RAG ingest 完成。",
                {"indexed_count": len(indexed)},
            )
        return indexed

    async def query_with_context(self, normalized: NormalizedPayload) -> QueryResponse:
        start_ms = time.perf_counter()
        trace_id = f"tr_{uuid4().hex[:10]}"

        context = await self.memory_client.get_context(normalized.uid)
        if normalized.request_id:
            progress_event(
                normalized.request_id,
                "pipeline.context_loaded",
                "会话上下文加载完成。",
                {"history_count": len(context.get("history") or [])},
            )
        preferences = context.get("preferences", {})
        response_pref = preferences.get("response") if isinstance(preferences, dict) else {}
        pref_hint = ""
        if isinstance(response_pref, dict):
            pref_hint = str(response_pref.get("style") or "").strip()
        if not pref_hint and isinstance(preferences, dict):
            pref_hint = str(preferences.get("answer_style") or "").strip()

        enhanced_query = normalized.query
        if pref_hint:
            enhanced_query = f"{normalized.query}\n[用户偏好回答风格]: {pref_hint}"
        # Force answer language and weather-focus constraints for general QA.
        if normalized.intent == "general_qa":
            constraints: list[str] = ["请使用简体中文回答，不要使用英文。"]
            if _is_weather_query(normalized.query):
                constraints.append(
                    "这是天气问答。只回答天气信息（如天气现象、温度范围、降雨、风力、湿度、空气质量、时间范围），"
                    "不要介绍城市历史、景点、旅游信息。若证据不足请明确说明。"
                )
                add_runtime_flag("general_qa_weather_focus")
            enhanced_query = f"{enhanced_query}\n\n[回答约束]\n" + "\n".join(constraints)

        if normalized.intent == "image_search":
            if normalized.request_id:
                progress_event(
                    normalized.request_id,
                    "image_search.answering",
                    "正在执行多模态排序与答案生成。",
                    {"max_images": normalized.max_images},
                )
            result = await build_image_search_vlm_response(
                query=(normalized.original_query or normalized.query),
                documents=normalized.documents,
                max_images=normalized.max_images,
                image_constraints=normalized.image_constraints,
                trace_id=trace_id,
            )
        else:
            if normalized.request_id:
                progress_event(
                    normalized.request_id,
                    "general_qa.answering",
                    "正在执行 RAG 查询与回答生成。",
                )

            async def _query() -> QueryResponse:
                return await self.rag_client.query(
                    query=enhanced_query,
                    uid=normalized.uid,
                    trace_id=trace_id,
                )

            result = await with_retry(_query)
        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)
        result.latency_ms = elapsed_ms
        result.route = normalized.intent
        result.runtime_flags = get_runtime_flags()

        await self.memory_client.update_context(
            uid=normalized.uid,
            query=normalized.query,
            answer=result.answer,
            intent=normalized.intent,
        )
        if normalized.request_id:
            progress_event(
                normalized.request_id,
                "pipeline.answer_done",
                "答案生成与上下文写回完成。",
                {"latency_ms": elapsed_ms},
            )
        return result

    @staticmethod
    def _from_source_doc(src: SourceDoc) -> NormalizedDocument:
        text = src.text_content or ""
        meta = dict(src.metadata)
        if src.structure:
            meta["crawl_structure"] = src.structure
        return NormalizedDocument(
            doc_id=src.doc_id,
            text=text,
            modal_elements=src.modal_elements,
            metadata=meta,
        )


def _is_weather_query(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    return bool(re.search(r"(天气|气温|下雨|降雨|温度|风力|湿度|空气质量|体感)", q))
