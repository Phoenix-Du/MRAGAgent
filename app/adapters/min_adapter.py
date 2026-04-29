from __future__ import annotations

import time
from uuid import uuid4

from app.core.runtime_flags import add_runtime_flag
from app.core.retry import with_retry
from app.core.runtime_flags import get_runtime_flags
from app.core.settings import settings
from app.models.schemas import (
    NormalizedDocument,
    NormalizedPayload,
    QueryRequest,
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

    async def normalize_input(self, payload: QueryRequest) -> NormalizedPayload:
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
            return []

        tags = {"uid": normalized.uid, "intent": normalized.intent}

        async def _ingest() -> list[str]:
            return await self.rag_client.ingest_documents(normalized.documents, tags)

        return await with_retry(_ingest)

    async def query_with_context(self, normalized: NormalizedPayload) -> QueryResponse:
        start_ms = time.perf_counter()
        trace_id = f"tr_{uuid4().hex[:10]}"

        context = await self.memory_client.get_context(normalized.uid)
        pref_hint = context.get("preferences", {}).get("answer_style", "")

        enhanced_query = normalized.query
        if pref_hint:
            enhanced_query = f"{normalized.query}\n[用户偏好回答风格]: {pref_hint}"

        if normalized.intent == "image_search":
            result = await build_image_search_vlm_response(
                query=(normalized.original_query or normalized.query),
                documents=normalized.documents,
                max_images=normalized.max_images,
                image_constraints=normalized.image_constraints,
                trace_id=trace_id,
            )
        else:

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

