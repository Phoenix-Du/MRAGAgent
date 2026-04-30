from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.metrics import metrics
from app.core.runtime_flags import add_runtime_flag
from app.core.settings import settings
from app.models.schemas import EvidenceItem, ImageItem, NormalizedDocument, QueryResponse

logger = logging.getLogger(__name__)


def _local_fallback_answer() -> str:
    return (
        "\u8fd9\u662f\u672c\u5730\u964d\u7ea7\u56de\u7b54\uff1a"
        "\u5df2\u5b8c\u6210\u4e0a\u4e0b\u6587\u68c0\u7d22\uff0c"
        "\u4f46\u5f53\u524d\u6ca1\u6709\u8fde\u63a5\u771f\u5b9e RAG \u751f\u6210\u670d\u52a1\u3002"
        "\u63a5\u5165 RAG Anything \u540e\u53ef\u8fd4\u56de\u771f\u5b9e\u591a\u6a21\u6001\u7b54\u6848\u3002"
    )


class RagClient:
    def __init__(self) -> None:
        self._store: dict[str, NormalizedDocument] = {}

    async def ingest_documents(
        self,
        documents: list[NormalizedDocument],
        tags: dict[str, Any],
    ) -> list[str]:
        if settings.rag_anything_endpoint:
            try:
                timeout = httpx.Timeout(
                    connect=float(settings.request_timeout_seconds),
                    read=float(settings.rag_ingest_timeout_seconds),
                    write=float(settings.request_timeout_seconds),
                    pool=float(settings.request_timeout_seconds),
                )
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    resp = await client.post(
                        f"{settings.rag_anything_endpoint.rstrip('/')}/ingest",
                        json={"documents": [doc.model_dump() for doc in documents], "tags": tags},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return self._map_ingest_response(data, documents)
            except (httpx.HTTPError, ValueError, TypeError):
                add_runtime_flag("rag_ingest_fallback")
                logger.warning("rag_ingest_fallback reason=remote_ingest_failed")
                metrics.inc("rag_ingest_fallback_total")

        indexed_ids: list[str] = []
        for doc in documents:
            key = f"{tags.get('uid', 'unknown')}::{doc.doc_id}"
            if key in self._store:
                self._store.pop(key)
            self._store[key] = doc
            indexed_ids.append(doc.doc_id)
        self._trim_store()
        return indexed_ids

    async def query(
        self,
        query: str,
        uid: str,
        trace_id: str,
    ) -> QueryResponse:
        if settings.rag_anything_endpoint:
            try:
                timeout = httpx.Timeout(
                    connect=float(settings.request_timeout_seconds),
                    read=float(settings.rag_query_timeout_seconds),
                    write=float(settings.request_timeout_seconds),
                    pool=float(settings.request_timeout_seconds),
                )
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    resp = await client.post(
                        f"{settings.rag_anything_endpoint.rstrip('/')}/query",
                        json={"query": query, "uid": uid, "trace_id": trace_id},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return self._map_query_response(data=data, trace_id=trace_id)
            except (httpx.HTTPError, ValueError, TypeError):
                add_runtime_flag("rag_query_fallback")
                logger.warning("rag_query_fallback reason=remote_query_failed trace_id=%s uid=%s", trace_id, uid)
                metrics.inc("rag_query_fallback_total")

        if not settings.allow_placeholder_fallback:
            add_runtime_flag("rag_unavailable")
            raise RuntimeError("RAG service unavailable and placeholder fallback is disabled")

        candidates = [
            doc
            for key, doc in self._store.items()
            if key.startswith(f"{uid}::")
        ][:3]

        evidence = [
            EvidenceItem(
                doc_id=doc.doc_id,
                score=0.8 - idx * 0.1,
                snippet=(doc.text[:120] if doc.text else "No text content."),
            )
            for idx, doc in enumerate(candidates)
        ]

        images = []
        for doc in candidates:
            for modal in doc.modal_elements:
                if modal.type == "image" and modal.url:
                    images.append(ImageItem(url=modal.url, desc=modal.desc))

        answer = (
            "这是本地降级回答：已完成上下文检索，但当前没有连接真实 RAG 生成服务。"
            "接入 RAG Anything 后可返回真实多模态答案。"
        )
        answer = _local_fallback_answer()
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            images=images[:5],
            trace_id=trace_id,
            latency_ms=0,
        )

    @staticmethod
    def _map_ingest_response(data: dict[str, Any], docs: list[NormalizedDocument]) -> list[str]:
        indexed = data.get("indexed_doc_ids")
        if isinstance(indexed, list):
            return [str(doc_id) for doc_id in indexed]

        ids = data.get("doc_ids")
        if isinstance(ids, list):
            return [str(doc_id) for doc_id in ids]

        return [doc.doc_id for doc in docs]

    @staticmethod
    def _map_query_response(data: dict[str, Any], trace_id: str) -> QueryResponse:
        if {"answer", "evidence", "images"}.issubset(data.keys()):
            payload = {
                "answer": data.get("answer") or "",
                "evidence": data.get("evidence") or [],
                "images": data.get("images") or [],
                "trace_id": data.get("trace_id") or trace_id,
                "latency_ms": int(data.get("latency_ms") or 0),
            }
            return QueryResponse(**payload)

        answer = data.get("answer") or data.get("response") or data.get("text") or ""
        if not answer:
            answer = "RAG 服务已返回结果，但未提供标准 answer 字段。"

        raw_evidence = data.get("evidence") or data.get("sources") or []
        evidence: list[EvidenceItem] = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                evidence.append(
                    EvidenceItem(
                        doc_id=str(item.get("doc_id") or item.get("id") or "unknown"),
                        score=float(item.get("score") or item.get("relevance") or 0.0),
                        snippet=str(item.get("snippet") or item.get("text") or ""),
                    )
                )

        raw_images = data.get("images") or data.get("modal_elements") or []
        images: list[ImageItem] = []
        if isinstance(raw_images, list):
            for item in raw_images:
                if not isinstance(item, dict):
                    continue
                if item.get("type") not in {None, "image"}:
                    continue
                url = item.get("url")
                if not url:
                    continue
                images.append(
                    ImageItem(
                        url=str(url),
                        desc=item.get("desc"),
                        local_path=(
                            str(item.get("local_path"))
                            if item.get("local_path") is not None
                            else None
                        ),
                    )
                )

        return QueryResponse(
            answer=str(answer),
            evidence=evidence,
            images=images,
            trace_id=str(data.get("trace_id") or trace_id),
            latency_ms=int(data.get("latency_ms") or 0),
        )

    def _trim_store(self) -> None:
        max_docs = max(1, int(settings.local_rag_store_max_docs))
        while len(self._store) > max_docs:
            oldest_key = next(iter(self._store))
            self._store.pop(oldest_key, None)
