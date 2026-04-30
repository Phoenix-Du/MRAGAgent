from __future__ import annotations

import asyncio
from hashlib import md5

from app.core.runtime_flags import add_runtime_flag
from app.core.settings import settings
from app.core.url_safety import is_safe_public_http_url
from app.models.schemas import ModalElement, QueryRequest, SourceDoc
from app.services.connectors import BGERerankClient, CrawlClient, ImagePipelineClient, SearchClient
from app.services.query_optimizer import optimize_web_query


class TaskDispatcher:
    def __init__(
        self,
        search_client: SearchClient,
        crawl_client: CrawlClient,
        bge_rerank_client: BGERerankClient,
        image_pipeline: ImagePipelineClient,
    ) -> None:
        self.search_client = search_client
        self.crawl_client = crawl_client
        self.bge_rerank_client = bge_rerank_client
        self.image_pipeline = image_pipeline

    async def prepare_documents(self, req: QueryRequest) -> tuple[list[SourceDoc], list[ModalElement]]:
        if req.intent == "image_search":
            return await self._image_search_branch(req)
        return await self._general_qa_branch(req)

    async def _general_qa_branch(self, req: QueryRequest) -> tuple[list[SourceDoc], list[ModalElement]]:
        if req.source_docs:
            return req.source_docs[: req.max_web_docs], []

        if req.url:
            crawled = [await self.crawl_client.crawl(req.url)]
            return crawled[: req.max_web_docs], []

        # n candidates from search engine (url + title/snippet summary)
        n_candidates = req.max_web_candidates or settings.web_search_candidates_n
        n_candidates = max(req.max_web_docs, n_candidates)
        optimized_web_query = optimize_web_query(req.query)
        search_hits = await self.search_client.search_web_hits(optimized_web_query, top_k=n_candidates)

        # m selected by BGE over search snippets (not full page body)
        m_limit = req.max_web_docs if req.max_web_docs > 0 else settings.web_url_select_m
        m_selected = max(1, m_limit)
        selected_hits = await self.bge_rerank_client.rerank(optimized_web_query, search_hits, top_k=m_selected)

        selected_urls: list[str] = []
        seen_urls: set[str] = set()
        for hit in selected_hits:
            url = str((hit.metadata or {}).get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            if not is_safe_public_http_url(url):
                add_runtime_flag("unsafe_crawl_url_skipped")
                continue
            if url:
                seen_urls.add(url)
                selected_urls.append(url)

        crawled = await self._crawl_urls(selected_urls)

        if settings.general_qa_body_rerank_enabled and len(crawled) > 1:
            reranked_body = await self.bge_rerank_client.rerank(
                optimized_web_query,
                crawled,
                top_k=req.max_web_docs,
            )
            if reranked_body:
                crawled = reranked_body
                add_runtime_flag("general_qa_body_rerank")

        return crawled[: req.max_web_docs], []

    async def _image_search_branch(self, req: QueryRequest) -> tuple[list[SourceDoc], list[ModalElement]]:
        images = req.images
        image_query = (req.image_search_query or "").strip() or req.query
        if not images:
            images = await self.image_pipeline.search_and_rank_images(image_query, top_k=req.max_images)

        doc = SourceDoc(
            doc_id=f"image_branch::{md5(image_query.encode('utf-8')).hexdigest()[:10]}",
            text_content=req.original_query or req.query,
            modal_elements=images,
            structure={"type": "image_search_result"},
            metadata={"source": "image_pipeline", "image_search_query": image_query},
        )
        return [doc], images

    async def _crawl_urls(self, urls: list[str]) -> list[SourceDoc]:
        if not urls:
            return []
        concurrency = max(1, min(settings.web_crawl_concurrency, len(urls)))
        semaphore = asyncio.Semaphore(concurrency)

        async def _crawl_one(url: str) -> SourceDoc:
            async with semaphore:
                return await self.crawl_client.crawl(url)

        return list(await asyncio.gather(*(_crawl_one(url) for url in urls)))
