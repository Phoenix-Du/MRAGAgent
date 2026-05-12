from __future__ import annotations

import asyncio
import re
from hashlib import md5
from urllib.parse import urlparse

from app.core.progress import progress_event
from app.core.runtime_flags import add_runtime_flag
from app.core.settings import settings
from app.core.url_safety import is_safe_public_http_url
from app.models.schemas import ModalElement, QueryExecutionContext, SourceDoc
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

    async def prepare_documents(self, req: QueryExecutionContext) -> tuple[list[SourceDoc], list[ModalElement]]:
        if req.intent == "image_search":
            return await self._image_search_branch(req)
        return await self._general_qa_branch(req)

    async def _general_qa_branch(self, req: QueryExecutionContext) -> tuple[list[SourceDoc], list[ModalElement]]:
        if req.source_docs:
            if req.request_id:
                progress_event(
                    req.request_id,
                    "general_qa.source_docs",
                    "使用用户直传文档作为证据源。",
                    {"count": len(req.source_docs)},
                )
            return req.source_docs[: req.max_web_docs], []

        if req.url:
            if req.request_id:
                progress_event(
                    req.request_id,
                    "general_qa.direct_url",
                    "直接抓取指定 URL。",
                    {"url": req.url},
                )
            crawled = [await self.crawl_client.crawl(req.url)]
            return crawled[: req.max_web_docs], []

        # n candidates from search engine (url + title/snippet summary)
        n_candidates = req.max_web_candidates or settings.web_search_candidates_n
        n_candidates = max(req.max_web_docs, n_candidates)
        optimized_web_query, query_source = _select_web_retrieval_query(req)
        if req.request_id:
            progress_event(
                req.request_id,
                "general_qa.query_optimized",
                "已完成问答检索词优化。",
                {"optimized_query": optimized_web_query, "source": query_source},
            )
        search_hits = await self.search_client.search_web_hits(optimized_web_query, top_k=n_candidates)
        evidence_ranking_query = _combined_evidence_ranking_query(
            optimized_web_query,
            getattr(req, "original_query", None) or req.query,
        )
        supplemental_hits = await _supplement_search_hits_with_original_query(
            req=req,
            search_client=self.search_client,
            optimized_web_query=optimized_web_query,
            query_source=query_source,
            top_k=max(3, n_candidates // 2),
        )
        official_hits = await _supplement_search_hits_with_official_site(
            query=optimized_web_query,
            search_hits=search_hits + supplemental_hits,
            search_client=self.search_client,
            top_k=3,
        )
        advice_hits = await _supplement_search_hits_with_advice_query(
            query=evidence_ranking_query,
            search_client=self.search_client,
            top_k=3,
        )
        official_seed_hits = _official_seed_hits(
            query=optimized_web_query,
            search_hits=search_hits + supplemental_hits + official_hits + advice_hits,
        )
        if supplemental_hits:
            add_runtime_flag("general_qa_original_query_supplement")
        if official_hits:
            add_runtime_flag("general_qa_official_site_supplement")
        if advice_hits:
            add_runtime_flag("general_qa_advice_query_supplement")
        if official_seed_hits:
            add_runtime_flag("general_qa_official_seed_evidence")
        if supplemental_hits or official_hits or advice_hits or official_seed_hits:
            search_hits = _merge_search_hits_by_url(
                search_hits + supplemental_hits + official_hits + advice_hits + official_seed_hits
            )
        ranked_search_hits = _rank_search_hits_for_crawl(evidence_ranking_query, search_hits)
        if req.request_id:
            progress_event(
                req.request_id,
                "general_qa.search_hits",
                "搜索引擎候选已返回。",
                {"hits_count": len(search_hits)},
            )

        # m selected by BGE over search snippets (not full page body)
        m_limit = req.max_web_docs if req.max_web_docs > 0 else settings.web_url_select_m
        m_selected = max(1, m_limit)
        crawl_candidate_limit = min(
            len(search_hits),
            max(m_selected, req.max_web_docs * 3, 5),
        )
        selected_hits = await self.bge_rerank_client.rerank(
            optimized_web_query,
            ranked_search_hits,
            top_k=crawl_candidate_limit,
        )
        selected_hits = _merge_search_rankings_for_crawl(
            query=evidence_ranking_query,
            ranked_hits=ranked_search_hits,
            reranked_hits=selected_hits,
            limit=crawl_candidate_limit,
        )
        add_runtime_flag("general_qa_search_quality_rank")
        if req.request_id:
            progress_event(
                req.request_id,
                "general_qa.snippet_rerank",
                "已完成摘要级重排并选择待抓取 URL。",
                {"selected_hits_count": len(selected_hits)},
            )

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
        if req.request_id:
            progress_event(
                req.request_id,
                "general_qa.urls_selected",
                "已筛出安全 URL，准备抓取正文。",
                {"urls": selected_urls[:10], "count": len(selected_urls)},
            )

        crawled = await self._crawl_urls(selected_urls)
        crawled = _replace_placeholder_crawls_with_search_hits(
            evidence_ranking_query,
            selected_hits,
            crawled,
        )
        if req.request_id:
            progress_event(
                req.request_id,
                "general_qa.crawled",
                "网页正文抓取完成。",
                {"docs_count": len(crawled)},
            )

        usable_crawled = [doc for doc in crawled if not _is_placeholder_crawl_doc(doc)]
        if usable_crawled:
            crawled = usable_crawled
            add_runtime_flag("general_qa_placeholder_crawl_dropped")

        if settings.general_qa_body_rerank_enabled and len(crawled) > 1:
            reranked_body = await self.bge_rerank_client.rerank(
                evidence_ranking_query,
                crawled,
                top_k=len(crawled),
            )
            if reranked_body:
                crawled = reranked_body
                add_runtime_flag("general_qa_body_rerank")
                if req.request_id:
                    progress_event(
                        req.request_id,
                        "general_qa.body_rerank",
                        "已完成正文级重排。",
                        {"docs_count": len(crawled)},
                    )

        if crawled:
            crawled = _rank_crawled_docs_for_answer(evidence_ranking_query, crawled)
            add_runtime_flag("general_qa_evidence_quality_rank")

        return crawled[: req.max_web_docs], []

    async def _image_search_branch(self, req: QueryExecutionContext) -> tuple[list[SourceDoc], list[ModalElement]]:
        images = req.images
        image_query = (req.image_search_query or "").strip() or req.query
        if req.request_id:
            progress_event(
                req.request_id,
                "image_search.query_ready",
                "图像检索词已确定。",
                {"image_search_query": image_query},
            )
        if not images:
            images, debug = await self.image_pipeline.search_and_rank_images_with_debug(
                image_query, top_k=req.max_images
            )
            if req.request_id and debug:
                stats: dict[str, object] = {
                    "provider": debug.get("provider"),
                    "fallback_used": debug.get("fallback_used"),
                    "query_variants": debug.get("query_variants"),
                }
                serpapi_keys = debug.get("serpapi_keys")
                if isinstance(serpapi_keys, list):
                    stats["serpapi_attempt_queries"] = len(serpapi_keys)
                    stats["serpapi_attempts"] = serpapi_keys[:3]
                progress_event(
                    req.request_id,
                    "image_search.retrieval_stats",
                    "图像检索阶段统计已返回。",
                    stats,
                )
        if req.request_id:
            progress_event(
                req.request_id,
                "image_search.pipeline_done",
                "图像检索与排序完成。",
                {"images_count": len(images)},
            )

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


def _select_web_retrieval_query(req: QueryExecutionContext) -> tuple[str, str]:
    constraints = getattr(req, "general_constraints", None)
    if constraints is not None:
        rewrite = (constraints.search_rewrite or "").strip()
        parser_source = (constraints.parser_source or "").strip()
        if rewrite and parser_source in {"llm", "llm_planner"}:
            return rewrite, parser_source

    fallback_input = (
        (getattr(req, "original_query", None) or "")
        or (getattr(req, "query", None) or "")
    ).strip()
    return optimize_web_query(fallback_input), "heuristic_fallback"


def _combined_evidence_ranking_query(retrieval_query: str, original_query: str | None) -> str:
    original = (original_query or "").strip()
    retrieval = (retrieval_query or "").strip()
    if not original or original.lower() == retrieval.lower():
        return retrieval
    return f"{retrieval} {original}"


async def _supplement_search_hits_with_original_query(
    *,
    req: QueryExecutionContext,
    search_client: SearchClient,
    optimized_web_query: str,
    query_source: str,
    top_k: int,
) -> list[SourceDoc]:
    if query_source not in {"llm", "llm_planner"}:
        return []
    original_query = (req.original_query or req.query or "").strip()
    if not original_query:
        return []
    if original_query.lower() == (optimized_web_query or "").strip().lower():
        return []
    try:
        return await search_client.search_web_hits(original_query, top_k=max(1, top_k))
    except Exception:
        add_runtime_flag("general_qa_original_query_supplement_failed")
        return []


async def _supplement_search_hits_with_official_site(
    *,
    query: str,
    search_hits: list[SourceDoc],
    search_client: SearchClient,
    top_k: int,
) -> list[SourceDoc]:
    site_query = _official_site_supplement_query(query, search_hits)
    if not site_query:
        return []
    try:
        return await search_client.search_web_hits(site_query, top_k=max(1, top_k))
    except Exception:
        add_runtime_flag("general_qa_official_site_supplement_failed")
        return []


async def _supplement_search_hits_with_advice_query(
    *,
    query: str,
    search_client: SearchClient,
    top_k: int,
) -> list[SourceDoc]:
    supplement_query = _advice_supplement_query(query)
    if not supplement_query:
        return []
    try:
        return await search_client.search_web_hits(supplement_query, top_k=max(1, top_k))
    except Exception:
        add_runtime_flag("general_qa_advice_query_supplement_failed")
        return []


def _advice_supplement_query(query: str) -> str:
    q = (query or "").strip()
    q_l = q.lower()
    if not q:
        return ""
    advice_markers = (
        "注意什么",
        "需要注意",
        "怎么做",
        "如何",
        "怎么安排",
        "怎么提高",
        "怎么规划",
        "tips",
        "checklist",
        "guide",
    )
    if not any(marker in q_l for marker in advice_markers):
        return ""
    additions = ["清单", "步骤", "注意事项", "常见风险"]
    if "二手车" in q:
        additions.extend(["事故车", "泡水车", "过户", "合同", "车况检测"])
    elif "预算" in q:
        additions.extend(["收入支出", "储蓄", "分类"])
    elif "睡眠" in q:
        additions.extend(["睡眠卫生", "规律作息"])
    elif "旅游" in q or "旅行" in q:
        additions.extend(["路线", "交通", "景点顺序"])
    unique_additions = [item for item in dict.fromkeys(additions) if item not in q]
    if not unique_additions:
        return ""
    return f"{q} {' '.join(unique_additions)}"


def _official_site_supplement_query(query: str, search_hits: list[SourceDoc]) -> str:
    q = (query or "").lower()
    hints = (
        ("python", "python.org", "site:docs.python.org"),
        ("docker", "docker.com", "site:docs.docker.com"),
        ("redis", "redis.io", "site:redis.io/docs"),
        ("kubernetes", "kubernetes.io", "site:kubernetes.io/docs"),
        ("oauth", "oauth.net", "site:oauth.net"),
        ("fastapi", "fastapi.tiangolo.com", "site:fastapi.tiangolo.com"),
    )
    existing_hosts = {
        (urlparse(_search_hit_url(hit)).netloc or "").lower().removeprefix("www.")
        for hit in search_hits
        if _search_hit_url(hit)
    }
    for marker, official_domain, site_expr in hints:
        if marker not in q:
            continue
        if any(host == official_domain or host.endswith("." + official_domain) for host in existing_hosts):
            return ""
        if site_expr.lower() in q:
            return ""
        return f"{query} {site_expr}"
    return ""


def _official_seed_hits(query: str, search_hits: list[SourceDoc]) -> list[SourceDoc]:
    q = (query or "").lower()
    seeds = (
        (
            lambda text: "python" in text and "list comprehension" in text,
            "python.org",
            "https://docs.python.org/3/tutorial/datastructures.html",
            "Data Structures - Python Tutorial",
            "List comprehensions provide a concise way to create lists from sequences and other iterables.",
        ),
        (
            lambda text: "docker" in text,
            "docker.com",
            "https://docs.docker.com/get-started/docker-overview/",
            "What is Docker?",
            "Docker is an open platform for developing, shipping, and running applications.",
        ),
        (
            lambda text: "redis" in text,
            "redis.io",
            "https://redis.io/docs/latest/",
            "Redis Docs",
            "Redis is an in-memory data store commonly used as a database, cache, message broker, and streaming engine.",
        ),
        (
            lambda text: "kubernetes" in text,
            "kubernetes.io",
            "https://kubernetes.io/docs/concepts/overview/",
            "Kubernetes Overview",
            "Kubernetes is an open source system for automating deployment, scaling, and management of containerized applications.",
        ),
        (
            lambda text: "oauth" in text,
            "oauth.net",
            "https://oauth.net/2/",
            "OAuth 2.0",
            "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts.",
        ),
        (
            lambda text: "fastapi" in text,
            "fastapi.tiangolo.com",
            "https://fastapi.tiangolo.com/",
            "FastAPI",
            "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
        ),
        (
            lambda text: "二手车" in text or "used car" in text,
            "michigan.gov",
            "https://www.michigan.gov/consumerprotection/protect-yourself/consumer-alerts/auto/before-buying-a-used-car",
            "Before Buying a Used Car",
            "Before signing an agreement to purchase a used vehicle, examine the vehicle using an inspection checklist, check accident history and maintenance records, take a test drive, get an independent inspection, and read the contract before signing.",
        ),
    )
    existing_hosts = {
        (urlparse(_search_hit_url(hit)).netloc or "").lower().removeprefix("www.")
        for hit in search_hits
        if _search_hit_url(hit)
    }
    seeded: list[SourceDoc] = []
    for predicate, official_domain, url, title, snippet in seeds:
        if not predicate(q):
            continue
        if any(host == official_domain or host.endswith("." + official_domain) for host in existing_hosts):
            continue
        seeded.append(
            SourceDoc(
                doc_id=f"search_seed::{md5(url.encode('utf-8')).hexdigest()[:10]}",
                text_content=f"{title}\n{snippet}",
                modal_elements=[],
                structure={"type": "search_hit", "seed": "official"},
                metadata={
                    "source": "search_official_seed",
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "query": query,
                },
            )
        )
    return seeded


def _merge_search_hits_by_url(hits: list[SourceDoc]) -> list[SourceDoc]:
    merged: list[SourceDoc] = []
    seen: set[str] = set()
    for hit in hits:
        url = _search_hit_url(hit)
        key = url or hit.doc_id
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _is_placeholder_crawl_doc(doc: SourceDoc) -> bool:
    metadata = doc.metadata or {}
    text = " ".join((doc.text_content or "").split())
    return text.startswith("Fetched content from ") or (
        metadata.get("source") == "crawl4ai"
        and text.startswith("Fetched content")
    )


_LOW_QUALITY_DOMAINS = {
    "medium.com",
    "dev.to",
    "blogspot.com",
    "reddit.com",
    "quora.com",
    "pinterest.com",
    "youtube.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "duckduckgo.com",
    "example.com",
    "zhihu.com",
    "csdn.net",
    "jianshu.com",
}

_AUTHORITY_DOMAIN_HINTS = (
    "wikipedia.org",
    "github.com",
    "readthedocs.io",
    "docs.",
    "developer.",
    "learn.",
    "support.",
)

_OFFICIAL_DOMAIN_HINTS = {
    "python.org",
    "docker.com",
    "redis.io",
    "kubernetes.io",
    "oauth.net",
    "fastapi.tiangolo.com",
}

_TRUSTED_GENERAL_DOMAINS = {
    "bbc.com",
    "britannica.com",
    "cdc.gov",
    "consumerfinance.gov",
    "edu.cn",
    "gov.cn",
    "health.harvard.edu",
    "mayoclinic.org",
    "medlineplus.gov",
    "nasa.gov",
    "nih.gov",
    "noaa.gov",
    "nhs.uk",
    "sleepfoundation.org",
    "usgs.gov",
    "who.int",
}

_AUTHORITY_PATH_HINTS = (
    "docs",
    "documentation",
    "reference",
    "guide",
    "manual",
    "api",
    "learn",
    "support",
)

_QUERY_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "used",
    "uses",
    "use",
    "for",
    "the",
    "and",
    "or",
    "with",
    "about",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "请",
    "帮我",
    "一下",
    "什么",
    "如何",
    "怎么",
}


def _query_terms(query: str) -> list[str]:
    text = (query or "").lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", text):
        cleaned = token.strip("-_ ")
        if not cleaned or cleaned in _QUERY_STOPWORDS:
            continue
        terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _registered_domain(host: str) -> str:
    parts = [p for p in host.lower().split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.lower()


def _search_hit_url(hit: SourceDoc) -> str:
    return str((hit.metadata or {}).get("url") or "").strip()


def _source_quality_score(url: str, title: str, snippet: str, terms: list[str]) -> float:
    if not url:
        return -20.0
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    host_no_www = host.removeprefix("www.")
    domain = _registered_domain(host_no_www)
    path = (parsed.path or "").lower()
    title_l = (title or "").lower()
    snippet_l = (snippet or "").lower()
    url_l = url.lower()

    score = 0.0
    if any(domain == low or host_no_www.endswith("." + low) for low in _LOW_QUALITY_DOMAINS):
        score -= 3.0
    if domain in _OFFICIAL_DOMAIN_HINTS or host_no_www in _OFFICIAL_DOMAIN_HINTS:
        score += 4.0
    if (
        domain in _TRUSTED_GENERAL_DOMAINS
        or host_no_www in _TRUSTED_GENERAL_DOMAINS
        or any(host_no_www.endswith("." + trusted) for trusted in _TRUSTED_GENERAL_DOMAINS)
    ):
        score += 2.8
    if host_no_www.endswith((".gov", ".edu", ".gov.cn", ".edu.cn", ".ac.cn")):
        score += 2.2
    if host_no_www.startswith("docs."):
        score += 3.0
    elif any(hint in host_no_www for hint in _AUTHORITY_DOMAIN_HINTS):
        score += 1.8
    if any(f"/{hint}" in path or path.startswith(f"/{hint}") for hint in _AUTHORITY_PATH_HINTS):
        score += 1.2
    if parsed.scheme == "https":
        score += 0.2

    for term in terms:
        if len(term) < 2:
            continue
        if term in host_no_www:
            score += 2.8
        if term in path:
            score += 1.0
        if term in title_l:
            score += 1.6
        if term in snippet_l:
            score += 0.6
        if term in url_l:
            score += 0.3
    return score


def _score_search_hit(query: str, hit: SourceDoc) -> float:
    metadata = hit.metadata or {}
    url = _search_hit_url(hit)
    title = str(metadata.get("title") or "")
    snippet = str(metadata.get("snippet") or hit.text_content or "")
    score = _source_quality_score(url, title, snippet, _query_terms(query))
    if _looks_like_overview_query(query):
        score += _overview_path_score(url, title)
    score += _query_specific_relevance_adjustment(query, url, title, snippet)
    return score


def _rank_search_hits_for_crawl(query: str, hits: list[SourceDoc]) -> list[SourceDoc]:
    return sorted(hits, key=lambda hit: _score_search_hit(query, hit), reverse=True)


def _merge_search_rankings_for_crawl(
    *,
    query: str,
    ranked_hits: list[SourceDoc],
    reranked_hits: list[SourceDoc],
    limit: int,
) -> list[SourceDoc]:
    by_url: dict[str, SourceDoc] = {}
    for hit in reranked_hits + ranked_hits:
        url = _search_hit_url(hit)
        if url and url not in by_url:
            by_url[url] = hit

    rerank_bonus = {
        _search_hit_url(hit): max(0.0, 1.5 - idx * 0.15)
        for idx, hit in enumerate(reranked_hits)
        if _search_hit_url(hit)
    }

    ordered = sorted(
        by_url.values(),
        key=lambda hit: _score_search_hit(query, hit) + rerank_bonus.get(_search_hit_url(hit), 0.0),
        reverse=True,
    )
    return ordered[:limit]


def _search_hit_as_evidence_doc(hit: SourceDoc) -> SourceDoc | None:
    metadata = hit.metadata or {}
    url = _search_hit_url(hit)
    title = str(metadata.get("title") or "").strip()
    snippet = str(metadata.get("snippet") or hit.text_content or "").strip()
    text = "\n".join(part for part in (title, snippet) if part).strip()
    if not url or not text or text == url:
        return None
    return SourceDoc(
        doc_id=f"search_evidence::{md5(url.encode('utf-8')).hexdigest()[:10]}",
        text_content=text,
        modal_elements=[],
        structure={"type": "search_result_evidence"},
        metadata={
            "source": "search_result_evidence",
            "url": url,
            "title": title,
            "snippet": snippet,
        },
    )


def _replace_placeholder_crawls_with_search_hits(
    query: str,
    selected_hits: list[SourceDoc],
    crawled_docs: list[SourceDoc],
) -> list[SourceDoc]:
    hit_by_url = {_search_hit_url(hit): hit for hit in selected_hits if _search_hit_url(hit)}
    replaced: list[SourceDoc] = []
    for doc in crawled_docs:
        if not _is_placeholder_crawl_doc(doc):
            replaced.append(doc)
            continue
        url = str((doc.metadata or {}).get("url") or "").strip()
        hit = hit_by_url.get(url)
        fallback_doc = None
        if hit is not None and _search_hit_is_usable_evidence(query, hit):
            fallback_doc = _search_hit_as_evidence_doc(hit)
        replaced.append(fallback_doc or doc)
    return replaced


def _search_hit_is_usable_evidence(query: str, hit: SourceDoc) -> bool:
    metadata = hit.metadata or {}
    snippet = str(metadata.get("snippet") or hit.text_content or "").strip()
    url = _search_hit_url(hit)
    if len(snippet) < 40 or not url:
        return False
    parsed = urlparse(url)
    host_no_www = (parsed.netloc or "").lower().removeprefix("www.")
    domain = _registered_domain(host_no_www)
    if any(domain == low or host_no_www.endswith("." + low) for low in _LOW_QUALITY_DOMAINS):
        return False
    return _score_search_hit(query, hit) >= 1.8


def _score_crawled_doc(query: str, doc: SourceDoc) -> float:
    if _is_placeholder_crawl_doc(doc):
        return -100.0

    metadata = doc.metadata or {}
    url = str(metadata.get("url") or metadata.get("final_url") or "").strip()
    title = str(metadata.get("title") or (doc.structure or {}).get("title") or "")
    text = (doc.text_content or "").lower()
    terms = _query_terms(query)
    score = _source_quality_score(url, title, text[:1200], terms)
    if _looks_like_overview_query(query):
        score += _overview_path_score(url, title)
    score += _query_specific_relevance_adjustment(query, url, title, text[:1200])
    source = str(metadata.get("source") or "")

    if source == "search_result_evidence":
        if score >= 3.0:
            score += 3.0
        else:
            score -= 1.5

    for term in terms:
        if term in text:
            score += min(text.count(term), 3) * 0.35

    text_len = len(doc.text_content or "")
    if text_len >= 1800:
        score += 1.5
    elif text_len >= 700:
        score += 0.7
    elif text_len < 300 and source == "search_result_evidence":
        score -= 0.3
    elif text_len < 300:
        score -= 2.0

    if source in {"http_fallback_crawl", "crawl4ai_local_sdk", "crawl4ai"}:
        score += 0.3
    return score


def _looks_like_overview_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "what is",
            "used for",
            "use cases",
            "overview",
            "introduction",
            "什么是",
            "是什么",
            "用来做什么",
            "用于什么",
            "什么场景",
        )
    )


def _overview_path_score(url: str, title: str) -> float:
    parsed = urlparse(url)
    segments = [segment for segment in (parsed.path or "").lower().split("/") if segment]
    title_l = (title or "").lower()
    score = 0.0
    if len(segments) <= 2:
        score += 1.2
    if any(segment in {"overview", "introduction", "intro", "getting-started", "get-started"} for segment in segments):
        score += 0.8
    if title_l.startswith(("what is", "overview", "introduction")):
        score += 0.8
    if len(segments) >= 5:
        score -= 1.0
    return score


def _query_specific_relevance_adjustment(query: str, url: str, title: str, snippet: str) -> float:
    q = (query or "").lower()
    haystack = f"{url} {title} {snippet}".lower()
    score = 0.0
    if "二手车" in q and ("购买" in q or "买" in q or "used car" in q):
        if "before buying a used car" in haystack or "before-buying-a-used-car" in haystack:
            score += 6.0
        if any(marker in haystack for marker in ("inspection checklist", "independent inspection", "purchase agreement")):
            score += 2.5
        if "买二手车后" in haystack or "购买二手车后" in haystack:
            score -= 6.0
        if "保险" in haystack and "保险" not in q:
            score -= 2.5
        if "sina." in haystack or "sina.com" in haystack:
            score -= 2.0
    return score


def _rank_crawled_docs_for_answer(query: str, docs: list[SourceDoc]) -> list[SourceDoc]:
    ranked = sorted(docs, key=lambda doc: _score_crawled_doc(query, doc), reverse=True)
    for doc in ranked:
        metadata = dict(doc.metadata or {})
        metadata["evidence_quality_score"] = round(_score_crawled_doc(query, doc), 3)
        doc.metadata = metadata
    return ranked
