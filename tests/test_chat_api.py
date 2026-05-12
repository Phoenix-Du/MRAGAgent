from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthz() -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_query_returns_runtime_flags(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import NormalizedDocument, NormalizedPayload, QueryResponse

    async def fake_normalize_input(payload):
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent or "general_qa",
            query=payload.query,
            documents=[
                NormalizedDocument(
                    doc_id="doc-1",
                    text="test document",
                    modal_elements=[],
                    metadata={"source": "test"},
                )
            ],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-1"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="ok",
            evidence=[],
            images=[],
            trace_id="tr_test_001",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test",
            "intent": "general_qa",
            "query": "hello",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    assert data["route"] == "general_qa"
    assert "runtime_flags" in data


def test_query_request_rejects_out_of_range_limits() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-validation",
            "intent": "general_qa",
            "query": "hello",
            "use_rasa_intent": False,
            "max_images": 99,
        },
    )
    assert resp.status_code == 422


def test_query_request_rejects_non_http_url() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-url-validation",
            "intent": "general_qa",
            "query": "hello",
            "url": "file:///etc/passwd",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 422


def test_query_request_rejects_internal_execution_fields() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-internal-fields",
            "intent": "general_qa",
            "query": "hello",
            "image_search_query": "internal should not be accepted",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 422


def test_query_request_rejects_localhost_url() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-url-localhost",
            "intent": "general_qa",
            "query": "hello",
            "url": "http://localhost/private",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 422


def test_clean_chat_heuristics_support_real_chinese() -> None:
    from app.api import chat

    assert chat._heuristic_image_search_intent("给我几张城市夜景照片")
    assert chat._heuristic_general_qa_intent("北京明天天气怎么样")
    assert chat._parse_count("给我三张图片") == 3


def test_image_proxy_rejects_loopback_url() -> None:
    client = TestClient(app)
    resp = client.get("/v1/chat/image-proxy", params={"url": "http://127.0.0.1/test.png"})
    assert resp.status_code == 400


def test_image_proxy_rejects_redirect_to_blocked_host(monkeypatch) -> None:
    from app.api import chat

    async def fake_blocked_host(hostname):
        return hostname == "127.0.0.1"

    class FakeResponse:
        status_code = 302
        headers = {"location": "http://127.0.0.1/private.png"}
        content = b""

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr(chat, "_is_blocked_image_proxy_host", fake_blocked_host)
    monkeypatch.setattr(chat.httpx, "AsyncClient", FakeAsyncClient)

    client = TestClient(app)
    resp = client.get(
        "/v1/chat/image-proxy",
        params={"url": "https://public.example/image.png"},
    )
    assert resp.status_code == 400


def test_image_query_parser_extracts_real_chinese_spatial_constraints() -> None:
    from app.services.image_query_parser import parse_image_search_constraints
    from app.services import image_query_parser
    import asyncio

    old_openai = image_query_parser.bridge_settings.openai_api_key
    old_qwen = image_query_parser.bridge_settings.qwen_api_key
    try:
        image_query_parser.bridge_settings.openai_api_key = None
        image_query_parser.bridge_settings.qwen_api_key = None
        result = asyncio.run(
            parse_image_search_constraints(
                "给我几张左边是边牧右边是金毛的照片",
                {},
            )
        )
    finally:
        image_query_parser.bridge_settings.openai_api_key = old_openai
        image_query_parser.bridge_settings.qwen_api_key = old_qwen

    assert result.count == 5
    assert "边牧" in result.entities["subjects"]
    assert result.relations[0].type == "spatial"
    assert result.spatial_relations
    rel = result.spatial_relations[0]
    assert rel.primary_subject == "边牧"
    assert rel.secondary_subject == "金毛"
    assert "边牧" in (result.search_rewrite or "")
    assert "金毛" in (result.search_rewrite or "")


def test_parser_cache_is_bounded(monkeypatch) -> None:
    from app.services.parser_cache import ParserCache

    cache = ParserCache[object](ttl_seconds=60, max_entries=2)

    cache.put("a", object(), 1.0)
    cache.put("b", object(), 2.0)
    cache.put("c", object(), 3.0)

    assert cache.keys() == {"b", "c"}


def test_parser_cache_expires_stale_entries() -> None:
    from app.services.parser_cache import ParserCache

    cache = ParserCache[str](ttl_seconds=5, max_entries=10)
    cache.put("a", "value", 10.0)

    assert cache.get("a", 14.0) == "value"
    assert cache.get("a", 16.0) is None


def test_rag_client_local_store_is_bounded(monkeypatch) -> None:
    from app.models.schemas import NormalizedDocument
    from app.services import rag_client
    import asyncio

    monkeypatch.setattr(rag_client.settings, "rag_anything_endpoint", None)
    monkeypatch.setattr(rag_client.settings, "local_rag_store_max_docs", 2)

    client = rag_client.RagClient()
    docs = [
        NormalizedDocument(doc_id=f"doc-{idx}", text=f"text {idx}")
        for idx in range(3)
    ]
    asyncio.run(client.ingest_documents(docs, {"uid": "u-bounded"}))

    assert list(client._store) == ["u-bounded::doc-1", "u-bounded::doc-2"]


def test_dispatcher_deduplicates_selected_urls() -> None:
    from app.models.schemas import QueryRequest, SourceDoc
    from app.services.dispatcher import TaskDispatcher
    import asyncio

    search_hits = [
        SourceDoc(doc_id="hit-1", text_content="a", metadata={"url": "https://example.com/a"}),
        SourceDoc(doc_id="hit-2", text_content="a2", metadata={"url": "https://example.com/a"}),
        SourceDoc(doc_id="hit-3", text_content="b", metadata={"url": "https://example.com/b"}),
    ]
    crawled_urls = []

    class FakeSearch:
        async def search_web_hits(self, _query, top_k):
            return search_hits[:top_k]

    class FakeRerank:
        async def rerank(self, _query, docs, top_k):
            return docs[:top_k]

    class FakeCrawl:
        async def crawl(self, url):
            crawled_urls.append(url)
            return SourceDoc(doc_id=url, text_content=url, metadata={"url": url})

    class FakeImagePipeline:
        pass

    dispatcher = TaskDispatcher(FakeSearch(), FakeCrawl(), FakeRerank(), FakeImagePipeline())
    req = QueryRequest(
        uid="u-dispatcher",
        intent="general_qa",
        query="hello",
        max_web_docs=3,
        use_rasa_intent=False,
    )
    asyncio.run(dispatcher.prepare_documents(req))

    assert crawled_urls == ["https://example.com/a", "https://example.com/b"]


def test_dispatcher_prefers_real_crawl_docs_over_placeholders() -> None:
    from app.models.schemas import QueryExecutionContext, SourceDoc
    from app.services.dispatcher import TaskDispatcher
    import asyncio

    class FakeSearch:
        async def search_web_hits(self, _query, top_k):
            return [
                SourceDoc(
                    doc_id=f"hit-{idx}",
                    text_content=f"title {idx}",
                    metadata={"url": f"https://example.com/{idx}"},
                )
                for idx in range(1, 4)
            ][:top_k]

    class FakeRerank:
        async def rerank(self, _query, docs, top_k):
            return docs[:top_k]

    class FakeCrawl:
        async def crawl(self, url):
            if url.endswith("/1"):
                return SourceDoc(
                    doc_id="placeholder",
                    text_content=f"Fetched content from {url}",
                    metadata={"source": "crawl4ai", "url": url},
                )
            return SourceDoc(
                doc_id=f"real-{url.rsplit('/', 1)[-1]}",
                text_content=f"real body from {url}",
                metadata={"source": "http_fallback_crawl", "url": url},
            )

    dispatcher = TaskDispatcher(
        search_client=FakeSearch(),
        crawl_client=FakeCrawl(),
        bge_rerank_client=FakeRerank(),
        image_pipeline=None,
    )

    docs, _images = asyncio.run(
        dispatcher.prepare_documents(
                QueryExecutionContext(
                    uid="u-dispatch-real-docs",
                    intent="general_qa",
                    query="hello",
                    original_query="hello",
                    max_web_docs=1,
                    max_web_candidates=3,
                )
        )
    )

    assert len(docs) == 1
    assert docs[0].doc_id == "real-2"
    assert docs[0].text_content.startswith("real body")


def test_search_quality_rank_prefers_official_docs_over_blog() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_search_hits_for_crawl

    hits = [
        SourceDoc(
            doc_id="medium",
            text_content="FastAPI use cases",
            metadata={
                "url": "https://medium.com/@someone/fastapi-use-cases",
                "title": "FastAPI use cases",
                "snippet": "A personal blog post.",
            },
        ),
        SourceDoc(
            doc_id="official",
            text_content="FastAPI framework documentation",
            metadata={
                "url": "https://fastapi.tiangolo.com/",
                "title": "FastAPI",
                "snippet": "FastAPI framework, high performance, ready for production.",
            },
        ),
    ]

    ranked = _rank_search_hits_for_crawl("What is FastAPI used for?", hits)

    assert ranked[0].doc_id == "official"


def test_search_quality_rank_prefers_language_official_docs() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_search_hits_for_crawl

    hits = [
        SourceDoc(
            doc_id="tutorial",
            text_content="Python list comprehension guide",
            metadata={
                "url": "https://mimo.org/glossary/python/list-comprehension",
                "title": "Python List Comprehension",
                "snippet": "List comprehension is an easy way to create a list.",
            },
        ),
        SourceDoc(
            doc_id="python-docs",
            text_content="Data Structures - Python Tutorial",
            metadata={
                "url": "https://docs.python.org/3/tutorial/datastructures.html",
                "title": "Data Structures - Python Tutorial",
                "snippet": "List comprehensions provide a concise way to create lists.",
            },
        ),
    ]

    ranked = _rank_search_hits_for_crawl("What is Python list comprehension used for?", hits)

    assert ranked[0].doc_id == "python-docs"


def test_dispatcher_supplements_llm_rewrite_with_original_query() -> None:
    from app.models.schemas import GeneralQueryConstraints, QueryExecutionContext, SourceDoc
    from app.services.dispatcher import TaskDispatcher
    import asyncio

    search_queries: list[str] = []

    class FakeSearch:
        async def search_web_hits(self, query, top_k):
            search_queries.append(query)
            if query == "redis time series use cases":
                return [
                    SourceDoc(
                        doc_id="narrow",
                        text_content="Redis time series use case",
                        metadata={
                            "url": "https://redis.io/docs/latest/develop/data-types/timeseries/use_cases/",
                            "title": "Use cases | Docs",
                            "snippet": "Redis Time Series use cases.",
                        },
                    )
                ]
            return [
                SourceDoc(
                    doc_id="overview",
                    text_content="Redis overview",
                    metadata={
                        "url": "https://redis.io/docs/latest/",
                        "title": "Redis Docs",
                        "snippet": "Redis is an in-memory data store used as a database, cache, and message broker.",
                    },
                )
            ]

    class FakeRerank:
        async def rerank(self, _query, docs, top_k):
            return docs[:top_k]

    class FakeCrawl:
        async def crawl(self, url):
            return SourceDoc(
                doc_id=url,
                text_content="Redis is an in-memory data store used as a database, cache, and message broker.",
                metadata={"source": "http_fallback_crawl", "url": url, "title": "Redis Docs"},
            )

    dispatcher = TaskDispatcher(FakeSearch(), FakeCrawl(), FakeRerank(), None)
    req = QueryExecutionContext(
        uid="u-redis-quality",
        intent="general_qa",
        query="redis time series use cases",
        original_query="What is Redis used for?",
        max_web_docs=2,
        max_web_candidates=4,
        general_constraints=GeneralQueryConstraints(
            raw_query="What is Redis used for?",
            search_rewrite="redis time series use cases",
            parser_source="llm_planner",
        ),
    )

    docs, _ = asyncio.run(dispatcher.prepare_documents(req))

    assert search_queries == ["redis time series use cases", "What is Redis used for?"]
    assert any(doc.metadata["url"] == "https://redis.io/docs/latest/" for doc in docs)


def test_official_site_supplement_query_when_official_missing() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _official_site_supplement_query

    hits = [
        SourceDoc(
            doc_id="tutorial",
            text_content="Python list comprehension tutorial",
            metadata={"url": "https://www.w3schools.com/python/python_lists_comprehension.asp"},
        )
    ]

    assert (
        _official_site_supplement_query("What is Python list comprehension used for?", hits)
        == "What is Python list comprehension used for? site:docs.python.org"
    )


def test_official_site_supplement_query_skips_when_official_present() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _official_site_supplement_query

    hits = [
        SourceDoc(
            doc_id="official",
            text_content="Python docs",
            metadata={"url": "https://docs.python.org/3/tutorial/datastructures.html"},
        )
    ]

    assert _official_site_supplement_query("What is Python list comprehension used for?", hits) == ""


def test_official_seed_hits_adds_python_docs_when_official_missing() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _official_seed_hits

    hits = [
        SourceDoc(
            doc_id="tutorial",
            text_content="Python list comprehension tutorial",
            metadata={"url": "https://www.w3schools.com/python/python_lists_comprehension.asp"},
        )
    ]

    seeded = _official_seed_hits("What is Python list comprehension used for?", hits)

    assert len(seeded) == 1
    assert seeded[0].metadata["source"] == "search_official_seed"
    assert seeded[0].metadata["url"] == "https://docs.python.org/3/tutorial/datastructures.html"


def test_official_seed_hits_skips_when_official_present() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _official_seed_hits

    hits = [
        SourceDoc(
            doc_id="official",
            text_content="Python docs",
            metadata={"url": "https://docs.python.org/3/tutorial/datastructures.html"},
        )
    ]

    assert _official_seed_hits("What is Python list comprehension used for?", hits) == []


def test_official_seed_hits_adds_used_car_consumer_protection_page() -> None:
    from app.services.dispatcher import _official_seed_hits

    seeded = _official_seed_hits("购买二手车需要注意什么？", [])

    assert any(hit.metadata["url"].startswith("https://www.michigan.gov/consumerprotection") for hit in seeded)


def test_advice_supplement_query_expands_used_car_checks() -> None:
    from app.services.dispatcher import _advice_supplement_query

    query = _advice_supplement_query("购买二手车需要注意什么？")

    assert "事故车" in query
    assert "泡水车" in query
    assert "过户" in query
    assert "合同" in query
    assert "车况检测" in query


def test_advice_supplement_query_ignores_factual_question() -> None:
    from app.services.dispatcher import _advice_supplement_query

    assert _advice_supplement_query("为什么海水是咸的？") == ""


def test_crawled_doc_quality_rank_prefers_official_relevant_body() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_crawled_docs_for_answer

    docs = [
        SourceDoc(
            doc_id="blog",
            text_content="FastAPI bootcamp day one. " * 80,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://example-blog.com/fastapi-bootcamp",
                "title": "FastAPI Bootcamp",
            },
        ),
        SourceDoc(
            doc_id="official",
            text_content="FastAPI is a modern, fast web framework for building APIs with Python. " * 60,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://fastapi.tiangolo.com/",
                "title": "FastAPI",
            },
        ),
    ]

    ranked = _rank_crawled_docs_for_answer("What is FastAPI used for?", docs)

    assert ranked[0].doc_id == "official"
    assert ranked[0].metadata["evidence_quality_score"] > ranked[1].metadata["evidence_quality_score"]


def test_crawled_doc_quality_rank_prefers_official_docs_over_keyword_dense_tutorial() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_crawled_docs_for_answer

    docs = [
        SourceDoc(
            doc_id="tutorial",
            text_content="Python list comprehension creates lists. " * 80,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://mimo.org/glossary/python/list-comprehension",
                "title": "Python List Comprehension",
            },
        ),
        SourceDoc(
            doc_id="python-docs",
            text_content="List comprehensions provide a concise way to create lists. " * 20,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://docs.python.org/3/tutorial/datastructures.html",
                "title": "Data Structures - Python Tutorial",
            },
        ),
    ]

    ranked = _rank_crawled_docs_for_answer("What is Python list comprehension used for?", docs)

    assert ranked[0].doc_id == "python-docs"


def test_placeholder_crawl_can_fall_back_to_search_snippet_evidence() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import (
        _rank_crawled_docs_for_answer,
        _replace_placeholder_crawls_with_search_hits,
    )

    hits = [
        SourceDoc(
            doc_id="hit-official",
            text_content="FastAPI framework, ready for production.",
            metadata={
                "url": "https://fastapi.tiangolo.com/",
                "title": "FastAPI",
                "snippet": "FastAPI framework, high performance, easy to learn, ready for production.",
            },
        ),
        SourceDoc(
            doc_id="hit-blog",
            text_content="Blog article",
            metadata={
                "url": "https://example-blog.com/fastapi",
                "title": "FastAPI blog",
                "snippet": "Personal article.",
            },
        ),
    ]
    crawled = [
        SourceDoc(
            doc_id="placeholder",
            text_content="Fetched content from https://fastapi.tiangolo.com/",
            metadata={"source": "crawl4ai", "url": "https://fastapi.tiangolo.com/"},
        ),
        SourceDoc(
            doc_id="blog",
            text_content="FastAPI bootcamp day one. " * 80,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://example-blog.com/fastapi",
                "title": "FastAPI blog",
            },
        ),
    ]

    replaced = _replace_placeholder_crawls_with_search_hits(
        "What is FastAPI used for?",
        hits,
        crawled,
    )
    ranked = _rank_crawled_docs_for_answer("What is FastAPI used for?", replaced)

    assert replaced[0].metadata["source"] == "search_result_evidence"
    assert ranked[0].doc_id.startswith("search_evidence::")
    assert "ready for production" in ranked[0].text_content


def test_placeholder_detection_handles_any_fetched_content_source() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _is_placeholder_crawl_doc

    assert _is_placeholder_crawl_doc(
        SourceDoc(
            doc_id="placeholder-local",
            text_content="Fetched content from https://example.org/page",
            metadata={"source": "crawl4ai_local_sdk", "url": "https://example.org/page"},
        )
    )


def test_search_snippet_evidence_omits_raw_url_from_text() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _search_hit_as_evidence_doc

    doc = _search_hit_as_evidence_doc(
        SourceDoc(
            doc_id="hit-budget",
            text_content="Monthly budget guide",
            metadata={
                "url": "https://consumerfinance.gov/consumer-tools/budgeting/",
                "title": "Budgeting",
                "snippet": "A budget helps you decide how to spend and save your money each month.",
            },
        )
    )

    assert doc is not None
    assert "consumerfinance.gov" not in doc.text_content
    assert "spend and save" in doc.text_content


def test_source_quality_promotes_trusted_general_domains() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_search_hits_for_crawl

    hits = [
        SourceDoc(
            doc_id="forum",
            text_content="budget discussion",
            metadata={
                "url": "https://reddit.com/r/personalfinance/example",
                "title": "How do you budget?",
                "snippet": "People discuss household budget tips.",
            },
        ),
        SourceDoc(
            doc_id="trusted",
            text_content="budget guide",
            metadata={
                "url": "https://consumerfinance.gov/consumer-tools/budgeting/",
                "title": "Budgeting",
                "snippet": "A budget helps you plan how to spend and save money.",
            },
        ),
    ]

    ranked = _rank_search_hits_for_crawl("普通家庭怎么做月度预算？ household budget", hits)

    assert ranked[0].doc_id == "trusted"


def test_used_car_query_prefers_before_buying_over_after_purchase() -> None:
    from app.models.schemas import SourceDoc
    from app.services.dispatcher import _rank_crawled_docs_for_answer

    docs = [
        SourceDoc(
            doc_id="after",
            text_content="买二手车后要做的五件事，更换机油、制动油、冷却液和轮胎。" * 10,
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://jtgl.beijing.gov.cn/jgj/94220/aqcs/127544/index.html",
                "title": "买二手车后要做的五件事",
            },
        ),
        SourceDoc(
            doc_id="before",
            text_content=(
                "Before signing an agreement to purchase a used vehicle, examine the vehicle "
                "using an inspection checklist, check accident history, ask for maintenance records, "
                "test drive the car, get an independent inspection, and read the purchase agreement."
            ),
            metadata={
                "source": "http_fallback_crawl",
                "url": "https://www.michigan.gov/consumerprotection/protect-yourself/consumer-alerts/auto/before-buying-a-used-car",
                "title": "Before Buying a Used Car",
            },
        ),
    ]

    ranked = _rank_crawled_docs_for_answer("购买二手车需要注意什么？", docs)

    assert ranked[0].doc_id == "before"


def test_web_retrieval_query_uses_llm_rewrite_without_second_optimization(monkeypatch) -> None:
    from app.models.schemas import GeneralQueryConstraints, QueryExecutionContext
    from app.services import dispatcher as dispatcher_module

    def fail_optimizer(_query):
        raise AssertionError("optimize_web_query should not run for LLM planner rewrite")

    monkeypatch.setattr(dispatcher_module, "optimize_web_query", fail_optimizer)

    ctx = QueryExecutionContext(
        uid="u-web-query",
        intent="general_qa",
        query="planner rewrite",
        original_query="original question",
        general_constraints=GeneralQueryConstraints(
            raw_query="original question",
            search_rewrite="planner rewrite",
            parser_source="llm_planner",
        ),
    )

    query, source = dispatcher_module._select_web_retrieval_query(ctx)

    assert query == "planner rewrite"
    assert source == "llm_planner"


def test_web_retrieval_query_falls_back_for_heuristic_constraints(monkeypatch) -> None:
    from app.models.schemas import GeneralQueryConstraints, QueryExecutionContext
    from app.services import dispatcher as dispatcher_module

    seen = {}

    def fake_optimizer(query):
        seen["query"] = query
        return f"fallback::{query}"

    monkeypatch.setattr(dispatcher_module, "optimize_web_query", fake_optimizer)

    ctx = QueryExecutionContext(
        uid="u-web-query-fallback",
        intent="general_qa",
        query="heuristic rewrite",
        original_query="original question",
        general_constraints=GeneralQueryConstraints(
            raw_query="original question",
            search_rewrite="heuristic rewrite",
            parser_source="heuristic",
        ),
    )

    query, source = dispatcher_module._select_web_retrieval_query(ctx)

    assert query == "fallback::original question"
    assert source == "heuristic_fallback"
    assert seen["query"] == "original question"


def test_dispatcher_skips_unsafe_selected_urls() -> None:
    from app.models.schemas import QueryRequest, SourceDoc
    from app.services.dispatcher import TaskDispatcher
    import asyncio

    search_hits = [
        SourceDoc(doc_id="hit-local", text_content="local", metadata={"url": "http://127.0.0.1/a"}),
        SourceDoc(doc_id="hit-public", text_content="public", metadata={"url": "https://example.com/a"}),
    ]
    crawled_urls = []

    class FakeSearch:
        async def search_web_hits(self, _query, top_k):
            return search_hits[:top_k]

    class FakeRerank:
        async def rerank(self, _query, docs, top_k):
            return docs[:top_k]

    class FakeCrawl:
        async def crawl(self, url):
            crawled_urls.append(url)
            return SourceDoc(doc_id=url, text_content=url, metadata={"url": url})

    class FakeImagePipeline:
        pass

    dispatcher = TaskDispatcher(FakeSearch(), FakeCrawl(), FakeRerank(), FakeImagePipeline())
    req = QueryRequest(
        uid="u-dispatcher-safe-url",
        intent="general_qa",
        query="hello",
        max_web_docs=2,
        use_rasa_intent=False,
    )
    asyncio.run(dispatcher.prepare_documents(req))

    assert crawled_urls == ["https://example.com/a"]


def test_image_pipeline_returns_clip_results_without_vlm_rerank(monkeypatch) -> None:
    from app.integrations import image_pipeline_bridge as bridge
    import asyncio
    import sys
    import types

    async def fake_search_source(_query, _limit):
        return [
            bridge.ImageCandidate(url="https://example.com/1.jpg", title="one", desc="one"),
            bridge.ImageCandidate(url="https://example.com/2.jpg", title="two", desc="two"),
        ]

    async def fake_clip_filter(_query, candidates, top_k):
        candidates[1].score = 0.9
        candidates[0].score = 0.5
        return [candidates[1], candidates[0]][:top_k]

    async def fake_accessible(preferred, fallback_pool, top_k):
        return preferred[:top_k]

    async def forbidden_vlm_rank_clip_pool(*_args, **_kwargs):
        raise AssertionError("image pipeline should not run VLM rerank after CLIP")

    fake_qwen_module = types.SimpleNamespace(
        has_vlm_credentials=lambda: True,
        vlm_rank_clip_pool=forbidden_vlm_rank_clip_pool,
    )
    monkeypatch.setitem(sys.modules, "app.services.qwen_vlm_images", fake_qwen_module)
    monkeypatch.setattr(bridge.bridge_settings, "image_search_provider", "unsplash")
    monkeypatch.setattr(bridge, "_search_unsplash_source", fake_search_source)
    monkeypatch.setattr(bridge, "_chinese_clip_filter", fake_clip_filter)
    monkeypatch.setattr(bridge, "_ensure_accessible_topk", fake_accessible)

    result = asyncio.run(bridge.search_rank(bridge.ImageSearchRequest(query="dog", top_k=2)))

    assert [item["url"] for item in result["images"]] == [
        "https://example.com/2.jpg",
        "https://example.com/1.jpg",
    ]


def test_memory_pref_value_decodes_json_string() -> None:
    from app.services.memory_client import MemoryClient

    assert MemoryClient._decode_pref_value('{"answer_style":"short"}') == {"answer_style": "short"}
    assert MemoryClient._decode_pref_value("plain") == "plain"


def test_image_query_parser_extracts_spatial_constraints() -> None:
    from app.services.image_query_parser import parse_image_search_constraints
    from app.services import image_query_parser
    import asyncio

    old_openai = image_query_parser.bridge_settings.openai_api_key
    old_qwen = image_query_parser.bridge_settings.qwen_api_key
    try:
        image_query_parser.bridge_settings.openai_api_key = None
        image_query_parser.bridge_settings.qwen_api_key = None
        result = asyncio.run(
            parse_image_search_constraints(
                "给我几张左边是边牧右边是金毛的照片",
                {"image_count": "几张"},
            )
        )
    finally:
        image_query_parser.bridge_settings.openai_api_key = old_openai
        image_query_parser.bridge_settings.qwen_api_key = old_qwen
    assert result.count == 5
    assert result.spatial_relations
    rel = result.spatial_relations[0]
    assert rel.primary_subject == "边牧"
    assert rel.secondary_subject == "金毛"
    assert "边牧" in (result.search_rewrite or "")
    assert "金毛" in (result.search_rewrite or "")


def test_optimize_image_query_prefers_structured_rewrite() -> None:
    from app.models.schemas import ImageSearchConstraints
    from app.services.query_optimizer import optimize_image_query_with_constraints

    constraints = ImageSearchConstraints(
        raw_query="给我几张左边是边牧右边是金毛的照片",
        search_rewrite="边牧 金毛 同框 合照 照片",
        subjects=["边牧", "金毛"],
    )
    rewritten = optimize_image_query_with_constraints(
        "给我几张左边是边牧右边是金毛的照片",
        constraints,
        {},
    )
    assert "边牧" in rewritten and "金毛" in rewritten
    assert "同框" in rewritten


def test_optimize_image_query_refine_spatial_rewrite_for_scene() -> None:
    from app.models.schemas import ImageSearchConstraints, SpatialRelationConstraint
    from app.services.query_optimizer import optimize_image_query_with_constraints

    constraints = ImageSearchConstraints(
        raw_query="给我几张左边是边牧右边是金毛的照片",
        search_rewrite="边牧 金毛 同框 对比",
        subjects=["边牧", "金毛"],
        spatial_relations=[
            SpatialRelationConstraint(
                relation="left_of",
                primary_subject="边牧",
                secondary_subject="金毛",
            )
        ],
    )
    rewritten = optimize_image_query_with_constraints(
        "给我几张左边是边牧右边是金毛的照片",
        constraints,
        {},
    )
    assert rewritten == "边牧 金毛 同框 对比"


def test_apply_image_constraints_consumes_search_rewrite_without_optimizer(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import ImageSearchConstraints, QueryExecutionContext

    def fail_optimizer(_query, _entities):
        raise AssertionError("image optimizer should only run when constraints are absent")

    monkeypatch.setattr(chat, "optimize_image_query", fail_optimizer)

    ctx = QueryExecutionContext(
        uid="u-image-constraints",
        intent="image_search",
        query="user semantic query",
        original_query="original image request",
    )

    updated = chat._apply_image_constraints(
        ctx,
        ImageSearchConstraints(
            raw_query="original image request",
            search_rewrite="planned image retrieval query",
            count=3,
            parser_source="llm_planner",
        ),
        {},
    )

    assert updated.image_search_query == "planned image retrieval query"
    assert updated.query == "user semantic query"
    assert updated.max_images == 3


def test_apply_image_constraints_without_rewrite_uses_current_query(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import ImageSearchConstraints, QueryExecutionContext

    def fail_optimizer(_query, _entities):
        raise AssertionError("image optimizer should only run when constraints are absent")

    monkeypatch.setattr(chat, "optimize_image_query", fail_optimizer)

    ctx = QueryExecutionContext(
        uid="u-image-constraints-empty",
        intent="image_search",
        query="current query",
        original_query="original image request",
    )

    updated = chat._apply_image_constraints(
        ctx,
        ImageSearchConstraints(
            raw_query="original image request",
            search_rewrite=None,
            parser_source="heuristic",
        ),
        {},
    )

    assert updated.image_search_query == "current query"


def test_general_query_parser_extracts_city_heuristically() -> None:
    from app.services.image_query_parser import parse_general_query_constraints
    from app.services import image_query_parser
    import asyncio

    old_openai = image_query_parser.bridge_settings.openai_api_key
    old_qwen = image_query_parser.bridge_settings.qwen_api_key
    try:
        image_query_parser.bridge_settings.openai_api_key = None
        image_query_parser.bridge_settings.qwen_api_key = None
        result = asyncio.run(parse_general_query_constraints("杭州今天会下雨吗"))
    finally:
        image_query_parser.bridge_settings.openai_api_key = old_openai
        image_query_parser.bridge_settings.qwen_api_key = old_qwen

    assert result.city == "杭州"
    assert result.entities["location"] == "杭州"
    assert "杭州" in (result.search_rewrite or "")


def test_chat_query_general_qa_keeps_runtime_flags(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import NormalizedDocument, NormalizedPayload, QueryResponse

    async def fake_normalize_input(payload):
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent or "general_qa",
            query=payload.query,
            original_query=payload.original_query or payload.query,
            general_constraints=payload.general_constraints,
            documents=[
                NormalizedDocument(
                    doc_id="doc-2",
                    text="test general document",
                    modal_elements=[],
                    metadata={"source": "test"},
                )
            ],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-2"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="general-ok",
            evidence=[],
            images=[],
            trace_id="tr_test_002",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-general",
            "intent": "general_qa",
            "query": "杭州今天会下雨吗",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general-ok"
    assert data["route"] == "general_qa"
    assert "runtime_flags" in data


def test_chat_query_with_source_docs_skips_general_llm_parser(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import NormalizedDocument, NormalizedPayload, QueryResponse

    captured = {}

    async def forbidden_parse_general(_query):
        raise AssertionError("source-doc QA should not need general query parser")

    async def fake_normalize_input(payload):
        captured["constraints_source"] = payload.general_constraints.parser_source
        captured["query"] = payload.query
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent,
            query=payload.query,
            original_query=payload.original_query,
            general_constraints=payload.general_constraints,
            documents=[NormalizedDocument(doc_id="doc-context", text="context")],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-context"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="context-ok",
            evidence=[],
            images=[],
            trace_id="tr_context",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "parse_general_query_constraints", forbidden_parse_general)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-source-docs-direct",
            "intent": "general_qa",
            "query": "summarize this material",
            "use_rasa_intent": False,
            "source_docs": [
                {
                    "doc_id": "doc-context",
                    "text_content": "material",
                    "modal_elements": [],
                    "structure": {},
                    "metadata": {},
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "context-ok"
    assert captured["constraints_source"] == "request_context"
    assert captured["query"] == "summarize this material"


def test_chat_query_rejects_rasa_image_for_weather_compare(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import GeneralQueryConstraints, NormalizedDocument, NormalizedPayload, QueryResponse

    async def fake_rasa_parse(_query):
        return "image_search", 0.99, {}

    async def fake_parse_general(_query, **_kwargs):
        return GeneralQueryConstraints(
            raw_query=_query,
            search_rewrite=_query,
            parser_source="heuristic",
        )

    async def fake_normalize_input(payload):
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent or "general_qa",
            query=payload.query,
            original_query=payload.original_query or payload.query,
            general_constraints=payload.general_constraints,
            documents=[
                NormalizedDocument(
                    doc_id="doc-weather",
                    text="weather doc",
                    modal_elements=[],
                    metadata={"source": "test"},
                )
            ],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-weather"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="weather-ok",
            evidence=[],
            images=[],
            trace_id="tr_test_weather",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat.rasa_client, "parse", fake_rasa_parse)
    monkeypatch.setattr(chat, "parse_general_query_constraints", fake_parse_general)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-weather",
            "query": "北京和上海未来三天天气对比",
            "use_rasa_intent": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "general_qa"


def test_image_search_uses_dedicated_search_query_without_overwriting_user_query(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import (
        ImageSearchConstraints,
        NormalizedDocument,
        NormalizedPayload,
        QueryResponse,
        SpatialRelationConstraint,
    )

    captured = {}

    async def fake_parse_image_constraints(_query, _entities, **_kwargs):
        return ImageSearchConstraints(
            raw_query=_query,
            search_rewrite="金毛 边牧 同框",
            subjects=["金毛", "边牧"],
            spatial_relations=[
                SpatialRelationConstraint(
                    relation="left_of",
                    primary_subject="金毛",
                    secondary_subject="边牧",
                )
            ],
            parser_source="llm",
        )

    async def fake_normalize_input(payload):
        captured["query"] = payload.query
        captured["image_search_query"] = payload.image_search_query
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent or "image_search",
            query=payload.query,
            image_search_query=payload.image_search_query,
            original_query=payload.original_query or payload.query,
            image_constraints=payload.image_constraints,
            documents=[
                NormalizedDocument(
                    doc_id="doc-img",
                    text="img doc",
                    modal_elements=[],
                    metadata={"source": "test"},
                )
            ],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-img"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="img-ok",
            evidence=[],
            images=[],
            trace_id="tr_test_img_query",
            latency_ms=1,
            route="image_search",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "parse_image_search_constraints", fake_parse_image_constraints)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-image-query",
            "intent": "image_search",
            "query": "给我几张左边是金毛右边是边牧的照片",
            "use_rasa_intent": False,
        },
    )
    assert resp.status_code == 200
    assert captured["query"] == "给我几张左边是金毛右边是边牧的照片"
    assert captured["image_search_query"] == "金毛 边牧 同框"


def test_chat_query_uses_planner_image_constraints_without_second_parser(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import ImageSearchConstraints, NormalizedDocument, NormalizedPayload, QueryResponse
    from app.services.query_planner import QueryPlan

    captured = {}

    async def fake_plan_query(_req, **_kwargs):
        return QueryPlan(
            intent="image_search",
            confidence=0.99,
            source="llm_planner",
            image_constraints=ImageSearchConstraints(
                raw_query=_req.query,
                search_rewrite="金毛 边牧 同框 照片",
                subjects=["金毛", "边牧"],
                count=4,
                parser_source="llm_planner",
            ),
            flags=["query_planner_llm"],
        )

    async def forbidden_parse_image_constraints(_query, _entities):
        raise AssertionError("planner path should not call image parser again")

    async def fake_normalize_input(payload):
        captured["intent"] = payload.intent
        captured["query"] = payload.query
        captured["image_search_query"] = payload.image_search_query
        captured["max_images"] = payload.max_images
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent,
            query=payload.query,
            image_search_query=payload.image_search_query,
            original_query=payload.original_query or payload.query,
            image_constraints=payload.image_constraints,
            documents=[NormalizedDocument(doc_id="doc-plan-img", text="img doc")],
        )

    async def fake_ingest_to_rag(_normalized):
        return []

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="planner-img-ok",
            evidence=[],
            images=[],
            trace_id="tr_planner_img",
            latency_ms=1,
            route="image_search",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "plan_query", fake_plan_query)
    monkeypatch.setattr(chat, "parse_image_search_constraints", forbidden_parse_image_constraints)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-planner-img",
            "query": "给我几张左边是金毛右边是边牧的照片",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    assert captured["intent"] == "image_search"
    assert captured["query"] == "给我几张左边是金毛右边是边牧的照片"
    assert captured["image_search_query"] == "金毛 边牧 同框 照片"
    assert captured["max_images"] == 4


def test_chat_query_uses_planner_general_constraints_without_second_parser(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import GeneralQueryConstraints, NormalizedDocument, NormalizedPayload, QueryResponse
    from app.services.query_planner import QueryPlan
    import asyncio

    captured = {}
    uid = "u-test-planner-general"
    asyncio.run(chat.memory_client.set_preference(uid, "profile", {"location": "杭州"}))
    asyncio.run(chat.memory_client.set_preference(uid, "response", {"style": "简洁"}))

    async def fake_plan_query(_req, **_kwargs):
        captured["user_context"] = _kwargs.get("user_context")
        return QueryPlan(
            intent="general_qa",
            confidence=0.98,
            source="llm_planner",
            general_constraints=GeneralQueryConstraints(
                raw_query=_req.query,
                search_rewrite="杭州 今天 降雨 天气",
                entities={"location": "杭州"},
                attributes=["天气", "降雨"],
                parser_source="llm_planner",
            ),
            flags=["query_planner_llm"],
        )

    async def forbidden_parse_general_constraints(_query):
        raise AssertionError("planner path should not call general parser again")

    async def fake_normalize_input(payload):
        captured["intent"] = payload.intent
        captured["query"] = payload.query
        captured["general_constraints"] = payload.general_constraints
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent,
            query=payload.query,
            original_query=payload.original_query or payload.query,
            general_constraints=payload.general_constraints,
            documents=[NormalizedDocument(doc_id="doc-plan-general", text="general doc")],
        )

    async def fake_ingest_to_rag(_normalized):
        return []

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="planner-general-ok",
            evidence=[],
            images=[],
            trace_id="tr_planner_general",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "plan_query", fake_plan_query)
    monkeypatch.setattr(chat, "parse_general_query_constraints", forbidden_parse_general_constraints)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": uid,
            "query": "杭州今天会下雨吗",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    assert captured["intent"] == "general_qa"
    assert captured["query"] == "杭州 今天 降雨 天气"
    assert captured["general_constraints"].entities["location"] == "杭州"
    assert captured["user_context"]["profile"]["location"] == "杭州"
    assert captured["user_context"]["response"]["style"] == "简洁"


def test_planner_failure_forces_legacy_parser_heuristic_mode(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import GeneralQueryConstraints, NormalizedDocument, NormalizedPayload, QueryResponse

    captured = {}

    async def fake_plan_query(_req, **_kwargs):
        return None

    async def fake_parse_general(_query, **kwargs):
        captured["allow_llm"] = kwargs.get("allow_llm")
        return GeneralQueryConstraints(
            raw_query=_query,
            search_rewrite=_query,
            parser_source="heuristic",
        )

    async def fake_normalize_input(payload):
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent,
            query=payload.query,
            original_query=payload.original_query,
            general_constraints=payload.general_constraints,
            documents=[NormalizedDocument(doc_id="doc-heuristic", text="doc")],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-heuristic"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="heuristic-ok",
            evidence=[],
            images=[],
            trace_id="tr_heuristic",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "plan_query", fake_plan_query)
    monkeypatch.setattr(chat, "parse_general_query_constraints", fake_parse_general)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": "u-test-planner-fail-heuristic-parser",
            "query": "What is FastAPI used for?",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    assert captured["allow_llm"] is False


def test_planner_clarification_writes_unified_pending_and_completes_progress(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import ImageSearchConstraints
    from app.services.query_planner import QueryPlan
    import asyncio

    uid = "u-test-planner-clarification"
    request_id = "reqclarify001"

    async def fake_plan_query(_req, **_kwargs):
        return QueryPlan(
            intent="image_search",
            confidence=0.99,
            source="llm_planner",
            image_constraints=ImageSearchConstraints(
                raw_query=_req.query,
                search_rewrite=_req.query,
                needs_clarification=True,
                clarification_question="你想看哪个主体的图片？",
                parser_source="llm_planner",
            ),
            flags=["query_planner_llm"],
        )

    async def forbidden_normalize_input(_payload):
        raise AssertionError("clarification should return before normalization")

    monkeypatch.setattr(chat, "plan_query", fake_plan_query)
    monkeypatch.setattr(chat.adapter, "normalize_input", forbidden_normalize_input)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": uid,
            "request_id": request_id,
            "query": "来几张图",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "你想看哪个主体的图片？"
    assert data["runtime_flags"] == ["clarification_needed"]

    context = asyncio.run(chat.memory_client.get_context(uid))
    pending = context["preferences"]["pending_clarification"]
    assert pending["type"] == "clarification"
    assert pending["route"] == "image_search"
    assert pending["missing"] == ["subject_or_landmark"]
    assert pending["original_query"] == "来几张图"

    progress = client.get("/v1/chat/progress", params={"request_id": request_id}).json()
    assert progress["status"] == "completed"


def test_pending_clarification_is_merged_before_planner(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import GeneralQueryConstraints, NormalizedDocument, NormalizedPayload, QueryResponse
    from app.services.query_planner import QueryPlan
    import asyncio

    uid = "u-test-pending-before-planner"
    captured = {}

    asyncio.run(
        chat.memory_client.set_preference(
            uid,
            "pending_clarification",
            {
                "type": "clarification",
                "route": "general_qa",
                "original_query": "今天会下雨吗",
                "question": "你想查询哪个城市的天气？",
                "missing": ["city"],
                "created_at": 1,
            },
        )
    )

    async def fake_plan_query(_req, **_kwargs):
        captured["planner_query"] = _req.query
        return QueryPlan(
            intent="general_qa",
            confidence=0.99,
            source="llm_planner",
            general_constraints=GeneralQueryConstraints(
                raw_query=_req.query,
                search_rewrite=_req.query,
                entities={"location": "北京"},
                parser_source="llm_planner",
            ),
            flags=["query_planner_llm"],
        )

    async def fake_normalize_input(payload):
        captured["normalized_query"] = payload.query
        return NormalizedPayload(
            uid=payload.uid,
            intent=payload.intent,
            query=payload.query,
            original_query=payload.original_query,
            general_constraints=payload.general_constraints,
            documents=[NormalizedDocument(doc_id="doc-pending", text="weather doc")],
        )

    async def fake_ingest_to_rag(_normalized):
        return ["doc-pending"]

    async def fake_query_with_context(_normalized):
        return QueryResponse(
            answer="pending-ok",
            evidence=[],
            images=[],
            trace_id="tr_pending",
            latency_ms=1,
            route="general_qa",
            runtime_flags=[],
        )

    monkeypatch.setattr(chat, "plan_query", fake_plan_query)
    monkeypatch.setattr(chat.adapter, "normalize_input", fake_normalize_input)
    monkeypatch.setattr(chat.adapter, "ingest_to_rag", fake_ingest_to_rag)
    monkeypatch.setattr(chat.adapter, "query_with_context", fake_query_with_context)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": uid,
            "query": "北京",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    assert captured["planner_query"] == "北京 今天会下雨吗"
    assert captured["normalized_query"] == "北京 今天会下雨吗"
    context = asyncio.run(chat.memory_client.get_context(uid))
    assert context["preferences"]["pending_clarification"] == {}


def test_general_constraints_clarification_signal_ignores_non_weather_missing_slot() -> None:
    from app.models.schemas import GeneralQueryConstraints
    from app.services.clarification import decide_clarification

    decision = decide_clarification(
        query="根据材料说明通用问答链路",
        intent="general_qa",
        entities={},
        preferences={},
        image_constraints=None,
        general_constraints=GeneralQueryConstraints(
            raw_query="根据材料说明通用问答链路",
            search_rewrite="通用问答链路",
            needs_clarification=True,
            clarification_question="您的输入内容无法识别，请问您具体想查询什么信息？",
            parser_source="llm",
        ),
    )

    assert not decision.should_ask


def test_generic_image_request_uses_deterministic_clarification_fallback(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import ImageSearchConstraints
    import asyncio

    uid = "u-test-deterministic-image-clarification"
    request_id = "reqclarify002"

    async def fake_parse_image_constraints(_query, _entities, **_kwargs):
        return ImageSearchConstraints(
            raw_query=_query,
            search_rewrite=_query,
            subjects=[],
            parser_source="heuristic",
        )

    async def forbidden_normalize_input(_payload):
        raise AssertionError("clarification should return before normalization")

    monkeypatch.setattr(chat, "parse_image_search_constraints", fake_parse_image_constraints)
    monkeypatch.setattr(chat.adapter, "normalize_input", forbidden_normalize_input)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/query",
        json={
            "uid": uid,
            "request_id": request_id,
            "intent": "image_search",
            "query": "找几张图",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "你想看哪个地点或主体的图片？"
    assert data["runtime_flags"] == ["clarification_needed"]

    context = asyncio.run(chat.memory_client.get_context(uid))
    pending = context["preferences"]["pending_clarification"]
    assert pending["type"] == "clarification"
    assert pending["route"] == "image_search"
    assert pending["missing"] == ["subject_or_landmark"]

    progress = client.get("/v1/chat/progress", params={"request_id": request_id}).json()
    assert progress["status"] == "completed"


def test_crawl_client_http_fallback_extracts_real_html(monkeypatch) -> None:
    from app.services import connectors
    import asyncio

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = """
        <html>
          <head><title>FastAPI docs</title></head>
          <body>
            <nav>Navigation should be removed</nav>
            <main>
              <h1>FastAPI</h1>
              <p>FastAPI is a modern web framework for building APIs with Python.</p>
              <img src="/logo.png" alt="FastAPI logo">
            </main>
          </body>
        </html>
        """
        url = "https://example.org/docs"

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr(connectors.settings, "crawl4ai_local_enabled", False)
    monkeypatch.setattr(connectors.settings, "crawl4ai_endpoint", None)
    monkeypatch.setattr(connectors.httpx, "AsyncClient", FakeAsyncClient)

    doc = asyncio.run(connectors.CrawlClient().crawl("https://example.org/docs"))

    assert doc.metadata["source"] == "http_fallback_crawl"
    assert "FastAPI is a modern web framework" in doc.text_content
    assert "Navigation should be removed" not in doc.text_content
    assert doc.modal_elements[0].url == "https://example.org/logo.png"


def test_clean_extracted_text_removes_unsupported_browser_banner() -> None:
    from app.services.connectors import _clean_extracted_text

    text = _clean_extracted_text(
        "Unsupported Browser Detected The web Browser you are currently using is unsupported, "
        "and some features of this site may not work as intended. Please update to a modern browser.\n"
        "Before signing an agreement to purchase a used vehicle, inspect it carefully."
    )

    assert "Unsupported Browser" not in text
    assert "Before signing an agreement" in text


def test_crawl_client_local_sdk_timeout_falls_back_to_http(monkeypatch) -> None:
    from app.services import connectors
    import asyncio

    async def slow_local(_url):
        await asyncio.sleep(0.05)
        return {"text_content": "slow local result"}

    async def fast_http(_url):
        return {
            "text_content": "fast http fallback result",
            "metadata": {"source": "http_fallback_crawl", "url": _url},
            "structure": {"type": "webpage"},
        }

    monkeypatch.setattr(connectors.settings, "crawl4ai_local_enabled", True)
    monkeypatch.setattr(connectors.settings, "crawl4ai_endpoint", None)
    monkeypatch.setattr(connectors.settings, "request_timeout_seconds", 0.001)
    monkeypatch.setattr(connectors.settings, "crawl4ai_local_timeout_seconds", 0.001)
    monkeypatch.setattr(connectors.CrawlClient, "_crawl_with_local_sdk", staticmethod(slow_local))
    monkeypatch.setattr(connectors.CrawlClient, "_crawl_with_http_fallback", staticmethod(fast_http))

    doc = asyncio.run(connectors.CrawlClient().crawl("https://example.org/docs"))

    assert doc.text_content == "fast http fallback result"
    assert doc.metadata["source"] == "http_fallback_crawl"


def test_duckduckgo_html_search_mapping_decodes_result_urls() -> None:
    from app.services.connectors import _map_duckduckgo_html

    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Ffastapi.tiangolo.com%2F">FastAPI</a>
      <a class="result__snippet">FastAPI is a modern Python web framework.</a>
    </div>
    """

    hits = _map_duckduckgo_html(html, query="FastAPI", top_k=3)

    assert len(hits) == 1
    assert hits[0].metadata["source"] == "search_duckduckgo_html"
    assert hits[0].metadata["url"] == "https://fastapi.tiangolo.com/"
    assert "modern Python web framework" in hits[0].text_content


def test_duckduckgo_html_search_mapping_skips_duckduckgo_self_links() -> None:
    from app.services.connectors import _map_duckduckgo_html

    html = """
    <div class="result">
      <a class="result__a" href="https://duckduckgo.com/feedback.html">Feedback</a>
      <a class="result__snippet">DuckDuckGo feedback page.</a>
    </div>
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fconsumerfinance.gov%2Fconsumer-tools%2Fbudgeting%2F">Budgeting</a>
      <a class="result__snippet">Budgeting helps you plan spending and saving.</a>
    </div>
    """

    hits = _map_duckduckgo_html(html, query="budgeting", top_k=3)

    assert len(hits) == 1
    assert hits[0].metadata["url"] == "https://consumerfinance.gov/consumer-tools/budgeting/"


def test_raganything_ingest_can_skip_full_rag_insert(monkeypatch) -> None:
    from app.integrations import raganything_bridge
    import asyncio

    async def forbidden_get_rag():
        raise AssertionError("full RAGAnything ingest should be skipped")

    uid = "u-test-fast-ingest"
    raganything_bridge._fallback_docs.pop(uid, None)
    monkeypatch.setattr(
        raganything_bridge.bridge_settings,
        "raganything_full_ingest_enabled",
        False,
    )
    monkeypatch.setattr(raganything_bridge, "_get_rag", forbidden_get_rag)

    result = asyncio.run(
        raganything_bridge.ingest(
            raganything_bridge.IngestRequest(
                documents=[
                    raganything_bridge.IngestDocument(
                        doc_id="doc-fast-ingest",
                        text="cached evidence",
                    )
                ],
                tags={"uid": uid},
            )
        )
    )

    assert result["status"] == "ok"
    assert result["mode"] == "uid_context_cache"
    assert result["indexed_doc_ids"] == ["doc-fast-ingest"]
    assert raganything_bridge._fallback_docs[uid][-1].text == "cached evidence"


def test_raganything_uid_docs_extractive_answer_when_llm_unavailable() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-answer"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="doc-extractive",
            text="系统通过 uid 级证据缓存减少同步入库耗时，并在查询阶段直接使用本轮证据生成回答。",
        )
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="系统如何降低延迟？",
            trace_id="tr_extract",
        )
    )

    assert answer is not None
    assert "根据当前证据" in answer
    assert "模型暂不可用" not in answer
    assert "uid 级证据缓存" in answer


def test_raganything_extractive_answer_localizes_common_technical_snippet() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-fastapi"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="search_evidence-fastapi",
            text=(
                "FastAPI FastAPI is a modern, fast (high-performance), web framework "
                "for building APIs with Python based on standard Python type hints. "
                "https://fastapi.tiangolo.com/"
            ),
            metadata={
                "source": "search_result_evidence",
                "url": "https://fastapi.tiangolo.com/",
                "title": "FastAPI",
                "evidence_quality_score": 8.5,
            },
        )
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="What is FastAPI used for?",
            trace_id="tr_extract_fastapi",
        )
    )

    assert answer is not None
    assert "FastAPI 是一个" in answer
    assert "构建 API" in answer


def test_raganything_extractive_answer_hides_source_prefix_for_common_snippet() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-docker"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="search_evidence-docker",
            text=(
                "What is Docker? Docker is an open platform for developing, shipping, "
                "and running applications. Docker enables you to separate your applications "
                "from your infrastructure."
            ),
            metadata={
                "source": "search_result_evidence",
                "url": "https://docs.docker.com/get-started/docker-overview/",
                "title": "What is Docker?",
                "evidence_quality_score": 8.0,
            },
        )
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="What is Docker used for?",
            trace_id="tr_extract_docker",
        )
    )

    assert answer is not None
    assert "Docker 是一个" in answer
    assert "来源类型=" not in answer.splitlines()[0]
    assert "证据来源：" not in answer
    assert "URL=" not in answer


def test_raganything_extractive_answer_uses_relevant_sentences_not_navigation() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-redis-noise"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="doc-redis-noisy",
            text=(
                "Resource Center Events & webinars Blog Videos Glossary Demo Center. "
                "Back to blog Blog Redis Use Case Examples for Developers. "
                "Developers rely on Redis Enterprise for critical use cases across several industries. "
                "Learn several scenarios where Redis has made a difference in application development "
                "for gaming, retail, IoT networking, and travel."
            ),
            metadata={
                "source": "http_fallback_crawl",
                "title": "Redis Use Case Examples for Developers | Redis",
                "evidence_quality_score": 7.0,
            },
        )
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="What is Redis used for?",
            trace_id="tr_extract_redis_noise",
        )
    )

    assert answer is not None
    assert "Redis 可用于" in answer
    assert "Resource Center" not in answer
    assert "Events & webinars" not in answer


def test_raganything_extractive_answer_localizes_used_car_checklist() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-used-car"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="doc-used-car",
            text=(
                "Before signing an agreement to purchase a used vehicle, you should examine "
                "the vehicle using an inspection checklist, find out if the vehicle was involved "
                "in an accident, ask for maintenance records, take it to a local mechanic for "
                "an independent inspection, and read the terms of the purchase agreement before signing."
            ),
            metadata={
                "source": "http_fallback_crawl",
                "title": "Before Buying a Used Car",
                "evidence_quality_score": 8.0,
            },
        )
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="购买二手车需要注意什么？",
            trace_id="tr_extract_used_car",
        )
    )

    assert answer is not None
    assert "购买二手车建议" in answer
    assert "独立技师或第三方机构检测" in answer


def test_raganything_extractive_answer_structures_used_car_advice() -> None:
    from app.integrations import raganything_bridge

    uid = "u-test-extractive-used-car-structured"
    raganything_bridge._fallback_docs[uid] = [
        raganything_bridge.IngestDocument(
            doc_id="doc-used-car-before",
            text=(
                "Before signing an agreement to purchase a used vehicle, examine the vehicle "
                "using an inspection checklist, check accident history and maintenance records, "
                "take a test drive, get an independent inspection, and read the contract before signing."
            ),
            metadata={
                "source": "http_fallback_crawl",
                "title": "Before Buying a Used Car",
                "evidence_quality_score": 9.0,
            },
        ),
        raganything_bridge.IngestDocument(
            doc_id="doc-used-car-after",
            text="买二手车后要做的五件事，更换机油及油格，更换波箱油，检查轮胎。",
            metadata={
                "source": "http_fallback_crawl",
                "title": "买二手车后要做的五件事",
                "evidence_quality_score": 3.0,
            },
        ),
    ]

    answer = raganything_bridge._extractive_answer_from_uid_docs(
        raganything_bridge.QueryRequest(
            uid=uid,
            query="购买二手车需要注意什么？",
            trace_id="tr_extract_used_car_structured",
        )
    )

    assert answer is not None
    assert "先查历史、再验车、再签约" in answer
    assert "买二手车后要做的五件事" not in answer
    assert "所有承诺写进合同" in answer


def test_raganything_generation_candidates_include_qwen_fallback(monkeypatch) -> None:
    from app.integrations import raganything_bridge

    monkeypatch.setattr(raganything_bridge.bridge_settings, "openai_api_key", "openai-key")
    monkeypatch.setattr(raganything_bridge.bridge_settings, "openai_base_url", "https://openai.example/v1")
    monkeypatch.setattr(raganything_bridge.bridge_settings, "raganything_llm_model", "openai-model")
    monkeypatch.setattr(raganything_bridge.bridge_settings, "qwen_api_key", "qwen-key")
    monkeypatch.setattr(raganything_bridge.bridge_settings, "qwen_base_url", "https://qwen.example/v1")
    monkeypatch.setattr(raganything_bridge.bridge_settings, "qwen_model", "qwen-model")

    candidates = raganything_bridge._chat_generation_candidates()

    assert [item["base_url"] for item in candidates] == [
        "https://openai.example/v1",
        "https://qwen.example/v1",
    ]


def test_raganything_prompt_guides_primary_answer_without_exposing_metadata() -> None:
    from app.integrations import raganything_bridge

    block = raganything_bridge._build_context_block(
        1,
        raganything_bridge.IngestDocument(
            doc_id="doc-docker",
            text="Docker is an open platform for developing, shipping, and running applications.",
            metadata={
                "source": "http_fallback_crawl",
                "title": "What is Docker?",
                "url": "https://docs.docker.com/get-started/docker-overview/",
                "evidence_quality_score": 8.0,
            },
        ),
    )
    prompt = raganything_bridge._build_uid_docs_prompt(
        raganything_bridge.QueryRequest(
            uid="u-test-prompt-quality",
            query="What is Docker used for?",
            trace_id="tr_prompt_quality",
        ),
        [block],
    )

    assert "先用一句话说明对象是什么和核心用途" in prompt
    assert "不要输出证据编号、内部字段名、链接、来源类型或质量分" in prompt
    assert "证据1（网页正文）" in prompt
    assert "source_type=webpage_body" not in prompt
    assert "quality_score=8.000" not in prompt
    assert "doc_id=doc-docker" not in prompt


def test_raganything_sanitizes_generated_internal_metadata() -> None:
    from app.integrations import raganything_bridge

    answer = raganything_bridge._sanitize_answer(
        "Redis 可以作为缓存和消息代理。\n来源类型=search_summary；URL=https://redis.io/docs\nquality_score=8.0",
        "What is Redis used for?",
    )

    assert "Redis 可以作为缓存和消息代理" in answer
    assert "来源类型" not in answer
    assert "search_summary" not in answer
    assert "https://redis.io" not in answer
