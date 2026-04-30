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


def test_chat_query_rejects_rasa_image_for_weather_compare(monkeypatch) -> None:
    from app.api import chat
    from app.models.schemas import GeneralQueryConstraints, NormalizedDocument, NormalizedPayload, QueryResponse

    async def fake_rasa_parse(_query):
        return "image_search", 0.99, {}

    async def fake_parse_general(_query):
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

    async def fake_parse_image_constraints(_query, _entities):
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

    async def fake_plan_query(_req):
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

    captured = {}

    async def fake_plan_query(_req):
        return QueryPlan(
            intent="general_qa",
            confidence=0.98,
            source="llm_planner",
            general_constraints=GeneralQueryConstraints(
                raw_query=_req.query,
                search_rewrite="杭州 今天 降雨 天气",
                city="杭州",
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
            "uid": "u-test-planner-general",
            "query": "杭州今天会下雨吗",
            "use_rasa_intent": False,
        },
    )

    assert resp.status_code == 200
    assert captured["intent"] == "general_qa"
    assert captured["query"] == "杭州 今天 降雨 天气"
    assert captured["general_constraints"].city == "杭州"
