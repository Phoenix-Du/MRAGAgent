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
