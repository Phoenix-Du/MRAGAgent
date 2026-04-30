from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any

from app.models.schemas import (
    ActionRelationConstraint,
    GeneralQueryConstraints,
    ImageSearchConstraints,
    IntentType,
    ObjectRelationConstraint,
    QueryRequest,
    SpatialRelationConstraint,
)
from app.services.llm_json_client import llm_env, post_llm_json

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    intent: IntentType
    confidence: float
    source: str
    entities: dict[str, str] = field(default_factory=dict)
    image_constraints: ImageSearchConstraints | None = None
    general_constraints: GeneralQueryConstraints | None = None
    flags: list[str] = field(default_factory=list)


async def plan_query(req: QueryRequest) -> QueryPlan | None:
    api_key, base, model = llm_env()
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_prompt(req)}],
        "temperature": 0,
        "max_tokens": 900,
    }
    try:
        obj = await post_llm_json(base=base, api_key=api_key, payload=payload, retries=1)
    except Exception:
        logger.exception("query_planner_llm_failed")
        return None

    if not obj:
        logger.warning("query_planner_bad_json")
        return None
    return plan_from_obj(req.query, obj)


def plan_from_obj(raw_query: str, obj: dict[str, Any]) -> QueryPlan | None:
    intent = _normalize_intent(obj.get("intent"))
    if intent is None:
        return None
    confidence = _coerce_confidence(obj.get("confidence"))
    entities = _string_dict(obj.get("entities"))
    search_rewrite = str(obj.get("search_rewrite") or raw_query).strip() or raw_query

    if intent == "image_search":
        constraints = _image_constraints_from_obj(raw_query, search_rewrite, obj.get("image_constraints") or {}, entities)
        return QueryPlan(
            intent=intent,
            confidence=confidence,
            source="llm_planner",
            entities=entities,
            image_constraints=constraints,
            flags=["query_planner_llm", "image_query_rewritten"],
        )

    constraints = _general_constraints_from_obj(raw_query, search_rewrite, obj.get("general_constraints") or {})
    return QueryPlan(
        intent=intent,
        confidence=confidence,
        source="llm_planner",
        entities=entities,
        general_constraints=constraints,
        flags=["query_planner_llm", "general_query_rewritten"],
    )


def _build_prompt(req: QueryRequest) -> str:
    schema = {
        "intent": "general_qa | image_search",
        "confidence": "0.0-1.0",
        "search_rewrite": "string optimized for the selected retrieval pipeline",
        "entities": {},
        "general_constraints": {
            "city": None,
            "attributes": [],
            "compare_targets": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
        "image_constraints": {
            "subjects": [],
            "attributes": [],
            "subject_synonyms": {},
            "style_terms": [],
            "exclude_terms": [],
            "count": None,
            "landmark": None,
            "time_of_day": None,
            "must_have_all_subjects": True,
            "spatial_relations": [],
            "action_relations": [],
            "object_relations": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
    }
    context = {
        "has_url": bool(req.url),
        "has_source_docs": bool(req.source_docs),
        "has_user_images": bool(req.images),
        "max_images": req.max_images,
        "max_web_docs": req.max_web_docs,
    }
    prompt = (
        "\u4f60\u662f\u591a\u6a21\u6001 RAG \u7cfb\u7edf\u7684\u7edf\u4e00 query planner\u3002"
        "\u53ea\u8f93\u51fa\u4e00\u4e2a JSON \u5bf9\u8c61\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\n"
        f"\u8f93\u51fa\u7ed3\u6784: {json.dumps(schema, ensure_ascii=False)}\n"
        "\u4efb\u52a1\uff1a\u4e00\u6b21\u6027\u5b8c\u6210 intent routing \u548c\u68c0\u7d22 query/constraints \u89c4\u5212\uff0c"
        "\u907f\u514d\u540e\u7eed\u518d\u6b21\u8c03\u7528 LLM\u3002\n"
        "\u573a\u666f\u89c4\u5219\uff1a\n"
        "1) general_qa\uff1a\u95ee\u7b54\u3001\u603b\u7ed3\u7f51\u9875/\u6587\u6863\u3001\u89e3\u91ca\u3001\u5206\u6790\u3001\u5bf9\u6bd4\u3001\u5929\u6c14\u3001\u63a8\u8350\u3001\u7406\u89e3\u7528\u6237\u4e0a\u4f20\u56fe\u7247/\u6587\u6863\u3002search_rewrite \u5e94\u9002\u5408\u7f51\u9875/RAG \u68c0\u7d22\u3002\n"
        "2) image_search\uff1a\u7528\u6237\u660e\u786e\u8981\u6c42\u6309\u6587\u672c\u627e\u5916\u90e8\u56fe\u7247/\u7167\u7247/\u58c1\u7eb8\u3002search_rewrite \u5e94\u9002\u5408\u56fe\u7247\u641c\u7d22\uff0c\u4fdd\u7559\u4e3b\u4f53\u3001\u573a\u666f\u3001\u98ce\u683c\u3001\u540c\u4e49\u8bcd\uff0c\u65b9\u4f4d\u5173\u7cfb\u6539\u5199\u6210\u540c\u6846/\u4e92\u52a8\u7b49\u53ef\u68c0\u7d22\u8868\u8fbe\u3002\n"
        "3) \u63d0\u5230\u56fe\u7247\u4e0d\u4e00\u5b9a\u662f image_search\uff1b\u5982\u679c\u662f\u5728\u95ee\u56fe\u7247\u5185\u5bb9\u3001\u5206\u6790\u56fe\u7247\u3001\u6bd4\u8f83\u56fe\u7247\uff0c\u9009 general_qa\u3002\n"
        "4) \u53ea\u586b\u6240\u9009 intent \u5bf9\u5e94\u7684 constraints\uff0c\u53e6\u4e00\u4e2a\u53ef\u4ee5\u4e3a null \u6216\u7a7a\u5bf9\u8c61\u3002\n"
        "5) \u4e0d\u786e\u5b9a\u65f6\u9009 general_qa \u5e76\u964d\u4f4e confidence\uff1b\u7f3a\u5fc5\u8981\u6761\u4ef6\u65f6\u8bbe\u7f6e needs_clarification\u3002\n"
        f"\u4e0a\u4e0b\u6587: {json.dumps(context, ensure_ascii=False)}\n"
        f"\u7528\u6237 query: {req.query}"
    )
    return prompt
    return (
        "你是多模态 RAG 系统的统一 query planner。只输出一个 JSON 对象，不要解释。\n"
        f"输出结构: {json.dumps(schema, ensure_ascii=False)}\n"
        "任务：一次性完成 intent routing 和检索 query/constraints 规划，避免后续再次调用 LLM。\n"
        "场景规则：\n"
        "1) general_qa：问答、总结网页/文档、解释、分析、对比、天气、推荐、理解用户上传图片/文档。search_rewrite 应适合网页/RAG 检索。\n"
        "2) image_search：用户明确要求按文本找外部图片/照片/壁纸。search_rewrite 应适合图片搜索，保留主体、场景、风格、同义词，方位关系改写成同框/互动等可检索表达。\n"
        "3) 提到图片不一定是 image_search；如果是在问图片内容、分析图片、比较图片，选 general_qa。\n"
        "4) 只填所选 intent 对应的 constraints，另一个可以为 null 或空对象。\n"
        "5) 不确定时选 general_qa 并降低 confidence；缺必要条件时设置 needs_clarification。\n"
        f"上下文: {json.dumps(context, ensure_ascii=False)}\n"
        f"用户 query: {req.query}"
    )


def _normalize_intent(value: Any) -> IntentType | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    if name in {"image_search", "search_image", "image", "image_query"}:
        return "image_search"
    if name in {"general_qa", "qa", "general_question", "text_qa"}:
        return "general_qa"
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in value.items()
        if isinstance(k, str) and isinstance(v, (str, int, float))
    }


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _image_constraints_from_obj(
    raw_query: str,
    search_rewrite: str,
    obj: Any,
    entities: dict[str, str],
) -> ImageSearchConstraints:
    data = obj if isinstance(obj, dict) else {}
    spatial_relations: list[SpatialRelationConstraint] = []
    for item in data.get("spatial_relations") or []:
        if not isinstance(item, dict):
            continue
        try:
            spatial_relations.append(
                SpatialRelationConstraint(
                    relation=str(item.get("relation") or "next_to"),
                    primary_subject=str(item.get("primary_subject") or ""),
                    secondary_subject=str(item.get("secondary_subject") or ""),
                )
            )
        except Exception:
            continue

    action_relations: list[ActionRelationConstraint] = []
    for item in data.get("action_relations") or []:
        if isinstance(item, dict):
            action_relations.append(
                ActionRelationConstraint(
                    subject=str(item.get("subject") or ""),
                    verb=str(item.get("verb") or ""),
                    object=str(item.get("object") or ""),
                )
            )

    object_relations: list[ObjectRelationConstraint] = []
    for item in data.get("object_relations") or []:
        if isinstance(item, dict):
            object_relations.append(
                ObjectRelationConstraint(
                    subject=str(item.get("subject") or ""),
                    relation=str(item.get("relation") or ""),
                    object=str(item.get("object") or ""),
                )
            )

    count = data.get("count") or entities.get("image_count")
    count = int(count) if isinstance(count, int) or (isinstance(count, str) and count.isdigit()) else None
    if count is not None:
        count = max(1, min(count, 12))

    synonyms_raw = data.get("subject_synonyms")
    subject_synonyms = {
        str(k): _str_list(v)
        for k, v in synonyms_raw.items()
    } if isinstance(synonyms_raw, dict) else {}

    return ImageSearchConstraints(
        raw_query=raw_query,
        search_rewrite=search_rewrite,
        subjects=_str_list(data.get("subjects")),
        attributes=_str_list(data.get("attributes")),
        subject_synonyms=subject_synonyms,
        style_terms=_str_list(data.get("style_terms")),
        exclude_terms=_str_list(data.get("exclude_terms")),
        count=count,
        landmark=str(data.get("landmark") or entities.get("landmark") or "").strip() or None,
        time_of_day=str(data.get("time_of_day") or entities.get("time_of_day") or "").strip() or None,
        must_have_all_subjects=bool(data.get("must_have_all_subjects", True)),
        spatial_relations=spatial_relations,
        action_relations=action_relations,
        object_relations=object_relations,
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=str(data.get("clarification_question") or "").strip() or None,
        parser_source="llm_planner",
    )


def _general_constraints_from_obj(raw_query: str, search_rewrite: str, obj: Any) -> GeneralQueryConstraints:
    data = obj if isinstance(obj, dict) else {}
    return GeneralQueryConstraints(
        raw_query=raw_query,
        search_rewrite=search_rewrite,
        city=str(data.get("city") or "").strip() or None,
        attributes=_str_list(data.get("attributes")),
        compare_targets=_str_list(data.get("compare_targets")),
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=str(data.get("clarification_question") or "").strip() or None,
        parser_source="llm_planner",
    )
