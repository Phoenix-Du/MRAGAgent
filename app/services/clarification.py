from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from app.models.schemas import GeneralQueryConstraints, ImageSearchConstraints, IntentType


@dataclass
class ClarificationDecision:
    should_ask: bool
    question: str = ""
    scenario: str = ""
    missing_slots: list[str] | None = None
    rewritten_query: str | None = None
    source: str = ""
    constraints_kind: str | None = None


@dataclass
class ClarificationSignal:
    needed: bool
    question: str = ""
    scenario: str = ""
    missing_slots: list[str] | None = None
    source: str = ""
    constraints_kind: str | None = None


WEATHER_KEYWORDS = ("天气", "气温", "下雨", "降雨", "温度", "风力", "空气质量")
GENERIC_IMAGE_PATTERNS = (
    "来几张图",
    "来几张图片",
    "给我几张图",
    "搜几张图",
    "找几张图",
)

DEFAULT_GENERAL_QUESTION = "请再补充一下你想查询的具体条件。"
DEFAULT_IMAGE_QUESTION = "你想看哪个地点或主体的图片？"
DEFAULT_WEATHER_QUESTION = "你想查询哪个城市的天气？"


def detect_weather_scenario(query: str) -> bool:
    q = query.strip()
    return any(k in q for k in WEATHER_KEYWORDS)


def extract_city(text: str) -> str | None:
    q = text.strip()
    direct = re.search(r"([\u4e00-\u9fa5]{2,8}(?:市|区|县|州|盟))", q)
    if direct:
        return direct.group(1)
    # handle frequent city names without suffix
    hot = (
        "北京",
        "上海",
        "广州",
        "深圳",
        "杭州",
        "武汉",
        "成都",
        "重庆",
        "南京",
        "西安",
        "苏州",
        "天津",
    )
    for c in hot:
        if c in q:
            return c
    return None


def decide_clarification(
    *,
    query: str,
    intent: IntentType,
    entities: dict[str, str],
    preferences: dict[str, Any],
    image_constraints: ImageSearchConstraints | None = None,
    general_constraints: GeneralQueryConstraints | None = None,
) -> ClarificationDecision:
    """
    Central clarification decision point.

    LLM planner/parser constraints are the primary signal. Local logic only resolves
    deterministic user preferences and covers obviously underspecified requests.
    """
    signal = _signal_from_constraints(
        intent=intent,
        query=query,
        image_constraints=image_constraints,
        general_constraints=general_constraints,
    )

    if intent == "general_qa" and detect_weather_scenario(query):
        city = _city_from_general_sources(query, entities, general_constraints)
        preferred_location = _preferred_location(preferences)
        if city:
            return ClarificationDecision(
                False,
                scenario="general_qa",
                rewritten_query=None if city in query else f"{city} {query}",
                source="deterministic_fallback",
                constraints_kind="general",
            )
        if preferred_location:
            return ClarificationDecision(
                False,
                scenario="general_qa",
                rewritten_query=None if preferred_location in query else f"{preferred_location} {query}",
                source="deterministic_fallback",
                constraints_kind="general",
            )

    if signal.needed:
        question = signal.question or _default_question(
            intent=intent,
            query=query,
            missing_slots=signal.missing_slots or [],
        )
        return ClarificationDecision(
            True,
            question=question,
            scenario=signal.scenario or intent,
            missing_slots=signal.missing_slots or [],
            source=signal.source,
            constraints_kind=signal.constraints_kind,
        )

    fallback = _deterministic_fallback(
        query=query,
        intent=intent,
        entities=entities,
        preferences=preferences,
        image_constraints=image_constraints,
        general_constraints=general_constraints,
    )
    if fallback is not None:
        return fallback

    return ClarificationDecision(False)


def build_pending_state(
    decision: ClarificationDecision,
    *,
    original_query: str,
) -> dict[str, Any]:
    route = _canonical_scenario(decision.scenario) or decision.scenario
    return {
        "type": "clarification",
        "route": route,
        "original_query": original_query.strip(),
        "question": decision.question,
        "missing": decision.missing_slots or [],
        "created_at": int(time.time()),
    }


def should_clarify(
    *,
    query: str,
    intent: IntentType,
    entities: dict[str, str],
    preferences: dict[str, Any],
) -> ClarificationDecision:
    # Backward-compatible wrapper for older callers/tests.
    return decide_clarification(
        query=query,
        intent=intent,
        entities=entities,
        preferences=preferences,
    )


def maybe_resolve_pending(
    *,
    query: str,
    pending: dict[str, Any] | None,
    preferences: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any] | None, str | None]:
    """
    Returns:
    - resolved: whether the pending clarification is resolved
    - rewritten_query: merged query if resolved
    - next_pending: pending state to keep if unresolved
    - resolved_intent: canonical intent hint when resolved
    """
    if not pending:
        return False, None, None, None
    scenario = _canonical_scenario(str(pending.get("route") or pending.get("scenario") or ""))
    original_query = str(pending.get("original_query") or "").strip()
    if not scenario or not original_query:
        return False, None, None, None

    reply = query.strip()
    if not reply:
        return False, None, pending, None

    missing_slots = [
        str(slot).strip()
        for slot in (pending.get("missing") or pending.get("missing_slots") or [])
        if str(slot).strip()
    ]

    if _pending_expects_city(
        scenario=scenario,
        original_query=original_query,
        missing_slots=missing_slots,
    ):
        city = extract_city(reply) or _preferred_location(preferences)
        if city:
            merged = original_query if city in original_query else f"{city} {original_query}"
            return True, merged, None, "general_qa"
        return False, None, pending, None

    if scenario == "general_qa":
        return True, f"{original_query} {reply}", None, "general_qa"

    if scenario == "image_search":
        return True, f"{original_query} {reply}", None, "image_search"

    return False, None, None, None


def _signal_from_constraints(
    *,
    intent: IntentType,
    query: str,
    image_constraints: ImageSearchConstraints | None,
    general_constraints: GeneralQueryConstraints | None,
) -> ClarificationSignal:
    if intent == "image_search" and image_constraints is not None:
        if image_constraints.needs_clarification:
            missing_slots = _infer_image_missing_slots(image_constraints)
            return ClarificationSignal(
                True,
                question=image_constraints.clarification_question or "",
                scenario="image_search",
                missing_slots=missing_slots,
                source=image_constraints.parser_source or "constraints",
                constraints_kind="image",
            )
        return ClarificationSignal(False)

    if intent == "general_qa" and general_constraints is not None:
        if general_constraints.needs_clarification:
            missing_slots = ["city"] if detect_weather_scenario(query) and not general_constraints.city else []
            if not missing_slots:
                return ClarificationSignal(False)
            return ClarificationSignal(
                True,
                question=general_constraints.clarification_question or "",
                scenario="general_qa",
                missing_slots=missing_slots,
                source=general_constraints.parser_source or "constraints",
                constraints_kind="general",
            )
    return ClarificationSignal(False)


def _deterministic_fallback(
    *,
    query: str,
    intent: IntentType,
    entities: dict[str, str],
    preferences: dict[str, Any],
    image_constraints: ImageSearchConstraints | None,
    general_constraints: GeneralQueryConstraints | None,
) -> ClarificationDecision | None:
    if intent == "general_qa" and detect_weather_scenario(query):
        city = _city_from_general_sources(query, entities, general_constraints)
        preferred_location = _preferred_location(preferences)
        if city or preferred_location:
            return None
        return ClarificationDecision(
            True,
            question=DEFAULT_WEATHER_QUESTION,
            scenario="general_qa",
            missing_slots=["city"],
            source="deterministic_fallback",
            constraints_kind="general",
        )

    if intent == "image_search":
        if _image_has_retrieval_anchor(image_constraints, entities):
            return None
        generic = any(p in query for p in GENERIC_IMAGE_PATTERNS)
        too_short = len(query.strip()) <= 6
        if generic or too_short:
            return ClarificationDecision(
                True,
                question=DEFAULT_IMAGE_QUESTION,
                scenario="image_search",
                missing_slots=["subject_or_landmark"],
                source="deterministic_fallback",
                constraints_kind="image",
            )
    return None


def _city_from_general_sources(
    query: str,
    entities: dict[str, str],
    constraints: GeneralQueryConstraints | None,
) -> str | None:
    entity_city = str(entities.get("city") or "").strip()
    if entity_city:
        return entity_city
    if constraints and constraints.city:
        return constraints.city
    return extract_city(query)


def _preferred_location(preferences: dict[str, Any]) -> str:
    profile = preferences.get("profile")
    if isinstance(profile, dict):
        location = str(profile.get("location") or "").strip()
        if location:
            return location
    # Backward-compatible read for old local memory; new writes should use profile.location.
    return str(preferences.get("default_city") or preferences.get("location") or "").strip()


def _image_has_retrieval_anchor(
    constraints: ImageSearchConstraints | None,
    entities: dict[str, str],
) -> bool:
    if constraints is not None:
        if constraints.subjects or constraints.landmark or constraints.attributes or constraints.style_terms:
            return True
        if constraints.spatial_relations or constraints.action_relations or constraints.object_relations:
            return True
    for key in ("subject", "landmark", "style", "time_of_day", "object"):
        if str(entities.get(key) or "").strip():
            return True
    return False


def _infer_image_missing_slots(constraints: ImageSearchConstraints) -> list[str]:
    if not _image_has_retrieval_anchor(constraints, {}):
        return ["subject_or_landmark"]
    return []


def _default_question(
    *,
    intent: IntentType,
    query: str,
    missing_slots: list[str],
) -> str:
    if intent == "general_qa" and ("city" in missing_slots or detect_weather_scenario(query)):
        return DEFAULT_WEATHER_QUESTION
    if intent == "image_search":
        return DEFAULT_IMAGE_QUESTION
    return DEFAULT_GENERAL_QUESTION


def _canonical_scenario(value: str) -> str | None:
    if value in {"weather", "general_constraints", "general_qa"}:
        return "general_qa"
    if value in {"image_constraints", "image_search"}:
        return "image_search"
    return None


def _pending_expects_city(
    *,
    scenario: str,
    original_query: str,
    missing_slots: list[str],
) -> bool:
    return scenario == "general_qa" and (
        "city" in missing_slots or detect_weather_scenario(original_query)
    )
