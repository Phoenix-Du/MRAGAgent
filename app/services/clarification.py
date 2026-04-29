from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models.schemas import IntentType


@dataclass
class ClarificationDecision:
    should_ask: bool
    question: str = ""
    scenario: str = ""
    missing_slots: list[str] | None = None
    rewritten_query: str | None = None


WEATHER_KEYWORDS = ("天气", "气温", "下雨", "降雨", "温度", "风力", "空气质量")
GENERIC_IMAGE_PATTERNS = (
    "来几张图",
    "来几张图片",
    "给我几张图",
    "搜几张图",
    "找几张图",
)


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


def should_clarify(
    *,
    query: str,
    intent: IntentType,
    entities: dict[str, str],
    preferences: dict[str, Any],
) -> ClarificationDecision:
    # Weather-style question should ask city if missing.
    if intent == "general_qa" and detect_weather_scenario(query):
        city = entities.get("city") or extract_city(query)
        default_city = str(preferences.get("default_city") or "").strip()
        if city:
            rewritten = f"{city} {query}"
            return ClarificationDecision(False, scenario="weather", rewritten_query=rewritten)
        if default_city:
            rewritten = f"{default_city} {query}"
            return ClarificationDecision(False, scenario="weather", rewritten_query=rewritten)
        return ClarificationDecision(
            True,
            question="你想查询哪个城市的天气？",
            scenario="weather",
            missing_slots=["city"],
        )

    # Image search can be optional for landmark, but too generic requests should be clarified.
    if intent == "image_search":
        landmark = entities.get("landmark", "").strip()
        generic = any(p in query for p in GENERIC_IMAGE_PATTERNS)
        too_short = len(query.strip()) <= 6
        if not landmark and (generic or too_short):
            return ClarificationDecision(
                True,
                question="你想看哪个地点或主体的图片？",
                scenario="image_search",
                missing_slots=["landmark"],
            )
    return ClarificationDecision(False)


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
    """
    if not pending:
        return False, None, None, None
    scenario = str(pending.get("scenario") or "")
    original_query = str(pending.get("original_query") or "").strip()
    if not scenario or not original_query:
        return False, None, None, None

    if scenario == "weather":
        city = extract_city(query) or str(preferences.get("default_city") or "").strip()
        if city:
            merged = original_query if city in original_query else f"{city} {original_query}"
            return True, merged, None, "general_qa"
        return False, None, pending, None

    if scenario == "image_search":
        # Treat user's reply as subject supplement.
        subject = query.strip()
        if subject:
            return True, f"{original_query} {subject}", None, "image_search"
        return False, None, pending, None

    return False, None, None, None
