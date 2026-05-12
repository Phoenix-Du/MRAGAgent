from __future__ import annotations

import re

from app.models.schemas import ImageSearchConstraints

_STOPWORDS = {
    "我想",
    "我想要",
    "我要",
    "帮我",
    "请帮我",
    "请",
    "给我",
    "来点",
    "来几张",
    "几张",
    "一些",
    "一下",
    "看看",
    "找找",
}

_WEATHER_KEYWORDS = ("天气", "气温", "下雨", "降雨", "温度", "风力", "湿度", "体感", "空气质量")


def optimize_image_query(original_query: str, entities: dict[str, str]) -> str:
    """
    Convert conversational image intent query into retrieval-friendly terms.
    """
    landmark = (entities.get("landmark") or "").strip()
    time_of_day = (entities.get("time_of_day") or "").strip()
    style = (entities.get("style") or "").strip()

    parts: list[str] = []
    if time_of_day:
        parts.append(time_of_day)
    if landmark:
        parts.append(landmark)
    if style:
        parts.append(style)

    if parts:
        parts.append("照片")
        return _dedup_join(parts)

    # No useful entities; still do a lightweight cleanup.
    cleaned = _strip_common_fillers(original_query)
    if not cleaned:
        cleaned = original_query
    return _dedup_join([cleaned, "图片"])


def optimize_image_query_with_constraints(
    original_query: str,
    constraints: ImageSearchConstraints | None,
    entities: dict[str, str],
) -> str:
    """Legacy helper kept for compatibility; the image execution path consumes search_rewrite directly."""
    if constraints is not None and (constraints.search_rewrite or "").strip():
        return str(constraints.search_rewrite).strip()
    return optimize_image_query(original_query, entities)


def optimize_web_query(original_query: str) -> str:
    """
    Convert conversational web QA query into search-engine-friendly query terms.
    """
    cleaned = _strip_common_fillers(original_query)
    text = cleaned or original_query

    # If user is asking for latest info, add freshness hint.
    freshness_hint = ""
    if any(k in text for k in ("最新", "最近", "今年", "当前", "现在")):
        freshness_hint = "最新"

    # If query is comparative, add retrieval intent hints.
    compare_hint = ""
    if any(k in text for k in ("对比", "区别", "哪个好", "优缺点", "差异")):
        compare_hint = "对比 评测"

    # Keep the original semantic core and add lightweight hints.
    hints = [h for h in (freshness_hint, compare_hint) if h]
    if any(k in text for k in _WEATHER_KEYWORDS):
        # Weather queries should bias toward forecast terms instead of city-intro pages.
        hints.extend(["天气预报", "气温", "降雨", "风力", "湿度"])
    if not hints:
        return text
    return _dedup_join([text, *hints])


def _strip_common_fillers(text: str) -> str:
    out = text.strip()
    for w in sorted(_STOPWORDS, key=len, reverse=True):
        out = out.replace(w, " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _dedup_join(parts: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        token = p.strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return " ".join(out)
