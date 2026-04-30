from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

import httpx

from app.integrations.bridge_settings import bridge_settings
from app.models.schemas import (
    ActionRelationConstraint,
    GeneralQueryConstraints,
    ImageSearchConstraints,
    ObjectRelationConstraint,
    SpatialRelationConstraint,
)
from app.services.llm_json_client import (
    extract_json_obj as _shared_extract_json_obj,
    llm_env as _shared_llm_env,
    post_llm_json as _shared_post_llm_json,
)
from app.services.parser_cache import ParserCache


logger = logging.getLogger(__name__)

_FILLERS = {
    "给我",
    "来几张",
    "几张",
    "来点",
    "找几张",
    "帮我",
    "我想要",
    "我想",
    "照片",
    "图片",
    "图",
}

_SPATIAL_KEYWORDS: dict[str, str] = {
    "左边": "left_right",
    "左侧": "left_right",
    "右边": "right_left",
    "右侧": "right_left",
    "前面": "front_back",
    "后面": "back_front",
    "上面": "above_below",
    "下面": "below_above",
    "旁边": "left_right",
    "中间": "left_right",
}

_COLOR_WORDS = ("红", "黄", "蓝", "白", "黑", "灰", "金", "银", "粉", "紫", "绿", "棕", "橙")
_STATE_WORDS = ("大", "小", "高", "低", "长", "短", "新", "旧", "干", "湿", "胖", "瘦")

_IMAGE_PARSE_CACHE_TTL_SECONDS = 180
_GENERAL_PARSE_CACHE_TTL_SECONDS = 180
_image_parse_cache: ParserCache[ImageSearchConstraints] = ParserCache(
    ttl_seconds=_IMAGE_PARSE_CACHE_TTL_SECONDS,
    max_entries=bridge_settings.parser_cache_max_entries,
)
_general_parse_cache: ParserCache[GeneralQueryConstraints] = ParserCache(
    ttl_seconds=_GENERAL_PARSE_CACHE_TTL_SECONDS,
    max_entries=bridge_settings.parser_cache_max_entries,
)

_FILLERS.update(
    {
        "给我",
        "来几张",
        "找几张",
        "搜几张",
        "发几张",
        "看几张",
        "帮我",
        "我想要",
        "我想",
        "几张",
        "图片",
        "照片",
        "图",
    }
)
_SPATIAL_KEYWORDS.update(
    {
        "左边": "left_right",
        "左侧": "left_right",
        "右边": "right_left",
        "右侧": "right_left",
        "前面": "front_back",
        "后面": "back_front",
        "上面": "above_below",
        "下面": "below_above",
        "旁边": "next_to",
        "中间": "next_to",
    }
)
_COLOR_WORDS = _COLOR_WORDS + ("红", "黄", "蓝", "白", "黑", "灰", "金", "银", "粉", "紫", "绿", "棕", "橙")
_STATE_WORDS = _STATE_WORDS + ("大", "小", "高", "低", "长", "短", "新", "旧", "宽", "瘦", "胖")


async def _post_llm_json(
    *,
    base: str,
    api_key: str,
    payload: dict[str, Any],
    retries: int = 2,
) -> dict[str, Any] | None:
    return await _shared_post_llm_json(base=base, api_key=api_key, payload=payload, retries=retries)
    timeout = httpx.Timeout(
        connect=float(bridge_settings.request_timeout_seconds),
        read=float(bridge_settings.parser_read_timeout_seconds),
        write=float(bridge_settings.request_timeout_seconds),
        pool=float(bridge_settings.request_timeout_seconds),
    )
    last_exc: Exception | None = None
    for i in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(
                    f"{base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message", {}).get("content", ""))
            obj = _extract_json_obj(str(content))
            if obj:
                return obj
            logger.warning("parser_llm_bad_json attempt=%s content_prefix=%s", i + 1, str(content)[:240])
            return None
        except Exception as exc:
            last_exc = exc
            if i >= retries:
                break
    if last_exc:
        raise last_exc
    return None


def _llm_env() -> tuple[str, str, str]:
    return _shared_llm_env()
    api_key = (bridge_settings.qwen_api_key or "").strip() or (
        bridge_settings.openai_api_key or ""
    ).strip()
    base = (
        (bridge_settings.qwen_base_url or "").strip()
        or (bridge_settings.openai_base_url or "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = (bridge_settings.qwen_parser_model or "").strip() or "qwen3.5-plus"
    return api_key, base, model


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    return _shared_extract_json_obj(text)
    raw = text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    obj_text = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_subject(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^(一只|一条|一个|一位|一名|一头|只|条|个|位|名|头)", "", value)
    value = re.sub(r"(照片|图片|图|的|是)+$", "", value)
    return value.strip(" ，,。.；;:")


def _parse_count_from_text(text: str) -> int | None:
    m_real = re.search(r"(\d{1,2})\s*(张|个|幅|份)?", text)
    if m_real:
        return int(m_real.group(1))
    real_mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "几": 5,
    }
    for key, value in real_mapping.items():
        if f"{key}张" in text or f"{key}个" in text or (key == "几" and "几张" in text):
            return value
    m = re.search(r"(\d{1,2})\s*张", text)
    if m:
        return int(m.group(1))
    mapping = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "几": 5,
    }
    for key, value in mapping.items():
        if f"{key}张" in text or (key == "几" and "几张" in text):
            return value
    return None


def _strip_fillers(text: str) -> str:
    out = text
    for token in sorted(_FILLERS, key=len, reverse=True):
        out = out.replace(token, " ")
    return re.sub(r"\s+", " ", out).strip()


def _build_search_rewrite(constraints: ImageSearchConstraints) -> str:
    parts: list[str] = []
    if constraints.time_of_day:
        parts.append(constraints.time_of_day)
    if constraints.landmark:
        parts.append(constraints.landmark)
    for subject in constraints.subjects:
        parts.append(subject)
        parts.extend(constraints.subject_synonyms.get(subject, []))
    parts.extend(constraints.attributes)
    parts.extend(constraints.style_terms)
    if constraints.count and constraints.count > 1:
        parts.extend(["同框", "合影"])
    if len(constraints.subjects) >= 2 and constraints.must_have_all_subjects:
        parts.append("互动")
    if constraints.action_relations:
        parts.extend(rel.verb for rel in constraints.action_relations if rel.verb.strip())
    if constraints.object_relations:
        parts.extend(rel.relation for rel in constraints.object_relations if rel.relation.strip())
    parts.append("照片")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return " ".join(out) or _strip_fillers(constraints.raw_query)


def _heuristic_constraints(query: str, entities: dict[str, str]) -> ImageSearchConstraints:
    text = query.strip()
    subjects: list[str] = []
    spatial_relations: list[SpatialRelationConstraint] = []
    action_relations: list[ActionRelationConstraint] = []
    object_relations: list[ObjectRelationConstraint] = []

    real_pair_patterns = [
        (r"左(?:边|侧)是?(?P<a>[^，。？！,?.\s右]{1,12}).*?右(?:边|侧)是?(?P<b>[^，。？！,?.\s]{1,12})", "left_right"),
        (r"右(?:边|侧)是?(?P<a>[^，。？！,?.\s左]{1,12}).*?左(?:边|侧)是?(?P<b>[^，。？！,?.\s]{1,12})", "right_left"),
        (r"前(?:面)?是?(?P<a>[^，。？！,?.\s后]{1,12}).*?后(?:面)?是?(?P<b>[^，。？！,?.\s]{1,12})", "front_back"),
        (r"后(?:面)?是?(?P<a>[^，。？！,?.\s前]{1,12}).*?前(?:面)?是?(?P<b>[^，。？！,?.\s]{1,12})", "back_front"),
    ]
    for pattern, rel in real_pair_patterns:
        m_real = re.search(pattern, text)
        if not m_real:
            continue
        a = _clean_subject(m_real.group("a"))
        b = _clean_subject(m_real.group("b"))
        if a and b:
            subjects.extend([a, b])
            spatial_relations.append(
                SpatialRelationConstraint(relation=rel, primary_subject=a, secondary_subject=b)
            )

    for word, rel in _SPATIAL_KEYWORDS.items():
        if word not in text:
            continue
        m = re.search(
            rf"(?P<a>[^\s，。,.]{{1,12}})(?:在|于|是)?(?P<b>[^\s，。,.]{{1,12}})?(?:的)?{re.escape(word)}",
            text,
        )
        if m:
            a = _clean_subject(m.group("a") or "")
            b = _clean_subject(m.group("b") or "")
            if a and b:
                subjects.extend([a, b])
                spatial_relations.append(
                    SpatialRelationConstraint(
                        relation=rel,
                        primary_subject=a,
                        secondary_subject=b,
                    )
                )
        lr = re.search(
            rf"{re.escape(word)}(?:是)?(?P<left>[^，。,.]{{1,12}})",
            text,
        )
        if lr:
            s = _clean_subject(lr.group("left"))
            if s:
                subjects.append(s)
    pair_patterns = [
        (r"左(?:边|侧)?(?:是)?(?P<a>[^，。,.]{1,12})右(?:边|侧)?(?:是)?(?P<b>[^，。,.]{1,12})", "left_right"),
        (r"右(?:边|侧)?(?:是)?(?P<a>[^，。,.]{1,12})左(?:边|侧)?(?:是)?(?P<b>[^，。,.]{1,12})", "right_left"),
    ]
    for pattern, rel in pair_patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        a = _clean_subject(m.group("a"))
        b = _clean_subject(m.group("b"))
        if a and b:
            subjects.extend([a, b])
            spatial_relations.append(
                SpatialRelationConstraint(relation=rel, primary_subject=a, secondary_subject=b)
            )

    for m in re.finditer(
        r"(?P<subject>[^\s，。,.]{1,12}?)(?P<verb>[^\s，。,.]{1,4}(?:着|了|过))(?P<object>[^\s，。,.]{1,12})",
        text,
    ):
        subj = _clean_subject(m.group("subject"))
        verb = m.group("verb").strip()
        obj = _clean_subject(m.group("object"))
        if subj and verb and obj:
            subjects.extend([subj, obj])
            action_relations.append(
                ActionRelationConstraint(subject=subj, verb=verb, object=obj)
            )
            object_relations.append(
                ObjectRelationConstraint(subject=subj, relation=verb, object=obj)
            )

    count = _parse_count_from_text(text)
    entity_count = entities.get("image_count")
    if count is None and entity_count:
        count = _parse_count_from_text(entity_count)

    landmark = (entities.get("landmark") or "").strip() or None
    time_of_day = (entities.get("time_of_day") or "").strip() or None
    style = (entities.get("style") or "").strip()
    style_terms = [style] if style else []

    if not subjects:
        stripped = _strip_fillers(text)
        tokens = [t for t in re.split(r"[ ，,。.!！？、]+", stripped) if t]
        subjects = [_clean_subject(t) for t in tokens[:3] if _clean_subject(t)]

    dedup_subjects = list(dict.fromkeys([s for s in subjects if s]))
    extracted_attributes: list[str] = []
    cleaned_subjects: list[str] = []
    for s in dedup_subjects[:6]:
        attr = ""
        core = s
        if len(s) >= 2 and s[0] in _COLOR_WORDS + _STATE_WORDS:
            attr = s[0]
            core = s[1:]
        if core:
            cleaned_subjects.append(core)
        if attr:
            extracted_attributes.append(attr)
    constraints = ImageSearchConstraints(
        raw_query=text,
        subjects=list(dict.fromkeys(cleaned_subjects))[:6],
        count=count,
        landmark=landmark,
        time_of_day=time_of_day,
        attributes=list(dict.fromkeys(extracted_attributes)),
        style_terms=style_terms,
        spatial_relations=spatial_relations,
        action_relations=action_relations,
        object_relations=object_relations,
        parser_source="heuristic",
    )
    constraints.search_rewrite = _build_search_rewrite(constraints)
    return constraints


def _constraints_from_obj(query: str, obj: dict[str, Any], entities: dict[str, str]) -> ImageSearchConstraints:
    spatial_relations: list[SpatialRelationConstraint] = []
    for item in obj.get("spatial_relations") or []:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation") or "").strip()
        primary = _clean_subject(str(item.get("primary_subject") or item.get("left") or ""))
        secondary = _clean_subject(str(item.get("secondary_subject") or item.get("right") or ""))
        if relation and primary and secondary:
            try:
                spatial_relations.append(
                    SpatialRelationConstraint(
                        relation=relation, primary_subject=primary, secondary_subject=secondary
                    )
                )
            except Exception:
                continue

    object_relations: list[ObjectRelationConstraint] = []
    for item in obj.get("object_relations") or []:
        if not isinstance(item, dict):
            continue
        subj = _clean_subject(str(item.get("subject") or ""))
        relation = str(item.get("relation") or "").strip()
        obj_name = _clean_subject(str(item.get("object") or ""))
        if subj and relation and obj_name:
            object_relations.append(
                ObjectRelationConstraint(subject=subj, relation=relation, object=obj_name)
            )
    action_relations: list[ActionRelationConstraint] = []
    for item in obj.get("action_relations") or []:
        if not isinstance(item, dict):
            continue
        subj = _clean_subject(str(item.get("subject") or ""))
        verb = str(item.get("verb") or item.get("relation") or "").strip()
        obj_name = _clean_subject(str(item.get("object") or ""))
        if subj and verb and obj_name:
            action_relations.append(
                ActionRelationConstraint(subject=subj, verb=verb, object=obj_name)
            )

    count = obj.get("count")
    if not isinstance(count, int):
        count = _parse_count_from_text(entities.get("image_count") or "")
    if isinstance(count, int):
        count = max(1, min(count, 12))

    subjects = [
        _clean_subject(str(v))
        for v in (obj.get("subjects") or [])
        if isinstance(v, (str, int, float))
    ]
    attributes = [
        str(v).strip()
        for v in (obj.get("attributes") or [])
        if isinstance(v, (str, int, float)) and str(v).strip()
    ]
    style_terms = [
        str(v).strip()
        for v in (obj.get("style_terms") or [])
        if isinstance(v, (str, int, float)) and str(v).strip()
    ]
    exclude_terms = [
        str(v).strip()
        for v in (obj.get("exclude_terms") or [])
        if isinstance(v, (str, int, float)) and str(v).strip()
    ]
    subject_synonyms_raw = obj.get("subject_synonyms") or {}
    subject_synonyms: dict[str, list[str]] = {}
    if isinstance(subject_synonyms_raw, dict):
        for k, v in subject_synonyms_raw.items():
            key = _clean_subject(str(k))
            if not key:
                continue
            vals = [
                _clean_subject(str(x))
                for x in (v if isinstance(v, list) else [v])
                if _clean_subject(str(x))
            ]
            if vals:
                subject_synonyms[key] = list(dict.fromkeys(vals))
    constraints = ImageSearchConstraints(
        raw_query=query,
        subjects=list(dict.fromkeys([s for s in subjects if s])),
        attributes=attributes,
        subject_synonyms=subject_synonyms,
        style_terms=style_terms,
        exclude_terms=exclude_terms,
        count=count,
        landmark=str(obj.get("landmark") or entities.get("landmark") or "").strip() or None,
        time_of_day=str(obj.get("time_of_day") or entities.get("time_of_day") or "").strip() or None,
        must_have_all_subjects=bool(obj.get("must_have_all_subjects", True)),
        spatial_relations=spatial_relations,
        action_relations=action_relations,
        object_relations=object_relations,
        needs_clarification=bool(obj.get("needs_clarification", False)),
        clarification_question=str(obj.get("clarification_question") or "").strip() or None,
        parser_source="llm",
    )
    rewrite = str(obj.get("search_rewrite") or "").strip()
    constraints.search_rewrite = rewrite or _build_search_rewrite(constraints)
    return constraints


async def parse_image_search_constraints(
    query: str,
    entities: dict[str, str],
) -> ImageSearchConstraints:
    cache_key = hashlib.md5(
        f"{query}::{json.dumps(entities, ensure_ascii=False, sort_keys=True)}".encode("utf-8")
    ).hexdigest()
    now = time.time()
    cached = _image_parse_cache.get(cache_key, now)
    if cached:
        return cached

    api_key, base, model = _llm_env()
    heuristic = _heuristic_constraints(query, entities)
    if not api_key:
        _image_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _image_parse_cache.put(cache_key, heuristic, now)
        return heuristic

    schema = {
        "type": "object",
        "properties": {
            "subjects": {"type": "array", "items": {"type": "string"}},
            "subject_synonyms": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "count": {"type": ["integer", "null"]},
            "spatial_relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "relation": {"type": "string"},
                        "primary_subject": {"type": "string"},
                        "secondary_subject": {"type": "string"},
                    },
                    "required": ["relation", "primary_subject", "secondary_subject"],
                },
            },
            "action_relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "verb": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "verb", "object"],
                },
            },
            "exclude_terms": {"type": "array", "items": {"type": "string"}},
            "must_have_all_subjects": {"type": "boolean"},
            "needs_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "search_rewrite": {"type": "string"},
        },
        "required": ["subjects", "spatial_relations", "action_relations", "search_rewrite"],
    }
    prompt = (
        "你是图搜意图解析器。请严格按 JSON Schema 输出，不要输出任何解释。\n"
        f"JSON Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        "要求：\n"
        "1) subjects 不限于动物，任意实体都可。\n"
        "2) spatial_relations 使用泛化关系词，如 left_of/right_of/in_front_of/behind/on/under/inside/next_to。\n"
        "3) action_relations 使用原动词语义。\n"
        "4) search_rewrite 适合搜索引擎：保留核心实体、同义词、动作、场景；丢弃左右等方位词并改为同框/互动表达。\n"
        "5) subject_synonyms 为主体同义词扩展，尽量补常用别名。\n"
        "Few-shot 示例A: 用户'茶几左边一杯咖啡右边一本书' => subjects:[咖啡,书,茶几], spatial_relations:[(咖啡,left_of,书)], search_rewrite:'咖啡 书 茶几 同框'\n"
        "Few-shot 示例B: 用户'一只大雁追着一只老鹰飞' => action_relations:[(大雁,追逐,老鹰)]\n"
        "Few-shot 示例C: 用户'海边散步，不要人，只要狗' => subjects:[狗], exclude_terms:[人], search_rewrite:'海边 散步 狗'\n"
        f"用户请求: {query}\n"
        f"已有轻解析实体: {json.dumps(entities, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 650,
    }
    try:
        obj = await _post_llm_json(base=base, api_key=api_key, payload=payload, retries=2)
        if not obj:
            logger.warning("image_query_parser_bad_json query=%s", query)
            _image_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
            _image_parse_cache.put(cache_key, heuristic, now)
            return heuristic
        parsed = _constraints_from_obj(query, obj, entities)
        _image_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _image_parse_cache.put(cache_key, parsed, now)
        return parsed
    except Exception:
        logger.exception("image_query_parser_failed query=%s", query)
        _image_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _image_parse_cache.put(cache_key, heuristic, now)
        return heuristic


def constraints_to_prompt_text(constraints: ImageSearchConstraints | None) -> str:
    if constraints is None:
        return ""
    parts: list[str] = []
    if constraints.subjects:
        parts.append(f"主体: {', '.join(constraints.subjects)}")
    if constraints.count:
        parts.append(f"数量约束: {constraints.count}")
    if constraints.spatial_relations:
        rel_text = []
        for rel in constraints.spatial_relations:
            rel_text.append(f"{rel.primary_subject}-{rel.relation}-{rel.secondary_subject}")
        parts.append(f"空间约束: {'; '.join(rel_text)}")
    if constraints.object_relations:
        rel_text = []
        for rel in constraints.object_relations:
            rel_text.append(f"{rel.subject}-{rel.relation}-{rel.object}")
        parts.append(f"对象关系: {'; '.join(rel_text)}")
    if constraints.action_relations:
        rel_text = []
        for rel in constraints.action_relations:
            rel_text.append(f"{rel.subject}-{rel.verb}-{rel.object}")
        parts.append(f"动作关系: {'; '.join(rel_text)}")
    if constraints.exclude_terms:
        parts.append(f"排除: {', '.join(constraints.exclude_terms)}")
    if constraints.must_have_all_subjects:
        parts.append("要求所有主体同时出现")
    return "\n".join(parts)


def _extract_city(text: str) -> str | None:
    direct = re.search(r"([\u4e00-\u9fa5]{2,8}(?:市|区|县|州|盟))", text)
    if direct:
        return direct.group(1)
    for c in ("北京", "上海", "广州", "深圳", "杭州", "武汉", "成都", "重庆", "南京", "西安", "苏州", "天津"):
        if c in text:
            return c
    return None


def _heuristic_general_constraints(query: str) -> GeneralQueryConstraints:
    city = _extract_city(query)
    compare_targets: list[str] = []
    if any(k in query for k in ("对比", "区别", "哪个好", "差异")):
        parts = [p.strip() for p in re.split(r"和|与|跟|vs|VS", query) if p.strip()]
        compare_targets = parts[:2]
    rewritten = _strip_fillers(query)
    return GeneralQueryConstraints(
        raw_query=query,
        city=city,
        compare_targets=compare_targets,
        search_rewrite=rewritten or query,
        parser_source="heuristic",
    )


async def parse_general_query_constraints(query: str) -> GeneralQueryConstraints:
    cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()
    now = time.time()
    cached = _general_parse_cache.get(cache_key, now)
    if cached:
        return cached

    api_key, base, model = _llm_env()
    heuristic = _heuristic_general_constraints(query)
    if not api_key:
        _general_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _general_parse_cache.put(cache_key, heuristic, now)
        return heuristic

    prompt = (
        "你是通用问题解析器。请把用户问题解析成结构化 JSON。"
        "重点识别：search_rewrite、city、attributes、compare_targets、needs_clarification、clarification_question。"
        "search_rewrite 应更适合搜索或检索，不要改变用户真实意图。"
        "如果是天气类问题且缺城市，可以设置 needs_clarification=true。"
        "只输出一个 JSON 对象，不要解释。\n"
        f"用户问题: {query}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 450,
    }
    try:
        obj = await _post_llm_json(base=base, api_key=api_key, payload=payload, retries=1)
        if not obj:
            logger.warning("general_query_parser_bad_json query=%s", query)
            _general_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
            _general_parse_cache.put(cache_key, heuristic, now)
            return heuristic
        parsed = GeneralQueryConstraints(
            raw_query=query,
            search_rewrite=str(obj.get("search_rewrite") or heuristic.search_rewrite or query).strip(),
            city=str(obj.get("city") or heuristic.city or "").strip() or None,
            attributes=[
                str(v).strip()
                for v in (obj.get("attributes") or [])
                if isinstance(v, (str, int, float)) and str(v).strip()
            ],
            compare_targets=[
                str(v).strip()
                for v in (obj.get("compare_targets") or [])
                if isinstance(v, (str, int, float)) and str(v).strip()
            ],
            needs_clarification=bool(obj.get("needs_clarification", False)),
            clarification_question=str(obj.get("clarification_question") or "").strip() or None,
            parser_source="llm",
        )
        _general_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _general_parse_cache.put(cache_key, parsed, now)
        return parsed
    except Exception:
        logger.exception("general_query_parser_failed query=%s", query)
        _general_parse_cache.max_entries = bridge_settings.parser_cache_max_entries
        _general_parse_cache.put(cache_key, heuristic, now)
        return heuristic
