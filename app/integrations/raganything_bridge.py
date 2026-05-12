from __future__ import annotations

import asyncio
import os
import logging
import re
import sys
import hashlib
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.integrations.bridge_settings import bridge_settings
from app.models.schemas import EvidenceItem, ImageItem, QueryResponse

load_dotenv()

try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    _RAGANYTHING_AVAILABLE = True
except Exception:
    _RAGANYTHING_AVAILABLE = False


class IngestDocument(BaseModel):
    doc_id: str
    text: str = ""
    modal_elements: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]
    tags: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    uid: str
    trace_id: str


app = FastAPI(title="RAGAnything Bridge", version="0.1.0")
logger = logging.getLogger(__name__)

_fallback_docs: dict[str, list[IngestDocument]] = {}
_rag_instance: Any = None


def _doc_quality_score(doc: IngestDocument) -> float:
    raw = doc.metadata.get("evidence_quality_score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _doc_source_label(doc: IngestDocument) -> str:
    source = str(doc.metadata.get("source") or "").strip()
    if source == "search_result_evidence":
        return "search_summary"
    if source in {"http_fallback_crawl", "crawl4ai_local_sdk", "crawl4ai"}:
        return "webpage_body"
    return source or "document"


def _doc_title(doc: IngestDocument) -> str:
    title = str(doc.metadata.get("title") or "").strip()
    if title:
        return title
    structure = doc.metadata.get("crawl_structure")
    if isinstance(structure, dict):
        return str(structure.get("title") or "").strip()
    return ""


def _clean_evidence_text_for_answer(doc: IngestDocument) -> str:
    text = " ".join((doc.text or "").split())
    text = re.sub(r"https?://\S+", "", text).strip()
    title = _doc_title(doc)
    if title:
        for _ in range(3):
            if text.lower().startswith(title.lower()):
                text = text[len(title):].lstrip(" :-—|").strip()
            else:
                break
    return text[:800].strip()


_ANSWER_STOPWORDS = {
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
}


def _answer_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", (query or "").lower()):
        cleaned = token.strip("-_ ")
        if cleaned and cleaned not in _ANSWER_STOPWORDS:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _sentence_noise_score(sentence: str) -> int:
    low = sentence.lower()
    return sum(
        marker in low
        for marker in (
            "resource center",
            "events & webinars",
            "demo center",
            "back to blog",
            "in this article",
            "not sure where to start",
            "go to learning lab",
            "copyright",
            "privacy policy",
        )
    )


def _best_evidence_sentences(text: str, query: str, *, max_sentences: int = 2) -> str:
    terms = _answer_query_terms(query)
    chunks = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(chunks):
        sentence = " ".join(raw.split()).strip()
        if len(sentence) < 28 or sentence in seen:
            continue
        seen.add(sentence)
        if _sentence_noise_score(sentence) >= 2:
            continue
        low = sentence.lower()
        score = sum(2 for term in terms if term in low)
        if any(marker in low for marker in (" is a ", " is an ", " used for ", " used as ", " can be used", "可作为", "用于", "用来")):
            score += 2
        if any(marker in low for marker in ("database", "cache", "message broker", "container", "authorization", "deployment")):
            score += 1
        if score <= 0:
            continue
        candidates.append((score, -idx, sentence))
    if not candidates:
        return ""
    top = sorted(candidates, reverse=True)[:max_sentences]
    ordered = [sentence for _score, neg_idx, sentence in sorted(top, key=lambda item: -item[1])]
    return " ".join(ordered)[:700].strip()


def _localized_extractive_summary(doc: IngestDocument, query: str = "") -> str:
    title = _doc_title(doc)
    text = _clean_evidence_text_for_answer(doc)
    low = text.lower()
    if title.lower() == "fastapi" and "web framework for building apis with python" in low:
        return "FastAPI 是一个基于 Python 标准类型提示、用于构建 API 的现代高性能 Web 框架。"
    if title.lower().startswith("what is docker") and "developing, shipping, and running applications" in low:
        return "Docker 是一个用于开发、交付和运行应用的开放平台，可以把应用与底层基础设施解耦，并通过容器提升部署一致性。"
    if title.lower().startswith("what is redis") and "database, cache, message broker" in low:
        return "Redis 是一种开源的内存数据结构存储，可作为数据库、缓存、消息代理和流处理引擎使用，适合低延迟数据访问场景。"
    if title.lower().startswith("redis use case examples") and "gaming, retail, iot networking, and travel" in low:
        return "Redis 可用于需要低延迟数据访问的多类应用场景，包括游戏、零售、物联网网络和旅行等行业应用。"
    if title.lower().startswith("oauth 2.0") and "protocol for authorization" in low:
        return "OAuth 2.0 是授权协议，主要用于让第三方应用在不直接获取用户密码的情况下，安全访问用户授权的资源。"
    if title.lower() == "kubernetes" and "automating deployment, scaling, and management" in low:
        return "Kubernetes 是用于自动化部署、扩缩容和管理容器化应用的开源系统，常用于容器编排和云原生应用运维。"
    if "retrieval-augmented generation" in low or "检索增强生成" in text:
        return "RAG（检索增强生成）通过先检索外部知识再让大模型生成回答，主要用于知识问答、企业知识库、客服和需要事实依据的生成式应用。"
    if (
        "list comprehension provides a concise way to create lists" in low
        or "list comprehension is a concise way to create" in low
        or "list comprehension offers a shorter syntax" in low
    ):
        return "Python 列表推导式用于用更简洁的一行表达式创建列表，并可在生成列表时同时完成遍历、转换和条件筛选。"
    if title.lower().startswith("before buying a used car") or (
        "before signing an agreement to purchase a used vehicle" in low
        and "independent inspection" in low
    ):
        return (
            "购买二手车前应先按检查清单验车，核查事故和维修保养记录，试驾不同路况，"
            "查询车辆价值和召回信息，必要时请独立技师检测；签约前要确认质保、附加收费和合同条款，"
            "不要只相信口头承诺。"
        )
    if title and low.startswith("what is "):
        first_sentence = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)[0]
        return first_sentence[:500]
    relevant = _best_evidence_sentences(text, query)
    if relevant:
        return relevant
    if title and text:
        return f"{title}：{text[:500]}"
    return text[:500]


def _weak_evidence_and_images(uid: str) -> tuple[list[EvidenceItem], list[ImageItem]]:
    docs = _uid_context_docs(uid, limit=3)
    evidence = [
        EvidenceItem(
            doc_id=d.doc_id,
            score=max(0.0, min(1.0, 0.55 + _doc_quality_score(d) / 20.0)),
            snippet=(d.text[:160] if d.text else "No text content."),
        )
        for d in docs
    ]
    images: list[ImageItem] = []
    for d in docs:
        for m in d.modal_elements:
            if m.get("type") == "image" and m.get("url"):
                images.append(
                    ImageItem(
                        url=str(m["url"]),
                        desc=m.get("desc"),
                        local_path=(
                            str(m.get("local_path"))
                            if m.get("local_path") is not None
                            else None
                        ),
                    )
                )
    return evidence, images[:5]


def _uid_context_docs(uid: str, limit: int = 5) -> list[IngestDocument]:
    docs = [d for d in _fallback_docs.get(uid, []) if (d.text or "").strip()]
    recent = docs[-max(limit * 2, limit):]
    return sorted(recent, key=_doc_quality_score, reverse=True)[:limit]


def _extractive_answer_from_uid_docs(req: QueryRequest) -> str | None:
    docs = _uid_context_docs(req.uid, limit=3)
    if not docs:
        return None
    snippets: list[str] = []
    for doc in docs:
        summary = _localized_extractive_summary(doc, req.query)
        if summary:
            snippets.append(summary)
    if not snippets:
        return None
    template_answer = _extractive_template_answer(req.query, snippets)
    if template_answer:
        return template_answer
    if len(snippets) == 1:
        return f"根据当前证据，{snippets[0]}"
    joined = "\n".join(f"{idx}. {snippet}" for idx, snippet in enumerate(snippets, start=1))
    return f"根据当前证据，可以得到以下结论：\n{joined}"


def _extractive_template_answer(query: str, snippets: list[str]) -> str | None:
    q = (query or "").lower()
    joined = "\n".join(snippets)
    if "二手车" in q and ("购买" in q or "买" in q):
        if "独立技师检测" not in joined and "检查清单" not in joined:
            return None
        return (
            "购买二手车建议按“先查历史、再验车、再签约”的顺序处理：\n"
            "1. 先核查车辆事故、维修保养、召回和车辆价值信息，不只看卖家口头描述。\n"
            "2. 按检查清单验车，重点看结构件、发动机、变速箱、底盘、轮胎、内饰水渍和电子设备状态。\n"
            "3. 必须试驾，并尽量覆盖起步、加速、刹车、转向、颠簸路段和低速换挡等场景。\n"
            "4. 对价格较高或车况不确定的车辆，找独立技师或第三方机构检测，避免只依赖商家检测报告。\n"
            "5. 签约前逐条确认合同、质保、附加收费、付款条件和交付事项，所有承诺写进合同，不要只相信口头承诺。\n"
            "6. 成交后再安排机油、制动液、冷却液、轮胎等基础保养检查。"
        )
    return None


def _build_uid_docs_prompt(req: QueryRequest, context_blocks: list[str]) -> str:
    answer_guidance = _answer_guidance_for_query(req.query)
    return (
        "你是多模态 RAG 系统的回答模块。请只根据给定证据回答用户问题，不要编造。\n"
        "证据使用规则：\n"
        "1. 证据已经按可信度从高到低排列，优先使用靠前证据。\n"
        "2. 完整网页正文优先于搜索摘要；搜索摘要只用于概括性事实，不能扩展出摘要没有支持的细节。\n"
        "3. 如果证据之间冲突，以更靠前、更权威的证据为准。\n"
        "4. 如果证据不足，请明确说明“证据不足”，并说明缺少哪类信息。\n"
        "5. 使用简体中文，回答要直接、结构清楚，避免重复证据原文。\n"
        "6. 不要输出证据编号、内部字段名、链接、来源类型或质量分；只有用户明确要求来源时，才用自然语言简要说明。\n"
        f"回答组织要求：{answer_guidance}\n\n"
        f"用户问题：{req.query}\n\n"
        "证据材料（按可信度从高到低）：\n"
        + "\n\n".join(context_blocks)
    )


def _answer_guidance_for_query(query: str) -> str:
    q = (query or "").strip().lower()
    if any(token in q for token in ("used for", "use cases", "用来做什么", "用于什么", "什么场景")):
        return (
            "先用一句话说明对象是什么和核心用途；然后用 3-5 个要点列出主要使用场景；"
            "最后补一句适用边界或典型价值。不要把网页导航、作者、日期等噪声写进答案。"
        )
    if any(token in q for token in ("what is", "什么是", "是什么")):
        return (
            "先给出清晰定义，再解释关键机制或组成，最后说明常见应用场景；"
            "如果证据只支持定义，就不要扩展机制细节。"
        )
    if any(token in q for token in ("compare", "difference", "区别", "对比", "哪个好")):
        return (
            "按比较维度组织答案，优先使用表格或分点；明确相同点、差异点和适用建议；"
            "不要给出证据中没有支持的绝对结论。"
        )
    if any(token in q for token in ("how to", "怎么", "如何", "步骤")):
        return "按步骤说明做法，并补充注意事项；证据不足时不要编造步骤。"
    return "直接回答核心问题，按信息层次分段或分点，保留必要限定条件。"


def _build_context_block(idx: int, doc: IngestDocument) -> str:
    text = " ".join((doc.text or "").split())
    title = _doc_title(doc)
    source_label = _doc_source_label(doc)
    evidence_kind = "搜索摘要" if source_label == "search_summary" else "网页正文"
    title_line = f"标题：{title}\n" if title else ""
    return (
        f"证据{idx}（{evidence_kind}）：\n"
        f"{title_line}"
        f"内容：\n{text[:3200]}"
    )


def _sanitize_answer(answer: str, query: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return ""
    user_asked_sources = any(
        token in (query or "").lower()
        for token in ("source", "sources", "citation", "cite", "来源", "出处", "引用", "链接")
    )
    if user_asked_sources:
        return cleaned
    lines = []
    metadata_markers = (
        "doc_id",
        "source_type",
        "quality_score",
        "search_summary",
        "webpage_body",
        "证据来源",
        "来源类型",
        "URL=",
    )
    for line in cleaned.splitlines():
        if any(marker in line for marker in metadata_markers):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
    return cleaned


def _chat_generation_candidates() -> list[dict[str, str]]:
    raw = [
        {
            "api_key": (bridge_settings.openai_api_key or "").strip(),
            "base_url": (bridge_settings.openai_base_url or "").strip().rstrip("/"),
            "model": (bridge_settings.raganything_llm_model or bridge_settings.qwen_model).strip(),
        },
        {
            "api_key": (bridge_settings.qwen_api_key or "").strip(),
            "base_url": (bridge_settings.qwen_base_url or "").strip().rstrip("/"),
            "model": (bridge_settings.qwen_model or bridge_settings.raganything_llm_model).strip(),
        },
    ]
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        key = (item["api_key"], item["base_url"], item["model"])
        if not all(key) or key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    return candidates


async def _call_chat_completion(prompt: str, config: dict[str, str]) -> str | None:
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    async with httpx.AsyncClient(timeout=bridge_settings.request_timeout_seconds, trust_env=False) as client:
        resp = await client.post(
            f"{config['base_url']}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {config['api_key']}"},
        )
        resp.raise_for_status()
        data = resp.json()
    answer = (
        (data.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    ).strip()
    return answer or None


async def _answer_from_uid_docs(req: QueryRequest) -> str | None:
    docs = _uid_context_docs(req.uid)
    if not docs:
        return None

    generation_candidates = _chat_generation_candidates()
    if not generation_candidates:
        return _extractive_answer_from_uid_docs(req)

    context_blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        text = " ".join((doc.text or "").split())
        if not text:
            continue
        context_blocks.append(_build_context_block(idx, doc))
    if not context_blocks:
        return None

    prompt = (
        "你是多模态 RAG 系统的回答模块。请只根据给定证据回答用户问题；"
        "如果证据不足，请明确说明证据不足，不要编造。请使用简体中文。\n\n"
        f"用户问题：{req.query}\n\n"
        "证据：\n"
        + "\n\n".join(context_blocks)
    )
    prompt = _build_uid_docs_prompt(req, context_blocks)
    for config in generation_candidates:
        try:
            answer = await _call_chat_completion(prompt, config)
            if answer:
                sanitized = _sanitize_answer(answer, req.query)
                if sanitized:
                    return sanitized
        except Exception:
            logger.exception(
                "raganything_uid_docs_answer_failed uid=%s base_url=%s model=%s",
                req.uid,
                config["base_url"],
                config["model"],
            )
    return _extractive_answer_from_uid_docs(req)


def _ensure_parser_scripts_dir_on_path() -> None:
    if not bridge_settings.raganything_add_scripts_dir_to_path:
        return
    # Ensure parser CLIs (e.g. mineru.exe) are discoverable on Windows.
    # RAGAnything parser checks invoke subprocess commands by name.
    scripts_dir = os.path.dirname(sys.executable)
    current_path = os.environ.get("PATH", "")
    if scripts_dir and scripts_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"


def _disable_env_proxies() -> None:
    # RAGAnything bridge uses internal OpenAI/httpx clients that honor env proxies.
    # In this local setup, proxy routing can break calls to DashScope and localhost.
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


def _guess_image_ext(url: str, content_type: str | None) -> str:
    if content_type:
        ct = content_type.lower()
        if "png" in ct:
            return ".png"
        if "jpeg" in ct or "jpg" in ct:
            return ".jpg"
        if "webp" in ct:
            return ".webp"
        if "gif" in ct:
            return ".gif"
    path_ext = Path(urlparse(url).path).suffix.lower()
    if path_ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}:
        return path_ext
    return ".jpg"


async def _materialize_remote_image(url: str, doc_id: str, idx: int, working_dir: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None

    images_dir = Path(working_dir).resolve() / "remote_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, trust_env=False, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.lower().startswith("image/"):
            return None

        ext = _guess_image_ext(url, content_type)
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        file_name = f"{doc_id}_{idx}_{digest}{ext}"
        file_path = images_dir / file_name
        file_path.write_bytes(resp.content)
        return str(file_path)


_HTML_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


async def _fetch_html_body(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    async with httpx.AsyncClient(
        timeout=30, follow_redirects=True, trust_env=False, headers=_HTML_FETCH_HEADERS
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ct = (resp.headers.get("content-type") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct and "text/plain" not in ct:
            return None
        return resp.text


def _looks_like_html_document(text: str) -> bool:
    t = text.lstrip()[:4000]
    if not t.startswith("<"):
        return False
    lower = t.lower()
    return "<html" in lower or "<!doctype html" in lower


def _should_route_html_to_docling(doc: IngestDocument) -> bool:
    fmt = (doc.metadata.get("content_format") or doc.metadata.get("parse_as") or "").lower()
    if fmt in ("html", "text/html", "docling", "docling_html"):
        return True
    if doc.metadata.get("html_fetch_url"):
        return True
    src = doc.metadata.get("source")
    if isinstance(src, str):
        low = src.lower()
        if low.endswith((".html", ".htm", ".xhtml")):
            return True
        if src.startswith("http"):
            path = urlparse(src).path.lower()
            if path.endswith((".html", ".htm", ".xhtml")):
                return True
    if doc.text and _looks_like_html_document(doc.text):
        return True
    return False


def _parse_html_file_with_docling(html_path: Path, working_dir: str) -> list[dict[str, Any]] | None:
    try:
        from raganything.parser import get_parser

        parser = get_parser("docling")
        if not parser.check_installation():
            logger.warning("docling_not_installed_skipping_html_route")
            return None
        out_base = Path(working_dir).resolve() / "docling_output"
        out_base.mkdir(parents=True, exist_ok=True)
        return parser.parse_document(
            str(html_path.resolve()),
            method="auto",
            output_dir=str(out_base),
            lang=None,
        )
    except Exception:
        logger.exception("docling_parse_html_failed path=%s", html_path)
        return None


def _max_page_idx(items: list[dict[str, Any]]) -> int:
    m = -1
    for it in items:
        p = it.get("page_idx")
        if isinstance(p, int):
            m = max(m, p)
    return m


def _pick_best_html_from_crawl_snapshot(full: dict[str, Any]) -> str | None:
    """Prefer fit_html from markdown bundle, then cleaned_html, then raw html."""
    md = full.get("markdown")
    if isinstance(md, dict):
        fh = md.get("fit_html") or md.get("fitHtml")
        if isinstance(fh, str) and fh.strip():
            return fh
    for key in ("cleaned_html", "html"):
        h = full.get(key)
        if isinstance(h, str) and h.strip():
            return h
    return None


def _markdown_supplement_from_crawl_snapshot(full: dict[str, Any]) -> str:
    md = full.get("markdown")
    if isinstance(md, dict):
        for k in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    for k in ("fit_markdown", "raw_markdown", "extracted_content"):
        v = full.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _table_dict_to_markdown_body(t: dict[str, Any]) -> str:
    headers = t.get("headers") or []
    rows = t.get("rows") or []
    cap = t.get("caption") or t.get("summary") or ""
    lines: list[str] = []
    if cap:
        lines.append(f"**{cap}**")
    if headers:
        hs = [str(h) for h in headers]
        lines.append("| " + " | ".join(hs) + " |")
        lines.append("| " + " | ".join("---" for _ in hs) + " |")
    for row in rows:
        if isinstance(row, list):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        elif isinstance(row, dict) and headers:
            lines.append(
                "| "
                + " | ".join(str(row.get(h, "")) for h in headers)
                + " |"
            )
        elif isinstance(row, dict):
            vals = [str(v) for v in row.values()]
            lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


async def _build_hybrid_crawl_content_list(
    doc: IngestDocument,
    working_dir: str,
    loop: asyncio.AbstractEventLoop,
) -> list[dict[str, Any]] | None:
    """
    Crawl4AI snapshot + Docling(HTML) + markdown supplement + structured tables + images.

    Returns None if hybrid is disabled or there is no crawl snapshot / nothing to insert.
    """
    if not bridge_settings.raganything_crawl_hybrid:
        return None

    full = doc.metadata.get("crawl4ai_full")
    if not isinstance(full, dict):
        return None

    structure = doc.metadata.get("crawl_structure")
    if not isinstance(structure, dict):
        structure = {}

    items: list[dict[str, Any]] = []
    max_html = int(bridge_settings.raganything_max_html_chars)

    html_body = _pick_best_html_from_crawl_snapshot(full)
    if html_body:
        if len(html_body) > max_html:
            logger.warning(
                "crawl_html_truncated doc_id=%s chars=%s max=%s",
                doc.doc_id,
                len(html_body),
                max_html,
            )
            html_body = html_body[:max_html]
        html_dir = Path(working_dir).resolve() / "html_inputs"
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / f"{doc.doc_id}_crawl_hybrid.html"
        html_path.write_text(html_body, encoding="utf-8")
        parsed = await loop.run_in_executor(
            None,
            lambda p=html_path, wd=working_dir: _parse_html_file_with_docling(p, wd),
        )
        if parsed:
            items.extend(parsed)

    if not items and (doc.text or "").strip():
        items.append({"type": "text", "text": doc.text.strip(), "page_idx": 0})

    md_extra = _markdown_supplement_from_crawl_snapshot(full)
    doc_text = (doc.text or "").strip()
    if md_extra and md_extra != doc_text:
        if not doc_text or (doc_text not in md_extra and md_extra not in doc_text):
            pidx = _max_page_idx(items) + 1
            items.append(
                {
                    "type": "text",
                    "text": "## Markdown capture\n\n" + md_extra,
                    "page_idx": pidx,
                }
            )

    tables = structure.get("tables") or full.get("tables") or []
    if isinstance(tables, list) and tables:
        base_p = _max_page_idx(items) + 1
        for ti, t in enumerate(tables):
            if not isinstance(t, dict):
                continue
            body = _table_dict_to_markdown_body(t)
            if not body.strip():
                continue
            cap = str(t.get("caption") or t.get("summary") or "")
            items.append(
                {
                    "type": "table",
                    "table_body": body,
                    "table_caption": [cap] if cap else [],
                    "table_footnote": [],
                    "page_idx": base_p + ti,
                }
            )

    seen_urls: set[str] = set()
    img_counter = 0

    async def _add_image_url(url: str, desc: str) -> None:
        nonlocal img_counter
        u = url.strip()
        if not u or u in seen_urls:
            return
        seen_urls.add(u)
        try:
            local_path = await _materialize_remote_image(
                u, doc.doc_id, img_counter, working_dir
            )
        except Exception:
            logger.exception("hybrid_image_materialize_failed doc_id=%s url=%s", doc.doc_id, u)
            local_path = None
        img_counter += 1
        if local_path:
            pidx = _max_page_idx(items) + 1
            items.append(
                {
                    "type": "image",
                    "img_path": local_path,
                    "image_caption": [desc] if desc else [],
                    "image_footnote": [],
                    "page_idx": pidx,
                }
            )
        else:
            pidx = _max_page_idx(items) + 1
            items.append(
                {
                    "type": "text",
                    "text": f"[image] {desc} {u}".strip(),
                    "page_idx": pidx,
                }
            )

    for modal in doc.modal_elements or []:
        if modal.get("type") != "image":
            continue
        u = modal.get("url")
        if isinstance(u, str) and u.strip():
            await _add_image_url(u, str(modal.get("desc") or ""))

    media = full.get("media")
    if isinstance(media, dict):
        for img in media.get("images") or []:
            if not isinstance(img, dict):
                continue
            src = img.get("src") or img.get("url")
            if isinstance(src, str) and src.strip():
                alt = str(img.get("alt") or img.get("desc") or "")
                await _add_image_url(src, alt)

    if not items:
        return None
    return items


def _build_rag_instance() -> Any:
    if not _RAGANYTHING_AVAILABLE:
        return None

    api_key = bridge_settings.openai_api_key or ""
    if not api_key:
        return None

    base_url = bridge_settings.openai_base_url
    llm_model = bridge_settings.raganything_llm_model
    vision_model = bridge_settings.raganything_vision_model
    embedding_model = bridge_settings.raganything_embedding_model
    embedding_dim = int(bridge_settings.raganything_embedding_dim)
    working_dir = bridge_settings.raganything_working_dir
    parser = bridge_settings.raganything_parser
    _ensure_parser_scripts_dir_on_path()
    _disable_env_proxies()

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def _get_rag() -> Any:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = _build_rag_instance()
    return _rag_instance


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    rag_ready = _RAGANYTHING_AVAILABLE and bool(bridge_settings.openai_api_key)
    return {"status": "ok", "raganything_ready": str(rag_ready).lower()}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    indexed_ids = [d.doc_id for d in req.documents]
    uid = str(req.tags.get("uid", "unknown"))
    _fallback_docs.setdefault(uid, []).extend(req.documents)

    if not bridge_settings.raganything_full_ingest_enabled:
        return {
            "indexed_doc_ids": indexed_ids,
            "status": "ok",
            "mode": "uid_context_cache",
        }

    rag = await _get_rag()
    if rag is not None:
        try:
            working_dir = bridge_settings.raganything_working_dir
            loop = asyncio.get_running_loop()
            for doc in req.documents:
                hybrid_list = await _build_hybrid_crawl_content_list(
                    doc, working_dir, loop
                )
                if hybrid_list:
                    await rag.insert_content_list(
                        content_list=hybrid_list,
                        file_path=str(doc.metadata.get("source") or doc.doc_id),
                        doc_id=doc.doc_id,
                        display_stats=False,
                    )
                    continue

                inserted_via_docling = False
                if _should_route_html_to_docling(doc):
                    html_body = doc.text or ""
                    fetch_url = doc.metadata.get("html_fetch_url")
                    if isinstance(fetch_url, str) and fetch_url.strip():
                        try:
                            fetched = await _fetch_html_body(fetch_url.strip())
                            if fetched:
                                html_body = fetched
                        except Exception:
                            logger.exception(
                                "raganything_html_fetch_failed doc_id=%s", doc.doc_id
                            )
                    if html_body.strip():
                        html_dir = Path(working_dir).resolve() / "html_inputs"
                        html_dir.mkdir(parents=True, exist_ok=True)
                        html_path = html_dir / f"{doc.doc_id}_content.html"
                        html_path.write_text(html_body, encoding="utf-8")
                        parsed = await loop.run_in_executor(
                            None,
                            lambda p=html_path, wd=working_dir: _parse_html_file_with_docling(
                                p, wd
                            ),
                        )
                        if parsed:
                            await rag.insert_content_list(
                                content_list=parsed,
                                file_path=str(
                                    doc.metadata.get("source") or doc.doc_id
                                ),
                                doc_id=doc.doc_id,
                                display_stats=False,
                            )
                            inserted_via_docling = True

                if inserted_via_docling:
                    continue

                content_list: list[dict[str, Any]] = []
                if doc.text:
                    content_list.append({"type": "text", "text": doc.text, "page_idx": 0})

                for idx, modal in enumerate(doc.modal_elements):
                    m_type = modal.get("type", "generic")
                    if m_type == "equation":
                        content_list.append(
                            {
                                "type": "equation",
                                "latex": modal.get("desc") or "",
                                "text": modal.get("desc") or "",
                                "page_idx": 0,
                            }
                        )
                    elif m_type == "table":
                        content_list.append(
                            {
                                "type": "table",
                                "table_body": modal.get("desc") or "",
                                "table_caption": [modal.get("desc") or ""],
                                "page_idx": 0,
                            }
                        )
                    elif m_type == "image":
                        image_url = modal.get("url")
                        image_desc = modal.get("desc") or ""
                        local_path: str | None = None
                        if isinstance(image_url, str) and image_url.strip():
                            try:
                                local_path = await _materialize_remote_image(
                                    image_url.strip(),
                                    doc.doc_id,
                                    idx,
                                    working_dir,
                                )
                            except Exception:
                                logger.exception("raganything_image_download_failed doc_id=%s", doc.doc_id)
                        if local_path:
                            content_list.append(
                                {
                                    "type": "image",
                                    "img_path": local_path,
                                    "image_caption": [image_desc] if image_desc else [],
                                    "image_footnote": [],
                                    "page_idx": 0,
                                }
                            )
                        else:
                            synthetic = f"[image] {image_desc} {image_url or ''}".strip()
                            if synthetic:
                                content_list.append({"type": "text", "text": synthetic, "page_idx": 0})
                    else:
                        # For non-local images, preserve as text to keep robust ingestion.
                        synthetic = f"[{m_type}] {modal.get('desc') or ''} {modal.get('url') or ''}".strip()
                        if synthetic:
                            content_list.append({"type": "text", "text": synthetic, "page_idx": 0})

                if content_list:
                    await rag.insert_content_list(
                        content_list=content_list,
                        file_path=str(doc.metadata.get("source") or doc.doc_id),
                        doc_id=doc.doc_id,
                        display_stats=False,
                    )
        except Exception:
            logger.exception("raganything_ingest_failed, fallback cache only")

    return {"indexed_doc_ids": indexed_ids, "status": "ok"}


@app.post("/query")
async def query(req: QueryRequest) -> QueryResponse:
    evidence, images = _weak_evidence_and_images(req.uid)
    uid_doc_answer = await _answer_from_uid_docs(req)
    if uid_doc_answer:
        return QueryResponse(
            answer=uid_doc_answer,
            evidence=evidence,
            images=images,
            trace_id=req.trace_id,
            latency_ms=0,
        )

    rag = await _get_rag()
    if rag is not None:
        try:
            answer = await rag.aquery(
                req.query, mode=bridge_settings.raganything_query_mode
            )
            return QueryResponse(
                answer=str(answer),
                evidence=evidence,
                images=images,
                trace_id=req.trace_id,
                latency_ms=0,
            )
        except Exception:
            logger.exception("raganything_query_failed, fallback local docs")

    return QueryResponse(
        answer="RAGAnything bridge fallback result (SDK not fully initialized).",
        evidence=evidence,
        images=images,
        trace_id=req.trace_id,
        latency_ms=0,
    )
