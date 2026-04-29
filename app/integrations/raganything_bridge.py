from __future__ import annotations

import asyncio
import os
import logging
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


def _weak_evidence_and_images(uid: str) -> tuple[list[EvidenceItem], list[ImageItem]]:
    docs = _fallback_docs.get(uid, [])[:3]
    evidence = [
        EvidenceItem(
            doc_id=d.doc_id,
            score=0.7 - i * 0.1,
            snippet=(d.text[:160] if d.text else "No text content."),
        )
        for i, d in enumerate(docs)
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
    rag = await _get_rag()
    if rag is not None:
        try:
            evidence, images = _weak_evidence_and_images(req.uid)
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

    evidence, images = _weak_evidence_and_images(req.uid)

    return QueryResponse(
        answer="RAGAnything bridge fallback result (SDK not fully initialized).",
        evidence=evidence,
        images=images,
        trace_id=req.trace_id,
        latency_ms=0,
    )
