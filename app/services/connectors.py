from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from html import unescape
from hashlib import md5
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from app.core.metrics import metrics
from app.core.retry import with_retry
from app.core.runtime_flags import add_runtime_flag
from app.core.settings import settings
from app.core.url_safety import is_safe_public_http_url
from app.models.schemas import ModalElement, SourceDoc

logger = logging.getLogger(__name__)


def _markdown_text_from_crawl_result(result: Any) -> str:
    """Resolve primary markdown string from CrawlResult (handles markdown object vs legacy fields)."""
    if result is None:
        return ""
    if isinstance(result, dict):
        md = result.get("markdown")
        if isinstance(md, dict):
            return str(
                md.get("fit_markdown")
                or md.get("raw_markdown")
                or md.get("markdown_with_citations")
                or ""
            )
        if isinstance(md, str):
            return md
        return str(
            result.get("fit_markdown")
            or result.get("raw_markdown")
            or result.get("markdown")
            or ""
        )
    md_obj = getattr(result, "markdown", None)
    if md_obj is None:
        return ""
    try:
        fm = getattr(md_obj, "fit_markdown", None)
        if fm:
            return str(fm)
        rm = getattr(md_obj, "raw_markdown", None)
        if rm:
            return str(rm)
        return str(md_obj)
    except Exception:
        return str(md_obj)


def _crawl_result_to_json_snapshot(result: Any) -> dict[str, Any]:
    """Serialize Crawl4AI CrawlResult for downstream RAG bridge (exclude raw PDF bytes)."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return deepcopy(result)
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump(mode="json", exclude={"pdf"})
        except Exception:
            logger.debug("crawl4ai model_dump json failed, trying python+fallback", exc_info=True)
        try:
            dumped = result.model_dump(mode="python", exclude={"pdf"})
            return json.loads(json.dumps(dumped, default=str))
        except Exception:
            logger.warning("crawl4ai full snapshot serialization failed, using partial fields")
            return {
                "url": getattr(result, "url", None),
                "success": getattr(result, "success", None),
                "html_len": len(getattr(result, "html", "") or ""),
                "cleaned_html_len": len(getattr(result, "cleaned_html", "") or ""),
                "media": getattr(result, "media", None),
                "tables": getattr(result, "tables", None),
                "links": getattr(result, "links", None),
                "extracted_content": getattr(result, "extracted_content", None),
                "error_message": getattr(result, "error_message", None),
            }
    return {}


def _media_dict_to_modal_items(media: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn crawl4ai media buckets into bridge-friendly modal dicts (video/audio as generic)."""
    items: list[dict[str, Any]] = []
    for img in media.get("images") or []:
        if not isinstance(img, dict):
            continue
        src = img.get("src") or img.get("url")
        if src:
            items.append(
                {
                    "type": "image",
                    "url": str(src),
                    "desc": str(img.get("alt") or img.get("desc") or ""),
                }
            )
    for vid in media.get("videos") or []:
        if not isinstance(vid, dict):
            continue
        src = vid.get("src") or vid.get("url")
        if src:
            alt = str(vid.get("alt") or "").strip()
            items.append(
                {
                    "type": "generic",
                    "url": str(src),
                    "desc": f"[video] {alt}".strip(),
                }
            )
    for aud in media.get("audios") or []:
        if not isinstance(aud, dict):
            continue
        src = aud.get("src") or aud.get("url")
        if src:
            alt = str(aud.get("alt") or "").strip()
            items.append(
                {
                    "type": "generic",
                    "url": str(src),
                    "desc": f"[audio] {alt}".strip(),
                }
            )
    return items


_CRAWL_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
}


def _clean_extracted_text(text: str, max_chars: int = 60_000) -> str:
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(
        r"(?is)Unsupported Browser Detected.*?Please update to a modern browser.*?(?=\n|$)",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"(?is)The web Browser you are currently using is unsupported.*?(?=\n|$)",
        " ",
        normalized,
    )
    return normalized.strip()[:max_chars]


def _extract_html_content(html: str, base_url: str) -> dict[str, Any] | None:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript", "svg", "template", "form"]):
            node.decompose()
        for node in soup(["header", "nav", "footer", "aside"]):
            node.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        content_root = soup.find("main") or soup.find("article") or soup.body or soup
        text = _clean_extracted_text(content_root.get_text("\n", strip=True))

        images: list[dict[str, str]] = []
        seen: set[str] = set()
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if not src:
                continue
            abs_url = urljoin(base_url, str(src).strip())
            if abs_url in seen:
                continue
            seen.add(abs_url)
            desc = str(img.get("alt") or img.get("title") or "").strip()
            images.append({"type": "image", "url": abs_url, "desc": desc})
            if len(images) >= 5:
                break
        if not text:
            return None
        return {"title": title, "text_content": text, "modal_elements": images}
    except Exception:
        logger.debug("beautifulsoup_html_extract_failed", exc_info=True)

    no_script = re.sub(r"(?is)<(script|style|noscript|svg|template).*?>.*?</\1>", " ", html)
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", no_script)
    title = unescape(re.sub(r"(?is)<.*?>", " ", title_match.group(1))).strip() if title_match else ""
    body = re.sub(r"(?is)<(header|nav|footer|aside|form).*?>.*?</\1>", " ", no_script)
    text = _clean_extracted_text(unescape(re.sub(r"(?is)<[^>]+>", "\n", body)))
    if not text:
        return None
    return {"title": title, "text_content": text, "modal_elements": []}


def _normalize_duckduckgo_href(href: str) -> str:
    candidate = href.strip()
    if not candidate:
        return ""
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    if candidate.startswith("/"):
        candidate = urljoin("https://duckduckgo.com", candidate)
    parsed = urlparse(candidate)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = (parse_qs(parsed.query).get("uddg") or [""])[0]
        return target.strip()
    return candidate


def _map_duckduckgo_html(html: str, query: str, top_k: int) -> list[SourceDoc]:
    mapped: list[SourceDoc] = []
    seen: set[str] = set()
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("a.result__a") or soup.find_all("a")
        for anchor in anchors:
            href = anchor.get("href")
            if not href:
                continue
            url = _normalize_duckduckgo_href(str(href))
            if not url or url in seen or not url.startswith(("http://", "https://")):
                continue
            if "duckduckgo.com" in (urlparse(url).netloc or "").lower():
                continue
            seen.add(url)
            title = anchor.get_text(" ", strip=True)
            parent = anchor.find_parent("div")
            snippet_node = parent.select_one(".result__snippet") if parent else None
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            text = f"{title}\n{snippet}".strip() or url
            mapped.append(
                SourceDoc(
                    doc_id=f"search::{md5(url.encode('utf-8')).hexdigest()[:10]}",
                    text_content=text,
                    modal_elements=[],
                    structure={"type": "search_hit"},
                    metadata={
                        "source": "search_duckduckgo_html",
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "query": query,
                    },
                )
            )
            if len(mapped) >= top_k:
                break
    except Exception:
        logger.debug("duckduckgo_html_parse_failed", exc_info=True)

    if mapped:
        return mapped

    for href, title in re.findall(r'(?is)<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html):
        url = _normalize_duckduckgo_href(unescape(re.sub(r"(?is)<[^>]+>", "", href)))
        if not url or url in seen or not url.startswith(("http://", "https://")):
            continue
        if "duckduckgo.com" in (urlparse(url).netloc or "").lower():
            continue
        seen.add(url)
        clean_title = _clean_extracted_text(unescape(re.sub(r"(?is)<[^>]+>", " ", title)), max_chars=200)
        mapped.append(
            SourceDoc(
                doc_id=f"search::{md5(url.encode('utf-8')).hexdigest()[:10]}",
                text_content=clean_title or url,
                modal_elements=[],
                structure={"type": "search_hit"},
                metadata={
                    "source": "search_duckduckgo_html",
                    "url": url,
                    "title": clean_title,
                    "snippet": "",
                    "query": query,
                },
            )
        )
        if len(mapped) >= top_k:
            break
    return mapped


class SearchClient:
    async def search_web_hits(self, query: str, top_k: int = 10) -> list[SourceDoc]:
        if settings.serpapi_endpoint:
            try:
                async def _call() -> list[SourceDoc]:
                    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                        resp = await client.post(
                            settings.serpapi_endpoint,
                            json={"query": query, "top_k": top_k},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        return self._map_search_hits(data=data, query=query, top_k=top_k)

                return await with_retry(_call)
            except Exception:
                add_runtime_flag("search_fallback")
                metrics.inc("search_fallback_total")
                logger.warning("search_fallback_used reason=remote_search_hits_failed")

        # Direct SerpAPI web search fallback (option A): no middle endpoint required.
        serpapi_keys = self._load_serpapi_keys()
        if serpapi_keys:
            try:
                async def _call_serpapi() -> list[SourceDoc]:
                    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False) as client:
                        for api_key in serpapi_keys:
                            params = {
                                "engine": "google",
                                "q": query,
                                "api_key": api_key,
                                "num": max(1, min(top_k, 20)),
                                "hl": "zh-cn",
                                "gl": "cn",
                            }
                            try:
                                resp = await client.get("https://serpapi.com/search.json", params=params)
                                if resp.status_code in {401, 403, 429}:
                                    continue
                                resp.raise_for_status()
                                data = resp.json()
                                if isinstance(data, dict) and data.get("error"):
                                    continue
                                mapped = self._map_search_hits(data=data, query=query, top_k=top_k)
                                if mapped:
                                    return mapped
                            except (httpx.HTTPError, ValueError, TypeError):
                                continue
                    return []

                hits = await with_retry(_call_serpapi)
                if hits:
                    return hits
            except Exception:
                add_runtime_flag("search_fallback")
                metrics.inc("search_fallback_total")
                logger.warning("search_fallback_used reason=direct_serpapi_failed")

        try:
            html_hits = await self._search_duckduckgo_html(query=query, top_k=top_k)
            if html_hits:
                add_runtime_flag("search_html_fallback")
                metrics.inc("search_fallback_total")
                return html_hits
        except Exception:
            add_runtime_flag("search_fallback")
            metrics.inc("search_fallback_total")
            logger.warning("search_fallback_used reason=duckduckgo_html_failed")

        if not settings.allow_placeholder_fallback:
            add_runtime_flag("search_unavailable")
            return []

        # Fallback placeholder hits.
        hits: list[SourceDoc] = []
        for idx in range(1, top_k + 1):
            url = f"https://example.com/search?q={query}&rank={idx}"
            hits.append(
                SourceDoc(
                    doc_id=f"search::{md5(url.encode('utf-8')).hexdigest()[:10]}",
                    text_content=f"{query} {url}",
                    modal_elements=[],
                    structure={"type": "search_hit"},
                    metadata={"source": "search_placeholder", "url": url, "title": "", "snippet": ""},
                )
            )
        return hits

    async def search_web(self, query: str, top_k: int = 5) -> list[str]:
        hits = await self.search_web_hits(query=query, top_k=top_k)
        out: list[str] = []
        for h in hits:
            url = str((h.metadata or {}).get("url", "")).strip()
            if url:
                out.append(url)
        return out[:top_k]

    @staticmethod
    def _map_search_hits(data: dict, query: str, top_k: int) -> list[SourceDoc]:
        items = data.get("hits") or data.get("results") or data.get("organic_results") or []
        if not isinstance(items, list) or not items:
            urls = data.get("urls") or []
            if isinstance(urls, list):
                items = [{"url": u} for u in urls if isinstance(u, str)]

        mapped: list[SourceDoc] = []
        for item in items[:top_k]:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link")
            if not url:
                continue
            title = str(item.get("title") or "")
            snippet = str(item.get("snippet") or item.get("desc") or item.get("content") or "")
            text = f"{title}\n{snippet}".strip() or str(url)
            mapped.append(
                SourceDoc(
                    doc_id=f"search::{md5(str(url).encode('utf-8')).hexdigest()[:10]}",
                    text_content=text,
                    modal_elements=[],
                    structure={"type": "search_hit"},
                    metadata={"source": "search_api", "url": str(url), "title": title, "snippet": snippet, "query": query},
                )
            )
        return mapped

    @staticmethod
    def _load_serpapi_keys() -> list[str]:
        raw_multi = (settings.serpapi_api_keys or "").strip()
        key_candidates: list[str] = []
        if raw_multi:
            key_candidates.extend([k.strip() for k in raw_multi.split(",") if k.strip()])
        single = (settings.serpapi_api_key or "").strip()
        if single:
            key_candidates.append(single)
        # dedupe preserving order
        return list(dict.fromkeys(key_candidates))

    @staticmethod
    async def _search_duckduckgo_html(query: str, top_k: int) -> list[SourceDoc]:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            trust_env=False,
            headers=_CRAWL_HTTP_HEADERS,
            follow_redirects=True,
        ) as client:
            resp = await client.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
            )
            resp.raise_for_status()
            return _map_duckduckgo_html(resp.text, query=query, top_k=top_k)


class CrawlClient:
    async def crawl(self, url: str) -> SourceDoc:
        if settings.crawl4ai_local_enabled:
            try:
                local_data = await asyncio.wait_for(
                    self._crawl_with_local_sdk(url),
                    timeout=max(
                        0.001,
                        min(
                            float(settings.request_timeout_seconds),
                            float(settings.crawl4ai_local_timeout_seconds),
                        ),
                    ),
                )
                if local_data is not None:
                    return self._map_crawl4ai_response(url=url, data=local_data)
            except asyncio.TimeoutError:
                add_runtime_flag("crawl_local_sdk_timeout")
                logger.warning("crawl_local_sdk_timeout url=%s", url)
            except Exception:
                logger.warning("crawl_local_sdk_failed url=%s", url)

        if settings.crawl4ai_endpoint:
            try:
                async with httpx.AsyncClient(
                    timeout=settings.request_timeout_seconds,
                    trust_env=False,
                ) as client:
                    # Prefer Crawl4AI Docker API schema.
                    resp = await client.post(
                        settings.crawl4ai_endpoint,
                        json={"urls": [url], "browser_config": {}, "crawler_config": {}},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return self._map_crawl4ai_response(url=url, data=data)
            except (httpx.HTTPError, ValueError, TypeError):
                # Backward-compatible fallback for custom simple crawl APIs.
                logger.warning("crawl_primary_failed url=%s", url)
                try:
                    async with httpx.AsyncClient(
                        timeout=settings.request_timeout_seconds,
                        trust_env=False,
                    ) as client:
                        resp = await client.post(settings.crawl4ai_endpoint, json={"url": url})
                        resp.raise_for_status()
                        data = resp.json()
                        return self._map_crawl4ai_response(url=url, data=data)
                except (httpx.HTTPError, ValueError, TypeError):
                    # Soft-fail to keep the QA chain available when external crawling is unstable.
                    add_runtime_flag("crawl_fallback")
                    logger.warning("crawl_fallback_used url=%s", url)
                    metrics.inc("crawl_fallback_total")

        try:
            http_data = await self._crawl_with_http_fallback(url)
            if http_data is not None:
                add_runtime_flag("crawl_http_fallback")
                metrics.inc("crawl_fallback_total")
                return self._map_crawl4ai_response(url=url, data=http_data)
        except Exception:
            logger.warning("crawl_http_fallback_failed url=%s", url)

        if not settings.allow_placeholder_fallback:
            add_runtime_flag("crawl_unavailable")
            raise RuntimeError(f"crawl unavailable for url={url}")

        # Fallback placeholder.
        return SourceDoc(
            doc_id=f"crawl::{md5(url.encode('utf-8')).hexdigest()[:10]}",
            text_content=f"Fetched content from {url}",
            modal_elements=[ModalElement(type="image", url=f"{url}/cover.png", desc="cover image")],
            structure={"type": "webpage"},
            metadata={"source": "crawl4ai", "url": url},
        )

    @staticmethod
    async def _crawl_with_http_fallback(url: str) -> dict[str, Any] | None:
        current_url = url
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            trust_env=False,
            headers=_CRAWL_HTTP_HEADERS,
        ) as client:
            for _ in range(6):
                resp = await client.get(current_url)
                if 300 <= resp.status_code < 400:
                    location = resp.headers.get("location")
                    if not location:
                        break
                    next_url = urljoin(current_url, location)
                    if not is_safe_public_http_url(next_url):
                        add_runtime_flag("crawl_redirect_blocked")
                        return None
                    current_url = next_url
                    continue
                break
            else:
                add_runtime_flag("crawl_redirect_limit")
                return None

        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/plain" in content_type:
            text = _clean_extracted_text(resp.text)
            if not text:
                return None
            return {
                "doc_id": f"crawl::{md5(url.encode('utf-8')).hexdigest()[:10]}",
                "text_content": text,
                "title": "",
                "modal_elements": [],
                "structure": {"type": "webpage"},
                "metadata": {
                    "source": "http_fallback_crawl",
                    "url": url,
                    "final_url": str(resp.url),
                    "content_type": content_type,
                },
            }
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None

        extracted = _extract_html_content(resp.text, str(resp.url))
        if extracted is None:
            return None
        return {
            "doc_id": f"crawl::{md5(url.encode('utf-8')).hexdigest()[:10]}",
            "text_content": extracted["text_content"],
            "title": extracted.get("title") or "",
            "modal_elements": extracted.get("modal_elements") or [],
            "structure": {
                "type": "webpage",
                "title": extracted.get("title") or "",
                "fallback": "http",
            },
            "metadata": {
                "source": "http_fallback_crawl",
                "url": url,
                "final_url": str(resp.url),
                "title": extracted.get("title") or "",
                "content_type": content_type,
            },
        }

    @staticmethod
    async def _crawl_with_local_sdk(url: str) -> dict | None:
        """
        Use crawl4ai as an embedded framework (no standalone HTTP service).
        Returns a crawl4ai-like dict payload when successful.
        """
        try:
            from crawl4ai import AsyncWebCrawler  # type: ignore
        except Exception:
            return None

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

        if result is None:
            return None

        full_snapshot = _crawl_result_to_json_snapshot(result)
        markdown_text = _markdown_text_from_crawl_result(result)
        if not markdown_text and isinstance(full_snapshot.get("markdown"), dict):
            md_inner = full_snapshot["markdown"]
            markdown_text = str(
                md_inner.get("fit_markdown")
                or md_inner.get("raw_markdown")
                or md_inner.get("markdown_with_citations")
                or ""
            )

        title = getattr(result, "title", None)
        if title is None and isinstance(result, dict):
            title = result.get("title")
        if not title:
            title = full_snapshot.get("title")

        media = getattr(result, "media", None)
        if media is None and isinstance(result, dict):
            media = result.get("media")
        if not isinstance(media, dict):
            media = full_snapshot.get("media") if isinstance(full_snapshot.get("media"), dict) else {}

        tables = getattr(result, "tables", None)
        if tables is None:
            tables = full_snapshot.get("tables")
        if not isinstance(tables, list):
            tables = []

        links = getattr(result, "links", None)
        if links is None:
            links = full_snapshot.get("links")
        if not isinstance(links, dict):
            links = {}

        return {
            "text_content": markdown_text,
            "title": title,
            "media": media,
            "tables": tables,
            "links": links,
            "crawl4ai_full": full_snapshot,
            "metadata": {"source": "crawl4ai_local_sdk", "url": url},
            "structure": {"type": "webpage", "tables": tables, "links": links},
        }

    @staticmethod
    def _map_crawl4ai_response(url: str, data: dict) -> SourceDoc:
        # Crawl4AI Docker style: {"results": [{...}]}
        if isinstance(data.get("results"), list) and data["results"]:
            first = data["results"][0]
            if isinstance(first, dict):
                data = first

        doc_id = data.get("doc_id")
        if not doc_id:
            doc_id = f"crawl::{md5(url.encode('utf-8')).hexdigest()[:10]}"

        md_field = data.get("markdown")
        md_from_obj = ""
        if isinstance(md_field, dict):
            md_from_obj = str(
                md_field.get("fit_markdown")
                or md_field.get("raw_markdown")
                or md_field.get("markdown_with_citations")
                or ""
            )

        # Support common crawl4ai-like payload fields.
        text_content = (
            data.get("text_content")
            or data.get("content")
            or md_from_obj
            or (str(data.get("markdown")) if isinstance(data.get("markdown"), str) else "")
            or data.get("fit_markdown")
            or data.get("raw_markdown")
            or data.get("cleaned_text")
            or ""
        )

        raw_modal: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in data.get("modal_elements") or []:
            if isinstance(item, dict) and (item.get("url") or item.get("src")):
                u = str(item.get("url") or item.get("src"))
                if u not in seen_urls:
                    seen_urls.add(u)
                    raw_modal.append(item)

        if data.get("images"):
            for img in data.get("images", []):
                if not isinstance(img, dict):
                    continue
                u = img.get("url") or img.get("src")
                if u and str(u) not in seen_urls:
                    seen_urls.add(str(u))
                    raw_modal.append(
                        {
                            "type": "image",
                            "url": str(u),
                            "desc": img.get("alt") or img.get("desc"),
                        }
                    )

        if isinstance(data.get("media"), dict):
            for m in _media_dict_to_modal_items(data["media"]):
                u = m.get("url")
                if u and str(u) not in seen_urls:
                    seen_urls.add(str(u))
                    raw_modal.append(m)

        modal_elements: list[ModalElement] = []
        for item in raw_modal:
            if not isinstance(item, dict):
                continue
            modal_type = item.get("type") or "image"
            if modal_type not in {"image", "table", "equation", "generic"}:
                modal_type = "generic"
            modal_elements.append(
                ModalElement(
                    type=modal_type,
                    url=item.get("url") or item.get("src"),
                    desc=item.get("desc") or item.get("alt"),
                )
            )

        structure = data.get("structure")
        if not isinstance(structure, dict):
            structure = {}
        structure.setdefault("type", "webpage")
        if data.get("title"):
            structure.setdefault("title", data.get("title"))
        if isinstance(data.get("tables"), list):
            structure["tables"] = data["tables"]
        if isinstance(data.get("links"), dict):
            structure["links"] = data["links"]

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("source", "crawl4ai")
        metadata.setdefault("url", url)
        if data.get("title"):
            metadata.setdefault("title", data.get("title"))

        full = data.get("crawl4ai_full")
        if isinstance(full, dict):
            metadata["crawl4ai_full"] = full
        else:
            metadata["crawl4ai_full"] = deepcopy(data)

        return SourceDoc(
            doc_id=doc_id,
            text_content=text_content,
            modal_elements=modal_elements,
            structure=structure,
            metadata=metadata,
        )


class BGERerankClient:
    _model = None
    _tokenizer = None
    _device = "cpu"

    async def rerank(self, query: str, docs: list[SourceDoc], top_k: int = 5) -> list[SourceDoc]:
        if not docs:
            return []
        try:
            scores = self._score_with_bge(query=query, docs=docs)
            ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception:
            add_runtime_flag("bge_rerank_fallback")
            metrics.inc("bge_rerank_fallback_total")
            logger.warning("bge_rerank_fallback_used reason=local_bge_rerank_failed")
            return docs[:top_k]

    def _ensure_loaded(self) -> None:
        if self.__class__._model is not None and self.__class__._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = settings.bge_reranker_model or "BAAI/bge-reranker-base"
        local_files_only = settings.bge_reranker_local_files_only
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self.__class__._tokenizer = tokenizer
        self.__class__._model = model
        self.__class__._device = device

    def _score_with_bge(self, query: str, docs: list[SourceDoc]) -> list[float]:
        import torch

        self._ensure_loaded()
        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        device = self.__class__._device

        pairs = []
        for doc in docs:
            text = (doc.text_content or "").strip()
            if not text:
                text = str((doc.metadata or {}).get("url", ""))
            pairs.append((query, text[:1200]))

        with torch.no_grad():
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits.view(-1).float().cpu().tolist()
        return [float(s) for s in logits]


class ImagePipelineClient:
    async def search_and_rank_images_with_debug(
        self, query: str, top_k: int = 5
    ) -> tuple[list[ModalElement], dict[str, Any] | None]:
        if settings.image_pipeline_endpoint:
            try:
                async with httpx.AsyncClient(
                    timeout=settings.image_pipeline_timeout_seconds,
                    trust_env=False,
                ) as client:
                    resp = await client.post(
                        settings.image_pipeline_endpoint,
                        json={"query": query, "top_k": top_k},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    mapped = self._map_image_pipeline_response(data)
                    if mapped:
                        debug = data.get("debug") if isinstance(data, dict) else None
                        return mapped[:top_k], (debug if isinstance(debug, dict) else None)
            except (httpx.HTTPError, ValueError, TypeError):
                # Fall back to mock results if external image service fails.
                add_runtime_flag("image_pipeline_fallback")
                logger.warning("image_pipeline_fallback_used reason=remote_image_pipeline_failed")
                metrics.inc("image_pipeline_fallback_total")
        # Placeholder for Google Image API + CLIP + QwenVLM pipeline.
        return (
            [
                ModalElement(
                    type="image",
                    url=f"https://images.example.com/{idx}.png",
                    desc=f"image result {idx} for {query}",
                )
                for idx in range(1, top_k + 1)
            ],
            {"provider": "placeholder", "fallback_used": True},
        )

    async def search_and_rank_images(self, query: str, top_k: int = 5) -> list[ModalElement]:
        images, _debug = await self.search_and_rank_images_with_debug(query, top_k=top_k)
        return images

    @staticmethod
    def _map_image_pipeline_response(data: dict) -> list[ModalElement]:
        # Supported formats:
        # 1) {"images":[{"url":"...","desc":"..."}]}
        # 2) {"results":[{"image_url":"...","caption":"..."}]}
        # 3) {"modal_elements":[...]}
        raw_items = (
            data.get("images")
            or data.get("results")
            or data.get("modal_elements")
            or []
        )
        if not isinstance(raw_items, list):
            return []

        mapped: list[ModalElement] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("image_url")
            if not url:
                continue
            desc = item.get("desc") or item.get("caption") or item.get("title")
            mapped.append(
                ModalElement(
                    type="image",
                    url=str(url),
                    desc=(str(desc) if desc is not None else None),
                    local_path=(
                        str(item.get("local_path"))
                        if item.get("local_path") is not None
                        else None
                    ),
                )
            )
        return mapped
