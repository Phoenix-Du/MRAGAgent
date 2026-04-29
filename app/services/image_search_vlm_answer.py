"""image_search: VLM-only multimodal answer (bypasses RAGAnything query template)."""

from __future__ import annotations

from app.core.runtime_flags import add_runtime_flag
from app.models.schemas import ImageItem, ImageSearchConstraints, NormalizedDocument, QueryResponse
from app.services.image_query_parser import constraints_to_prompt_text
from app.services.qwen_vlm_images import (
    filter_reachable_image_rows,
    vlm_answer_from_image_urls,
    vlm_filter_strict_match_indices,
    vlm_rank_and_answer_from_image_urls,
)


def _collect_image_rows(
    documents: list[NormalizedDocument],
) -> list[tuple[str, str | None, str | None]]:
    rows: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set()
    for doc in documents:
        for m in doc.modal_elements or []:
            if m.type != "image" or not m.url:
                continue
            u = str(m.url).strip()
            if not u or u in seen:
                continue
            seen.add(u)
            rows.append((u, m.desc, m.local_path))
    return rows


def _has_spatial_constraint(query: str) -> bool:
    text = query.lower()
    return any(token in text for token in ("左边", "右边", "左侧", "右侧", "left", "right"))


async def build_image_search_vlm_response(
    *,
    query: str,
    documents: list[NormalizedDocument],
    max_images: int,
    image_constraints: ImageSearchConstraints | None,
    trace_id: str,
) -> QueryResponse:
    rows = await filter_reachable_image_rows(_collect_image_rows(documents))
    if not rows:
        add_runtime_flag("image_search_vlm_no_images")
        return QueryResponse(
            answer="未找到可用于回答的图片结果。",
            evidence=[],
            images=[],
            trace_id=trace_id,
            latency_ms=0,
        )

    top_k = max(1, min(max_images, len(rows)))
    prompt_query = query
    constraints_text = constraints_to_prompt_text(image_constraints)
    if constraints_text:
        prompt_query = f"{query}\n\n[结构化约束]\n{constraints_text}"
    indices, selected, answer = await vlm_rank_and_answer_from_image_urls(
        prompt_query, rows, top_k=top_k
    )
    if selected:
        top = [rows[i] for i in selected[:top_k]]
        add_runtime_flag("image_search_vlm_selected_applied")
        add_runtime_flag("image_search_vlm_rank_answer_combined")
    elif indices:
        top = [rows[i] for i in indices[:top_k]]
        add_runtime_flag("image_search_vlm_rank_answer_combined")
    else:
        top = rows[:top_k]
        answer = answer or await vlm_answer_from_image_urls(
            prompt_query, top, max_images=top_k
        )
        add_runtime_flag("image_search_vlm_rank_answer_fallback")

    if top and _has_spatial_constraint(prompt_query):
        before_spatial_count = len(top)
        strict_indices = await vlm_filter_strict_match_indices(
            prompt_query,
            top,
            max_keep=top_k,
        )
        if strict_indices is not None:
            top = [top[i] for i in strict_indices]
            add_runtime_flag("image_search_vlm_spatial_filter_applied")
            if not top:
                answer = "已检索到候选图片，但其中没有严格满足左右位置约束的结果。"
            elif len(top) != before_spatial_count:
                answer = await vlm_answer_from_image_urls(
                    prompt_query,
                    top,
                    max_images=len(top),
                )
                add_runtime_flag("image_search_vlm_answer_regenerated_after_spatial_filter")

    # Final hardening: only return images that can be locally proxied.
    top = await filter_reachable_image_rows(top)

    if not answer.strip():
        add_runtime_flag("image_search_vlm_answer_empty")
        answer = (
            "未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，"
            "或稍后重试）。以下为检索到的图片链接，便于人工查看。"
        )
        add_runtime_flag("image_search_vlm_answer_degraded")
    else:
        add_runtime_flag("image_search_vlm_answer_generated")

    images = [ImageItem(url=u, desc=d, local_path=local_path) for u, d, local_path in top]
    return QueryResponse(
        answer=answer.strip(),
        evidence=[],
        images=images,
        trace_id=trace_id,
        latency_ms=0,
    )
