from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    docid: str
    text: str = ""


class RerankRequest(BaseModel):
    query: str
    candidates: list[Candidate] = Field(default_factory=list)
    top_k: int | None = None


app = FastAPI(title="RankLLM Bridge", version="0.1.0")


def _tokenize(value: str) -> list[str]:
    return [tok for tok in value.lower().split() if tok]


def _score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(text)
    if not q_tokens or not d_tokens:
        return 0.0

    d_counts = Counter(d_tokens)
    overlap = sum(d_counts.get(t, 0) for t in q_tokens)
    return overlap / max(len(d_tokens), 1)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/rerank")
async def rerank(req: RerankRequest) -> dict[str, Any]:
    candidates = []
    for cand in req.candidates:
        candidates.append(
            {
                "docid": cand.docid,
                "score": _score(req.query, cand.text),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if req.top_k is not None and req.top_k > 0:
        candidates = candidates[: req.top_k]

    return {
        "artifacts": [
            {
                "name": "rerank-results",
                "value": [
                    {
                        "query": req.query,
                        "candidates": candidates,
                    }
                ],
            }
        ]
    }
