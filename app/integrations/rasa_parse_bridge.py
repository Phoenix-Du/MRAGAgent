from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Rasa Parse Bridge", version="0.1.0")


class ParseRequest(BaseModel):
    text: str


def _infer_intent(text: str) -> tuple[str, float]:
    value = text.lower()
    image_keywords = ("图", "图片", "image", "photo", "show me", "找图", "搜图")
    if any(k in value for k in image_keywords):
        return "image_search", 0.92
    return "general_qa", 0.88


@app.get("/version")
async def version() -> dict[str, str]:
    return {"version": "bridge-0.1.0"}


@app.post("/model/parse")
async def model_parse(req: ParseRequest) -> dict:
    intent, confidence = _infer_intent(req.text)
    return {
        "text": req.text,
        "intent": {"name": intent, "confidence": confidence},
        "entities": [],
        "intent_ranking": [{"name": intent, "confidence": confidence}],
    }
