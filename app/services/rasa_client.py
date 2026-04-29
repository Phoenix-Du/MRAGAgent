from __future__ import annotations

from typing import Any

import httpx

from app.core.settings import settings
from app.models.schemas import IntentType


class RasaClient:
    async def parse(self, text: str) -> tuple[IntentType | None, float, dict[str, str]]:
        if not settings.rasa_endpoint:
            return None, 0.0, {}

        try:
            async with httpx.AsyncClient(
                timeout=settings.request_timeout_seconds,
                trust_env=False,
            ) as client:
                resp = await client.post(settings.rasa_endpoint, json={"text": text})
                resp.raise_for_status()
                data = resp.json()
                intent, confidence = self._map_intent_response(data)
                entities = self._extract_entities(data)
                return intent, confidence, entities
        except (httpx.HTTPError, ValueError, TypeError):
            return None, 0.0, {}

    async def parse_intent(self, text: str) -> tuple[IntentType | None, float]:
        intent, confidence, _ = await self.parse(text)
        return intent, confidence

    @staticmethod
    def _map_intent_response(data: dict[str, Any]) -> tuple[IntentType | None, float]:
        # Common Rasa parse format: {"intent":{"name":"...","confidence":0.92}, ...}
        intent_node = data.get("intent")
        if isinstance(intent_node, dict):
            name = intent_node.get("name")
            confidence = float(intent_node.get("confidence") or 0.0)
            mapped = RasaClient._map_name(name)
            return mapped, confidence

        # Fallback format: {"intent":"...", "confidence":...}
        mapped = RasaClient._map_name(data.get("intent"))
        confidence = float(data.get("confidence") or 0.0)
        return mapped, confidence

    @staticmethod
    def _map_name(raw_name: Any) -> IntentType | None:
        if not isinstance(raw_name, str):
            return None
        name = raw_name.strip().lower()
        if name in {"image_search", "search_image", "image", "image_query"}:
            return "image_search"
        if name in {"general_qa", "qa", "general_question", "text_qa"}:
            return "general_qa"
        return None

    @staticmethod
    def _extract_entities(data: dict[str, Any]) -> dict[str, str]:
        entities: dict[str, str] = {}
        raw_entities = data.get("entities")
        if not isinstance(raw_entities, list):
            return entities
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            key = item.get("entity")
            value = item.get("value")
            if isinstance(key, str) and isinstance(value, str) and key not in entities:
                entities[key] = value
        return entities
