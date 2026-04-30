from __future__ import annotations

from collections import defaultdict, deque
import json
from typing import Any

import aiomysql
import redis.asyncio as redis

from app.core.settings import settings


class MemoryClient:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.backend = settings.memory_backend.lower()
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )
        self._prefs: dict[str, dict[str, Any]] = defaultdict(dict)
        self._redis: redis.Redis | None = None
        self._mysql_pool: aiomysql.Pool | None = None
        self._mysql_ready = False

        if self.backend in {"redis", "hybrid"} and settings.redis_url:
            self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def get_context(self, uid: str) -> dict[str, Any]:
        if self.backend in {"redis", "hybrid"} and self._redis:
            context = await self._get_from_redis(uid)
            if context:
                return context

        if self.backend in {"mysql", "hybrid"}:
            context = await self._get_from_mysql(uid)
            if context:
                return context

        return {
            "history": list(self._history[uid]),
            "preferences": self._prefs[uid],
        }

    async def update_context(
        self,
        uid: str,
        query: str,
        answer: str,
        intent: str,
    ) -> None:
        if self.backend in {"redis", "hybrid"} and self._redis:
            await self._write_to_redis(uid=uid, query=query, answer=answer, intent=intent)

        if self.backend in {"mysql", "hybrid"}:
            await self._write_to_mysql(uid=uid, query=query, answer=answer, intent=intent)

        self._history[uid].append(
            {"query": query, "answer": answer, "intent": intent}
        )

    async def set_preference(self, uid: str, key: str, value: Any) -> None:
        if self.backend in {"redis", "hybrid"} and self._redis:
            await self._redis.hset(
                self._redis_pref_key(uid),
                key,
                json.dumps(value, ensure_ascii=True),
            )

        if self.backend in {"mysql", "hybrid"}:
            await self._set_preference_mysql(uid=uid, key=key, value=value)

        self._prefs[uid][key] = value

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
        if self._mysql_pool is not None:
            self._mysql_pool.close()
            await self._mysql_pool.wait_closed()

    def _redis_history_key(self, uid: str) -> str:
        return f"{settings.redis_prefix}:history:{uid}"

    def _redis_pref_key(self, uid: str) -> str:
        return f"{settings.redis_prefix}:prefs:{uid}"

    async def _get_from_redis(self, uid: str) -> dict[str, Any] | None:
        assert self._redis is not None
        raw_history = await self._redis.lrange(self._redis_history_key(uid), 0, self.max_turns - 1)
        raw_prefs = await self._redis.hgetall(self._redis_pref_key(uid))
        if not raw_history and not raw_prefs:
            return None

        history = [json.loads(item) for item in raw_history]
        preferences = {k: json.loads(v) for k, v in raw_prefs.items()}
        return {"history": history, "preferences": preferences}

    async def _write_to_redis(self, uid: str, query: str, answer: str, intent: str) -> None:
        assert self._redis is not None
        item = json.dumps({"query": query, "answer": answer, "intent": intent}, ensure_ascii=True)
        key = self._redis_history_key(uid)
        await self._redis.lpush(key, item)
        await self._redis.ltrim(key, 0, self.max_turns - 1)

    async def _ensure_mysql_pool(self) -> bool:
        if self._mysql_pool is None:
            if not all(
                [
                    settings.mysql_host,
                    settings.mysql_user,
                    settings.mysql_password,
                    settings.mysql_database,
                ]
            ):
                return False
            self._mysql_pool = await aiomysql.create_pool(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                db=settings.mysql_database,
                autocommit=True,
            )
        if not self._mysql_ready:
            assert self._mysql_pool is not None
            async with self._mysql_pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_memory_history (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            uid VARCHAR(128) NOT NULL,
                            query_text TEXT NOT NULL,
                            answer_text TEXT NOT NULL,
                            intent VARCHAR(64) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    await cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_preferences (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            uid VARCHAR(128) NOT NULL,
                            pref_key VARCHAR(128) NOT NULL,
                            pref_value JSON NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            UNIQUE KEY uniq_user_pref (uid, pref_key)
                        )
                        """
                    )
            self._mysql_ready = True
        return True

    async def _get_from_mysql(self, uid: str) -> dict[str, Any] | None:
        ready = await self._ensure_mysql_pool()
        if not ready:
            return None
        assert self._mysql_pool is not None
        async with self._mysql_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT query_text, answer_text, intent
                    FROM user_memory_history
                    WHERE uid=%s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (uid, self.max_turns),
                )
                history_rows = await cur.fetchall()
                await cur.execute(
                    """
                    SELECT pref_key, pref_value
                    FROM user_preferences
                    WHERE uid=%s
                    """,
                    (uid,),
                )
                pref_rows = await cur.fetchall()
        if not history_rows and not pref_rows:
            return None
        history = [
            {"query": row["query_text"], "answer": row["answer_text"], "intent": row["intent"]}
            for row in history_rows
        ]
        preferences = {
            row["pref_key"]: self._decode_pref_value(row["pref_value"])
            for row in pref_rows
        }
        return {"history": history, "preferences": preferences}

    async def _write_to_mysql(self, uid: str, query: str, answer: str, intent: str) -> None:
        ready = await self._ensure_mysql_pool()
        if not ready:
            return
        assert self._mysql_pool is not None
        async with self._mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_memory_history(uid, query_text, answer_text, intent)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (uid, query, answer, intent),
                )

    @staticmethod
    def _decode_pref_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def _set_preference_mysql(self, uid: str, key: str, value: Any) -> None:
        ready = await self._ensure_mysql_pool()
        if not ready:
            return
        assert self._mysql_pool is not None
        async with self._mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_preferences(uid, pref_key, pref_value)
                    VALUES (%s, %s, CAST(%s AS JSON))
                    ON DUPLICATE KEY UPDATE pref_value=VALUES(pref_value)
                    """,
                    (uid, key, json.dumps(value, ensure_ascii=True)),
                )

