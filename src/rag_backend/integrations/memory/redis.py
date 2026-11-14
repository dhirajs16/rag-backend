"""Redis-backed bounded conversation memory."""

import json
import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag_backend.integrations.llm.base import ChatMessage
from rag_backend.integrations.memory.base import MemoryError


class RedisConversationMemory:
    """Store recent role-based messages in a Redis list with a TTL."""

    def __init__(self, *, url: str, ttl_seconds: int, max_messages: int) -> None:
        if ttl_seconds <= 0 or max_messages <= 0:
            raise ValueError("Redis memory limits must be greater than zero.")
        self._client: Redis = Redis.from_url(url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        self._max_messages = max_messages

    async def get(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        try:
            values = await self._client.lrange(self._key(conversation_id), 0, -1)
            return [self._decode(value) for value in values]
        except (RedisError, KeyError, TypeError, ValueError) as error:
            raise MemoryError("Redis could not read conversation memory.") from error

    async def append(
        self,
        conversation_id: uuid.UUID,
        messages: list[ChatMessage],
    ) -> None:
        if not messages:
            return
        key = self._key(conversation_id)
        encoded = [
            json.dumps({"role": message.role, "content": message.content})
            for message in messages
        ]
        try:
            async with self._client.pipeline(transaction=True) as pipeline:
                pipeline.rpush(key, *encoded)
                pipeline.ltrim(key, -self._max_messages, -1)
                pipeline.expire(key, self._ttl_seconds)
                await pipeline.execute()
        except RedisError as error:
            raise MemoryError("Redis could not update conversation memory.") from error

    @staticmethod
    def _key(conversation_id: uuid.UUID) -> str:
        return f"chat:{conversation_id}:messages"

    @staticmethod
    def _decode(value: str) -> ChatMessage:
        payload: dict[str, Any] = json.loads(value)
        role = payload["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Invalid stored chat role")
        return ChatMessage(role=role, content=str(payload["content"]))

    async def close(self) -> None:
        await self._client.aclose()
