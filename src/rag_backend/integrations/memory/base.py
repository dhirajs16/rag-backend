"""Conversation-memory contracts."""

import uuid
from typing import Protocol

from rag_backend.integrations.llm.base import ChatMessage


class MemoryError(RuntimeError):
    """Raised when conversation history cannot be read or written."""


class ConversationMemory(Protocol):
    """Contract for bounded multi-turn conversation storage."""

    async def get(self, conversation_id: uuid.UUID) -> list[ChatMessage]: ...

    async def append(
        self,
        conversation_id: uuid.UUID,
        messages: list[ChatMessage],
    ) -> None: ...

    async def close(self) -> None: ...
