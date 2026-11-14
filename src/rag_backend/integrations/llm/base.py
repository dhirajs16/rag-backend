"""Chat language-model contracts."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol


class LLMError(RuntimeError):
    """Raised when a chat model cannot produce a valid response."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One role-based message sent to or stored from a chat model."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel(Protocol):
    """Contract for conversational text generation."""

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate one assistant response."""
        ...

    async def close(self) -> None:
        """Release model network resources."""
        ...
