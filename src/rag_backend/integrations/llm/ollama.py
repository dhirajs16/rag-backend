"""Ollama chat-model integration."""

from typing import Any

import httpx

from rag_backend.integrations.llm.base import ChatMessage, LLMError


class OllamaChatModel:
    """Generate ordinary or schema-constrained responses through Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        if not messages:
            raise LLMError("At least one chat message is required.")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": self._temperature},
        }
        if response_schema is not None:
            payload["format"] = response_schema

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise TypeError("message content must be a non-empty string")
            return content.strip()
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise LLMError(
                f"Ollama could not generate a response with '{self._model}'."
            ) from error

    async def close(self) -> None:
        await self._client.aclose()
