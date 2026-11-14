"""Ollama embedding provider."""

from typing import Any

import httpx

from rag_backend.integrations.embeddings.base import EmbeddingError


class OllamaEmbeddingProvider:
    """Generate normalized embeddings with Ollama's batch API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        batch_size: int,
        timeout_seconds: float = 120.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._model = model
        self._batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text in bounded batches and preserve its original order."""
        if not texts:
            raise EmbeddingError("At least one text is required for embedding.")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            embeddings.extend(await self._embed_batch(batch))

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "Ollama returned a different number of embeddings than requested."
            )

        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingError("Ollama returned inconsistent embedding dimensions.")

        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/api/embed",
                json={
                    "model": self._model,
                    "input": texts,
                    "truncate": True,
                    "keep_alive": "10m",
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            raw_embeddings = payload["embeddings"]
            if not isinstance(raw_embeddings, list):
                raise TypeError("embeddings must be a list")
            return [
                [float(value) for value in vector]
                for vector in raw_embeddings
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise EmbeddingError(
                f"Ollama could not generate embeddings with '{self._model}'."
            ) from error

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

