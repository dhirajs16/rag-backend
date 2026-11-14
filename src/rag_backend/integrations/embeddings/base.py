"""Embedding provider contracts."""

from typing import Protocol


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot generate valid vectors."""


class EmbeddingProvider(Protocol):
    """Contract for converting text into numeric vectors."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate one embedding for every supplied text."""
        ...

    async def close(self) -> None:
        """Release provider network resources."""
        ...

