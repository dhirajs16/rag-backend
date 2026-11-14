"""Embedding provider integrations."""

from rag_backend.integrations.embeddings.base import EmbeddingProvider
from rag_backend.integrations.embeddings.ollama import OllamaEmbeddingProvider

__all__ = ["EmbeddingProvider", "OllamaEmbeddingProvider"]

