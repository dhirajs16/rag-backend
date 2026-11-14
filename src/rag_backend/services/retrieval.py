"""Semantic document-retrieval orchestration."""

from rag_backend.integrations.embeddings.base import EmbeddingProvider
from rag_backend.integrations.vector_store.base import VectorSearchResult, VectorStore


class RetrievalService:
    """Embed a user query and retrieve its most relevant document chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
    ) -> list[VectorSearchResult]:
        """Convert a query to a vector and find the closest stored chunks."""
        normalized_query = " ".join(query.split())
        query_vectors = await self._embedding_provider.embed_texts([normalized_query])
        return await self._vector_store.search(
            query_vectors[0],
            limit=top_k,
            score_threshold=score_threshold,
        )
