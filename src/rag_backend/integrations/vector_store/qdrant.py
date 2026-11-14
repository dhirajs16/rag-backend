"""Qdrant vector-store integration."""

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from rag_backend.integrations.vector_store.base import (
    VectorSearchResult,
    VectorStoreError,
)
from rag_backend.models.document import Document
from rag_backend.models.document_chunk import DocumentChunk


class QdrantVectorStore:
    """Persist document chunk vectors and retrieval payloads in Qdrant."""

    def __init__(self, *, url: str, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client = AsyncQdrantClient(url=url)

    async def upsert_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Ensure the collection exists, then upsert all document chunks."""
        if not chunks or len(chunks) != len(embeddings):
            raise VectorStoreError("Every document chunk must have one embedding.")

        vector_size = len(embeddings[0])
        if vector_size == 0 or any(len(vector) != vector_size for vector in embeddings):
            raise VectorStoreError("Embedding vectors have inconsistent dimensions.")

        try:
            await self._ensure_collection(vector_size)
            await self._client.upsert(
                collection_name=self._collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=str(chunk.id),
                        vector=embedding,
                        payload={
                            "document_id": str(document.id),
                            "chunk_id": str(chunk.id),
                            "filename": document.original_filename,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "text": chunk.text,
                        },
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError("Qdrant could not store document vectors.") from error

    async def _ensure_collection(self, vector_size: int) -> None:
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = await self._client.get_collection(self._collection_name)
        vectors_config = collection.config.params.vectors
        existing_size = getattr(vectors_config, "size", None)
        if existing_size is not None and existing_size != vector_size:
            raise VectorStoreError(
                "The Qdrant collection dimension does not match the embedding model."
            )

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Find the document chunks most similar to a query vector."""
        if not query_vector:
            raise VectorStoreError("A query embedding is required for vector search.")
        if limit <= 0:
            raise VectorStoreError("The vector search limit must be greater than zero.")

        try:
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            return [self._to_search_result(point) for point in response.points]
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError("Qdrant could not search document vectors.") from error

    @staticmethod
    def _to_search_result(point: models.ScoredPoint) -> VectorSearchResult:
        payload = point.payload or {}
        try:
            page_number = payload.get("page_number")
            return VectorSearchResult(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                filename=str(payload["filename"]),
                chunk_index=int(payload["chunk_index"]),
                page_number=int(page_number) if page_number is not None else None,
                text=str(payload["text"]),
                score=float(point.score),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VectorStoreError("Qdrant returned an invalid chunk payload.") from error

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
