"""Embedding-aware semantic text chunking."""

import math
import re

from rag_backend.chunkers.base import ChunkingError, TextChunk
from rag_backend.extractors.base import ExtractedDocument
from rag_backend.integrations.embeddings.base import EmbeddingProvider


class SemanticChunker:
    """Group adjacent sentences until their meaning changes significantly."""

    _sentence_boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        breakpoint_percentile: float,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> None:
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be between 0 and 100")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than zero")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be at least min_chunk_size")

        self._embedding_provider = embedding_provider
        self._breakpoint_percentile = breakpoint_percentile
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    async def split(self, document: ExtractedDocument) -> list[TextChunk]:
        """Create page-aware chunks using changes in adjacent sentence meaning."""
        chunks: list[TextChunk] = []

        for page in document.pages:
            sentences = self._split_sentences(page.text)
            if not sentences:
                continue

            page_chunks = await self._split_page(sentences)
            for text in page_chunks:
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        text=text,
                        page_number=page.page_number,
                    )
                )

        if not chunks:
            raise ChunkingError("The document produced no text chunks.")

        return chunks

    async def _split_page(self, sentences: list[str]) -> list[str]:
        if len(sentences) == 1:
            return self._split_oversized_sentence(sentences[0])

        embeddings = await self._embedding_provider.embed_texts(sentences)
        distances = [
            self._cosine_distance(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        threshold = self._percentile(distances, self._breakpoint_percentile)

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for index, sentence in enumerate(sentences):
            added_size = len(sentence) + (1 if current else 0)
            exceeds_maximum = current and current_size + added_size > self._max_chunk_size
            previous_is_breakpoint = (
                index > 0
                and distances[index - 1] >= threshold
                and current_size >= self._min_chunk_size
            )

            if exceeds_maximum or previous_is_breakpoint:
                chunks.append(" ".join(current))
                current = []
                current_size = 0

            if len(sentence) > self._max_chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_size = 0
                chunks.extend(self._split_oversized_sentence(sentence))
                continue

            current.append(sentence)
            current_size += len(sentence) + (1 if current_size else 0)

        if current:
            chunks.append(" ".join(current))

        return chunks

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in cls._sentence_boundary.split(text.strip())
            if sentence.strip()
        ]

    def _split_oversized_sentence(self, text: str) -> list[str]:
        """Keep hard limits for text that has no usable sentence boundaries."""
        chunks: list[str] = []
        remaining = text.strip()
        while len(remaining) > self._max_chunk_size:
            end = remaining.rfind(" ", 0, self._max_chunk_size + 1)
            if end <= 0:
                end = self._max_chunk_size
            chunks.append(remaining[:end].strip())
            remaining = remaining[end:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _cosine_distance(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            raise ChunkingError("Sentence embeddings have incompatible dimensions.")

        dot_product = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            raise ChunkingError("Sentence embeddings cannot be zero vectors.")
        return 1.0 - (dot_product / (left_norm * right_norm))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * percentile / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
