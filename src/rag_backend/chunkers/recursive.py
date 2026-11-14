"""Boundary-aware recursive character chunking."""

from rag_backend.chunkers.base import ChunkingError, TextChunk
from rag_backend.extractors.base import ExtractedDocument


class RecursiveCharacterChunker:
    """Split text near paragraphs, lines, sentences, then words."""

    _separators = ("\n\n", "\n", ". ", " ")

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def split(self, document: ExtractedDocument) -> list[TextChunk]:
        """Split every page independently so source page metadata is preserved."""
        chunks: list[TextChunk] = []

        for page in document.pages:
            for text in self._split_page(page.text):
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

    def _split_page(self, text: str) -> list[str]:
        """Create overlapping chunks while preferring natural text boundaries."""
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            maximum_end = min(start + self._chunk_size, len(text))
            end = self._find_boundary(text, start, maximum_end)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            proposed_start = max(start + 1, end - self._chunk_overlap)
            start = self._align_overlap_to_word(text, start, proposed_start)

        return chunks

    def _find_boundary(self, text: str, start: int, maximum_end: int) -> int:
        """Find the strongest available boundary in the latter half of a window."""
        if maximum_end >= len(text):
            return len(text)

        minimum_end = start + ((maximum_end - start) // 2)
        for separator in self._separators:
            position = text.rfind(separator, minimum_end, maximum_end)
            if position != -1:
                return position + len(separator)

        return maximum_end

    @staticmethod
    def _align_overlap_to_word(text: str, previous_start: int, start: int) -> int:
        """Move an overlap boundary backward so a chunk does not start mid-word."""
        while start > previous_start + 1 and not text[start - 1].isspace():
            start -= 1
        return start
