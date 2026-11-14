"""Document extractor selection."""

from rag_backend.extractors.base import DocumentExtractor, ExtractedDocument
from rag_backend.extractors.pdf import PdfExtractor
from rag_backend.extractors.text import TextExtractor


def get_extractor(extension: str) -> DocumentExtractor:
    """Return the extractor registered for a validated file extension."""
    extractors: dict[str, DocumentExtractor] = {
        ".pdf": PdfExtractor(),
        ".txt": TextExtractor(),
    }
    return extractors[extension]


__all__ = ["ExtractedDocument", "get_extractor"]
