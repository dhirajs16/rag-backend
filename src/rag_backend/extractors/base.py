"""Shared types and contracts for document text extraction."""

from dataclasses import dataclass
from typing import Protocol


class ExtractionError(ValueError):
    """Raised when text cannot be extracted from a supported document."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one logical document page."""

    text: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Page-aware text produced by a document extractor."""

    pages: tuple[ExtractedPage, ...]

    @property
    def text(self) -> str:
        """Return all extracted pages as one normalized text value."""
        return "\n\n".join(page.text for page in self.pages)


class DocumentExtractor(Protocol):
    """Contract implemented by every supported file extractor."""

    def extract(self, content: bytes) -> ExtractedDocument:
        """Extract page-aware text from raw file bytes."""
        ...
