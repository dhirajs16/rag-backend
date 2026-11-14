"""Plain-text document extraction."""

from rag_backend.extractors.base import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionError,
)


class TextExtractor:
    """Decode UTF-8 plain-text documents."""

    def extract(self, content: bytes) -> ExtractedDocument:
        """Decode bytes and return one logical page."""
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ExtractionError("The text file must use UTF-8 encoding.") from error

        if not text:
            raise ExtractionError("The text file contains no extractable text.")

        return ExtractedDocument(pages=(ExtractedPage(text=text),))
