"""PDF document extraction."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_backend.extractors.base import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionError,
)


class PdfExtractor:
    """Extract text from each page of a PDF document."""

    def extract(self, content: bytes) -> ExtractedDocument:
        """Read PDF bytes while preserving one-based page numbers."""
        try:
            reader = PdfReader(BytesIO(content))
        except (PdfReadError, ValueError, OSError) as error:
            raise ExtractionError("The PDF file is corrupt or unreadable.") from error

        if reader.is_encrypted:
            raise ExtractionError("Encrypted PDF files are not supported.")

        pages: list[ExtractedPage] = []
        try:
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(
                        ExtractedPage(text=text, page_number=page_number)
                    )
        except (PdfReadError, ValueError, OSError) as error:
            raise ExtractionError("Text could not be extracted from the PDF.") from error

        if not pages:
            raise ExtractionError(
                "The PDF contains no extractable text; it may be a scanned document."
            )

        return ExtractedDocument(pages=tuple(pages))
